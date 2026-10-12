"""Real Scheduler/cache lifecycle; only model execution uses a scripted runner."""

import random
from collections import OrderedDict

import pytest
import torch

from nanovllm.engine.input import prepare_decode, prepare_prefill, prepare_mixed
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from numeric_helpers import CASE_SEEDS
from prefix_helpers import assert_block_conservation, assert_idle_cache, make_cache


class ScriptedRunner:
    run_batch = ModelRunner.run_batch

    def __init__(self, manager, scripts=None):
        self.manager = manager
        self.scripts = {} if scripts is None else scripts
        self.events = []
        self.fail_after_first_layer = False

    def run(self, seqs, is_prefill):
        prepare = {True: prepare_prefill, False: prepare_decode, None: prepare_mixed}[is_prefill]
        ids, positions, ctx = prepare(seqs, self.manager.k_pool.device)
        ctx = self.manager.prepare_ctx(ctx)
        self.events.append((is_prefill, tuple(
            (seq.seq_id, seq.num_cached_tokens, seq.num_scheduled_tokens) for seq in seqs
        )))
        assert len(ids) == sum(seq.num_scheduled_tokens for seq in seqs)
        for layer in range(self.manager.num_layers):
            for row, seq in enumerate(seqs):
                lo, hi = ctx.cu_seq_lens_q[row:row + 2]
                # Synthetic causal values depend on the actual prefix, so the
                # same token prefix has identical data regardless of request ID.
                values = [[sum(seq.token_ids[:int(t) + 1]), int(t) + layer]
                          for t in positions[lo:hi]]
                k = torch.tensor(values, device=ids.device, dtype=torch.float32)[:, None, :]
                self.manager.append(layer, seq.seq_id, k, -k, ctx.cache_lens[row],
                                    ctx.slot_mapping[lo:hi])
            if self.fail_after_first_layer:
                raise RuntimeError("injected partial forward failure")
        return [self.scripts[seq.seq_id][seq.num_completion_tokens]
                if seq.seq_id in self.scripts else 10 + seq.seq_id + seq.num_completion_tokens
                for seq in seqs]


def make_scheduler(blocks=4, size=4, budget=16):
    manager = make_cache(blocks, size)
    scheduler = Scheduler(manager, 2, budget, max_model_len=128)
    return scheduler, ScriptedRunner(manager)


def step(scheduler, runner):
    plan = scheduler.schedule()
    if plan is None:
        return []
    seqs = plan.seqs
    for seq in seqs:
        assert scheduler.cache_manager.lengths(seq.seq_id) == seq.num_cached_tokens
    tokens = runner.run_batch(plan)
    scheduler.postprocess(plan, tokens)
    assert_block_conservation(scheduler.cache_manager.block_manager)
    return [(seq.seq_id, list(seq.completion_token_ids))
            for seq in seqs if seq.status == SequenceStatus.FINISHED]


def drain(scheduler, runner, limit=64):
    outputs = {}
    for _ in range(limit):
        for sid, tokens in step(scheduler, runner):
            assert sid not in outputs
            outputs[sid] = tokens
        if scheduler.is_finished():
            assert_idle_cache(scheduler.cache_manager)
            return outputs
    raise AssertionError("generation failed to make bounded progress")


def test_live_share_cold_hit_and_decode_publication():
    scheduler, runner = make_scheduler()
    a = Sequence(0, list(range(9)), 5)
    scheduler.add(a)
    assert step(scheduler, runner) == []
    b = Sequence(1, list(range(8)) + [29], 1)
    scheduler.add(b)
    assert step(scheduler, runner) == [(1, [11])]
    assert runner.events[:2] == [(True, ((0, 0, 9),)), (True, ((1, 8, 1),))]
    assert drain(scheduler, runner) == {0: [10, 11, 12, 13, 14]}
    assert len(scheduler.cache_manager.block_manager.hash_to_block_id) == 3
    cold = Sequence(2, list(range(8)) + [30], 1)
    scheduler.add(cold)
    assert drain(scheduler, runner) == {2: [12]}
    assert runner.events[-1] == (True, ((2, 8, 1),))


def test_shared_preemption_frees_only_the_private_page(monkeypatch):
    scheduler, runner = make_scheduler(blocks=3, budget=12)
    a, b = Sequence(0, [10,11,12,13,18], 5), Sequence(1, [10,11,12,13,29], 5)
    scheduler.add(a)
    step(scheduler, runner)
    scheduler.add(b)
    step(scheduler, runner)
    manager = scheduler.cache_manager.block_manager
    assert manager.table(0) == (0, 1) and manager.table(1) == (0, 2)
    assert manager.blocks[0].ref_count == 2
    original = scheduler.preempt
    victims = []

    def observe(seq):
        history = list(seq.token_ids)
        free_before = len(manager.free_block_id)
        original(seq)
        assert seq.token_ids == history
        assert seq.num_cached_tokens == seq.num_scheduled_tokens == 0
        assert manager.blocks[0].ref_count == 1
        assert len(manager.free_block_id) == free_before + 1
        victims.append(seq.seq_id)

    monkeypatch.setattr(scheduler, "preempt", observe)
    assert drain(scheduler, runner) == {0: [10,11,12,13,14], 1: [11,12,13,14,15]}
    assert victims == [1]
    assert runner.events[-1] == (True, ((1, 4, 5),))


def test_partial_forward_failure_does_not_publish_and_abort_releases_sharers():
    scheduler, runner = make_scheduler()
    scheduler.add(Sequence(0, list(range(9)), 5))
    step(scheduler, runner)
    index = dict(scheduler.cache_manager.block_manager.hash_to_block_id)
    scheduler.add(Sequence(1, list(range(8)) + [29], 1))
    plan = scheduler.schedule()
    runner.fail_after_first_layer = True
    with pytest.raises(RuntimeError, match="partial forward"):
        runner.run_batch(plan)
    assert scheduler.cache_manager.layer_lengths[1] == [9, 8]
    with pytest.raises(RuntimeError, match="every layer"):
        scheduler.postprocess(plan, [11])
    assert scheduler.cache_manager.block_manager.hash_to_block_id == index
    assert scheduler.cache_manager.block_manager.blocks[3].hash == -1
    scheduler.abort()
    assert scheduler.is_finished()
    assert_idle_cache(scheduler.cache_manager)


def test_zero_budget_and_eos_release_only_the_ending_request():
    scheduler, runner = make_scheduler()
    scheduler.add(Sequence(0, [1, 2], 0))
    assert scheduler.is_finished() and step(scheduler, runner) == []
    a = Sequence(1, list(range(9)), 3)
    scheduler.add(a)
    step(scheduler, runner)
    b = Sequence(2, list(range(8)) + [29], 4, eos_token_id=12)
    scheduler.add(b)
    assert step(scheduler, runner) == [(2, [12])]
    assert scheduler.cache_manager.block_manager.blocks[0].ref_count == 1
    assert scheduler.cache_manager.lengths(1) == 9
    assert drain(scheduler, runner) == {1: [11, 12, 13]}


@pytest.mark.parametrize("seed", CASE_SEEDS, ids=lambda seed: f"seed-{seed}")
def test_seeded_arrivals_shared_lifetimes_and_bounded_progress(seed):
    rng = random.Random(seed)
    size, blocks = rng.choice([1, 2, 3, 4, 7]), rng.randint(4, 7)
    budget = size * blocks
    scheduler, runner = make_scheduler(blocks, size, budget)
    order = list(scheduler.cache_manager.block_manager.free_block_id)
    rng.shuffle(order)
    scheduler.cache_manager.block_manager.free_block_id.clear()
    scheduler.cache_manager.block_manager.free_block_id.update(OrderedDict.fromkeys(order))
    common = [rng.randrange(97) for _ in range(rng.randint(1, 2) * size)]
    first_budget = min(4, budget - len(common))
    first = Sequence(0, common + [3], first_budget)
    scheduler.add(first)
    expected = {0: [10 + i for i in range(first_budget)]}
    outputs = dict(step(scheduler, runner))
    scheduler.add(Sequence(1, common + [4], 1))
    expected[1] = [11]
    outputs.update(step(scheduler, runner))
    assert runner.events[1] == (True, ((1, len(common), 1),)), f"seed={seed} did not hit prefix"

    arrivals = []
    for sid in range(2, rng.randint(6, 11)):
        prompt = common + [rng.randrange(97)] if rng.random() < 0.6 else [rng.randrange(97)]
        maximum = min(5, budget - len(prompt) + 1)
        count = rng.randint(0, maximum)
        script = [rng.randrange(97) for _ in range(count)]
        eos = rng.choice(script) if script and rng.random() < 0.3 else None
        expected[sid] = script[:script.index(eos) + 1] if eos in script else script
        runner.scripts[sid] = script
        arrivals.append((rng.randint(0, 5), Sequence(sid, prompt, count, eos)))
    limit = 8 + first_budget + sum(seq.max_tokens for _, seq in arrivals)
    for tick in range(limit):
        for arrival, seq in arrivals:
            if arrival == tick:
                scheduler.add(seq)
                if seq.status == SequenceStatus.FINISHED:
                    outputs[seq.seq_id] = []  # Engine's ready_outputs behavior is checked by HF tests.
        for sid, tokens in step(scheduler, runner):
            assert sid not in outputs, f"seed={seed}, repeated completion {sid}"
            outputs[sid] = tokens
        for seq in scheduler.running:
            assert seq.num_cached_tokens == seq.num_tokens - 1 and seq.num_scheduled_tokens == 0
        for seq in scheduler.waiting:
            assert seq.num_cached_tokens == seq.num_scheduled_tokens == 0
        if tick >= 5 and scheduler.is_finished():
            break
    assert outputs == expected, f"seed={seed}"
    assert_idle_cache(scheduler.cache_manager)
