"""Stage 8 scheduling/storage contracts; synthetic KV never replaces the allocator."""

import random
from collections import OrderedDict

import pytest
import torch

from nanovllm.engine.input import prepare_decode, prepare_prefill
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from numeric_helpers import CASE_SEEDS
from prefix_helpers import assert_block_conservation, assert_idle_cache, block_snapshot, make_cache
from test_prefix_scheduler import ScriptedRunner


def make_chunk_scheduler(budget=3, size=4, blocks=8, count=2, enabled=True):
    manager = make_cache(blocks=blocks, size=size, enabled=enabled)
    return Scheduler(manager, count, budget, size * blocks), ScriptedRunner(manager)


def execute_plan(scheduler, runner, plan):
    """Check reservation, real all-layer writes, then logical commit separately."""
    seqs = plan.seqs
    manager = scheduler.cache_manager
    assert 0 < sum(s.num_scheduled_tokens for s in seqs) <= scheduler.max_num_batched_tokens
    assert len(seqs) <= scheduler.max_num_seqs
    states = [(list(s.token_ids), s.num_cached_tokens, s.num_scheduled_tokens) for s in seqs]
    for seq, (_, c, q) in zip(seqs, states):
        assert manager.lengths(seq.seq_id) == c
        assert len(manager.block_manager.table(seq.seq_id)) * manager.block_size >= c + q
    published_before = dict(manager.block_manager.hash_to_block_id)
    tokens = runner.run_batch(plan)
    assert manager.block_manager.hash_to_block_id == published_before
    for seq, (history, c, q) in zip(seqs, states):
        assert (seq.token_ids, seq.num_cached_tokens, seq.num_scheduled_tokens) == (history, c, q)
        assert manager.lengths(seq.seq_id) == c + q
    scheduler.postprocess(plan, tokens)
    for seq, token, (history, c, q) in zip(seqs, tokens, states):
        assert seq.num_cached_tokens == c + q and seq.num_scheduled_tokens == 0
        if c + q < len(history):
            assert seq.token_ids == history and seq.status == SequenceStatus.WAITING
            assert seq in scheduler.waiting and seq not in scheduler.running
        else:
            assert seq.token_ids == history + [token]
    assert scheduler.inflight is None
    assert_block_conservation(manager.block_manager)
    return [(seq.seq_id, list(seq.completion_token_ids)) for seq in seqs
            if seq.status == SequenceStatus.FINISHED]


def drain(scheduler, runner, limit=128):
    outputs = {}
    for _ in range(limit):
        plan = scheduler.schedule()
        if plan is None:
            assert_idle_cache(scheduler.cache_manager)
            return outputs
        for sid, tokens in execute_plan(scheduler, runner, plan):
            assert sid not in outputs
            outputs[sid] = tokens
    pytest.fail("chunked requests failed to make bounded progress")


@pytest.mark.parametrize("budget", [1, 3, 4, 10, 11])
@pytest.mark.parametrize("enabled", [False, True], ids=["cache-off", "cache-on"])
def test_chunk_progress_reserves_once_and_ignores_intermediate_eos(budget, enabled, monkeypatch):
    scheduler, runner = make_chunk_scheduler(budget=budget, enabled=enabled)
    manager = runner.manager
    seq = Sequence(17, list(range(10)), 3, eos_token_id=99)
    runner.scripts[17] = [99]
    allocations = []
    original = manager.allocate_prefill

    def allocate(request, hits):
        allocations.append(request.seq_id)
        return original(request, hits)

    monkeypatch.setattr(manager, "allocate_prefill", allocate)
    scheduler.add(seq)  # N and model capacity may exceed the per-step budget.
    assert manager.block_manager.table(17) == ()
    assert manager.lengths(17) == 0
    scheduled = []
    while not scheduler.is_finished():
        plan = scheduler.schedule()
        assert plan.seqs == (seq,) and plan.kind == "prefill"
        q = seq.num_scheduled_tokens
        scheduled.append(q)
        assert len(manager.block_manager.table(17)) == 3  # Entire prompt reserved on first chunk.
        assert seq.status == (SequenceStatus.WAITING if seq.num_cached_tokens + q < 10
                              else SequenceStatus.RUNNING)
        with pytest.raises(ValueError, match="previous batch"):
            scheduler.schedule()
        finished = execute_plan(scheduler, runner, plan)
        assert finished == ([(17, [99])] if seq.num_cached_tokens == 10 else [])
    assert scheduled == [min(budget, 10 - c) for c in range(0, 10, budget)]
    assert allocations == [17]
    assert seq.completion_token_ids == [99]  # Earlier EOS candidates did not finish the request.
    assert_idle_cache(manager)


def test_non_block_aligned_slots_publication_and_previous_kv_preservation():
    scheduler, runner = make_chunk_scheduler(blocks=4)
    manager = runner.manager
    manager.block_manager.free_block_id = OrderedDict.fromkeys([2, 0, 3, 1])
    seq = Sequence(0, list(range(10, 20)), 1)
    scheduler.add(seq)
    expected_slots = [(8, 9, 10), (11, 0, 1), (2, 3, 12), (13,)]
    for turn, slots in enumerate(expected_slots):
        plan = scheduler.schedule()
        c, q = seq.num_cached_tokens, seq.num_scheduled_tokens
        ids, positions, ctx = prepare_prefill(plan.seqs, "cpu")
        ctx = manager.prepare_ctx(ctx)
        assert manager.block_manager.table(0) == (2, 0, 3)
        assert ctx.slot_mapping == slots
        assert ids.tolist() == list(range(10 + c, 10 + c + q))
        assert positions.tolist() == list(range(c, c + q))
        assert ctx.cache_lens == (c,) and ctx.context_lens == (c + q,)
        assert ctx.cu_seq_lens_q == (0, q) and ctx.cu_seq_lens_k == (0, c + q)
        previous = [tuple(t.clone() for t in manager.gather(layer, 0, c, ctx.block_tables[0]))
                    for layer in range(manager.num_layers)]
        execute_plan(scheduler, runner, plan)
        assert sum(b.hash != -1 for b in manager.block_manager.blocks) == [0, 1, 2, 2][turn]
        if seq.status != SequenceStatus.FINISHED:
            for layer, wanted in enumerate(previous):
                actual = manager.gather(layer, 0, c, ctx.block_tables[0])
                for a, b in zip(actual, wanted):
                    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert_idle_cache(manager)


@pytest.mark.parametrize("second_length,expected_ids", [(2, [0, 1]), (5, [0])])
def test_tail_chunk_batches_only_with_a_complete_later_suffix(second_length, expected_ids):
    scheduler, runner = make_chunk_scheduler(budget=4, enabled=False)
    first = Sequence(0, list(range(6)), 1)
    second = Sequence(1, list(range(second_length)), 1)
    scheduler.add(first)
    execute_plan(scheduler, runner, scheduler.schedule())
    scheduler.add(second)
    plan = scheduler.schedule()
    assert [s.seq_id for s in plan.seqs] == expected_ids
    assert first.num_scheduled_tokens == 2
    if second_length == 5:
        assert runner.manager.block_manager.table(1) == ()
        assert second.num_cached_tokens == second.num_scheduled_tokens == 0
    else:
        ids, positions, ctx = prepare_prefill(plan.seqs, "cpu")
        assert ids.tolist() == [4, 5, 0, 1] and positions.tolist() == [4, 5, 0, 1]
        assert ctx.last_indices == (1, 3)
    execute_plan(scheduler, runner, plan)
    drain(scheduler, runner)


def test_sequence_limit_and_decode_token_budget_are_both_enforced():
    scheduler, runner = make_chunk_scheduler(budget=1, count=3, enabled=False)
    for sid in range(3):
        scheduler.add(Sequence(sid, [sid], 2))
    for _ in range(3):
        plan = scheduler.schedule()
        assert plan.kind == "prefill" and len(plan.entries) == 1
        execute_plan(scheduler, runner, plan)
    plan = scheduler.schedule()
    assert plan.kind == "decode" and len(plan.entries) == 1
    execute_plan(scheduler, runner, plan)
    drain(scheduler, runner)
    scheduler, runner = make_chunk_scheduler(budget=8, count=1)
    scheduler.add(Sequence(0, [1], 1))
    scheduler.add(Sequence(1, [2], 1))
    assert len(scheduler.schedule().entries) == 1
    scheduler.abort()
    assert_idle_cache(runner.manager)


def test_capacity_wait_is_not_confused_with_a_small_compute_budget():
    scheduler, runner = make_chunk_scheduler(budget=4, blocks=3, enabled=False)
    a, b = Sequence(0, list(range(9)), 1), Sequence(1, list(range(5)), 1)
    scheduler.add(a)
    scheduler.add(b)
    plans = []
    for _ in range(5):
        plan = scheduler.schedule()
        plans.append([(seq.seq_id, seq.num_scheduled_tokens) for seq in plan.seqs])
        if len(plans) <= 3:
            assert runner.manager.block_manager.table(1) == ()
            assert b.num_cached_tokens == b.num_scheduled_tokens == 0
        execute_plan(scheduler, runner, plan)
    assert plans == [[(0, 4)], [(0, 4)], [(0, 1)], [(1, 4)], [(1, 1)]]
    assert_idle_cache(runner.manager)


@pytest.mark.parametrize("enabled", [False, True])
def test_partial_layer_failure_cannot_publish_and_abort_releases_waiting(enabled):
    scheduler, runner = make_chunk_scheduler(enabled=enabled)
    seq = Sequence(0, list(range(10)), 2)
    scheduler.add(seq)
    execute_plan(scheduler, runner, scheduler.schedule())  # C=3, no full block yet.
    plan = scheduler.schedule()
    assert seq.status == SequenceStatus.WAITING and seq.num_cached_tokens == 3
    before = dict(runner.manager.block_manager.hash_to_block_id)
    runner.fail_after_first_layer = True
    with pytest.raises(RuntimeError, match="partial forward"):
        runner.run_batch(plan)
    assert runner.manager.layer_lengths[0] == [6, 3]
    with pytest.raises(RuntimeError, match="every layer"):
        scheduler.postprocess(plan, [None])
    assert runner.manager.block_manager.hash_to_block_id == before
    assert seq.num_cached_tokens == 3 and seq.num_scheduled_tokens == 3
    scheduler.abort()
    assert scheduler.is_finished() and not scheduler.request
    assert_idle_cache(runner.manager)


@pytest.mark.parametrize("tokens,maximum,model_len", [(9, 1, 8), (7, 3, 16)])
def test_infeasible_request_is_rejected_before_registration(tokens, maximum, model_len):
    manager = make_cache(blocks=2, size=4)
    scheduler = Scheduler(manager, 2, 1, model_len)
    before = block_snapshot(manager.block_manager)
    with pytest.raises(AssertionError):
        scheduler.add(Sequence(0, list(range(tokens)), maximum))
    assert block_snapshot(manager.block_manager) == before
    assert_idle_cache(manager)


def test_zero_generation_budget_uses_no_chunk_or_storage():
    scheduler, runner = make_chunk_scheduler(budget=1)
    scheduler.add(Sequence(0, list(range(10)), 0))
    assert scheduler.schedule() is None and runner.events == []
    assert_idle_cache(runner.manager)


@pytest.mark.parametrize("status,c,q,prefill", [
    (SequenceStatus.FINISHED, 0, 1, True),
    (SequenceStatus.WAITING, 0, 0, True),
    (SequenceStatus.WAITING, -1, 1, True),
    (SequenceStatus.WAITING, 3, 2, True),
    (SequenceStatus.WAITING, 3, 1, False),
    (SequenceStatus.RUNNING, 2, 2, False),
    (SequenceStatus.RUNNING, 1, 1, False),
])
def test_invalid_chunk_and_decode_plans_are_rejected(status, c, q, prefill):
    seq = Sequence(0, [1, 2, 3, 4], 2)
    seq.status, seq.num_cached_tokens, seq.num_scheduled_tokens = status, c, q
    prepare = prepare_prefill if prefill else prepare_decode
    with pytest.raises(ValueError):
        prepare([seq], "cpu")


@pytest.mark.parametrize("seed", CASE_SEEDS, ids=lambda seed: f"seed-{seed}")
def test_seeded_chunked_preemption_recovers_generated_history(seed, monkeypatch):
    rng = random.Random(seed)
    size = rng.choice([4, 5, 7])
    budget = rng.randint(2, size - 1)  # Decode must select both requests to force eviction.
    scheduler, runner = make_chunk_scheduler(budget, size, blocks=3, enabled=False)
    order = list(runner.manager.block_manager.free_block_id)
    rng.shuffle(order)
    runner.manager.block_manager.free_block_id = OrderedDict.fromkeys(order)
    seqs = [Sequence(sid, [rng.randrange(97) for _ in range(size)], 3) for sid in range(2)]
    for seq in seqs:
        scheduler.add(seq)
    victims = []
    original = scheduler.preempt

    def preempt(seq):
        history, completion = list(seq.token_ids), list(seq.completion_token_ids)
        original(seq)
        assert seq.token_ids == history and seq.completion_token_ids == completion
        assert seq.num_cached_tokens == seq.num_scheduled_tokens == 0
        assert runner.manager.block_manager.table(seq.seq_id) == ()
        victims.append(seq.seq_id)

    monkeypatch.setattr(scheduler, "preempt", preempt)
    outputs = drain(scheduler, runner)
    assert outputs == {0: [10, 11, 12], 1: [11, 12, 13]}
    assert victims == [1], f"seed={seed}: no actual eviction"
    recovery = [(c, q) for prefill, rows in runner.events if prefill
                for sid, c, q in rows if sid == 1 and c + q > size]
    assert recovery and recovery[-1][0] + recovery[-1][1] == size + 1
    # At least two rounds are needed to reconstruct N=P+1, preserving prior completion.
    assert sum(1 for prefill, rows in runner.events if prefill
               for sid, c, q in rows if sid == 1 and c == 0) == 2


@pytest.mark.parametrize("seed", CASE_SEEDS, ids=lambda seed: f"seed-{seed}")
def test_seeded_arrivals_partial_waiting_eos_and_zero_budget(seed):
    rng = random.Random(seed + 300_000)
    size, blocks = rng.choice([2, 4, 7]), rng.randint(6, 9)
    budget = rng.randint(1, size)
    scheduler, runner = make_chunk_scheduler(budget, size, blocks, enabled=True)
    arrivals, expected = [], {}
    for sid in range(rng.randint(5, 9)):
        length = 2 * size + 1 if sid == 0 else rng.randint(1, 2 * size + 1)
        count = 3 if sid == 0 else rng.randint(0, 4)
        prompt = [rng.randrange(97) for _ in range(length)]
        script = [rng.randrange(97) for _ in range(count)]
        eos = rng.choice(script) if script and rng.choice([False, True]) else None
        seq = Sequence(sid, prompt, count, eos)
        arrivals.append((0 if sid == 0 else rng.randint(1, 6), seq))
        runner.scripts[sid] = script
        expected[sid] = script[:script.index(eos) + 1] if eos in script else script
    outputs, partial_steps = {}, 0
    for tick in range(512):
        for arrival, seq in arrivals:
            if arrival == tick:
                scheduler.add(seq)
                if seq.status == SequenceStatus.FINISHED:
                    outputs[seq.seq_id] = []  # Mirrors Engine.ready_outputs for G=0.
        plan = scheduler.schedule()
        if plan is not None:
            partial_steps += sum(s.num_cached_tokens + s.num_scheduled_tokens < s.num_tokens
                                 for s in plan.seqs)
            for sid, tokens in execute_plan(scheduler, runner, plan):
                assert sid not in outputs
                outputs[sid] = tokens
        for seq in scheduler.waiting:
            assert 0 <= seq.num_cached_tokens < seq.num_tokens and seq.num_scheduled_tokens == 0
        for seq in scheduler.running:
            assert seq.num_cached_tokens == seq.num_tokens - 1 and seq.num_scheduled_tokens == 0
        if tick >= 6 and scheduler.is_finished():
            break
    assert partial_steps > 0, f"seed={seed}: no chunk was actually exercised"
    assert outputs == expected, f"seed={seed}: incomplete or incorrect completions"
    assert_idle_cache(runner.manager)
