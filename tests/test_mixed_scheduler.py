"""Unified plans and three policies against the production scheduler/KV pool."""

import random
from dataclasses import FrozenInstanceError, replace

import pytest
from transformers import Qwen3Config

from nanovllm import Config
from nanovllm.engine.batch import BatchEntry, BatchPlan
from nanovllm.engine.input import prepare_mixed
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from numeric_helpers import CASE_SEEDS, make_context
from prefix_helpers import assert_block_conservation, assert_idle_cache, block_snapshot, make_cache
from test_chunked_prefill import execute_plan
from test_prefix_scheduler import ScriptedRunner


POLICIES = ("separate", "interleaved", "mixed")


def make_scheduler(policy="mixed", budget=4, count=2, size=4, blocks=8, enabled=False):
    manager = make_cache(blocks=blocks, size=size, enabled=enabled)
    return Scheduler(manager, count, budget, size * blocks, scheduling_policy=policy), ScriptedRunner(manager)


@pytest.mark.parametrize("policy", ["unknown", "", None, True])
def test_invalid_policy_is_rejected_before_any_request_or_model_load(policy, tmp_path):
    manager = make_cache()
    before = block_snapshot(manager.block_manager)
    with pytest.raises(ValueError, match="policy"):
        Scheduler(manager, 2, 4, 16, scheduling_policy=policy)
    with pytest.raises(ValueError, match="policy"):
        Config(str(tmp_path), scheduling_policy=policy)
    assert block_snapshot(manager.block_manager) == before
    assert_idle_cache(manager)


@pytest.mark.parametrize("policy", POLICIES)
def test_config_accepts_each_explicit_policy(policy, tmp_path):
    Qwen3Config().save_pretrained(tmp_path)
    assert Config(str(tmp_path), scheduling_policy=policy).scheduling_policy == policy


@pytest.mark.parametrize("policy,expected", [
    ("separate", ["prefill", "prefill", "decode", "decode", "decode"]),
    ("interleaved", ["decode", "prefill", "decode", "prefill", "decode"]),
    ("mixed", ["mixed", "mixed", "mixed"]),
])
def test_three_policies_have_distinct_rounds_and_identical_outputs(policy, expected):
    scheduler, runner = make_scheduler(policy, budget=3)
    scheduler.add(Sequence(0, [1, 2, 3], 4))
    execute_plan(scheduler, runner, scheduler.schedule())
    scheduler.add(Sequence(1, [4, 5, 6, 7, 8, 9], 1))
    kinds, outputs = [], {}
    for _ in range(16):
        if scheduler.is_finished():
            break
        plan = scheduler.schedule()
        kinds.append(plan.kind)
        outputs.update(execute_plan(scheduler, runner, plan))
    assert scheduler.is_finished()
    assert kinds == expected
    assert outputs == {0: [10, 11, 12, 13], 1: [11]}
    assert_idle_cache(runner.manager)


def test_mixed_partial_prefill_commits_kv_without_sampling_and_packs_boundaries():
    scheduler, runner = make_scheduler()
    a, b = Sequence(0, [1, 2, 3, 4], 3), Sequence(1, [5, 6, 7, 8, 9, 10], 1)
    runner.scripts = {0: [0, 1, 2], 1: [99]}
    scheduler.add(a)
    execute_plan(scheduler, runner, scheduler.schedule())
    scheduler.add(b)
    plan = scheduler.schedule()
    assert plan.kind == "mixed" and plan.legacy_flag is None
    assert [(e.phase, e.cache_len, e.query_len, e.should_sample) for e in plan.entries] == [
        ("decode", 4, 1, True), ("prefill", 0, 3, False)]
    ids, positions, ctx = prepare_mixed(plan.seqs, "cpu")
    assert ids.tolist() == [0, 5, 6, 7]
    assert positions.tolist() == [4, 0, 1, 2]
    assert ctx.cu_seq_lens_q == (0, 1, 4) and ctx.cu_seq_lens_k == (0, 5, 8)
    assert ctx.cache_lens == (4, 0) and ctx.context_lens == (5, 3)
    assert ctx.last_indices == (0, 3)
    tokens = runner.run_batch(plan)
    assert tokens == [1, None]
    assert a.completion_token_ids == [0] and b.completion_token_ids == []
    scheduler.postprocess(plan, tokens)
    assert a.completion_token_ids == [0, 1]
    assert b.num_cached_tokens == 3 and b.num_scheduled_tokens == 0
    assert b.status == SequenceStatus.WAITING and b in scheduler.waiting
    scheduler.abort()
    assert_idle_cache(runner.manager)


def test_prefix_hit_with_single_query_remains_prefill_and_samples_token_zero():
    scheduler, runner = make_scheduler(enabled=True, budget=8)
    a, b = Sequence(0, [1, 2, 3, 4], 3), Sequence(1, [1, 2, 3, 4, 9], 1)
    runner.scripts[1] = [0]
    scheduler.add(a)
    execute_plan(scheduler, runner, scheduler.schedule())
    scheduler.add(b)
    plan = scheduler.schedule()
    assert [(e.phase, e.cache_len, e.query_len, e.should_sample) for e in plan.entries] == [
        ("decode", 4, 1, True), ("prefill", 4, 1, True)]
    shared = runner.manager.block_manager.table(0)[0]
    assert runner.manager.block_manager.table(1)[0] == shared
    assert runner.manager.block_manager.blocks[shared].ref_count == 2
    assert execute_plan(scheduler, runner, plan) == [(1, [0])]
    assert runner.manager.block_manager.blocks[shared].ref_count == 1
    scheduler.abort()
    assert_idle_cache(runner.manager)


@pytest.mark.parametrize("budget,count", [(1, 3), (8, 1), (2, 2)])
def test_token_and_sequence_limits_apply_to_combined_batch(budget, count):
    scheduler, runner = make_scheduler(budget=budget, count=count)
    scheduler.add(Sequence(0, [1], 4))
    execute_plan(scheduler, runner, scheduler.schedule())
    scheduler.add(Sequence(1, list(range(6)), 2))
    kinds = []
    for _ in range(32):
        if scheduler.is_finished():
            break
        plan = scheduler.schedule()
        assert plan.query_tokens <= budget and len(plan.entries) <= count
        kinds.append(plan.kind)
        execute_plan(scheduler, runner, plan)
    assert scheduler.is_finished()
    assert ("mixed" in kinds) == (budget > 1 and count > 1)
    assert_idle_cache(runner.manager)


def test_new_prefill_cannot_preempt_selected_decode():
    scheduler, runner = make_scheduler(blocks=2, budget=8)
    a, b = Sequence(0, [1, 2, 3, 4], 3), Sequence(1, list(range(5)), 1)
    scheduler.add(a)
    execute_plan(scheduler, runner, scheduler.schedule())
    scheduler.add(b)
    plan = scheduler.schedule()
    assert plan.kind == "decode" and plan.seqs == (a,)
    assert b in scheduler.waiting and runner.manager.block_manager.table(1) == ()
    assert a.num_cached_tokens == 4
    execute_plan(scheduler, runner, plan)
    scheduler.abort()
    assert_idle_cache(runner.manager)


@pytest.mark.parametrize("policy", POLICIES)
def test_reserved_partial_prefill_recovers_when_preempted_head_cannot_fit(policy):
    scheduler, runner = make_scheduler(policy, budget=4, size=4, blocks=3)
    a, b = Sequence(0, [1, 2, 3], 4), Sequence(1, list(range(8)), 1)
    scheduler.add(a)
    execute_plan(scheduler, runner, scheduler.schedule())
    plan = scheduler.schedule()
    assert plan.kind == "decode"
    execute_plan(scheduler, runner, plan)  # A has C=4,N=5, occupying one full page.
    scheduler.add(b)
    scheduler.max_num_batched_tokens = 3
    # Seed the valid resource state with one manual prefill plan. Thereafter
    # schedule() must independently restore progress using the chosen policy.
    selected = scheduler._schedule_prefill(3, 2)
    assert selected == [b]
    plan = BatchPlan((BatchEntry(b, "prefill", 0, 3, 8),))
    scheduler.inflight = plan
    execute_plan(scheduler, runner, plan)  # B reserves two pages but has only C=3.
    scheduler.next_phase = "decode"
    assert runner.manager.lengths(1) == 3
    assert not runner.manager.block_manager.free_block_id
    outputs, recovered = {}, []
    for _ in range(16):
        if scheduler.is_finished():
            break
        plan = scheduler.schedule()
        recovered.extend(e for e in plan.entries if e.seq is a and e.phase == "prefill")
        outputs.update(execute_plan(scheduler, runner, plan))
    assert scheduler.is_finished()
    if policy != "separate":
        assert recovered, "decode must preempt A and resume the reserved B behind it"
    assert outputs == {0: [10, 11, 12, 13], 1: [11]}
    assert_idle_cache(runner.manager)


@pytest.mark.parametrize("bad_tokens", [[10], [10, None], [10, True], [10, 1.5]])
def test_all_results_are_validated_before_any_sequence_or_hash_commit(bad_tokens):
    scheduler, runner = make_scheduler(budget=8, enabled=True)
    seqs = [Sequence(i, [i, 2, 3, 4], 2) for i in range(2)]
    for seq in seqs:
        scheduler.add(seq)
    plan = scheduler.schedule()
    tokens = runner.run_batch(plan)
    before = block_snapshot(runner.manager.block_manager)
    states = [(list(s.token_ids), s.num_cached_tokens, s.num_scheduled_tokens) for s in seqs]
    with pytest.raises(ValueError):
        scheduler.postprocess(plan, bad_tokens)
    assert block_snapshot(runner.manager.block_manager) == before
    assert [(s.token_ids, s.num_cached_tokens, s.num_scheduled_tokens) for s in seqs] == states
    assert scheduler.inflight is plan
    scheduler.postprocess(plan, tokens)
    scheduler.abort()
    assert_idle_cache(runner.manager)


def test_partial_result_must_be_none_and_later_incomplete_layer_blocks_whole_commit():
    scheduler, runner = make_scheduler()
    scheduler.add(Sequence(0, [1], 4))
    execute_plan(scheduler, runner, scheduler.schedule())
    scheduler.add(Sequence(1, list(range(6)), 1))
    plan = scheduler.schedule()
    tokens = runner.run_batch(plan)
    histories = [list(s.token_ids) for s in plan.seqs]
    before = block_snapshot(runner.manager.block_manager)
    with pytest.raises(ValueError, match="partial prefill"):
        scheduler.postprocess(plan, [tokens[0], 99])
    runner.manager.layer_lengths[1][-1] -= 1
    with pytest.raises(RuntimeError, match="every layer"):
        scheduler.postprocess(plan, tokens)
    assert [s.token_ids for s in plan.seqs] == histories
    assert block_snapshot(runner.manager.block_manager) == before
    scheduler.abort()
    assert_idle_cache(runner.manager)


def test_snapshot_identity_reentrancy_and_repeated_commit_are_rejected():
    scheduler, runner = make_scheduler()
    seq = Sequence(0, [1], 2)
    scheduler.add(seq)
    plan = scheduler.schedule()
    with pytest.raises(FrozenInstanceError):
        plan.entries[0].cache_len = 9
    with pytest.raises(ValueError, match="previous batch"):
        scheduler.schedule()
    with pytest.raises(AssertionError):
        scheduler.add(Sequence(1, [2], 1))
    with pytest.raises(ValueError, match="inflight"):
        scheduler.postprocess(replace(plan), [10])
    seq.num_scheduled_tokens += 1
    with pytest.raises(ValueError, match="stale"):
        runner.run_batch(plan)
    seq.num_scheduled_tokens -= 1
    tokens = runner.run_batch(plan)
    scheduler.request[0] = Sequence(0, [1], 2)
    with pytest.raises(ValueError, match="identity"):
        scheduler.postprocess(plan, tokens)
    scheduler.request[0] = seq
    scheduler.postprocess(plan, tokens)
    with pytest.raises(ValueError, match="inflight"):
        scheduler.postprocess(plan, tokens)
    scheduler.abort()
    assert_idle_cache(runner.manager)


def test_abort_resets_interleaved_preference_and_releases_partial_waiting():
    scheduler, runner = make_scheduler("interleaved", budget=2)
    scheduler.add(Sequence(0, list(range(6)), 1))
    execute_plan(scheduler, runner, scheduler.schedule())
    assert scheduler.next_phase == "decode"
    assert scheduler.waiting[0].num_cached_tokens == 2
    scheduler.abort()
    assert scheduler.next_phase == "prefill" and scheduler.is_finished()
    assert_idle_cache(runner.manager)
    scheduler.add(Sequence(1, [4], 1))
    assert execute_plan(scheduler, runner, scheduler.schedule()) == [(1, [11])]
    assert_idle_cache(runner.manager)


@pytest.mark.parametrize("phase,c,q,status", [
    ("unknown", 0, 1, SequenceStatus.WAITING),
    ("prefill", 0, 0, SequenceStatus.WAITING),
    ("prefill", -1, 1, SequenceStatus.WAITING),
    ("prefill", 1, 2, SequenceStatus.WAITING),
    ("prefill", 0, 1, SequenceStatus.FINISHED),
    ("decode", 0, 2, SequenceStatus.RUNNING),
    ("decode", 1, 1, SequenceStatus.WAITING),
])
def test_invalid_entry_intervals_and_phases_are_rejected(phase, c, q, status):
    seq = Sequence(0, [1, 2], 2)
    seq.num_cached_tokens, seq.num_scheduled_tokens, seq.status = c, q, status
    with pytest.raises(ValueError):
        BatchEntry(seq, phase, c, q, 2).validate_live()


def test_empty_duplicate_plans_and_wrong_candidate_count_are_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        BatchPlan(()).validate_live()
    seq = Sequence(0, [1], 2)
    seq.num_scheduled_tokens = 1
    entry = BatchEntry(seq, "prefill", 0, 1, 1)
    with pytest.raises(ValueError, match="unique"):
        BatchPlan((entry, entry)).validate_live()
    runner = ModelRunner.__new__(ModelRunner)
    runner.run = lambda seqs, flag: []
    with pytest.raises(ValueError, match="candidate"):
        runner.run_batch(BatchPlan((entry,)))


@pytest.mark.parametrize("flag", [False, None])
def test_unallocated_pool_rejects_decode_and_mixed_warmup(flag):
    from nanovllm.runtime.paged_cache import PagedKVCacheManager
    manager = PagedKVCacheManager(1, None, 4, 1, 2)
    ctx = replace(make_context([0], [0], [1]), is_prefill=flag)
    with pytest.raises(ValueError, match="Unallocated"):
        manager.prepare_ctx(ctx)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("seed", CASE_SEEDS, ids=lambda seed: f"seed-{seed}")
def test_seeded_dynamic_arrivals_eos_zero_budget_and_resource_conservation(policy, seed):
    rng = random.Random(seed + 400_000)
    scheduler, runner = make_scheduler(policy, budget=rng.randint(1, 7), count=rng.randint(1, 4),
                                       size=rng.choice([1, 3, 4]), blocks=12, enabled=bool(seed % 2))
    arrivals, expected, outputs = [], {}, {}
    for sid in range(8):
        prompt = [rng.randrange(97) for _ in range(rng.randint(1, 6))]
        count = 0 if sid == 0 else rng.randint(1, 5)
        script = [rng.randrange(97) for _ in range(count)]
        eos = script[0] if script and sid % 3 == 0 else None
        runner.scripts[sid] = script
        expected[sid] = script[:1] if eos is not None else script
        arrivals.append((rng.randint(0, 5), Sequence(sid, prompt, count, eos)))
    for tick in range(256):
        for arrival, seq in arrivals:
            if arrival == tick:
                scheduler.add(seq)
                if seq.status == SequenceStatus.FINISHED:
                    outputs[seq.seq_id] = []
        plan = scheduler.schedule()
        if plan is not None:
            for sid, tokens in execute_plan(scheduler, runner, plan):
                assert sid not in outputs
                outputs[sid] = tokens
        assert_block_conservation(runner.manager.block_manager)
        for seq in scheduler.waiting:
            assert seq.num_cached_tokens < seq.num_tokens and seq.num_scheduled_tokens == 0
        for seq in scheduler.running:
            assert seq.num_cached_tokens == seq.num_tokens - 1 and seq.num_scheduled_tokens == 0
        if tick >= 5 and scheduler.is_finished():
            break
    assert scheduler.is_finished() and outputs == expected, f"seed={seed}, policy={policy}"
    assert_idle_cache(runner.manager)
