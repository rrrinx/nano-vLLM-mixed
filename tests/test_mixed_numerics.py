"""Actual mixed Qwen3 forwards: every query logit/layer KV and greedy vs HF."""

import atexit

import pytest
import torch

from nanovllm import LLM, SamplingParams
from numeric_helpers import hf_greedy_tokens, random_tokens, shuffled_blocks
from prefix_helpers import (
    assert_block_conservation, assert_blocks_unchanged, assert_idle_cache,
    observe_numerical_runner, save_blocks,
)
from test_chunked_numerics import close_engine, collect_steps


POLICIES = ("separate", "interleaved", "mixed")


def observe_plans(engine, monkeypatch):
    records = []
    original = engine.scheduler.schedule

    def schedule():
        plan = original()
        if plan is not None:
            records.append({"kind": plan.kind, "entries": tuple(
                (e.seq.seq_id, e.phase, e.cache_len, e.query_len, e.num_tokens, e.should_sample)
                for e in plan.entries)})
        return plan

    monkeypatch.setattr(engine.scheduler, "schedule", schedule)
    return records


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("enabled", [False, True], ids=["cache-off", "cache-on"])
@torch.inference_mode()
def test_three_policies_dynamic_chunks_eos_and_public_outputs_match_hf(model_case, monkeypatch,
                                                                     policy, enabled):
    seed, rng, reference, model_dir, device = model_case
    prompts = [random_tokens(rng, n, reference.config.vocab_size) for n in (4, 9, 2)]
    # This scenario needs a cold B; random_tokens deliberately emits repeated
    # boundary patterns, so two independently generated prompts can share A.
    prompts[1][0] = (prompts[0][0] + 1) % reference.config.vocab_size
    engine = LLM(
        str(model_dir), device=str(device), dtype="float32", scheduling_policy=policy,
        enable_prefix_cache=enabled, max_model_len=24, max_num_batched_tokens=4,
        max_num_seqs=3, kvcache_block_size=4, num_kvcache_blocks=16,
    )
    manager = engine.model_runner.kv_cache_manager
    manager.k_pool.fill_(torch.nan)
    manager.v_pool.fill_(torch.nan)
    shuffled_blocks(manager, rng)
    events, handle = observe_numerical_runner(engine, reference, monkeypatch)
    plans = observe_plans(engine, monkeypatch)
    try:
        a = engine.add_request(prompts[0], SamplingParams(max_tokens=6, ignore_eos=True))
        assert engine.step() == []
        b = engine.add_request(prompts[1], SamplingParams(max_tokens=3, ignore_eos=True))
        assert engine.step() == []
        eos = hf_greedy_tokens(reference, prompts[2], 1)[0]
        c = engine.add_request(" ".join(f"t{token}_" for token in prompts[2]),
                               SamplingParams(max_tokens=4), eos_token_id=eos)
        zero = engine.add_request([0], SamplingParams(max_tokens=0))
        expected = {a: hf_greedy_tokens(reference, prompts[0], 6),
                    b: hf_greedy_tokens(reference, prompts[1], 3), c: [eos], zero: []}
        assert collect_steps(engine) == expected, f"seed={seed}, policy={policy}"
        assert len(events) == len(plans), "each plan must execute exactly one packed forward"
        assert all(sum(event["queries"]) <= 4 and len(event["seq_ids"]) <= 3 for event in events)
        assert any(not sample for plan in plans for _, _, _, _, _, sample in plan["entries"])
        if policy == "mixed":
            assert plans[1]["kind"] == "mixed"
            assert plans[1]["entries"][0][1:4] == ("decode", 4, 1)
            assert plans[1]["entries"][1][1:4] == ("prefill", 0, 3)
            assert events[1]["prefill"] is None
            assert events[1]["ids"] == expected[a][:1] + prompts[1][:3]
            assert events[1]["positions"] == [4, 0, 1, 2]
        else:
            assert all(plan["kind"] != "mixed" for plan in plans)
        assert_idle_cache(manager)
        # Public generate is reusable after all dynamic requests complete.
        actual = engine.generate([prompts[2]], SamplingParams(max_tokens=2, ignore_eos=True), use_tqdm=False)
        assert actual[0]["token_ids"] == hf_greedy_tokens(reference, prompts[2], 2)
        assert_idle_cache(manager)
    finally:
        close_engine(engine, handle)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("enabled", [False, True], ids=["cache-off", "cache-on"])
@torch.inference_mode()
def test_three_policies_preempt_and_rebuild_complete_generated_history(model_case, monkeypatch,
                                                                     policy, enabled):
    seed, rng, reference, model_dir, device = model_case
    common = random_tokens(rng, 4 if enabled else 3, reference.config.vocab_size)
    private = rng.randrange(reference.config.vocab_size)
    prompts = [common + [private], common + [(private + 1) % reference.config.vocab_size]]
    engine = LLM(
        str(model_dir), device=str(device), dtype="float32", scheduling_policy=policy,
        enable_prefix_cache=enabled, max_model_len=12, max_num_batched_tokens=3,
        max_num_seqs=2, kvcache_block_size=4, num_kvcache_blocks=3,
    )
    manager = engine.model_runner.kv_cache_manager
    shuffled_blocks(manager, rng)
    events, handle = observe_numerical_runner(engine, reference, monkeypatch)
    plans = observe_plans(engine, monkeypatch)
    victims = []
    original = engine.scheduler.preempt

    def preempt(seq):
        history, completion = list(seq.token_ids), list(seq.completion_token_ids)
        victims.append((seq.seq_id, history, completion, len(plans)))
        original(seq)
        assert seq.token_ids == history and seq.completion_token_ids == completion
        assert seq.num_cached_tokens == seq.num_scheduled_tokens == 0
        assert_block_conservation(manager.block_manager)

    monkeypatch.setattr(engine.scheduler, "preempt", preempt)
    try:
        a = engine.add_request(prompts[0], SamplingParams(max_tokens=6, ignore_eos=True))
        while not engine.scheduler.request[a].num_completion_tokens:
            assert engine.step() == []
        shared = manager.block_manager.table(a)[:1]
        before = save_blocks(manager, shared)
        engine.add_request(prompts[1], SamplingParams(max_tokens=6, ignore_eos=True))
        actual = collect_steps(engine)
        assert actual == {i: hf_greedy_tokens(reference, p, 6) for i, p in enumerate(prompts)}
        assert victims, f"seed={seed}, policy={policy}: no actual preemption"
        recovered = []
        for sid, history, completion, start in victims:
            assert completion and history == prompts[sid] + completion
            recovered.extend(e for plan in plans[start:] for e in plan["entries"]
                             if e[0] == sid and e[1] == "prefill" and e[2] + e[3] >= len(history))
        assert recovered, f"seed={seed}: did not reconstruct generated history"
        if enabled:
            assert_blocks_unchanged(manager, shared, before)
        assert_idle_cache(manager)
    finally:
        close_engine(engine, handle)


@torch.inference_mode()
def test_mixed_one_query_prefix_suffix_and_decode_are_numerically_independent(model_case, monkeypatch):
    _, rng, reference, model_dir, device = model_case
    common = random_tokens(rng, 4, reference.config.vocab_size)
    b_prompt = common + [rng.randrange(reference.config.vocab_size)]
    engine = LLM(
        str(model_dir), device=str(device), dtype="float32", scheduling_policy="mixed",
        enable_prefix_cache=True, max_model_len=16, max_num_batched_tokens=8,
        max_num_seqs=2, kvcache_block_size=4, num_kvcache_blocks=8,
    )
    events, handle = observe_numerical_runner(engine, reference, monkeypatch)
    plans = observe_plans(engine, monkeypatch)
    manager = engine.model_runner.kv_cache_manager
    try:
        engine.add_request(common, SamplingParams(max_tokens=3, ignore_eos=True))
        assert engine.step() == []
        shared = manager.block_manager.table(0)[:1]
        before = save_blocks(manager, shared)
        engine.add_request(b_prompt, SamplingParams(max_tokens=1, ignore_eos=True))
        result = dict(engine.step())
        assert result == {1: hf_greedy_tokens(reference, b_prompt, 1)}
        assert plans[-1]["kind"] == "mixed"
        assert [(e[1], e[2], e[3], e[5]) for e in plans[-1]["entries"]] == [
            ("decode", 4, 1, True), ("prefill", 4, 1, True)]
        assert events[-1]["queries"] == (1, 1) and events[-1]["cached"] == (4, 4)
        assert collect_steps(engine) == {0: hf_greedy_tokens(reference, common, 3)}
        assert_blocks_unchanged(manager, shared, before)
        assert_idle_cache(manager)
    finally:
        close_engine(engine, handle)


@pytest.mark.parametrize("enabled", [False, True], ids=["cache-off", "cache-on"])
@torch.inference_mode()
def test_mixed_failure_after_kv_write_aborts_both_running_and_partial_waiting(model_case, monkeypatch,
                                                                           enabled):
    _, rng, reference, model_dir, device = model_case
    engine = LLM(
        str(model_dir), device=str(device), dtype="float32", scheduling_policy="mixed",
        enable_prefix_cache=enabled, max_model_len=16, max_num_batched_tokens=4,
        max_num_seqs=2, kvcache_block_size=4, num_kvcache_blocks=8,
    )
    manager = engine.model_runner.kv_cache_manager
    try:
        a_prompt = random_tokens(rng, 4, reference.config.vocab_size)
        b_prompt = random_tokens(rng, 9, reference.config.vocab_size)
        b_prompt[0] = (a_prompt[0] + 1) % reference.config.vocab_size
        engine.add_request(a_prompt,
                           SamplingParams(max_tokens=3, ignore_eos=True))
        assert engine.step() == []
        engine.add_request(b_prompt,
                           SamplingParams(max_tokens=1, ignore_eos=True))
        a, b = engine.scheduler.request.values()
        histories = [list(a.token_ids), list(b.token_ids)]
        published = dict(manager.block_manager.hash_to_block_id)
        backend = engine.model_runner.model.model.layers[0].self_attn.backend
        original = backend.attend

        def fail(*args, **kwargs):
            original(*args, **kwargs)
            assert manager.layer_lengths[0][0] == 5
            assert manager.layer_lengths[1][0] == 3
            raise RuntimeError("injected mixed first-layer failure")

        monkeypatch.setattr(backend, "attend", fail)
        with pytest.raises(RuntimeError, match="mixed first-layer"):
            engine.step()
        assert engine.failed and engine.scheduler.inflight is None
        assert [a.token_ids, b.token_ids] == histories
        assert a.num_cached_tokens == b.num_cached_tokens == 0
        assert a.num_scheduled_tokens == b.num_scheduled_tokens == 0
        assert manager.block_manager.hash_to_block_id == published
        assert_idle_cache(manager)
        for operation in (engine.step, engine.is_finished, lambda: engine.add_request([0]),
                          lambda: engine.generate([[0]], use_tqdm=False)):
            with pytest.raises(RuntimeError, match="failed"):
                operation()
    finally:
        engine.exit()
        atexit.unregister(engine.exit)
