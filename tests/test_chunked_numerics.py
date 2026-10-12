"""Actual chunked Qwen3 execution: all query logits/KV vs independent full-history HF."""

import atexit

import pytest
import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import SequenceStatus
from numeric_helpers import hf_greedy_tokens, random_tokens, shuffled_blocks
from prefix_helpers import (
    assert_blocks_unchanged, assert_idle_cache, observe_numerical_runner, save_blocks,
)


def close_engine(engine, handle):
    handle.remove()
    engine.exit()
    atexit.unregister(engine.exit)


def collect_steps(engine, limit=128):
    results = {}
    for _ in range(limit):
        if engine.is_finished():
            assert_idle_cache(engine.model_runner.kv_cache_manager)
            return results
        for sid, tokens in engine.step():
            assert sid not in results, "request completed twice"
            results[sid] = tokens
    pytest.fail("chunked engine failed to make bounded progress")


@pytest.mark.parametrize("enabled", [False, True], ids=["cache-off", "cache-on"])
@pytest.mark.parametrize("budget", [1, 3], ids=["one-query", "three-queries"])
@torch.inference_mode()
def test_chunked_and_full_public_generation_all_logits_kv_and_tokens(model_case, monkeypatch,
                                                                    enabled, budget):
    seed, rng, reference, model_dir, device = model_case
    size = rng.choice([2, 4, 5])  # T=1/3 differs from the physical block size.
    length = 2 * size + budget + 1
    prompt = random_tokens(rng, length, reference.config.vocab_size)
    short = random_tokens(rng, 2, reference.config.vocab_size)
    expected = [[], hf_greedy_tokens(reference, prompt, 3), hf_greedy_tokens(reference, short, 2)]
    outputs_by_budget = []
    for token_budget in (budget, 32):
        engine = LLM(
            str(model_dir), device=str(device), dtype="float32", enable_prefix_cache=enabled,
            scheduling_policy="separate",
            max_model_len=32, max_num_batched_tokens=token_budget, max_num_seqs=2,
            kvcache_block_size=size, num_kvcache_blocks=32,
        )
        manager = engine.model_runner.kv_cache_manager
        shuffled_blocks(manager, rng)
        events, handle = observe_numerical_runner(engine, reference, monkeypatch)
        try:
            inputs = [short, " ".join(f"t{token}_" for token in prompt), short]
            params = [SamplingParams(max_tokens=count, ignore_eos=True) for count in (0, 3, 2)]
            actual = engine.generate(inputs, params, use_tqdm=False)
            actual = [result["token_ids"] for result in actual]
            assert actual == expected, f"seed={seed}, budget={token_budget}"
            outputs_by_budget.append(actual)
            assert events and all(sum(e["queries"]) <= token_budget for e in events)
            long_prefill = [(e["cached"][i], e["queries"][i])
                            for e in events if e["prefill"]
                            for i, sid in enumerate(e["seq_ids"]) if sid == 1]
            assert sum(q for _, q in long_prefill) == length
            assert bool(any(c + q < length for c, q in long_prefill)) == (token_budget == budget)
            assert all(0 not in e["seq_ids"] for e in events)  # Zero budget never executes.
            assert_idle_cache(manager)
        finally:
            close_engine(engine, handle)
    assert outputs_by_budget[0] == outputs_by_budget[1]


@torch.inference_mode()
def test_chunked_suffix_after_live_prefix_hit_and_cold_reuse(model_case, monkeypatch):
    _, rng, reference, model_dir, device = model_case
    size = rng.choice([3, 4, 5])
    budget = size - 1
    common = random_tokens(rng, size, reference.config.vocab_size)
    private = rng.randrange(reference.config.vocab_size)
    a_prompt = common + [private]
    b_prompt = common + [(private + 1) % reference.config.vocab_size]
    b_prompt += random_tokens(rng, 2 * size, reference.config.vocab_size)
    engine = LLM(
        str(model_dir), device=str(device), dtype="float32", enable_prefix_cache=True,
        scheduling_policy="separate",
        max_model_len=24, max_num_batched_tokens=budget, max_num_seqs=2,
        kvcache_block_size=size, num_kvcache_blocks=12,
    )
    manager = engine.model_runner.kv_cache_manager
    events, handle = observe_numerical_runner(engine, reference, monkeypatch)
    try:
        sid_a = engine.add_request(a_prompt, SamplingParams(max_tokens=3, ignore_eos=True))
        a = engine.scheduler.request[sid_a]
        while not a.num_completion_tokens:
            assert engine.step() == []
        shared = manager.block_manager.table(sid_a)[:1]
        before = save_blocks(manager, shared)
        sid_b = engine.add_request(b_prompt, SamplingParams(max_tokens=2, ignore_eos=True))
        b = engine.scheduler.request[sid_b]
        start = len(events)
        assert engine.step() == []
        assert b.status == SequenceStatus.WAITING and b.completion_token_ids == []
        assert events[-1]["seq_ids"] == (sid_b,)
        assert events[-1]["cached"] == (size,) and events[-1]["queries"] == (budget,)
        assert events[-1]["positions"] == list(range(size, size + budget))
        assert manager.block_manager.blocks[shared[0]].ref_count == 2
        assert_blocks_unchanged(manager, shared, before)
        actual = collect_steps(engine)
        assert actual == {sid_a: hf_greedy_tokens(reference, a_prompt, 3),
                          sid_b: hf_greedy_tokens(reference, b_prompt, 2)}
        b_prefill = [(e["cached"][i], e["queries"][i]) for e in events[start:] if e["prefill"]
                     for i, sid in enumerate(e["seq_ids"]) if sid == sid_b]
        assert len(b_prefill) > 1 and sum(q for _, q in b_prefill) == len(b_prompt) - size
        assert_blocks_unchanged(manager, shared, before)
        # A completed chunk donor remains a cold source in the same engine.
        start = len(events)
        eos = hf_greedy_tokens(reference, b_prompt, 1)[0]
        output = engine.generate([b_prompt], SamplingParams(max_tokens=3),
                                 use_tqdm=False, eos_token_id=eos)
        assert output[0]["token_ids"] == [eos]
        assert events[start]["cached"][0] >= size
        assert_idle_cache(manager)
    finally:
        close_engine(engine, handle)


@torch.inference_mode()
def test_tail_prefill_batch_uses_each_request_last_query(model_case, monkeypatch):
    _, rng, reference, model_dir, device = model_case
    long_prompt = random_tokens(rng, 6, reference.config.vocab_size)
    short_prompt = random_tokens(rng, 2, reference.config.vocab_size)
    engine = LLM(
        str(model_dir), device=str(device), dtype="float32", scheduling_policy="separate",
        max_model_len=16, max_num_batched_tokens=4, max_num_seqs=2,
        kvcache_block_size=3, num_kvcache_blocks=8,
    )
    events, handle = observe_numerical_runner(engine, reference, monkeypatch)
    try:
        a = engine.add_request(long_prompt, SamplingParams(max_tokens=1, ignore_eos=True))
        assert engine.step() == []
        b = engine.add_request(short_prompt, SamplingParams(max_tokens=1, ignore_eos=True))
        actual = dict(engine.step())
        assert events[-1]["seq_ids"] == (a, b) and events[-1]["queries"] == (2, 2)
        assert events[-1]["ids"] == long_prompt[4:] + short_prompt
        assert events[-1]["positions"] == [4, 5, 0, 1]
        assert actual == {a: hf_greedy_tokens(reference, long_prompt, 1),
                          b: hf_greedy_tokens(reference, short_prompt, 1)}
        assert engine.is_finished()
        assert_idle_cache(engine.model_runner.kv_cache_manager)
    finally:
        close_engine(engine, handle)


@torch.inference_mode()
def test_numerical_preemption_rebuilds_prompt_and_generated_tokens_in_chunks(model_case, monkeypatch):
    seed, rng, reference, model_dir, device = model_case
    size = rng.choice([4, 5, 7])
    budget = size - 1
    prompts = [random_tokens(rng, size, reference.config.vocab_size) for _ in range(2)]
    engine = LLM(
        str(model_dir), device=str(device), dtype="float32", enable_prefix_cache=False,
        scheduling_policy="separate",
        max_model_len=3 * size, max_num_batched_tokens=budget, max_num_seqs=2,
        kvcache_block_size=size, num_kvcache_blocks=3,
    )
    events, handle = observe_numerical_runner(engine, reference, monkeypatch)
    victims = []
    original = engine.scheduler.preempt

    def preempt(seq):
        victims.append((seq.seq_id, list(seq.token_ids), list(seq.completion_token_ids), len(events)))
        original(seq)

    monkeypatch.setattr(engine.scheduler, "preempt", preempt)
    try:
        shuffled_blocks(engine.model_runner.kv_cache_manager, rng)
        for prompt in prompts:
            engine.add_request(prompt, SamplingParams(max_tokens=3, ignore_eos=True))
        assert collect_steps(engine) == {sid: hf_greedy_tokens(reference, p, 3)
                                        for sid, p in enumerate(prompts)}
        assert len(victims) == 1 and victims[0][0] == 1, f"seed={seed}: eviction not exercised"
        _, history, completion, start = victims[0]
        assert len(history) == size + 1 and len(completion) == 1
        recovery = [(e["cached"][i], e["queries"][i]) for e in events[start:] if e["prefill"]
                    for i, sid in enumerate(e["seq_ids"]) if sid == 1]
        assert recovery == [(0, budget), (budget, 2)]
        assert history == prompts[1] + completion
    finally:
        close_engine(engine, handle)


@torch.inference_mode()
def test_shared_preemption_resumes_cached_prefix_with_multiple_chunks(model_case, monkeypatch):
    seed, rng, reference, model_dir, device = model_case
    size = rng.choice([3, 4, 5])
    budget, generation = size - 1, size + 2
    common = random_tokens(rng, size, reference.config.vocab_size)
    private = rng.randrange(reference.config.vocab_size)
    prompts = [common + [private], common + [(private + 1) % reference.config.vocab_size]]
    engine = LLM(
        str(model_dir), device=str(device), dtype="float32", enable_prefix_cache=True,
        scheduling_policy="separate",
        max_model_len=3 * size, max_num_batched_tokens=budget, max_num_seqs=2,
        kvcache_block_size=size, num_kvcache_blocks=3,
    )
    manager = engine.model_runner.kv_cache_manager
    events, handle = observe_numerical_runner(engine, reference, monkeypatch)
    victims = []
    original = engine.scheduler.preempt

    def preempt(seq):
        free_before = len(manager.block_manager.free_block_id)
        original(seq)
        assert len(manager.block_manager.free_block_id) == free_before + 1
        victims.append((seq.seq_id, len(events)))

    monkeypatch.setattr(engine.scheduler, "preempt", preempt)
    try:
        engine.add_request(prompts[0], SamplingParams(max_tokens=generation, ignore_eos=True))
        a = engine.scheduler.request[0]
        while not a.num_completion_tokens:
            assert engine.step() == []
        shared = manager.block_manager.table(0)[:1]
        before = save_blocks(manager, shared)
        engine.add_request(prompts[1], SamplingParams(max_tokens=generation, ignore_eos=True))
        assert collect_steps(engine) == {sid: hf_greedy_tokens(reference, prompt, generation)
                                        for sid, prompt in enumerate(prompts)}
        assert len(victims) == 1 and victims[0][0] == 1, f"seed={seed}: shared eviction not exercised"
        recovery = [(e["cached"][i], e["queries"][i]) for e in events[victims[0][1]:]
                    if e["prefill"] for i, sid in enumerate(e["seq_ids"]) if sid == 1]
        assert recovery == [(size, budget), (size + budget, 2)]
        assert_blocks_unchanged(manager, shared, before)
    finally:
        close_engine(engine, handle)


@torch.inference_mode()
def test_real_first_layer_failure_during_waiting_chunk_does_not_publish(model_case, monkeypatch):
    _, rng, reference, model_dir, device = model_case
    engine = LLM(
        str(model_dir), device=str(device), dtype="float32", enable_prefix_cache=True,
        scheduling_policy="separate",
        max_model_len=16, max_num_batched_tokens=3, max_num_seqs=2,
        kvcache_block_size=4, num_kvcache_blocks=4,
    )
    manager = engine.model_runner.kv_cache_manager
    events, handle = observe_numerical_runner(engine, reference, monkeypatch)
    try:
        engine.add_request(random_tokens(rng, 10, reference.config.vocab_size),
                           SamplingParams(max_tokens=2, ignore_eos=True))
        assert engine.step() == []
        assert list(engine.scheduler.waiting)[0].num_cached_tokens == 3
        before = dict(manager.block_manager.hash_to_block_id)
        backend = engine.model_runner.model.model.layers[0].self_attn.backend
        original = backend.attend

        def fail(*args, **kwargs):
            original(*args, **kwargs)
            assert manager.layer_lengths[0] == [6] + [3] * (manager.num_layers - 1)
            raise RuntimeError("injected failure after chunk first-layer write")

        monkeypatch.setattr(backend, "attend", fail)
        with pytest.raises(RuntimeError, match="chunk first-layer"):
            engine.step()
        assert engine.failed and engine.scheduler.inflight is None
        assert not engine.scheduler.waiting and not engine.scheduler.running
        assert manager.block_manager.hash_to_block_id == before
        assert_idle_cache(manager)
        with pytest.raises(RuntimeError, match="failed"):
            engine.add_request([0])
        with pytest.raises(RuntimeError, match="failed"):
            engine.step()
    finally:
        close_engine(engine, handle)
