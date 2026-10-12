"""Seeded real Qwen3 forwards, independent HF logits/KV and public generation."""

import atexit

import pytest
import torch

from nanovllm import Config, LLM, SamplingParams
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import postprocess_one
from nanovllm.engine.sequence import Sequence
from numeric_helpers import (
    checked_runner_logits, hf_greedy_tokens, random_tokens, shuffled_blocks, allocate_decode,
)
from prefix_helpers import (
    assert_block_conservation, assert_blocks_unchanged, assert_idle_cache,
    attach_request, observe_numerical_runner, save_blocks,
)


@pytest.mark.parametrize("enabled", [False, True], ids=["cache-off", "cache-on"])
@torch.inference_mode()
def test_random_histories_suffix_logits_kv_and_mixed_cold_batch(model_case, enabled):
    seed, rng, reference, model_dir, device = model_case
    hf = reference.config
    size = rng.choice([1, 2, 3, 4, 7])
    runner = ModelRunner(Config(
        str(model_dir), device=str(device), dtype="float32", enable_prefix_cache=enabled,
        max_model_len=hf.max_position_embeddings,
        max_num_batched_tokens=3 * hf.max_position_embeddings,
        kvcache_block_size=size,
        num_kvcache_blocks=3 * ((hf.max_position_embeddings + size - 1) // size),
    ))
    manager = runner.kv_cache_manager
    assert manager.enable_prefix_cache is enabled
    assert manager.block_manager.enable_prefix_cache is enabled
    manager.k_pool.fill_(torch.nan)
    manager.v_pool.fill_(torch.nan)
    shuffled_blocks(manager, rng)
    common = random_tokens(rng, 2 * size, hf.vocab_size)
    private_a = rng.randrange(hf.vocab_size)
    private_b = (private_a + rng.randint(1, hf.vocab_size - 1)) % hf.vocab_size
    a = Sequence(17, common + [private_a], 8)
    b = Sequence(203, common + [private_b]
                 + random_tokens(rng, rng.randrange(size + 1), hf.vocab_size), 8)
    calls = []

    def capture(module, inputs):
        ids, positions, ctx = inputs
        calls.append((ids.tolist(), positions.tolist(), ctx))

    handle = runner.model.register_forward_pre_hook(capture)

    def forward_and_commit(seqs, prefill):
        if not prefill:
            allocate_decode(manager, seqs)
        # Full independent HF history compares every query, every layer's KV,
        # the last logits and argmax. Forced next tokens diversify decode paths.
        checked_runner_logits(runner, seqs, prefill, reference)
        for seq in seqs:
            manager.publish_seq(seq)
        for seq in seqs:
            postprocess_one(seq, rng.randrange(hf.vocab_size))
        assert_block_conservation(manager.block_manager)

    try:
        assert attach_request(manager, a) == 0
        forward_and_commit([a], True)
        shared = manager.block_manager.table(a.seq_id)[:2]
        before = save_blocks(manager, shared)
        assert attach_request(manager, b) == (2 if enabled else 0), f"seed={seed}"
        cached = 2 * size if enabled else 0
        suffix = list(b.token_ids[cached:])
        forward_and_commit([b], True)
        ids, positions, ctx = calls[-1]
        assert ids == suffix
        assert positions == list(range(cached, cached + len(suffix)))
        assert ctx.cache_lens == (cached,)
        for _ in range(3):
            b.num_scheduled_tokens = 1
            forward_and_commit([b], False)
        assert_blocks_unchanged(manager, shared, before)
        # Release either sharer first, then reactivate cold entries in a mixed
        # batch whose other request has a deliberately different first token.
        exits = [a.seq_id, b.seq_id]
        rng.shuffle(exits)
        for sid in exits:
            manager.release(sid)
            assert_block_conservation(manager.block_manager)
        assert_idle_cache(manager)
        shuffled_blocks(manager, rng)
        warm = Sequence(307, common + [private_b], 1)
        different = list(common)
        different[0] = (different[0] + 1) % hf.vocab_size
        cold = Sequence(419, different + random_tokens(rng, rng.randint(1, size), hf.vocab_size), 1)
        assert attach_request(manager, warm) == (2 if enabled else 0)
        assert attach_request(manager, cold) == 0
        batch = [warm, cold]
        rng.shuffle(batch)
        forward_and_commit(batch, True)
        ids, positions, ctx = calls[-1]
        assert ctx.cache_lens == tuple(cached if seq is warm else 0 for seq in batch)
        assert len(ids) == sum(seq.num_tokens - 1 - old for seq, old in zip(batch, ctx.cache_lens))
        assert bool(manager.block_manager.hash_to_block_id) is enabled
    finally:
        handle.remove()
        for sid in list(manager.layer_lengths):
            manager.release(sid)
    assert_idle_cache(manager)


@torch.inference_mode()
def test_random_deep_histories_reuse_exact_hot_and_cold_prefixes(model_case):
    seed, rng, reference, model_dir, device = model_case
    hf = reference.config
    size, depth = rng.choice([1, 2, 3, 4]), rng.randint(3, 6)
    runner = ModelRunner(Config(
        str(model_dir), device=str(device), dtype="float32", enable_prefix_cache=True,
        max_model_len=(depth + 1) * size, max_num_batched_tokens=2 * (depth + 1) * size,
        max_num_seqs=3, kvcache_block_size=size, num_kvcache_blocks=2 * depth + 3,
    ))
    manager, blocks = runner.kv_cache_manager, runner.kv_cache_manager.block_manager
    manager.k_pool.fill_(torch.nan)
    manager.v_pool.fill_(torch.nan)
    shuffled_blocks(manager, rng)
    history = random_tokens(rng, depth * size, hf.vocab_size)
    changed = list(history)
    changed[0] = (changed[0] + rng.randint(1, hf.vocab_size - 1)) % hf.vocab_size
    a = Sequence(17, history + [rng.randrange(hf.vocab_size)], 1)
    b_tail = (changed[(depth - 1) * size] + rng.randint(1, hf.vocab_size - 1)) % hf.vocab_size
    b = Sequence(203, changed[:(depth - 1) * size] + [b_tail], 1)
    probe = Sequence(307, changed + [rng.randrange(hf.vocab_size)], 1)
    try:
        for seq in [a, b]:
            assert attach_request(manager, seq) == 0
            checked_runner_logits(runner, [seq], True, reference)
            manager.publish_seq(seq)
        assert attach_request(manager, probe) == depth - 1, f"seed={seed}"
        shared = blocks.table(b.seq_id)[:depth - 1]
        assert blocks.table(probe.seq_id)[:depth - 1] == shared
        assert blocks.table(probe.seq_id)[depth - 1] != blocks.table(a.seq_id)[depth - 1]
        before = save_blocks(manager, shared)
        checked_runner_logits(runner, [probe], True, reference)
        manager.publish_seq(probe)
        assert_blocks_unchanged(manager, shared, before)
        assert_block_conservation(blocks)
        expected_prefix = blocks.table(probe.seq_id)[:depth]
        exits = [a.seq_id, b.seq_id, probe.seq_id]
        rng.shuffle(exits)
        for sid in exits:
            manager.release(sid)
        assert_idle_cache(manager)
        cold = Sequence(419, changed + [rng.randrange(hf.vocab_size)], 1)
        assert attach_request(manager, cold) == depth
        assert blocks.table(cold.seq_id)[:depth] == expected_prefix
        checked_runner_logits(runner, [cold], True, reference)
        assert_block_conservation(blocks)
    finally:
        for sid in list(manager.layer_lengths):
            manager.release(sid)
    assert_idle_cache(manager)


@torch.inference_mode()
def test_public_greedy_shared_preemption_and_cold_reuse_match_hf(model_case, monkeypatch):
    seed, rng, reference, model_dir, device = model_case
    size = rng.choice([2, 3, 4, 7])
    hf = reference.config
    engine = LLM(
        str(model_dir), device=str(device), dtype="float32", enable_prefix_cache=True,
        scheduling_policy="separate",
        max_model_len=3 * size, max_num_batched_tokens=3 * size,
        max_num_seqs=2, kvcache_block_size=size, num_kvcache_blocks=3,
    )
    manager = engine.model_runner.kv_cache_manager
    shuffled_blocks(manager, rng)
    common = random_tokens(rng, size, hf.vocab_size)
    prompts = [common + [0], common + [hf.vocab_size - 1]]
    budget = size + 1
    params = SamplingParams(max_tokens=budget, ignore_eos=True)
    expected = {sid: hf_greedy_tokens(reference, prompt, budget) for sid, prompt in enumerate(prompts)}
    actual, victims = {}, []
    original_preempt = engine.scheduler.preempt
    events, handle = observe_numerical_runner(engine, reference, monkeypatch)

    def observe_preemption(seq):
        history = list(seq.token_ids)
        original_preempt(seq)
        assert seq.token_ids == history
        assert seq.num_cached_tokens == seq.num_scheduled_tokens == 0
        assert_block_conservation(manager.block_manager)
        victims.append(seq.seq_id)

    monkeypatch.setattr(engine.scheduler, "preempt", observe_preemption)
    try:
        assert engine.add_request(prompts[0], params) == 0
        actual.update(engine.step())
        shared = manager.block_manager.table(0)[0]
        before = save_blocks(manager, [shared])
        assert engine.add_request(prompts[1], params) == 1
        actual.update(engine.step())
        assert events[1]["cached"] == (size,) and events[1]["queries"] == (1,)
        assert events[1]["ids"] == [hf.vocab_size - 1]
        assert events[1]["positions"] == [size]
        assert manager.block_manager.blocks[shared].ref_count == 2
        assert_blocks_unchanged(manager, [shared], before)
        for _ in range(2 * budget + 10):
            if engine.is_finished():
                break
            for sid, tokens in engine.step():
                assert sid not in actual
                actual[sid] = tokens
        assert engine.is_finished(), f"seed={seed}: failed to make bounded progress"
        assert victims == [1], f"seed={seed}: shared eviction was not exercised"
        assert any(event["prefill"] and event["seq_ids"] == (1,)
                   and event["cached"] == (size,) and event["queries"][0] > 1
                   for event in events), f"seed={seed}: cached recovery was not exercised"
        assert actual == expected
        assert_idle_cache(manager)

        # Public generate must preserve zero-budget output ordering and use the
        # already committed cold prefix for an EOS-ending request in a new wave.
        eos = hf_greedy_tokens(reference, prompts[1], 1)[0]
        start = len(events)
        outputs = engine.generate(
            [prompts[0], prompts[1]],
            [SamplingParams(max_tokens=0), SamplingParams(max_tokens=2)],
            use_tqdm=False, eos_token_id=eos,
        )
        assert [output["token_ids"] for output in outputs] == [[], [eos]]
        assert len(events) == start + 1
        assert events[-1]["cached"] == (size,) and events[-1]["queries"] == (1,)
        assert_idle_cache(manager)
    finally:
        handle.remove()
        engine.exit()
        atexit.unregister(engine.exit)


@torch.inference_mode()
def test_public_engine_partial_forward_failure_releases_all_references(model_case, monkeypatch):
    _, rng, reference, model_dir, device = model_case
    size = rng.choice([1, 2, 3, 4, 7])
    engine = LLM(
        str(model_dir), device=str(device), dtype="float32", enable_prefix_cache=True,
        scheduling_policy="separate",
        max_model_len=4 * size, max_num_batched_tokens=4 * size,
        kvcache_block_size=size, num_kvcache_blocks=4,
    )
    manager = engine.model_runner.kv_cache_manager
    common = random_tokens(rng, 2 * size, reference.config.vocab_size)
    events, handle = observe_numerical_runner(engine, reference, monkeypatch)
    try:
        engine.add_request(common + [0], SamplingParams(max_tokens=2, ignore_eos=True))
        engine.step()
        index = dict(manager.block_manager.hash_to_block_id)
        engine.add_request(common + [reference.config.vocab_size - 1],
                           SamplingParams(max_tokens=1, ignore_eos=True))
        backend = engine.model_runner.model.model.layers[0].self_attn.backend
        original_attend = backend.attend

        def fail_after_first_layer(*args, **kwargs):
            original_attend(*args, **kwargs)
            assert manager.layer_lengths[1] == [2 * size + 1] + [2 * size] * (manager.num_layers - 1)
            raise RuntimeError("injected failure after real first-layer KV write")

        monkeypatch.setattr(backend, "attend", fail_after_first_layer)
        with pytest.raises(RuntimeError, match="real first-layer"):
            engine.step()
        assert engine.failed
        assert manager.block_manager.hash_to_block_id == index
        assert_idle_cache(manager)
        with pytest.raises(RuntimeError):
            engine.step()
    finally:
        handle.remove()
        engine.exit()
        atexit.unregister(engine.exit)
