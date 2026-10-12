"""Opt-in active-framework CUDA/local checkpoint equivalence; no downloads."""

import atexit
import gc
import os
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer, Qwen3ForCausalLM as HFQwen3

from nanovllm import Config, LLM, SamplingParams
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import postprocess_one
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.loader import resolve_checkpoint_files
from numeric_helpers import allocate_decode, checked_runner_logits, hf_greedy_tokens, shuffled_blocks
from prefix_helpers import attach_request, assert_idle_cache, assert_block_conservation


pytestmark = pytest.mark.skipif(os.getenv("NANO_VLLM_TEST_REAL") != "1",
                                reason="Set NANO_VLLM_TEST_REAL=1 for the local trained checkpoint")
REAL_COUNT = int(os.getenv("MY_NANO_VLLM_REAL_CASES", "8"))
if REAL_COUNT < 1:
    raise ValueError("MY_NANO_VLLM_REAL_CASES must be positive")
REAL_SEED = int(os.getenv("MY_NANO_VLLM_REAL_SEED", "20263004"))
REAL_SEEDS = tuple(REAL_SEED + i for i in range(REAL_COUNT))
# The existing real-checkpoint suites also use 1e-4 for float32. Tiny random
# model tests retain their original 1e-5 absolute tolerance. Tokens are exact.
REAL_ATOL = REAL_RTOL = 1e-4


@pytest.fixture(scope="module")
def real_case():
    directory = Path(os.getenv("NANO_VLLM_TEST_MODEL", Config.model_path))
    resolve_checkpoint_files(directory)
    device = torch.device(os.getenv("NANO_VLLM_TEST_DEVICE", "cuda"))
    if device.type == "cuda":
        assert torch.cuda.is_available(), "Requested CUDA is unavailable"
        device = torch.device("cuda", 0 if device.index is None else device.index)
    reference = HFQwen3.from_pretrained(
        directory, torch_dtype=torch.float32, attn_implementation="eager",
        local_files_only=True,
    ).eval().to(device).requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    case = SimpleNamespace(directory=str(directory), device=device, reference=reference,
                           tokenizer=tokenizer)
    yield case
    reference.cpu()
    del case, reference
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def cleanup_engine(engine):
    runner = engine.model_runner
    engine.exit()
    atexit.unregister(engine.exit)
    cleanup_runner(runner)


def cleanup_runner(runner):
    # Pytest keeps failed assertion tracebacks alive. Move their retained model
    # off GPU explicitly, so a numerical failure cannot exhaust later cases.
    manager = runner.kv_cache_manager
    for sid in list(manager.layer_lengths):
        manager.release(sid)
    runner.model.cpu()
    manager.k_pool = manager.v_pool = None
    gc.collect()
    if runner.device.type == "cuda":
        torch.cuda.empty_cache()


@pytest.mark.parametrize("seed", REAL_SEEDS)
@torch.inference_mode()
def test_real_shared_suffix_decode_boundary_and_cold_batch(real_case, seed):
    case, rng = real_case, random.Random(seed)
    size = rng.choice([4, 8, 16])
    runner = ModelRunner(Config(
        case.directory, device=str(case.device), dtype="float32", enable_prefix_cache=True,
        max_model_len=8 * size, max_num_batched_tokens=16 * size,
        max_num_seqs=3, kvcache_block_size=size, num_kvcache_blocks=32,
    ))
    manager = runner.kv_cache_manager
    assert next(runner.model.parameters()).device == case.device
    shuffled_blocks(manager, rng)
    source = case.tokenizer.encode(
        "Explain how a KV cache reuses a common prefix during language model inference. " * 8)
    offset = rng.randrange(len(source) - 3 * size)
    common = source[offset:offset + 2 * size]
    tail = source[offset + 2 * size:offset + 3 * size - 2]
    a = Sequence(17, common + tail, 8)
    other_tail = list(tail)
    other_tail[0] = (other_tail[0] + 1) % runner.config.hf_config.vocab_size
    b = Sequence(203, common + other_tail, 8)

    def commit(seqs, prefill):
        if not prefill:
            allocate_decode(manager, seqs)
        logits = checked_runner_logits(runner, seqs, prefill, case.reference,
                                       atol=REAL_ATOL, rtol=REAL_RTOL)
        for seq in seqs:
            manager.publish_seq(seq)
        for seq, token in zip(seqs, logits.argmax(-1).tolist()):
            postprocess_one(seq, token)
        assert_block_conservation(manager.block_manager)

    try:
        assert attach_request(manager, a) == 0
        commit([a], True)
        assert attach_request(manager, b) == 2
        commit([b], True)
        grew = False
        for _ in range(3):
            before = len(manager.block_manager.table(b.seq_id))
            b.num_scheduled_tokens = 1
            commit([b], False)
            grew |= len(manager.block_manager.table(b.seq_id)) > before
        assert grew, "real decode must cross a physical block boundary"
        for sid in [a.seq_id, b.seq_id]:
            manager.release(sid)
        assert_idle_cache(manager)
        shuffled_blocks(manager, rng)
        warm = Sequence(307, common + [other_tail[0]], 1)
        different = list(common)
        different[0] = (different[0] + 1) % runner.config.hf_config.vocab_size
        cold = Sequence(419, different + [tail[0]], 1)
        assert attach_request(manager, warm) == 2
        assert attach_request(manager, cold) == 0
        seqs = [warm, cold]
        rng.shuffle(seqs)
        commit(seqs, True)
    finally:
        for sid in list(manager.layer_lengths):
            manager.release(sid)
        cleanup_runner(runner)
    assert_idle_cache(manager)


@pytest.mark.parametrize("seed", REAL_SEEDS)
@torch.inference_mode()
def test_real_public_greedy_preemption_and_natural_text(real_case, seed, monkeypatch):
    case, rng = real_case, random.Random(seed)
    size = rng.choice([4, 7, 8])
    engine = LLM(case.directory, device=str(case.device), dtype="float32", enable_prefix_cache=True,
                 scheduling_policy="separate",
                 max_model_len=3 * size, max_num_batched_tokens=3 * size,
                 max_num_seqs=2, kvcache_block_size=size, num_kvcache_blocks=3)
    runner, manager = engine.model_runner, engine.model_runner.kv_cache_manager
    original_logits, original_preempt = runner.run_logits, engine.scheduler.preempt
    victims, recovered = [], []

    def checked(seqs, prefill):
        recovered.extend(seq.seq_id for seq in seqs if prefill and seq.num_completion_tokens > 0)
        return checked_runner_logits(runner, seqs, prefill, case.reference, original_logits,
                                     atol=REAL_ATOL, rtol=REAL_RTOL)

    def preempt(seq):
        victims.append(seq.seq_id)
        original_preempt(seq)

    monkeypatch.setattr(runner, "run_logits", checked)
    monkeypatch.setattr(engine.scheduler, "preempt", preempt)
    source = case.tokenizer.encode("Explain KV caching in a language model.")
    common = (source * 4)[:size]
    prompts = [common + [source[-1]], common + [source[-2]]]
    params = SamplingParams(max_tokens=size + 1, ignore_eos=True)
    expected = {sid: hf_greedy_tokens(case.reference, prompt, params.max_tokens)
                for sid, prompt in enumerate(prompts)}
    actual = {}
    try:
        shuffled_blocks(manager, rng)
        assert engine.add_request(prompts[0], params) == 0
        actual.update(engine.step())
        assert engine.add_request(prompts[1], params) == 1
        actual.update(engine.step())
        for _ in range(2 * params.max_tokens + 10):
            if engine.is_finished():
                break
            actual.update(engine.step())
        assert engine.is_finished()
        assert actual == expected
        assert victims and recovered, "real model must exercise preemption and cached recovery"
        assert_idle_cache(manager)
    finally:
        cleanup_engine(engine)


@pytest.mark.parametrize("seed", REAL_SEEDS)
@torch.inference_mode()
def test_real_random_deep_histories_preserve_logits_and_kv(real_case, seed):
    """Same middle chunks after different roots, with the production hash."""
    case, rng = real_case, random.Random(seed)
    size, depth = rng.choice([1, 2, 4, 8]), rng.randint(3, 6)
    runner = ModelRunner(Config(
        case.directory, device=str(case.device), dtype="float32", enable_prefix_cache=True,
        max_model_len=8 * size, max_num_batched_tokens=16 * size, max_num_seqs=3,
        kvcache_block_size=size, num_kvcache_blocks=2 * depth + 3,
    ))
    manager, blocks = runner.kv_cache_manager, runner.kv_cache_manager.block_manager
    shuffled_blocks(manager, rng)
    source = case.tokenizer.encode("Explain why a language model caches its attention keys and values. " * 12)
    offset = rng.randrange(len(source) - depth * size - 3)
    root = source[offset:offset + size]
    middle = source[offset + size:offset + (depth - 1) * size]
    target = source[offset + (depth - 1) * size:offset + depth * size]
    changed = list(root)
    changed[0] = (changed[0] + rng.randint(1, runner.config.hf_config.vocab_size - 1)) % runner.config.hf_config.vocab_size
    a = Sequence(1, root + middle + target + [source[offset + depth * size]], 1)
    b_tail = (target[0] + rng.randint(1, runner.config.hf_config.vocab_size - 1)) % runner.config.hf_config.vocab_size
    b = Sequence(2, changed + middle + [b_tail], 1)
    probe = Sequence(3, changed + middle + target + [source[offset + depth * size + 2]], 1)
    try:
        for seq in [a, b]:
            assert attach_request(manager, seq) == 0
            checked_runner_logits(runner, [seq], True, case.reference,
                                  atol=REAL_ATOL, rtol=REAL_RTOL)
            manager.publish_seq(seq)
        assert attach_request(manager, probe) == depth - 1
        assert blocks.table(probe.seq_id)[:depth - 1] == blocks.table(b.seq_id)[:depth - 1]
        assert blocks.table(probe.seq_id)[depth - 1] != blocks.table(a.seq_id)[depth - 1]
        checked_runner_logits(runner, [probe], True, case.reference,
                              atol=REAL_ATOL, rtol=REAL_RTOL)
        manager.publish_seq(probe)
        assert probe.num_cached_tokens == (depth - 1) * size
        assert_block_conservation(blocks)
        exits = [a.seq_id, b.seq_id, probe.seq_id]
        rng.shuffle(exits)
        for sid in exits:
            manager.release(sid)
        assert_idle_cache(manager)
        cold = Sequence(4, list(probe.token_ids), 1)
        assert attach_request(manager, cold) == depth
        checked_runner_logits(runner, [cold], True, case.reference,
                              atol=REAL_ATOL, rtol=REAL_RTOL)
        assert_block_conservation(blocks)
    finally:
        for sid in list(manager.layer_lengths):
            manager.release(sid)
        cleanup_runner(runner)

@pytest.mark.parametrize("seed", REAL_SEEDS)
@torch.inference_mode()
def test_real_natural_text_public_tokenizer_and_greedy(real_case, seed):
    case, rng = real_case, random.Random(seed)
    size = rng.choice([4, 7, 8])
    # Longer, complete natural text inputs exercise the public tokenizer path
    # with zero-budget ordering and ordinary greedy generation.
    engine = LLM(case.directory, device=str(case.device), dtype="float32", enable_prefix_cache=True,
                 scheduling_policy="separate",
                 max_model_len=128, max_num_batched_tokens=256, max_num_seqs=3,
                 kvcache_block_size=size, num_kvcache_blocks=96)
    texts = ["Hello, introduce yourself in one sentence.",
             "Explain why language models use a KV cache.",
             "What is the difference between prefill and decode?"]
    budgets = [0, rng.randint(2, 4), rng.randint(2, 4)]
    expected = [hf_greedy_tokens(case.reference, case.tokenizer.encode(text), budget)
                for text, budget in zip(texts, budgets)]
    try:
        outputs = engine.generate(texts, [SamplingParams(max_tokens=n, ignore_eos=True) for n in budgets],
                                  use_tqdm=False)
        assert [output["token_ids"] for output in outputs] == expected
        assert_idle_cache(engine.model_runner.kv_cache_manager)
    finally:
        cleanup_engine(engine)


@pytest.mark.parametrize("policy", ["separate", "interleaved", "mixed"])
@pytest.mark.parametrize("seed", REAL_SEEDS)
@torch.inference_mode()
def test_real_three_policies_mixed_prefix_and_preemption_match_hf(real_case, seed, policy, monkeypatch):
    case, rng = real_case, random.Random(seed)
    size = rng.choice([4, 7, 8])
    source = case.tokenizer.encode("Explain why attention caches keys and values during generation. " * 8)
    common = source[:size]
    prompts = [common + [source[size]], common + [(source[size] + 1) % case.reference.config.vocab_size]]
    generation = size + 2
    engine = LLM(
        case.directory, device=str(case.device), dtype="float32", scheduling_policy=policy,
        enable_prefix_cache=True, max_model_len=3 * size, max_num_batched_tokens=size - 1,
        max_num_seqs=2, kvcache_block_size=size, num_kvcache_blocks=3,
    )
    runner, manager = engine.model_runner, engine.model_runner.kv_cache_manager
    original_logits, original_schedule = runner.run_logits, engine.scheduler.schedule
    original_preempt = engine.scheduler.preempt
    kinds, phases, victims, recovery = [], [], [], []

    def schedule():
        plan = original_schedule()
        if plan is not None:
            kinds.append(plan.kind)
            phases.append(tuple(e.phase for e in plan.entries))
            recovery.extend(e.seq.seq_id for e in plan.entries
                            if e.phase == "prefill" and e.seq.num_completion_tokens > 0)
        return plan

    def checked(seqs, flag):
        return checked_runner_logits(runner, seqs, flag, case.reference, original_logits,
                                     atol=REAL_ATOL, rtol=REAL_RTOL)

    def preempt(seq):
        victims.append((seq.seq_id, list(seq.token_ids)))
        original_preempt(seq)
        assert_block_conservation(manager.block_manager)

    monkeypatch.setattr(engine.scheduler, "schedule", schedule)
    monkeypatch.setattr(engine.scheduler, "preempt", preempt)
    monkeypatch.setattr(runner, "run_logits", checked)
    try:
        shuffled_blocks(manager, rng)
        engine.add_request(prompts[0], SamplingParams(max_tokens=generation, ignore_eos=True))
        while not engine.scheduler.request[0].num_completion_tokens:
            assert engine.step() == []
        engine.add_request(prompts[1], SamplingParams(max_tokens=generation, ignore_eos=True))
        actual = {}
        for _ in range(4 * generation + 20):
            if engine.is_finished():
                break
            for sid, tokens in engine.step():
                assert sid not in actual
                actual[sid] = tokens
        assert engine.is_finished(), f"seed={seed}, policy={policy}: no bounded progress"
        assert actual == {i: hf_greedy_tokens(case.reference, p, generation) for i, p in enumerate(prompts)}
        assert victims and recovery, "trained checkpoint must exercise actual eviction and recovery"
        assert ("mixed" in kinds) == (policy == "mixed")
        if policy == "mixed":
            assert ("decode", "prefill") in phases
        assert_idle_cache(manager)
    finally:
        cleanup_engine(engine)
