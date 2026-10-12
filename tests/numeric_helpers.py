import math
import os
import random
from collections import OrderedDict
from itertools import accumulate

import torch

from nanovllm.engine.context import Context
from nanovllm.engine.sequence import Sequence, SequenceStatus


CASE_COUNT = int(os.environ.get("MY_NANO_VLLM_TEST_CASES", "24"))
BASE_SEED = int(os.environ.get("MY_NANO_VLLM_TEST_SEED", "20261003"))
if CASE_COUNT < 1:
    raise ValueError("MY_NANO_VLLM_TEST_CASES must be positive")
CASE_SEEDS = tuple(BASE_SEED + i for i in range(CASE_COUNT))
TENSOR_SEEDS = tuple(BASE_SEED + 100_000 + i for i in range(4 * CASE_COUNT))
ATOL = 1e-5
RTOL = 1e-4


def random_model_config(seed, config_class):
    rng = random.Random(seed)
    heads = rng.choice([2, 4, 6, 8])
    kv_heads = rng.choice([h for h in range(1, heads + 1) if heads % h == 0])
    config = config_class(
        vocab_size=rng.choice([31, 64, 97, 131]),
        hidden_size=rng.choice([16, 24, 32, 48]),
        intermediate_size=rng.choice([32, 48, 64, 96]),
        num_hidden_layers=rng.randint(1, 4),
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=rng.choice([4, 6, 8, 12, 16]),
        max_position_embeddings=rng.choice([32, 64, 96]),
        rope_theta=rng.choice([100.0, 10_000.0, 1_000_000.0]),
        rms_norm_eps=rng.choice([1e-6, 1e-5, 1e-3]),
        initializer_range=rng.choice([0.02, 0.05, 0.1]),
        attention_dropout=0.0,
        tie_word_embeddings=rng.choice([False, True]),
        bos_token_id=None, eos_token_id=1, pad_token_id=0,
    )
    config._attn_implementation = "eager"
    return config, rng


def random_tensor(rng, shape, device, scale=1.0):
    generator = torch.Generator().manual_seed(rng.randrange(2**31))
    return (torch.randn(shape, generator=generator) * scale).to(device)


def random_tokens(rng, length, vocab_size):
    # Include repeated tokens and both vocabulary boundaries, as well as
    # uniformly random IDs, instead of relying on random greedy trajectories.
    mode = rng.randrange(4)
    if mode == 0:
        return [rng.randrange(vocab_size)] * length
    if mode == 1:
        return [0 if i % 2 == 0 else vocab_size - 1 for i in range(length)]
    return [rng.randrange(vocab_size) for _ in range(length)]


def shuffled_blocks(manager, rng):
    blocks = list(manager.block_manager.free_block_id)
    rng.shuffle(blocks)
    manager.block_manager.free_block_id = OrderedDict.fromkeys(blocks)


def allocate_registered_prefill(manager, seqs):
    """Perform the scheduling preparation for direct Runner numerical tests.

    These requests are already registered. Warmup fixtures may have scheduled
    their full prompt before storage existed; real admission starts with Q=0.
    The caller, like Scheduler, owns Sequence progress after cache allocation.
    """
    for seq in seqs:
        assert seq.num_cached_tokens == 0
        seq.num_scheduled_tokens = 0
        possible, hits = manager.can_allocate_prefill(seq)
        assert possible, "numerical fixture must fit the configured KV pool"
        cached = manager.allocate_prefill(seq, hits)
        assert seq.num_cached_tokens == 0, "cache allocation must not commit Sequence progress"
        seq.num_cached_tokens = cached
        seq.num_scheduled_tokens = seq.num_tokens - cached
        seq.status = SequenceStatus.RUNNING


def allocate_decode(manager, seqs):
    """Explicit resource preparation; never hide allocation in the HF oracle."""
    for seq in seqs:
        assert manager.can_allocate_decode(seq), "decode fixture must fit its pool"
        manager.allocate_decode(seq)


def make_context(seq_ids, old_lengths, query_lengths):
    lengths = tuple(old + new for old, new in zip(old_lengths, query_lengths))
    return Context(
        is_prefill=not any(old_lengths), seq_ids=tuple(seq_ids),
        cu_seq_lens_q=(0, *accumulate(query_lengths)),
        cu_seq_lens_k=(0, *accumulate(lengths)),
        cache_lens=tuple(old_lengths), context_lens=lengths,
        max_seq_lens_q=max(query_lengths), max_seq_lens_k=max(lengths),
    )


def dense_attention_reference(q, k, v, positions):
    # Independent float64 reference. Query heads select their KV head by
    # indexing, rather than the runtime's repeat_interleave implementation.
    groups = q.shape[1] // k.shape[1]
    head_indices = torch.arange(q.shape[1], device=q.device) // groups
    keys = k.double()[:, head_indices]
    values = v.double()[:, head_indices]
    scores = torch.einsum("thd,shd->hts", q.double(), keys) / math.sqrt(q.shape[-1])
    mask = torch.arange(k.shape[0], device=q.device)[None] > positions[:, None]
    probabilities = scores.masked_fill(mask[None], -torch.inf).softmax(-1)
    return torch.einsum("hts,shd->thd", probabilities, values).float()


def make_sequences(rng, vocab_size, block_size, count=None, max_length=None):
    count = rng.randint(2, 5) if count is None else count
    ids = rng.sample(range(10, 10_000), count)
    lengths = [rng.choice([1, block_size, block_size + 1, 2 * block_size - 1])
               for _ in ids]
    if max_length is not None:
        lengths[0] = max_length
        lengths[1:] = [rng.choice([length, rng.randint(1, max_length)])
                       for length in lengths[1:]]
    seqs = [Sequence(sid, random_tokens(rng, length, vocab_size), max_tokens=16)
            for sid, length in zip(ids, lengths)]
    for seq in seqs:
        seq.status = SequenceStatus.RUNNING
        seq.num_scheduled_tokens = seq.num_tokens
    return seqs


def advance_with_random_tokens(seqs, rng, vocab_size):
    for seq in seqs:
        seq.num_cached_tokens = seq.num_tokens
        seq.append_token(rng.randrange(vocab_size))
        seq.num_scheduled_tokens = 1


@torch.inference_mode()
def assert_packed_matches_hf(logits, ctx, seqs, reference, manager, *, atol=ATOL, rtol=RTOL):
    expected_last = []
    for i, seq in enumerate(seqs):
        ids = torch.tensor([seq.token_ids], device=logits.device)
        # Recompute the entire history independently each time. The oracle
        # never shares or trusts the runtime's cached values or cache lengths.
        expected = reference(ids, use_cache=True)
        lo, hi = ctx.cu_seq_lens_q[i:i + 2]
        old, end = ctx.cache_lens[i], ctx.context_lens[i]
        assert hi - lo == end - old
        assert torch.isfinite(logits[lo:hi]).all()
        torch.testing.assert_close(
            logits[lo:hi], expected.logits[0, old:end], atol=atol, rtol=rtol,
            msg=lambda msg: f"request={seq.seq_id}, query=[{old},{end}): {msg}",
        )
        expected_last.append(expected.logits[0, end - 1])
        if manager.k_pool is not None:
            table = manager.block_manager.table(seq.seq_id)
            for layer in range(manager.num_layers):
                actual_k, actual_v = manager.gather(layer, seq.seq_id, end, table)
                expected_k, expected_v = expected.past_key_values[layer]
                for actual, wanted in zip((actual_k, actual_v), (expected_k, expected_v)):
                    # HF [batch, Hkv, T, D] -> paged logical [T, Hkv, D].
                    torch.testing.assert_close(
                        actual, wanted[0, :, :end].transpose(0, 1), atol=atol, rtol=rtol,
                        msg=lambda msg: f"request={seq.seq_id}, layer={layer}: {msg}",
                    )
    return torch.stack(expected_last)


def checked_runner_logits(runner, seqs, is_prefill, reference, original=None, *, atol=ATOL, rtol=RTOL):
    captured = {}

    def capture(module, inputs, output):
        captured["logits"] = output
        captured["ctx"] = inputs[2]

    hook = runner.model.register_forward_hook(capture)
    try:
        actual = (original or runner.run_logits)(seqs, is_prefill)
    finally:
        hook.remove()
    expected = assert_packed_matches_hf(
        captured["logits"], captured["ctx"], seqs, reference, runner.kv_cache_manager,
        atol=atol, rtol=rtol,
    )
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual.argmax(-1), expected.argmax(-1), atol=0, rtol=0)
    return actual


@torch.inference_mode()
def hf_greedy_tokens(reference, prompt, max_tokens, eos_token_id=None):
    history = list(prompt)
    device = next(reference.parameters()).device
    for _ in range(max_tokens):
        ids = torch.tensor([history], device=device)
        token = int(reference(ids, use_cache=False).logits[0, -1].argmax())
        history.append(token)
        if token == eos_token_id:
            break
    return history[len(prompt):]
