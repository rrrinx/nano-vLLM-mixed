import random

import pytest
import torch

from nanovllm.engine.sequence import Sequence
from nanovllm.runtime.paged_cache import PagedKVCacheManager
from numeric_helpers import (
    ATOL, RTOL, TENSOR_SEEDS, dense_attention_reference, make_context,
    random_tensor, shuffled_blocks,
)


@pytest.mark.parametrize("seed", TENSOR_SEEDS, ids=lambda seed: f"seed-{seed}")
@torch.inference_mode()
def test_random_paged_attention_and_exact_kv_storage(seed, numeric_device):
    rng = random.Random(seed)
    heads = rng.choice([1, 2, 4, 6, 8])
    kv_heads = rng.choice([h for h in range(1, heads + 1) if heads % h == 0])
    dim = rng.choice([2, 4, 8, 12, 16])
    layers = rng.randint(1, 3)
    block_size = rng.choice([1, 2, 3, 4, 7, 8])
    count = rng.randint(2, 5)
    seqs = [Sequence(sid, [0], max_tokens=1) for sid in rng.sample(range(1000), count)]
    lengths = [rng.choice([1, block_size, block_size + 1, 2 * block_size - 1])
               for _ in seqs]
    manager = PagedKVCacheManager(layers, None, block_size, kv_heads, dim)
    backends = [manager.make_backend(layer) for layer in range(layers)]
    initial = []
    initial_ctx = make_context([s.seq_id for s in seqs], [0] * count, lengths)
    positions = torch.cat([torch.arange(n, device=numeric_device) for n in lengths])

    # Exercise the actual pre-allocation backend using the same tensors later
    # passed through physical pages. Expected attention never uses that backend.
    for backend in backends:
        q = random_tensor(rng, (sum(lengths), heads, dim), numeric_device,
                          scale=rng.choice([0.1, 1.0, 3.0]))
        k = random_tensor(rng, (sum(lengths), kv_heads, dim), numeric_device,
                          scale=rng.choice([0.1, 1.0, 3.0]))
        v = random_tensor(rng, k.shape, numeric_device,
                          scale=rng.choice([0.1, 1.0, 3.0]))
        output = backend.attend(q, k, v, positions, manager.prepare_ctx(initial_ctx))
        for lo, hi in zip(initial_ctx.cu_seq_lens_q, initial_ctx.cu_seq_lens_q[1:]):
            expected = dense_attention_reference(q[lo:hi], k[lo:hi], v[lo:hi], positions[lo:hi])
            torch.testing.assert_close(output[lo:hi], expected, atol=ATOL, rtol=RTOL)
        initial.append((q, k, v, output))

    blocks = count * ((2 * block_size + 10 + block_size - 1) // block_size)
    manager.reserve_storage(blocks, numeric_device, torch.float32)
    # Unwritten padding contains NaNs: reading it into attention must fail a
    # numerical comparison, even if the physical address is otherwise valid.
    manager.k_pool.fill_(torch.nan)
    manager.v_pool.fill_(torch.nan)
    shuffled_blocks(manager, rng)
    manager.register(seqs)
    histories = {}

    for step in range(rng.randint(5, 9)):
        if step == 0:
            batch = list(seqs)
            query_lengths = lengths
        else:
            if step == 2:
                victim = rng.choice(seqs)
                manager.release(victim.seq_id)
                seqs.remove(victim)
                for layer in range(layers):
                    histories.pop((layer, victim.seq_id))
                replacement = Sequence(10_000 + victim.seq_id, [0], max_tokens=1)
                seqs.append(replacement)
                manager.register([replacement])
                shuffled_blocks(manager, rng)
            batch = rng.sample(seqs, rng.randint(1, len(seqs)))
            # Cached requests append a single decode token. A replacement starts
            # with a fresh ragged prefill; the kernel accepts their packed rows.
            query_lengths = [1 if (0, seq.seq_id) in histories else rng.randint(1, block_size + 1)
                             for seq in batch]
        old_lengths = [histories[(0, seq.seq_id)][0].shape[0]
                       if (0, seq.seq_id) in histories else 0 for seq in batch]
        # Tensor-only fixtures have synthetic token histories. Keep their
        # lengths aligned with the actual KV target before explicit allocation.
        for seq, old, new in zip(batch, old_lengths, query_lengths):
            seq.token_ids = [0] * (old + new)
            seq.num_cached_tokens = old
            if old == 0:
                possible, hits = manager.can_allocate_prefill(seq)
                assert possible and not hits  # This suite disables prefix reuse.
                assert manager.allocate_prefill(seq, hits) == 0
            else:
                assert manager.can_allocate_decode(seq)
                manager.allocate_decode(seq)
        ctx = manager.prepare_ctx(make_context(
            [s.seq_id for s in batch], old_lengths, query_lengths,
        ))
        positions = torch.cat([torch.arange(old, old + new, device=numeric_device)
                               for old, new in zip(old_lengths, query_lengths)])
        for layer, backend in enumerate(backends):
            if step == 0:
                q, k, v, uncached = initial[layer]
            else:
                q = random_tensor(rng, (sum(query_lengths), heads, dim), numeric_device,
                                  scale=rng.choice([0.1, 1.0, 3.0]))
                k = random_tensor(rng, (sum(query_lengths), kv_heads, dim), numeric_device,
                                  scale=rng.choice([0.1, 1.0, 3.0]))
                v = random_tensor(rng, k.shape, numeric_device, scale=rng.choice([0.1, 1.0, 3.0]))
            actual = backend.attend(q, k, v, positions, ctx)
            for i, seq in enumerate(batch):
                lo, hi = ctx.cu_seq_lens_q[i:i + 2]
                old_k, old_v = histories.get((layer, seq.seq_id), (k[:0], v[:0]))
                full_k = torch.cat((old_k, k[lo:hi]))
                full_v = torch.cat((old_v, v[lo:hi]))
                histories[(layer, seq.seq_id)] = full_k, full_v
                expected = dense_attention_reference(q[lo:hi], full_k, full_v, positions[lo:hi])
                torch.testing.assert_close(actual[lo:hi], expected, atol=ATOL, rtol=RTOL)
            if step == 0:
                torch.testing.assert_close(actual, uncached, atol=ATOL, rtol=RTOL)
        # Check every live request, including requests absent from this batch.
        # Storage must preserve the original float32 values bit for bit.
        for (layer, sid), expected in histories.items():
            actual = manager.gather(layer, sid, expected[0].shape[0], manager.block_manager.table(sid))
            for stored, wanted in zip(actual, expected):
                torch.testing.assert_close(stored, wanted, atol=0, rtol=0)
