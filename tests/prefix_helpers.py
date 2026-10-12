"""Stage 7 assertions against the active runtime, never a replacement allocator."""

from collections import Counter
from copy import deepcopy

import torch

from nanovllm.engine.input import prepare_prefill
from nanovllm.engine.sequence import SequenceStatus
from nanovllm.runtime.paged_cache import PagedKVCacheManager
from numeric_helpers import checked_runner_logits


def assert_block_conservation(manager):
    free = list(manager.free_block_id)
    used = {block.block_id for block in manager.blocks if block.ref_count > 0}
    assert len(free) == len(set(free))
    assert set(free).isdisjoint(used)
    assert set(free) | used == set(range(manager.num_blocks))
    references = Counter(bid for table in manager.tables.values() for bid in table)
    for table in manager.tables.values():
        assert len(table) == len(set(table))
    for block in manager.blocks:
        assert block.ref_count == references[block.block_id]
        assert (block.ref_count == 0) == (block.block_id in free)
        assert (block.ref_count > 0) == (block.block_id in used)
        if block.ref_count > 1:
            assert block.hash != -1, "shared blocks must have committed complete KV"
        if block.hash != -1:
            assert len(block.token_ids) == manager.block_size
            assert len(block.previous_token_ids) in (0, manager.block_size)
    for key, bid in manager.hash_to_block_id.items():
        assert manager.blocks[bid].hash == key


def block_snapshot(manager):
    return deepcopy((manager.tables, list(manager.free_block_id),
                     [vars(block) for block in manager.blocks], manager.hash_to_block_id))


def assert_idle_cache(manager):
    assert not manager.layer_lengths
    assert not manager.block_manager.tables
    assert all(block.ref_count == 0 for block in manager.block_manager.blocks)
    assert_block_conservation(manager.block_manager)
    assert len(manager.block_manager.free_block_id) == manager.block_manager.num_blocks
    # Committed cold-cache keys are allowed to remain after every request exits.


def make_cache(blocks=4, size=4, layers=2, heads=1, dim=2, device="cpu", enabled=True):
    manager = PagedKVCacheManager(
        num_layers=layers, num_blocks=blocks, block_size=size,
        num_kv_heads=heads, head_dim=dim, device=device, dtype=torch.float32,
        enable_prefix_cache=enabled,
    )
    manager.k_pool.fill_(torch.nan)
    manager.v_pool.fill_(torch.nan)
    return manager


def attach_request(manager, seq):
    manager.register([seq])
    possible, hits = manager.can_allocate_prefill(seq)
    assert possible, "test request must fit its configured pool"
    cached = manager.allocate_prefill(seq, hits)
    assert seq.num_cached_tokens == 0
    seq.num_cached_tokens = cached
    seq.status = SequenceStatus.RUNNING
    seq.num_scheduled_tokens = seq.num_tokens - seq.num_cached_tokens
    return len(hits)


def write_planned_kv(manager, seq, rng):
    """Real append/gather storage, with synthetic values rather than model math."""
    _, _, ctx = prepare_prefill([seq], manager.k_pool.device)
    ctx = manager.prepare_ctx(ctx)
    count = seq.num_scheduled_tokens
    generator = torch.Generator().manual_seed(rng.randrange(2**31))
    expected = []
    for layer in range(manager.num_layers):
        shape = (count, manager.num_kv_heads, manager.head_dim)
        k = torch.randn(shape, generator=generator).to(manager.k_pool.device)
        v = torch.randn(shape, generator=generator).to(manager.v_pool.device)
        manager.append(layer, seq.seq_id, k, v, ctx.cache_lens[0], ctx.slot_mapping)
        expected.append((k, v))
    return ctx, expected


def save_blocks(manager, block_ids):
    index = torch.tensor(block_ids, dtype=torch.long, device=manager.k_pool.device)
    return (manager.k_pool.index_select(1, index).clone(),
            manager.v_pool.index_select(1, index).clone())


def assert_blocks_unchanged(manager, block_ids, before):
    for actual, expected in zip(save_blocks(manager, block_ids), before):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)


def observe_numerical_runner(engine, reference, monkeypatch):
    """Observe real forward inputs; independently check all logits/KV against HF."""
    runner = engine.model_runner
    original = runner.run_logits
    events = []

    def capture(module, inputs):
        ids, positions, ctx = inputs
        events.append({
            "prefill": ctx.is_prefill, "ids": ids.tolist(), "positions": positions.tolist(),
            "seq_ids": ctx.seq_ids, "cached": ctx.cache_lens,
            "queries": tuple(b - a for a, b in zip(ctx.cu_seq_lens_q, ctx.cu_seq_lens_q[1:])),
            "tables": ctx.block_tables, "slots": ctx.slot_mapping,
        })

    handle = runner.model.register_forward_pre_hook(capture)

    def checked(seqs, is_prefill):
        return checked_runner_logits(runner, seqs, is_prefill, reference, original)

    monkeypatch.setattr(runner, "run_logits", checked)
    return events, handle
