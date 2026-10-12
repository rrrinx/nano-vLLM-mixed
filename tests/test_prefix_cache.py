"""Real shared KV storage, suffix attention and full-layer publication barriers."""

import random
from collections import OrderedDict

import pytest
import torch

from nanovllm.engine.input import prepare_prefill
from nanovllm.engine.sequence import Sequence
from numeric_helpers import ATOL, RTOL, CASE_SEEDS, dense_attention_reference, random_tensor
from prefix_helpers import (
    assert_block_conservation, assert_blocks_unchanged, assert_idle_cache,
    attach_request, make_cache, save_blocks, write_planned_kv,
)


def test_publication_waits_for_all_layers_and_rejects_repeated_registration():
    manager = make_cache()
    seq = Sequence(17, list(range(9)), 5)
    attach_request(manager, seq)
    assert manager.lengths(seq.seq_id) == 0
    _, _, ctx = prepare_prefill([seq], "cpu")
    ctx = manager.prepare_ctx(ctx)
    values = torch.ones((9, 1, 2))
    manager.append(0, seq.seq_id, values, values, 0, ctx.slot_mapping)
    with pytest.raises(RuntimeError, match="every layer"):
        manager.publish_seq(seq)
    assert not manager.block_manager.hash_to_block_id
    assert manager.layer_lengths[seq.seq_id] == [9, 0]
    with pytest.raises(RuntimeError, match="every layer"):
        manager.lengths(seq.seq_id)
    with pytest.raises(KeyError, match="registered"):
        manager.register([seq])
    assert manager.layer_lengths[seq.seq_id] == [9, 0]
    manager.append(1, seq.seq_id, values, values, 0, ctx.slot_mapping)
    assert manager.lengths(seq.seq_id) == 9
    manager.publish_seq(seq)
    assert len(manager.block_manager.hash_to_block_id) == 2
    manager.release(seq.seq_id)
    assert_idle_cache(manager)


@pytest.mark.parametrize("first_exit", [1, 2], ids=["donor-exits", "receiver-exits"])
def test_shared_release_preserves_other_request_and_private_tail(first_exit):
    manager = make_cache()
    rng = random.Random(20261004)
    a, b = Sequence(1, list(range(9)), 5), Sequence(2, list(range(8)) + [29], 5)
    assert attach_request(manager, a) == 0
    write_planned_kv(manager, a, rng)
    manager.publish_seq(a)
    assert attach_request(manager, b) == 2
    assert manager.lengths(b.seq_id) == 8
    shared = manager.block_manager.table(1)[:2]
    before = save_blocks(manager, shared)
    ctx, _ = write_planned_kv(manager, b, rng)
    assert ctx.cache_lens == (8,) and ctx.context_lens == (9,)
    assert ctx.slot_mapping == (12,)
    manager.publish_seq(b)
    assert_blocks_unchanged(manager, shared, before)
    other = 3 - first_exit
    other_table = manager.block_manager.table(other)
    other_values = save_blocks(manager, other_table)
    manager.release(first_exit)
    assert manager.block_manager.table(other) == other_table
    assert manager.lengths(other) == 9
    assert all(manager.block_manager.blocks[bid].ref_count == 1 for bid in shared)
    assert_blocks_unchanged(manager, other_table, other_values)
    assert_block_conservation(manager.block_manager)
    manager.release(other)
    assert_idle_cache(manager)


def test_published_blocks_are_read_only_even_with_one_reference():
    manager = make_cache()
    seq = Sequence(1, list(range(9)), 5)
    attach_request(manager, seq)
    write_planned_kv(manager, seq, random.Random(7))
    manager.publish_seq(seq)
    before = save_blocks(manager, manager.block_manager.table(1))
    assert manager.block_manager.blocks[0].ref_count == 1
    with pytest.raises(ValueError, match="private unpublished"):
        manager._assert_writable(1, 0, 1)
    manager._assert_writable(1, 9, 1)  # The existing partial private tail is writable.
    assert_blocks_unchanged(manager, manager.block_manager.table(1), before)


def test_bad_suffix_slots_are_rejected_before_any_pool_write():
    manager = make_cache()
    a, b = Sequence(1, list(range(9)), 1), Sequence(2, list(range(8)) + [29], 1)
    attach_request(manager, a)
    write_planned_kv(manager, a, random.Random(9))
    manager.publish_seq(a)
    attach_request(manager, b)
    before = save_blocks(manager, list(range(4)))
    values = torch.ones((1, 1, 2))
    for invalid_slots in [(0,), ()]:
        with pytest.raises(ValueError):
            manager.append(0, 2, values, values, 8, invalid_slots)
        assert manager.lengths(2) == 8
        assert_blocks_unchanged(manager, list(range(4)), before)


@pytest.mark.parametrize("seed", CASE_SEEDS, ids=lambda seed: f"seed-{seed}")
@torch.inference_mode()
def test_random_prefix_suffix_attention_matches_independent_float64(seed, numeric_device):
    rng = random.Random(seed)
    size = rng.choice([1, 2, 3, 4, 7])
    layers = rng.randint(1, 3)
    heads = rng.choice([1, 2, 4, 6, 8])
    kv_heads = rng.choice([h for h in range(1, heads + 1) if heads % h == 0])
    dim = rng.choice([2, 4, 8, 12])
    manager = make_cache(8, size, layers, kv_heads, dim, numeric_device)
    order = list(manager.block_manager.free_block_id)
    rng.shuffle(order)
    manager.block_manager.free_block_id.clear()
    manager.block_manager.free_block_id.update(OrderedDict.fromkeys(order))
    common = [rng.randrange(97) for _ in range(2 * size)]
    a = Sequence(100, common + [3], 5)
    attach_request(manager, a)
    _, _, ctx_a = prepare_prefill([a], numeric_device)
    ctx_a = manager.prepare_ctx(ctx_a)
    positions_a = torch.arange(a.num_tokens, device=numeric_device)
    donor_values = []
    for layer in range(layers):
        q = random_tensor(rng, (a.num_tokens, heads, dim), numeric_device)
        k = random_tensor(rng, (a.num_tokens, kv_heads, dim), numeric_device)
        v = random_tensor(rng, k.shape, numeric_device)
        actual = manager.make_backend(layer).attend(q, k, v, positions_a, ctx_a)
        expected = dense_attention_reference(q, k, v, positions_a)
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
        donor_values.append((k[:2 * size].clone(), v[:2 * size].clone()))
    manager.publish_seq(a)
    shared = manager.block_manager.table(100)[:2]
    before = save_blocks(manager, shared)

    count = rng.randint(1, size + 1)
    b = Sequence(900, common + [4] + [rng.randrange(97) for _ in range(count - 1)], 5)
    assert attach_request(manager, b) == 2
    assert manager.lengths(900) == 2 * size
    _, positions, ctx_b = prepare_prefill([b], numeric_device)
    ctx_b = manager.prepare_ctx(ctx_b)
    assert positions.tolist() == list(range(2 * size, 2 * size + count))
    assert len(ctx_b.slot_mapping) == count
    for layer in range(layers):
        q = random_tensor(rng, (count, heads, dim), numeric_device, scale=rng.choice([0.1, 1.0, 3.0]))
        k = random_tensor(rng, (count, kv_heads, dim), numeric_device)
        v = random_tensor(rng, k.shape, numeric_device)
        actual = manager.make_backend(layer).attend(q, k, v, positions, ctx_b)
        full_k = torch.cat((donor_values[layer][0], k))
        full_v = torch.cat((donor_values[layer][1], v))
        expected = dense_attention_reference(q, full_k, full_v, positions)
        torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
        stored = manager.gather(layer, 900, b.num_tokens, manager.block_manager.table(900))
        for value, wanted in zip(stored, (full_k, full_v)):
            torch.testing.assert_close(value, wanted, atol=0, rtol=0)
    manager.publish_seq(b)
    assert_blocks_unchanged(manager, shared, before)
    assert_block_conservation(manager.block_manager)
    exits = [100, 900]
    rng.shuffle(exits)
    for sid in exits:
        manager.release(sid)
        assert_block_conservation(manager.block_manager)
    assert_idle_cache(manager)
