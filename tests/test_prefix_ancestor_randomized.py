"""Seeded multi-block histories using the production hash without injection."""

from collections import OrderedDict
import random

import pytest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from numeric_helpers import BASE_SEED, CASE_COUNT
from prefix_helpers import assert_block_conservation, block_snapshot


@pytest.mark.parametrize("seed", [BASE_SEED + 200_000 + i for i in range(CASE_COUNT)])
def test_random_deep_histories_hot_and_cold_reuse_keep_their_own_ancestors(seed):
    rng = random.Random(seed)
    size, depth = rng.choice([1, 2, 3, 4, 7, 8]), rng.randint(3, 6)
    a_root = [rng.randrange(100) for _ in range(size)]
    b_root = list(a_root)
    b_root[0] += 100
    middle = [[1000 + level * 100 + rng.randrange(100) for _ in range(size)]
              for level in range(depth - 2)]
    target = tuple(9000 + rng.randrange(100) for _ in range(size))
    common_middle = [token for chunk in middle for token in chunk]
    manager = BlockManager(2 * depth + 3, size, True)
    order = list(manager.free_block_id)
    rng.shuffle(order)
    manager.free_block_id = OrderedDict.fromkeys(order)
    a = Sequence(1, a_root + common_middle + list(target) + [1], 1)
    manager.register([a])
    possible, hits = manager.can_allocate_prefix(a)
    assert possible and not hits
    manager.allocate_prefix(a, hits)
    a.num_scheduled_tokens = a.num_tokens
    manager.hash_blocks(a)

    b = Sequence(2, b_root + common_middle + [2], 1)
    manager.register([b])
    possible, hits = manager.can_allocate_prefix(b)
    assert possible and not hits
    manager.allocate_prefix(b, hits)
    b.num_scheduled_tokens = b.num_tokens
    manager.hash_blocks(b)

    probe = Sequence(3, b_root + common_middle + list(target) + [3], 1)
    manager.register([probe])
    before = block_snapshot(manager)
    possible, hits = manager.can_allocate_prefix(probe)
    assert possible
    assert block_snapshot(manager) == before
    expected = list(manager.table(b.seq_id)[:depth - 1])
    assert hits == expected
    assert manager.allocate_prefix(probe, hits) == (depth - 1) * size
    probe.num_cached_tokens = (depth - 1) * size
    probe.num_scheduled_tokens = probe.num_tokens - probe.num_cached_tokens
    manager.hash_blocks(probe)
    probe_table = manager.table(probe.seq_id)
    assert probe_table[depth - 1] != manager.table(a.seq_id)[depth - 1]
    assert manager.blocks[probe_table[depth - 1]].hash != manager.blocks[manager.table(a.seq_id)[depth - 1]].hash
    assert_block_conservation(manager)

    # Release both donors and the receiver in a seeded order, then reuse the
    # newly published full prefix from cold pages before any overwrite.
    exits = [a.seq_id, b.seq_id, probe.seq_id]
    rng.shuffle(exits)
    for sid in exits:
        manager.release(sid)
        assert_block_conservation(manager)
    cold = Sequence(4, probe.token_ids[:-1] + [4], 1)
    manager.register([cold])
    before = block_snapshot(manager)
    possible, hits = manager.can_allocate_prefix(cold)
    assert possible and hits == list(probe_table[:depth])
    assert block_snapshot(manager) == before
    assert manager.allocate_prefix(cold, hits) == depth * size
    assert manager.table(cold.seq_id)[:depth] == probe_table[:depth]
    manager.release(cold.seq_id)
    assert_block_conservation(manager)

    # Recycle every physical page using an unrelated first token. All old
    # representatives must be removed and the same history must become cold.
    order = list(manager.free_block_id)
    rng.shuffle(order)
    manager.free_block_id = OrderedDict.fromkeys(order)
    overwrite = Sequence(5, [777] * (manager.num_blocks * size), 1)
    manager.register([overwrite])
    possible, hits = manager.can_allocate_prefix(overwrite)
    assert possible and not hits
    manager.allocate_prefix(overwrite, hits)
    assert not manager.hash_to_block_id
    manager.release(overwrite.seq_id)
    fresh = Sequence(6, list(cold.token_ids), 1)
    manager.register([fresh])
    assert manager.can_allocate_prefix(fresh) == (True, [])
    assert_block_conservation(manager)
