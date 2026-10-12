"""Prefix indexing, reference counts and eviction using the real BlockManager."""

from collections import OrderedDict
import random

import pytest

from nanovllm.engine.block_manager import BlockManager, compute_hash
from nanovllm.engine.sequence import Sequence
from numeric_helpers import CASE_SEEDS
from prefix_helpers import assert_block_conservation, block_snapshot


def publish_fixture(manager, seq, query_count=None):
    # Integer-only fixture: simulate successful forward completion to isolate
    # BlockManager.hash_blocks. Actual layer completion is tested in cache/HF tests.
    seq.num_scheduled_tokens = seq.num_tokens if query_count is None else query_count
    manager.hash_blocks(seq)


def test_hash_depends_on_history_and_uses_valid_token_ids():
    first = compute_hash([10, 11, 12, 13])
    different = compute_hash([90, 91, 92, 93])
    assert compute_hash([20, 21, 22, 23], first) != compute_hash([20, 21, 22, 23], different)
    assert first == compute_hash((10, 11, 12, 13))
    for invalid in ([-1], [1.5], [True]):
        with pytest.raises(ValueError):
            compute_hash(invalid)


def test_new_allocation_has_one_reference_and_release_is_idempotent():
    manager = BlockManager(4, 4, True)
    seq = Sequence(17, list(range(9)), 1)
    manager.register([seq])
    assert manager.can_allocate_prefix(seq) == (True, [])
    assert manager.allocate_prefix(seq, []) == 0
    assert [manager.blocks[bid].ref_count for bid in manager.table(17)] == [1, 1, 1]
    assert_block_conservation(manager)
    manager.release(17)
    before = block_snapshot(manager)
    manager.release(17)
    assert block_snapshot(manager) == before
    assert_block_conservation(manager)


@pytest.mark.parametrize("length,hits", [(1, 0), (3, 0), (4, 0), (8, 1), (9, 2)])
def test_last_logical_block_is_never_reused(length, hits):
    manager = BlockManager(8, 4, True)
    donor = Sequence(1, list(range(9)), 1)
    manager.register([donor])
    assert manager.can_allocate_prefix(donor) == (True, [])
    manager.allocate_prefix(donor, [])
    publish_fixture(manager, donor)
    probe = Sequence(2, list(range(length)), 1)
    manager.register([probe])
    before = block_snapshot(manager)
    possible, block_ids = manager.can_allocate_prefix(probe)
    assert possible and len(block_ids) == hits
    assert block_snapshot(manager) == before
    cached = manager.allocate_prefix(probe, block_ids)
    assert cached == hits * 4 and length - cached > 0
    assert_block_conservation(manager)


def test_active_hit_waits_for_private_tail_then_reprobes_after_release():
    manager = BlockManager(3, 4, True)
    a, b = Sequence(1, list(range(9)), 1), Sequence(2, list(range(8)) + [29], 1)
    manager.register([a, b])
    assert manager.can_allocate_prefix(a) == (True, [])
    manager.allocate_prefix(a, [])
    publish_fixture(manager, a)
    before = block_snapshot(manager)
    possible, hits = manager.can_allocate_prefix(b)
    assert not possible and hits == list(manager.table(a.seq_id)[:2])
    # A failed probe is read-only and must never be passed to allocation.
    assert block_snapshot(manager) == before
    manager.release(1)
    # Releasing a request changes resource state: obtain a fresh probe.
    possible, hits = manager.can_allocate_prefix(b)
    assert possible and len(hits) == 2  # Cold hits also consume free slots.
    manager.allocate_prefix(b, hits)
    assert_block_conservation(manager)


def test_cold_hits_are_pinned_before_fifo_can_overwrite_them():
    manager = BlockManager(3, 4, True)
    a = Sequence(1, list(range(9)), 1)
    manager.register([a])
    assert manager.can_allocate_prefix(a) == (True, [])
    manager.allocate_prefix(a, [])
    publish_fixture(manager, a)
    manager.release(1)
    manager.free_block_id = OrderedDict.fromkeys([0, 1, 2])
    b = Sequence(2, list(range(8)) + [29], 1)
    manager.register([b])
    possible, hits = manager.can_allocate_prefix(b)
    assert possible and hits == [0, 1]
    manager.allocate_prefix(b, hits)
    assert manager.table(2) == (0, 1, 2)
    assert manager.blocks[0].hash != -1 and manager.blocks[1].hash != -1
    assert_block_conservation(manager)


def test_duplicate_representative_eviction_does_not_delete_the_newer_key():
    manager = BlockManager(6, 4, True)
    a, b = Sequence(1, list(range(9)), 1), Sequence(2, list(range(9)), 1)
    manager.register([a, b])
    assert manager.can_allocate_prefix(a) == (True, [])
    manager.allocate_prefix(a, [])
    assert manager.can_allocate_prefix(b) == (True, [])  # Allocation is not publication.
    manager.allocate_prefix(b, [])
    publish_fixture(manager, a)
    publish_fixture(manager, b)
    latest = dict(manager.hash_to_block_id)
    assert set(latest.values()) == {3, 4}
    manager.release(1)
    c = Sequence(3, [29] * 9, 1)
    manager.register([c])
    manager.may_allocate(c)  # Overwrite old representatives 2, 1, 0.
    assert manager.hash_to_block_id == latest
    manager.release(2)
    d = Sequence(4, [30] * 9, 1)
    manager.register([d])
    manager.may_allocate(d)  # Overwrite current representatives 5, 4, 3.
    assert not manager.hash_to_block_id
    assert_block_conservation(manager)


def test_eviction_of_nonrepresentative_with_an_already_missing_key():
    manager = BlockManager(6, 4, True)
    a, b = Sequence(1, list(range(9)), 1), Sequence(2, list(range(9)), 1)
    manager.register([a, b])
    assert manager.can_allocate_prefix(a) == (True, [])
    manager.allocate_prefix(a, [])
    assert manager.can_allocate_prefix(b) == (True, [])
    manager.allocate_prefix(b, [])
    publish_fixture(manager, a)
    publish_fixture(manager, b)
    manager.release(1)
    manager.release(2)
    # Evict current representatives first, leaving duplicate old blocks whose
    # metadata still carries a hash no longer present in the index.
    manager.free_block_id = OrderedDict.fromkeys([5, 4, 3, 2, 1, 0])
    probe = Sequence(3, [90] * 24, 1)
    manager.register([probe])
    manager.may_allocate(probe)
    assert not manager.hash_to_block_id
    assert_block_conservation(manager)


@pytest.mark.parametrize("seed", CASE_SEEDS, ids=lambda seed: f"seed-{seed}")
def test_random_shared_lifetimes_cold_hits_and_overwrites(seed):
    rng = random.Random(seed)
    size = rng.choice([1, 2, 3, 4, 7])
    shared_count = rng.randint(1, 3)
    count = rng.randint(2, 5)
    manager = BlockManager(shared_count + count + 3, size, True)
    order = list(manager.free_block_id)
    rng.shuffle(order)
    manager.free_block_id = OrderedDict.fromkeys(order)
    common = [rng.randrange(97) for _ in range(shared_count * size)]
    a = Sequence(rng.randrange(100, 200), common + [1], 1)
    manager.register([a])
    assert manager.can_allocate_prefix(a) == (True, [])
    manager.allocate_prefix(a, [])
    publish_fixture(manager, a)
    shared = manager.table(a.seq_id)[:shared_count]
    live = [a]
    for sid in range(200, 200 + count - 1):
        seq = Sequence(sid, common + [rng.randrange(97)], 1)
        manager.register([seq])
        before = block_snapshot(manager)
        possible, hits = manager.can_allocate_prefix(seq)
        assert possible and len(hits) == shared_count
        assert block_snapshot(manager) == before
        manager.allocate_prefix(seq, hits)
        assert manager.table(sid)[:shared_count] == shared
        live.append(seq)
        assert_block_conservation(manager)
    rng.shuffle(live)
    for remaining, seq in zip(range(count - 1, -1, -1), live):
        manager.release(seq.seq_id)
        assert all(manager.blocks[bid].ref_count == remaining for bid in shared)
        assert_block_conservation(manager)
    order = list(manager.free_block_id)
    rng.shuffle(order)
    manager.free_block_id = OrderedDict.fromkeys(order)
    cold = Sequence(999, common + [2], 1)
    manager.register([cold])
    possible, hits = manager.can_allocate_prefix(cold)
    assert possible and len(hits) == shared_count
    manager.allocate_prefix(cold, hits)
    assert manager.table(999)[:shared_count] == shared
    manager.release(999)
    overwrite = Sequence(1000, [96] * manager.num_blocks * size, 1)
    manager.register([overwrite])
    manager.may_allocate(overwrite)
    assert not manager.hash_to_block_id
    assert all(block.hash == -1 for block in manager.blocks)
    assert_block_conservation(manager)
