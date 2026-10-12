"""Resource boundaries after the CacheManager facade refactor.

Lookup fixtures explicitly seed already committed metadata to isolate lookup
from publication failures. They never replace production lookup/allocation.
Numerical suites separately verify publication against real all-layer KV.
"""

from copy import deepcopy

import pytest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from numeric_helpers import make_context
from prefix_helpers import assert_block_conservation, block_snapshot, make_cache


def seed_committed_metadata(manager, seq, count):
    """Represent full blocks from a completed donor; no model execution here."""
    table = manager.table(seq.seq_id)
    previous_hash = -1
    previous_tokens = ()
    for index in range(count):
        block = manager.blocks[table[index]]
        tokens = tuple(seq.token_ids[index * manager.block_size:(index + 1) * manager.block_size])
        assert len(tokens) == manager.block_size
        block.hash = manager.compute_hash(tokens, previous_hash)
        block.token_ids = tokens
        block.previous_token_ids = previous_tokens
        manager.hash_to_block_id[block.hash] = block.block_id
        previous_hash, previous_tokens = block.hash, tokens


@pytest.mark.parametrize("size", [1, 2, 4, 7])
def test_decode_allocates_only_missing_capacity_and_preserves_progress(size):
    manager = make_cache(blocks=8, size=size, enabled=False)
    seq = Sequence(1, [5] * size, 8)
    manager.register([seq])
    possible, hits = manager.can_allocate_prefill(seq)
    assert possible and hits == []
    assert manager.allocate_prefill(seq, hits) == 0
    seq.status = SequenceStatus.RUNNING
    for length in range(size, 5 * size + 1):
        seq.token_ids = [5] * length
        seq.num_cached_tokens = length - 1
        seq.num_scheduled_tokens = 1
        manager.layer_lengths[seq.seq_id] = [length - 1] * manager.num_layers
        before = block_snapshot(manager.block_manager)
        assert manager.can_allocate_decode(seq)
        assert block_snapshot(manager.block_manager) == before
        lengths = manager.lengths(seq.seq_id)
        manager.allocate_decode(seq)
        assert len(manager.block_manager.table(seq.seq_id)) == (length + size - 1) // size
        assert manager.lengths(seq.seq_id) == lengths
        assert (seq.num_cached_tokens, seq.num_scheduled_tokens) == (length - 1, 1)
        after = block_snapshot(manager.block_manager)
        manager.allocate_decode(seq)
        assert block_snapshot(manager.block_manager) == after
        assert_block_conservation(manager.block_manager)


def test_decode_capacity_failure_does_not_mutate_resources():
    manager = make_cache(blocks=1, size=4, enabled=False)
    seq = Sequence(1, [1] * 4, 2)
    manager.register([seq])
    assert manager.can_allocate_prefill(seq) == (True, [])
    manager.allocate_prefill(seq, [])
    seq.append_token(2)
    before = block_snapshot(manager.block_manager)
    assert not manager.can_allocate_decode(seq)
    with pytest.raises(RuntimeError, match="insufficient"):
        manager.allocate_decode(seq)
    assert block_snapshot(manager.block_manager) == before


def test_failed_prefill_probe_does_not_mutate_resources():
    manager = BlockManager(1, 4, True)
    seq = Sequence(1, list(range(9)), 1)
    manager.register([seq])
    before = block_snapshot(manager)
    possible, hits = manager.can_allocate_prefix(seq)
    assert not possible and hits == []
    assert block_snapshot(manager) == before
    # A rejected probe must not be passed to allocate_prefix. Its failure
    # behavior outside that precondition is deliberately not part of this test.
    assert manager.table(seq.seq_id) == ()


def test_successful_prefill_probe_is_consumed_without_repeating_lookup(monkeypatch):
    manager = BlockManager(3, 4, True)
    seq = Sequence(1, list(range(9)), 1)
    manager.register([seq])
    before = block_snapshot(manager)
    possible, hits = manager.can_allocate_prefix(seq)
    assert possible and hits == []
    assert block_snapshot(manager) == before

    def unexpected(*args):
        raise AssertionError("allocation must consume the successful probe without another lookup")

    monkeypatch.setattr(manager, "_cached_block_ids", unexpected)
    monkeypatch.setattr(manager, "_prefix_items", unexpected)
    assert manager.allocate_prefix(seq, hits) == 0
    assert len(manager.table(1)) == 3
    assert_block_conservation(manager)


@pytest.mark.parametrize("budget", [4, 8])
def test_scheduler_consumes_successful_probe_before_any_resource_change(monkeypatch, budget):
    manager = make_cache(blocks=4, size=4, enabled=False)
    scheduler = Scheduler(manager, 2, budget, 16)
    seq = Sequence(1, list(range(4)), 1)
    scheduler.add(seq)
    query, allocate = manager.can_allocate_prefill, manager.allocate_prefill
    events = []
    plan = {}

    def observed_query(candidate):
        possible, hits = query(candidate)
        events.append("query")
        plan.update(possible=possible, hits=hits,
                    blocks=block_snapshot(manager.block_manager),
                    lengths=deepcopy(manager.layer_lengths), sequence=deepcopy(vars(candidate)))
        return possible, hits

    def observed_allocate(candidate, hits):
        assert events == ["query"] and plan["possible"]
        assert hits is plan["hits"]
        assert block_snapshot(manager.block_manager) == plan["blocks"]
        assert manager.layer_lengths == plan["lengths"]
        assert vars(candidate) == plan["sequence"]
        events.append("allocate")
        return allocate(candidate, hits)

    monkeypatch.setattr(manager, "can_allocate_prefill", observed_query)
    monkeypatch.setattr(manager, "allocate_prefill", observed_allocate)
    assert scheduler._schedule_prefill(budget, 2) == [seq]
    assert events == ["query", "allocate"]
    assert seq.num_cached_tokens == 0 and seq.num_scheduled_tokens == 4
    assert seq.status == SequenceStatus.RUNNING
    assert not scheduler.waiting
    assert_block_conservation(manager.block_manager)


def test_scheduler_does_not_allocate_after_a_failed_probe(monkeypatch):
    manager = make_cache(blocks=2, size=4, enabled=False)
    scheduler = Scheduler(manager, 2, 8, 16)
    seq = Sequence(1, list(range(5)), 1)
    scheduler.add(seq)
    blocker = Sequence(2, list(range(5)), 1)
    manager.register([blocker])
    possible, hits = manager.can_allocate_prefill(blocker)
    assert possible
    manager.allocate_prefill(blocker, hits)
    before = block_snapshot(manager.block_manager), deepcopy(manager.layer_lengths), deepcopy(vars(seq))
    query = manager.can_allocate_prefill
    probes = []

    def observed_query(candidate):
        result = query(candidate)
        probes.append(result)
        return result

    def unexpected(*args):
        raise AssertionError("a rejected probe must not reach allocation")

    monkeypatch.setattr(manager, "can_allocate_prefill", observed_query)
    monkeypatch.setattr(manager, "allocate_prefill", unexpected)
    assert scheduler._schedule_prefill(8, 2) == []
    assert probes == [(False, [])]
    assert (block_snapshot(manager.block_manager), manager.layer_lengths, vars(seq)) == before
    assert list(scheduler.waiting) == [seq]


def test_scheduler_does_not_allocate_when_remaining_batch_budget_is_insufficient(monkeypatch):
    manager = make_cache(blocks=4, size=4, enabled=False)
    scheduler = Scheduler(manager, 2, 4, 16)
    first, second = Sequence(1, [1, 2, 3], 1), Sequence(2, [4, 5, 6], 1)
    scheduler.add(first)
    scheduler.add(second)
    allocate = manager.allocate_prefill
    allocations = []

    def observed_allocate(candidate, hits):
        allocations.append(candidate.seq_id)
        return allocate(candidate, hits)

    monkeypatch.setattr(manager, "allocate_prefill", observed_allocate)
    assert scheduler._schedule_prefill(4, 2) == [first]
    assert allocations == [first.seq_id]
    assert list(scheduler.waiting) == [second]
    assert manager.block_manager.table(second.seq_id) == ()
    assert second.num_cached_tokens == second.num_scheduled_tokens == 0
    assert manager.lengths(second.seq_id) == 0
    assert_block_conservation(manager.block_manager)


def test_prepare_ctx_requires_preallocation_and_does_not_allocate():
    manager = make_cache(blocks=3, size=4, enabled=False)
    seq = Sequence(1, list(range(9)), 1)
    manager.register([seq])
    ctx = make_context([1], [0], [9])
    before = block_snapshot(manager.block_manager)
    with pytest.raises(ValueError, match="allocate"):
        manager.prepare_ctx(ctx)
    assert block_snapshot(manager.block_manager) == before
    assert manager.lengths(1) == 0
    assert manager.can_allocate_prefill(seq) == (True, [])
    manager.allocate_prefill(seq, [])
    before = block_snapshot(manager.block_manager)
    lengths = deepcopy(manager.layer_lengths)
    prepared = manager.prepare_ctx(ctx)
    assert block_snapshot(manager.block_manager) == before
    assert manager.layer_lengths == lengths
    assert prepared.slot_mapping == tuple(range(9))
    assert ctx.block_tables is ctx.slot_mapping is None


def test_prefill_returns_cached_length_without_modifying_sequence():
    manager = make_cache(blocks=3, size=4)
    donor = Sequence(1, [10, 11, 12, 13, 20], 1)
    manager.register([donor])
    assert manager.can_allocate_prefill(donor) == (True, [])
    manager.allocate_prefill(donor, [])
    seed_committed_metadata(manager.block_manager, donor, 1)
    receiver = Sequence(2, [10, 11, 12, 13, 21], 1)
    manager.register([receiver])
    shared = manager.block_manager.table(1)[0]
    possible, hits = manager.can_allocate_prefill(receiver)
    assert possible and hits == [shared]
    cached = manager.allocate_prefill(receiver, hits)
    assert cached == 4
    assert receiver.num_cached_tokens == receiver.num_scheduled_tokens == 0
    assert receiver.status == SequenceStatus.WAITING
    assert manager.lengths(2) == 4
    assert manager.block_manager.blocks[shared].ref_count == 2


def test_prefix_lookup_uses_donor_metadata_with_an_empty_request_table():
    manager = BlockManager(4, 4, True)
    donor = Sequence(1, list(range(5)), 1)
    probe = Sequence(2, list(range(4)) + [20], 1)
    manager.register([donor, probe])
    assert manager.can_allocate_prefix(donor) == (True, [])
    manager.allocate_prefix(donor, [])
    seed_committed_metadata(manager, donor, 1)
    before = block_snapshot(manager)
    assert manager.can_allocate_prefix(probe) == (True, [manager.table(1)[0]])
    assert manager.table(2) == ()
    assert block_snapshot(manager) == before


def test_publication_without_a_new_full_block_does_not_hash(monkeypatch):
    manager = BlockManager(4, 4, True)
    seq = Sequence(1, list(range(10)), 1)
    manager.register([seq])
    assert manager.can_allocate_prefix(seq) == (True, [])
    manager.allocate_prefix(seq, [])
    seed_committed_metadata(manager, seq, 2)
    seq.num_cached_tokens, seq.num_scheduled_tokens = 9, 1
    def unexpected(*args):
        raise AssertionError("no complete new block should require hashing")
    monkeypatch.setattr(manager, "compute_hash", unexpected)
    before = block_snapshot(manager)
    manager.hash_blocks(seq)
    assert block_snapshot(manager) == before


def test_publication_hashes_only_the_newly_completed_block(monkeypatch):
    manager = BlockManager(4, 4, True)
    seq = Sequence(1, list(range(12)), 1)
    manager.register([seq])
    assert manager.can_allocate_prefix(seq) == (True, [])
    manager.allocate_prefix(seq, [])
    seed_committed_metadata(manager, seq, 2)
    seq.num_cached_tokens, seq.num_scheduled_tokens = 11, 1
    calls = []
    original = manager.compute_hash
    def counted(tokens, prefix=-1):
        calls.append((tuple(tokens), prefix))
        return original(tokens, prefix)
    monkeypatch.setattr(manager, "compute_hash", counted)
    manager.hash_blocks(seq)
    assert calls == [(tuple(range(8, 12)), manager.blocks[manager.table(1)[1]].hash)]
    assert manager.blocks[manager.table(1)[2]].hash != -1
    # Incremental publication must link to the immediately preceding page,
    # including when physical IDs are unrelated to logical block positions.
    assert manager.blocks[manager.table(1)[2]].previous_token_ids == tuple(range(4, 8))


def test_equal_prefixes_in_distinct_parent_pages_can_reuse_the_same_suffix():
    manager = BlockManager(7, 4, True)
    a = Sequence(1, list(range(9)), 1)
    b = Sequence(2, list(range(9)), 1)
    manager.register([a, b])
    assert manager.can_allocate_prefix(a) == (True, [])
    manager.allocate_prefix(a, [])
    assert manager.can_allocate_prefix(b) == (True, [])
    manager.allocate_prefix(b, [])
    seed_committed_metadata(manager, a, 2)
    seed_committed_metadata(manager, b, 1)
    probe = Sequence(3, list(range(8)) + [30], 1)
    manager.register([probe])
    # The index currently represents B's first page and A's second page.
    # Both have the exact same causal history despite distinct parent IDs.
    assert manager.can_allocate_prefix(probe) == (True, [manager.table(2)[0], manager.table(1)[1]])
