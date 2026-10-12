from collections import OrderedDict
from hashlib import blake2b

class Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.previous_token_ids: tuple[int, ...] = ()
        self.token_ids: tuple[int, ...] = ()

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = ()
        self.previous_token_ids = ()


def compute_hash(token_ids, prefix: int = -1) -> int:
    h = blake2b(digest_size=8)
    h.update(b"wo shi shanli ling huo de gou")
    if prefix == -1:
        h.update(b"\x00")
    else:
        h.update(b"\x01")
        h.update(prefix.to_bytes(8, "little", signed=False))
    for token in token_ids:
        if type(token) is not int or token < 0:
            raise ValueError("token IDs must be nonnegative integers")
        h.update(token.to_bytes(8, "little", signed=False))
    return int.from_bytes(h.digest(), "little", signed=False)


class BlockManager:
    def __init__(self, num_blocks, block_size, enable_prefix_cache=False):

        if any(type(x) is not int or x <= 0
               for x in (num_blocks, block_size)):
            raise ValueError("pool dimensions must be positive integers")

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_cache = enable_prefix_cache
        self.free_block_id = OrderedDict({i:None for i in range(num_blocks)})
        self.tables: dict[int, list[int]] = {}
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}

        if block_size <= 0:
            raise ValueError("block_size must be positive")

    def table(self, seq_id):
        return tuple(self.tables[seq_id])

    compute_hash = staticmethod(compute_hash)

    def register(self, seqs):
        for seq in seqs:
            if seq.seq_id in self.tables:
                raise KeyError(f"seq {seq.seq_id} has been registered")

        if len(seqs) != len(set(seq.seq_id for seq in seqs)):
            raise KeyError(f"seq_id in seqs cannot be repeated")

        for seq in seqs:
            self.tables[seq.seq_id] = []


    def _allocate_block(self):
        block_id = self.free_block_id.popitem(False)[0]
        block = self.blocks[block_id]
        assert block.ref_count == 0
        old_hash = block.hash
        if old_hash != -1 and self.hash_to_block_id.get(old_hash) == block_id:
            del self.hash_to_block_id[old_hash]
        block.reset()
        return block_id



    def _requirements_allocate(self, seq):
            seq_id, length = seq.seq_id, seq.num_tokens
            if seq_id not in self.tables:
                raise KeyError(f"seq {seq_id} has not been registered")
            
            total = (length + self.block_size - 1) // self.block_size
            return max(0, total - len(self.tables[seq_id]))

    def can_allocate(self, seq) -> bool:
        return self._requirements_allocate(seq) <= len(self.free_block_id)
    
    def may_allocate(self, seq):
        seq_id = seq.seq_id
        require = self._requirements_allocate(seq)

        if(require) > len(self.free_block_id):
            raise RuntimeError("insufficient free kv blocks")

        for _ in range(require):
            self.tables[seq_id].append(self._allocate_block())
        return require

    def _prefix_items(self, seq, begin, end, initial_hash):
        h = initial_hash

        for i in range(begin, end):
            stop = (i + 1) * self.block_size
            tokens = tuple(seq.token_ids[i * self.block_size:stop])
            assert len(tokens) == self.block_size
            h = self.compute_hash(tokens, h)

            yield i, h, tokens


    def _cached_block_ids(self, seq):
        if not self.enable_prefix_cache:
            return []
        hits = []
        end =  (seq.num_tokens - 1) // self.block_size
        previous_token_ids = ()
        for _, h, tokens in self._prefix_items(seq, 0, end, -1):
            block_id = self.hash_to_block_id.get(h)
            if block_id is None:
                break
            block = self.blocks[block_id]
            if (block.hash != h or block.token_ids != tokens
                or block.previous_token_ids != previous_token_ids):
                break
            previous_token_ids = tokens
            hits.append(block_id)
        return hits



    def _acquire_block_prefix(self, block_id):
        block = self.blocks[block_id]
        if block.ref_count == 0:
            self.free_block_id.pop(block_id)
        block.ref_count += 1

    def can_allocate_prefix(self, seq) -> int:
        if self.tables[seq.seq_id]:
            raise ValueError("prefill attachment requires an empty table")
        hits = self._cached_block_ids(seq)
        total = (seq.num_tokens + self.block_size - 1) // self.block_size
        cold_hits = sum(self.blocks[block_id].ref_count == 0 for block_id in hits)
        required_free = total - len(hits) + cold_hits
        return required_free <= len(self.free_block_id), hits # bool block_ids

    # must execute after can_allocate_prefix
    # initial admission or recovery after preemption
    def allocate_prefix(self, seq, hits) -> int:
        table = self.tables[seq.seq_id]

        assert not table

        for block_id in hits:
            self._acquire_block_prefix(block_id)
            table.append(block_id)
        total = (seq.num_tokens + self.block_size - 1) // self.block_size
        for _ in range(total - len(hits)):
            table.append(self._allocate_block())

        return len(hits) * self.block_size


    def hash_blocks(self, seq):
        if not self.enable_prefix_cache:
            return
        c, q = seq.num_cached_tokens, seq.num_scheduled_tokens
        start = c // self.block_size
        end = (c + q) // self.block_size
        table = self.tables[seq.seq_id]

        initial_hash = -1 if start == 0 else self.blocks[table[start - 1]].hash
        previous_token_ids = () if start == 0 else self.blocks[table[start - 1]].token_ids

        if start == end:
            return

        for i, h, tokens in self._prefix_items(seq, start, end, initial_hash):
            block = self.blocks[table[i]]
            assert block.ref_count == 1 and block.hash == -1
            block.hash = h
            block.token_ids = tokens
            block.previous_token_ids = previous_token_ids
            self.hash_to_block_id[h] = block.block_id
            previous_token_ids = block.token_ids

    def release(self, seq_id):
        table = self.tables.pop(seq_id, None)
        if table is None:
            return
        for bid in reversed(table):
            block = self.blocks[bid]
            assert block.ref_count > 0
            block.ref_count -= 1
            if block.ref_count == 0:
                self.free_block_id[bid]=None
