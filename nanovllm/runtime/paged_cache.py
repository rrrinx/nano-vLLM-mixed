import torch
from ..engine.block_manager import BlockManager
from dataclasses import replace
from .paged_attention_backend import PagedAttentionBackend


class PagedKVCacheManager:
    def __init__(
        self, num_layers, num_blocks, block_size,
        num_kv_heads, head_dim, device=None, dtype=None,
        *, enable_prefix_cache=False
    ):
        self.num_layers = num_layers
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.enable_prefix_cache = enable_prefix_cache
        self.num_blocks = num_blocks
        # None defers storage allocation until no-cache prefill warmup finishes.
        self.k_pool: torch.Tensor | None = None
        self.v_pool: torch.Tensor | None = None
        self.layer_lengths: dict[int, list[int]] = {}
        self.block_manager: BlockManager | None = None
        if num_blocks is not None:
            self.reserve_storage(num_blocks, device, dtype)

    def reserve_storage(self, num_blocks, device, dtype):
        if self.k_pool is not None:
            raise RuntimeError("KV pool is already allocated")
        shape = (self.num_layers, num_blocks, self.block_size, self.num_kv_heads, self.head_dim)
        k_pool = torch.empty(shape, device=device, dtype=dtype)
        v_pool = torch.empty(shape, device=device, dtype=dtype)
        self.block_manager = BlockManager(num_blocks, self.block_size, self.enable_prefix_cache)
        self.k_pool, self.v_pool = k_pool, v_pool
        self.num_blocks=num_blocks

        # print("block_size: ", self.block_size)
        # print("num_blocks: ", self.num_blocks)

    def register(self, seqs):
        if self.block_manager is None:
            raise RuntimeError("Allocate the KV pool before registering requests")
        self.block_manager.register(seqs)
        for seq in seqs:
            if seq.seq_id in self.layer_lengths:
                raise KeyError(f"seq {seq.seq_id} has been registered")
            self.layer_lengths[seq.seq_id] = [0 for i in range(self.num_layers)]

    def lengths(self, seq_id) -> int:
        """Return the common KV length between forwards, rejecting partial writes."""
        lengths = self.layer_lengths[seq_id]
        assert len(lengths) == self.num_layers
        length = lengths[0]
        if any(layer_length != length for layer_length in lengths):
            raise RuntimeError(
                f"every layer must have the same KV length for seq {seq_id}, got {tuple(lengths)}"
            )
        return length

    def append(self, layer_idx, seq_id, k_new, v_new, expected_old, slots):
        if seq_id not in self.layer_lengths:
            raise ValueError(f"seq {seq_id} has not been registered")
        if self.layer_lengths[seq_id][layer_idx] != expected_old:
            raise ValueError(f"the length of (seq {seq_id}, layer {layer_idx}) must equal to expected_old{expected_old}")
        if k_new.shape[-2:] != self.k_pool.shape[-2:] or v_new.shape[-2: ] != self.v_pool.shape[-2:]:
            raise ValueError(f"Invalid shape of k_new or v_new: {k_new.shape}")
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"Invalid layer_idx {layer_idx}")

        if k_new.shape[0] != v_new.shape[0]:
            raise ValueError(f"the length of k and v must be the same")
        if k_new.dtype != self.k_pool.dtype or v_new.dtype != self.k_pool.dtype:
            raise ValueError(f"the dtype of k or v is not valid")


        nq = k_new.shape[0]
        table = self.block_manager.table(seq_id)
        for i, slot in enumerate(slots):
            if slot != table[(expected_old + i) // self.block_size] * self.block_size + (expected_old + i) % self.block_size:
                raise ValueError("slots do not match the actual physical address")

        k_flat = self.k_pool[layer_idx].view(-1, *self.k_pool.shape[-2:])
        v_flat = self.v_pool[layer_idx].view(-1, *self.v_pool.shape[-2:])
        index = torch.tensor(slots, dtype=torch.long, device=k_flat.device)

        if len(slots) != nq:
            raise ValueError("one physical slot is required for each query")
        self._assert_writable(seq_id, expected_old, nq)


        k_flat.index_copy_(0, index, k_new)
        v_flat.index_copy_(0, index, v_new)

        self.layer_lengths[seq_id][layer_idx] += nq

    def gather(self, layer_idx, seq_id, valid_length, table_snapshot):
        if seq_id not in self.layer_lengths:
            raise ValueError(f"seq {seq_id} has not been registered")
        if table_snapshot != self.block_manager.table(seq_id):
            raise ValueError("table_snapshot is not equal to the actual table")

        if not 0 <= valid_length <= self.layer_lengths[seq_id][layer_idx]:
            raise ValueError(f"valid_length {valid_length} is invalid")

        needed = (valid_length + self.block_size - 1) // self.block_size
        blocks = torch.tensor(table_snapshot[:needed], dtype=torch.long, device=self.k_pool.device)
        k_gathered = self.k_pool[layer_idx].index_select(0, blocks).flatten(0, 1)
        v_gathered = self.v_pool[layer_idx].index_select(0, blocks).flatten(0, 1)


        return k_gathered[:valid_length], v_gathered[:valid_length]

    def release(self, seq_id):
        self.block_manager.release(seq_id)
        if seq_id not in self.layer_lengths:
            return
        self.layer_lengths.pop(seq_id)


    def prepare_ctx(self, ctx):
        if self.k_pool is None:
            if (ctx.is_prefill is not True or any(ctx.cache_lens)
                    or ctx.cu_seq_lens_q != ctx.cu_seq_lens_k
                    or ctx.block_tables is not None or ctx.slot_mapping is not None):
                raise ValueError("Unallocated KV pool only supports full prefill without cached tokens")
            return ctx


        for seq_id, cache_len, context_len in zip(ctx.seq_ids, ctx.cache_lens, ctx.context_lens):
            if self.lengths(seq_id) != cache_len:
                raise ValueError("cache_lens of context and manager do not match")
            require = (context_len + self.block_size - 1) // self.block_size
            if len(self.block_manager.table(seq_id)) < require:
                raise ValueError(f"allocate memory for seq {seq_id} before run")


        block_tables = tuple(self.block_manager.table(seq_id) for seq_id in ctx.seq_ids)

        slot_mapping = tuple( block_table[i // self.block_size] * self.block_size + (i % self.block_size)
                              for block_table, context_len, cache_len
                              in zip(block_tables, ctx.context_lens, ctx.cache_lens)
                              for i in range(cache_len, context_len))

        new_ctx = replace(ctx, block_size=self.block_size,
                          block_tables=block_tables, slot_mapping=slot_mapping
        )

        return new_ctx

    def make_backend(self, layer_idx):
        return PagedAttentionBackend(self, layer_idx)



    def allocate_prefill(self, seq, hits):
        seq_id = seq.seq_id
        if(seq.num_cached_tokens != 0 or seq.num_scheduled_tokens != 0 or
            self.lengths(seq_id) != 0 or
            self.block_manager.table(seq_id)):
            raise ValueError("prefix attachment requires a fresh registration")

        c = self.block_manager.allocate_prefix(seq, hits)
        self.layer_lengths[seq_id] = [c] * self.num_layers
        return c

    def can_allocate_prefill(self, seq):
        return self.block_manager.can_allocate_prefix(seq)

    def allocate_decode(self, seq):
        return self.block_manager.may_allocate(seq)

    def can_allocate_decode(self, seq):
        return self.block_manager.can_allocate(seq)


    def publish_seq(self, seq):
        expected = seq.num_cached_tokens + seq.num_scheduled_tokens
        if seq.num_scheduled_tokens <= 0:
            raise ValueError("there is no executed query to publish")
        if self.lengths(seq.seq_id) != expected:
            raise RuntimeError("cannot publish KV before every layer finishes")
        self.block_manager.hash_blocks(seq)

    def _assert_writable(self, seq_id, expected_old, count):
        table = self.block_manager.table(seq_id)
        touched = {table[t // self.block_size]
                   for t in range(expected_old, expected_old + count)}
        for block_id in touched:
            block = self.block_manager.blocks[block_id]
            if block.ref_count != 1 or block.hash != -1:
                raise ValueError("KV writes must target private unpublished blocks")
