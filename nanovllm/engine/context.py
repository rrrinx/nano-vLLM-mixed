from dataclasses import dataclass

@dataclass(frozen=True)
class Context:
    is_prefill: bool | None # None for mixed
    seq_ids: tuple[int, ...]
    cu_seq_lens_q: tuple[int, ...] # prefix sum of lens of q
    cu_seq_lens_k: tuple[int, ...]
    cache_lens: tuple[int, ...] # len of history kv cache
    context_lens: tuple[int, ...] # len of kv cache after write kv
    max_seq_lens_q: int
    max_seq_lens_k: int
    block_size: int = 0
    block_tables: tuple[tuple[int, ...], ...] | None = None
    slot_mapping: tuple[int, ...] | None = None

    @property
    def last_indices(self):
        return tuple(end - 1 for end in self.cu_seq_lens_q[1:])
