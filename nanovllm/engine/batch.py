from dataclasses import dataclass
from typing import Literal
from .sequence import Sequence, SequenceStatus

Phase = Literal["prefill", "decode"]

@dataclass(frozen=True)
class BatchEntry:
    seq: Sequence
    phase: Phase
    cache_len: int
    query_len: int
    num_tokens: int

    @property
    def should_sample(self):
        return self.cache_len + self.query_len == self.num_tokens

    def validate_live(self):
        s = self.seq
        if self.phase not in ("prefill", "decode"):
            raise ValueError("unknown request phase")
        if s.status not in (SequenceStatus.WAITING, SequenceStatus.RUNNING):
            raise ValueError("batch needs live requests")
        if (s.num_cached_tokens, s.num_scheduled_tokens, s.num_tokens) != (
            self.cache_len, self.query_len, self.num_tokens
        ):
            raise ValueError("stale batch entry")
        if not (0 <= self.cache_len and 0 < self.query_len
                and self.cache_len + self.query_len <= self.num_tokens):
            raise ValueError("invalid token interval")
        if self.phase == "decode" and not (
            s.status == SequenceStatus.RUNNING
            and self.query_len == 1 and self.cache_len == self.num_tokens - 1
        ):
            raise ValueError("decode must consume the uncached last token")

@dataclass(frozen=True)
class BatchPlan:
    entries: tuple[BatchEntry, ...]

    @property
    def seqs(self):
        return tuple(e.seq for e in self.entries)

    @property
    def kind(self):
        phases = {e.phase for e in self.entries}
        return next(iter(phases)) if len(phases) == 1 else "mixed"

    @property
    def query_tokens(self):
        return sum(e.query_len for e in self.entries)

    @property
    def legacy_flag(self):
        return {"prefill": True, "decode": False, "mixed": None}[self.kind]

    def validate_live(self):
        if not self.entries or len({e.seq.seq_id for e in self.entries}) != len(self.entries):
            raise ValueError("batch needs nonempty unique requests")
        for entry in self.entries:
            entry.validate_live()