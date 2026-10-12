from dataclasses import dataclass, field
from enum import Enum, auto

class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

@dataclass
class Sequence:
    seq_id: int
    token_ids: list[int]
    max_tokens: int
    eos_token_id: int | None = None
    status: SequenceStatus = field(default=SequenceStatus.WAITING, init=False)
    num_prompt_tokens: int = field(init=False)
    num_cached_tokens: int = field(default=0, init=False)
    num_scheduled_tokens: int = field(default=0, init=False)

    def __post_init__(self):
        self.token_ids = list(self.token_ids)

        if not self.token_ids:
            raise ValueError("need to set prompt")

        if self.max_tokens < 0:
            raise ValueError("max_tokens >= 0")

        self.num_prompt_tokens = len(self.token_ids)

        if self.max_tokens == 0:
            self.status = SequenceStatus.FINISHED

    @property
    def num_tokens(self):
        return len(self.token_ids)

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    def append_token(self, token_id):
        self.token_ids.append(int(token_id))
