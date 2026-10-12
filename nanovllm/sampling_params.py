from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 0.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature == 0.0, "Only greedy sampling is implemented"
        assert type(self.max_tokens) is int and self.max_tokens >= 0
        assert type(self.ignore_eos) is bool
