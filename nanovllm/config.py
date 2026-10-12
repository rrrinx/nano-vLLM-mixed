import os
import logging
from dataclasses import dataclass
from typing import ClassVar

from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 2048
    max_num_seqs: int = 32
    max_model_len: int = 1024
    gpu_memory_utilization: float = 0.9
    kvcache_block_size: int = 128
    num_kvcache_blocks: int = -1
    scheduling_policy: str = "mixed"
    enable_prefix_cache: bool = True
    enable_tqdm: bool = True
    log_level: str = "INFO"
    device: str = "cuda"
    dtype: str | None = "float32"
    hf_config: AutoConfig | None = None
    eos: int | None = None

    # Default checkpoint for example.py; callers can pass another local model.
    model_path: ClassVar[str] = "D:/models/Qwen3-0.6B"

    def __post_init__(self):
        if self.scheduling_policy not in ("separate", "interleaved", "mixed"):
            raise ValueError("unknown scheduling policy")
        assert os.path.isdir(self.model), f"Model directory does not exist: {self.model}"
        assert all(type(value) is int and value > 0 for value in (
            self.max_num_batched_tokens, self.max_num_seqs,
            self.max_model_len, self.kvcache_block_size,
        )), "Engine limits must be positive integers"
        assert 0 < self.gpu_memory_utilization <= 1
        assert type(self.enable_prefix_cache) is bool, "enable_prefix_cache must be a boolean"
        assert type(self.enable_tqdm) is bool, "enable_tqdm must be a boolean"
        assert isinstance(self.log_level, str) and getattr(
            logging, self.log_level.upper(), None
        ) in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL), (
            "log_level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL"
        )
        self.log_level = self.log_level.upper()
        assert type(self.num_kvcache_blocks) is int and (
            self.num_kvcache_blocks == -1 or self.num_kvcache_blocks > 0
        )
        self.hf_config = AutoConfig.from_pretrained(self.model, local_files_only=True)
        assert self.hf_config.model_type == "qwen3", "Only Qwen3 is supported"
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        # Chunked prefill limits each forward's query count independently of
        # the total history length. Scheduler still checks model/KV capacity.
