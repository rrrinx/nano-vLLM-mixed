# Implementation Tests

[中文](../docs/zh-CN/tests/README.md)

Run `python -m pytest tests -q` from the repository root. Default numerical tests use random small models on CPU float32 and do not require downloading real weights.

- `test_layers_numerics.py`, `test_model_numerics.py`, `test_generation_numerics.py`: layer calculations, model logits/KV, and greedy tokens.
- `test_chunked_*`, `test_mixed_*`: chunked prefill, mixed scheduling, and numerical consistency.
- `test_prefix_*`, `test_cache_contracts.py`: shared prefixes, reference counts, cache boundaries, and recovery.
- `test_online_*`, `test_math500_*`: experiment timing, metrics, configuration, and platform behavior.
- `test_experiment_blocks.py`: execution order, complete-record reuse, and paired statistics.

`MY_NANO_VLLM_TEST_CASES` and `MY_NANO_VLLM_TEST_SEED` control randomized test scale and seeds. Real-checkpoint tests require explicit opt-in; see `test_real_checkpoint.py` for the applicable variables.

These tests do not establish identical Qwen3-4B / BF16 outputs across different batch shapes. Output-hash differences in the burst and serving workloads remain unresolved.
