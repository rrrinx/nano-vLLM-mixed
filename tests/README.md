# 当前实现的测试

在仓库根目录执行 `python -m pytest tests -q`。默认数值测试在 CPU float32 上使用随机小模型，不需要下载真实权重。

- `test_layers_numerics.py`、`test_model_numerics.py`、`test_generation_numerics.py`：层计算、模型 logits/KV 与 greedy token。
- `test_chunked_*`、`test_mixed_*`：分块 Prefill、混合批调度和数值一致性。
- `test_prefix_*`、`test_cache_contracts.py`：共享前缀、引用计数、缓存边界和恢复。
- `test_online_*`、`test_math500_*`：实验计时、指标、配置及平台行为。
- `test_experiment_blocks.py`：执行顺序、完整记录复用及配对统计。

随机规模和种子由 `MY_NANO_VLLM_TEST_CASES`、`MY_NANO_VLLM_TEST_SEED` 控制。真实 checkpoint 测试需要显式启用，具体变量以 `test_real_checkpoint.py` 为准。

这些测试不证明Qwen3-4B / BF16 不同批次形状下的输出完全一致；突发请求与长输入在线负载输出哈希差异仍是未解决问题。
