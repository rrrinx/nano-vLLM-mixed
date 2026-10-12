# 来源与声明

[English](../../THIRD_PARTY_NOTICES.md)

- 推理引擎架构参考 GeeeekExplorer/nano-vllm。对应 MIT 版权和许可原文保留在 `licenses/nano-vllm-MIT.txt`；此文件不将作者自己的新增代码自动改授同一许可证。
- 使用 Hugging Face Transformers 的 Qwen3 实现进行数值对照，通过本地 checkpoint 加载模型。仓库不分发模型权重和 tokenizer。
- 实验输入来自 NVIDIA SPEED-Bench 的 `throughput_2k`。不分发 Parquet、原文或可解码的输入 token 序列；公开副本只包含样本索引、长度、指纹和性能测量。重新推理需要使用者自行准备数据。
- 本次整理未替项目自有代码选择新的开源许可证。维护者可在正式发布时按自己的意愿设置仓库许可证；上游声明应保留。
