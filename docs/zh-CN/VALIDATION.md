# 项目验证记录

[English](../../VALIDATION.md)

日期：2026-10-12。以下检查在本项目的公开目录结构下执行，没有运行新的真实模型性能实验。

## 测试

```text
python -m pytest tests -q --confcutdir=. --basetemp=artifacts/layout-tests
1550 passed, 61 skipped, 16 subtests passed in 150.96s
```

运行环境：Windows、Python 3.12.14、PyTorch 2.7.1+cu128、Transformers 4.57.6；数值测试默认使用 CPU float32。跳过项为未启用的真实 checkpoint 测试及 Linux / POSIX 专用测试。

实验命令最终命名调整后另行运行 `tests/test_experiment_blocks.py`，结果为 6 passed、16 subtests passed，覆盖 burst/online 转发、执行顺序、完整调用复用和配对统计。

## 入口、数据和报告

- `python examples/generate.py --help` 与 `python experiments/launch.py --help` 正常运行。
- `python tools/verify_results.py` 读取两种负载的全部 72 份正式结果，核验策略次序、组合完整性、逐请求时间戳、延迟恒等式、吞吐及均值。突发请求包含 864 条请求测量，在线服务包含 10800 条请求测量。
- 推理引擎源文件与性能实验记录的指纹一致。实验辅助模块为公开目录调整了导入和路径解析；历史指纹、当前指纹及文件位置映射保留在 `reports/data/PROVENANCE.json`。
- `python tools/render_burst_report.py` 重新生成全部 12 个突发配置的结果页。
- `python tools/render_serving_report.py` 完整执行 Notebook 的 9 个代码单元，生成 5 张图及 Markdown 阅读版，无执行错误。
- 文档链接、Python 语法、文件清单和压缩包完整性均经过检查。

## 边界

性能报告来自 Linux、RTX 4090、Qwen3-4B / BF16 的既有实验；此次 CPU 测试不替代 GPU 性能或输出等价验证。跨策略输出哈希差异仍未解释，结论保留固定长度回放和 TTFT 权衡等限定。

## 双语文档验证

中英文 Notebook 均已完整重新执行，每版 9 个代码单元、5 张图；生成表格的数值与原 Notebook 一致。两版突发报告均成功重新生成。已检查全部本地文档链接和运行文件清单哈希，72 份测量文件的完整核验通过。本次文档更新未重新运行推理或全套引擎测试。
