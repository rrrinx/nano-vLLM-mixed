# 突发请求与长输入在线负载复现指南

[English](../../REPRODUCE.md)

这里区分重新分析已公开的测量，以及重新加载模型进行推理。前者不需要 GPU、模型或数据集；后者需要自行准备这些资源。此发布快照只包含已完成实验的代码和证据，命令不会在阅读或安装时自动执行。

## 1. 重算已有结果

从仓库根目录运行：

```bash
python tools/verify_results.py
python -m pip install -r requirements-report.txt
python tools/render_serving_report.py --language zh-CN
```

Notebook 的全部代码单元将重新执行，更新 `docs/zh-CN/reports/long_input_serving/` 下的 Notebook、Markdown 和五张图。重算只使用公开测量，不加载模型。Linux 若缺少中文字体，安装 Noto Sans CJK 后再执行；数字计算不受字体影响。

原始配对分析器也已保留。若要重跑并保留原发布表，先复制到忽略目录：

```bash
python -c "import shutil; shutil.copytree('reports/data/burst_requests/formal', 'artifacts/reanalysis/burst_requests')"
python experiments/analyze.py artifacts/reanalysis/burst_requests --mode burst
python -c "import shutil; shutil.copytree('reports/data/long_input_serving/formal', 'artifacts/reanalysis/long_input_serving')"
python experiments/analyze.py artifacts/reanalysis/long_input_serving --mode stream
```

这些目标目录需尚不存在，避免覆盖先前分析。

## 2. 准备独立推理环境

正式测量环境为 Linux、Python 3.12.3、PyTorch 2.8.0+cu128、Transformers 4.57.6、单张 RTX 4090、Qwen3-4B / BF16。仓库整理测试可能使用另一环境，见 `VALIDATION.md`；二者不要混淆。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export MODEL=/absolute/path/to/Qwen3-4B
```

模型目录应包含 config、tokenizer 和 safetensors。准备 SPEED-Bench `throughput_2k` 文件到：

```text
offline/SPEED-Bench/throughput_2k/test-00000-of-00001.parquet
```

本次冻结文件 SHA-256 为 `90df6c01a3cb7dc0243a54545c4fc9f6c40aa5ea472f2a8f085e76a4744704ee`。数据来源为 [SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench)。准备脚本检查该哈希与样本身份；若文件发生变化，应记录为新的数据版本，不能直接合并旧实验。

探索阶段已用样本的索引已随包保留，用于排除已用输入并重建相同的确认集合；正文不在包内。抽样种子固定为 20261010。

## 3. 冻结计划

```bash
bash run_experiments.sh --stage prepare --model "$MODEL" \
  --kv-blocks 565 --delay 1 --warmup 1 --window 5 --timeout 3600 \
  --no-install
```

记录命令输出的 `FROZEN PLAN` 路径，然后填入以下变量：

```bash
export EXPERIMENT_PLAN=/absolute/path/to/this/repository/plans/GENERATED_TIMESTAMP
```

通用配置：max_seqs=16、KV block_size=128、num_blocks=565、prefix_cache=false、ignore_eos=true、每策略一次完整预热。max_model_len 由冻结输入计算，原实验为 2298。`prepare` 仅准备输入，不执行模型推理。

## 4. 突发请求调度

```bash
bash run_experiments.sh --stage burst --model "$MODEL" --plan "$EXPERIMENT_PLAN" \
  --budgets 128,256,384,512 --block-ids 0,3,4 --no-install
```

三个确认输入组；每组四个已有 Decode 请求和四个后到 Prefill 请求。已有请求使用 64-token 前缀、输出 256 token；后到请求使用完整长文本、输出 32 token；共同延迟为 1 秒。

正式部分为 3 输入组 × 4 预算 × 3 策略 × 3 block，即 108 次策略回放。入口先执行 pilot，检查后到请求与已有 Decode 的重叠；pilot 不计入正式样本。新机器若不能满足重叠条件，应在新计划中统一校准 delay，不能按策略分别调参。

原突发请求实验复用了中断批次中 12 个完整调用，具体见 `reports/data/burst_requests/formal/reuse.json`。从零复现不需要 `--reuse-from`，也不要拿公开分析副本做执行恢复。

## 5. 长输入在线服务

```bash
bash run_experiments.sh --stage online --model "$MODEL" --plan "$EXPERIMENT_PLAN" \
  --budgets 384,512 --rates 0.5,0.75,1.0 \
  --arrival-seeds 20261012,20261013 --block-ids 0,3,4 --no-install
```

同一套 100 条完整长输入，50 条 high_entropy、50 条 low_entropy，每请求输出 128 token。正式部分为 3 速率 × 2 到达种子 × 2 预算 × 3 策略 × 3 block，即 108 次策略回放；pilot 和预热另外执行。

这些速率复现原报告条件，在不同 GPU 或模型上未必仍代表相同负载强度。不能将新机器的指标与旧机器直接当作调度对照。

## 6. 结果与版本

新输出写入 `runs/`，默认不被 Git 收录。`--dry-run` 可加入突发请求与长输入在线负载命令，执行配置预检但不加载模型推理。`--stage analyze --results ... --mode burst/stream` 只做分析。

`PACKAGE_MANIFEST.json` 固定运行代码和元数据，防止测量时混入代码修改；清单也包含少量目录 README；本次翻译只更新对应文档的哈希，运行代码指纹保持不变。报告重算不改动运行代码。若主动修改运行代码，应明确产生新版本和新的指纹清单，不要绕过检查来混合旧结果。

当前未开展 GPU 合批效率的独立消融，也未解决跨策略生成哈希差异。重新推理可观察性能重复性，不自动完成答案等价验证。
