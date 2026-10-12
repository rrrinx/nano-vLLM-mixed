# Reproducing Burst and Long-Input Serving Experiments

[中文](zh-CN/docs/REPRODUCE.md)

Published measurements can be reanalyzed without a GPU, model, or dataset. Rerunning inference requires supplying those resources. This snapshot contains code and evidence for completed experiments; reading or installing it does not automatically launch experiments.

## 1. Recompute existing results

Run from the repository root:

```bash
python tools/verify_results.py
python -m pip install -r requirements-report.txt
python tools/render_serving_report.py
```

All notebook cells execute again, updating the English notebook, Markdown report, and five figures in `reports/long_input_serving/`. This only reads published measurements and does not load a model. Both report generators default to English; use `--language zh-CN` to update their Chinese editions under `docs/zh-CN/`. Chinese figures require a CJK font such as Noto Sans CJK; English figures do not.

The original paired analyzer is also included. To preserve the published tables, first copy the data into an ignored directory:

```bash
python -c "import shutil; shutil.copytree('reports/data/burst_requests/formal', 'artifacts/reanalysis/burst_requests')"
python experiments/analyze.py artifacts/reanalysis/burst_requests --mode burst
python -c "import shutil; shutil.copytree('reports/data/long_input_serving/formal', 'artifacts/reanalysis/long_input_serving')"
python experiments/analyze.py artifacts/reanalysis/long_input_serving --mode stream
```

These destination directories must not already exist, to avoid overwriting previous analyses.

## 2. Prepare an inference environment

Formal measurements used Linux, Python 3.12.3, PyTorch 2.8.0+cu128, Transformers 4.57.6, one RTX 4090, and Qwen3-4B / BF16. Repository validation used a different environment; see [VALIDATION.md](../VALIDATION.md).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export MODEL=/absolute/path/to/Qwen3-4B
```

The model directory should contain the config, tokenizer, and safetensors files. Place SPEED-Bench `throughput_2k` at:

```text
offline/SPEED-Bench/throughput_2k/test-00000-of-00001.parquet
```

The frozen file SHA-256 is `90df6c01a3cb7dc0243a54545c4fc9f6c40aa5ea472f2a8f085e76a4744704ee`. Source: [SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench). Preparation checks this hash and sample identities. A changed file requires a new data version; do not merge it into the old experiment.

Indices of samples used during exploration are included to exclude previously used inputs and reconstruct the confirmation sets. Text is not included. The sampling seed is fixed at 20261010.

## 3. Freeze the plan

```bash
bash run_experiments.sh --stage prepare --model "$MODEL" \
  --kv-blocks 565 --delay 1 --warmup 1 --window 5 --timeout 3600 \
  --no-install
```

Record the `FROZEN PLAN` path printed by the command and set:

```bash
export EXPERIMENT_PLAN=/absolute/path/to/this/repository/plans/GENERATED_TIMESTAMP
```

Shared settings: max_seqs=16, KV block_size=128, num_blocks=565, prefix_cache=false, ignore_eos=true, and one full warmup per policy. max_model_len is calculated from the frozen inputs; it was 2298 in the original experiment. `prepare` prepares inputs without running model inference.

## 4. Burst scheduling

```bash
bash run_experiments.sh --stage burst --model "$MODEL" --plan "$EXPERIMENT_PLAN" \
  --budgets 128,256,384,512 --block-ids 0,3,4 --no-install
```

There are three confirmation cohorts, each with four existing decode requests and four late prefill requests. Existing requests use 64-token prefixes and generate 256 tokens. Late requests use full long texts and generate 32 tokens. The shared arrival delay is 1 second.

The formal matrix has 3 cohorts × 4 budgets × 3 policies × 3 blocks = 108 policy replays. The entry point first runs a pilot to check overlap between late arrivals and existing decoding; pilots are excluded from formal samples. If another machine fails this overlap condition, calibrate one shared delay in a new plan, rather than tuning it separately per policy.

The original burst experiment reused 12 complete calls from an interrupted batch; see `reports/data/burst_requests/formal/reuse.json`. A fresh reproduction does not require `--reuse-from`. Do not use published analysis copies to resume execution.

## 5. Long-input serving

```bash
bash run_experiments.sh --stage online --model "$MODEL" --plan "$EXPERIMENT_PLAN" \
  --budgets 384,512 --rates 0.5,0.75,1.0 \
  --arrival-seeds 20261012,20261013 --block-ids 0,3,4 --no-install
```

The same 100 complete long inputs are used: 50 high_entropy and 50 low_entropy texts, with 128 output tokens per request. The formal matrix has 3 rates × 2 arrival seeds × 2 budgets × 3 policies × 3 blocks = 108 policy replays. Pilots and warmups run separately.

These rates reproduce the original report's conditions; they may represent different load levels on another GPU or model. Measurements across machines are not controlled scheduling comparisons.

## 6. Outputs and versions

New outputs go to `runs/`, which is ignored by Git. Add `--dry-run` to burst or serving commands to perform configuration preflight checks without model inference. `--stage analyze --results ... --mode burst/stream` only analyzes results.

`PACKAGE_MANIFEST.json` pins runtime code and metadata to detect changes during measurement; it also lists some directory READMEs. This translation updates only those documentation hashes, preserving runtime-code fingerprints. Report regeneration does not change runtime code. Deliberate runtime changes require an explicit new version and fingerprint manifest; do not bypass checks to mix results across versions.

Independent ablations of GPU batching efficiency have not been performed, and cross-policy generation-hash differences remain unresolved. Repeating inference can assess performance reproducibility but does not automatically establish answer equivalence.
