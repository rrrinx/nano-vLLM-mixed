# Qwen3 Inference Engine with Mixed Prefill/Decode Scheduling

[中文](docs/zh-CN/README.md)

A PyTorch implementation of a Qwen3 inference engine based on the nano-vLLM architecture, extended with chunked prefill, mixed prefill/decode batches, paged KV storage, prefix sharing, and preemption recovery. Controlled request bursts and continuous arrivals evaluate the scheduling behavior.

**Main finding:** On an RTX 4090 with Qwen3-4B, decode-priority mixed scheduling reduces completion latency and generation stalls in long-input replays, with throughput gains at high load. The tradeoff is a longer wait for the first token of new requests.

## Start here

| Question | Reading |
| --- | --- |
| What is implemented, and how do the modules interact? | [Architecture and code guide](docs/ARCHITECTURE.md) |
| What are the benefits and costs under continuous long-input arrivals? | [Serving performance report](reports/long_input_serving/README.md) · [Notebook](reports/long_input_serving/mixed_scheduling_report.ipynb) |
| What happens when new prefill work arrives during decoding? | [Burst scheduling report](reports/burst_requests/README.md) |
| How can I verify the measurements or rerun experiments? | [Reproduction guide](docs/REPRODUCE.md) · [Validation record](VALIDATION.md) |

## Implementation

- **Scheduling:** Prefill priority (`separate`), prefill/decode alternation across iterations (`interleaved`), and mixed iterations (`mixed`). Mixed reserves token and sequence slots for decoding before scheduling prefill chunks.
- **Execution:** Packs prefill chunks and decode tokens from different requests into one model forward pass, maintaining query ranges, absolute positions, and cache lengths. Partial-prefill commits are separate from sampling.
- **Caching:** Paged KV storage, chained-hash prefix sharing with reference counts, recomputation after preemption, chunked continuation, and recovery of generation history.
- **Validation:** Numerical comparisons against Hugging Face, cache-boundary and recovery tests, three-policy replays, rotated execution order, and paired statistics.

The current implementation uses PyTorch paged attention and greedy sampling. Optimized upstream kernels, CUDA Graphs, tensor parallelism, and an HTTP server are outside its current scope.

## Representative results

Configuration: RTX 4090, Qwen3-4B / BF16, approximately 2K-token SPEED-Bench inputs, fixed 128-token outputs, Poisson arrivals at 1 req/s, token budget 384, max_seqs=16, prefix cache disabled. Each value is the mean of three execution blocks. The baseline is prefill-priority `separate`.

| Arrival seed | Output throughput: separate → mixed | Mean E2E: separate → mixed | Mean TTFT: separate → mixed |
| --- | --- | --- | --- |
| 20261012 | 78.26 → 84.61 token/s (+8.11%) | 59.65 → 48.03 s (−19.50%) | 14.06 → 26.51 s |
| 20261013 | 78.10 → 83.77 token/s (+7.27%) | 44.19 → 32.96 s (−25.40%) | 2.17 → 12.06 s |

![Time to first token and subsequent generation time](reports/long_input_serving/figures/04_latency_decomposition.png)

The full serving experiment covers three arrival rates, two arrival seeds, two budgets, and three policies, totaling 108 formal replays. The burst experiment uses three independent input cohorts, four budgets, and three repeats, also retaining all 108 formal replays.

Results are limited to one model, one GPU, synthetic arrivals, and fixed output lengths. Throughput includes arrival waits and final draining; it is not maximum steady-state capacity. Latency excludes network overhead. The reports retain all benefits, negative results, and limitations.

## Repository layout

```text
nanovllm/                    Inference engine, scheduling, model layers, and paged KV
examples/generate.py         Generation example using a local checkpoint
tests/                       Numerical, cache, scheduling, and experiment-tool tests
experiments/                 Shared preparation, execution, timing, and statistics
  metadata/                  Sample indices for reconstructing input splits
  common/                    Timing, replay, and diagnostics
  online/                    Request arrivals and serving metrics
  simulation/                CPU scheduling simulations
reports/
  burst_requests/            Burst scheduling report
  long_input_serving/        Serving report, notebook, and figures
  data/                      All formal measurements and provenance
tools/                       Measurement verification and report generation
docs/                        Architecture and reproduction guides
  zh-CN/                     Chinese documentation, reports, and notebook
```

Experiment entry points are organized by function. Model paths and device arguments adapt the same code to different environments.

## Usage and reproduction

Python 3.12 is recommended. Dependencies use the formal experiment environment (PyTorch 2.8.0 / Transformers 4.57.6) as a reference. Supply your own model and dataset; neither is included.

```bash
python -m pip install -r requirements.txt
python examples/generate.py /path/to/Qwen3-4B --dtype bfloat16 --device cuda
python -m pytest tests -q
```

Verify published measurements without a model, GPU, or PyTorch:

```bash
python tools/verify_results.py
```

Regenerate the English reports:

```bash
python -m pip install -r requirements-report.txt
python tools/render_burst_report.py
python tools/render_serving_report.py
```

Both report commands also accept `--language zh-CN` for the Chinese edition. Full inference experiments use `python experiments/launch.py`; see the [reproduction guide](docs/REPRODUCE.md).

## Sources

Architecture reference: [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm), with its [MIT notice](licenses/nano-vllm-MIT.txt) retained. Inputs come from [SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench); Hugging Face Qwen3 provides the numerical reference. See [published data](reports/data/README.md) for export rules and [third-party notices](THIRD_PARTY_NOTICES.md) for attribution.
