# Validation Record

[中文](docs/zh-CN/VALIDATION.md)

Date: 2026-10-12. These checks were performed using the public repository layout. No new real-model performance experiments were run.

## Tests

```text
python -m pytest tests -q --confcutdir=. --basetemp=artifacts/layout-tests
1550 passed, 61 skipped, 16 subtests passed in 150.96s
```

Environment: Windows, Python 3.12.14, PyTorch 2.7.1+cu128, Transformers 4.57.6. Numerical tests default to CPU float32. Skips cover disabled real-checkpoint tests and Linux / POSIX-specific tests.

After the final experiment-command renaming, `tests/test_experiment_blocks.py` was run separately: 6 passed, 16 subtests passed. It covers burst/online dispatch, execution order, complete-call reuse, and paired statistics.

## Entry points, measurements, and reports

- `python examples/generate.py --help` and `python experiments/launch.py --help` ran successfully.
- `python tools/verify_results.py` read all 72 formal result files across both workloads and checked policy order, configuration completeness, request timestamps, latency identities, throughput, and means. Burst requests include 864 request measurements; serving includes 10800.
- Inference-engine source fingerprints match the measured code. Experiment helpers have adjusted imports and path resolution for the public layout; historical and current fingerprints and file mappings are retained in `reports/data/PROVENANCE.json`.
- `python tools/render_burst_report.py` regenerated the report for all 12 burst configurations.
- `python tools/render_serving_report.py` executed all 9 notebook code cells and produced 5 figures and a Markdown edition without execution errors.
- Documentation links, Python syntax, file manifests, and archive integrity were checked.

## Scope

The performance reports use existing Linux / RTX 4090 / Qwen3-4B / BF16 experiments. CPU tests do not replace GPU performance or output-equivalence validation. Cross-policy output-hash differences remain unexplained; conclusions retain the fixed-length replay and TTFT-tradeoff qualifications.

## Bilingual documentation verification

The English and Chinese notebooks were each re-executed successfully (9 code cells and 5 figures per edition). Generated numerical table values match the original notebook. Both burst editions regenerated successfully. All local documentation links and runtime-manifest hashes were checked, and the full 72-file measurement audit passed. This documentation update did not rerun inference or the full engine test suite.
