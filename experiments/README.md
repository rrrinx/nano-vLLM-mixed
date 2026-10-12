# Experiment Tools

[中文](../docs/zh-CN/experiments/README.md)

All environments share these entry points. Model paths, devices, and experiment parameters determine execution.

| File or directory | Purpose |
| --- | --- |
| `launch.py` | Unified entry point for preparation, preflight checks, bursts, and continuous arrivals |
| `prepare.py` | Frozen plans from the model tokenizer, dataset, and sample indices |
| `run_matrix.py`, `block_design.py` | Configuration matrices, policy execution order, and complete-call reuse |
| `audit.py`, `analyze.py` | Integrity checks and paired analysis |
| `common/`, `online/` | Timing, request replay, serving metrics, and reporting |
| `benchmark.py` | Basic measurement and metric utilities |
| `simulation/` | CPU scheduling simulations without model forward passes |
| `math500/` | Helpers used by online input encoding and related tests |
| `metadata/` | Exploration and exclusion indices for reconstructing input sets |

See the [reproduction guide](../docs/REPRODUCE.md) for commands and the [report index](../reports/README.md) for results.
