# Scheduling Performance Evaluation

[中文](../docs/zh-CN/reports/README.md)

| Workload | Question | Report |
| --- | --- | --- |
| Request bursts | How does mixing affect existing decode requests and newly arriving long prefill requests? | [Burst scheduling](burst_requests/README.md) |
| Long-input serving | How do throughput, completion latency, generation stalls, and first-token waits change under continuous arrivals? | [Performance report](long_input_serving/README.md) · [Notebook](long_input_serving/mixed_scheduling_report.ipynb) |

Both workloads retain all formal configurations, including negative results. All values can be recomputed from published measurements. See [published data](data/README.md) for export rules and limitations.
