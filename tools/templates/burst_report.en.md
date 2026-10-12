# Burst Scheduling: Controlled Request Bursts

[中文](../../docs/zh-CN/reports/burst_requests/README.md)

Three confirmation input cohorts, four token budgets, three policies, and three execution blocks yield 108 formal policy replays. Each cohort has four existing decode requests (64-token input prefixes, 256-token outputs) and four late prefill requests (complete long texts, 32-token outputs). Each value below is a three-block mean.

## All configurations relative to interleaved

Arrows indicate interleaved → mixed. Percentages use `(mixed / interleaved - 1) × 100%`. Positive throughput changes and negative latency changes are favorable. Old-request window maximum ITL is the largest token interval among existing requests within the specified window of each replay, averaged over three replays. It is not the p95 of all request latencies.

| Cohort / budget | Output throughput token/s | Late-request mean TTFT ms | Late-request mean E2E ms | Old-request window max ITL ms |
| --- | --- | --- | --- | --- |
{rows}

## Interpretation

At T384, all three cohorts improve late-request TTFT, completion time, and old-request window maximum ITL relative to interleaved. This controlled workload shows that mixed can advance both request types together; it does not imply improved TTFT in every scenario or against every baseline.

See the [full paired analysis](../data/burst_requests/formal/analysis.md) for benefits and costs relative to separate; late-request TTFT is longer against that baseline. The serving experiment further demonstrates the tradeoff between improved completion time and increased first-token waits under continuous arrivals. Benefits across the two workloads must not be combined into a claim that all latency metrics improve.

## Records and limitations

- [All formal measurements](../data/burst_requests/formal/) are indexed by block0, block3, block4 and confirm0/1/2_T*. See the [data export documentation](../data/README.md).
- [reuse.json](../data/burst_requests/formal/reuse.json) records 12 complete calls reused from an earlier interrupted batch; each is counted once.
- No formal call has preemption, but output hashes differ across policies. Benefits have not been established under an equivalent-answer guarantee.
- Each configuration has three execution repeats. Paired confidence intervals describe execution variability, are unadjusted for multiple comparisons, and do not guarantee generalization to arbitrary workloads.
- Regenerate this page with `python tools/render_burst_report.py`; verify all data with `python tools/verify_results.py`.
