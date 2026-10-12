# Burst Scheduling: Controlled Request Bursts

[中文](../../docs/zh-CN/reports/burst_requests/README.md)

Three confirmation input cohorts, four token budgets, three policies, and three execution blocks yield 108 formal policy replays. Each cohort has four existing decode requests (64-token input prefixes, 256-token outputs) and four late prefill requests (complete long texts, 32-token outputs). Each value below is a three-block mean.

## All configurations relative to interleaved

Arrows indicate interleaved → mixed. Percentages use `(mixed / interleaved - 1) × 100%`. Positive throughput changes and negative latency changes are favorable. Old-request window maximum ITL is the largest token interval among existing requests within the specified window of each replay, averaged over three replays. It is not the p95 of all request latencies.

| Cohort / budget | Output throughput token/s | Late-request mean TTFT ms | Late-request mean E2E ms | Old-request window max ITL ms |
| --- | --- | --- | --- | --- |
| confirm0_T128 | 62.85 → 69.01 (+9.81%) | 4346.58 → 3373.44 (-22.39%) | 7300.30 → 5872.39 (-19.56%) | 124.50 → 99.05 (-20.44%) |
| confirm0_T256 | 67.44 → 69.87 (+3.61%) | 2277.41 → 1960.14 (-13.93%) | 5245.54 → 4670.43 (-10.96%) | 131.95 → 108.31 (-17.92%) |
| confirm0_T384 | 67.16 → 69.10 (+2.89%) | 1772.64 → 1489.18 (-15.99%) | 4902.46 → 4413.50 (-9.97%) | 153.24 → 131.52 (-14.18%) |
| confirm0_T512 | 68.36 → 69.55 (+1.75%) | 1534.43 → 1355.33 (-11.67%) | 4601.33 → 4306.28 (-6.41%) | 176.33 → 151.28 (-14.21%) |
| confirm1_T128 | 63.21 → 69.88 (+10.56%) | 4085.78 → 3191.36 (-21.89%) | 7028.30 → 5636.08 (-19.81%) | 133.78 → 88.38 (-33.94%) |
| confirm1_T256 | 66.61 → 68.81 (+3.30%) | 2187.48 → 1892.90 (-13.47%) | 5239.51 → 4656.28 (-11.13%) | 137.96 → 112.62 (-18.37%) |
| confirm1_T384 | 67.71 → 69.62 (+2.82%) | 1714.39 → 1457.74 (-14.97%) | 4787.93 → 4351.24 (-9.12%) | 154.71 → 126.35 (-18.33%) |
| confirm1_T512 | 68.09 → 69.34 (+1.84%) | 1424.39 → 1294.03 (-9.15%) | 4532.40 → 4264.49 (-5.91%) | 182.58 → 162.62 (-10.93%) |
| confirm2_T128 | 64.18 → 66.91 (+4.26%) | 3899.99 → 3400.46 (-12.81%) | 6764.11 → 6044.14 (-10.64%) | 112.99 → 101.12 (-10.51%) |
| confirm2_T256 | 67.60 → 69.88 (+3.38%) | 2070.19 → 1840.00 (-11.12%) | 5049.08 → 4525.30 (-10.37%) | 133.69 → 105.73 (-20.91%) |
| confirm2_T384 | 67.96 → 70.15 (+3.22%) | 1709.55 → 1432.97 (-16.18%) | 4750.05 → 4284.47 (-9.80%) | 147.60 → 127.70 (-13.48%) |
| confirm2_T512 | 68.07 → 69.41 (+1.98%) | 1328.56 → 1224.39 (-7.84%) | 4410.76 → 4158.47 (-5.72%) | 176.74 → 146.24 (-17.26%) |

## Interpretation

At T384, all three cohorts improve late-request TTFT, completion time, and old-request window maximum ITL relative to interleaved. This controlled workload shows that mixed can advance both request types together; it does not imply improved TTFT in every scenario or against every baseline.

See the [full paired analysis](../data/burst_requests/formal/analysis.md) for benefits and costs relative to separate; late-request TTFT is longer against that baseline. The serving experiment further demonstrates the tradeoff between improved completion time and increased first-token waits under continuous arrivals. Benefits across the two workloads must not be combined into a claim that all latency metrics improve.

## Records and limitations

- [All formal measurements](../data/burst_requests/formal/) are indexed by block0, block3, block4 and confirm0/1/2_T*. See the [data export documentation](../data/README.md).
- [reuse.json](../data/burst_requests/formal/reuse.json) records 12 complete calls reused from an earlier interrupted batch; each is counted once.
- No formal call has preemption, but output hashes differ across policies. Benefits have not been established under an equivalent-answer guarantee.
- Each configuration has three execution repeats. Paired confidence intervals describe execution variability, are unadjusted for multiple comparisons, and do not guarantee generalization to arbitrary workloads.
- Regenerate this page with `python tools/render_burst_report.py`; verify all data with `python tools/verify_results.py`.
