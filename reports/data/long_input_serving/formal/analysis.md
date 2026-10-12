# Remote experiment audit and paired results

Mode: stream; complete replays: 108; source variants: 1.

Selected block IDs: [0, 3, 4]; execution positions balanced: True.
All planned records were read. Rows are paired by configuration and block.
Each configuration has 3 repeats of one workload. Intervals concern repeat variability, not arbitrary workload generalization.
One block has no repeat-variance estimate: SD and confidence intervals are unavailable, not zero.
Small-sample t intervals are exploratory and unadjusted for multiple comparisons.
No outliers were removed. Cross-budget Pareto comparisons and independent-set inference require separate interpretation.

| Case | Baseline | Metric | Baseline mean | Mixed mean | Change % | Paired 95% interval | + / - |
|---|---|---|---:|---:|---:|---|---|
| E_rate0_seed20261012_T384 | separate | throughput_tok_s | 69.049 | 69.094 | +0.06 | [0.000, 0.089] | 3 / 0 |
| E_rate0_seed20261012_T384 | separate | all_ttft_ms | 386.895 | 1008.670 | +160.71 | [543.439, 700.111] | 3 / 0 |
| E_rate0_seed20261012_T384 | separate | all_e2e_ms | 17948.730 | 13774.726 | -23.26 | [-4735.910, -3612.099] | 0 / 3 |
| E_rate0_seed20261012_T384 | separate | request_max_itl_p95_ms | 3546.215 | 180.377 | -94.91 | [-3783.627, -2948.049] | 0 / 3 |
| E_rate0_seed20261012_T384 | separate | preemptions | 0.000 | 0.000 | n/a | [0.000, 0.000] | 0 / 0 |
| E_rate0_seed20261012_T384 | interleaved | throughput_tok_s | 68.989 | 69.094 | +0.15 | [0.072, 0.136] | 3 / 0 |
| E_rate0_seed20261012_T384 | interleaved | all_ttft_ms | 1462.204 | 1008.670 | -31.02 | [-497.017, -410.050] | 0 / 3 |
| E_rate0_seed20261012_T384 | interleaved | all_e2e_ms | 18374.849 | 13774.726 | -25.03 | [-5022.387, -4177.859] | 0 / 3 |
| E_rate0_seed20261012_T384 | interleaved | request_max_itl_p95_ms | 464.200 | 180.377 | -61.14 | [-1023.271, 455.625] | 0 / 3 |
| E_rate0_seed20261012_T384 | interleaved | preemptions | 0.000 | 0.000 | n/a | [0.000, 0.000] | 0 / 0 |
| E_rate0_seed20261012_T512 | separate | throughput_tok_s | 69.099 | 69.073 | -0.04 | [-0.253, 0.201] | 1 / 2 |
| E_rate0_seed20261012_T512 | separate | all_ttft_ms | 375.924 | 942.372 | +150.68 | [452.138, 680.759] | 3 / 0 |
| E_rate0_seed20261012_T512 | separate | all_e2e_ms | 17741.392 | 15122.073 | -14.76 | [-2738.298, -2500.341] | 0 / 3 |
| E_rate0_seed20261012_T512 | separate | request_max_itl_p95_ms | 3138.539 | 219.171 | -93.02 | [-4546.592, -1292.144] | 0 / 3 |
| E_rate0_seed20261012_T512 | separate | preemptions | 0.000 | 0.000 | n/a | [0.000, 0.000] | 0 / 0 |
| E_rate0_seed20261012_T512 | interleaved | throughput_tok_s | 69.038 | 69.073 | +0.05 | [-0.035, 0.105] | 3 / 0 |
| E_rate0_seed20261012_T512 | interleaved | all_ttft_ms | 1029.817 | 942.372 | -8.49 | [-125.331, -49.560] | 0 / 3 |
| E_rate0_seed20261012_T512 | interleaved | all_e2e_ms | 18340.775 | 15122.073 | -17.55 | [-3454.274, -2983.131] | 0 / 3 |
| E_rate0_seed20261012_T512 | interleaved | request_max_itl_p95_ms | 1893.817 | 219.171 | -88.43 | [-2213.157, -1136.135] | 0 / 3 |
| E_rate0_seed20261012_T512 | interleaved | preemptions | 0.000 | 0.000 | n/a | [0.000, 0.000] | 0 / 0 |
| E_rate0_seed20261013_T384 | separate | throughput_tok_s | 52.791 | 52.775 | -0.03 | [-0.050, 0.018] | 0 / 3 |
| E_rate0_seed20261013_T384 | separate | all_ttft_ms | 347.402 | 632.973 | +82.20 | [255.171, 315.970] | 3 / 0 |
| E_rate0_seed20261013_T384 | separate | all_e2e_ms | 12062.121 | 10137.395 | -15.96 | [-2324.531, -1524.923] | 0 / 3 |
| E_rate0_seed20261013_T384 | separate | request_max_itl_p95_ms | 722.623 | 170.893 | -76.35 | [-588.328, -515.132] | 0 / 3 |
| E_rate0_seed20261013_T384 | separate | preemptions | 0.000 | 0.000 | n/a | [0.000, 0.000] | 0 / 0 |
| E_rate0_seed20261013_T384 | interleaved | throughput_tok_s | 52.754 | 52.775 | +0.04 | [-0.026, 0.068] | 3 / 0 |
| E_rate0_seed20261013_T384 | interleaved | all_ttft_ms | 783.229 | 632.973 | -19.18 | [-171.783, -128.730] | 0 / 3 |
| E_rate0_seed20261013_T384 | interleaved | all_e2e_ms | 12126.695 | 10137.395 | -16.40 | [-2536.833, -1441.768] | 0 / 3 |
| E_rate0_seed20261013_T384 | interleaved | request_max_itl_p95_ms | 202.056 | 170.893 | -15.42 | [-42.305, -20.022] | 0 / 3 |
| E_rate0_seed20261013_T384 | interleaved | preemptions | 0.000 | 0.000 | n/a | [0.000, 0.000] | 0 / 0 |
| E_rate0_seed20261013_T512 | separate | throughput_tok_s | 52.776 | 52.784 | +0.02 | [-0.114, 0.131] | 2 / 1 |
| E_rate0_seed20261013_T512 | separate | all_ttft_ms | 338.194 | 572.139 | +69.17 | [203.515, 264.375] | 3 / 0 |
| E_rate0_seed20261013_T512 | separate | all_e2e_ms | 11830.848 | 10663.335 | -9.87 | [-1598.646, -736.380] | 0 / 3 |
| E_rate0_seed20261013_T512 | separate | request_max_itl_p95_ms | 696.862 | 199.801 | -71.33 | [-521.200, -472.923] | 0 / 3 |
| E_rate0_seed20261013_T512 | separate | preemptions | 0.000 | 0.000 | n/a | [0.000, 0.000] | 0 / 0 |
| E_rate0_seed20261013_T512 | interleaved | throughput_tok_s | 52.747 | 52.784 | +0.07 | [-0.035, 0.108] | 3 / 0 |
| E_rate0_seed20261013_T512 | interleaved | all_ttft_ms | 607.603 | 572.139 | -5.84 | [-54.699, -16.229] | 0 / 3 |
| E_rate0_seed20261013_T512 | interleaved | all_e2e_ms | 11779.564 | 10663.335 | -9.48 | [-1604.726, -627.733] | 0 / 3 |
| E_rate0_seed20261013_T512 | interleaved | request_max_itl_p95_ms | 239.546 | 199.801 | -16.59 | [-43.684, -35.806] | 0 / 3 |
| E_rate0_seed20261013_T512 | interleaved | preemptions | 0.000 | 0.000 | n/a | [0.000, 0.000] | 0 / 0 |
| E_rate1_seed20261012_T384 | separate | throughput_tok_s | 78.320 | 83.605 | +6.75 | [2.616, 7.954] | 3 / 0 |
| E_rate1_seed20261012_T384 | separate | all_ttft_ms | 3382.882 | 14658.224 | +333.31 | [8573.010, 13977.676] | 3 / 0 |
| E_rate1_seed20261012_T384 | separate | all_e2e_ms | 46411.498 | 36214.731 | -21.97 | [-14270.018, -6123.517] | 0 / 3 |
| E_rate1_seed20261012_T384 | separate | request_max_itl_p95_ms | 30926.978 | 246.113 | -99.20 | [-30902.094, -30459.635] | 0 / 3 |
| E_rate1_seed20261012_T384 | separate | preemptions | 1.333 | 0.000 | -100.00 | [-2.768, 0.101] | 0 / 3 |
| E_rate1_seed20261012_T384 | interleaved | throughput_tok_s | 78.197 | 83.605 | +6.92 | [3.152, 7.664] | 3 / 0 |
| E_rate1_seed20261012_T384 | interleaved | all_ttft_ms | 7814.713 | 14658.224 | +87.57 | [4395.258, 9291.764] | 3 / 0 |
| E_rate1_seed20261012_T384 | interleaved | all_e2e_ms | 45928.043 | 36214.731 | -21.15 | [-12404.936, -7021.688] | 0 / 3 |
| E_rate1_seed20261012_T384 | interleaved | request_max_itl_p95_ms | 26404.343 | 246.113 | -99.07 | [-26667.980, -25648.479] | 0 / 3 |
| E_rate1_seed20261012_T384 | interleaved | preemptions | 0.000 | 0.000 | n/a | [0.000, 0.000] | 0 / 0 |
| E_rate1_seed20261012_T512 | separate | throughput_tok_s | 78.772 | 82.568 | +4.82 | [2.988, 4.604] | 3 / 0 |
| E_rate1_seed20261012_T512 | separate | all_ttft_ms | 3032.948 | 15250.839 | +402.84 | [11660.903, 12774.878] | 3 / 0 |
| E_rate1_seed20261012_T512 | separate | all_e2e_ms | 45751.655 | 37255.492 | -18.57 | [-8899.242, -8093.084] | 0 / 3 |
| E_rate1_seed20261012_T512 | separate | request_max_itl_p95_ms | 30778.629 | 247.880 | -99.19 | [-31577.046, -29484.453] | 0 / 3 |
| E_rate1_seed20261012_T512 | separate | preemptions | 1.333 | 0.000 | -100.00 | [-2.768, 0.101] | 0 / 3 |
| E_rate1_seed20261012_T512 | interleaved | throughput_tok_s | 78.559 | 82.568 | +5.10 | [3.231, 4.787] | 3 / 0 |
| E_rate1_seed20261012_T512 | interleaved | all_ttft_ms | 4550.701 | 15250.839 | +235.13 | [10038.083, 11362.191] | 3 / 0 |
| E_rate1_seed20261012_T512 | interleaved | all_e2e_ms | 46033.286 | 37255.492 | -19.07 | [-10599.412, -6956.176] | 0 / 3 |
| E_rate1_seed20261012_T512 | interleaved | request_max_itl_p95_ms | 28473.164 | 247.880 | -99.13 | [-29026.553, -27424.016] | 0 / 3 |
| E_rate1_seed20261012_T512 | interleaved | preemptions | 0.667 | 0.000 | -100.00 | [-2.101, 0.768] | 0 / 2 |
| E_rate1_seed20261013_T384 | separate | throughput_tok_s | 74.989 | 76.350 | +1.82 | [0.713, 2.009] | 3 / 0 |
| E_rate1_seed20261013_T384 | separate | all_ttft_ms | 386.773 | 1902.607 | +391.92 | [1319.627, 1712.043] | 3 / 0 |
| E_rate1_seed20261013_T384 | separate | all_e2e_ms | 24136.368 | 19075.181 | -20.97 | [-5767.194, -4355.180] | 0 / 3 |
| E_rate1_seed20261013_T384 | separate | request_max_itl_p95_ms | 10798.722 | 229.772 | -97.87 | [-11833.152, -9304.747] | 0 / 3 |
| E_rate1_seed20261013_T384 | separate | preemptions | 0.000 | 0.000 | n/a | [0.000, 0.000] | 0 / 0 |
| E_rate1_seed20261013_T384 | interleaved | throughput_tok_s | 74.557 | 76.350 | +2.41 | [1.442, 2.146] | 3 / 0 |
| E_rate1_seed20261013_T384 | interleaved | all_ttft_ms | 1963.477 | 1902.607 | -3.10 | [-327.375, 205.637] | 1 / 2 |
| E_rate1_seed20261013_T384 | interleaved | all_e2e_ms | 24630.083 | 19075.181 | -22.55 | [-6315.150, -4794.655] | 0 / 3 |
| E_rate1_seed20261013_T384 | interleaved | request_max_itl_p95_ms | 6948.397 | 229.772 | -96.69 | [-7945.239, -5492.011] | 0 / 3 |
| E_rate1_seed20261013_T384 | interleaved | preemptions | 0.000 | 0.000 | n/a | [0.000, 0.000] | 0 / 0 |
| E_rate1_seed20261013_T512 | separate | throughput_tok_s | 75.271 | 76.052 | +1.04 | [0.421, 1.141] | 3 / 0 |
| E_rate1_seed20261013_T512 | separate | all_ttft_ms | 377.543 | 2105.649 | +457.73 | [1522.096, 1934.117] | 3 / 0 |
| E_rate1_seed20261013_T512 | separate | all_e2e_ms | 23475.215 | 20059.137 | -14.55 | [-3612.667, -3219.489] | 0 / 3 |
| E_rate1_seed20261013_T512 | separate | request_max_itl_p95_ms | 10512.340 | 248.465 | -97.64 | [-11073.031, -9454.719] | 0 / 3 |
| E_rate1_seed20261013_T512 | separate | preemptions | 0.000 | 0.000 | n/a | [0.000, 0.000] | 0 / 0 |
| E_rate1_seed20261013_T512 | interleaved | throughput_tok_s | 75.092 | 76.052 | +1.28 | [0.620, 1.301] | 3 / 0 |
| E_rate1_seed20261013_T512 | interleaved | all_ttft_ms | 1137.627 | 2105.649 | +85.09 | [744.090, 1191.955] | 3 / 0 |
| E_rate1_seed20261013_T512 | interleaved | all_e2e_ms | 23776.970 | 20059.137 | -15.64 | [-3927.712, -3507.955] | 0 / 3 |
| E_rate1_seed20261013_T512 | interleaved | request_max_itl_p95_ms | 8955.748 | 248.465 | -97.23 | [-9101.026, -8313.540] | 0 / 3 |
| E_rate1_seed20261013_T512 | interleaved | preemptions | 0.000 | 0.000 | n/a | [0.000, 0.000] | 0 / 0 |
| E_rate2_seed20261012_T384 | separate | throughput_tok_s | 78.260 | 84.607 | +8.11 | [5.585, 7.109] | 3 / 0 |
| E_rate2_seed20261012_T384 | separate | all_ttft_ms | 14057.876 | 26511.243 | +88.59 | [11625.412, 13281.321] | 3 / 0 |
| E_rate2_seed20261012_T384 | separate | all_e2e_ms | 59654.766 | 48025.023 | -19.50 | [-12463.026, -10796.459] | 0 / 3 |
| E_rate2_seed20261012_T384 | separate | request_max_itl_p95_ms | 32548.713 | 227.575 | -99.30 | [-33194.052, -31448.224] | 0 / 3 |
| E_rate2_seed20261012_T384 | separate | preemptions | 2.000 | 0.000 | -100.00 | [-2.000, -2.000] | 0 / 3 |
| E_rate2_seed20261012_T384 | interleaved | throughput_tok_s | 78.268 | 84.607 | +8.10 | [5.609, 7.070] | 3 / 0 |
| E_rate2_seed20261012_T384 | interleaved | all_ttft_ms | 19743.952 | 26511.243 | +34.28 | [5785.017, 7749.565] | 3 / 0 |
| E_rate2_seed20261012_T384 | interleaved | all_e2e_ms | 59106.974 | 48025.023 | -18.75 | [-11996.345, -10167.557] | 0 / 3 |
| E_rate2_seed20261012_T384 | interleaved | request_max_itl_p95_ms | 26332.953 | 227.575 | -99.14 | [-27022.577, -25188.178] | 0 / 3 |
| E_rate2_seed20261012_T384 | interleaved | preemptions | 0.000 | 0.000 | n/a | [0.000, 0.000] | 0 / 0 |
| E_rate2_seed20261012_T512 | separate | throughput_tok_s | 78.649 | 82.678 | +5.12 | [3.354, 4.704] | 3 / 0 |
| E_rate2_seed20261012_T512 | separate | all_ttft_ms | 13815.516 | 27929.706 | +102.16 | [13057.639, 15170.741] | 3 / 0 |
| E_rate2_seed20261012_T512 | separate | all_e2e_ms | 59147.393 | 50133.410 | -15.24 | [-10100.320, -7927.647] | 0 / 3 |
| E_rate2_seed20261012_T512 | separate | request_max_itl_p95_ms | 32189.781 | 270.368 | -99.16 | [-32404.342, -31434.484] | 0 / 3 |
| E_rate2_seed20261012_T512 | separate | preemptions | 2.000 | 0.000 | -100.00 | [-2.000, -2.000] | 0 / 3 |
| E_rate2_seed20261012_T512 | interleaved | throughput_tok_s | 78.735 | 82.678 | +5.01 | [3.176, 4.711] | 3 / 0 |
| E_rate2_seed20261012_T512 | interleaved | all_ttft_ms | 15369.045 | 27929.706 | +81.73 | [11635.082, 13486.241] | 3 / 0 |
| E_rate2_seed20261012_T512 | interleaved | all_e2e_ms | 59152.332 | 50133.410 | -15.25 | [-10193.110, -7844.735] | 0 / 3 |
| E_rate2_seed20261012_T512 | interleaved | request_max_itl_p95_ms | 30590.656 | 270.368 | -99.12 | [-30902.737, -29737.838] | 0 / 3 |
| E_rate2_seed20261012_T512 | interleaved | preemptions | 2.000 | 0.000 | -100.00 | [-2.000, -2.000] | 0 / 3 |
| E_rate2_seed20261013_T384 | separate | throughput_tok_s | 78.098 | 83.774 | +7.27 | [5.379, 5.973] | 3 / 0 |
| E_rate2_seed20261013_T384 | separate | all_ttft_ms | 2170.595 | 12061.413 | +455.67 | [9649.873, 10131.763] | 3 / 0 |
| E_rate2_seed20261013_T384 | separate | all_e2e_ms | 44186.826 | 32961.937 | -25.40 | [-12044.795, -10404.981] | 0 / 3 |
| E_rate2_seed20261013_T384 | separate | request_max_itl_p95_ms | 29040.046 | 230.994 | -99.20 | [-29466.349, -28151.756] | 0 / 3 |
| E_rate2_seed20261013_T384 | separate | preemptions | 2.000 | 0.000 | -100.00 | [-2.000, -2.000] | 0 / 3 |
| E_rate2_seed20261013_T384 | interleaved | throughput_tok_s | 77.791 | 83.774 | +7.69 | [4.719, 7.247] | 3 / 0 |
| E_rate2_seed20261013_T384 | interleaved | all_ttft_ms | 7766.957 | 12061.413 | +55.29 | [3323.064, 5265.848] | 3 / 0 |
| E_rate2_seed20261013_T384 | interleaved | all_e2e_ms | 43216.375 | 32961.937 | -23.73 | [-12307.918, -8200.957] | 0 / 3 |
| E_rate2_seed20261013_T384 | interleaved | request_max_itl_p95_ms | 23101.940 | 230.994 | -99.00 | [-23495.105, -22246.787] | 0 / 3 |
| E_rate2_seed20261013_T384 | interleaved | preemptions | 0.000 | 0.000 | n/a | [0.000, 0.000] | 0 / 0 |
| E_rate2_seed20261013_T512 | separate | throughput_tok_s | 78.738 | 82.216 | +4.42 | [1.357, 5.599] | 3 / 0 |
| E_rate2_seed20261013_T512 | separate | all_ttft_ms | 1799.858 | 13136.923 | +629.89 | [10408.951, 12265.179] | 3 / 0 |
| E_rate2_seed20261013_T512 | separate | all_e2e_ms | 42873.039 | 34732.422 | -18.99 | [-10123.147, -6158.088] | 0 / 3 |
| E_rate2_seed20261013_T512 | separate | request_max_itl_p95_ms | 28663.217 | 254.866 | -99.11 | [-29548.516, -27268.185] | 0 / 3 |
| E_rate2_seed20261013_T512 | separate | preemptions | 0.333 | 0.000 | -100.00 | [-1.768, 1.101] | 0 / 1 |
| E_rate2_seed20261013_T512 | interleaved | throughput_tok_s | 78.459 | 82.216 | +4.79 | [2.964, 4.550] | 3 / 0 |
| E_rate2_seed20261013_T512 | interleaved | all_ttft_ms | 3923.625 | 13136.923 | +234.82 | [8681.791, 9744.805] | 3 / 0 |
| E_rate2_seed20261013_T512 | interleaved | all_e2e_ms | 42612.267 | 34732.422 | -18.49 | [-8768.668, -6991.023] | 0 / 3 |
| E_rate2_seed20261013_T512 | interleaved | request_max_itl_p95_ms | 26185.169 | 254.866 | -99.03 | [-27194.922, -24665.686] | 0 / 3 |
| E_rate2_seed20261013_T512 | interleaved | preemptions | 0.000 | 0.000 | n/a | [0.000, 0.000] | 0 / 0 |
