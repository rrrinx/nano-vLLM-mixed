# Remote experiment audit and paired results

Mode: burst; complete replays: 108; source variants: 1.

Reused completed calls: 12; provenance in reuse.json. Old/new run periods may differ.

Selected block IDs: [0, 3, 4]; execution positions balanced: True.
All planned records were read. Rows are paired by configuration and block.
Each configuration has 3 repeats of one workload. Intervals concern repeat variability, not arbitrary workload generalization.
One block has no repeat-variance estimate: SD and confidence intervals are unavailable, not zero.
Small-sample t intervals are exploratory and unadjusted for multiple comparisons.
No outliers were removed. Cross-budget Pareto comparisons and independent-set inference require separate interpretation.

| Case | Baseline | Metric | Baseline mean | Mixed mean | Change % | Paired 95% interval | + / - |
|---|---|---|---:|---:|---:|---|---|
| confirm0_T128 | separate | throughput_tok_s | 63.233 | 69.012 | +9.14 | [3.647, 7.913] | 3 / 0 |
| confirm0_T128 | separate | old_window_max_itl_ms | 2405.170 | 99.053 | -95.88 | [-2463.908, -2148.324] | 0 / 3 |
| confirm0_T128 | separate | old_full_max_itl_ms | 2405.170 | 99.053 | -95.88 | [-2463.908, -2148.324] | 0 / 3 |
| confirm0_T128 | separate | late_ttft_ms | 1485.292 | 3373.439 | +127.12 | [1629.360, 2146.933] | 3 / 0 |
| confirm0_T128 | separate | late_e2e_ms | 5245.320 | 5872.392 | +11.95 | [290.546, 963.596] | 3 / 0 |
| confirm0_T128 | interleaved | throughput_tok_s | 62.847 | 69.012 | +9.81 | [2.193, 10.139] | 3 / 0 |
| confirm0_T128 | interleaved | old_window_max_itl_ms | 124.501 | 99.053 | -20.44 | [-90.999, 40.104] | 0 / 3 |
| confirm0_T128 | interleaved | old_full_max_itl_ms | 140.240 | 99.053 | -29.37 | [-89.597, 7.224] | 0 / 3 |
| confirm0_T128 | interleaved | late_ttft_ms | 4346.575 | 3373.439 | -22.39 | [-1436.931, -509.342] | 0 / 3 |
| confirm0_T128 | interleaved | late_e2e_ms | 7300.299 | 5872.392 | -19.56 | [-1981.246, -874.568] | 0 / 3 |
| confirm0_T256 | separate | throughput_tok_s | 67.233 | 69.871 | +3.92 | [-0.777, 6.053] | 3 / 0 |
| confirm0_T256 | separate | old_window_max_itl_ms | 1417.434 | 108.305 | -92.36 | [-1357.559, -1260.698] | 0 / 3 |
| confirm0_T256 | separate | old_full_max_itl_ms | 1417.434 | 108.305 | -92.36 | [-1357.559, -1260.698] | 0 / 3 |
| confirm0_T256 | separate | late_ttft_ms | 880.469 | 1960.142 | +122.62 | [721.675, 1437.669] | 3 / 0 |
| confirm0_T256 | separate | late_e2e_ms | 4229.478 | 4670.428 | +10.43 | [-30.056, 911.955] | 3 / 0 |
| confirm0_T256 | interleaved | throughput_tok_s | 67.437 | 69.871 | +3.61 | [-0.127, 4.994] | 3 / 0 |
| confirm0_T256 | interleaved | old_window_max_itl_ms | 131.949 | 108.305 | -17.92 | [-26.485, -20.803] | 0 / 3 |
| confirm0_T256 | interleaved | old_full_max_itl_ms | 131.949 | 108.305 | -17.92 | [-26.485, -20.803] | 0 / 3 |
| confirm0_T256 | interleaved | late_ttft_ms | 2277.410 | 1960.142 | -13.93 | [-782.839, 148.302] | 0 / 3 |
| confirm0_T256 | interleaved | late_e2e_ms | 5245.536 | 4670.428 | -10.96 | [-1155.160, 4.943] | 0 / 3 |
| confirm0_T384 | separate | throughput_tok_s | 66.872 | 69.099 | +3.33 | [-0.296, 4.750] | 3 / 0 |
| confirm0_T384 | separate | old_window_max_itl_ms | 1312.954 | 131.517 | -89.98 | [-1225.681, -1137.194] | 0 / 3 |
| confirm0_T384 | separate | old_full_max_itl_ms | 1312.954 | 131.517 | -89.98 | [-1225.681, -1137.194] | 0 / 3 |
| confirm0_T384 | separate | late_ttft_ms | 811.990 | 1489.177 | +83.40 | [670.100, 684.274] | 3 / 0 |
| confirm0_T384 | separate | late_e2e_ms | 4188.407 | 4413.504 | +5.37 | [-6.966, 457.159] | 3 / 0 |
| confirm0_T384 | interleaved | throughput_tok_s | 67.158 | 69.099 | +2.89 | [0.655, 3.226] | 3 / 0 |
| confirm0_T384 | interleaved | old_window_max_itl_ms | 153.244 | 131.517 | -14.18 | [-34.848, -8.606] | 0 / 3 |
| confirm0_T384 | interleaved | old_full_max_itl_ms | 153.244 | 131.517 | -14.18 | [-34.848, -8.606] | 0 / 3 |
| confirm0_T384 | interleaved | late_ttft_ms | 1772.645 | 1489.177 | -15.99 | [-497.917, -69.019] | 0 / 3 |
| confirm0_T384 | interleaved | late_e2e_ms | 4902.465 | 4413.504 | -9.97 | [-722.936, -254.987] | 0 / 3 |
| confirm0_T512 | separate | throughput_tok_s | 67.201 | 69.552 | +3.50 | [-0.790, 5.492] | 3 / 0 |
| confirm0_T512 | separate | old_window_max_itl_ms | 1293.285 | 151.278 | -88.30 | [-1168.295, -1115.720] | 0 / 3 |
| confirm0_T512 | separate | old_full_max_itl_ms | 1293.285 | 151.278 | -88.30 | [-1168.295, -1115.720] | 0 / 3 |
| confirm0_T512 | separate | late_ttft_ms | 799.022 | 1355.332 | +69.62 | [466.742, 645.878] | 3 / 0 |
| confirm0_T512 | separate | late_e2e_ms | 4192.710 | 4306.277 | +2.71 | [-417.601, 644.734] | 2 / 1 |
| confirm0_T512 | interleaved | throughput_tok_s | 68.359 | 69.552 | +1.75 | [0.010, 2.376] | 3 / 0 |
| confirm0_T512 | interleaved | old_window_max_itl_ms | 176.330 | 151.278 | -14.21 | [-32.730, -17.375] | 0 / 3 |
| confirm0_T512 | interleaved | old_full_max_itl_ms | 176.330 | 151.278 | -14.21 | [-32.730, -17.375] | 0 / 3 |
| confirm0_T512 | interleaved | late_ttft_ms | 1534.427 | 1355.332 | -11.67 | [-251.257, -106.931] | 0 / 3 |
| confirm0_T512 | interleaved | late_e2e_ms | 4601.333 | 4306.277 | -6.41 | [-445.484, -144.629] | 0 / 3 |
| confirm1_T128 | separate | throughput_tok_s | 63.469 | 69.884 | +10.11 | [4.043, 8.786] | 3 / 0 |
| confirm1_T128 | separate | old_window_max_itl_ms | 2304.464 | 88.376 | -96.17 | [-2276.428, -2155.747] | 0 / 3 |
| confirm1_T128 | separate | old_full_max_itl_ms | 2304.464 | 88.376 | -96.17 | [-2276.428, -2155.747] | 0 / 3 |
| confirm1_T128 | separate | late_ttft_ms | 1430.691 | 3191.364 | +123.06 | [1529.184, 1992.162] | 3 / 0 |
| confirm1_T128 | separate | late_e2e_ms | 5079.740 | 5636.079 | +10.95 | [380.993, 731.686] | 3 / 0 |
| confirm1_T128 | interleaved | throughput_tok_s | 63.211 | 69.884 | +10.56 | [0.363, 12.983] | 3 / 0 |
| confirm1_T128 | interleaved | old_window_max_itl_ms | 133.777 | 88.376 | -33.94 | [-113.577, 22.776] | 0 / 3 |
| confirm1_T128 | interleaved | old_full_max_itl_ms | 133.777 | 88.376 | -33.94 | [-113.577, 22.776] | 0 / 3 |
| confirm1_T128 | interleaved | late_ttft_ms | 4085.779 | 3191.364 | -21.89 | [-1419.786, -369.043] | 0 / 3 |
| confirm1_T128 | interleaved | late_e2e_ms | 7028.300 | 5636.079 | -19.81 | [-2289.172, -495.270] | 0 / 3 |
| confirm1_T256 | separate | throughput_tok_s | 65.967 | 68.809 | +4.31 | [-1.754, 7.439] | 3 / 0 |
| confirm1_T256 | separate | old_window_max_itl_ms | 1388.996 | 112.620 | -91.89 | [-1432.950, -1119.803] | 0 / 3 |
| confirm1_T256 | separate | old_full_max_itl_ms | 1388.996 | 112.620 | -91.89 | [-1432.950, -1119.803] | 0 / 3 |
| confirm1_T256 | separate | late_ttft_ms | 845.969 | 1892.896 | +123.75 | [661.396, 1432.459] | 3 / 0 |
| confirm1_T256 | separate | late_e2e_ms | 4318.421 | 4656.281 | +7.82 | [208.286, 467.433] | 3 / 0 |
| confirm1_T256 | interleaved | throughput_tok_s | 66.614 | 68.809 | +3.30 | [-1.237, 5.627] | 3 / 0 |
| confirm1_T256 | interleaved | old_window_max_itl_ms | 137.965 | 112.620 | -18.37 | [-42.910, -7.780] | 0 / 3 |
| confirm1_T256 | interleaved | old_full_max_itl_ms | 137.965 | 112.620 | -18.37 | [-42.910, -7.780] | 0 / 3 |
| confirm1_T256 | interleaved | late_ttft_ms | 2187.481 | 1892.896 | -13.47 | [-569.448, -19.722] | 0 / 3 |
| confirm1_T256 | interleaved | late_e2e_ms | 5239.509 | 4656.281 | -11.13 | [-1011.045, -155.411] | 0 / 3 |
| confirm1_T384 | separate | throughput_tok_s | 67.890 | 69.618 | +2.55 | [-0.616, 4.072] | 3 / 0 |
| confirm1_T384 | separate | old_window_max_itl_ms | 1264.050 | 126.348 | -90.00 | [-1165.599, -1109.804] | 0 / 3 |
| confirm1_T384 | separate | old_full_max_itl_ms | 1264.050 | 126.348 | -90.00 | [-1165.599, -1109.804] | 0 / 3 |
| confirm1_T384 | separate | late_ttft_ms | 755.164 | 1457.736 | +93.04 | [670.718, 734.426] | 3 / 0 |
| confirm1_T384 | separate | late_e2e_ms | 4049.188 | 4351.243 | +7.46 | [131.320, 472.789] | 3 / 0 |
| confirm1_T384 | interleaved | throughput_tok_s | 67.707 | 69.618 | +2.82 | [-1.067, 4.889] | 3 / 0 |
| confirm1_T384 | interleaved | old_window_max_itl_ms | 154.714 | 126.348 | -18.33 | [-51.183, -5.548] | 0 / 3 |
| confirm1_T384 | interleaved | old_full_max_itl_ms | 154.714 | 126.348 | -18.33 | [-51.183, -5.548] | 0 / 3 |
| confirm1_T384 | interleaved | late_ttft_ms | 1714.386 | 1457.736 | -14.97 | [-316.865, -196.435] | 0 / 3 |
| confirm1_T384 | interleaved | late_e2e_ms | 4787.932 | 4351.243 | -9.12 | [-538.362, -335.017] | 0 / 3 |
| confirm1_T512 | separate | throughput_tok_s | 68.845 | 69.341 | +0.72 | [-1.903, 2.896] | 1 / 2 |
| confirm1_T512 | separate | old_window_max_itl_ms | 1216.774 | 162.616 | -86.64 | [-1086.949, -1021.367] | 0 / 3 |
| confirm1_T512 | separate | old_full_max_itl_ms | 1216.774 | 162.616 | -86.64 | [-1086.949, -1021.367] | 0 / 3 |
| confirm1_T512 | separate | late_ttft_ms | 749.972 | 1294.026 | +72.54 | [483.062, 605.047] | 3 / 0 |
| confirm1_T512 | separate | late_e2e_ms | 3985.968 | 4264.487 | +6.99 | [148.372, 408.667] | 3 / 0 |
| confirm1_T512 | interleaved | throughput_tok_s | 68.088 | 69.341 | +1.84 | [-0.629, 3.137] | 3 / 0 |
| confirm1_T512 | interleaved | old_window_max_itl_ms | 182.575 | 162.616 | -10.93 | [-64.012, 24.094] | 1 / 2 |
| confirm1_T512 | interleaved | old_full_max_itl_ms | 182.575 | 162.616 | -10.93 | [-64.012, 24.094] | 1 / 2 |
| confirm1_T512 | interleaved | late_ttft_ms | 1424.386 | 1294.026 | -9.15 | [-223.983, -36.738] | 0 / 3 |
| confirm1_T512 | interleaved | late_e2e_ms | 4532.403 | 4264.487 | -5.91 | [-479.557, -56.276] | 0 / 3 |
| confirm2_T128 | separate | throughput_tok_s | 63.690 | 66.908 | +5.05 | [-11.072, 17.509] | 2 / 1 |
| confirm2_T128 | separate | old_window_max_itl_ms | 2420.418 | 101.120 | -95.82 | [-3059.195, -1579.403] | 0 / 3 |
| confirm2_T128 | separate | old_full_max_itl_ms | 2420.418 | 101.888 | -95.79 | [-3059.874, -1577.187] | 0 / 3 |
| confirm2_T128 | separate | late_ttft_ms | 1492.365 | 3400.462 | +127.86 | [871.056, 2945.138] | 3 / 0 |
| confirm2_T128 | separate | late_e2e_ms | 5295.445 | 6044.143 | +14.14 | [-1362.357, 2859.753] | 2 / 1 |
| confirm2_T128 | interleaved | throughput_tok_s | 64.176 | 66.908 | +4.26 | [-10.056, 15.520] | 2 / 1 |
| confirm2_T128 | interleaved | old_window_max_itl_ms | 112.991 | 101.120 | -10.51 | [-65.355, 41.612] | 1 / 2 |
| confirm2_T128 | interleaved | old_full_max_itl_ms | 112.991 | 101.888 | -9.83 | [-63.634, 41.427] | 1 / 2 |
| confirm2_T128 | interleaved | late_ttft_ms | 3899.987 | 3400.462 | -12.81 | [-1272.359, 273.310] | 0 / 3 |
| confirm2_T128 | interleaved | late_e2e_ms | 6764.110 | 6044.143 | -10.64 | [-2381.555, 941.622] | 1 / 2 |
| confirm2_T256 | separate | throughput_tok_s | 67.734 | 69.883 | +3.17 | [0.128, 4.170] | 3 / 0 |
| confirm2_T256 | separate | old_window_max_itl_ms | 1300.927 | 105.732 | -91.87 | [-1229.620, -1160.769] | 0 / 3 |
| confirm2_T256 | separate | old_full_max_itl_ms | 1300.927 | 105.732 | -91.87 | [-1229.620, -1160.769] | 0 / 3 |
| confirm2_T256 | separate | late_ttft_ms | 796.852 | 1840.005 | +130.91 | [1025.178, 1061.128] | 3 / 0 |
| confirm2_T256 | separate | late_e2e_ms | 4072.350 | 4525.301 | +11.12 | [416.427, 489.475] | 3 / 0 |
| confirm2_T256 | interleaved | throughput_tok_s | 67.596 | 69.883 | +3.38 | [-1.017, 5.591] | 3 / 0 |
| confirm2_T256 | interleaved | old_window_max_itl_ms | 133.686 | 105.732 | -20.91 | [-32.862, -23.046] | 0 / 3 |
| confirm2_T256 | interleaved | old_full_max_itl_ms | 133.686 | 105.732 | -20.91 | [-32.862, -23.046] | 0 / 3 |
| confirm2_T256 | interleaved | late_ttft_ms | 2070.188 | 1840.005 | -11.12 | [-313.926, -146.440] | 0 / 3 |
| confirm2_T256 | interleaved | late_e2e_ms | 5049.079 | 4525.301 | -10.37 | [-694.174, -353.383] | 0 / 3 |
| confirm2_T384 | separate | throughput_tok_s | 67.874 | 70.151 | +3.35 | [0.684, 3.870] | 3 / 0 |
| confirm2_T384 | separate | old_window_max_itl_ms | 1260.952 | 127.697 | -89.87 | [-1141.220, -1125.291] | 0 / 3 |
| confirm2_T384 | separate | old_full_max_itl_ms | 1260.952 | 127.697 | -89.87 | [-1141.220, -1125.291] | 0 / 3 |
| confirm2_T384 | separate | late_ttft_ms | 765.548 | 1432.975 | +87.18 | [639.060, 695.795] | 3 / 0 |
| confirm2_T384 | separate | late_e2e_ms | 4054.881 | 4284.472 | +5.66 | [146.052, 313.128] | 3 / 0 |
| confirm2_T384 | interleaved | throughput_tok_s | 67.960 | 70.151 | +3.22 | [1.614, 2.768] | 3 / 0 |
| confirm2_T384 | interleaved | old_window_max_itl_ms | 147.599 | 127.697 | -13.48 | [-23.039, -16.765] | 0 / 3 |
| confirm2_T384 | interleaved | old_full_max_itl_ms | 147.599 | 127.697 | -13.48 | [-23.039, -16.765] | 0 / 3 |
| confirm2_T384 | interleaved | late_ttft_ms | 1709.550 | 1432.975 | -16.18 | [-318.821, -234.328] | 0 / 3 |
| confirm2_T384 | interleaved | late_e2e_ms | 4750.046 | 4284.472 | -9.80 | [-520.547, -410.602] | 0 / 3 |
| confirm2_T512 | separate | throughput_tok_s | 68.121 | 69.412 | +1.89 | [1.041, 1.540] | 3 / 0 |
| confirm2_T512 | separate | old_window_max_itl_ms | 1178.119 | 146.237 | -87.59 | [-1047.036, -1016.727] | 0 / 3 |
| confirm2_T512 | separate | old_full_max_itl_ms | 1178.119 | 146.237 | -87.59 | [-1047.036, -1016.727] | 0 / 3 |
| confirm2_T512 | separate | late_ttft_ms | 728.587 | 1224.386 | +68.05 | [461.459, 530.138] | 3 / 0 |
| confirm2_T512 | separate | late_e2e_ms | 3996.694 | 4158.475 | +4.05 | [101.748, 221.814] | 3 / 0 |
| confirm2_T512 | interleaved | throughput_tok_s | 68.067 | 69.412 | +1.98 | [0.762, 1.928] | 3 / 0 |
| confirm2_T512 | interleaved | old_window_max_itl_ms | 176.743 | 146.237 | -17.26 | [-33.211, -27.801] | 0 / 3 |
| confirm2_T512 | interleaved | old_full_max_itl_ms | 176.743 | 146.237 | -17.26 | [-33.211, -27.801] | 0 / 3 |
| confirm2_T512 | interleaved | late_ttft_ms | 1328.558 | 1224.386 | -7.84 | [-146.682, -61.661] | 0 / 3 |
| confirm2_T512 | interleaved | late_e2e_ms | 4410.763 | 4158.475 | -5.72 | [-299.765, -204.812] | 0 / 3 |
