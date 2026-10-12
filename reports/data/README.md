# Published Measurements

[中文](../../docs/zh-CN/reports/data/README.md)

`burst_requests/formal/` and `long_input_serving/formal/` each retain 36 `results.json` files, with three policies per file and 108 formal replays per workload. Schedule, block, reuse, and original analysis files identify configurations and repeats. The burst experiment reused 12 complete calls from an earlier interrupted batch; these are already included in the 36 files and are not counted twice.

## Differences from the original local records

1. All `runs` are retained: summary metrics, request timestamps, output lengths, output hashes, preemptions, and mixed-step fractions. No measured values are changed.
2. `trace.requests[].prompt_token_ids` is replaced with `input_tokens` and `prompt_token_ids_sha256`. SHA-256 is computed over compact JSON of the original token list (`separators=(',', ':')`, UTF-8). Decodable input content is not distributed.
3. The duplicate top-level `summary` is removed; every `runs[].summary` remains complete.
4. A `publication_export` marker is added. Public JSON files are analysis copies, not executable input traces or byte-for-byte copies of the originals.
5. `PROVENANCE.json` records original relative paths and SHA-256 hashes, published paths and hashes, and file-level mappings between measured and published code. Inference-engine files retain their original contents. Experiment tools have adjusted imports and path resolution for the unified layout; historical and current fingerprints are recorded separately without claiming byte-for-byte identity for modified files.

Directories map one-to-one to the original results: bursts come from `20261011-231131-087401-confirm/formal`; serving comes from `20261012-015122-127636-online/formal`. Historical `/root/...` paths provide provenance only; report regeneration does not access them.

`source_sha256` contains the source fingerprints recorded at measurement time. `checkpoint_files` mainly records shard sizes and modification times plus the index-file SHA-256; it is not a complete content hash of all model weights.

## Verification

```bash
python tools/verify_results.py
```

This command checks file fingerprints, all configurations and blocks, policy order, request completeness, latency identities, and summary means. It also prints the two configurations used in the CV. No model is loaded.

## Limitations

All formal files have cross-policy output-hash mismatches; serving repeats also differ within the same policy. The cause is undetermined. Full generated token sequences were not retained, so these records cannot establish equivalent answer quality.

Each configuration has only three execution blocks; individual requests are not independent repeats. Paired intervals describe repeat variability and are not adjusted for multiple comparisons. Both serving arrival seeds use the same 100 texts. Results are limited to one model, one GPU, synthetic arrivals, and fixed output lengths.
