"""Dependency-free, complete-schedule audit and within-config paired summaries."""
import argparse
import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

from audit import audit, read
from block_design import ORDERS, T95, position_balanced

FIELDS = {
    "throughput_tok_s": "summary.output_tok_s",
    "elapsed_s": "summary.elapsed_s",
    "old_window_max_itl_ms": "online.window_old_itl_ms.max",
    "old_full_max_itl_ms": "online.groups.old_decode.max_itl_ms.max",
    "late_ttft_ms": "online.groups.late_prefill.ttft_ms.mean",
    "late_e2e_ms": "online.groups.late_prefill.e2e_ms.mean",
    "all_ttft_ms": "online.groups.all.ttft_ms.mean",
    "all_e2e_ms": "online.groups.all.e2e_ms.mean",
    "request_max_itl_p95_ms": "online.groups.all.max_itl_ms.p95",
    "request_max_itl_max_ms": "online.groups.all.max_itl_ms.max",
    "preemptions": "summary.counts.preemptions",
}


def get(value, path):
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root, mode="burst"):
    root = Path(root).resolve()
    manifest = read(root / "schedule.json")
    schedule = manifest["entries"]
    block_ids = manifest.get("block_ids", sorted({e['block'] for e in schedule}))
    if (not block_ids or len(set(block_ids)) != len(block_ids)
            or any(b not in range(6) for b in block_ids)
            or manifest["blocks"] != len(block_ids)):
        raise ValueError("Invalid selected block manifest")
    expected_blocks = set(block_ids)
    if read(root / "blocks.json")["entries"] != schedule:
        raise ValueError("Not all planned calls completed; do not analyze partial matrices")
    scheduled = {(Path(e['config']).stem, e['block']) for e in schedule}
    cases = {case for case, _ in scheduled}
    if len(scheduled) != len(schedule) or scheduled != {(c, b) for c in cases for b in block_ids}:
        raise ValueError("Schedule must contain each case exactly once per selected block")
    rows, identities, seen = [], set(), set()
    for entry in schedule:
        if entry['policy_order'] != list(ORDERS[entry['block']]):
            raise ValueError("Scheduled policy order differs from stable block ID")
        case = Path(entry["config"]).stem
        # Resolve relative to the copied result directory, so results can be moved home.
        path = root / f"block{entry['block']}" / case / "results.json"
        data = read(path)
        audit(data, mode=mode)
        identities.add(json.dumps(data["source_sha256"], sort_keys=True))
        if [r["policy"] for r in data["runs"]] != entry["policy_order"]:
            raise ValueError(f"Policy order differs: {path}")
        if any(r.get('policy_position') != i for i, r in enumerate(data['runs'], 1)):
            raise ValueError(f"Policy position differs: {path}")
        if data["settings"]["token_budget"] != entry["token_budget"]:
            raise ValueError(f"Budget differs: {path}")
        for run in data["runs"]:
            key = (case, entry["block"], run["policy"])
            if key in seen:
                raise ValueError(f"Duplicate replay: {key}")
            seen.add(key)
            rows.append({"case": case, "block": entry["block"], "policy": run["policy"],
                         "budget": entry["token_budget"], "max_seqs": data["settings"]["max_seqs"],
                         "workload_sha256": data["trace"]["workload_sha256"],
                         "source_record": str(path.relative_to(root)),
                         **{k: get(run, field) for k, field in FIELDS.items()}})
    if len(identities) != 1:
        raise ValueError("Multiple source fingerprints; do not silently pool")
    cells = defaultdict(list)
    for row in rows:
        cells[(row["case"], row["policy"])].append(row)
    for case in {r["case"] for r in rows}:
        selected = [r for r in rows if r["case"] == case]
        if len({r["workload_sha256"] for r in selected}) != 1:
            raise ValueError(f"Workload drift: {case}")
    summary = []
    for (case, policy), values in sorted(cells.items()):
        for metric in FIELDS:
            samples = [r[metric] for r in values if r[metric] is not None]
            if samples:
                summary.append({"case": case, "policy": policy, "metric": metric,
                                "n": len(samples), "mean": st.fmean(samples),
                                "median": st.median(samples), "sd": st.stdev(samples) if len(samples) > 1 else None,
                                "min": min(samples), "max": max(samples)})
    compared = []
    for case in sorted({r["case"] for r in rows}):
        mixed = {r["block"]: r for r in cells[(case, "mixed")]}
        for baseline in ("separate", "interleaved"):
            base = {r["block"]: r for r in cells[(case, baseline)]}
            if set(mixed) != expected_blocks or set(base) != expected_blocks:
                raise ValueError(f"Expected all selected paired blocks {block_ids}: {case}")
            for metric in FIELDS:
                if any(mixed[b][metric] is None or base[b][metric] is None for b in base):
                    continue
                differences = [mixed[b][metric] - base[b][metric] for b in sorted(base)]
                mean_a = st.fmean(base[b][metric] for b in base)
                mean_b = st.fmean(mixed[b][metric] for b in base)
                delta = st.fmean(differences)
                n = len(differences)
                half = T95[n - 1] * st.stdev(differences) / math.sqrt(n) if n > 1 else None
                compared.append({"case": case, "baseline": baseline, "metric": metric,
                                 "n": n, "block_ids": ','.join(map(str, block_ids)),
                                 "baseline_mean": mean_a, "mixed_mean": mean_b,
                                 "change_pct": (mean_b / mean_a - 1) * 100 if mean_a else None,
                                 "paired_delta": delta, "ci95_low": delta - half if half is not None else None,
                                 "ci95_high": delta + half if half is not None else None,
                                 "positive": sum(x > 0 for x in differences),
                                 "negative": sum(x < 0 for x in differences)})
    write_csv(root / "runs_analysis.csv", rows)
    write_csv(root / "summary_analysis.csv", summary)
    write_csv(root / "comparisons.csv", compared)
    lines = ["# Remote experiment audit and paired results", "",
             f"Mode: {mode}; complete replays: {len(rows)}; source variants: {len(identities)}.", "",
             f"Selected block IDs: {block_ids}; execution positions balanced: {position_balanced(block_ids)}.",
             "All planned records were read. Rows are paired by configuration and block.",
             f"Each configuration has {len(block_ids)} repeats of one workload. Intervals concern repeat variability, not arbitrary workload generalization.",
             "One block has no repeat-variance estimate: SD and confidence intervals are unavailable, not zero.",
             "Small-sample t intervals are exploratory and unadjusted for multiple comparisons.",
             "No outliers were removed. Cross-budget Pareto comparisons and independent-set inference require separate interpretation.", "",
             "| Case | Baseline | Metric | Baseline mean | Mixed mean | Change % | Paired 95% interval | + / - |",
             "|---|---|---|---:|---:|---:|---|---|"]
    wanted = {"throughput_tok_s", "old_window_max_itl_ms", "old_full_max_itl_ms", "late_ttft_ms", "late_e2e_ms"}
    if mode == "stream":
        wanted = {"throughput_tok_s", "all_ttft_ms", "all_e2e_ms", "request_max_itl_p95_ms", "preemptions"}
    reuse_path = root / 'reuse.json'
    if reuse_path.exists():
        reused = read(reuse_path)['entries']
        lines[4:4] = [f"Reused completed calls: {len(reused)}; provenance in reuse.json. Old/new run periods may differ.", ""]
    for r in compared:
        if r["metric"] in wanted:
            pct = "n/a" if r["change_pct"] is None else f"{r['change_pct']:+.2f}"
            ci = "n/a (one block)" if r['ci95_low'] is None else f"[{r['ci95_low']:.3f}, {r['ci95_high']:.3f}]"
            lines.append(f"| {r['case']} | {r['baseline']} | {r['metric']} | {r['baseline_mean']:.3f} | "
                         f"{r['mixed_mean']:.3f} | {pct} | {ci} | "
                         f"{r['positive']} / {r['negative']} |")
    (root / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("REPORT", root / "analysis.md")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--mode", choices=["burst", "boundary", "stream"], default="burst")
    args = parser.parse_args()
    analyze(args.root, args.mode)
