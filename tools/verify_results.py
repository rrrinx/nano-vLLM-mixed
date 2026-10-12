"""Audit all published burst and serving measurements with the standard library."""
import hashlib
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ("separate", "interleaved", "mixed")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    provenance = read(ROOT / "reports/data/PROVENANCE.json")
    assert len(provenance["records"]) == 72
    records = {r["published_path"]: r for r in provenance["records"]}
    assert len(records) == 72
    for stage, checks in provenance['source_checks'].items():
        for name, item in checks.items():
            source = ROOT / item['published_path']
            assert hashlib.sha256(source.read_bytes()).hexdigest() == item['published']
            if name.startswith('nanovllm/'):
                assert item['published'] == item['recorded']
        print(f"{stage}: source mapping verified; inference implementation unchanged")
    by_config = defaultdict(lambda: defaultdict(list))
    for stage in ("burst_requests", "long_input_serving"):
        directory = ROOT / "reports/data" / stage / "formal"
        schedule = read(directory / "schedule.json")
        assert schedule["block_ids"] == [0, 3, 4]
        assert read(directory / "blocks.json")["entries"] == schedule["entries"]
        assert len(schedule["entries"]) == 36
        seen, identities, text_sets, groups, count = set(), set(), set(), defaultdict(set), 0
        expected_requests = 8 if stage == "burst_requests" else 100
        for entry in schedule["entries"]:
            case = Path(entry["config"]).stem
            path = directory / f"block{entry['block']}" / case / "results.json"
            evidence = records[path.relative_to(ROOT).as_posix()]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == evidence["published_sha256"]
            data = read(path)
            assert data["publication_export"]["original_file_sha256"] == evidence["original_sha256"]
            assert data["status"] == "complete" and data["matching_outputs"] is False
            assert data["settings"]["token_budget"] == entry["token_budget"]
            assert data["settings"]["warmup"] == data["settings"]["rounds"] == 1
            assert tuple(entry["policy_order"]) == list(itertools.permutations(POLICIES))[entry["block"]]
            assert [r["policy"] for r in data["runs"]] == entry["policy_order"]
            identities.add(json.dumps({k: data[k] for k in ("source_sha256", "environment", "checkpoint_files")}, sort_keys=True))
            specs = {q["id"]: q for q in data["trace"]["requests"]}
            assert len(specs) == len(data["trace"]["requests"]) == expected_requests
            assert all("prompt_token_ids" not in q for q in specs.values())
            if stage == "long_input_serving":
                generation = data["trace"]["generation"]
                key = (generation["arrival_rate"], generation["seed"], entry["token_budget"])
                text_sets.add(json.dumps([{k: v for k, v in q.items() if k != "arrival_s"} for q in specs.values()], sort_keys=True))
            for position, run in enumerate(data["runs"], 1):
                unique = (case, entry["block"], run["policy"])
                assert unique not in seen
                seen.add(unique)
                groups[case, run["policy"]].add(entry["block"])
                assert run["policy_position"] == position
                requests, summary = run["requests"], run["summary"]
                assert len(requests) == len({q['id'] for q in requests}) == expected_requests
                assert {q['id'] for q in requests} == set(specs)
                assert summary["requests"] == expected_requests
                for q in requests:
                    assert q["output_tokens"] == specs[q["id"]]["max_tokens"]
                    assert q["input_tokens"] == specs[q["id"]]["input_tokens"]
                    assert q["arrival_s"] == specs[q["id"]]["arrival_s"]
                    assert q["arrival_s"] <= q["admission_s"] <= q["first_token_s"] <= q["last_token_s"]
                    assert math.isclose(q["e2e_ms"], q["ttft_ms"] + (q["output_tokens"] - 1) * q["tpot_ms"], abs_tol=1e-6)
                    assert math.isclose(q["e2e_ms"], (q["last_token_s"] - q["arrival_s"]) * 1000, abs_tol=1e-6)
                    assert math.isclose(q["ttft_ms"], (q["first_token_s"] - q["arrival_s"]) * 1000, abs_tol=1e-6)
                for metric in ("ttft_ms", "e2e_ms", "tpot_ms"):
                    assert all(math.isfinite(q[metric]) and q[metric] >= 0 for q in requests)
                    assert math.isclose(statistics.mean(q[metric] for q in requests), summary[metric]["mean"], abs_tol=1e-6)
                assert math.isclose(summary["elapsed_s"] * summary["output_tok_s"], sum(q["output_tokens"] for q in requests), abs_tol=1e-6)
                count += len(requests)
                if stage == "long_input_serving":
                    by_config[key][run["policy"]].append(summary)
        assert len(seen) == 108 and len(identities) == 1
        assert len(groups) == 36 and all(b == {0, 3, 4} for b in groups.values())
        assert len(list(directory.rglob("results.json"))) == 36
        if stage == "long_input_serving":
            assert len(text_sets) == 1
            assert set(by_config) == set(itertools.product((.5, .75, 1.), (20261012, 20261013), (384, 512)))
        print(f"{stage}: 36 files / 108 runs / {count} request records; audit passed")
    for seed in (20261012, 20261013):
        cell = by_config[1., seed, 384]
        a, b = cell['separate'], cell['mixed']
        throughput_a = statistics.mean(r['output_tok_s'] for r in a)
        throughput_b = statistics.mean(r['output_tok_s'] for r in b)
        e2e_a = statistics.mean(r['e2e_ms']['mean'] for r in a) / 1000
        e2e_b = statistics.mean(r['e2e_ms']['mean'] for r in b) / 1000
        print(f"长输入在线实验 seed={seed}, T384, 1 req/s: throughput {throughput_a:.2f} -> {throughput_b:.2f} ({100*(throughput_b/throughput_a-1):+.2f}%); E2E {e2e_a:.2f} -> {e2e_b:.2f} s ({100*(e2e_b/e2e_a-1):+.2f}%)")
    print("This audit validates measurements, not generated-answer equivalence.")


if __name__ == "__main__":
    main()
