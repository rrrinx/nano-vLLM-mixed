"""Validate complete replays, with stricter gates for burst mechanism pilots."""
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def audit(payload, mode="burst", details=False):
    if payload.get("status") != "complete":
        raise ValueError("Replay is incomplete")
    specs = {r["id"]: r for r in payload["trace"]["requests"]}
    expected = sum(r["max_tokens"] for r in specs.values())
    findings = []
    for run in payload["runs"]:
        summary = run["summary"]
        if len(run["requests"]) != len(specs) or summary["output_tokens"] != expected:
            raise ValueError("Request/output count mismatch")
        if {r["id"] for r in run["requests"]} != set(specs):
            raise ValueError("Request identity mismatch")
        if not math.isclose(summary["output_tok_s"] * summary["elapsed_s"], expected, abs_tol=1e-7):
            raise ValueError("Throughput identity failed")
        for request in run["requests"]:
            if request["output_tokens"] != specs[request["id"]]["max_tokens"]:
                raise ValueError("Per-request output limit mismatch")
            if not math.isclose(request["e2e_ms"], request["ttft_ms"] +
                                (request["output_tokens"] - 1) * request["tpot_ms"], abs_tol=1e-6):
                raise ValueError("Latency identity failed")
        preemptions = summary["counts"].get("preemptions", 0)
        if mode != "stream" and preemptions:
            raise ValueError("Unexpected preemption: review fixed KV capacity")
        if mode == "burst":
            overlap = run["online"]["overlap"]
            if (overlap["old_decoding_at_arrival"] != 4 or
                    overlap["old_decoding_at_admission"] != 4):
                raise ValueError("Burst overlap failed; recalibrate a COMMON --delay in a NEW plan")
        joint = None
        if details:
            steps = run["details"]["steps"]
            joint = sum(
                any(e["phase"] == "decode" and specs[e["id"]]["group"] == "old_decode"
                    for e in step["entries"])
                and any(e["phase"] == "prefill" and specs[e["id"]]["group"] == "late_prefill"
                        for e in step["entries"])
                for step in steps)
            if mode == "burst" and run["policy"] == "mixed" and not joint:
                raise ValueError("No actual old-decode / late-prefill joint step")
        findings.append({"policy": run["policy"], "preemptions": preemptions,
                         "joint_old_late_steps": joint, "overlap": run["online"]["overlap"]})
    return findings
