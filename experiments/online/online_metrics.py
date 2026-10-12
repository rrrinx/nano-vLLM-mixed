"""Cohort statistics computed after a complete replay, never inside GPU timing."""

import math

from experiments.benchmark import distribution


LATENCIES = ("ttft_ms", "tpot_ms", "itl_ms", "max_itl_ms", "e2e_ms", "admission_delay_ms")


def cohort_metrics(records, elapsed_s, ttft_slo_ms=None, itl_slo_ms=None):
    result = {"requests": len(records), "output_tokens": sum(r["output_tokens"] for r in records)}
    for metric in LATENCIES:
        values = ([gap for r in records for gap in r.get("itl_ms", [])] if metric == "itl_ms" else
                  [r[metric] for r in records if r.get(metric) is not None])
        result[metric] = distribution(values)
        result[metric]["max"] = max(values) if values else None
    result["token_weighted_tpot_ms"] = result["itl_ms"]["mean"]
    if ttft_slo_ms is not None or itl_slo_ms is not None:
        # A single-token request has no ITL requirement to violate, not a zero-ms ITL.
        passed = sum(
            (ttft_slo_ms is None or r["ttft_ms"] <= ttft_slo_ms)
            and (itl_slo_ms is None or r["max_itl_ms"] is None or r["max_itl_ms"] <= itl_slo_ms)
            for r in records)
        result.update(slo_pass_requests=passed, slo_pass_fraction=passed / len(records) if records else None,
                      slo_goodput_requests_s=passed / elapsed_s)
    return result


def summarize_online(run, requests, *, window_s=5.0, ttft_slo_ms=None, itl_slo_ms=None):
    if not math.isfinite(window_s) or window_s <= 0:
        raise ValueError("window_s must be finite and positive")
    specs = {r["id"]: r for r in requests}
    records = run["requests"]
    if len(records) != len(specs) or {r["id"] for r in records} != set(specs):
        raise ValueError("request records do not match the complete workload")
    if any(not r["marks"] for r in records):
        raise ValueError("latency timestamps required")
    elapsed = run["summary"]["elapsed_s"]
    scopes = {"all": cohort_metrics(records, elapsed, ttft_slo_ms, itl_slo_ms)}
    for name in sorted({r["group"] for r in requests}):
        chosen = [r for r in records if specs[r["id"]]["group"] == name]
        scopes[name] = cohort_metrics(chosen, elapsed, ttft_slo_ms, itl_slo_ms)
    old = [r for r in records if specs[r["id"]]["group"] == "old_decode"]
    late = [r for r in records if specs[r["id"]]["group"] == "late_prefill"]
    onset = min(r["arrival_s"] for r in late) if late else 0.0
    end = onset + window_s
    affected = [(right - left) * 1000 for r in old
                for left, right in zip(r["marks"], r["marks"][1:])
                if left < end and right > onset]
    window = distribution(affected)
    window["max"] = max(affected) if affected else None
    admission = min(r["admission_s"] for r in late) if late else None
    # Strict bounds: a token exactly at arrival is not evidence of prior decode.
    overlap = {"old_requests": len(old), "late_requests": len(late),
               "old_decoding_at_arrival": sum(r["marks"][0] < onset < r["marks"][-1] for r in old) if late else None,
               "old_decoding_at_admission": sum(r["marks"][0] < admission < r["marks"][-1] for r in old) if late else None}
    return {"groups": scopes, "window_start_s": onset, "window_end_s": end,
            "window_old_itl_ms": window, "overlap": overlap}


def request_rows(run, requests):
    specs = {r["id"]: r for r in requests}
    return [{**{k: v for k, v in r.items() if k not in ("marks", "itl_ms")},
             "group": specs[r["id"]]["group"], "max_tokens": specs[r["id"]]["max_tokens"],
             "first_token_s": r["marks"][0], "last_token_s": r["marks"][-1],
             "itl_samples": len(r["itl_ms"])} for r in run["requests"]]
