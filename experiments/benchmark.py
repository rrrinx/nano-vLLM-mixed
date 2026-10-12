import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Replay identical offline/online traces; see the mixed scheduling guide.

The current framework supports only `separate`. The other modes require the
guide's Config.scheduling_policy and BatchPlan interface. No scheduler is patched
by this script. --self-test and --make-trace need only the Python standard library.
"""

import argparse
import hashlib
import json
import math
import platform
import random
import statistics
import sys
from collections import Counter
from dataclasses import fields
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter, sleep


POLICIES = ("separate", "interleaved", "mixed")


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def distribution(values):
    values = sorted(values)
    if not values:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    result = {"n": len(values), "mean": statistics.fmean(values)}
    for name, fraction in (("p50", .50), ("p95", .95), ("p99", .99)):
        result[name] = values[math.ceil(fraction * len(values)) - 1]
    return result


def token_metrics(arrival, admission, marks):
    intervals = [(b - a) * 1000 for a, b in zip(marks, marks[1:])]
    return {
        "admission_delay_ms": (admission - arrival) * 1000,
        "ttft_ms": (marks[0] - arrival) * 1000,
        "e2e_ms": (marks[-1] - arrival) * 1000,
        "tpot_ms": statistics.fmean(intervals) if intervals else None,
        "itl_ms": intervals,
        "max_itl_ms": max(intervals) if intervals else None,
    }


def validate_trace(trace):
    if trace.get("schema") != 1 or not trace.get("requests"):
        raise ValueError("trace needs schema=1 and nonempty requests")
    seen, last = set(), -1.0
    for request in trace["requests"]:
        rid = request["id"]
        arrival = request["arrival_s"]
        tokens = request["prompt_token_ids"]
        output = request["output_tokens"]
        if type(rid) is not str or not rid or rid in seen:
            raise ValueError("request IDs must be unique nonempty strings")
        if type(arrival) not in (int, float) or not math.isfinite(arrival) or arrival < 0 or arrival < last:
            raise ValueError("arrivals must be finite, nonnegative and sorted")
        if not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("prompts must contain nonnegative integer token IDs")
        if type(output) is not int or output < 1:
            raise ValueError("output_tokens must be a positive integer")
        seen.add(rid)
        last = arrival
    if trace["requests"][0]["arrival_s"] != 0:
        raise ValueError("normalize the first arrival to zero")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def make_trace(args):
    prompt_lens = [int(x) for x in args.prompt_lengths.split(",")]
    output_lens = [int(x) for x in args.output_lengths.split(",")]
    if min(prompt_lens + output_lens) < 1 or args.vocab_size < 2 or args.requests < 1:
        raise ValueError("lengths and request count must be positive; vocab-size >= 2")
    if not math.isfinite(args.arrival_interval_ms) or args.arrival_interval_ms < 0:
        raise ValueError("arrival interval must be finite and nonnegative")
    rng = random.Random(args.seed)
    requests = []
    for i in range(args.requests):
        requests.append({
            "id": f"r{i}",
            "arrival_s": 0.0 if args.arrival_mode == "offline" else i * args.arrival_interval_ms / 1000,
            "prompt_token_ids": [rng.randrange(args.vocab_size) for _ in range(prompt_lens[i % len(prompt_lens)])],
            "output_tokens": output_lens[i % len(output_lens)],
        })
    trace = {"schema": 1, "seed": args.seed, "synthetic": True, "requests": requests}
    validate_trace(trace)
    write_json(args.make_trace, trace)
    print(f"trace saved: {args.make_trace}; workload_sha256={digest(requests)}")


def inspect_plan(plan):
    """Observe a legacy tuple or the guide's BatchPlan.entries after selection."""
    if plan is None:
        return []
    if hasattr(plan, "entries"):
        return [(entry.seq, entry.phase, entry.cache_len, entry.query_len) for entry in plan.entries]
    seqs, is_prefill = plan
    return [(seq, "prefill" if is_prefill else "decode",
             seq.num_cached_tokens, seq.num_scheduled_tokens) for seq in seqs]


def replay(engine, requests, timing, sync, timeout_s, ttft_slo_ms, itl_slo_ms):
    """Open-loop arrivals are measured from their scheduled time, even if late.

    Sequence references survive release from the scheduler, so token deltas can
    be observed after postprocess without changing Engine.step's return type.
    """
    scheduler = engine.scheduler
    original_schedule, original_preempt = scheduler.schedule, scheduler.preempt
    selected, counts = [], Counter()
    steps, records, by_sid, outputs = [], {}, {}, {}

    def observe_schedule():
        plan = original_schedule()
        selected[:] = inspect_plan(plan)
        return plan

    def observe_preempt(seq):
        counts["preemptions"] += 1
        return original_preempt(seq)

    scheduler.schedule, scheduler.preempt = observe_schedule, observe_preempt
    from nanovllm import SamplingParams

    sync()
    t0, cursor = perf_counter(), 0
    try:
        while cursor < len(requests) or not engine.is_finished():
            elapsed = perf_counter() - t0
            if elapsed > timeout_s:
                raise RuntimeError("replay timeout: do not report an incomplete run as throughput")
            # Admit all requests already due, including time spent in add_request.
            while cursor < len(requests) and requests[cursor]["arrival_s"] <= perf_counter() - t0:
                request = requests[cursor]
                sid = engine.add_request(request["prompt_token_ids"], SamplingParams(
                    max_tokens=request["output_tokens"], ignore_eos=True))
                by_sid[sid] = request["id"]
                records[request["id"]] = {
                    "id": request["id"], "arrival_s": request["arrival_s"],
                    "admission_s": perf_counter() - t0, "marks": [],
                }
                cursor += 1
            if engine.is_finished():
                sleep(min(.01, max(0.0, requests[cursor]["arrival_s"] - (perf_counter() - t0))))
                continue
            before = {sid: seq.num_completion_tokens for sid, seq in scheduler.request.items()} if timing == "latency" else {}
            start = perf_counter()
            completed = engine.step()
            if timing == "latency":
                sync()
            end = perf_counter()
            phases = {phase for _, phase, _, _ in selected}
            if not phases or not phases <= {"prefill", "decode"}:
                raise RuntimeError("invalid or missing observed batch phases")
            kind = "mixed" if len(phases) == 2 else next(iter(phases))
            counts[kind + "_steps"] += 1
            counts["steps"] += 1
            queries = sum(q for _, _, _, q in selected)
            counts["query_tokens"] += queries
            for seq, phase, c, q in selected:
                counts[phase + "_query_tokens"] += q
                counts["kv_token_visits"] += c + q
                if timing == "latency":
                    delta = seq.num_completion_tokens - before[seq.seq_id]
                    if delta not in (0, 1):
                        raise RuntimeError("ordinary greedy decoding should append zero or one token per step")
                    if delta:
                        records[by_sid[seq.seq_id]]["marks"].append(end - t0)
            if timing == "latency":
                steps.append({
                    "kind": kind, "wall_ms": (end - start) * 1000,
                    "start_s": start - t0, "end_s": end - t0,
                    "requests": [{"id": by_sid[s.seq_id], "phase": phase, "c": c, "q": q}
                                 for s, phase, c, q in selected],
                })
            for sid, tokens in completed:
                rid = by_sid[sid]
                if rid in outputs:
                    raise RuntimeError("duplicate completed request")
                outputs[rid] = list(tokens)
        sync()
        elapsed = perf_counter() - t0
    finally:
        scheduler.schedule, scheduler.preempt = original_schedule, original_preempt

    for request in requests:
        rid = request["id"]
        if len(outputs.get(rid, [])) != request["output_tokens"]:
            raise RuntimeError(f"wrong output length for {rid}")
        if timing == "latency" and len(records[rid]["marks"]) != request["output_tokens"]:
            raise RuntimeError(f"missing emitted-token timestamps for {rid}")
    if scheduler.request or scheduler.cache_manager.layer_lengths or scheduler.cache_manager.block_manager.tables:
        raise RuntimeError("live cache state leaked after drain")

    per_request = []
    if timing == "latency":
        for record in records.values():
            metrics = token_metrics(record["arrival_s"], record["admission_s"], record["marks"])
            per_request.append({**record, **metrics})
    output_count = sum(len(tokens) for tokens in outputs.values())
    slo_count = None
    if timing == "latency" and ttft_slo_ms is not None and itl_slo_ms is not None:
        slo_count = sum(r["ttft_ms"] <= ttft_slo_ms and
                        (r["max_itl_ms"] is None or r["max_itl_ms"] <= itl_slo_ms)
                        for r in per_request)
    summary = {
        "elapsed_s": elapsed, "output_tokens": output_count,
        "output_tok_s": output_count / elapsed, "requests_s": len(requests) / elapsed,
        "counts": dict(counts),
        "mixed_step_fraction": counts["mixed_steps"] / counts["steps"],
        "mean_query_tokens_per_step": counts["query_tokens"] / counts["steps"],
        "ttft_ms": distribution([r["ttft_ms"] for r in per_request]),
        "itl_ms": distribution([x for r in per_request for x in r["itl_ms"]]),
        "request_tpot_ms": distribution([r["tpot_ms"] for r in per_request if r["tpot_ms"] is not None]),
        "e2e_ms": distribution([r["e2e_ms"] for r in per_request]),
        "admission_delay_ms": distribution([r["admission_delay_ms"] for r in per_request]),
        "step_ms": distribution([s["wall_ms"] for s in steps]),
        "slo_pass_requests": slo_count,
        "slo_pass_fraction": None if slo_count is None else slo_count / len(requests),
        "slo_goodput_requests_s": None if slo_count is None else slo_count / elapsed,
    }
    return {"summary": summary, "requests": per_request, "steps": steps,
            "outputs": outputs, "output_sha256": digest(outputs)}


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def self_test():
    assert distribution([])["p99"] is None
    assert distribution(list(range(1, 101)))["p95"] == 95
    m = token_metrics(0.0, .25, [1.0, 1.5, 2.0])
    assert m["ttft_ms"] == 1000 and m["itl_ms"] == [500, 500] and m["admission_delay_ms"] == 250
    assert token_metrics(0.0, .25, [1.0])["tpot_ms"] is None
    trace = {"schema": 1, "requests": [{"id": "a", "arrival_s": 0, "prompt_token_ids": [1], "output_tokens": 1}]}
    validate_trace(trace)
    for invalid in (-1, float("nan"), float("inf")):
        trace["requests"][0]["arrival_s"] = invalid
        try:
            validate_trace(trace)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid arrival accepted")
    assert digest({"b": 2, "a": 1}) == digest({"a": 1, "b": 2})
    print("mixed benchmark metric/trace checks passed; no model or CUDA executed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--policy", choices=POLICIES, default="separate")
    parser.add_argument("--timing", choices=("throughput", "latency"), default="throughput")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--token-budget", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--blocks", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=3600)
    parser.add_argument("--ttft-slo-ms", type=float)
    parser.add_argument("--itl-slo-ms", type=float)
    parser.add_argument("--out", type=Path, default=Path("artifacts/mixed-scheduling/result.json"))
    parser.add_argument("--make-trace", type=Path)
    parser.add_argument("--arrival-mode", choices=("offline", "online"), default="offline")
    parser.add_argument("--arrival-interval-ms", type=float, default=100)
    parser.add_argument("--requests", type=int, default=12)
    parser.add_argument("--prompt-lengths", default="128,512,1024")
    parser.add_argument("--output-lengths", default="32,64")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.make_trace:
        make_trace(args)
        return
    if not args.model or not args.trace:
        parser.error("model and --trace are required for replay")
    for name in ("batch_size", "token_budget", "block_size", "blocks", "warmup", "repeats"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("timeout-s must be finite and positive")
    if (args.ttft_slo_ms is None) != (args.itl_slo_ms is None):
        parser.error("provide both SLO thresholds or neither")
    for value in (args.ttft_slo_ms, args.itl_slo_ms):
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error("SLO thresholds must be finite and positive")
    trace = json.loads(args.trace.read_text(encoding="utf-8-sig"))
    validate_trace(trace)
    requests = trace["requests"]

    import torch
    from nanovllm import Config, LLM, SamplingParams

    available = {field.name for field in fields(Config)}
    policy_supported = "scheduling_policy" in available
    if not policy_supported and args.policy != "separate":
        parser.error("mixed scheduling guide is not implemented: Config.scheduling_policy is missing")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            parser.error("CUDA requested but unavailable; CPU results cannot stand in for GPU performance")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    max_len = max(len(r["prompt_token_ids"]) + r["output_tokens"] - 1 for r in requests)
    config = {
        "device": str(device), "dtype": args.dtype,
        "max_num_seqs": args.batch_size, "max_num_batched_tokens": args.token_budget,
        "max_model_len": max_len, "kvcache_block_size": args.block_size,
        "num_kvcache_blocks": args.blocks, "enable_prefix_cache": False,
    }
    # If stage 9 is implemented later, keep its independent optimizations off.
    config.update({k: v for k, v in {"decode_backend": "python", "enable_compile": False,
                                   "enforce_eager": True}.items() if k in available})
    if policy_supported:
        config["scheduling_policy"] = args.policy
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    engine = LLM(args.model, **config)
    try:
        if policy_supported and getattr(engine.scheduler, "scheduling_policy", None) != args.policy:
            parser.error("Config.scheduling_policy was not connected to Scheduler")
        vocab = engine.model_runner.model.model.embed_tokens.num_embeddings
        if any(max(r["prompt_token_ids"]) >= vocab for r in requests):
            parser.error("trace token ID exceeds the model vocabulary")
        if max_len > engine.config.max_model_len:
            parser.error("trace exceeds model position capacity")
        warm = requests[0]
        for _ in range(args.warmup):
            engine.generate([warm["prompt_token_ids"]],
                            SamplingParams(max_tokens=warm["output_tokens"], ignore_eos=True), use_tqdm=False)
        # Exercise the exact policy and workload once, including mixed shapes.
        replay(engine, requests, "throughput", sync, args.timeout_s, None, None)
        runs = []
        for _ in range(args.repeats):
            if device.type == "cuda":
                sync()
                torch.cuda.reset_peak_memory_stats(device)
            run = replay(engine, requests, args.timing, sync, args.timeout_s, args.ttft_slo_ms, args.itl_slo_ms)
            run["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            runs.append(run)
        if len({run["output_sha256"] for run in runs}) != 1:
            raise RuntimeError("greedy outputs changed between repetitions; investigate before comparing speed")
        result = {
            "schema": 1, "policy": args.policy, "policy_api": "unified" if policy_supported else "legacy",
            "timing": args.timing, "workload_sha256": digest(requests),
            "model": str(Path(args.model).resolve()), "model_config_sha256": digest(engine.config.hf_config.to_dict()),
            "config": config, "seed": args.seed, "warmup": args.warmup,
            "trace_warmup_replays": 1,
            "slo": {"ttft_ms": args.ttft_slo_ms, "max_itl_ms": args.itl_slo_ms},
            "environment": {"python": sys.version, "platform": platform.platform(),
                            "torch": torch.__version__, "transformers": package_version("transformers"),
                            "cuda": torch.version.cuda, "cpu_threads": torch.get_num_threads(),
                            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                            "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory if device.type == "cuda" else None,
                            "tf32": False},
            "median_output_tok_s": statistics.median(r["summary"]["output_tok_s"] for r in runs),
            "median_elapsed_s": statistics.median(r["summary"]["elapsed_s"] for r in runs),
            "runs": runs,
        }
        write_json(args.out, result)
        print(f"{args.policy}: median={result['median_output_tok_s']:.3f} output tok/s; {args.out}")
        if args.policy == "mixed" and not any(r["summary"]["counts"].get("mixed_steps", 0) for r in runs):
            print("No mixed batches occurred: this workload does not measure mixed execution benefits.")
    finally:
        engine.exit()


if __name__ == "__main__":
    main()
