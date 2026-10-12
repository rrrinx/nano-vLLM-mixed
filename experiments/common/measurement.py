"""Time actual engine steps; token timestamps refer to committed outputs."""

import atexit
import json
import os
import platform
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter, sleep

from experiments.benchmark import digest, distribution, package_version, token_metrics, write_json
from experiments.common.diagnostics import DiagnosticMonitor


def assert_idle(engine):
    manager = engine.model_runner.kv_cache_manager
    blocks = manager.block_manager
    if (not engine.is_finished() or engine.scheduler.request or manager.layer_lengths
            or blocks.tables or any(b.ref_count for b in blocks.blocks)
            or len(blocks.free_block_id) != manager.num_blocks):
        raise RuntimeError("live requests or KV references remain after replay")


def reset_cache(engine):
    """Reuse the one KV pool, resetting cold prefixes and physical free order.

    Between replays every reference is gone. Attention backends retain the
    same Manager; replacing its metadata allocator never creates a second pool.
    Old tensor contents cannot be read until new requests write valid KV.
    """
    from nanovllm.engine.block_manager import BlockManager

    assert_idle(engine)
    manager = engine.model_runner.kv_cache_manager
    manager.block_manager = BlockManager(manager.num_blocks, manager.block_size,
                                        manager.enable_prefix_cache)


def replay(engine, requests, max_tokens, timing, sync, timeout_s, *, ignore_eos=True,
           save_steps=False, clock=perf_counter, wait=sleep, diagnostics=None, label=None,
           request_limits=None):
    from nanovllm import SamplingParams

    if timing not in ("latency", "throughput"):
        raise ValueError("timing must be latency or throughput")
    assert_idle(engine)

    ids = [request["id"] for request in requests]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate logical request ID")
    if request_limits is None:
        limits = {rid: max_tokens for rid in ids}
    else:
        limits = dict(request_limits)
        if set(limits) != set(ids):
            raise ValueError("request_limits must cover exactly this workload")
    if any(type(value) is not int or value < 1 for value in limits.values()):
        raise ValueError("each output limit must be a positive integer")


    scheduler = engine.scheduler
    original_schedule, original_preempt = scheduler.schedule, scheduler.preempt
    selected, counts = [], Counter()
    records, by_sid, outputs, steps = {}, {}, {}, []

    def schedule():
        plan = original_schedule()
        selected[:] = [] if plan is None else list(plan.entries)
        return plan

    def preempt(seq):
        counts["preemptions"] += 1
        return original_preempt(seq)

    scheduler.schedule, scheduler.preempt = schedule, preempt
    try:
        sync()
        if diagnostics is not None:
            diagnostics.begin(label)
        start_time, cursor = clock(), 0
        while cursor < len(requests) or not engine.is_finished():
            elapsed = clock() - start_time
            if elapsed > timeout_s:
                raise TimeoutError("replay exceeded timeout; incomplete work has no throughput result")
            while cursor < len(requests) and requests[cursor]["arrival_s"] <= clock() - start_time:
                request = requests[cursor]
                sid = engine.add_request(request["prompt_token_ids"], SamplingParams(
                    max_tokens=limits[request["id"]], ignore_eos=ignore_eos))
                by_sid[sid] = request["id"]
                records[request["id"]] = {
                    "id": request["id"], "dataset_index": request["dataset_index"],
                    "input_tokens": len(request["prompt_token_ids"]),
                    "arrival_s": request["arrival_s"], "admission_s": clock() - start_time,
                    "marks": [],
                }
                cursor += 1
            if engine.is_finished():
                if cursor < len(requests):
                    wait(min(.01, max(0.0, requests[cursor]["arrival_s"] - (clock() - start_time))))
                continue
            before = {sid: s.num_completion_tokens for sid, s in scheduler.request.items()} if timing == "latency" else {}
            step_start = clock()
            completed = engine.step()
            if timing == "latency":
                sync()
            step_end = clock()
            if not selected:
                raise RuntimeError("engine step executed without a batch plan")
            phases = {e.phase for e in selected}
            kind = "mixed" if len(phases) == 2 else next(iter(phases))
            counts["steps"] += 1
            counts[kind + "_steps"] += 1
            for entry in selected:
                counts["query_tokens"] += entry.query_len
                counts[entry.phase + "_query_tokens"] += entry.query_len
                counts["kv_token_visits"] += entry.cache_len + entry.query_len
                if timing == "latency":
                    delta = entry.seq.num_completion_tokens - before[entry.seq.seq_id]
                    if delta != int(entry.should_sample):
                        raise RuntimeError("committed output delta disagrees with batch sampling plan")
                    if delta:
                        records[by_sid[entry.seq.seq_id]]["marks"].append(step_end - start_time)
            if save_steps:
                steps.append({"kind": kind, "start_s": step_start - start_time,
                              "end_s": step_end - start_time,
                              "entries": [{"id": by_sid[e.seq.seq_id], "phase": e.phase,
                                           "c": e.cache_len, "q": e.query_len,
                                           "should_sample": e.should_sample} for e in selected]})
            for sid, tokens in completed:
                rid = by_sid[sid]
                if rid in outputs:
                    raise RuntimeError("request completed twice")
                outputs[rid] = list(tokens)
        sync()
        elapsed = clock() - start_time
        if diagnostics is not None:
            diagnostics.end(elapsed)
        if elapsed > timeout_s:
            raise TimeoutError("replay completed after timeout; do not report successful throughput")
    finally:
        scheduler.schedule, scheduler.preempt = original_schedule, original_preempt

    assert_idle(engine)
    if len(outputs) != len(requests) or elapsed <= 0:
        raise RuntimeError("replay must complete every request with a positive elapsed time")
    per_request = []
    for request in requests:
        rid = request["id"]
        n = len(outputs[rid])
        cap = limits[rid]
        if not 1 <= n <= cap or (ignore_eos and n != cap):
            raise RuntimeError(f"unexpected output length for {rid}: {n}")
        record = records[rid]
        if timing == "latency":
            if len(record["marks"]) != n:
                raise RuntimeError(f"missing committed token timestamps for {rid}")
            record.update(token_metrics(record["arrival_s"], record["admission_s"], record["marks"]))
        record["output_tokens"] = n
        per_request.append(record)
    input_count = sum(len(r["prompt_token_ids"]) for r in requests)
    output_count = sum(len(t) for t in outputs.values())
    summary = {
        "elapsed_s": elapsed, "requests": len(requests),
        "input_tokens": input_count, "output_tokens": output_count,
        "output_tok_s": output_count / elapsed, "input_tok_s": input_count / elapsed,
        "total_tok_s": (input_count + output_count) / elapsed,
        "requests_s": len(requests) / elapsed,
        "counts": dict(counts), "mixed_step_fraction": counts["mixed_steps"] / counts["steps"],
    }
    for name, field in (("ttft_ms", "ttft_ms"), ("tpot_ms", "tpot_ms"),
                        ("e2e_ms", "e2e_ms"), ("admission_delay_ms", "admission_delay_ms")):
        summary[name] = distribution([r[field] for r in per_request if r.get(field) is not None])
    summary["itl_ms"] = distribution([x for r in per_request for x in r.get("itl_ms", [])])
    summary["token_weighted_tpot_ms"] = summary["itl_ms"]["mean"]
    return {"summary": summary, "requests": per_request, "steps": steps,
            "outputs": outputs, "output_sha256": digest(outputs)}


def config_line(job, config, cache=None):
    """One configuration line; also marks successful model initialization."""
    settings, factors = job["settings"], job["factors"]
    allocation = cache["allocation"] if cache else ("auto" if settings["blocks"] == -1 else "manual")
    blocks = cache["num_blocks"] if cache else settings["blocks"]
    capacity = cache["capacity_tokens"] if cache else "pending"
    round_info = (f" round={job['round']}/{job['round_count']} policy_position={job['policy_position']}"
                  if job.get("round") is not None else "")
    if job.get("shared_worker_id"):
        round_info += " warmup_scope=policy_once"
    return (f"{job['case_id']}: config policy={factors['policy']} N={job['request_count']} "
            f"dtype={config['dtype']} device={config['device']} "
            f"max_tokens={factors['max_tokens']} max_seqs={factors['max_seqs']} "
            f"token_budget={factors['token_budget']} max_model_len={config['max_model_len']} "
            f"prefix_cache={settings['prefix_cache']} ignore_eos={settings['ignore_eos']} "
            f"warmup={settings['warmup']} repeats={settings['repeats']} "
            f"timings={','.join(settings['timings'])} KV={allocation} blocks={blocks} "
            f"capacity={capacity} utilization={config.get('gpu_memory_utilization', 'engine-default')} "
            f"diagnostics={settings.get('diagnostics', False)} "
            f"diagnostics_interval={settings.get('diagnostics_interval_s', 1.0)}s{round_info}")


def metric_line(case_id, run):
    summary = run["summary"]

    def number(value):
        return "-" if value is None else f"{value:.3f}"

    line = (f"{case_id} {run['timing']} repeat {run['repeat']}: "
            f"elapsed={summary['elapsed_s']:.3f}s output={summary['output_tok_s']:.3f}tok/s "
            f"input={summary['input_tok_s']:.3f}tok/s total={summary['total_tok_s']:.3f}tok/s "
            f"requests={summary['requests_s']:.3f}/s output_tokens={summary['output_tokens']} "
            f"steps={summary['counts']['steps']} preemptions={summary['counts'].get('preemptions', 0)} "
            f"mixed={summary['mixed_step_fraction']:.3f} "
            f"peak_cuda={number(summary['peak_cuda_memory_mb'])}MiB")
    if run["timing"] == "latency":
        for label, name in (("TTFT", "ttft_ms"), ("TPOT", "tpot_ms"), ("ITL", "itl_ms"), ("E2E", "e2e_ms")):
            line += f" {label}_mean={number(summary[name]['mean'])}ms {label}_p95={number(summary[name]['p95'])}ms"
    if "process_cpu_s" in summary:
        line += (f" CPU={number(summary['process_cpu_pct'])}% "
                 f"peak_reserved={number(summary['peak_cuda_reserved_mb'])}MiB")
    return line


def run_worker(job):
    if "members" in job:
        results = run_jobs(job["members"], shared_job=job)
        return {"status": "ok" if len({r["output_sha256"] for result in results for r in result["runs"]}) == 1
                else "unstable"}
    return run_jobs([job])[0]


def run_jobs(jobs, *, shared_job=None):
    """One model/KV pool per parameter combination; warm each policy once.

    Each replay starts with cold KV metadata and a fresh idle scheduler in
    shared mode. No requests survive a policy switch, and no model or pool is
    reallocated between measured rounds. All case files are written at the end.
    """
    import torch
    from nanovllm import LLM
    from nanovllm.engine.scheduler import Scheduler

    job = jobs[0]
    settings = job["settings"]
    torch.set_num_threads(settings["cpu_threads"])
    torch.manual_seed(settings["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(settings["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        torch.cuda.set_device(device)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    workload = json.loads(Path(job["workload_path"]).read_text(encoding="utf-8-sig"))
    if digest(workload["requests"]) != job["workload_sha256"]:
        raise ValueError("workload changed after experiment preparation")
    config = dict(job["engine_config"], device=str(device))
    for member in jobs:
        common = {k: v for k, v in member["engine_config"].items() if k != "scheduling_policy"}
        if (common != {k: v for k, v in job["engine_config"].items() if k != "scheduling_policy"}
                or member["workload_sha256"] != job["workload_sha256"]
                or member["settings"] != settings
                or member["factors"]["max_tokens"] != job["factors"]["max_tokens"]):
            raise ValueError("shared jobs must use the same model, workload, limits and settings")
    engine = LLM(settings["model"], **config)
    manager = engine.model_runner.kv_cache_manager
    resolved_config = dict(config, num_kvcache_blocks=engine.config.num_kvcache_blocks,
                           max_model_len=engine.config.max_model_len,
                           gpu_memory_utilization=engine.config.gpu_memory_utilization)
    cache = {"allocation": "auto" if config.get("num_kvcache_blocks", -1) == -1 else "manual",
             "num_blocks": manager.num_blocks, "block_size": manager.block_size,
             "capacity_tokens": manager.num_blocks * manager.block_size,
             "pool_mb": (manager.k_pool.numel() * manager.k_pool.element_size()
                         + manager.v_pool.numel() * manager.v_pool.element_size()) / 2**20,
             "dtype": str(manager.k_pool.dtype)}
    environment = {"python": sys.version, "platform": platform.platform(),
                   "torch": torch.__version__, "transformers": package_version("transformers"),
                   "cuda": torch.version.cuda, "cpu_threads": torch.get_num_threads(), "tf32": False,
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    environment.update(cpu_count=os.cpu_count(),
                       cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                       thread_environment={key: os.environ.get(key) for key in (
                           "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                           "TOKENIZERS_PARALLELISM", "CUDA_VISIBLE_DEVICES")})
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        environment.update(gpu_uuid=str(properties.uuid), gpu_total_memory_mb=properties.total_memory / 2**20)
    monitor = (DiagnosticMonitor(settings["diagnostics_interval_s"], torch, device)
               if settings.get("diagnostics", False) else None)
    results, labels, warmups = [], {}, {}

    def select_policy(member):
        if shared_job is not None:
            assert_idle(engine)
            policy = member["factors"]["policy"]
            # Use the real Scheduler constructor to reset phase/history, while
            # retaining ownership of the same model and paged KV tensor pool.
            engine.scheduler = Scheduler(
                manager, engine.config.max_num_seqs, engine.config.max_num_batched_tokens,
                engine.config.max_model_len, scheduling_policy=policy)
            engine.config.scheduling_policy = policy

    def measure(member, timing, label, *, warmup=False):
        select_policy(member)
        reset_cache(engine)
        if device.type == "cuda" and (not warmup or monitor is not None):
            sync()
            torch.cuda.reset_peak_memory_stats(device)
        run = replay(engine, workload["requests"], member["factors"]["max_tokens"], timing, sync,
                     settings["timeout_s"], ignore_eos=settings["ignore_eos"],
                     diagnostics=monitor, label=label)
        run["summary"].update(num_kvcache_blocks=cache["num_blocks"],
                              kv_capacity_tokens=cache["capacity_tokens"], kv_pool_mb=cache["pool_mb"],
                              peak_cuda_memory_mb=(torch.cuda.max_memory_allocated(device) / 2**20
                                                   if device.type == "cuda" else None))
        if monitor is not None:
            run["summary"].update(monitor.summary(label))
        return run

    try:
        if monitor is not None:
            monitor.start()
        # This header also marks successful initialization for native retries.
        print(config_line(shared_job or job, resolved_config, cache), flush=True)
        policy_jobs = {}
        for member in jobs:
            policy_jobs.setdefault(member["factors"]["policy"], member)
        for policy, member in policy_jobs.items():
            warmups[policy] = []
            for repeat in range(settings["warmup"]):
                label = f"warmup/{policy}/{repeat + 1}" if shared_job else f"warmup/{repeat + 1}"
                warmup = measure(member, "throughput", label, warmup=True)
                if monitor is not None:
                    warmups[policy].append({"repeat": repeat + 1, "summary": warmup["summary"]})
                    labels[(policy, "warmup", repeat + 1)] = label
                del warmup
        for member in jobs:
            policy = member["factors"]["policy"]
            member_config = dict(config, scheduling_policy=policy)
            member_resolved = dict(resolved_config, scheduling_policy=policy)
            result = {"schema": 2, "case_id": member["case_id"], "fingerprint": member["fingerprint"],
                      "factors": member["factors"], "requested_config": member_config, "config": member_resolved,
                      "cache": cache, "status": "running", "runs": [], "environment": environment,
                      "workload_sha256": member["workload_sha256"]}
            if member.get("round") is not None:
                result.update(group_id=member["group_id"], round=member["round"],
                              policy_position=member["policy_position"], shared_worker_id=member.get("shared_worker_id"))
            if monitor is not None:
                result["warmup_runs"] = warmups[policy] if policy_jobs[policy] is member else []
            if shared_job is not None:
                print(config_line(member, member_resolved, cache), flush=True)
            for timing in settings["timings"]:
                for repeat in range(settings["repeats"]):
                    repeat_number = member.get("round") or repeat + 1
                    label = (f"{member['case_id']}/{timing}/{repeat_number}" if shared_job
                             else f"{timing}/{repeat_number}")
                    run = measure(member, timing, label)
                    compact = {"timing": timing, "repeat": repeat_number,
                               "summary": run["summary"], "output_sha256": run["output_sha256"]}
                    result["runs"].append(compact)
                    labels[(member["case_id"], timing, repeat_number)] = label
                    del run
                    print(metric_line(member["case_id"], compact), flush=True)
            results.append(result)
        if monitor is not None:
            monitor.stop()
            # Drain delayed NVIDIA samples before exporting final aggregates.
            for result in results:
                policy = result["factors"]["policy"]
                for run in result["runs"]:
                    run["summary"].update(monitor.summary(labels[(result["case_id"], run["timing"], run["repeat"])]))
                for warmup in result["warmup_runs"]:
                    warmup["summary"].update(monitor.summary(labels[(policy, "warmup", warmup["repeat"])]))
                result["diagnostics"] = monitor.metadata()
        for member, result in zip(jobs, results):
            result["status"] = "ok" if len({r["output_sha256"] for r in result["runs"]}) == 1 else "unstable"
            write_json(Path(member["result_path"]), result)
            print(f"{member['case_id']}: status={result['status']} "
                  f"matching_outputs={result['status'] == 'ok'}", flush=True)
        return results
    finally:
        try:
            if monitor is not None:
                monitor.stop()
        finally:
            engine.exit()
            atexit.unregister(engine.exit)
