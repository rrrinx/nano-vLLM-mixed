"""Replay configurable online arrivals with one resident model and cold-cache repeats."""

import argparse
import atexit
import json
import math
import platform
import sys
from datetime import datetime
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from experiments.benchmark import package_version, write_json
from experiments.common.diagnostics import DiagnosticMonitor
from experiments.common.measurement import assert_idle, metric_line, replay, reset_cache
from experiments.online.online_metrics import request_rows, summarize_online
from experiments.online.online_workload import check_identity, file_hash, prepare_trace, validate_trace
from experiments.online.reporting import save_report, summarize_runs


POLICIES = ("separate", "interleaved", "mixed")
DEFAULTS = {
    "model": None, "trace": None, "source": None, "dataset": None, "out": None,
    "profile": "poisson", "seed": 114514,
    "offset": 0, "shuffle": False, "max_tokens": 128,
    "old_count": 4, "late_count": 2, "short_length": 64, "long_length": 1024,
    "old_output": 128, "late_output": 16, "delay_s": 1.0, "late_interval_s": 0.0,
    "request_count": 100, "arrival_rate": 1.0,
    "prompt_lengths": [64, 1024], "output_lengths": [128, 16],
    "policies": list(POLICIES), "policy_orders": None, "rounds": 12, "warmup": 1,
    "device": "cuda:0", "dtype": "bfloat16", "max_seqs": 16, "token_budget": 256,
    "block_size": 128, "blocks": -1, "max_model_len": 0, "gpu_memory_utilization": .7,
    "prefix_cache": False, "ignore_eos": True, "cpu_threads": 1,
    "timeout_s": 3600.0, "window_s": 5.0, "trim_fraction": .1,
    "ttft_slo_ms": None, "itl_slo_ms": None,
    "save_details": False, "diagnostics": False, "diagnostics_interval_s": 1.0, "dry_run": False,
}
INTS = ("seed", "offset", "max_tokens", "old_count", "late_count", "short_length", "long_length", "old_output", "late_output",
        "request_count", "rounds", "warmup", "max_seqs", "token_budget", "block_size", "blocks",
        "max_model_len", "cpu_threads")
FLOATS = ("delay_s", "late_interval_s", "arrival_rate", "gpu_memory_utilization", "timeout_s",
          "window_s", "trim_fraction", "ttft_slo_ms", "itl_slo_ms", "diagnostics_interval_s")
BOOLS = ("prefix_cache", "ignore_eos", "save_details", "diagnostics", "dry_run", "shuffle")


def parse_options(argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path)
    known, _ = preliminary.parse_known_args(argv)
    options = dict(DEFAULTS)
    if known.config:
        supplied = json.loads(known.config.read_text(encoding="utf-8-sig"))
        if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
            raise ValueError("config must be an object containing only documented options")
        options.update(supplied)
    parser = argparse.ArgumentParser(description=__doc__, parents=[preliminary],
                                     argument_default=argparse.SUPPRESS)
    for name in INTS:
        aliases = ("--rounds", "--repeats") if name == "rounds" else ("--" + name.replace("_", "-"),)
        if name == "request_count":
            aliases = ("--request-count", "--num-problems")
        parser.add_argument(*aliases, dest=name, type=int)
    for name in FLOATS:
        parser.add_argument("--" + name.replace("_", "-"), type=float)
    for name in BOOLS:
        parser.add_argument("--" + name.replace("_", "-"), action=argparse.BooleanOptionalAction)
    for name in ("model", "trace", "source", "dataset", "out", "profile", "device", "dtype",
                 "policies", "policy_orders", "prompt_lengths", "output_lengths"):
        parser.add_argument("--" + name.replace("_", "-"))
    options.update({k: v for k, v in vars(parser.parse_args(argv)).items() if k != "config"})
    for name in INTS:
        if type(options[name]) is not int:
            raise ValueError(f"{name} must be an integer")
    for name in ("old_count", "late_count", "short_length", "long_length", "old_output", "late_output",
                 "max_tokens", "request_count", "rounds", "max_seqs", "token_budget", "block_size", "cpu_threads"):
        if options[name] < 1:
            raise ValueError(f"{name} must be positive")
    if any(options[key] < 0 for key in ("seed", "offset", "warmup", "max_model_len")):
        raise ValueError("seed, offset, warmup and max_model_len must be nonnegative")
    if options["blocks"] != -1 and options["blocks"] < 1:
        raise ValueError("blocks must be -1 (automatic) or positive")
    for name in FLOATS:
        value = options[name]
        if value is None and name in ("ttft_slo_ms", "itl_slo_ms"):
            continue
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        if value < 0 or (value == 0 and name not in ("late_interval_s", "trim_fraction")):
            raise ValueError(f"invalid {name}")
    if options["gpu_memory_utilization"] > 1 or options["trim_fraction"] >= .5:
        raise ValueError("invalid memory utilization or trim fraction")
    if options["diagnostics_interval_s"] < .1:
        raise ValueError("diagnostics_interval_s must be >= 0.1")
    for name in BOOLS:
        if type(options[name]) is not bool:
            raise ValueError(f"{name} must be a JSON boolean")
    for name in ("model", "device", "dtype", "profile"):
        if not isinstance(options[name], str) or not options[name].strip():
            raise ValueError(f"{name} is required")
    for name in ("trace", "source", "dataset", "out"):
        if options[name] is not None and (not isinstance(options[name], str) or not options[name].strip()):
            raise ValueError(f"{name} must be a path or null")
    if options["trace"] and options["source"]:
        raise ValueError("choose --trace or --source, not both")
    if not options["trace"] and options["dataset"]:
        if options["source"]:
            raise ValueError("choose --dataset or --source, not both")
        if options["profile"] == "burst":
            raise ValueError("dataset uses original prompts: choose interval or poisson")
    if options["profile"] not in ("burst", "interval", "poisson"):
        raise ValueError("profile must be burst, interval or poisson")
    if options["dtype"] not in ("float32", "float16", "bfloat16"):
        raise ValueError("unsupported dtype")
    for name in ("policies", "prompt_lengths", "output_lengths"):
        values = options[name]
        if isinstance(values, str):
            values = values.split(",")
            values = [v.strip() for v in values] if name == "policies" else [int(v) for v in values]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{name} must be a nonempty list")
        if name == "policies":
            if any(p not in POLICIES for p in values) or len(set(values)) != len(values):
                raise ValueError("policies must contain unique known policies")
        elif any(type(n) is not int or n < 1 for n in values):
            raise ValueError(f"{name} must contain positive integers")
        options[name] = values
    if len(options["prompt_lengths"]) != len(options["output_lengths"]):
        raise ValueError("prompt_lengths and output_lengths must have equal length (paired profiles)")
    orders = options["policy_orders"]
    if isinstance(orders, str):
        orders = [[p.strip() for p in row.split(",")] for row in orders.split(";")]
    if orders is not None:
        if not isinstance(orders, list) or not orders:
            raise ValueError("policy_orders must be a nonempty list of permutations")
        for row in orders:
            if (not isinstance(row, list) or any(p not in POLICIES for p in row)
                    or len(set(row)) != len(row) or not set(options["policies"]) <= set(row)):
                raise ValueError("each policy order must include every selected policy once")
        options["policy_orders"] = [[p for p in row if p in options["policies"]] for row in orders]
    if options["out"] is None:
        options["out"] = str(APP_ROOT / "experiments/results/online" / datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    return options


def policy_order(options, index):
    if options["policy_orders"]:
        return options["policy_orders"][index % len(options["policy_orders"])]
    policies = options["policies"]
    shift = index % len(policies)
    return policies[shift:] + policies[:shift]


def fresh_policy(engine, policy):
    from nanovllm.engine.scheduler import Scheduler

    assert_idle(engine)
    manager = engine.model_runner.kv_cache_manager
    reset_cache(engine)
    config = engine.config
    engine.scheduler = Scheduler(manager, config.max_num_seqs, config.max_num_batched_tokens,
                                 config.max_model_len, scheduling_policy=policy)
    config.scheduling_policy = policy


def measure(engine, requests, options, sync, policy, *, monitor=None, label=None, warmup=False):
    import torch

    fresh_policy(engine, policy)
    device = engine.model_runner.device
    sync()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    result = replay(
        engine, requests, max(r["max_tokens"] for r in requests), "latency", sync, options["timeout_s"],
        ignore_eos=options["ignore_eos"], request_limits={r["id"]: r["max_tokens"] for r in requests},
        save_steps=options["save_details"] and not warmup, diagnostics=monitor, label=label,
    )
    result["summary"]["counts"].setdefault("preemptions", 0)
    result["summary"]["peak_cuda_memory_mb"] = (
        torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None)
    result["online"] = summarize_online(result, requests, window_s=options["window_s"],
                                        ttft_slo_ms=options["ttft_slo_ms"], itl_slo_ms=options["itl_slo_ms"])
    assert_idle(engine)
    return result


def preflight(options):
    trace = prepare_trace(options)
    hf = check_identity(trace, options["model"])
    requests = validate_trace(trace)
    required = max(len(r["prompt_token_ids"]) + r["max_tokens"] - 1 for r in requests)
    max_len = min(options["max_model_len"] or required, hf["max_position_embeddings"])
    validate_trace(trace, vocab_size=hf["vocab_size"], max_model_len=max_len,
                   block_size=options["block_size"], num_blocks=None if options["blocks"] == -1 else options["blocks"])
    if options["blocks"] > 0 and max_len > options["blocks"] * options["block_size"]:
        raise ValueError("max_model_len exceeds fixed KV capacity")
    if requests[-1]["arrival_s"] >= options["timeout_s"]:
        raise ValueError("last planned arrival is at/after replay timeout")
    outputs = sum(r["max_tokens"] for r in requests)
    inputs = sum(len(r["prompt_token_ids"]) for r in requests)
    if options["save_details"] and (len(requests) > 32 or outputs > 4096 or inputs > 100000):
        raise ValueError("save_details requires <=32 requests, <=4096 output and <=100000 input tokens")
    plan = {"requests": len(requests), "input_tokens": inputs, "max_output_tokens": outputs,
            "last_arrival_s": requests[-1]["arrival_s"], "max_model_len": max_len,
            "kv_blocks_upper_bound_without_sharing": sum(
                (len(r["prompt_token_ids"]) + r["max_tokens"] - 2 + options["block_size"]) // options["block_size"] for r in requests),
            "workload_sha256": trace["workload_sha256"],
            "formal_replays": options["rounds"] * len(options["policies"]),
            "warmup_replays": options["warmup"] * len(options["policies"])}
    return trace, hf, plan


def result_warnings(runs, trace, options):
    warnings = []
    for policy in options["policies"]:
        selected = [r for r in runs if r["policy"] == policy]
        if len({r["output_sha256"] for r in selected}) > 1:
            warnings.append(f"{policy} 的重复输出哈希不同：在线批次形状可随耗时变化，需另行检查数值一致性。")
        overlap = [r["online"]["overlap"] for r in selected]
        if overlap and overlap[0]["old_requests"] and overlap[0]["late_requests"]:
            bad = sum(o["old_decoding_at_arrival"] == 0 or o["old_decoding_at_admission"] == 0 for o in overlap)
            partial = sum(0 < o["old_decoding_at_arrival"] < o["old_requests"] for o in overlap)
            if bad:
                warnings.append(f"{policy}: {bad}/{len(selected)} 次没有形成已有 decode 与后来请求的有效重叠；先校准 delay_s。")
            if partial:
                warnings.append(f"{policy}: {partial} 次计划到达时只有部分 old 请求处于 decode；详见 overlap 计数。")
        if policy == "mixed" and any(r["summary"]["counts"].get("mixed_steps", 0) == 0 for r in selected):
            warnings.append("mixed 存在 mixed_steps=0 的运行，未覆盖合批机制；检查请求名额与到达时间。")
    if len({r["output_sha256"] for r in runs}) > 1:
        warnings.append("跨策略/重复的输出不完全一致；执行完成不代表输出等价。固定生成上限也不能证明答案质量相同。")
    if any(r["summary"]["counts"].get("preemptions", 0) for r in runs):
        warnings.append("发生了抢占，延迟差异同时包含资源接纳和重算影响。")
    if len(trace["requests"]) < 100:
        warnings.append("请求样本较少，p95/p99 常接近最大值；这适合机制试验，不代表稳定的服务尾延迟。")
    if options["save_details"]:
        warnings.append("本次保存逐轮明细，有观察开销；正式数据请关闭 save_details，并与诊断结果分开。")
    if options["diagnostics"]:
        warnings.append("本次开启主机/GPU 后台采样；与关闭采样的正式实验分开比较。")
    return warnings


def execute(options):
    trace, hf, plan = preflight(options)
    if options["dry_run"]:
        print(json.dumps({"plan": plan, "settings": options}, ensure_ascii=False, indent=2))
        return {"status": "planned", "plan": plan}
    out = Path(options["out"])
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("output directory must be new or empty (no implicit overwrite/resume)")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "workload.json", trace)
    write_json(out / "settings.json", options)
    requests = trace["requests"]
    engine, monitor = None, None
    runs = []
    try:
        import torch
        from nanovllm import LLM

        torch.set_num_threads(options["cpu_threads"])
        torch.manual_seed(options["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device(options["device"])
        if device.type not in ("cuda", "cpu"):
            raise ValueError("only cpu and cuda devices are supported")
        if device.type == "cuda":
            torch.cuda.set_device(device)
            device = torch.device("cuda", torch.cuda.current_device())
        sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
        print(f"Loading {options['model']}; requests={len(requests)} policy repeats={options['rounds']}", flush=True)
        engine = LLM(options["model"], device=str(device), dtype=options["dtype"],
                     max_num_seqs=options["max_seqs"], max_num_batched_tokens=options["token_budget"],
                     max_model_len=plan["max_model_len"], kvcache_block_size=options["block_size"],
                     num_kvcache_blocks=options["blocks"], gpu_memory_utilization=options["gpu_memory_utilization"],
                     enable_prefix_cache=options["prefix_cache"], scheduling_policy=options["policies"][0],
                     enable_tqdm=False, log_level="WARNING")
        manager = engine.model_runner.kv_cache_manager
        validate_trace(trace, vocab_size=hf["vocab_size"], max_model_len=engine.config.max_model_len,
                       block_size=manager.block_size, num_blocks=manager.num_blocks)
        environment = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                       "transformers": package_version("transformers"), "cuda": torch.version.cuda,
                       "cpu_threads": torch.get_num_threads(), "device": str(device), "tf32": False,
                       "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                       "num_kvcache_blocks": manager.num_blocks, "block_size": manager.block_size,
                       "max_model_len": engine.config.max_model_len,
                       "kv_capacity_tokens": manager.num_blocks * manager.block_size,
                       "kv_pool_mb": (manager.k_pool.numel() * manager.k_pool.element_size()
                                      + manager.v_pool.numel() * manager.v_pool.element_size()) / 2**20}
        if device.type == "cuda":
            props = torch.cuda.get_device_properties(device)
            environment.update(gpu_uuid=str(props.uuid), gpu_total_memory_mb=props.total_memory / 2**20)
        sources = sorted((APP_ROOT / "nanovllm").rglob("*.py")) + sorted((APP_ROOT / "experiments").rglob("*.py"))
        sources.append(APP_ROOT / "experiments/benchmark.py")
        source_hashes = {p.relative_to(APP_ROOT).as_posix(): file_hash(p) for p in sources
                         if "results" not in p.relative_to(APP_ROOT).parts and ".ipynb_checkpoints" not in p.parts}
        model = Path(options["model"])
        checkpoints = {p.name: {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                       for p in sorted(model.glob("*.safetensors"))}
        index = model / "model.safetensors.index.json"
        if index.exists():
            checkpoints[index.name] = {"sha256": file_hash(index)}
        print(f"KV blocks={manager.num_blocks}, capacity={environment['kv_capacity_tokens']}, "
              f"pool={environment['kv_pool_mb']:.1f}MiB; S={options['max_seqs']} T={options['token_budget']}", flush=True)
        if options["diagnostics"]:
            monitor = DiagnosticMonitor(options["diagnostics_interval_s"], torch, device)
            monitor.start()
        for policy in policy_order(options, 0):
            for i in range(options["warmup"]):
                measure(engine, requests, options, sync, policy, monitor=monitor,
                        label=f"warmup/{policy}/{i + 1}", warmup=True)
                print(f"warmup {policy} {i + 1}/{options['warmup']} complete", flush=True)
        for round_index in range(options["rounds"]):
            for position, policy in enumerate(policy_order(options, round_index), 1):
                label = f"round{round_index + 1}/{policy}"
                measured = measure(engine, requests, options, sync, policy, monitor=monitor, label=label)
                row = {"round": round_index + 1, "repeat": round_index + 1, "timing": "latency",
                       "policy_position": position, "policy": policy, "label": label,
                       "summary": measured["summary"], "online": measured["online"],
                       "requests": request_rows(measured, requests), "output_sha256": measured["output_sha256"]}
                if options["save_details"]:
                    row["details"] = {key: measured[key] for key in ("requests", "steps", "outputs")}
                runs.append(row)
                del measured
                print(metric_line(policy, row), flush=True)
        if monitor is not None:
            monitor.stop()
            for row in runs:
                row["summary"].update(monitor.summary(row["label"]))
        payload = {"schema": "online-results-v1", "status": "complete", "timing": "latency",
                   "settings": options, "plan": plan, "trace": trace, "environment": environment,
                   "source_sha256": source_hashes, "checkpoint_files": checkpoints,
                   "diagnostics": monitor.metadata() if monitor is not None else None,
                   "runs": runs, "summary": summarize_runs(runs, options["policies"], options["trim_fraction"]),
                   "matching_outputs": len({r["output_sha256"] for r in runs}) == 1,
                   "warnings": result_warnings(runs, trace, options)}
        save_report(out, payload)
        for warning in payload["warnings"]:
            print("NOTE:", warning, flush=True)
        print("Report:", out / "report.md", flush=True)
        return payload
    except BaseException as error:
        write_json(out / "failed.json", {"status": "failed", "error": repr(error), "settings": options,
                                         "workload_sha256": trace["workload_sha256"], "completed_runs": runs})
        raise
    finally:
        try:
            if monitor is not None:
                monitor.stop()
        finally:
            if engine is not None:
                atexit.unregister(engine.exit)
                engine.exit()


def main(argv=None):
    return execute(parse_options(argv))


if __name__ == "__main__":
    main()
