"""Run a real Math500 policy/max_tokens/max_seqs matrix, exporting JSON/CSV/MD."""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
from datetime import datetime
from itertools import product
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = APP_ROOT
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from experiments.benchmark import digest, distribution, write_json
from experiments.common.diagnostics import DIAGNOSTIC_FIELDS
from experiments.common.measurement import config_line, metric_line, run_worker
from experiments.common.repeat_stats import summarize_repeats


DEFAULTS = {
    "model": ("D:/models/Qwen3-0.6B" if sys.platform == "win32"
              else str(REPO_ROOT / "offline/Qwen3-0.6B")),
    "dataset": str(REPO_ROOT / "offline/MATH-500/test.jsonl"),
    "policies": ["separate", "interleaved", "mixed"], "max_tokens": [16, 32, 64],
    "max_seqs": [1, 4, 8], "token_budgets": [2048], "timings": ["latency"],
    "device": "cuda", "dtype": "float32", "num_problems": 0, "offset": 0,
    "block_size": 128, "blocks": -1, "max_model_len": 0, "prefix_cache": False,
    "gpu_memory_utilization": None,
    "diagnostics": False, "diagnostics_interval_s": 1.0,
    "ignore_eos": True, "warmup": 1, "repeats": 3, "cpu_threads": 1,
    "seed": 20261006, "shuffle": False, "arrival_interval_ms": 0.0,
    "timeout_s": 3600.0, "save_steps": False, "out": None,
    "worker_retries": 1,
    "repeat_schedule": "grouped", "policy_orders": None, "trim_fraction": 0.1,
}
POLICIES = ("separate", "interleaved", "mixed")
SCALARS = ("elapsed_s", "requests", "input_tokens", "output_tokens", "output_tok_s",
           "input_tok_s", "total_tok_s", "requests_s", "mixed_step_fraction",
           "token_weighted_tpot_ms", "peak_cuda_memory_mb", "num_kvcache_blocks",
           "kv_capacity_tokens", "kv_pool_mb") + DIAGNOSTIC_FIELDS
LATENCIES = ("ttft_ms", "tpot_ms", "itl_ms", "e2e_ms", "admission_delay_ms")
COUNTS = ("steps", "prefill_steps", "decode_steps", "mixed_steps", "query_tokens",
          "prefill_query_tokens", "decode_query_tokens", "kv_token_visits", "preemptions")
REPEAT_METRICS = ("elapsed_s", "output_tok_s", "input_tok_s", "total_tok_s", "requests_s",
                  "token_weighted_tpot_ms") + tuple(
    f"{name[:-3]}_{stat}_ms" for name in LATENCIES for stat in ("mean", "p50", "p95", "p99"))


def parse_policy_orders(value, policies):
    if value is None:
        return None
    if isinstance(value, str):
        value = [part.split(",") for part in value.split(";")]
    if not isinstance(value, list) or not value:
        raise ValueError("policy_orders must be a nonempty list of policy permutations")
    orders = []
    for row in value:
        if not isinstance(row, list) or not row or any(not isinstance(p, str) for p in row):
            raise ValueError("each policy_orders row must be a nonempty policy list")
        row = [p.strip() for p in row]
        if len(set(row)) != len(row) or any(p not in POLICIES for p in row) or set(policies) - set(row):
            raise ValueError("each policy_orders row must contain every selected policy exactly once")
        # CLI --policies can select a subset without rewriting the JSON orders.
        orders.append([p for p in row if p in policies])
    return orders


def list_values(value, *, integers=False):
    values = value.split(",") if isinstance(value, str) else value
    if isinstance(values, int) and not isinstance(values, bool):
        values = [values]
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("scan parameters must be nonempty lists or comma-separated values")
    result = []
    for item in values:
        if integers:
            if isinstance(item, str):
                item = int(item.strip())
            if type(item) is not int or item < 1:
                raise ValueError("scan values must be positive integers")
        else:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("scan values must be nonempty strings")
            item = item.strip()
        if item not in result:
            result.append(item)
    return result


def parse_options(argv=None):
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path)
    known, _ = preliminary.parse_known_args(argv)
    defaults = dict(DEFAULTS)
    if known.config:
        supplied = json.loads(known.config.read_text(encoding="utf-8-sig"))
        if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
            raise ValueError(f"unknown configuration keys: {set(supplied) - set(DEFAULTS) if isinstance(supplied, dict) else supplied}")
        defaults.update(supplied)
    parser = argparse.ArgumentParser(description=__doc__, parents=[preliminary])
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--policies", help="separate,interleaved,mixed")
    parser.add_argument("--max-tokens", help="per-request generation caps, e.g. 16,32,64")
    parser.add_argument("--max-seqs", "--max-num-seqs", dest="max_seqs", help="per-forward request limits")
    parser.add_argument("--token-budgets", "--token-budget", dest="token_budgets", help="max_num_batched_tokens")
    parser.add_argument("--timings", help="latency, throughput, or both")
    parser.add_argument("--repeat-schedule", choices=("grouped", "round_robin"))
    parser.add_argument("--policy-orders", help="policy orders separated by semicolons; policies separated by commas")
    parser.add_argument("--trim-fraction", type=float, help="fraction trimmed from each tail, default 0.1")
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"))
    for name in ("num_problems", "offset", "block_size", "blocks", "max_model_len",
                 "warmup", "repeats", "cpu_threads", "seed", "worker_retries"):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    for name in ("arrival_interval_ms", "timeout_s", "diagnostics_interval_s"):
        parser.add_argument("--" + name.replace("_", "-"), type=float)
    parser.add_argument("--gpu-memory-utilization", type=float, help="engine GPU memory fraction; default uses Config")
    for name in ("prefix_cache", "ignore_eos", "shuffle", "save_steps", "diagnostics"):
        parser.add_argument("--" + name.replace("_", "-"), action=argparse.BooleanOptionalAction)
    parser.add_argument("--out", help="directory for all artifacts")
    parser.add_argument("--dry-run", action="store_true", help="prepare prompts and inspect the matrix without loading weights")
    parser.add_argument("--resume", action="store_true", help="reuse completed matching cases from --out")
    parser.set_defaults(**defaults)
    options = vars(parser.parse_args(argv))
    for name in ("max_tokens", "max_seqs", "token_budgets"):
        options[name] = list_values(options[name], integers=True)
    for name in ("policies", "timings"):
        if name == "timings" and options[name] == "both":
            options[name] = ["throughput", "latency"]
        options[name] = list_values(options[name])
        permitted = POLICIES if name == "policies" else ("throughput", "latency")
        if any(v not in permitted for v in options[name]):
            raise ValueError(f"{name} must use {permitted}")
    options["policy_orders"] = parse_policy_orders(options["policy_orders"], options["policies"])
    if options["repeat_schedule"] not in ("grouped", "round_robin"):
        raise ValueError("repeat_schedule must be grouped or round_robin")
    trim = options["trim_fraction"]
    if type(trim) not in (float, int) or not math.isfinite(trim) or not 0 <= trim < .5:
        raise ValueError("trim_fraction must be finite and in [0, 0.5)")
    for name in ("num_problems", "offset", "max_model_len", "warmup", "worker_retries"):
        if type(options[name]) is not int or options[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    for name in ("block_size", "repeats", "cpu_threads"):
        if type(options[name]) is not int or options[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(options["blocks"]) is not int or (options["blocks"] != -1 and options["blocks"] <= 0):
        raise ValueError("blocks must be -1 for engine auto allocation or a positive integer")
    utilization = options["gpu_memory_utilization"]
    if utilization is not None and (type(utilization) not in (float, int)
            or not math.isfinite(utilization) or not 0 < utilization <= 1):
        raise ValueError("gpu_memory_utilization must be null or a finite value in (0, 1]")
    for name in ("arrival_interval_ms", "timeout_s"):
        value = options[name]
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0 or (name == "timeout_s" and value == 0):
            raise ValueError(f"{name} must be finite and {'positive' if name == 'timeout_s' else 'nonnegative'}")
    interval = options["diagnostics_interval_s"]
    if type(interval) not in (float, int) or not math.isfinite(interval) or interval < .1:
        raise ValueError("diagnostics_interval_s must be finite and at least 0.1 seconds")
    for name in ("prefix_cache", "ignore_eos", "shuffle", "save_steps", "diagnostics"):
        if type(options[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    if options["save_steps"]:
        raise ValueError("save_steps=true is no longer supported; experiment output contains metrics only")
    if type(options["seed"]) is not int:
        raise ValueError("seed must be an integer")
    if options["dtype"] not in ("float32", "float16", "bfloat16"):
        raise ValueError("unsupported dtype")
    for name in ("model", "dataset"):
        if not isinstance(options[name], str) or not options[name].strip():
            raise ValueError(f"{name} must be a nonempty local path")
        options[name] = str(Path(options[name]).resolve())
    if options["device"] != "cpu" and not (isinstance(options["device"], str)
            and (options["device"] == "cuda" or (options["device"].startswith("cuda:")
                                                 and options["device"][5:].isdigit()))):
        raise ValueError("device must be cpu, cuda, or cuda:N")
    if options["resume"] and options["out"] is None:
        raise ValueError("--resume requires --out")
    if options["out"] is not None and (not isinstance(options["out"], str) or not options["out"].strip()):
        raise ValueError("out must be a nonempty path or null")
    options["out"] = str(Path(options["out"]).resolve()) if options["out"] else str(
        APP_ROOT / "experiments/results/math500" / datetime.now().strftime("math500-%Y%m%d-%H%M%S-%f"))
    options.pop("config")
    return options


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode_problem(tokenizer, problem):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": problem}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    return tokenizer.encode(rendered)


def prepare_workload(options):
    from transformers import AutoConfig, AutoTokenizer

    dataset, model = Path(options["dataset"]), Path(options["model"])
    if options["dry_run"]:
        # Planning still supports a local config/tokenizer without weights.
        checkpoint_paths = sorted(model.glob("*.safetensors"))
        if (model / "model.safetensors.index.json").is_file():
            checkpoint_paths.append(model / "model.safetensors.index.json")
    else:
        from nanovllm.utils.loader import resolve_checkpoint_files

        checkpoint_paths, weight_map = resolve_checkpoint_files(model)
        if weight_map is not None:
            checkpoint_paths.append(model / "model.safetensors.index.json")
    checkpoint_files = {
        path.relative_to(model).as_posix(): {
            "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
        } for path in checkpoint_paths
    }
    if "model.safetensors.index.json" in checkpoint_files:
        # The index decides where each tensor comes from; a changed mapping
        # must invalidate resume even when the shard files are unchanged.
        checkpoint_files["model.safetensors.index.json"]["sha256"] = sha256(
            model / "model.safetensors.index.json")
    config = AutoConfig.from_pretrained(model, local_files_only=True)
    if config.model_type != "qwen3":
        raise ValueError("only local Qwen3 checkpoints are supported")
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    selected = list(enumerate(rows))[options["offset"]:]
    if options["num_problems"]:
        selected = selected[:options["num_problems"]]
    if not selected:
        raise ValueError("selected Math500 cohort is empty")
    if options["shuffle"]:
        random.Random(options["seed"]).shuffle(selected)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    requests = []
    for order, (index, row) in enumerate(selected):
        problem = row.get("problem")
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"dataset row {index} has no nonempty problem")
        tokens = encode_problem(tokenizer, problem)
        if not tokens or any(type(t) is not int or not 0 <= t < config.vocab_size for t in tokens):
            raise ValueError(f"dataset row {index}: invalid prompt token IDs")
        requests.append({"id": str(row.get("unique_id", index)), "dataset_index": index,
                         "problem": problem, "subject": row.get("subject"), "level": row.get("level"),
                         "arrival_s": order * options["arrival_interval_ms"] / 1000,
                         "prompt_token_ids": tokens})
    if len({r["id"] for r in requests}) != len(requests):
        raise ValueError("selected dataset request IDs are not unique")
    pool_capacity = options["block_size"] * options["blocks"] if options["blocks"] > 0 else None
    if pool_capacity is not None and options["max_model_len"] > pool_capacity:
        raise ValueError("max_model_len exceeds the KV pool; reduce it or increase --blocks")
    required = max(len(r["prompt_token_ids"]) for r in requests) + max(options["max_tokens"]) - 1
    max_len = min(options["max_model_len"] or required, config.max_position_embeddings)
    if pool_capacity is not None:
        max_len = min(max_len, pool_capacity)
    workload = {"schema": 1, "requests": requests, "workload_sha256": digest(requests),
                "dataset_sha256": sha256(dataset), "selected_problem_count": len(requests),
                "prompt_lengths": distribution([len(r["prompt_token_ids"]) for r in requests]),
                "template": "Qwen user chat, enable_thinking=False, render then encode",
                "max_model_len": max_len, "model_position_capacity": config.max_position_embeddings,
                "kv_allocation": "auto" if options["blocks"] == -1 else "manual",
                "planned_kv_capacity_tokens": pool_capacity,
                "checkpoint_files": checkpoint_files,
                "model_config_sha256": sha256(model / "config.json"),
                "tokenizer_sha256": {p.name: sha256(p) for p in sorted(model.glob("*token*.json"))}}
    return workload


def prepare_workload_process(options):
    """Exit the tokenizer/torch process before starting a model worker.

    Keeping the controller in stdlib avoids a second PyTorch/CUDA DLL runtime
    consuming Windows system commit while the checkpoint is being loaded.
    """
    out = Path(options["out"])
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=out) as temporary:
        options_path, workload_path = Path(temporary) / "options.json", Path(temporary) / "workload.json"
        write_json(options_path, options)
        completed = subprocess.run(
            [sys.executable, "-X", "utf8", str(Path(__file__).resolve()),
             "--prepare-workload", str(options_path), str(workload_path)],
            cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300,
        )
        if completed.returncode != 0:
            raise ValueError(f"workload preparation exited {completed.returncode}: {completed.stderr[-6000:]}")
        return json.loads(workload_path.read_text(encoding="utf-8"))


def build_jobs(options, workload):
    out = Path(options["out"])
    # Hash this runner and its shared dependencies after the package split.
    # Future online-only edits must not invalidate a Math500 resume.
    sources = (sorted((APP_ROOT / "nanovllm").rglob("*.py"))
               + sorted((APP_ROOT / "experiments/common").rglob("*.py"))
               + sorted((APP_ROOT / "experiments/math500").rglob("*.py"))
               + [APP_ROOT / "experiments/__init__.py", APP_ROOT / "experiments/benchmark.py"])
    source_hashes = {p.relative_to(APP_ROOT).as_posix(): sha256(p) for p in sources}
    settings = {k: v for k, v in options.items() if k not in ("out", "resume", "dry_run")}
    jobs = []
    for index, (cap, seqs, budget) in enumerate(product(options["max_tokens"], options["max_seqs"], options["token_budgets"])):
        policies = options["policies"]
        shift = index % len(policies)
        orders = options["policy_orders"]
        if options["repeat_schedule"] == "round_robin":
            visits = []
            for repeat in range(1, options["repeats"] + 1):
                order = orders[(repeat - 1) % len(orders)] if orders else (
                    policies[(repeat - 1) % len(policies):] + policies[:(repeat - 1) % len(policies)])
                visits.extend((repeat, position, policy) for position, policy in enumerate(order, 1))
        else:
            order = orders[index % len(orders)] if orders else policies[shift:] + policies[:shift]
            visits = [(None, position, policy) for position, policy in enumerate(order, 1)]
        for repeat, position, policy in visits:
            factors = {"policy": policy, "max_tokens": cap, "max_seqs": seqs, "token_budget": budget}
            group_id = f"{policy}-g{cap}-s{seqs}-t{budget}"
            case_id = group_id if repeat is None else f"{group_id}-r{repeat:03d}"
            config = {"device": options["device"], "dtype": options["dtype"],
                      "scheduling_policy": policy, "max_num_seqs": seqs, "max_num_batched_tokens": budget,
                      "max_model_len": workload["max_model_len"], "kvcache_block_size": options["block_size"],
                      "enable_prefix_cache": options["prefix_cache"],
                      "enable_tqdm": False, "log_level": "WARNING"}
            # Omitting the override uses Config.num_kvcache_blocks=-1. Only the
            # real ModelRunner can profile the loaded model and size its KV pool.
            if options["blocks"] > 0:
                config["num_kvcache_blocks"] = options["blocks"]
            if options["gpu_memory_utilization"] is not None:
                config["gpu_memory_utilization"] = options["gpu_memory_utilization"]
            invalid = [r for r in workload["requests"]
                       if len(r["prompt_token_ids"]) + cap - 1 > workload["max_model_len"]]
            error = (f"{len(invalid)} requests exceed the configured/model sequence limit: "
                     f"{invalid[0]['id']} needs {len(invalid[0]['prompt_token_ids']) + cap - 1}, "
                     f"limit {workload['max_model_len']}; cohort was not filtered") if invalid else None
            worker_settings = settings if repeat is None else dict(settings, repeats=1)
            job = {"schema": 1, "case_id": case_id, "group_id": group_id,
                   "round": repeat, "round_count": options["repeats"] if repeat is not None else None,
                   "policy_position": position, "factors": factors, "settings": worker_settings,
                   "engine_config": config, "workload_sha256": workload["workload_sha256"],
                   "request_count": len(workload["requests"]),
                   "workload_path": str(out / "workload.json"), "result_path": str(out / "cases" / f"{case_id}.json"),
                   "preflight_error": error}
            if repeat is not None:
                job["shared_worker_id"] = f"g{cap}-s{seqs}-t{budget}-shared"
            job["fingerprint"] = digest({"settings": settings, "factors": factors, "config": config,
                                         "round": repeat, "policy_position": position,
                                         "workload": workload["workload_sha256"], "sources": source_hashes,
                                         "model_config": workload["model_config_sha256"],
                                         "checkpoint": workload["checkpoint_files"],
                                         "tokenizer": workload["tokenizer_sha256"]})
            jobs.append(job)
    return jobs, source_hashes


def execution_units(jobs):
    """Bundle all rounds of one G/S/T cell into one resident model worker."""
    units = []
    shared = {}
    for job in jobs:
        worker_id = job.get("shared_worker_id")
        if worker_id is None:
            units.append(job)
        elif worker_id not in shared:
            unit = dict(job, case_id=worker_id, round=None,
                        settings=dict(job["settings"], repeats=job["round_count"]), members=[])
            units.append(unit)
            shared[worker_id] = unit
        if worker_id is not None:
            shared[worker_id]["members"].append(job)
    return units


def terminate_worker_tree(process):
    """Stop only our worker tree; POSIX workers own a newly created session."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass  # The entire owned process group has already exited.
    return process.wait()


def run_worker_process(job, job_path, log_path):
    members = job.get("members", [job])
    warmups = job["settings"]["warmup"] * len({m["factors"]["policy"] for m in members}) if "members" in job else job["settings"]["warmup"]
    measurements = sum(m["settings"]["repeats"] * len(m["settings"]["timings"]) for m in members)
    timeout = 300 + job["settings"]["timeout_s"] * (warmups + measurements)
    attempts = []
    for attempt in range(job["settings"]["worker_retries"] + 1):
        path = log_path if attempt == 0 else log_path.with_name(f"{log_path.stem}-retry{attempt}{log_path.suffix}")
        # Drain the pipe while the child runs, showing only configuration and
        # metrics. Buffer the small log in RAM and write once after child exit.
        lines = []
        prefixes = (f"{job['case_id']}: config ",) + tuple(
            prefix for member in members for prefix in (
                f"{member['case_id']}: config ", f"{member['case_id']} throughput repeat ",
                f"{member['case_id']} latency repeat ", f"{member['case_id']}: status="))
        process = subprocess.Popen([sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve()),
                                    "--worker-job", str(job_path)], cwd=REPO_ROOT,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace",
                                   start_new_session=sys.platform != "win32")

        def read_output():
            for line in process.stdout:
                lines.append(line)
                if line.startswith(prefixes):
                    print(line, end="", flush=True)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        interrupted = None
        try:
            returncode = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            # Also terminate NVIDIA samplers and other descendants. Killing
            # only the POSIX worker would leave children holding this pipe open.
            returncode = terminate_worker_tree(process)
            interrupted = exc
        finally:
            if sys.platform != "win32" and interrupted is None:
                # A native worker crash may skip its finally/monitor.stop.
                # Clean up descendants before waiting for pipe EOF, even when
                # the worker itself has already exited.
                terminate_worker_tree(process)
            reader.join()
            process.stdout.close()
            path.write_text("".join(line for line in lines if line.startswith(prefixes)), encoding="utf-8")
        attempts.append({"attempt": attempt + 1, "exit_code": returncode, "log": str(path)})
        if interrupted is not None:
            interrupted.worker_attempts = attempts
            raise interrupted
        if returncode == 0:
            return attempts
        # A Python/model error or a death after initialization is a failed case.
        # Only Windows native exits before model loading completed are retried.
        initialization_exit = (sys.platform == "win32" and returncode >= 0x80000000
                               and not any(line.startswith(prefixes[0]) for line in lines))
        if initialization_exit and attempt < job["settings"]["worker_retries"]:
            continue
        detail = next((line.strip() for line in reversed(lines) if line.strip()), "no Python error message")
        error = RuntimeError(f"worker exited {returncode}: {detail[:500]}; see {path}")
        error.worker_attempts = attempts
        raise error


def flatten_summary(summary):
    values = {name: summary.get(name) for name in SCALARS}
    values.update({name: summary["counts"].get(name, 0) for name in COUNTS})
    for name in LATENCIES:
        for stat in ("n", "mean", "p50", "p95", "p99"):
            values[f"{name[:-3]}_{stat}_ms" if stat != "n" else f"{name[:-3]}_samples"] = summary[name][stat]
    return values


def make_report(options, workload, jobs, results):
    rows, runs, groups = [], [], {}
    case_jobs = {}
    for job in jobs:
        case_jobs.setdefault(job.get("group_id", job["case_id"]), []).append(job)
    for group_id, members in case_jobs.items():
        job = members[0]
        completed = [(member, results.get(member["case_id"], {"status": "planned", "runs": []}))
                     for member in members]
        statuses = [result["status"] for _, result in completed]
        hashes = {r["output_sha256"] for _, result in completed for r in result["runs"]}
        if "failed" in statuses:
            status = "failed"
        elif "infeasible" in statuses:
            status = "infeasible"
        elif "unstable" in statuses or len(hashes) > 1:
            status = "unstable"
        elif all(s == "ok" for s in statuses):
            status = "ok"
        elif all(s == "planned" for s in statuses):
            status = "planned"
        else:
            status = "partial"
        error = next((result.get("error") for _, result in completed if result.get("error")), None)
        for timing in options["timings"]:
            selected_pairs = sorted(((member, r) for member, result in completed
                                     for r in result["runs"] if r["timing"] == timing),
                                    key=lambda pair: pair[1]["repeat"])
            selected = [r for _, r in selected_pairs]
            row = {"case_id": group_id, **job["factors"], "timing": timing,
                   "status": status, "repeats_completed": len(selected),
                   "repeats_expected": options["repeats"], "trim_fraction": options["trim_fraction"], "error": error}
            if selected:
                flattened = [flatten_summary(r["summary"]) for r in selected]
                for key in flattened[0]:
                    values = [r[key] for r in flattened if r[key] is not None]
                    row[key] = statistics.median(values) if values else None
                for metric in REPEAT_METRICS:
                    for stat, value in summarize_repeats([r.get(metric) for r in flattened], options["trim_fraction"]).items():
                        row[f"{metric}_{stat}"] = value
                row["output_sha256"] = selected[0]["output_sha256"]
                for member, run in selected_pairs:
                    runs.append({"case_id": group_id, **job["factors"], "timing": timing,
                                 "status": status, "repeat": run["repeat"],
                                 "round": member.get("round"), "policy_position": member.get("policy_position"),
                                 "worker_case_id": member["case_id"],
                                 "worker_id": member.get("shared_worker_id", member["case_id"]),
                                 **flatten_summary(run["summary"]), "output_sha256": run["output_sha256"]})
            if selected:
                entries = [(group_id, r) for r in selected]
                groups.setdefault((job["factors"]["max_tokens"], timing), []).extend(entries)
                if len(options["timings"]) > 1:
                    groups.setdefault((job["factors"]["max_tokens"], "all"), []).extend(entries)
            rows.append(row)
    comparisons = []
    for (cap, timing), group in groups.items():
        hashes = {r["output_sha256"] for _, r in group}
        expected_cases = sum(m[0]["factors"]["max_tokens"] == cap for m in case_jobs.values())
        comparison = {"max_tokens": cap, "timing": timing, "matching_outputs": len(hashes) == 1,
                      "case_count": len({cid for cid, _ in group}), "runs_checked": len(group),
                      "expected_case_count": expected_cases,
                      "first_mismatch": None}
        comparison["complete"] = (comparison["case_count"] == expected_cases
                                  and len(group) == expected_cases * options["repeats"]
                                  * (len(options["timings"]) if timing == "all" else 1))
        base_id, base = group[0]
        if len(hashes) > 1:
            other_id, other = next((cid, r) for cid, r in group if r["output_sha256"] != base["output_sha256"])
            comparison["first_mismatch"] = {"reference_case": base_id, "actual_case": other_id,
                                             "reference_timing": base["timing"], "reference_repeat": base["repeat"],
                                             "actual_timing": other["timing"], "actual_repeat": other["repeat"],
                                             "reference_sha256": base["output_sha256"],
                                             "actual_sha256": other["output_sha256"]}
        comparisons.append(comparison)
    lookup = {(c["max_tokens"], c["timing"]): c["matching_outputs"] for c in comparisons}
    for row in rows:
        row["matching_outputs"] = (lookup.get((row["max_tokens"], "all"),
                                              lookup.get((row["max_tokens"], row["timing"])))
                                   if row["repeats_completed"] else None)
    return {"schema": 2, "settings": options, "workload_sha256": workload["workload_sha256"],
            "selected_problem_count": workload["selected_problem_count"], "prompt_lengths": workload["prompt_lengths"],
            "summary": rows, "runs": runs, "output_comparisons": comparisons}


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r)) or [
        "case_id", "policy", "max_tokens", "max_seqs", "token_budget", "timing", "status", "repeat"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_value(value):
    if value is None:
        return "-"
    if type(value) is bool:
        return "yes" if value else "no"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def validate_result(job, result):
    if (result.get("schema") != 2 or result.get("fingerprint") != job["fingerprint"]
            or result.get("status") not in ("ok", "unstable")
            or len(result.get("runs", [])) != job["settings"]["repeats"] * len(job["settings"]["timings"])):
        raise ValueError("worker did not return all matching measurements")
    for timing in job["settings"]["timings"]:
        runs = [r for r in result["runs"] if r["timing"] == timing]
        expected = [job["round"]] if job.get("round") is not None else list(range(1, job["settings"]["repeats"] + 1))
        if sorted(r["repeat"] for r in runs) != expected:
            raise ValueError("missing or duplicate repeat")
        for run in runs:
            summary, output_hash = run["summary"], run["output_sha256"]
            count, cap = job["request_count"], job["factors"]["max_tokens"]
            if (summary["requests"] != count or not count <= summary["output_tokens"] <= count * cap
                    or (job["settings"]["ignore_eos"] and summary["output_tokens"] != count * cap)
                    or not isinstance(output_hash, str) or len(output_hash) != 64
                    or any(c not in "0123456789abcdef" for c in output_hash)):
                raise ValueError("missing or corrupted completion counts/hash")
            if any(key in run for key in ("outputs", "output_texts", "requests", "steps")):
                raise ValueError("worker result must contain aggregate metrics only")
    stable = len({r["output_sha256"] for r in result["runs"]}) == 1
    if (result["status"] == "ok") != stable:
        raise ValueError("worker status disagrees with output hashes")


def save_report(out, report):
    write_json(out / "results.json", report)
    write_json(out / "output_equivalence.json", report["output_comparisons"])
    write_csv(out / "summary.csv", report["summary"])
    write_csv(out / "runs.csv", report["runs"])
    lines = ["# Math500 实际推理实验", "", f"请求数：{report['selected_problem_count']}；"
             "summary.csv 保留原有中位数字段，并增加跨重复的统计字段；runs.csv 保留全部原始测量。", "",
             f"执行模式：{report['settings']['repeat_schedule']}；每策略正式重复 {report['settings']['repeats']} 次。"
             "按轮模式每组 G/S/T 只加载一次模型，每策略在正式测量前预热一次阶段；正式 repeat 之间不重复预热。",
             "配置的策略顺序循环：" + ("；".join(" → ".join(order) for order in report['settings']['policy_orders'])
                                       if report['settings']['policy_orders'] else "默认轮换所选策略"),
             f"截尾比例：每端 {report['settings']['trim_fraction']:.1%}，每端排除 floor(N×比例) 个值；"
             "只影响截尾均值计算，标准差、CV 和极值使用全部有效 repeat。N<2 时样本标准差/CV 留空。", "",
             "max_tokens=生成上限；max_seqs=每轮请求上限；token_budget=每轮 query token 上限。",
             "TPOT 为每请求相邻输出 token 间隔的平均值；单输出 token 为缺失值。"
             "TTFT/E2E 从计划到达算起，含排队；latency 模式每轮同步 CUDA。"
             "throughput 模式不记录 token 时刻，延迟列为空，两种 timing 分开比较。", "",
             "| 策略 | max_tokens | max_seqs | token_budget | timing | 状态 | KV块数 | 输出 tok/s | TTFT mean ms | TPOT mean ms | ITL p95 ms | mixed比例 | 抢占 | 输出一致 |",
             "|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in report["summary"]:
        values = [row.get(k) for k in ("policy", "max_tokens", "max_seqs", "token_budget", "timing", "status",
                                       "num_kvcache_blocks", "output_tok_s", "ttft_mean_ms", "tpot_mean_ms", "itl_p95_ms",
                                       "mixed_step_fraction", "preemptions", "matching_outputs")]
        lines.append("| " + " | ".join(format_value(v) for v in values) + " |")
    lines += ["", "## 输出吞吐的重复测量统计", "",
              "单位 tok/s；标准差为样本标准差（N−1），CV=标准差/均值×100%。保留全部原始 repeat。", "",
              "| 组合 | timing | 完成/计划 | 均值 | 截尾均值 | 中位数 | 标准差 | CV % | 最小值 | 最大值 | 每端截尾数 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["summary"]:
        values = [row["case_id"], row["timing"], f"{row['repeats_completed']}/{row['repeats_expected']}"]
        values += [row.get(f"output_tok_s_{stat}") for stat in (
            "mean", "trimmed_mean", "median", "std", "cv_pct", "min", "max", "trimmed_each_tail")]
        lines.append("| " + " | ".join(format_value(v) for v in values) + " |")
    latency_rows = [(row, metric) for row in report["summary"]
                    for metric in ("ttft_mean_ms", "tpot_mean_ms", "itl_p95_ms", "e2e_mean_ms")
                    if row.get(f"{metric}_n", 0)]
    if latency_rows:
        lines += ["", "## 延迟指标的重复测量统计", "",
                  "对每次 replay 的请求均值或 ITL p95 做跨重复统计，单位 ms；不合并请求分布。", "",
                  "| 组合 | 指标 | N | 均值 | 截尾均值 | 中位数 | 标准差 | CV % | 最小值 | 最大值 | 每端截尾数 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row, metric in latency_rows:
            values = [row["case_id"], metric] + [row.get(f"{metric}_{stat}") for stat in (
                "n", "mean", "trimmed_mean", "median", "std", "cv_pct", "min", "max", "trimmed_each_tail")]
            lines.append("| " + " | ".join(format_value(v) for v in values) + " |")
    allocation = ("KV 块数由引擎自动推算；实际块数、容量和池大小见结果。"
                  if report["settings"]["blocks"] == -1 else "本次显式指定了 KV 块数。")
    lines += ["", allocation + "每次测量前清空前缀元数据和空闲块顺序。模型加载、分词、引擎 profiling、预热和写盘不计入耗时。",
              "结果只保存配置、汇总指标和输出一致性哈希，不解码或保存生成文本、token IDs、逐请求记录或逐步记录。测量期间不写结果和日志；worker 结束后保存 case，实验结束后生成汇总。",
              "生成使用 greedy；ignore_eos=true 时固定生成到上限，false 时按真实 EOS 结束。",
              "不筛除超过容量的题目；相关组合标为 infeasible。未完成组合的部分测量仅供诊断，不能当作完整成功结果。",
              "Math500 标准答案没有用作模型输入，本实验不评价解题正确率。",
              "输出不一致或 mixed 比例为 0 的组合，应查看 output_equivalence.json 和 case 汇总指标。"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(options, *, prepare=prepare_workload):
    out = Path(options["out"])
    if (out / "manifest.json").exists() and not options["resume"]:
        raise ValueError("output already contains an experiment; use --resume or a new --out")
    workload = prepare(options)
    jobs, source_hashes = build_jobs(options, workload)
    identity = digest([job["fingerprint"] for job in jobs])
    if options["resume"] and (out / "manifest.json").exists():
        previous = json.loads((out / "manifest.json").read_text(encoding="utf-8-sig"))
        if previous["identity"] != identity:
            raise ValueError("resume configuration/data/checkpoint/source differs from saved experiment")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("jobs", "cases", "logs"):
        (out / name).mkdir(exist_ok=True)
    write_json(out / "workload.json", workload)
    write_json(out / "manifest.json", {"schema": 1, "identity": identity, "settings": options,
                                       "source_sha256": source_hashes, "jobs": jobs})
    results = {}
    try:
        for unit in execution_units(jobs):
            members = unit.get("members", [unit])
            if options["resume"]:
                try:
                    cached_results = []
                    for job in members:
                        cached = json.loads(Path(job["result_path"]).read_text(encoding="utf-8-sig"))
                        if cached.get("status") != "ok":
                            raise ValueError("worker group is not complete")
                        validate_result(job, cached)
                        cached_results.append(cached)
                    # A resident worker is the recovery unit. Never mix old
                    # rounds with a rerun that has a different warmup phase.
                    for job, cached in zip(members, cached_results):
                        results[job["case_id"]] = cached
                        print(config_line(job, cached["config"], cached["cache"]), flush=True)
                        for run in cached["runs"]:
                            print(metric_line(job["case_id"], run), flush=True)
                        print(f"{job['case_id']}: status=reused matching_outputs=True", flush=True)
                    continue
                except (OSError, ValueError, KeyError):
                    pass  # A missing/invalid round causes the entire cell to rerun.
            if unit["preflight_error"]:
                for job in members:
                    result = {"case_id": job["case_id"], "fingerprint": job["fingerprint"],
                              "status": "infeasible", "error": job["preflight_error"], "runs": []}
                    write_json(Path(job["result_path"]), result)
                    results[job["case_id"]] = result
                    print(config_line(job, job["engine_config"]), flush=True)
                    print(f"{job['case_id']}: status=infeasible error={job['preflight_error']}", flush=True)
            elif not options["dry_run"]:
                job_path = out / "jobs" / f"{unit['case_id']}.json"
                write_json(job_path, unit)
                log_path = out / "logs" / f"{unit['case_id']}.log"
                try:
                    attempts = run_worker_process(unit, job_path, log_path)
                    completed = []
                    for job in members:
                        result = json.loads(Path(job["result_path"]).read_text(encoding="utf-8-sig"))
                        validate_result(job, result)
                        result["worker_attempts"] = attempts or []
                        completed.append(result)
                except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    completed = [{"case_id": job["case_id"], "fingerprint": job["fingerprint"],
                                  "status": "failed", "error": str(exc), "runs": [],
                                  "worker_attempts": getattr(exc, "worker_attempts", [])}
                                 for job in members]
                    detail = str(exc).splitlines()[0][:500] if str(exc) else type(exc).__name__
                    print(f"{unit['case_id']}: status=failed error={detail}", flush=True)
                for job, result in zip(members, completed):
                    write_json(Path(job["result_path"]), result)
                    results[job["case_id"]] = result
            else:
                for job in members:
                    print(config_line(job, job["engine_config"]), flush=True)
    except KeyboardInterrupt:
        save_report(out, make_report(options, workload, jobs, results))
        raise
    report = make_report(options, workload, jobs, results)
    save_report(out, report)
    print(f"Results: {out / 'summary.csv'}; repeats: {out / 'runs.csv'}", flush=True)
    bad = any(r["status"] not in ("ok", "planned") for r in report["summary"])
    mismatch = any(not c["matching_outputs"] for c in report["output_comparisons"])
    return report, 1 if bad or mismatch else 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 3 and argv[0] == "--prepare-workload":
        try:
            options = json.loads(Path(argv[1]).read_text(encoding="utf-8-sig"))
            write_json(Path(argv[2]), prepare_workload(options))
            return 0
        except (OSError, ValueError) as exc:
            print(f"Workload error: {exc}", file=sys.stderr)
            return 2
    if len(argv) == 2 and argv[0] == "--worker-job":
        job = json.loads(Path(argv[1]).read_text(encoding="utf-8-sig"))
        try:
            result = run_worker(job)
            # Preserve unstable measurements and output hashes. The parent
            # reports them and returns a failing exit code for the experiment.
            return 0 if result["status"] in ("ok", "unstable") else 1
        except Exception as exc:
            write_json(Path(job["result_path"]), {"case_id": job["case_id"], "fingerprint": job["fingerprint"],
                                                  "status": "failed", "error": str(exc), "runs": []})
            detail = str(exc).splitlines()[0][:500] if str(exc) else type(exc).__name__
            print(f"{job['case_id']}: status=failed error={detail}", file=sys.stderr, flush=True)
            return 1
    try:
        _, code = execute(parse_options(argv), prepare=prepare_workload_process)
        return code
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"Experiment error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; completed cases are saved. Use the same parameters with --resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
