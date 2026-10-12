"""Unpack-and-run entry point for the reproducible scheduling experiments."""
import argparse
import importlib.metadata as metadata
import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from audit import audit, read
from block_design import DEFAULT_BLOCK_IDS, parse_block_ids
from verify_package import verify

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
APP = ROOT
LOG = None


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(command, env=None):
    print("RUN", subprocess.list2cmdline([str(x) for x in command]), flush=True)
    with subprocess.Popen([str(x) for x in command], cwd=ROOT, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
            if LOG:
                LOG.write(line)
                LOG.flush()
        status = process.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)


def dependencies(no_install=False):
    required = {"torch": "torch==2.7.1", "transformers": "transformers>=4.51,<5",
                "pyarrow": "pyarrow", "safetensors": "safetensors", "tqdm": "tqdm"}
    missing = []
    for name in required:
        try:
            metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
    if missing and no_install:
        raise ValueError("Missing dependencies: " + ", ".join(missing) +
                         ". Install requirements.txt, or omit --no-install.")
    if "torch" in missing:
        run([sys.executable, "-m", "pip", "install", required["torch"],
             "--index-url", "https://download.pytorch.org/whl/cu128"])
    packages = [required[name] for name in missing if name != "torch"]
    if packages:
        run([sys.executable, "-m", "pip", "install", *packages])
    version = metadata.version("transformers").split(".")
    if int(version[0]) != 4 or int(version[1]) < 51:
        raise ValueError("Use Transformers >=4.51,<5 in the active environment; existing packages were not replaced")


def environment(model, gpu):
    # Probe in a short-lived process: the orchestration parent must not retain a
    # CUDA context or consume device memory while measured children are running.
    probe = """
import importlib.metadata as metadata
import json
import sys
import torch
info = {'python': sys.version, 'model': sys.argv[1], 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'transformers': metadata.version('transformers'),
        'pyarrow': metadata.version('pyarrow')}
if sys.argv[2] == '1':
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError('A CUDA GPU with bf16 support is required; inspect the driver and active PyTorch build')
    info.update(gpu=torch.cuda.get_device_name(0),
                gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory)
print(json.dumps(info))
"""
    result = subprocess.run([sys.executable, "-c", probe, str(model), str(int(gpu))],
                            capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        raise ValueError("Environment check failed:\n" + result.stderr)
    return json.loads(result.stdout)


def main():
    global LOG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/root/model/Qwen3-4B"))
    parser.add_argument("--stage", choices=["check", "prepare", "pilot", "scan", "burst", "boundary", "online", "analyze"], default="scan")
    parser.add_argument("--plan", type=Path, help="Reuse a frozen plan for subsequent stages")
    parser.add_argument("--out", type=Path, help="A new stage output directory")
    parser.add_argument("--budgets", help="Comma-separated budgets; burst requires an explicit selection")
    parser.add_argument("--block-ids", default=DEFAULT_BLOCK_IDS,
                        help="Stable block IDs; balanced three-block design: 0,3,4 (default: all six)")
    parser.add_argument("--reuse-from", type=Path,
                        help="Stopped prior stage/formal directory; copy matching complete calls into new output")
    parser.add_argument("--delay", type=float, default=None)
    parser.add_argument("--kv-blocks", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--window", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--rates", help="Online: calibrated common arrival rates in req/s")
    parser.add_argument("--arrival-seeds", default="20261011,20261012,20261013")
    parser.add_argument("--dry-run", action="store_true", help="Prepare and preflight only; no model inference")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--no-install", action="store_true", help="Fail on missing dependencies instead of installing")
    parser.add_argument("--results", type=Path, help="Analyze: path to a complete formal matrix")
    parser.add_argument("--mode", choices=["burst", "boundary", "stream"], default="burst")
    args = parser.parse_args()
    try:
        block_ids = parse_block_ids(args.block_ids)
    except ValueError as error:
        parser.error(str(error))
    if args.reuse_from and (args.stage not in ("scan", "burst", "boundary", "online") or args.pilot_only):
        parser.error("--reuse-from requires a formal scan/burst/boundary/online stage")
    os.chdir(ROOT)
    verify(ROOT)
    if args.stage == "analyze":
        if not args.results:
            parser.error("--stage analyze requires --results")
        from analyze import analyze
        analyze(args.results, args.mode)
        return
    model = args.model.resolve()
    if not (model / "config.json").is_file() or not (model / "tokenizer_config.json").is_file():
        parser.error(f"Model/tokenizer files not found at {model}; supply --model /absolute/model/path")
    if not args.dry_run and args.stage not in ("prepare", "check") and not list(model.glob("*.safetensors")):
        parser.error("No local safetensors weights found; this package does not download models")
    if args.stage in ("burst", "boundary", "online") and not args.plan:
        parser.error("This stage requires --plan pointing to the frozen preparation from scan")
    if args.stage == "burst" and not args.budgets:
        parser.error("Freeze candidate budgets first, then specify --budgets, e.g. 256,384")
    if args.stage == "online" and not args.rates:
        parser.error("Online requires --rates with calibrated common rates, e.g. 0.1,0.2 (examples, not calibrated defaults)")
    defaults = "128,256,384,512" if args.stage in ("scan", "pilot", "prepare", "check") else "256,512"
    budgets = [int(x) for x in (args.budgets or defaults).split(",")]
    if len(set(budgets)) != len(budgets) or any(x not in (128, 256, 384, 512) for x in budgets):
        parser.error("Use unique budgets from 128,256,384,512 in this frozen package")
    for name in ("delay", "window", "timeout"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name} must be finite and positive")
    for name in ("kv_blocks", "warmup"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    dependencies(args.no_install)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ["PYTHONUNBUFFERED"] = "1"
    info = environment(model, gpu=not args.dry_run and args.stage != "prepare")
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
    if args.stage == "check":
        return
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    out = (args.out or ROOT / "runs" / f"{stamp}-{args.stage}").resolve()
    if out.exists():
        raise ValueError("Stage output must be new; existing results are never overwritten")
    out.mkdir(parents=True)
    LOG = (out / "run.log").open("w", encoding="utf-8")
    write(out / "environment.json", info)
    try:
        if args.plan:
            plan = args.plan.resolve()
            manifest = read(plan / "manifest.json")
            if Path(manifest["model"]).resolve() != model:
                raise ValueError("Frozen plan model differs from --model; prepare a new plan")
            controls = {"delay": manifest["delay_s"], "kv_blocks": manifest["blocks"],
                        "warmup": manifest["common"]["warmup"], "window": manifest["common"]["window_s"],
                        "timeout": manifest["common"]["timeout_s"]}
            for key, value in controls.items():
                if getattr(args, key) is not None and getattr(args, key) != value:
                    raise ValueError(f"Cannot change {key} while reusing a frozen plan; omit --plan to prepare a new one")
        else:
            plan = ROOT / "plans" / stamp
            env = dict(os.environ, MODEL=str(model), EXPERIMENT_PLAN=str(plan),
                       BURST_DELAY=str(args.delay if args.delay is not None else 1.0),
                       KV_BLOCKS=str(args.kv_blocks if args.kv_blocks is not None else 565),
                       EXPERIMENT_WARMUP=str(args.warmup if args.warmup is not None else 1),
                       EXPERIMENT_WINDOW=str(args.window if args.window is not None else 5.0),
                       EXPERIMENT_TIMEOUT=str(args.timeout if args.timeout is not None else 3600.0))
            run([sys.executable, EXPERIMENTS / "prepare.py"], env=env)
            manifest = read(plan / "manifest.json")
        write(out / "stage.json", {"stage": args.stage, "plan": str(plan), "budgets": budgets,
                                  "block_ids": block_ids,
                                  "dry_run": args.dry_run, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}})
        print("FROZEN PLAN:", plan, flush=True)
        if args.stage == "prepare":
            return
        mode = "burst"
        if args.stage in ("scan", "pilot", "burst"):
            names = ["confirm0", "confirm1", "confirm2"] if args.stage == "burst" else ["explore"]
            configs = [plan / f"{name}_T{budget}.json" for name in names for budget in budgets]
        else:
            config_root = out / "configs"
            config_root.mkdir()
            configs = []
            base = read(plan / "explore_T512.json")
            if args.stage == "boundary":
                mode = "boundary"
                for budget in budgets:
                    for slots in (4, 16):
                        path = config_root / f"D_T{budget}_S{slots}.json"
                        write(path, dict(base, token_budget=budget, max_seqs=slots))
                        configs.append(path)
            else:
                mode = "stream"
                sys.path.insert(0, str(APP))
                from experiments.benchmark import digest
                from experiments.online.online_workload import validate_trace
                rates = [float(x) for x in args.rates.split(",")]
                seeds = [int(x) for x in args.arrival_seeds.split(",")]
                if any(not math.isfinite(x) or x <= 0 for x in rates) or len(set(rates)) != len(rates):
                    raise ValueError("Rates must be distinct, finite and positive")
                if not seeds or len(set(seeds)) != len(seeds) or any(x < 0 for x in seeds):
                    raise ValueError("Arrival seeds must be distinct nonnegative integers")
                bank = read(plan / "stream_bank.json")
                for rate_index, rate in enumerate(rates):
                    for seed in seeds:
                        rng, arrival, requests = random.Random(seed), 0.0, []
                        for index, original in enumerate(bank["requests"]):
                            if index:
                                arrival += rng.expovariate(rate)
                            requests.append(dict(original, arrival_s=arrival))
                        trace = dict(bank, requests=requests, workload_sha256=digest(requests),
                                     generation={"profile": "poisson", "arrival_rate": rate, "seed": seed})
                        validate_trace(trace)
                        trace_path = config_root / f"trace_rate{rate_index}_seed{seed}.json"
                        write(trace_path, trace)
                        for budget in budgets:
                            path = config_root / f"E_rate{rate_index}_seed{seed}_T{budget}.json"
                            write(path, dict(base, trace=str(trace_path), profile="poisson", token_budget=budget))
                            configs.append(path)
        matrix = [sys.executable, EXPERIMENTS / "run_matrix.py", "--configs", *configs,
                  "--out", out / "formal", "--mode", mode,
                  "--block-ids", ",".join(map(str, block_ids))]
        if args.reuse_from:
            matrix += ["--reuse-from", args.reuse_from.resolve()]
        if args.dry_run:
            run(matrix)
            print("DRY RUN COMPLETE: no GPU inference performed", flush=True)
            return
        # Validate every configuration before allocating model weights in any pilot.
        run(matrix)
        pilot_results = []
        for path in configs:
            destination = out / "pilot" / path.stem
            command = [sys.executable, APP / "experiments/online/run_online.py", "--config", path,
                       "--rounds", "1", "--out", destination]
            if mode != "stream":
                command.append("--save-details")
            run(command)
            payload = read(destination / "results.json")
            findings = audit(payload, mode, details=mode != "stream")
            pilot_results.append({"config": str(path), "out": str(destination), "findings": findings})
            write(out / "pilots.json", pilot_results)
        if args.stage == "pilot" or args.pilot_only:
            print("PILOTS COMPLETE:", out, flush=True)
            return
        run(matrix + ["--execute", "--reference-result",
                      Path(pilot_results[0]["out"]) / "results.json"])
        from analyze import analyze
        analyze(out / "formal", mode)
        print("COMPLETE:", out, flush=True)
        print("NEXT: retain this --plan for burst/boundary/online:", plan, flush=True)
    except Exception as error:
        write(out / "failure.json", {"error": repr(error), "stage": args.stage})
        raise
    finally:
        LOG.close()
        LOG = None


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
        print("STOP:", error, file=sys.stderr)
        raise SystemExit(1)
