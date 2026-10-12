"""Experiment accounting with real scheduling, plus a real CPU checkpoint matrix."""

import json
import subprocess
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from experiments.benchmark import digest, write_json
from experiments.math500 import run_math500 as experiment
from experiments.common import measurement
from experiments.common.measurement import replay, reset_cache, run_worker
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from numeric_helpers import hf_greedy_tokens
from prefix_helpers import make_cache
from test_prefix_scheduler import ScriptedRunner, step


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def wait(self, duration):
        self.now += duration


class TimedEngine:
    def __init__(self, clock, durations=(1, .5, .75, .5), prefix_cache=False):
        manager = make_cache(8, 4, enabled=prefix_cache)
        self.scheduler = Scheduler(manager, 2, 4, 32, scheduling_policy="mixed")
        self.runner = ScriptedRunner(manager)
        self.model_runner = SimpleNamespace(kv_cache_manager=manager)
        self.clock = clock
        self.durations = iter(durations)
        self.next_id = 0

    def is_finished(self):
        return self.scheduler.is_finished()

    def add_request(self, prompt, sampling):
        sid = self.next_id
        self.next_id += 1
        self.scheduler.add(Sequence(sid, prompt, sampling.max_tokens,
                                    eos_token_id=None if sampling.ignore_eos else 11))
        return sid

    def step(self):
        finished = step(self.scheduler, self.runner)
        self.clock.now += next(self.durations)
        return finished


def request(rid="a", tokens=None, arrival=0):
    return {"id": rid, "dataset_index": 0, "prompt_token_ids": tokens or [1], "arrival_s": arrival}


def test_latency_includes_planned_queue_delay_and_only_committed_tokens():
    clock = Clock()
    engine = TimedEngine(clock)
    run = replay(engine, [request(), request("b", [2, 3, 4], .25)], 3,
                 "latency", lambda: None, 10, clock=clock, wait=clock.wait, save_steps=True)
    a, b = run["requests"]
    assert a["marks"] == [1, 1.5, 2.25]
    assert b["marks"] == [1.5, 2.25, 2.75]
    assert a["ttft_ms"] == 1000 and b["ttft_ms"] == 1250
    assert b["admission_delay_ms"] == 750
    assert a["e2e_ms"] == 2250 and b["e2e_ms"] == 2500
    assert a["tpot_ms"] == b["tpot_ms"] == 625
    summary = run["summary"]
    assert summary["itl_ms"]["n"] == 4
    assert summary["token_weighted_tpot_ms"] == 625
    assert summary["output_tok_s"] == pytest.approx(6 / 2.75)
    assert summary["counts"]["mixed_steps"] == 1
    assert summary["mixed_step_fraction"] == .25
    assert [s["kind"] for s in run["steps"]] == ["prefill", "mixed", "decode", "decode"]


@pytest.mark.parametrize("ignore_eos,expected,expected_tpot", [(False, 2, 500), (True, 3, 625)])
def test_actual_eos_lengths_are_used_for_throughput(ignore_eos, expected, expected_tpot):
    clock = Clock()
    run = replay(TimedEngine(clock), [request()], 3, "latency", lambda: None, 10,
                 ignore_eos=ignore_eos, clock=clock, wait=clock.wait)
    assert run["summary"]["output_tokens"] == expected
    assert run["requests"][0]["tpot_ms"] == expected_tpot
    assert run["summary"]["output_tok_s"] == pytest.approx(expected / clock.now)


@pytest.mark.parametrize("timing", ["throughput", "latency"])
def test_single_token_has_no_tpot_and_throughput_has_no_token_timestamps(timing):
    clock = Clock()
    run = replay(TimedEngine(clock), [request()], 1, timing, lambda: None, 10,
                 clock=clock, wait=clock.wait)
    assert run["summary"]["tpot_ms"] == {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    assert run["summary"]["itl_ms"]["n"] == 0
    assert run["requests"][0]["marks"] == ([1] if timing == "latency" else [])
    assert run["summary"]["ttft_ms"]["n"] == int(timing == "latency")


def test_idle_arrival_wait_is_inside_elapsed_time():
    clock = Clock()
    run = replay(TimedEngine(clock), [request(arrival=2)], 1, "latency", lambda: None, 10,
                 clock=clock, wait=clock.wait)
    assert run["summary"]["elapsed_s"] == pytest.approx(3)
    assert run["requests"][0]["ttft_ms"] == pytest.approx(1000)
    assert run["summary"]["output_tok_s"] == pytest.approx(1 / 3)


def test_timeout_including_final_step_restores_scheduler_methods():
    clock = Clock()
    engine = TimedEngine(clock)
    schedule, preempt = engine.scheduler.schedule, engine.scheduler.preempt
    with pytest.raises(TimeoutError, match="completed after timeout"):
        replay(engine, [request()], 1, "latency", lambda: None, .5, clock=clock)
    assert engine.scheduler.schedule == schedule and engine.scheduler.preempt == preempt


def test_sync_failure_restores_scheduler_methods():
    clock = Clock()
    engine = TimedEngine(clock)
    schedule, preempt = engine.scheduler.schedule, engine.scheduler.preempt

    def bad_sync():
        raise RuntimeError("sync failure")

    with pytest.raises(RuntimeError, match="sync failure"):
        replay(engine, [request()], 1, "latency", bad_sync, 10, clock=clock)
    assert engine.scheduler.schedule == schedule and engine.scheduler.preempt == preempt


def test_reset_clears_warm_prefixes_and_keeps_same_kv_tensors():
    clock = Clock()
    engine = TimedEngine(clock, prefix_cache=True)
    manager = engine.model_runner.kv_cache_manager
    pool_ptrs = manager.k_pool.data_ptr(), manager.v_pool.data_ptr()
    prompts = [request(tokens=[1, 2, 3, 4, 5])]
    replay(engine, prompts, 1, "throughput", lambda: None, 10, clock=clock)
    assert manager.block_manager.hash_to_block_id
    reset_cache(engine)
    assert engine.model_runner.kv_cache_manager is manager
    assert (manager.k_pool.data_ptr(), manager.v_pool.data_ptr()) == pool_ptrs
    assert not manager.block_manager.hash_to_block_id
    assert list(manager.block_manager.free_block_id) == list(range(8))


@pytest.fixture
def local_experiment(tmp_path):
    model = tmp_path / "qwen"
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=8, max_position_embeddings=64, tie_word_embeddings=False,
                         eos_token_id=1, attention_dropout=0)
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        reference = Qwen3ForCausalLM(config).eval().requires_grad_(False)
    reference.save_pretrained(model, safe_serialization=True)
    tokenizer = Tokenizer(WordLevel({f"t{i}_": i for i in range(32)}, unk_token="t0_"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="t0_", pad_token="t0_", eos_token="t1_",
        chat_template="{% for m in messages %}{{ m['content'] }} {% endfor %}{% if add_generation_prompt %}t2_{% endif %}",
    ).save_pretrained(model)
    dataset = tmp_path / "test.jsonl"
    rows = [{"unique_id": str(i), "problem": p, "answer": "secret answer",
             "solution": "secret solution", "subject": "Algebra", "level": 1}
            for i, p in enumerate(["t3_ t4_ t5_", "t6_ t7_ t8_ t9_ t10_", "t11_"])]
    dataset.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    config_path = tmp_path / "config.json"
    write_json(config_path, {"model": str(model), "dataset": str(dataset), "device": "cpu",
                            "dtype": "float32", "max_tokens": [1, 3], "max_seqs": [1, 2],
                            "token_budgets": [2], "timings": ["throughput", "latency"],
                            "block_size": 4, "warmup": 0, "repeats": 1,
                            "out": str(tmp_path / "results")})
    return reference, config_path


def direct_worker(job, job_path, log_path):
    run_worker(job)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_sharded_checkpoint_runs_three_policy_matrix(local_experiment, monkeypatch, dtype):
    reference, config = local_experiment
    options = experiment.parse_options([
        "--config", str(config), "--dtype", dtype, "--max-tokens", "3", "--max-seqs", "2",
    ])
    model = Path(options["model"])
    reference.to(dtype=getattr(torch, dtype))
    reference.save_pretrained(model, safe_serialization=True, max_shard_size="1KB")
    (model / "model.safetensors").unlink(missing_ok=True)
    monkeypatch.setattr(experiment, "run_worker_process", direct_worker)
    report, code = experiment.execute(options)
    assert code == 0 and len(report["summary"]) == 6
    assert all(row["matching_outputs"] and row["complete"] for row in report["output_comparisons"])
    workload = json.loads((Path(options["out"]) / "workload.json").read_text(encoding="utf-8"))
    checkpoint = workload["checkpoint_files"]
    assert checkpoint["model.safetensors.index.json"]["sha256"] == experiment.sha256(
        model / "model.safetensors.index.json")
    expected = {row["id"]: hf_greedy_tokens(reference, row["prompt_token_ids"], 3)
                for row in workload["requests"]}
    for path in (Path(options["out"]) / "cases").glob("*.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        assert all(run["output_sha256"] == digest(expected) for run in result["runs"])


def test_shard_index_content_changes_resume_fingerprint(local_experiment):
    import os

    reference, config = local_experiment
    options = experiment.parse_options(["--config", str(config)])
    model = Path(options["model"])
    reference.save_pretrained(model, safe_serialization=True, max_shard_size="1KB")
    (model / "model.safetensors").unlink(missing_ok=True)
    first = experiment.prepare_workload(options)
    first_jobs, _ = experiment.build_jobs(options, first)
    index = model / "model.safetensors.index.json"
    stat = index.stat()
    original = index.read_bytes()
    # Keep size and mtime identical, but change valid JSON whitespace. Shard
    # metadata alone must not let a changed index resume an old experiment.
    changed = original.replace(b"\n", b" ", 1)
    assert changed != original and len(changed) == len(original)
    index.write_bytes(changed)
    os.utime(index, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    second = experiment.prepare_workload(options)
    second_jobs, _ = experiment.build_jobs(options, second)
    first_entry, second_entry = [workload["checkpoint_files"][index.name] for workload in (first, second)]
    assert first_entry["size"] == second_entry["size"]
    assert first_entry["mtime_ns"] == second_entry["mtime_ns"]
    assert first_entry["sha256"] != second_entry["sha256"]
    assert first_jobs[0]["fingerprint"] != second_jobs[0]["fingerprint"]


def test_resume_tracks_moved_dependencies_and_ignores_online_only_changes(tmp_path, monkeypatch):
    app = tmp_path / "app"
    paths = ("nanovllm/__init__.py", "experiments/__init__.py", "experiments/common/__init__.py",
             "experiments/common/measurement.py", "experiments/math500/__init__.py",
             "experiments/math500/run_math500.py", "experiments/online/run_online.py",
             "experiments/benchmark.py")
    for relative in paths:
        path = app / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# baseline\n", encoding="utf-8")
    monkeypatch.setattr(experiment, "APP_ROOT", app)
    options = experiment.parse_options(["--model", str(tmp_path / "model"),
                                        "--dataset", str(tmp_path / "dataset.jsonl")])
    workload = {"requests": [{"id": "a", "prompt_token_ids": [1]}],
                "max_model_len": 128, "workload_sha256": "workload",
                "model_config_sha256": "config", "checkpoint_files": {},
                "tokenizer_sha256": {}}
    first_jobs, sources = experiment.build_jobs(options, workload)
    assert set(sources) == set(paths) - {"experiments/online/run_online.py"}
    baseline = first_jobs[0]["fingerprint"]

    # Online experiments evolve independently of the Math500 resume contract.
    (app / "experiments/online/run_online.py").write_text("# online edit\n", encoding="utf-8")
    jobs, _ = experiment.build_jobs(options, workload)
    assert jobs[0]["fingerprint"] == baseline

    # A changed runner or replay implementation must force a new result set.
    for relative in ("experiments/common/measurement.py", "experiments/math500/run_math500.py"):
        path = app / relative
        path.write_text("# modified\n", encoding="utf-8")
        jobs, _ = experiment.build_jobs(options, workload)
        assert jobs[0]["fingerprint"] != baseline
        path.write_text("# baseline\n", encoding="utf-8")


def test_experiment_preflight_rejects_incomplete_sharded_checkpoint(local_experiment):
    reference, config = local_experiment
    options = experiment.parse_options(["--config", str(config)])
    model = Path(options["model"])
    reference.save_pretrained(model, safe_serialization=True, max_shard_size="1KB")
    (model / "model.safetensors").unlink(missing_ok=True)
    shard = next(model.glob("model-*.safetensors"))
    shard.unlink()
    with pytest.raises(FileNotFoundError, match=shard.name):
        experiment.prepare_workload(options)


def test_dry_run_still_accepts_config_and_tokenizer_without_weights(local_experiment):
    _, config = local_experiment
    options = experiment.parse_options(["--config", str(config), "--dry-run"])
    (Path(options["model"]) / "model.safetensors").unlink()
    workload = experiment.prepare_workload(options)
    assert workload["checkpoint_files"] == {} and len(workload["requests"]) == 3


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_real_checkpoint_matrix_metrics_outputs_csv_and_resume(local_experiment, monkeypatch, dtype):
    reference, config = local_experiment
    monkeypatch.setattr(experiment, "run_worker_process", direct_worker)
    arguments = ["--config", str(config), "--dtype", dtype]
    options = experiment.parse_options(arguments)
    assert options["blocks"] == -1
    report, code = experiment.execute(options)
    assert code == 0 and len(report["summary"]) == len(report["runs"]) == 24
    assert all(c["matching_outputs"] and c["complete"] and c["case_count"] == 6
               for c in report["output_comparisons"])
    workload = json.loads((Path(options["out"]) / "workload.json").read_text(encoding="utf-8"))
    assert [len(r["prompt_token_ids"]) for r in workload["requests"]] == [4, 6, 2]
    assert all("answer" not in r and "solution" not in r for r in workload["requests"])
    reference.to(dtype=getattr(torch, dtype))
    expected = {}
    for r in workload["requests"]:
        tokens = list(r["prompt_token_ids"])
        with torch.inference_mode():
            for _ in range(3):
                next_token = reference(torch.tensor([tokens]), use_cache=False).logits[0, -1].argmax().item()
                tokens.append(next_token)
        expected[r["id"]] = tokens[-3:]
    for path in (Path(options["out"]) / "cases").glob("*.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        assert "num_kvcache_blocks" not in result["requested_config"]
        assert result["cache"]["allocation"] == "auto"
        assert result["config"]["num_kvcache_blocks"] == result["cache"]["num_blocks"] > 0
        assert result["cache"]["dtype"] == f"torch.{dtype}"
        # The CPU engine sizes its pool from model length and the batch limit;
        # this cohort needs two pages per request, so S=1 and S=2 differ.
        assert result["cache"]["num_blocks"] == 2 * result["factors"]["max_seqs"]
        for run in result["runs"]:
            cap = result["factors"]["max_tokens"]
            assert run["output_sha256"] == digest({rid: ids[:cap] for rid, ids in expected.items()})
            assert set(run) == {"timing", "repeat", "summary", "output_sha256"}
            assert run["summary"]["output_tokens"] == 3 * cap
            assert run["summary"]["num_kvcache_blocks"] == result["cache"]["num_blocks"]
            assert run["summary"]["kv_capacity_tokens"] == 4 * result["cache"]["num_blocks"]
            assert run["summary"]["kv_pool_mb"] == result["cache"]["pool_mb"] > 0
            assert run["summary"]["counts"]["query_tokens"] >= 12 + 3 * (cap - 1)
            if run["timing"] == "throughput":
                assert run["summary"]["ttft_ms"]["mean"] is None
            else:
                assert run["summary"]["ttft_ms"]["n"] == 3
                assert run["summary"]["tpot_ms"]["n"] == (3 if cap > 1 else 0)
    assert (Path(options["out"]) / "summary.csv").read_bytes().startswith(b"\xef\xbb\xbf")

    def no_launch(*args):
        pytest.fail("completed cases must be reused")

    monkeypatch.setattr(experiment, "run_worker_process", no_launch)
    resumed, code = experiment.execute(experiment.parse_options(arguments + ["--resume"]))
    assert code == 0 and resumed["summary"] == report["summary"]
    with pytest.raises(ValueError, match="differs"):
        experiment.execute(experiment.parse_options(arguments + ["--resume", "--max-tokens", "2"]))


def test_infeasible_cases_do_not_filter_math500_or_report_success(local_experiment, monkeypatch):
    _, config = local_experiment
    monkeypatch.setattr(experiment, "run_worker_process", lambda *args: pytest.fail("infeasible work must not launch"))
    options = experiment.parse_options(["--config", str(config), "--max-model-len", "1"])
    report, code = experiment.execute(options)
    assert code == 1 and report["selected_problem_count"] == 3
    assert all(r["status"] == "infeasible" and "output_tok_s" not in r for r in report["summary"])
    assert report["runs"] == [] and report["output_comparisons"] == []


def test_worker_failure_with_corrupt_partial_file_continues(local_experiment, monkeypatch):
    _, config = local_experiment
    calls = []

    def fail_once(job, job_path, log_path):
        calls.append(job["case_id"])
        if len(calls) == 1:
            Path(job["result_path"]).write_text("{partial", encoding="utf-8")
            raise RuntimeError("injected failure")
        run_worker(job)

    monkeypatch.setattr(experiment, "run_worker_process", fail_once)
    report, code = experiment.execute(experiment.parse_options([
        "--config", str(config), "--policies", "separate,mixed", "--max-tokens", "1", "--max-seqs", "1"]))
    assert code == 1 and len(calls) == 2
    failed = [r for r in report["summary"] if r["status"] == "failed"]
    assert len(failed) == 2 and all("output_tok_s" not in r and r["matching_outputs"] is None for r in failed)
    assert all(c["matching_outputs"] and not c["complete"] for c in report["output_comparisons"])


def test_unstable_results_preserve_mismatching_run_hashes(local_experiment, monkeypatch):
    _, config = local_experiment

    def unstable_worker(job, job_path, log_path):
        result = run_worker(job)
        run = result["runs"][-1]
        run["output_sha256"] = digest("injected changed output")
        result["status"] = "unstable"
        write_json(Path(job["result_path"]), result)

    monkeypatch.setattr(experiment, "run_worker_process", unstable_worker)
    report, code = experiment.execute(experiment.parse_options([
        "--config", str(config), "--policies", "mixed", "--max-tokens", "1",
        "--max-seqs", "1", "--repeats", "2", "--timings", "latency"]))
    assert code == 1 and report["summary"][0]["status"] == "unstable"
    comparison = report["output_comparisons"][0]
    assert not comparison["matching_outputs"] and comparison["complete"]
    assert comparison["first_mismatch"]["reference_repeat"] == 1
    assert comparison["first_mismatch"]["actual_repeat"] == 2
    assert comparison["first_mismatch"]["reference_sha256"] != comparison["first_mismatch"]["actual_sha256"]
    monkeypatch.setattr(experiment, "run_worker", lambda job: {"status": "unstable"})
    job_path = next((Path(report["settings"]["out"]) / "jobs").glob("*.json"))
    assert experiment.main(["--worker-job", str(job_path)]) == 0


def test_cross_timing_divergence_is_reported_even_when_each_mode_is_consistent(local_experiment, monkeypatch):
    _, config = local_experiment

    def changed_timing(job, job_path, log_path):
        result = run_worker(job)
        run = result["runs"][-1]
        run["output_sha256"] = digest("injected changed output")
        result["status"] = "unstable"
        write_json(Path(job["result_path"]), result)

    monkeypatch.setattr(experiment, "run_worker_process", changed_timing)
    report, code = experiment.execute(experiment.parse_options([
        "--config", str(config), "--policies", "mixed", "--max-tokens", "1", "--max-seqs", "1"]))
    assert code == 1 and all(not r["matching_outputs"] for r in report["summary"])
    comparison = next(c for c in report["output_comparisons"] if c["timing"] == "all")
    assert not comparison["matching_outputs"]
    assert comparison["first_mismatch"]["reference_timing"] == "throughput"
    assert comparison["first_mismatch"]["actual_timing"] == "latency"


@pytest.mark.parametrize("exit_code,initialized,retry", [
    (0xC000070A, False, True), (1, False, False), (0xC000070A, True, False),
])
def test_only_native_initialization_failure_is_retried_with_logs(tmp_path, monkeypatch, exit_code, initialized, retry):
    monkeypatch.setattr(experiment.sys, "platform", "win32")
    calls = []

    def subprocess_popen(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=StringIO("case: config dtype=bfloat16\n" if initialized else ""),
                               wait=lambda timeout=None: exit_code if len(calls) == 1 else 0)

    monkeypatch.setattr(experiment.subprocess, "Popen", subprocess_popen)
    job = {"case_id": "case", "settings": {"worker_retries": 1, "timeout_s": 1,
                                             "warmup": 0, "repeats": 1, "timings": ["latency"]}}
    log_path = tmp_path / "case.log"
    if retry:
        attempts = experiment.run_worker_process(job, tmp_path / "job.json", log_path)
        assert [a["exit_code"] for a in attempts] == [exit_code, 0]
        assert log_path.exists() and (tmp_path / "case-retry1.log").exists()
        assert "utf8" in calls[0]
    else:
        with pytest.raises(RuntimeError) as caught:
            experiment.run_worker_process(job, tmp_path / "job.json", log_path)
        assert caught.value.worker_attempts[0]["exit_code"] == exit_code
    assert len(calls) == (2 if retry else 1)


def test_worker_buffers_log_until_exit_and_streams_only_config_and_metrics(tmp_path, monkeypatch, capsys):
    original_popen = subprocess.Popen
    log_path = tmp_path / "case.log"
    processes = []
    script = ("from pathlib import Path\n"
              "print('case: config dtype=bfloat16', flush=True)\n"
              "print('unwanted initialization details and generated tokens', flush=True)\n"
              "print('case throughput repeat 1: output=12.000tok/s', flush=True)\n"
              f"assert not Path({str(log_path)!r}).exists(), 'log written before worker exit'\n")

    def child(command, **kwargs):
        process = original_popen([command[0], "-u", "-X", "utf8", "-c", script], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(experiment.subprocess, "Popen", child)
    job = {"case_id": "case", "settings": {"worker_retries": 0, "timeout_s": 1,
                                            "warmup": 0, "repeats": 1, "timings": ["throughput"]}}
    attempts = experiment.run_worker_process(job, tmp_path / "job.json", log_path)
    assert attempts[0]["exit_code"] == 0 and processes[0].returncode == 0
    expected = "case: config dtype=bfloat16\ncase throughput repeat 1: output=12.000tok/s\n"
    assert capsys.readouterr().out == expected
    assert log_path.read_text(encoding="utf-8") == expected


def test_repeats_never_decode_or_write_results_and_aggregate_saved_once(local_experiment, monkeypatch, capsys):
    _, config = local_experiment
    worker_writes, report_writes = [], []
    original_write = measurement.write_json
    original_save = experiment.save_report

    def write_completed_case(path, result):
        assert result["status"] == "ok"
        assert len(result["runs"]) == 6
        assert all(set(r) == {"timing", "repeat", "summary", "output_sha256"} for r in result["runs"])
        worker_writes.append(path)
        original_write(path, result)

    def save_once(out, report):
        assert len(report["runs"]) == 12
        report_writes.append(out)
        original_save(out, report)

    def no_decode(*args, **kwargs):
        pytest.fail("benchmark must not decode generated output")

    monkeypatch.setattr(PreTrainedTokenizerFast, "decode", no_decode)
    monkeypatch.setattr(measurement, "write_json", write_completed_case)
    monkeypatch.setattr(experiment, "save_report", save_once)
    monkeypatch.setattr(experiment, "run_worker_process", direct_worker)
    options = experiment.parse_options(["--config", str(config), "--policies", "separate,mixed",
                                        "--max-tokens", "3", "--max-seqs", "1", "--repeats", "3"])
    report, code = experiment.execute(options)
    assert code == 0 and len(worker_writes) == 2 and len(report_writes) == 1
    lines = capsys.readouterr().out.splitlines()
    assert len([line for line in lines if ": config " in line]) == 2
    assert len([line for line in lines if " repeat " in line]) == 12
    assert not any("initializing" in line or "loading checkpoint" in line for line in lines)
    assert report["schema"] == 2


def test_dry_run_prepares_real_prompts_without_launching_weights(local_experiment, monkeypatch):
    _, config = local_experiment
    monkeypatch.setattr(experiment, "run_worker_process", lambda *args: pytest.fail("dry-run must not launch"))
    options = experiment.parse_options(["--config", str(config), "--dry-run", "--num-problems", "2",
                                        "--policies", "interleaved", "--max-tokens", "16",
                                        "--max-seqs", "16", "--token-budgets", "2048", "--diagnostics"])
    report, code = experiment.execute(options)
    assert code == 0 and report["selected_problem_count"] == 2
    assert all(r["status"] == "planned" for r in report["summary"])
    assert len(report["summary"]) == 2  # One configuration, two timing modes.
    workload = json.loads((Path(options["out"]) / "workload.json").read_text(encoding="utf-8"))
    assert [r["dataset_index"] for r in workload["requests"]] == [0, 1]
    manifest = json.loads((Path(options["out"]) / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["jobs"]) == 1 and manifest["jobs"][0]["settings"]["diagnostics"]


def test_auto_capacity_is_deferred_to_engine_not_clipped_by_minus_one(local_experiment, monkeypatch):
    _, config = local_experiment
    monkeypatch.setattr(experiment, "run_worker_process", lambda *args: pytest.fail("dry-run must not load a model"))
    options = experiment.parse_options(["--config", str(config), "--dry-run", "--max-tokens", "48"])
    workload = experiment.prepare_workload(options)
    assert workload["max_model_len"] == 53
    assert workload["kv_allocation"] == "auto" and workload["planned_kv_capacity_tokens"] is None
    jobs, _ = experiment.build_jobs(options, workload)
    assert all("num_kvcache_blocks" not in j["engine_config"] and not j["preflight_error"] for j in jobs)
    report, code = experiment.execute(options)
    assert code == 0 and all(r["status"] == "planned" for r in report["summary"])


def test_explicit_block_override_is_recorded_and_capacity_checked(local_experiment, monkeypatch):
    _, config = local_experiment
    monkeypatch.setattr(experiment, "run_worker_process", direct_worker)
    options = experiment.parse_options(["--config", str(config), "--blocks", "8",
                                        "--policies", "mixed", "--max-tokens", "1", "--max-seqs", "1"])
    report, code = experiment.execute(options)
    assert code == 0 and all(r["num_kvcache_blocks"] == 8 for r in report["summary"])
    result = json.loads(next((Path(options["out"]) / "cases").glob("*.json")).read_text(encoding="utf-8"))
    assert result["cache"]["allocation"] == "manual" and result["cache"]["capacity_tokens"] == 32
    assert result["requested_config"]["num_kvcache_blocks"] == 8
    small = experiment.parse_options(["--config", str(config), "--blocks", "1"])
    workload = experiment.prepare_workload(small)
    assert workload["planned_kv_capacity_tokens"] == workload["max_model_len"] == 4
    jobs, _ = experiment.build_jobs(small, workload)
    assert all(j["preflight_error"] for j in jobs)


def test_isolated_preparation_matches_local_tokenizer_and_cleans_temporary_files(local_experiment):
    _, config = local_experiment
    options = experiment.parse_options(["--config", str(config), "--num-problems", "2"])
    assert experiment.prepare_workload_process(options) == experiment.prepare_workload(options)
    assert not list(Path(options["out"]).glob(".prepare-*"))


@pytest.mark.parametrize("key,value", [
    ("max_tokens", [0]), ("max_seqs", []), ("token_budgets", [True]), ("policies", ["bogus"]),
    ("timings", ["both"]), ("warmup", -1), ("repeats", 0), ("num_problems", True),
    ("ignore_eos", "false"), ("arrival_interval_ms", float("nan")), ("timeout_s", 0),
    ("blocks", -2), ("blocks", 0), ("blocks", True),
    ("dtype", "float64"), ("device", "cuda:x"), ("out", 42), ("unknown", 1),
    ("worker_retries", -1), ("save_steps", True), ("diagnostics", "true"),
    ("diagnostics_interval_s", 0), ("diagnostics_interval_s", .01),
    ("diagnostics_interval_s", float("nan")), ("diagnostics_interval_s", True),
    ("gpu_memory_utilization", 0), ("gpu_memory_utilization", -0.1),
    ("gpu_memory_utilization", 1.1), ("gpu_memory_utilization", float("nan")),
    ("gpu_memory_utilization", True),
])
def test_invalid_configuration_rejected_before_model_load(tmp_path, key, value):
    config = tmp_path / "config.json"
    write_json(config, {key: value})
    with pytest.raises(ValueError):
        experiment.parse_options(["--config", str(config)])


def test_cli_overrides_json_and_aliases(local_experiment):
    _, config = local_experiment
    options = experiment.parse_options(["--config", str(config), "--max-tokens", "2,2,4",
                                        "--max-num-seqs", "3", "--token-budget", "5",
                                        "--timings", "both", "--no-ignore-eos", "--prefix-cache",
                                        "--gpu-memory-utilization", "0.75"])
    assert options["max_tokens"] == [2, 4] and options["max_seqs"] == [3]
    assert options["token_budgets"] == [5] and options["timings"] == ["throughput", "latency"]
    assert options["prefix_cache"] and not options["ignore_eos"]
    assert options["gpu_memory_utilization"] == .75
    jobs, _ = experiment.build_jobs(options, experiment.prepare_workload(options))
    assert all(j["engine_config"]["gpu_memory_utilization"] == .75
               and "num_kvcache_blocks" not in j["engine_config"] for j in jobs)


def test_diagnostic_toggle_preserves_outputs_and_exports_warmup_metrics(local_experiment, monkeypatch):
    _, config = local_experiment
    monkeypatch.setattr(experiment, "run_worker_process", direct_worker)
    arguments = ["--config", str(config), "--policies", "interleaved", "--max-tokens", "3",
                 "--max-seqs", "2", "--warmup", "1", "--repeats", "2"]
    off_options = experiment.parse_options(arguments + ["--no-diagnostics"])
    off_report, code = experiment.execute(off_options)
    assert code == 0
    on_options = experiment.parse_options(arguments + ["--diagnostics", "--diagnostics-interval-s", "0.1",
                                                       "--out", str(Path(off_options["out"]).parent / "diagnostics")])
    on_report, code = experiment.execute(on_options)
    assert code == 0 and len(on_report["runs"]) == 4
    assert [r["output_sha256"] for r in on_report["runs"]] == [r["output_sha256"] for r in off_report["runs"]]
    assert all(r["process_cpu_s"] >= 0 and r["process_cpu_pct"] >= 0 for r in on_report["runs"])
    assert all(r["peak_cuda_reserved_mb"] is None and r["gpu_samples"] == 0 for r in on_report["runs"])
    result_path = next((Path(on_options["out"]) / "cases").glob("*.json"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["diagnostics"]["enabled"] and result["diagnostics"]["interval_s"] == .1
    assert result["diagnostics"]["gpu_backend"] is None
    assert len(result["warmup_runs"]) == 1 and result["warmup_runs"][0]["summary"]["process_cpu_s"] >= 0
    assert all(set(r) == {"timing", "repeat", "summary", "output_sha256"} for r in result["runs"])
    off_result = json.loads(next((Path(off_options["out"]) / "cases").glob("*.json")).read_text(encoding="utf-8"))
    assert "diagnostics" not in off_result and "warmup_runs" not in off_result


def test_failed_replay_stops_diagnostic_threads(local_experiment, monkeypatch):
    _, config = local_experiment
    monitors = []
    original_monitor = measurement.DiagnosticMonitor

    def monitor(*args):
        instance = original_monitor(*args)
        monitors.append(instance)
        return instance

    def failed(*args, **kwargs):
        raise TimeoutError("injected replay failure")

    monkeypatch.setattr(measurement, "DiagnosticMonitor", monitor)
    monkeypatch.setattr(measurement, "replay", failed)
    options = experiment.parse_options(["--config", str(config), "--policies", "mixed", "--max-tokens", "1",
                                        "--max-seqs", "1", "--diagnostics"])
    workload = experiment.prepare_workload(options)
    jobs, _ = experiment.build_jobs(options, workload)
    write_json(Path(jobs[0]["workload_path"]), workload)
    with pytest.raises(TimeoutError, match="injected replay failure"):
        run_worker(jobs[0])
    assert len(monitors) == 1 and monitors[0].stop_event.is_set()
    assert all(not thread.is_alive() for thread in monitors[0].threads)
