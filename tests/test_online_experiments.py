"""Online load/report contracts, including the real CPU engine and CLI."""

import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.benchmark import digest, token_metrics, write_json
from experiments.online import run_online as experiment
from experiments.online.online_metrics import summarize_online
from experiments.online.online_workload import prepare_trace
from test_math500_experiments import Clock, TimedEngine, local_experiment


def options_for(local_experiment, tmp_path, *extra):
    _, config_path = local_experiment
    model = json.loads(config_path.read_text())["model"]
    return experiment.parse_options([
        "--profile", "burst",
        "--model", model, "--device", "cpu", "--dtype", "float32", "--blocks", "8",
        "--block-size", "4", "--max-seqs", "3", "--token-budget", "4",
        "--old-count", "2", "--late-count", "1", "--short-length", "3", "--long-length", "6",
        "--old-output", "4", "--late-output", "2", "--delay-s", "0.001",
        "--rounds", "1", "--warmup", "0", "--out", str(tmp_path / "online"), *extra])


@pytest.mark.parametrize("profile", ["burst", "interval", "poisson"])
def test_generated_load_is_reproducible_and_trace_reuse_is_exact(local_experiment, tmp_path, profile):
    options = options_for(local_experiment, tmp_path, "--profile", profile,
                          "--request-count", "5", "--arrival-rate", "8",
                          "--prompt-lengths", "3,6", "--output-lengths", "4,2")
    trace, _, plan = experiment.preflight(options)
    assert prepare_trace(options) == trace
    requests = trace["requests"]
    assert plan["max_model_len"] == 7
    assert plan["requests"] == (3 if profile == "burst" else 5)
    if profile != "burst":
        assert [len(r["prompt_token_ids"]) for r in requests] == [3, 6, 3, 6, 3]
        assert [r["max_tokens"] for r in requests] == [4, 2, 4, 2, 4]
        assert all(a["arrival_s"] < b["arrival_s"] for a, b in zip(requests, requests[1:]))
        if profile == "interval":
            assert [r["arrival_s"] for r in requests] == [0, .125, .25, .375, .5]
    path = tmp_path / "trace.json"
    write_json(path, trace)
    options.update(trace=str(path), seed=99, old_count=99, long_length=99999)
    assert experiment.preflight(options)[0] == trace


@pytest.mark.parametrize("changes", [
    {"rounds": True}, {"delay_s": float("nan")}, {"arrival_rate": 0}, {"blocks": 0},
    {"ignore_eos": "false"}, {"policies": ["mixed", "mixed"]},
    {"prompt_lengths": [1], "output_lengths": [1, 2]}, {"typo_option": 10},
])
def test_bad_config_fails_before_model_load(tmp_path, changes):
    config = tmp_path / "bad.json"
    write_json(config, {"model": "unused", **changes})
    with pytest.raises(ValueError):
        experiment.parse_options(["--config", str(config)])


def test_config_overrides_and_policy_rotation(tmp_path):
    config = tmp_path / "config.json"
    write_json(config, {"model": "unused", "rounds": 12, "prefix_cache": True})
    options = experiment.parse_options(["--config", str(config), "--rounds", "3", "--no-prefix-cache"])
    assert options["rounds"] == 3 and options["prefix_cache"] is False
    assert [experiment.policy_order(options, i)[0] for i in range(3)] == list(experiment.POLICIES)


def test_source_prompt_lengths_and_staggered_burst(local_experiment, tmp_path):
    options = options_for(local_experiment, tmp_path, "--late-count", "2", "--late-interval-s", ".5")
    source = prepare_trace(options)
    source["schema"] = 1  # Math500's workload envelope, with the same fingerprints.
    for r in source["requests"]:
        r["prompt_token_ids"] = [3, 4]
    source["workload_sha256"] = digest(source["requests"])
    path = tmp_path / "source.json"
    write_json(path, source)
    options["source"] = str(path)
    trace = prepare_trace(options)
    assert trace["requests"][0]["prompt_token_ids"] == [3, 4, 3]
    assert trace["requests"][-1]["prompt_token_ids"] == [3, 4, 3, 4, 3, 4]
    assert trace["requests"][-1]["arrival_s"] == pytest.approx(.501)
    options["old_count"] = 100
    with pytest.raises(ValueError, match="source needs"):
        prepare_trace(options)


@pytest.mark.parametrize("damage", ["digest", "arrival", "duplicate", "vocab", "cap", "capacity", "identity"])
def test_bad_trace_fails_before_model_load(local_experiment, tmp_path, damage):
    options = options_for(local_experiment, tmp_path)
    trace = prepare_trace(options)
    if damage == "digest":
        trace["workload_sha256"] = "wrong"
    elif damage == "arrival":
        trace["requests"][-1]["arrival_s"] = -1
    elif damage == "duplicate":
        trace["requests"][-1]["id"] = "r0"
    elif damage == "vocab":
        trace["requests"][0]["prompt_token_ids"][0] = 32
    elif damage == "cap":
        trace["requests"][0]["max_tokens"] = True
    elif damage == "capacity":
        trace["requests"][0]["max_tokens"] = 40
    else:
        trace["model_config_sha256"] = "wrong"
    if damage != "digest":
        trace["workload_sha256"] = digest(trace["requests"])
    path = tmp_path / "broken.json"
    write_json(path, trace)
    options["trace"] = str(path)
    with pytest.raises(ValueError):
        experiment.preflight(options)


def test_cohort_metrics_preserve_tail_gaps_and_slo_semantics():
    specs = [{"id": "old", "group": "old_decode"}, {"id": "new", "group": "late_prefill"}]
    records = []
    for rid, arrival, admission, marks in [
        ("old", 0, 0, [.1, .2, .9, 1.0]), ("new", .25, .4, [.8]),
    ]:
        records.append({"id": rid, "arrival_s": arrival, "admission_s": admission,
                        "marks": marks, "output_tokens": len(marks),
                        **token_metrics(arrival, admission, marks)})
    run = {"summary": {"elapsed_s": 1.0}, "requests": records}
    stats = summarize_online(run, specs, window_s=.2, ttft_slo_ms=600, itl_slo_ms=600)
    assert stats["groups"]["old_decode"]["tpot_ms"]["mean"] == pytest.approx(300)
    assert stats["groups"]["late_prefill"]["ttft_ms"]["mean"] == pytest.approx(550)
    assert stats["groups"]["late_prefill"]["tpot_ms"]["n"] == 0
    assert stats["window_old_itl_ms"]["max"] == pytest.approx(700)  # Whole gap, not clipped.
    assert stats["window_old_itl_ms"]["n"] == 1
    assert stats["overlap"]["old_decoding_at_arrival"] == 1
    assert stats["overlap"]["old_decoding_at_admission"] == 1
    assert stats["groups"]["all"]["slo_pass_fraction"] == .5
    assert stats["groups"]["all"]["slo_goodput_requests_s"] == 1
    assert stats["groups"]["late_prefill"]["itl_ms"]["mean"] is None
    with pytest.raises(ValueError, match="complete workload"):
        summarize_online(run, specs[:1])


def test_policy_switch_keeps_kv_storage():
    engine = TimedEngine(Clock(), prefix_cache=True)
    engine.config = SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=4, max_model_len=32)
    manager = engine.model_runner.kv_cache_manager
    pointers = manager.k_pool.data_ptr(), manager.v_pool.data_ptr()
    for policy in experiment.POLICIES:
        experiment.fresh_policy(engine, policy)
        assert engine.model_runner.kv_cache_manager is manager
        assert (manager.k_pool.data_ptr(), manager.v_pool.data_ptr()) == pointers
        assert engine.config.scheduling_policy == policy


def test_real_cpu_three_policy_replays_and_reports(local_experiment, tmp_path):
    options = options_for(local_experiment, tmp_path, "--rounds", "3", "--warmup", "1",
                          "--save-details", "--ttft-slo-ms", "10000", "--itl-slo-ms", "10000")
    payload = experiment.execute(options)
    assert payload["status"] == "complete"
    assert len(payload["runs"]) == 9
    assert [r["policy"] for r in payload["runs"]] == [
        "separate", "interleaved", "mixed", "interleaved", "mixed", "separate", "mixed", "separate", "interleaved"]
    for run in payload["runs"]:
        assert run["summary"]["output_tokens"] == 10
        assert run["online"]["groups"]["old_decode"]["output_tokens"] == 8
        assert run["online"]["groups"]["late_prefill"]["output_tokens"] == 2
        assert len(run["details"]["requests"]) == 3
        assert len(run["details"]["outputs"]["r2"]) == 2
        assert run["details"]["steps"]
        for r in run["requests"]:
            assert r["output_tokens"] == r["max_tokens"]
            assert r["ttft_ms"] >= r["admission_delay_ms"] >= 0
            assert r["last_token_s"] >= r["first_token_s"]
    out = Path(options["out"])
    with (out / "requests.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 27
    with (out / "runs.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
        assert len(rows) == 9 and "online.window_old_itl_ms.max" in rows[0]
    assert json.loads((out / "results.json").read_text(encoding="utf-8"))["plan"] == payload["plan"]
    assert (out / "summary.csv").exists() and (out / "report.md").exists()
    assert not (out / "failed.json").exists()
    with pytest.raises(ValueError, match="new or empty"):
        experiment.execute(options)


def test_dry_run_never_constructs_engine(local_experiment, tmp_path, monkeypatch):
    import nanovllm

    def forbidden(*args, **kwargs):
        pytest.fail("dry run constructed model")

    monkeypatch.setattr(nanovllm, "LLM", forbidden)
    options = options_for(local_experiment, tmp_path, "--dry-run")
    assert experiment.execute(options)["status"] == "planned"
    assert not Path(options["out"]).exists()


def test_failure_saves_partial_runs_and_releases_engine(local_experiment, tmp_path, monkeypatch):
    import nanovllm

    options = options_for(local_experiment, tmp_path)
    original_measure, constructor = experiment.measure, nanovllm.LLM
    instances = []

    def construct(*args, **kwargs):
        instance = constructor(*args, **kwargs)
        instances.append(instance)
        return instance

    def fail_second(*args, **kwargs):
        if len(instances) and instances[0].config.scheduling_policy == "separate" and fail_second.called:
            raise TimeoutError("injected online timeout")
        fail_second.called = True
        return original_measure(*args, **kwargs)

    fail_second.called = False
    monkeypatch.setattr(nanovllm, "LLM", construct)
    monkeypatch.setattr(experiment, "measure", fail_second)
    with pytest.raises(TimeoutError):
        experiment.execute(options)
    failed = json.loads((Path(options["out"]) / "failed.json").read_text())
    assert len(failed["completed_runs"]) == 1 and failed["status"] == "failed"
    assert not (Path(options["out"]) / "results.json").exists()
    assert instances[0].model_runner is None and instances[0].scheduler is None


@pytest.mark.parametrize("profile", ["burst", "interval", "poisson"])
def test_experiment_cli_and_diagnostics(local_experiment, tmp_path, profile):
    options = options_for(local_experiment, tmp_path, "--policies", "mixed", "--diagnostics",
                          "--profile", profile, "--request-count", "3", "--arrival-rate", "1000",
                          "--prompt-lengths", "3,6", "--output-lengths", "4,2")
    config = tmp_path / "online-config.json"
    write_json(config, options)
    result = subprocess.run([sys.executable, str(Path(experiment.__file__)), "--config", str(config)],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads((Path(options["out"]) / "results.json").read_text(encoding="utf-8"))
    assert payload["diagnostics"]["enabled"]
    assert len(payload["runs"]) == 1 and "details" not in payload["runs"][0]
    assert payload["runs"][0]["summary"]["process_cpu_s"] >= 0


def dataset_path(local_experiment):
    return json.loads(local_experiment[1].read_text())["dataset"]


@pytest.mark.parametrize("profile", ["interval", "poisson"])
def test_dataset_preserves_original_prompts_and_selection(local_experiment, tmp_path, profile):
    from transformers import AutoTokenizer
    from experiments.math500.run_math500 import encode_problem

    options = options_for(local_experiment, tmp_path, "--profile", profile,
                          "--dataset", dataset_path(local_experiment), "--num-problems", "2",
                          "--offset", "1", "--max-tokens", "3", "--arrival-rate", "2")
    trace = prepare_trace(options)
    assert trace == prepare_trace(options)
    tokenizer = AutoTokenizer.from_pretrained(options["model"], local_files_only=True)
    rows = [json.loads(line) for line in Path(options["dataset"]).read_text().splitlines()]
    assert [r["dataset_index"] for r in trace["requests"]] == [1, 2]
    for r in trace["requests"]:
        assert r["prompt_token_ids"] == encode_problem(tokenizer, rows[r["dataset_index"]]["problem"])
        assert r["max_tokens"] == 3 and r["group"] == "requests"
        assert "answer" not in r and "solution" not in r
    assert trace["synthetic"] is False
    assert trace["generation"]["offset"] == 1
    assert trace["requests"][1]["arrival_s"] > 0
    if profile == "interval":
        assert trace["requests"][1]["arrival_s"] == .5
    options["request_count"] = 3
    with pytest.raises(ValueError, match="dataset has"):
        prepare_trace(options)


def test_dataset_shuffle_and_trace_override(local_experiment, tmp_path):
    options = options_for(local_experiment, tmp_path, "--profile", "poisson",
                          "--dataset", dataset_path(local_experiment), "--request-count", "3",
                          "--shuffle", "--max-tokens", "3")
    trace = prepare_trace(options)
    assert prepare_trace(options) == trace
    assert sorted(r["dataset_index"] for r in trace["requests"]) == [0, 1, 2]
    path = tmp_path / "frozen.json"
    write_json(path, trace)
    options.update(trace=str(path), dataset="does-not-exist.jsonl", max_tokens=999, seed=1)
    assert experiment.preflight(options)[0] == trace


def test_source_stream_keeps_tokens_and_uses_uniform_output_cap(local_experiment, tmp_path):
    options = options_for(local_experiment, tmp_path)
    source = prepare_trace(options)
    path = tmp_path / "source.json"
    write_json(path, source)
    options.update(profile="interval", source=str(path), offset=1, request_count=2, max_tokens=3)
    trace = prepare_trace(options)
    assert [r["prompt_token_ids"] for r in trace["requests"]] == [r["prompt_token_ids"] for r in source["requests"][1:]]
    assert [r["max_tokens"] for r in trace["requests"]] == [3, 3]
    assert {r["group"] for r in trace["requests"]} == {"requests"}


def test_first_100_dataset_requests_real_three_policy_run(local_experiment, tmp_path):
    original = [json.loads(line) for line in Path(dataset_path(local_experiment)).read_text().splitlines()]
    rows = [{**original[i % len(original)], "unique_id": str(i)} for i in range(101)]
    dataset = tmp_path / "math100.jsonl"
    dataset.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    options = options_for(local_experiment, tmp_path, "--profile", "interval", "--dataset", str(dataset),
                          "--request-count", "100", "--max-tokens", "3", "--arrival-rate", "1000",
                          "--blocks", "256", "--max-seqs", "16", "--token-budget", "32")
    payload = experiment.execute(options)
    assert [r["dataset_index"] for r in payload["trace"]["requests"]] == list(range(100))
    assert len(payload["runs"]) == 3
    for run in payload["runs"]:
        all_requests = run["online"]["groups"]["all"]
        assert all_requests["requests"] == 100
        assert all_requests["output_tokens"] == 300
        assert all_requests["tpot_ms"]["n"] == 100
        assert all_requests["itl_ms"]["n"] == 200
    report = (Path(options["out"]) / "report.md").read_text(encoding="utf-8")
    assert "整体延迟分布" in report
    assert "| 请求组 |" not in report and "干扰窗口老请求" not in report
