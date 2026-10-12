"""Round ordering, raw sample retention, and robust repeat summaries."""

import json
from pathlib import Path

import pytest

from experiments.benchmark import digest, distribution, write_json
from experiments.common import measurement
from experiments.math500 import run_math500 as experiment
from experiments.common.repeat_stats import summarize_repeats
from test_math500_experiments import direct_worker, local_experiment


def test_trimmed_mean_retains_full_sample_variance_and_extremes():
    result = summarize_repeats([0, 60, 65, 68, 69, 70, 71, 73, 75, 1000], .1)
    assert result["n"] == 10 and result["trimmed_each_tail"] == 1
    assert result["mean"] == pytest.approx(155.1)
    assert result["trimmed_mean"] == pytest.approx(68.875)
    assert result["median"] == 69.5
    assert result["min"] == 0 and result["max"] == 1000
    assert result["std"] > 250 and result["cv_pct"] > 150
    regular = summarize_repeats([1, 2, 3], .1)
    assert regular["std"] == 1 and regular["cv_pct"] == 50
    assert regular["trimmed_each_tail"] == 0 and regular["trimmed_mean"] == 2


def test_missing_and_single_samples_do_not_invent_variance():
    empty = summarize_repeats([None, float("nan"), float("inf")], .1)
    assert empty["n"] == 0 and empty["mean"] is None and empty["std"] is None
    single = summarize_repeats([7], .1)
    assert single["median"] == single["trimmed_mean"] == 7
    assert single["std"] is None and single["cv_pct"] is None
    zeros = summarize_repeats([0, 0], 0)
    assert zeros["std"] == 0 and zeros["cv_pct"] is None


@pytest.mark.parametrize("key,value", [
    ("repeat_schedule", "unknown"), ("repeat_schedule", True),
    ("policy_orders", []), ("policy_orders", ""), ("policy_orders", ["separate"]),
    ("policy_orders", [["separate", "mixed"]]),
    ("policy_orders", [["separate", "interleaved", "mixed", "mixed"]]),
    ("policy_orders", [["separate", "interleaved", "bogus"]]),
    ("policy_orders", [[True]]),
    ("trim_fraction", -.1), ("trim_fraction", .5), ("trim_fraction", True),
    ("trim_fraction", float("nan")), ("trim_fraction", float("inf")),
])
def test_invalid_order_and_trimming_options_fail_before_loading(tmp_path, key, value):
    config = tmp_path / "invalid.json"
    write_json(config, {key: value})
    with pytest.raises(ValueError):
        experiment.parse_options(["--config", str(config)])


def test_grouped_order_cli_and_single_policy_override(local_experiment):
    _, config = local_experiment
    options = experiment.parse_options([
        "--config", str(config), "--policy-orders", "mixed,separate,interleaved",
        "--trim-fraction", "0.2"])
    jobs, _ = experiment.build_jobs(options, experiment.prepare_workload(options))
    assert [j["factors"]["policy"] for j in jobs[:6]] == ["mixed", "separate", "interleaved"] * 2
    assert all(j["round"] is None for j in jobs)
    subset = experiment.parse_options([
        "--config", str(config), "--policies", "interleaved", "--repeat-schedule", "round_robin",
        "--policy-orders", "mixed,separate,interleaved;separate,interleaved,mixed", "--repeats", "2"])
    assert subset["policy_orders"] == [["interleaved"], ["interleaved"]]
    jobs, _ = experiment.build_jobs(subset, experiment.prepare_workload(subset))
    assert [j["round"] for j in jobs[:2]] == [1, 2]


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_round_order_outputs_statistics_diagnostics_and_resume(local_experiment, monkeypatch, dtype):
    import nanovllm

    _, config = local_experiment
    calls, engines, replays = [], [], []
    original_llm, original_replay, original_write = nanovllm.LLM, measurement.replay, measurement.write_json

    def load_once(*args, **kwargs):
        engine = original_llm(*args, **kwargs)
        engines.append(engine)
        return engine

    def observe_replay(engine, *args, **kwargs):
        manager = engine.model_runner.kv_cache_manager
        assert engine.scheduler.next_phase == "prefill"
        assert not engine.scheduler.seen_ids and not manager.block_manager.tables
        assert not manager.block_manager.hash_to_block_id
        replays.append((engine.scheduler.scheduling_policy, args[2], id(manager.k_pool)))
        return original_replay(engine, *args, **kwargs)

    def write_after_all_rounds(path, result):
        assert len(replays) == 21  # Three warmups, nine cases with both timings.
        original_write(path, result)

    def worker(job, job_path, log_path):
        calls.append([(m["round"], m["policy_position"], m["factors"]["policy"]) for m in job["members"]])
        direct_worker(job, job_path, log_path)

    monkeypatch.setattr(nanovllm, "LLM", load_once)
    monkeypatch.setattr(measurement, "replay", observe_replay)
    monkeypatch.setattr(measurement, "write_json", write_after_all_rounds)
    monkeypatch.setattr(experiment, "run_worker_process", worker)
    arguments = ["--config", str(config), "--max-tokens", "3", "--max-seqs", "2",
                 "--repeat-schedule", "round_robin", "--repeats", "3", "--warmup", "1", "--diagnostics",
                 "--dtype", dtype,
                 "--policy-orders", "mixed,separate,interleaved;interleaved,mixed,separate"]
    report, code = experiment.execute(experiment.parse_options(arguments))
    assert code == 0
    expected_order = [(r, p, policy) for r, order in enumerate([
        ["mixed", "separate", "interleaved"], ["interleaved", "mixed", "separate"],
        ["mixed", "separate", "interleaved"]], 1) for p, policy in enumerate(order, 1)]
    assert calls == [expected_order] and len(engines) == 1
    assert len({pool for _, _, pool in replays}) == 1
    assert [(policy, timing) for policy, timing, _ in replays] == (
        [(policy, "throughput") for policy in ["mixed", "separate", "interleaved"]]
        + [(policy, timing) for _, _, policy in expected_order for timing in ["throughput", "latency"]])
    assert len(report["summary"]) == 6 and len(report["runs"]) == 18
    assert all(c["complete"] and c["case_count"] == c["expected_case_count"] == 3
               for c in report["output_comparisons"])
    assert len({r["output_sha256"] for r in report["runs"]}) == 1
    for row in report["summary"]:
        assert row["repeats_completed"] == row["repeats_expected"] == row["output_tok_s_n"] == 3
        assert row["output_tok_s"] == row["output_tok_s_median"]
        assert row["output_tok_s_min"] <= row["output_tok_s_mean"] <= row["output_tok_s_max"]
        assert row["output_tok_s_trimmed_each_tail"] == 0
        assert row["ttft_mean_ms_n"] == (3 if row["timing"] == "latency" else 0)
    assert all(r["repeat"] == r["round"] and r["process_cpu_s"] >= 0 for r in report["runs"])
    out = Path(report["settings"]["out"])
    assert len(list((out / "cases").glob("*.json"))) == 9
    cases = [json.loads(p.read_text(encoding="utf-8")) for p in (out / "cases").glob("*.json")]
    assert sum(len(case["warmup_runs"]) for case in cases) == 3
    assert all(len(case["warmup_runs"]) == (1 if case["round"] == 1 else 0) for case in cases)
    assert len({r["worker_id"] for r in report["runs"]}) == 1
    assert "截尾均值" in (out / "report.md").read_text(encoding="utf-8")
    assert "output_tok_s_cv_pct" in (out / "summary.csv").read_text(encoding="utf-8-sig")
    assert "worker_case_id" in (out / "runs.csv").read_text(encoding="utf-8-sig")
    monkeypatch.setattr(experiment, "run_worker_process", lambda *args: pytest.fail("resume relaunched a completed round"))
    resumed, code = experiment.execute(experiment.parse_options(arguments + ["--resume"]))
    assert code == 0 and resumed["summary"] == report["summary"]


def test_missing_round_is_partial_and_cross_round_hash_changes_are_unstable(local_experiment):
    _, config = local_experiment
    options = experiment.parse_options([
        "--config", str(config), "--policies", "mixed", "--max-tokens", "1", "--max-seqs", "1",
        "--timings", "throughput", "--repeat-schedule", "round_robin", "--repeats", "2"])
    workload = experiment.prepare_workload(options)
    jobs, _ = experiment.build_jobs(options, workload)
    summary = {"elapsed_s": 1, "requests": 3, "output_tokens": 3, "output_tok_s": 3,
               "counts": {}, **{name: distribution([]) for name in experiment.LATENCIES}}
    results = {jobs[0]["case_id"]: {"status": "ok", "runs": [
        {"timing": "throughput", "repeat": 1, "summary": summary, "output_sha256": digest("first")}]}}
    partial = experiment.make_report(options, workload, jobs, results)
    assert partial["summary"][0]["status"] == "partial"
    assert partial["summary"][0]["repeats_completed"] == 1
    assert not partial["output_comparisons"][0]["complete"]
    results[jobs[1]["case_id"]] = {"status": "ok", "runs": [
        {"timing": "throughput", "repeat": 2, "summary": summary, "output_sha256": digest("second")}]}
    unstable = experiment.make_report(options, workload, jobs, results)
    assert unstable["summary"][0]["status"] == "unstable"
    assert not unstable["output_comparisons"][0]["matching_outputs"]
    assert unstable["output_comparisons"][0]["first_mismatch"]["actual_repeat"] == 2


def test_ten_repeat_report_keeps_extremes_and_exports_exact_trimming(local_experiment):
    _, config = local_experiment
    options = experiment.parse_options([
        "--config", str(config), "--policies", "mixed", "--max-tokens", "1", "--max-seqs", "1",
        "--timings", "throughput", "--repeats", "10"])
    workload = experiment.prepare_workload(options)
    jobs, _ = experiment.build_jobs(options, workload)
    rates = [10, 60, 65, 68, 69, 70, 71, 73, 75, 1000]
    runs = [{"timing": "throughput", "repeat": i,
             "summary": {"elapsed_s": 3 / rate, "requests": 3, "output_tokens": 3, "output_tok_s": rate,
                         "counts": {}, **{name: distribution([]) for name in experiment.LATENCIES}},
             "output_sha256": digest("same outputs")} for i, rate in enumerate(rates, 1)]
    report = experiment.make_report(options, workload, jobs, {jobs[0]["case_id"]: {"status": "ok", "runs": runs}})
    row = report["summary"][0]
    assert row["output_tok_s_trimmed_each_tail"] == 1 and row["output_tok_s_n"] == 10
    assert row["output_tok_s_trimmed_mean"] == pytest.approx(68.875)
    assert row["output_tok_s_min"] == 10 and row["output_tok_s_max"] == 1000
    assert row["output_tok_s_std"] > 250
    assert row["ttft_mean_ms_n"] == 0 and row["ttft_mean_ms_trimmed_mean"] is None
    assert [r["output_tok_s"] for r in report["runs"]] == rates
    out = Path(options["out"])
    out.mkdir()
    experiment.save_report(out, report)
    saved = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert len(saved["runs"]) == 10 and saved["summary"][0]["output_tok_s_max"] == 1000


def test_round_mode_with_native_worker_subprocesses(local_experiment, monkeypatch):
    _, config = local_experiment
    calls = []
    original_worker = experiment.run_worker_process

    def launch(job, job_path, log_path):
        calls.append(job["case_id"])
        return original_worker(job, job_path, log_path)

    monkeypatch.setattr(experiment, "run_worker_process", launch)
    report, code = experiment.execute(experiment.parse_options([
        "--config", str(config), "--policies", "mixed,separate", "--max-tokens", "1", "--max-seqs", "1",
        "--timings", "throughput", "--repeat-schedule", "round_robin", "--repeats", "2",
        "--policy-orders", "mixed,separate;separate,mixed"]))
    assert code == 0 and len(report["runs"]) == 4 and len(report["summary"]) == 2
    assert len(calls) == 1 and len({r["worker_id"] for r in report["runs"]}) == 1
    assert [(r["round"], r["policy_position"]) for r in report["runs"]] == [(1, 1), (2, 2), (1, 2), (2, 1)]
    assert all(c["complete"] and c["matching_outputs"] for c in report["output_comparisons"])
    for path in (Path(report["settings"]["out"]) / "cases").glob("*.json"):
        case = json.loads(path.read_text(encoding="utf-8"))
        assert case["worker_attempts"][0]["exit_code"] == 0


def test_resume_missing_round_replays_entire_resident_worker(local_experiment, monkeypatch):
    _, config = local_experiment
    calls = []

    def worker(job, job_path, log_path):
        calls.append([member["case_id"] for member in job["members"]])
        direct_worker(job, job_path, log_path)

    monkeypatch.setattr(experiment, "run_worker_process", worker)
    arguments = ["--config", str(config), "--max-tokens", "1", "--max-seqs", "1",
                 "--timings", "throughput", "--repeat-schedule", "round_robin", "--repeats", "2"]
    report, code = experiment.execute(experiment.parse_options(arguments))
    assert code == 0 and len(calls) == 1
    out = Path(report["settings"]["out"])
    (out / "cases" / f"{calls[0][-1]}.json").unlink()
    resumed, code = experiment.execute(experiment.parse_options(arguments + ["--resume"]))
    assert code == 0 and calls == [calls[0], calls[0]]
    assert len(resumed["runs"]) == 6
    assert all(c["complete"] for c in resumed["output_comparisons"])


def test_resident_worker_failure_does_not_accept_stale_rounds(local_experiment, monkeypatch):
    _, config = local_experiment
    monkeypatch.setattr(experiment, "run_worker_process", direct_worker)
    arguments = ["--config", str(config), "--max-tokens", "1", "--max-seqs", "1",
                 "--timings", "throughput", "--repeat-schedule", "round_robin", "--repeats", "2"]
    report, code = experiment.execute(experiment.parse_options(arguments))
    assert code == 0
    out = Path(report["settings"]["out"])
    next((out / "cases").glob("*.json")).unlink()

    def failure(*args):
        raise RuntimeError("resident worker failed after starting")

    monkeypatch.setattr(experiment, "run_worker_process", failure)
    failed, code = experiment.execute(experiment.parse_options(arguments + ["--resume"]))
    assert code == 1 and not failed["runs"]
    assert all(row["status"] == "failed" for row in failed["summary"])
    assert len(list((out / "cases").glob("*.json"))) == 6
    assert all(json.loads(p.read_text(encoding="utf-8"))["status"] == "failed"
               for p in (out / "cases").glob("*.json"))
