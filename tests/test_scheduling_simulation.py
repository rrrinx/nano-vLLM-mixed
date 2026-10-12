"""Independent hand counts and a real Scheduler comparison for round estimates."""

import csv
import json

import pytest

from experiments.simulation.scheduler import Limits, RequestSpec, RoundSimulator, compare, main, simulate
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from test_chunked_prefill import execute_plan, make_chunk_scheduler


def requests(lengths):
    return [RequestSpec(str(i), prompt, output) for i, (prompt, output) in enumerate(lengths)]


@pytest.mark.parametrize("policy", ["separate", "interleaved", "mixed"])
def test_single_request_hand_count_and_first_prefill_token(policy):
    result = simulate(requests([(10, 4)]), Limits(8, 3), policy, record_steps=True)
    assert result["status"] == "ok"
    # Four prefill chunks (3,3,3,1), then three decode rounds; not four decodes.
    assert result["forward_rounds"] == 7
    assert result["query_tokens"] == 13
    assert result["requests"][0]["first_token_round"] == 4
    assert [step["query_tokens"] for step in result["steps"]] == [3, 3, 3, 1, 1, 1, 1]


def test_mixing_saves_two_rounds_when_short_outputs_overlap_decode():
    results = compare(requests([(3, 4), (3, 1), (3, 1)]), Limits(4, 4), record_steps=True)
    assert [r["forward_rounds"] for r in results] == [6, 6, 4]
    assert results[2]["round_reduction_pct"] == pytest.approx(100 / 3)
    assert results[2]["mixed_rounds"] == 2
    assert [r["query_tokens"] for r in results] == [12, 12, 12]


def test_chunking_can_increase_rounds_so_report_negative_savings():
    results = compare(requests([(8, 4), (8, 4)]), Limits(2, 8))
    assert [r["forward_rounds"] for r in results] == [5, 6, 6]
    assert results[2]["round_reduction_pct"] == -20


def test_only_first_prefill_can_chunk_inside_sub_batch():
    result = simulate(requests([(3, 1), (3, 1)]), Limits(2, 4), "separate", record_steps=True)
    assert result["forward_rounds"] == 2
    assert [s["query_tokens"] for s in result["steps"]] == [3, 3]


@pytest.mark.parametrize("limits", [Limits(1, 16), Limits(8, 1)])
def test_single_slot_or_single_token_cannot_produce_mixed_batches(limits):
    result = simulate(requests([(3, 4), (3, 4)]), limits, "mixed")
    assert result["status"] == "ok" and result["mixed_rounds"] == 0


def test_running_order_is_preserved_instead_of_rotated():
    result = simulate(requests([(1, 3)] * 3), Limits(1, 8), "separate", record_steps=True)
    decoded = [s["entries"][0]["id"] for s in result["steps"] if s["kind"] == "decode"]
    assert decoded == ["0", "0", "1", "1", "2", "2"]


def test_idle_arrivals_and_zero_output_are_not_counted_as_forwards():
    specs = [RequestSpec("a", 1, 1), RequestSpec("b", 1, 1, 10), RequestSpec("c", 9, 0, 12)]
    result = simulate(specs, Limits(2, 4), "mixed")
    assert (result["forward_rounds"], result["idle_rounds"], result["elapsed_rounds"]) == (2, 10, 12)
    assert [r["finish_round"] for r in result["requests"]] == [1, 11, 12]
    assert result["requests"][2]["first_token_round"] is None


def test_full_kv_pool_allows_decode_until_block_boundary_then_recomputes():
    result = simulate(requests([(3, 4), (3, 4)]), Limits(2, 8, 4, 2),
                      "separate", record_steps=True)
    assert result["status"] == "ok" and result["forward_rounds"] == 6
    assert result["preemptions"] == 1 and result["extra_recompute_tokens"] == 4
    assert result["query_tokens"] == 16 and result["peak_kv_blocks"] == 2
    assert result["steps"][0]["free_blocks_after_selection"] == 0
    assert result["steps"][1]["kind"] == "decode"


@pytest.mark.parametrize("policy", ["interleaved", "mixed"])
def test_guide_recovery_unblocks_reserved_partial_prefill(policy):
    result = simulate(requests([(3, 4), (8, 1)]), Limits(2, 4, 4, 3), policy)
    assert result["status"] == "ok" and result["forward_rounds"] == 7
    assert result["reserved_prefill_rescues"] == 1 and result["preemptions"] == 1
    assert result["extra_recompute_tokens"] == 4


def test_infeasible_capacity_and_round_limit_do_not_report_a_gain():
    results = compare(requests([(8, 4)]), Limits(2, 8, 4, 2))
    assert all(r["status"] == "infeasible" and r["round_reduction_pct"] is None for r in results)
    limited = simulate(requests([(8, 4)]), Limits(2, 8), "mixed", max_rounds=1)
    assert limited["status"] == "round_limit" and limited["forward_rounds"] == 1


@pytest.mark.parametrize("budget,count,blocks,lengths", [
    (1, 2, 16, [(1, 0), (2, 1), (3, 4), (7, 3)]),
    (3, 1, 16, [(3, 4), (7, 3)]),
    (8, 2, 16, [(3, 4), (7, 3), (2, 1)]),
    (8, 2, 2, [(3, 4), (3, 4)]),
])
def test_separate_matches_current_production_scheduler(budget, count, blocks, lengths):
    scheduler, runner = make_chunk_scheduler(budget=budget, size=4, blocks=blocks,
                                           count=count, enabled=False)
    for i, (prompt, output) in enumerate(lengths):
        scheduler.add(Sequence(i, list(range(prompt)), output))
    actual_rounds, actual_query_tokens = 0, 0
    while not scheduler.is_finished():
        assert actual_rounds < 100
        plan = scheduler.schedule()
        actual_query_tokens += plan.query_tokens
        execute_plan(scheduler, runner, plan)
        actual_rounds += 1
    result = simulate(requests(lengths), Limits(count, budget, 4, blocks), "separate")
    assert result["status"] == "ok"
    assert (result["forward_rounds"], result["query_tokens"]) == (actual_rounds, actual_query_tokens)


def test_cli_writes_reusable_json_csv_and_step_trace(tmp_path, capsys):
    source, out = tmp_path / "trace.json", tmp_path / "results.json"
    source.write_text(json.dumps({"requests": [
        {"id": "a", "prompt_tokens": 3, "output_tokens": 4},
        {"id": "b", "prompt_tokens": 3, "output_tokens": 1},
        {"id": "c", "prompt_tokens": 3, "output_tokens": 1},
    ]}), encoding="utf-8-sig")
    assert main(["--trace-file", str(source), "--max-num-seqs", "4", "--token-budgets", "4",
                 "--record-steps", "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert [r["forward_rounds"] for r in report["comparisons"][0]["results"]] == [6, 6, 4]
    with out.with_suffix(".csv").open(encoding="utf-8-sig", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 3
    assert "Not a runtime speedup" in capsys.readouterr().out


def test_physical_block_boundary_and_tail_waste_are_visible():
    result = simulate(requests([(127, 3)]), Limits(4, 2048, 128, 78),
                      "separate", record_steps=True)
    assert [step["entries"][0]["block_table"] for step in result["steps"]] == [[0], [0], [0, 1]]
    assert result["steps"][0]["free_blocks_after_selection"] == 77
    assert result["steps"][2]["free_blocks_after_selection"] == 76
    assert result["steps"][-1]["free_blocks_after_commit"] == 78
    assert result["block_allocations"] == result["peak_kv_blocks"] == 2


def test_identical_lengths_do_not_imply_identical_prompts():
    result = simulate(requests([(256, 1)] * 2), Limits(1, 2048, 128, 78, 1024, True), "separate")
    assert result["prefix_hit_tokens"] == result["prefix_saved_query_tokens"] == 0


def test_shared_prefix_uses_real_reference_counts_and_cold_pages():
    specs = [RequestSpec(str(i), 257, 4, shared_prefix_tokens=256) for i in range(2)]
    result = simulate(specs, Limits(1, 2048, 128, 78, 1024, True),
                      "separate", record_steps=True)
    assert result["status"] == "ok" and result["prefix_saved_query_tokens"] == 256
    assert result["query_tokens"] == result["base_query_tokens"] - 256
    second_prefill = result["steps"][1]
    assert [e["block_table"] for e in second_prefill["entries"]] == [[0, 1, 3]]
    assert second_prefill["block_ref_counts_after_commit"][:4] == [2, 2, 1, 1]
    assert result["peak_kv_blocks"] == 4 and result["peak_block_references"] == 6
    assert result["cold_cached_blocks_at_finish"] > 0
    assert not any(result["steps"][-1]["block_ref_counts_after_commit"])


def test_real_max_model_len_admission_is_distinct_from_pool_capacity():
    result = simulate(requests([(768, 512)]), Limits(32, 2048, 128, 78, 1024, True), "mixed")
    assert result["status"] == "infeasible" and "max_model_len" in result["error"]


def test_admission_cap_is_independent_of_actual_output_and_block_allocation():
    limits = Limits(32, 2048, 128, 78, 1024, True)
    low = compare([RequestSpec("0", 127, 3, max_tokens=3)], limits, record_steps=True)
    high = compare([RequestSpec("0", 127, 3, max_tokens=256)], limits, record_steps=True)
    assert low == high  # Including physical blocks and every round's trace.
    assert all(r["peak_kv_blocks"] == 2 for r in high)
    invalid = compare([RequestSpec("0", 127, 3, max_tokens=900)], limits)
    assert all(r["status"] == "infeasible" and "max_model_len" in r["error"] for r in invalid)
    invalid_pool = compare([RequestSpec("0", 127, 3, max_tokens=256)], Limits(4, 2048, 128, 2))
    assert all(r["status"] == "infeasible" and "KV capacity" in r["error"] for r in invalid_pool)


def test_real_prompt_ids_are_retained_and_enable_exact_prefix_hits():
    common = list(range(128))
    specs = [RequestSpec(str(i), 129, 2, max_tokens=64,
                         prompt_token_ids=common + [1000 + i]) for i in range(2)]
    simulator = RoundSimulator(specs, Limits(1, 2048, 128, 78, 1024, True), "separate")
    assert [s.token_ids for s in simulator.states] == [common + [1000], common + [1001]]
    assert min(s.token_base for s in simulator.states) > 1001
    result = simulator.run()
    assert result["status"] == "ok"
    assert result["prefix_saved_query_tokens"] == 128


@pytest.mark.parametrize("kwargs", [
    {"output_tokens": 5, "max_tokens": 4},
    {"output_tokens": 1, "max_tokens": -1},
    {"output_tokens": 1, "prompt_token_ids": [1]},
    {"output_tokens": 1, "prompt_token_ids": [1, -2]},
    {"output_tokens": 1, "prompt_token_ids": [1, 2], "shared_prefix_tokens": 1},
])
def test_real_prompt_and_cap_validation(kwargs):
    with pytest.raises(ValueError):
        RequestSpec("a", 2, **kwargs)


def test_math500_template_matches_existing_entrypoint():
    from experiments.simulation.math500 import encode_problem

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "question"}]
            assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
            return "rendered question"

        def encode(self, rendered):
            assert rendered == "rendered question"
            return [11, 12, 13]

    assert encode_problem(Tokenizer(), "question") == [11, 12, 13]


def test_math500_matrix_reports_invalid_batches_and_common_cohort(tmp_path, capsys):
    from experiments.simulation.math500 import main as math_main, sha256

    dataset, model, cache = tmp_path / "test.jsonl", tmp_path / "model", tmp_path / "prompts.json"
    model.mkdir()
    dataset.write_text("dataset fingerprint", encoding="utf-8")
    (model / "tokenizer.json").write_text("tokenizer fingerprint", encoding="utf-8")
    rows = [{"id": str(i), "prompt_token_ids": list(range(length))} for i, length in enumerate((3, 7, 1))]
    cache.write_text(json.dumps({
        "schema": 1, "fingerprints": {"dataset": sha256(dataset),
                                        "tokenizer_files": {"tokenizer.json": sha256(model / "tokenizer.json")}},
        "length_stats": {"count": 3, "min": 1, "median": 3, "mean": 11 / 3, "p95": 7, "max": 7},
        "prompts": rows,
    }), encoding="utf-8")
    out, report_path = tmp_path / "results.json", tmp_path / "report.md"
    assert math_main([
        "--dataset", str(dataset), "--model", str(model), "--prompt-cache", str(cache),
        "--requests", "3", "--output-lengths", "1,4", "--max-tokens", "1,4,8",
        "--max-num-seqs", "1,2", "--token-budgets", "4,8", "--block-size", "4", "--num-blocks", "8",
        "--max-model-len", "8", "--out", str(out), "--report", str(report_path),
    ]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["cohorts"][0]["common_count"] == 1
    assert report["cohorts"][0]["invalid_by_cap"] == {"1": [], "4": ["1"], "8": ["0", "1"]}
    assert len(report["comparisons"]) == 48
    assert {c["limits"]["token_budget"] for c in report["comparisons"]} == {4, 8}
    for case in report["comparisons"]:
        invalid = case["factors"]["cohort"] == "original" and case["factors"]["max_tokens"] > 1
        assert all(r["status"] == ("infeasible" if invalid else "ok") for r in case["results"])
        if invalid:
            assert all(r["round_reduction_pct"] is None for r in case["results"])
    assert report["cap_invariance"]["groups_with_multiple_valid_caps"] > 0
    assert report["cap_invariance"]["identical_rounds"] == report["cap_invariance"]["groups_with_multiple_valid_caps"]
    from experiments.simulation.math500 import merge_reports
    merge_reports(report, [out])
    assert len(report["comparisons"]) == 48  # An overlapping sweep is not counted twice.
    assert "Math500" in report_path.read_text(encoding="utf-8")
    capsys.readouterr()


def test_surviving_cold_prefix_reduces_recompute_after_preemption():
    specs = requests([(3, 4)] * 3)
    cached = simulate(specs, Limits(3, 16, 4, 3, 0, True), "separate")
    uncached = simulate(specs, Limits(3, 16, 4, 3), "separate")
    assert cached["status"] == uncached["status"] == "ok"
    assert cached["prefix_hit_tokens"] > 0
    assert cached["extra_recompute_tokens"] < cached["discarded_cached_tokens"]
    assert cached["query_tokens"] < uncached["query_tokens"]


def test_experiment_matrix_sweeps_lengths_independently_and_writes_report(tmp_path, capsys):
    from experiments.simulation.sweep import main as experiment_main

    out, report_path = tmp_path / "matrix.json", tmp_path / "report.md"
    assert experiment_main([
        "--prompt-lengths", "3,8", "--output-lengths", "1,4", "--requests", "2",
        "--max-num-seqs", "1,2", "--token-budget", "8", "--block-size", "4", "--num-blocks", "8",
        "--max-model-len", "64", "--no-prefix-cache", "--no-supplemental", "--no-record-example",
        "--out", str(out), "--report", str(report_path),
    ]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert len(report["comparisons"]) == report["summary"]["complete_comparisons"] == 8
    assert len(report["workloads"]) == 4
    for comparison in report["comparisons"]:
        workload = report["workloads"][comparison["workload_id"]]
        assert {r["prompt_tokens"] for r in workload} == {comparison["factors"]["prompt_tokens"]}
        assert {r["output_tokens"] for r in workload} == {comparison["factors"]["output_tokens"]}
    assert "4×8" in report_path.read_text(encoding="utf-8")
    capsys.readouterr()


class MetadataCache:
    """Real production allocator and publication, with successful KV layer lengths.

    No numerical forward is needed to independently compare allocation/schedule
    traces. The actual Scheduler performs all transitions and token appends.
    """

    def __init__(self, limits):
        self.block_size, self.num_blocks = limits.block_size, limits.num_blocks
        self.block_manager = BlockManager(self.num_blocks, self.block_size, limits.enable_prefix_cache)
        self.layer_lengths = {}

    def register(self, seqs):
        self.block_manager.register(seqs)
        for seq in seqs:
            self.layer_lengths[seq.seq_id] = [0]

    def lengths(self, sid):
        return self.layer_lengths[sid][0]

    def can_allocate_prefill(self, seq):
        return self.block_manager.can_allocate_prefix(seq)

    def allocate_prefill(self, seq, hits):
        c = self.block_manager.allocate_prefix(seq, hits)
        self.layer_lengths[seq.seq_id] = [c]
        return c

    def can_allocate_decode(self, seq):
        return self.block_manager.can_allocate(seq)

    def allocate_decode(self, seq):
        return self.block_manager.may_allocate(seq)

    def publish_seq(self, seq):
        assert self.layer_lengths[seq.seq_id] == [seq.num_cached_tokens + seq.num_scheduled_tokens]
        self.block_manager.hash_blocks(seq)

    def release(self, sid):
        self.block_manager.release(sid)
        self.layer_lengths.pop(sid, None)


def test_real_prompt_ids_and_early_eos_match_current_scheduler_round_by_round():
    specs = [RequestSpec(str(i), 129, length, max_tokens=64,
                         prompt_token_ids=list(range(128)) + [1000 + i])
             for i, length in enumerate((3, 7, 1))]
    limits = Limits(1, 2048, 128, 4, 512, True)
    simulator = RoundSimulator(specs, limits, "separate", record_steps=True)
    cache = MetadataCache(limits)
    scheduler = Scheduler(cache, limits.max_num_seqs, limits.token_budget, limits.max_model_len)
    actual_seqs = [Sequence(i, list(state.token_ids), 64, eos_token_id=-1)
                   for i, state in enumerate(simulator.states)]
    for seq in actual_seqs:
        scheduler.add(seq)
    result = simulator.run()
    assert result["status"] == "ok"
    for expected in result["steps"]:
        plan = scheduler.schedule()
        selected = plan.seqs
        assert [str(s.seq_id) for s in selected] == [e["id"] for e in expected["entries"]]
        assert [s.num_scheduled_tokens for s in selected] == [e["query_len"] for e in expected["entries"]]
        assert [s.num_cached_tokens for s in selected] == [e["cache_len"] for e in expected["entries"]]
        assert [list(cache.block_manager.table(s.seq_id)) for s in selected] == [e["block_table"] for e in expected["entries"]]
        tokens = []
        for seq in selected:
            cache.layer_lengths[seq.seq_id][0] += seq.num_scheduled_tokens
            token = simulator.states[seq.seq_id].token_base + seq.num_tokens
            if seq.num_completion_tokens + 1 == specs[seq.seq_id].output_tokens:
                token = -1  # Last sampled token is not written to KV after finishing.
            tokens.append(token if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens else None)
        scheduler.postprocess(plan, tokens)
        assert [b.ref_count for b in cache.block_manager.blocks] == expected["block_ref_counts_after_commit"]
    assert scheduler.is_finished()
    assert [seq.num_completion_tokens for seq in actual_seqs] == [s.output_tokens for s in specs]


@pytest.mark.parametrize("prefix_cache", [False, True])
@pytest.mark.parametrize("count,prompt,output,seq_cap,shared", [
    (4, 127, 3, 2, 0),
    (80, 128, 3, 32, 0),
    (32, 768, 16, 32, 0),
    (96, 64, 256, 32, 0),
    (16, 768, 256, 32, 0),
    (32, 768, 128, 32, 512),
])
def test_128x78_round_by_round_blocks_match_actual_scheduler(
        prefix_cache, count, prompt, output, seq_cap, shared):
    specs = [RequestSpec(str(i), prompt, output, shared_prefix_tokens=shared) for i in range(count)]
    limits = Limits(seq_cap, 2048, 128, 78, 1024, prefix_cache)
    simulator = RoundSimulator(specs, limits, "separate", record_steps=True)
    cache = MetadataCache(limits)
    scheduler = Scheduler(cache, seq_cap, 2048, 1024)
    for i, state in enumerate(simulator.states):
        scheduler.add(Sequence(i, list(state.token_ids), output))
    result = simulator.run()
    assert result["status"] == "ok"
    actual_queries, actual_preemptions = 0, []
    original_preempt = scheduler.preempt

    def observe_preempt(seq):
        actual_preemptions.append(str(seq.seq_id))
        original_preempt(seq)

    scheduler.preempt = observe_preempt
    for expected in result["steps"]:
        before_preemptions = len(actual_preemptions)
        plan = scheduler.schedule()
        selected = plan.seqs
        actual = [{"id": str(s.seq_id), "phase": plan.kind,
                   "cache_len": s.num_cached_tokens, "query_len": s.num_scheduled_tokens,
                   "history_len": s.num_tokens, "block_table": list(cache.block_manager.table(s.seq_id))}
                  for s in selected]
        keys = tuple(actual[0])
        assert actual == [{k: entry[k] for k in keys} for entry in expected["entries"]]
        assert len(cache.block_manager.free_block_id) == expected["free_blocks_after_selection"]
        assert actual_preemptions[before_preemptions:] == expected["preempted"]
        for seq in selected:
            assert cache.layer_lengths[seq.seq_id] == [seq.num_cached_tokens]
            cache.layer_lengths[seq.seq_id][0] += seq.num_scheduled_tokens
            actual_queries += seq.num_scheduled_tokens
        tokens = [simulator.states[e.seq.seq_id].token_base + e.num_tokens if e.should_sample else None
                  for e in plan.entries]
        scheduler.postprocess(plan, tokens)
        assert [b.ref_count for b in cache.block_manager.blocks] == expected["block_ref_counts_after_commit"]
    assert scheduler.is_finished() and not cache.block_manager.tables
    assert result["query_tokens"] == actual_queries
    assert result["preemptions"] == len(actual_preemptions)
