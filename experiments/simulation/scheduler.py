import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

"""Count model forwards for the guide's separate/interleaved/mixed policies.

No model, GPU, or torch. Load the repository's real BlockManager directly,
including block tables, FIFO free blocks, references, hashes, and cold reuse.
A forward takes one simulated round regardless of its token count; counts are
scheduling estimates, not wall-clock speedups.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import statistics
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from itertools import product
from pathlib import Path


POLICIES = ("separate", "interleaved", "mixed")
BLOCK_MANAGER_SOURCE = Path(__file__).resolve().parents[2] / "nanovllm/engine/block_manager.py"


@lru_cache(maxsize=1)
def load_block_manager():
    # Import the stdlib-only allocator file without executing nanovllm.__init__,
    # which would otherwise load torch, transformers, and the model runner.
    spec = importlib.util.spec_from_file_location("scheduling_sim_block_manager", BLOCK_MANAGER_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.BlockManager


def check_integer(name, value, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


@dataclass(frozen=True)
class RequestSpec:
    id: str
    prompt_tokens: int
    output_tokens: int
    arrival_round: int = 0
    shared_prefix_tokens: int = 0
    max_tokens: int | None = None  # Admission cap; output_tokens is the actual finish length.
    prompt_token_ids: tuple[int, ...] | None = None

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("request id must be a nonempty string")
        check_integer("prompt_tokens", self.prompt_tokens, 1)
        check_integer("output_tokens", self.output_tokens, 0)
        check_integer("arrival_round", self.arrival_round, 0)
        check_integer("shared_prefix_tokens", self.shared_prefix_tokens, 0)
        if self.shared_prefix_tokens > self.prompt_tokens:
            raise ValueError("shared_prefix_tokens cannot exceed prompt_tokens")
        if self.max_tokens is not None:
            check_integer("max_tokens", self.max_tokens, 0)
            if self.output_tokens > self.max_tokens:
                raise ValueError("output_tokens cannot exceed max_tokens")
        if self.prompt_token_ids is not None:
            object.__setattr__(self, "prompt_token_ids", tuple(self.prompt_token_ids))
            if len(self.prompt_token_ids) != self.prompt_tokens:
                raise ValueError("prompt_token_ids length must equal prompt_tokens")
            for token in self.prompt_token_ids:
                check_integer("prompt token ID", token, 0)
            if self.shared_prefix_tokens:
                raise ValueError("use real token IDs or synthetic shared_prefix_tokens, not both")

    @property
    def requested_max_tokens(self):
        return self.output_tokens if self.max_tokens is None else self.max_tokens


@dataclass(frozen=True)
class Limits:
    max_num_seqs: int
    token_budget: int
    block_size: int = 128
    num_blocks: int = 0  # 0 materializes enough physical blocks to avoid pressure.
    max_model_len: int = 0  # 0 disables the model position limit.
    enable_prefix_cache: bool = False

    def __post_init__(self):
        check_integer("max_num_seqs", self.max_num_seqs, 1)
        check_integer("token_budget", self.token_budget, 1)
        check_integer("block_size", self.block_size, 1)
        check_integer("num_blocks", self.num_blocks, 0)
        check_integer("max_model_len", self.max_model_len, 0)
        if type(self.enable_prefix_cache) is not bool:
            raise ValueError("enable_prefix_cache must be a boolean")


@dataclass(eq=False)
class RequestState:
    spec: RequestSpec
    cached_tokens: int = 0
    generated_tokens: int = 0
    token_ids: list[int] = field(default_factory=list)
    token_base: int = 0
    num_scheduled_tokens: int = 0
    history_high_water: int = 0  # Positions ever computed or reused, even after preemption.
    first_token_round: int | None = None
    finish_round: int | None = None
    last_token_round: int | None = None

    @property
    def num_tokens(self):
        # Generated tokens stay in the logical history after preemption.
        return len(self.token_ids)

    @property
    def seq_id(self):
        return self.spec.id

    @property
    def num_cached_tokens(self):
        return self.cached_tokens


@dataclass(frozen=True)
class Entry:
    seq: RequestState
    phase: str
    cache_len: int
    query_len: int
    history_len: int
    prefix_hit_tokens: int = 0

    @property
    def should_sample(self):
        # Only the last prefill chunk and ordinary decode produce a token.
        return self.cache_len + self.query_len == self.history_len


class RoundSimulator:
    def __init__(self, requests, limits, policy, record_steps=False, max_rounds=1_000_000):
        if policy not in POLICIES:
            raise ValueError(f"unknown policy: {policy}")
        check_integer("max_rounds", max_rounds, 1)
        requests = list(requests)
        if not requests or len({s.id for s in requests}) != len(requests):
            raise ValueError("requests must be nonempty with unique IDs")
        self.limits = limits
        self.policy = policy
        self.states = [RequestState(s) for s in requests]
        span = max(s.prompt_tokens + s.output_tokens for s in requests) + 1
        shared_span = max(s.shared_prefix_tokens for s in requests)
        real_token_max = max((max(s.prompt_token_ids) for s in requests
                              if s.prompt_token_ids is not None), default=-1)
        synthetic_start = max(shared_span, real_token_max + 1) + 1
        for index, seq in enumerate(self.states):
            seq.token_base = synthetic_start + index * span
            shared = seq.spec.shared_prefix_tokens
            # Explicitly shared prompt tokens are identical across requests.
            # All other prompts/completions are disjoint synthetic token IDs.
            if seq.spec.prompt_token_ids is not None:
                seq.token_ids = list(seq.spec.prompt_token_ids)
            else:
                seq.token_ids = list(range(shared)) + [
                    seq.token_base + position for position in range(shared, seq.spec.prompt_tokens)
                ]
        capacity = limits.num_blocks or max(1, sum(
            (s.prompt_tokens + s.requested_max_tokens - 1 + limits.block_size - 1) // limits.block_size
            for s in requests if s.requested_max_tokens
        ))
        self.block_manager = load_block_manager()(capacity, limits.block_size,
                                                 limits.enable_prefix_cache)
        # Stable sorting preserves FIFO input order for simultaneous arrivals.
        self.incoming = deque(sorted(self.states, key=lambda s: s.spec.arrival_round))
        self.waiting = deque()
        self.running = deque()
        self.next_phase = "prefill"
        self.round = 0
        self.forward_rounds = 0
        self.idle_rounds = 0
        self.peak_kv_blocks = 0
        self.peak_block_references = 0
        self.block_allocations = 0
        self.prefix_hit_tokens = 0
        self.prefix_saved_query_tokens = 0
        self.recomputed_query_tokens = 0
        self.prefill_capacity_stalls = 0
        self.full_pool_rounds = 0
        self.preemptions = 0
        self.discarded_cached_tokens = 0
        self.reserved_prefill_rescues = 0
        self.query_tokens = 0
        self.prefill_query_tokens = 0
        self.decode_query_tokens = 0
        self.kind_counts = Counter()
        self.inter_token_gaps = []
        self.step_preemptions = []
        self.steps = [] if record_steps else None
        self.max_rounds = max_rounds

    def needed_blocks(self, seq):
        return (seq.num_tokens + self.limits.block_size - 1) // self.limits.block_size

    @property
    def allocated_blocks(self):
        return self.block_manager.num_blocks - len(self.block_manager.free_block_id)

    def update_block_peaks(self):
        self.peak_kv_blocks = max(self.peak_kv_blocks, self.allocated_blocks)
        references = sum(block.ref_count for block in self.block_manager.blocks)
        self.peak_block_references = max(self.peak_block_references, references)

    def can_reserve(self, seq):
        return self.block_manager.can_allocate(seq)

    def reserve(self, seq):
        allocated = self.block_manager.may_allocate(seq)
        self.block_allocations += allocated
        # Ordinary decode within a page changes no block references or peaks.
        # Avoid scanning the full physical pool once per token in large sweeps.
        if allocated:
            self.update_block_peaks()

    def release(self, seq):
        self.block_manager.release(seq.seq_id)

    def free_blocks(self):
        if not self.limits.num_blocks:
            return None
        return len(self.block_manager.free_block_id)

    def preempt(self, seq):
        self.preemptions += 1
        self.discarded_cached_tokens += seq.cached_tokens
        self.step_preemptions.append(seq.spec.id)
        self.release(seq)
        seq.cached_tokens = 0
        seq.num_scheduled_tokens = 0
        self.block_manager.register([seq])
        self.waiting.appendleft(seq)

    def select_prefill(self, token_budget, seq_budget):
        selected, used = [], 0
        while self.waiting and len(selected) < seq_budget:
            remaining = token_budget - used
            if remaining <= 0:
                break
            seq = self.waiting[0]
            fresh = not self.block_manager.table(seq.seq_id)
            if fresh:
                possible, hits = self.block_manager.can_allocate_prefix(seq)
                if not possible:
                    self.prefill_capacity_stalls += 1
                    break  # FIFO: do not skip a blocked head during normal selection.
                c = len(hits) * self.limits.block_size
            else:
                c = seq.cached_tokens
            missing = seq.num_tokens - c
            if missing <= 0:
                raise RuntimeError("waiting request has no missing history")
            if selected and missing > remaining:
                break  # Only the first request in this prefill sub-batch can chunk.
            if fresh:
                attached = self.block_manager.allocate_prefix(seq, hits)
                if attached != c:
                    raise RuntimeError("prefix attachment length mismatch")
                seq.cached_tokens = c
                self.prefix_hit_tokens += c
                self.prefix_saved_query_tokens += max(0, c - seq.history_high_water)
                seq.history_high_water = max(seq.history_high_water, c)
                self.block_allocations += self.needed_blocks(seq) - len(hits)
                self.update_block_peaks()
            q = min(missing, remaining)
            seq.num_scheduled_tokens = q
            selected.append(Entry(seq, "prefill", c, q, seq.num_tokens, c if fresh else 0))
            used += q
            if seq.cached_tokens + q == seq.num_tokens:
                self.waiting.popleft()
                self.running.append(seq)
            else:
                break
        return selected

    def select_decode(self, token_budget, seq_budget):
        selected = []
        while self.running and len(selected) < min(token_budget, seq_budget):
            seq = self.running.popleft()
            while not self.can_reserve(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                self.reserve(seq)
                seq.num_scheduled_tokens = 1
                selected.append(Entry(seq, "decode", seq.cached_tokens, 1, seq.num_tokens))
        # Preserve production queue order. This is not per-request round robin.
        self.running.extendleft(reversed([entry.seq for entry in selected]))
        return selected

    def schedule(self):
        t, s = self.limits.token_budget, self.limits.max_num_seqs
        if self.policy == "mixed":
            selected = self.select_decode(t, s)
            d = len(selected)
            selected += self.select_prefill(t - d, s - d)
        else:
            first = "prefill" if self.policy == "separate" else self.next_phase
            choose_first = self.select_prefill if first == "prefill" else self.select_decode
            choose_second = self.select_decode if first == "prefill" else self.select_prefill
            selected = choose_first(t, s)
            if not selected:
                selected = choose_second(t, s)

        # Shared liveness handling from guide section 4.5. Decode may have
        # released blocks by preempting a request; retry prefill once.
        if not selected:
            selected = self.select_prefill(t, s)
        if not selected:
            for seq in self.waiting:
                if self.block_manager.table(seq.seq_id) and seq.cached_tokens < seq.num_tokens:
                    self.waiting.remove(seq)
                    self.waiting.appendleft(seq)
                    self.reserved_prefill_rescues += 1
                    selected = self.select_prefill(t, s)
                    break
        return selected

    def commit(self, entries):
        end_round = self.round + 1
        finished = []
        # Simulate a successful all-layer KV write before publishing complete
        # blocks. Writes must touch private, unpublished pages, as in the runtime.
        for entry in entries:
            table = self.block_manager.table(entry.seq.seq_id)
            for i in range(entry.cache_len // self.limits.block_size,
                           (entry.cache_len + entry.query_len - 1) // self.limits.block_size + 1):
                block = self.block_manager.blocks[table[i]]
                if block.ref_count != 1 or block.hash != -1:
                    raise RuntimeError("KV writes must target private unpublished blocks")
            self.block_manager.hash_blocks(entry.seq)
        for entry in entries:
            seq = entry.seq
            self.recomputed_query_tokens += max(
                0, min(entry.cache_len + entry.query_len, seq.history_high_water) - entry.cache_len
            )
            seq.history_high_water = max(seq.history_high_water, entry.cache_len + entry.query_len)
            seq.cached_tokens += entry.query_len
            seq.num_scheduled_tokens = 0
            if entry.should_sample:
                seq.token_ids.append(seq.token_base + seq.num_tokens)
                seq.generated_tokens += 1
                if seq.first_token_round is None:
                    seq.first_token_round = end_round
                if seq.last_token_round is not None:
                    self.inter_token_gaps.append(end_round - seq.last_token_round)
                seq.last_token_round = end_round
                if seq.generated_tokens == seq.spec.output_tokens:
                    seq.finish_round = end_round
                    self.running.remove(seq)
                    self.release(seq)
                    finished.append(seq.spec.id)
        return finished

    def run(self):
        for seq in self.states:
            maximum_context = seq.spec.prompt_tokens + seq.spec.requested_max_tokens - 1
            if (seq.spec.requested_max_tokens and self.limits.num_blocks
                    and maximum_context > self.limits.num_blocks * self.limits.block_size):
                return self.result("infeasible", (
                    f"request {seq.spec.id}: maximum context {maximum_context} exceeds "
                    f"KV capacity {self.limits.num_blocks * self.limits.block_size}"
                ))
            if (seq.spec.requested_max_tokens and self.limits.max_model_len
                    and maximum_context > self.limits.max_model_len):
                return self.result("infeasible", (
                    f"request {seq.spec.id}: maximum context {maximum_context} exceeds "
                    f"max_model_len {self.limits.max_model_len}"
                ))

        while self.incoming or self.waiting or self.running:
            while self.incoming and self.incoming[0].spec.arrival_round <= self.round:
                seq = self.incoming.popleft()
                if seq.spec.output_tokens == 0:
                    seq.finish_round = self.round
                else:
                    self.block_manager.register([seq])
                    self.waiting.append(seq)
            if not self.waiting and not self.running:
                if not self.incoming:
                    break
                next_arrival = self.incoming[0].spec.arrival_round
                self.idle_rounds += next_arrival - self.round
                self.round = next_arrival
                continue
            if self.forward_rounds >= self.max_rounds:
                return self.result("round_limit", "increase --max-rounds to finish this workload")

            self.step_preemptions = []
            free_before = self.free_blocks()
            entries = self.schedule()
            if not entries:
                return self.result("no_progress", "unfinished requests could not form a batch")
            phases = {entry.phase for entry in entries}
            kind = next(iter(phases)) if len(phases) == 1 else "mixed"
            q = sum(entry.query_len for entry in entries)
            if (len(entries) > self.limits.max_num_seqs or not 0 < q <= self.limits.token_budget
                    or len({entry.seq.spec.id for entry in entries}) != len(entries)):
                raise RuntimeError("simulator formed an invalid batch")
            for entry in entries:
                if (entry.query_len <= 0 or entry.cache_len + entry.query_len > entry.history_len
                        or (entry.phase == "decode" and entry.cache_len != entry.history_len - 1)):
                    raise RuntimeError("simulator formed an invalid token interval")
            self.kind_counts[kind] += 1
            self.query_tokens += q
            for entry in entries:
                if entry.phase == "prefill":
                    self.prefill_query_tokens += entry.query_len
                else:
                    self.decode_query_tokens += entry.query_len
            free_after_selection = self.free_blocks()
            if free_after_selection == 0:
                self.full_pool_rounds += 1
            selected_tables = ({entry.seq.seq_id: list(self.block_manager.table(entry.seq.seq_id))
                                for entry in entries} if self.steps is not None else None)
            finished = self.commit(entries)
            if self.policy == "interleaved":
                self.next_phase = "decode" if kind == "prefill" else "prefill"
            if self.steps is not None:
                self.steps.append({
                    "round": self.round, "kind": kind, "query_tokens": q,
                    "entries": [{"id": e.seq.spec.id, "phase": e.phase,
                                 "cache_len": e.cache_len, "query_len": e.query_len,
                                 "history_len": e.history_len, "sampled": e.should_sample,
                                 "prefix_hit_tokens": e.prefix_hit_tokens,
                                 "block_table": selected_tables[e.seq.seq_id]}
                                for e in entries],
                    "preempted": list(self.step_preemptions), "finished": finished,
                    "free_blocks_before": free_before,
                    "free_blocks_after_selection": free_after_selection,
                    "free_blocks_after_commit": self.free_blocks(),
                    "waiting": [seq.spec.id for seq in self.waiting],
                    "running": [seq.spec.id for seq in self.running],
                    "block_tables_after_commit": {sid: list(table) for sid, table
                                                  in self.block_manager.tables.items()},
                    "block_ref_counts_after_commit": [b.ref_count for b in self.block_manager.blocks],
                })
            self.forward_rounds += 1
            self.round += 1
        return self.result("ok")

    def result(self, status, error=None):
        expected_query_tokens = sum(
            seq.spec.prompt_tokens + seq.spec.output_tokens - 1
            for seq in self.states if seq.spec.output_tokens
        )
        ttft, completion = [], []
        requests = []
        for seq in self.states:
            first_delay = (None if seq.first_token_round is None else
                           seq.first_token_round - seq.spec.arrival_round)
            completion_delay = (None if seq.finish_round is None else
                                seq.finish_round - seq.spec.arrival_round)
            if first_delay is not None:
                ttft.append(first_delay)
            if completion_delay is not None:
                completion.append(completion_delay)
            requests.append({"id": seq.spec.id, "generated_tokens": seq.generated_tokens,
                             "first_token_round": seq.first_token_round,
                             "finish_round": seq.finish_round,
                             "ttft_rounds": first_delay,
                             "completion_rounds": completion_delay})
        if status == "ok":
            if (self.allocated_blocks != 0 or self.block_manager.tables
                    or self.query_tokens != expected_query_tokens - self.prefix_saved_query_tokens
                    + self.recomputed_query_tokens
                    or self.query_tokens != expected_query_tokens + self.discarded_cached_tokens
                    - self.prefix_hit_tokens):
                raise RuntimeError("simulator resource/work accounting mismatch")
        result = {
            "policy": self.policy, "status": status, "error": error,
            "forward_rounds": self.forward_rounds, "elapsed_rounds": self.round,
            "idle_rounds": self.idle_rounds,
            "prefill_only_rounds": self.kind_counts["prefill"],
            "decode_only_rounds": self.kind_counts["decode"],
            "mixed_rounds": self.kind_counts["mixed"],
            "query_tokens": self.query_tokens,
            "prefill_query_tokens": self.prefill_query_tokens,
            "decode_query_tokens": self.decode_query_tokens,
            "base_query_tokens": expected_query_tokens,
            "extra_recompute_tokens": (self.recomputed_query_tokens if status == "ok" else None),
            "prefix_hit_tokens": self.prefix_hit_tokens,
            "prefix_saved_query_tokens": self.prefix_saved_query_tokens,
            "discarded_cached_tokens": self.discarded_cached_tokens,
            "preemptions": self.preemptions,
            "reserved_prefill_rescues": self.reserved_prefill_rescues,
            "peak_kv_blocks": self.peak_kv_blocks,
            "peak_block_references": self.peak_block_references,
            "block_allocations": self.block_allocations,
            "prefill_capacity_stalls": self.prefill_capacity_stalls,
            "full_pool_rounds": self.full_pool_rounds,
            "cold_cached_blocks_at_finish": sum(b.hash != -1 and b.ref_count == 0
                                                for b in self.block_manager.blocks),
            "mean_ttft_rounds": statistics.fmean(ttft) if ttft else None,
            "mean_completion_rounds": statistics.fmean(completion) if completion else None,
            "mean_inter_token_gap_rounds": (statistics.fmean(self.inter_token_gaps)
                                           if self.inter_token_gaps else None),
            "requests": requests,
        }
        if self.steps is not None:
            result["steps"] = self.steps
        return result


def simulate(requests, limits, policy, **kwargs):
    return RoundSimulator(requests, limits, policy, **kwargs).run()


def compare(requests, limits, **kwargs):
    requests = list(requests)
    results = [simulate(requests, limits, policy, **kwargs) for policy in POLICIES]
    baseline = results[0]
    for result in results:
        valid = baseline["status"] == result["status"] == "ok"
        rounds = result["forward_rounds"]
        base_rounds = baseline["forward_rounds"]
        result["round_reduction_pct"] = (
            100 * (base_rounds - rounds) / base_rounds if valid and base_rounds else None
        )
        result["round_ratio"] = base_rounds / rounds if valid and rounds else None
    return results


def int_list(value, minimum=1):
    try:
        values = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use comma-separated integers") from exc
    if not values or any(v < minimum for v in values):
        raise argparse.ArgumentTypeError(f"all values must be >= {minimum}")
    return values


def nonnegative_list(value):
    return int_list(value, 0)


def load_workload(path):
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    rows = value.get("requests") if isinstance(value, dict) else value
    if not isinstance(rows, list) or not rows:
        raise ValueError("trace-file must contain a nonempty requests list")
    return [RequestSpec(**row) for row in rows]


def simulation_assumptions(enable_prefix_cache):
    return {
        "clock": "integer arrival rounds; one forward consumes one round, independent of shape",
        "allocator": "repository production BlockManager: physical IDs/tables/free queue/refcounts/hashes",
        "block_manager_sha256": hashlib.sha256(BLOCK_MANAGER_SOURCE.read_bytes()).hexdigest(),
        "prefix_cache": enable_prefix_cache,
        "tokens": "real prompt_token_ids when supplied; otherwise unique synthetic prompts except explicit shared prefixes; synthetic completions",
        "output": "fixed actual output_tokens; max_tokens separately controls admission; final prefill produces first output token",
        "prefill": "reserve full history upfront; only first prefill in sub-batch can chunk",
        "decode": "one query per request; preserve queue order; preempt unselected tail",
        "finite_kv": "all three use guide section 4.5 shared liveness recovery",
        "warning": "round ratio is not measured runtime speedup; arrivals use rounds, not seconds",
    }


def write_report(report, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_rows = []
    for comparison in report["comparisons"]:
        workload_id = comparison["workload_id"]
        for result in comparison["results"]:
            csv_rows.append({
                "workload_id": workload_id,
                "request_count": len(report["workloads"][workload_id]),
                **comparison.get("factors", {}), **comparison["limits"],
                **{k: v for k, v in result.items() if k not in ("requests", "steps")},
            })
    csv_path = out.with_suffix(".csv")
    fieldnames = list(dict.fromkeys(key for row in csv_rows for key in row))
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    return csv_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int_list, default=[32, 64],
                        help="request counts to sweep, default: 32,64")
    parser.add_argument("--max-num-seqs", type=int_list, default=[8, 32],
                        help="per-forward sequence caps, default: 8,32")
    parser.add_argument("--token-budgets", type=int_list, default=[2048],
                        help="per-forward query token budgets, default: current 2048")
    parser.add_argument("--prompt-lengths", type=int_list, default=[64, 512, 768],
                        help="cycle these prompt lengths inside each workload, default: 64,512,768")
    parser.add_argument("--output-lengths", type=nonnegative_list, default=[32, 64],
                        help="cycle these output lengths; first token is produced by final prefill")
    parser.add_argument("--arrival-intervals", type=nonnegative_list, default=[0],
                        help="sweep spacing in simulated rounds; 0 means all arrive together")
    parser.add_argument("--num-blocks", type=nonnegative_list, default=[78],
                        help="sweep physical KV pool sizes; 0 means unlimited, default: 78")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=1024,
                        help="maximum prompt + output - 1; 0 disables position limit")
    parser.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction, default=True,
                        help="use real full-block prefix hits/cold reuse, default: enabled")
    parser.add_argument("--shared-prefix-tokens", type=int, default=0,
                        help="explicit identical prefix in all generated prompts; default: unique prompts")
    parser.add_argument("--trace-file", type=Path,
                        help="explicit length/arrival trace; replaces generated request workloads")
    parser.add_argument("--record-steps", action="store_true", help="include each round in JSON")
    parser.add_argument("--max-rounds", type=int, default=1_000_000)
    parser.add_argument("--out", type=Path, default=Path("artifacts/scheduling-simulation/results.json"))
    args = parser.parse_args(argv)
    try:
        check_integer("block-size", args.block_size, 1)
        check_integer("max-rounds", args.max_rounds, 1)
        check_integer("max-model-len", args.max_model_len, 0)
        check_integer("shared-prefix-tokens", args.shared_prefix_tokens, 0)
        if args.out.suffix.lower() != ".json":
            raise ValueError("--out must end in .json; a matching .csv is also written")
        if args.trace_file:
            workloads = [load_workload(args.trace_file)]
        else:
            workloads = [[RequestSpec(
                id=str(i), prompt_tokens=args.prompt_lengths[i % len(args.prompt_lengths)],
                output_tokens=args.output_lengths[i % len(args.output_lengths)],
                arrival_round=i * interval,
                shared_prefix_tokens=args.shared_prefix_tokens,
            ) for i in range(count)] for count, interval in product(args.requests, args.arrival_intervals)]

        report = {
            "schema": 2,
            "assumptions": simulation_assumptions(args.prefix_cache),
            "workloads": {}, "comparisons": [],
        }
        print("N     S      T     KV    gap   separate interleaved   mixed   save(I) save(M)")
        for requests in workloads:
            rows = [asdict(spec) for spec in requests]
            workload_id = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16]
            report["workloads"][workload_id] = rows
            gaps = {b.arrival_round - a.arrival_round for a, b in zip(
                sorted(requests, key=lambda r: r.arrival_round),
                sorted(requests, key=lambda r: r.arrival_round)[1:],
            )}
            interval = next(iter(gaps)) if len(gaps) == 1 else (0 if not gaps else "var")
            for s, t, blocks in product(args.max_num_seqs, args.token_budgets, args.num_blocks):
                limits = Limits(s, t, args.block_size, blocks, args.max_model_len, args.prefix_cache)
                results = compare(requests, limits, record_steps=args.record_steps, max_rounds=args.max_rounds)
                report["comparisons"].append({
                    "workload_id": workload_id, "limits": asdict(limits), "results": results,
                    "factors": {"arrival_interval_rounds": interval},
                })
                counts = [str(r["forward_rounds"]) if r["status"] == "ok" else r["status"] for r in results]
                savings = [f"{r['round_reduction_pct']:+.1f}%" if r["round_reduction_pct"] is not None
                           else "n/a" for r in results[1:]]
                print(f"{len(requests):<5} {s:<5} {t:<5} {blocks or 'inf':<5} {interval!s:<5} "
                      f"{counts[0]:>8} {counts[1]:>11} {counts[2]:>7} {savings[0]:>8} {savings[1]:>7}")
        csv_path = write_report(report, args.out)
        print(f"\nJSON: {args.out}\nCSV:  {csv_path}")
        print("Positive save = fewer forwards; negative save = more forwards. Not a runtime speedup.")
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
