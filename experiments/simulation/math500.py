import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

"""Tokenize local Math500, then compare three schedulers without model forwards.

The prompt path matches math500.py: render Qwen's non-thinking chat template,
then tokenizer.encode, as LLMEngine.add_request does. Completions use fixed
lengths and unique synthetic IDs; max_tokens is an independent admission cap.
"""

import argparse
import hashlib
import json
import math
import statistics
from dataclasses import asdict
from itertools import product
from pathlib import Path

from experiments.simulation.sweep import runtime_defaults, summarize
from experiments.simulation.scheduler import (
    Limits, RequestSpec, compare, int_list, simulation_assumptions, write_report,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts/scheduling-simulation/math500-128x78/results.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def length_stats(rows):
    values = sorted(len(row["prompt_token_ids"]) for row in rows)
    return {"count": len(values), "min": min(values), "median": statistics.median(values),
            "mean": statistics.fmean(values), "p95": values[math.ceil(len(values) * .95) - 1],
            "max": max(values)} if values else {"count": 0}


def encode_problem(tokenizer, problem):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": problem}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    return tokenizer.encode(rendered)


def prepare_prompts(dataset, model, cache, refresh=False):
    fingerprints = {
        "dataset": sha256(dataset),
        "tokenizer_files": {name: sha256(model / name) for name in (
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"
        ) if (model / name).is_file()},
    }
    if not fingerprints["tokenizer_files"]:
        raise ValueError(f"no tokenizer files in {model}")
    if cache.is_file() and not refresh:
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if cached.get("fingerprints") == fingerprints and cached.get("schema") == 1:
            return cached
    # Optional dependency only for preparation; cached simulations are stdlib-only.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]
    prompts = [{"id": row.get("unique_id", str(index)), "dataset_index": index,
                "prompt_token_ids": encode_problem(tokenizer, row["problem"])}
               for index, row in enumerate(rows)]
    if not prompts or len({p["id"] for p in prompts}) != len(prompts):
        raise ValueError("dataset must have nonempty prompts and unique IDs")
    cached = {
        "schema": 1, "dataset": str(dataset.resolve()), "model": str(model.resolve()),
        "fingerprints": fingerprints,
        "template": "apply_chat_template(tokenize=False, add_generation_prompt=True, enable_thinking=False), then encode",
        "length_stats": length_stats(prompts), "prompts": prompts,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(cached, ensure_ascii=False, indent=2), encoding="utf-8")
    return cached


def invalid_rows(rows, cap, args):
    return [row for row in rows if (
        args.max_model_len and len(row["prompt_token_ids"]) + cap - 1 > args.max_model_len
    ) or (args.num_blocks and len(row["prompt_token_ids"]) + cap - 1 > args.num_blocks * args.block_size)]


def generation_pairs(output_lengths, caps):
    # The diagonal models ignore_eos / hitting the cap. Off-diagonal pairs model
    # finishing at a fixed length before the cap; never silently truncate G.
    return sorted({(cap, cap) for cap in caps} |
                  {(length, cap) for length, cap in product(output_lengths, caps) if length <= cap})


def requests_for(rows, output_tokens, cap):
    return [RequestSpec(row["id"], len(row["prompt_token_ids"]), output_tokens,
                        max_tokens=cap, prompt_token_ids=row["prompt_token_ids"]) for row in rows]


def render_row(case):
    f = case["factors"]
    r = case["results"]
    counts = [str(v["forward_rounds"]) if v["status"] == "ok" else v["status"] for v in r]
    savings = f"{r[2]['round_reduction_pct']:+.2f}%" if r[2]["round_reduction_pct"] is not None else "—"
    preemptions = "/".join(str(v["preemptions"]) for v in r)
    return (f"| {f['requested_count']} | {f['request_count']} | {f['cohort']} | {f['output_tokens']} "
            f"| {f['max_tokens']} | {case['limits']['max_num_seqs']} | {case['limits']['token_budget']} | "
            f"{' | '.join(counts)} | {savings} | {preemptions} |")


def cap_invariance(cases):
    groups = {}
    for case in cases:
        if not all(r["status"] == "ok" for r in case["results"]):
            continue
        f = case["factors"]
        key = (f["requested_count"], f["cohort"], f["output_tokens"],
               case["limits"]["max_num_seqs"], case["limits"]["token_budget"])
        groups.setdefault(key, []).append(case)
    comparable = [group for group in groups.values() if len(group) > 1]
    identical = sum(len({tuple(r["forward_rounds"] for r in c["results"]) for c in group}) == 1
                    for group in comparable)
    return {"groups_with_multiple_valid_caps": len(comparable), "identical_rounds": identical}


def merge_reports(report, paths):
    """Combine completed sweeps without counting overlapping configurations twice."""
    for path in paths:
        previous = json.loads(path.read_text(encoding="utf-8"))
        if (previous["dataset"]["fingerprints"] != report["dataset"]["fingerprints"]
                or previous["assumptions"] != report["assumptions"]
                or previous["cohorts"] != json.loads(json.dumps(report["cohorts"]))):
            raise ValueError(f"cannot merge incompatible dataset/allocator/cohorts: {path}")
        for name in ("block_size", "num_blocks", "max_model_len", "prefix_cache"):
            if previous["experiment"][name] != report["experiment"][name]:
                raise ValueError(f"cannot merge different {name}: {path}")
        report["workloads"].update(previous["workloads"])
        comparisons = {}
        for case in previous["comparisons"] + report["comparisons"]:
            key = (case["workload_id"], json.dumps(case["limits"], sort_keys=True))
            if key in comparisons and comparisons[key]["results"] != case["results"]:
                raise ValueError(f"overlapping simulations disagree: {path}")
            comparisons[key] = case
        report["comparisons"] = list(comparisons.values())
        exp = previous["experiment"]
        previous_budgets = exp.get("token_budgets", [exp.get("token_budget")])
        report["experiment"]["token_budgets"] = sorted(set(previous_budgets + report["experiment"]["token_budgets"]))
        for name in ("caps", "output_lengths", "request_counts", "seq_caps"):
            report["experiment"][name] = sorted(set(exp[name] + report["experiment"][name]))


def write_markdown(report, path, out, cache_path):
    exp = dict(report["experiment"])
    exp.setdefault("token_budgets", [exp.get("token_budget", 2048)])
    stats = report["dataset"]["length_stats"]
    reference_budget = 2048 if 2048 in exp["token_budgets"] else exp["token_budgets"][0]
    representative_seqs = {s for s in (32, 64) if s in exp["seq_caps"]} or {max(exp["seq_caps"])}
    budget_seq_cap = 64 if 64 in exp["seq_caps"] else max(exp["seq_caps"])
    fixed_budgets = sorted({c["limits"]["token_budget"] for c in report["comparisons"]
                            if c["factors"]["generation_mode"] == "fixed_finish"})
    completed = sum(all(r["status"] == "ok" for r in c["results"]) for c in report["comparisons"])
    lines = ["# Math500 块结构调度模拟实验", "",
             f"使用本地 Math500 的 {stats['count']} 道题，按 `math500.py` 的 Qwen3 聊天模板构造 prompt。"
             "不添加解题前缀，不使用标准答案或 solution 作为输入。",
             "", f"全量 prompt token 长度：最短 {stats['min']}，中位数 {stats['median']}，"
             f"平均 {stats['mean']:.2f}，P95 {stats['p95']}，最长 {stats['max']}。",
             "", "## 配置和生成长度", "",
             f"固定 block_size={exp['block_size']}、block_num={exp['num_blocks']}、"
             f"max_model_len={exp['max_model_len']}、"
             f"prefix cache={'开启' if exp['prefix_cache'] else '关闭'}。",
             f"每轮 query token 预算 T（max_num_batched_tokens）扫描 {exp['token_budgets']}；"
             "max_tokens 指单请求生成上限，与 T 分开设置。",
             f"请求数取数据集原始顺序的前 {exp['request_counts']} 条；"
             f"max_num_seqs={exp['seq_caps']}，max_tokens={exp['caps']}；"
             f"独立的实际生成长度 G={exp['output_lengths']}。所有请求在第 0 轮到达。",
             f"生成到上限的组合扫描全部 T；固定 G、改变上限的组合在 T={fixed_budgets} 下运行。"
             f"合并后共 {len(report['comparisons'])} 组配置、{3 * len(report['comparisons'])} 个策略结果。",
             f"其中 {completed} 组三策略全部完成，{len(report['comparisons']) - completed} 组未全部完成；"
             "具体原因保留在每个结果的 status/error 中。",
             "", "G=max_tokens 的组合模拟生成到上限；G<max_tokens 的组合模拟在指定长度结束。"
             "G 包含最后一个采样 token，最后一块 prefill 产生第一个输出 token。"
             "这两类长度均为人为设定，没有运行模型采样，也没有测量真实 EOS 分布。",
             "", "prompt 使用真实 token ID，完整 128-token 前缀块的共享由生产 BlockManager 判断。"
             "输出采用每请求互不相同的合成 token ID，因此不模拟真实输出之间的前缀共享。"
             "每种策略重新初始化相同的 78 块；保留 FIFO 空闲块队列、块表、引用计数、冷缓存、跨块扩容和抢占重算。",
             "", "## 长度限制与可比样本", "",
             "入队校验使用 `P + max_tokens - 1`，即使实际 G 更短，也必须通过"
             " `max_model_len` 和单请求 KV 容量校验。无效原始批次记作 infeasible，不把 0 轮当作收益。",
             "", "original 为原始前 N 条。若其中任何题不满足最大被测 max_tokens，"
             "额外提供 common 子集：保留同时满足所有被测上限的题，顺序不变，"
             "每个上限使用同一批题。original 与 common 的请求数和被排除 ID 都在 JSON 中记录。",
             "", "| 原始 N | prompt 最短/中位数/P95/最长 | 各 max_tokens 不可入队数 | common 保留数 |",
             "|---:|---|---|---:|"]
    for cohort in report["cohorts"]:
        s = cohort["length_stats"]
        invalid = ", ".join(f"{cap}: {len(ids)}" for cap, ids in cohort["invalid_by_cap"].items())
        lines.append(f"| {cohort['requested_count']} | {s['min']}/{s['median']}/{s['p95']}/{s['max']} "
                     f"| {invalid} | {cohort['common_count']} |")
    headers = ["| 原始 N | 实际 N | 样本 | G | max_tokens | max_seq | T | separate | interleaved | mixed | mixed轮次减少 | 抢占 S/I/M |",
               "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    common_counts = {c["factors"]["requested_count"] for c in report["comparisons"] if c["factors"]["cohort"] == "common"}
    lines += ["", f"## {reference_budget} token 预算的代表结果", "",
              "有 common 子集的请求数使用同一子集，便于比较不同生成长度；其余使用 original。", ""] + headers
    lines += [render_row(c) for c in report["comparisons"]
              if c["limits"]["token_budget"] == reference_budget and c["limits"]["max_num_seqs"] in representative_seqs
              and c["factors"]["output_tokens"] == c["factors"]["max_tokens"]
              and c["factors"]["cohort"] == ("common" if c["factors"]["requested_count"] in common_counts else "original")]
    lines += ["", "## 轮次结果", "",
              "下表列出生成到上限的所有组合；固定 G、改变上限的完整结果在 CSV/JSON。"
              "S/I/M 分别表示 separate（当前 prefill 优先）、interleaved（交替纯批）、mixed（decode 优先混批）。"
              "负的减少比例表示 mixed 轮次更多。",
              "", "| 原始 N | 实际 N | 样本 | G | max_tokens | max_seq | T | separate | interleaved | mixed | mixed轮次减少 | 抢占 S/I/M |",
              "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    lines += [render_row(c) for c in report["comparisons"]
              if c["factors"]["output_tokens"] == c["factors"]["max_tokens"]]
    lines += ["", "## 固定生成长度，只改变 max_tokens", ""]
    invariant = report["cap_invariance"]
    lines += [f"有 {invariant['groups_with_multiple_valid_caps']} 组相同请求、相同 G 和相同 max_seq，"
              f"通过多个上限的入队校验；其中 {invariant['identical_rounds']} 组三策略轮次完全相同。",
              "", "当前策略按已有历史长度分配块，decode 跨块才增加块，"
              "不会按 max_tokens 一次性预留未来全部 KV。"
              "因此固定 G 时，增大 max_tokens 主要改变能否入队；"
              "通过校验后不会单独改变每轮分配和调度。生成到上限的实验中，轮次变化则包含了 G 的变化。",
              "", "| 原始 N | 实际 N | 样本 | G | max_tokens | max_seq | T | separate | interleaved | mixed | mixed轮次减少 | 抢占 S/I/M |",
              "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    lines += [render_row(c) for c in report["comparisons"] if c["factors"]["output_tokens"] == min(exp["output_lengths"])
              and c["limits"]["max_num_seqs"] == min(representative_seqs)]
    if len(exp["token_budgets"]) > 1:
        lines += ["", "## 每轮 token 预算的影响", "",
                  f"固定生成长度等于生成上限，比较 T={exp['token_budgets']}；下表取 max_seq={budget_seq_cap}。"
                  "更大的 T 会改变 prefill 接纳数量，也可能改变 KV 拥挤和抢占，因此不能假定轮次随 T 单调减少。", ""] + headers
        lines += [render_row(c) for c in sorted(report["comparisons"], key=lambda c: (
                    c["factors"]["requested_count"], c["factors"]["output_tokens"], c["limits"]["token_budget"]))
                  if c["limits"]["max_num_seqs"] == budget_seq_cap
                  and c["factors"]["output_tokens"] == c["factors"]["max_tokens"]
                  and c["factors"]["cohort"] == ("common" if c["factors"]["requested_count"] in common_counts else "original")]
    lines += ["", "## 收益分布与解释", ""]
    for label, summary in report["summaries"].items():
        mixed = summary["mixed"]
        lines.append(f"{label}：{summary['comparisons']} 组配置，{summary['complete_comparisons']} 组可完成；"
                     f"mixed 更少/相同/更多轮次={mixed['fewer_rounds']}/{mixed['same_rounds']}/{mixed['more_rounds']}。")
        if mixed["median_reduction_pct"] is not None:
            lines += ["", f"mixed 减少比例中位数 {mixed['median_reduction_pct']:.2f}%，"
                      f"范围 {mixed['min_reduction_pct']:.2f}% 至 {mixed['max_reduction_pct']:.2f}%。", ""]
    valid = [c for c in report["comparisons"] if all(r["status"] == "ok" for r in c["results"])]
    if valid:
        for label, case in (("mixed 减少最多", max(valid, key=lambda c: c["results"][2]["round_reduction_pct"])),
                            ("mixed 减少最少", min(valid, key=lambda c: c["results"][2]["round_reduction_pct"]))):
            f, r = case["factors"], case["results"]
            lines += [f"{label}：{f['cohort']} N={f['request_count']}，G={f['output_tokens']}，"
                      f"max_tokens={f['max_tokens']}，max_seq={case['limits']['max_num_seqs']}，"
                      f"T={case['limits']['token_budget']}；"
                      f"轮次 {'/'.join(str(v['forward_rounds']) for v in r)}，"
                      f"减少 {r[2]['round_reduction_pct']:.2f}%。", ""]
        latency_example = next((c for c in valid if c["factors"]["requested_count"] == 50
                                and c["factors"]["cohort"] == "original"
                                and c["factors"]["output_tokens"] == c["factors"]["max_tokens"] == 16
                                and c["limits"]["token_budget"] == 2048
                                and c["limits"]["max_num_seqs"] == 4), None)
        if latency_example:
            r = latency_example["results"]
            first_token_text = "/".join(f"{v['mean_ttft_rounds']:.2f}" for v in r)
            lines += ["总轮次也不能代替首 token 延迟。例如前 50 题、G=max_tokens=16、max_seq=4、T=2048："
                      f"三策略轮次为 {'/'.join(str(v['forward_rounds']) for v in r)}，"
                      f"平均首 token 轮次为 {first_token_text}。"
                      "mixed 的 decode 优先规则在批次槽位占满时会延后后续 prompt 的接纳。", ""]
    rescues = max((r["reserved_prefill_rescues"] for c in report["comparisons"] for r in c["results"]), default=0)
    if report.get("validation"):
        lines += ["## 校验记录", "", report["validation"], ""]
    lines += [f"三策略共用指南中的无进展恢复逻辑，本次 reserved_prefill_rescues 最大值={rescues}。"
              "为 0 表示 separate 没有依赖该补充逻辑才能完成。",
              "", "轮次是 forward 次数，不是 GPU 耗时；合批、抢占和缓存复用的变化共同影响结果。"
              "应结合 query_tokens、extra_recompute_tokens、prefix_hit_tokens、mean_ttft_rounds 一起看收益。",
              "", "## 复现与文件", "", "```powershell",
              r".\.venv\Scripts\python.exe experiments/simulation/math500.py --prepare-only",
              "python experiments/simulation/math500.py", "```", "",
              "首次分词需要现有虚拟环境中的 transformers；缓存与数据、tokenizer 文件哈希一致后，"
              "模拟只需 Python 标准库。使用 `--refresh-prompts` 重新分词。",
              "", "可调整 `--requests`、`--output-lengths`、`--max-tokens`、`--max-num-seqs`、`--token-budgets`；"
              "例如 `--requests 50 --output-lengths 32,128 --max-tokens 64,128,256 --max-num-seqs 8,32`。",
              "", f"- 实际 prompt token 缓存：`{cache_path.as_posix()}`",
              f"- 完整原始结果：`{out.as_posix()}`",
              f"- 每策略一行的结果：`{out.with_suffix('.csv').as_posix()}`", ""]
    if len(exp["token_budgets"]) > 1:
        flags = (f" --requests {','.join(map(str, exp['request_counts']))}"
                 f" --output-lengths {','.join(map(str, exp['output_lengths']))}"
                 f" --max-tokens {','.join(map(str, exp['caps']))}"
                 f" --max-num-seqs {','.join(map(str, exp['seq_caps']))}"
                 f" --block-size {exp['block_size']} --num-blocks {exp['num_blocks']}"
                 f" --max-model-len {exp['max_model_len']}"
                 + (" --prefix-cache" if exp['prefix_cache'] else " --no-prefix-cache"))
        baseline = "artifacts/scheduling-simulation/math500-128x78/results.json"
        lines += ["完整复现本次两阶段扫描并合并去重：", "", "```powershell",
                  "python experiments/simulation/math500.py" + flags + f" --token-budgets 2048 --out {baseline}",
                  "python experiments/simulation/math500.py" + flags + " --to-cap-only --token-budgets "
                  + ','.join(map(str, exp['token_budgets'])) + f" --merge-results {baseline} --out {out.as_posix()}",
                  "```", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    defaults = runtime_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "offline/MATH-500/test.jsonl")
    parser.add_argument("--model", type=Path, default=Path("D:/models/Qwen3-0.6B"))
    parser.add_argument("--prompt-cache", type=Path, default=DEFAULT_OUT.parent / "prompts.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-only", type=Path, help="regenerate Markdown from a completed results JSON")
    parser.add_argument("--refresh-prompts", action="store_true")
    parser.add_argument("--requests", type=int_list, default=[50, 500])
    parser.add_argument("--output-lengths", type=int_list, default=[16, 64, 256])
    parser.add_argument("--max-tokens", type=int_list, default=[16, 64, 128, 256, 512])
    parser.add_argument("--max-num-seqs", type=int_list, default=[4, 16, 32, 64])
    parser.add_argument("--token-budget", type=int, default=defaults["max_num_batched_tokens"])
    parser.add_argument("--token-budgets", type=int_list,
                        help="per-forward token budgets to sweep; overrides --token-budget")
    parser.add_argument("--to-cap-only", action="store_true", help="only run G=max_tokens combinations")
    parser.add_argument("--merge-results", type=Path, nargs="*", default=[],
                        help="merge completed compatible sweeps into the final report")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=78)
    parser.add_argument("--max-model-len", type=int, default=defaults["max_model_len"])
    parser.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction, default=defaults["enable_prefix_cache"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path,
                        default=ROOT / "doc/混合调度与三策略性能对照实验/Math500块结构调度模拟实验报告.md")
    args = parser.parse_args(argv)
    budgets = args.token_budgets if args.token_budgets is not None else [args.token_budget]
    try:
        if args.summarize_only:
            report = json.loads(args.summarize_only.read_text(encoding="utf-8"))
            write_markdown(report, args.report, args.summarize_only, args.prompt_cache)
            print(f"Report: {args.report}")
            return 0
        for budget in budgets:
            Limits(1, budget, args.block_size, args.num_blocks, args.max_model_len, args.prefix_cache)
        if args.out.suffix.lower() != ".json":
            raise ValueError("--out must end in .json")
        if not args.to_cap_only and not args.prepare_only and max(args.output_lengths) > max(args.max_tokens):
            raise ValueError("every requested output length must fit at least one max_tokens cap")
        cached = prepare_prompts(args.dataset, args.model, args.prompt_cache, args.refresh_prompts)
        print(f"Math500 prompt lengths: {cached['length_stats']}", flush=True)
        if args.prepare_only:
            print(f"Prompt cache: {args.prompt_cache}")
            return 0
        if max(args.requests) > len(cached["prompts"]):
            raise ValueError("request count exceeds dataset size; prompts are not repeated")
        report = {
            "schema": 3, "assumptions": simulation_assumptions(args.prefix_cache),
            "dataset": {k: v for k, v in cached.items() if k != "prompts"},
            "experiment": {"block_size": args.block_size, "num_blocks": args.num_blocks,
                           "token_budgets": budgets, "max_model_len": args.max_model_len,
                           "prefix_cache": args.prefix_cache, "request_counts": args.requests,
                           "seq_caps": args.max_num_seqs, "caps": args.max_tokens,
                           "output_lengths": args.output_lengths, "to_cap_only": args.to_cap_only},
            "cohorts": [], "workloads": {}, "comparisons": [],
        }
        report["assumptions"]["tokens"] = "real Qwen3 Math500 prompt IDs; unique synthetic completion IDs"
        configurations = []
        for count in args.requests:
            original = cached["prompts"][:count]
            invalid = {cap: invalid_rows(original, cap, args) for cap in args.max_tokens}
            invalid_ids = {row["id"] for row in invalid[max(args.max_tokens)]}
            common = [row for row in original if row["id"] not in invalid_ids]
            report["cohorts"].append({
                "requested_count": count, "length_stats": length_stats(original),
                "invalid_by_cap": {cap: [row["id"] for row in rows] for cap, rows in invalid.items()},
                "common_count": len(common), "common_length_stats": length_stats(common),
            })
            cohorts = [("original", original)]
            if invalid_ids and common:
                cohorts.append(("common", common))
            for cohort, rows in cohorts:
                pairs = ([(cap, cap) for cap in args.max_tokens] if args.to_cap_only else
                         generation_pairs(args.output_lengths, args.max_tokens))
                for (output, cap), seq_cap, budget in product(pairs, args.max_num_seqs, budgets):
                    configurations.append((count, cohort, rows, output, cap, seq_cap, budget))
        print(f"B={args.block_size}, blocks={args.num_blocks}, T={budgets}; "
              f"{len(configurations)} configurations x 3 policies", flush=True)
        for index, (count, cohort, rows, output, cap, seq_cap, budget) in enumerate(configurations, 1):
            requests = requests_for(rows, output, cap)
            serial = [asdict(spec) for spec in requests]
            workload_id = hashlib.sha256(json.dumps(serial, sort_keys=True).encode()).hexdigest()[:16]
            report["workloads"].setdefault(workload_id, serial)
            limits = Limits(seq_cap, budget, args.block_size, args.num_blocks,
                            args.max_model_len, args.prefix_cache)
            results = compare(requests, limits)
            report["comparisons"].append({
                "workload_id": workload_id, "limits": asdict(limits), "results": results,
                "factors": {"requested_count": count, "request_count": len(rows), "cohort": cohort,
                            "output_tokens": output, "max_tokens": cap,
                            "generation_mode": "to_cap" if output == cap else "fixed_finish"},
            })
            if index % 4 == 0 or index == len(configurations):
                print(f"Completed {index}/{len(configurations)}; {cohort} N={len(rows)} "
                      f"G={output} cap={cap} S={seq_cap} T={budget}: "
                      f"{[r['forward_rounds'] if r['status'] == 'ok' else r['status'] for r in results]}", flush=True)
        merge_reports(report, args.merge_results)
        report["cap_invariance"] = cap_invariance(report["comparisons"])
        report["summaries"] = {
            f"N{count}/{cohort}/T{budget}/生成到上限": summarize([c for c in report["comparisons"]
                if c["factors"]["requested_count"] == count and c["factors"]["cohort"] == cohort
                and c["limits"]["token_budget"] == budget
                and c["factors"]["generation_mode"] == "to_cap"])
            for count, cohort, budget in dict.fromkeys((c["factors"]["requested_count"], c["factors"]["cohort"],
                                                       c["limits"]["token_budget"])
                                              for c in report["comparisons"])
        }
        csv_path = write_report(report, args.out)
        write_markdown(report, args.report, args.out, args.prompt_cache)
        print(json.dumps({"summaries": report["summaries"], "cap_invariance": report["cap_invariance"]}, indent=2))
        print(f"JSON: {args.out}\nCSV: {csv_path}\nReport: {args.report}")
    except (ValueError, TypeError, OSError, ImportError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
