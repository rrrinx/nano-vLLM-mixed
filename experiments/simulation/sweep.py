import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

"""Run length/request/batch experiments with the real block allocator, no model."""

import argparse
import ast
import hashlib
import json
import statistics
from dataclasses import asdict
from itertools import product
from pathlib import Path

from experiments.simulation.scheduler import (
    Limits, RequestSpec, compare, int_list, simulation_assumptions, write_report,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_SOURCE = ROOT / "nanovllm/config.py"


def runtime_defaults():
    """Read numeric dataclass defaults without importing/loading any model."""
    tree = ast.parse(CONFIG_SOURCE.read_text(encoding="utf-8"))
    config = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Config")
    names = {"max_num_batched_tokens", "max_num_seqs", "max_model_len",
             "kvcache_block_size", "enable_prefix_cache"}
    return {node.target.id: ast.literal_eval(node.value) for node in config.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
            and node.target.id in names}


def request_workload(profile, count, prompt, output):
    if profile == "uniform":
        return [RequestSpec(str(i), prompt, output) for i in range(count)]
    if profile in ("mixed_alternating", "online_mixed"):
        pairs = [(64, 16), (768, 256)] * ((count + 1) // 2)
    elif profile == "mixed_cross":
        pairs = [(64, 256), (768, 16)] * ((count + 1) // 2)
    elif profile in ("short_first", "long_first"):
        pairs = [(64, 16)] * (count // 2) + [(768, 256)] * (count - count // 2)
        if profile == "long_first":
            pairs.reverse()
    elif profile == "shared_prefix_512":
        return [RequestSpec(str(i), 768, 128, shared_prefix_tokens=512) for i in range(count)]
    else:
        raise ValueError(f"unknown workload profile: {profile}")
    return [RequestSpec(str(i), p, g, arrival_round=4 * i if profile == "online_mixed" else 0)
            for i, (p, g) in enumerate(pairs[:count])]


def markdown_row(case):
    factors, limits = case["factors"], case["limits"]
    results = case["results"]
    rounds = [str(r["forward_rounds"]) if r["status"] == "ok" else r["status"] for r in results]
    saving = results[2]["round_reduction_pct"]
    saving_text = f"{saving:+.2f}%" if saving is not None else "—"
    preemptions = "/".join(str(r["preemptions"]) for r in results)
    return (f"| {factors['prompt_tokens']} | {factors['output_tokens']} | {factors['request_count']} "
            f"| {limits['max_num_seqs']} | {rounds[0]} | {rounds[1]} | {rounds[2]} "
            f"| {saving_text} | {preemptions} |")


def describe(case):
    f = case["factors"]
    return (f"P={f['prompt_tokens']}、G={f['output_tokens']}、N={f['request_count']}、"
            f"S={case['limits']['max_num_seqs']}")


def summarize(cases):
    complete = [case for case in cases if all(r["status"] == "ok" for r in case["results"])]
    summary = {"comparisons": len(cases), "complete_comparisons": len(complete)}
    for index, policy in enumerate(("interleaved", "mixed"), 1):
        values = [case["results"][index]["round_reduction_pct"] for case in complete]
        summary[policy] = {
            "fewer_rounds": sum(v > 1e-9 for v in values),
            "same_rounds": sum(abs(v) <= 1e-9 for v in values),
            "more_rounds": sum(v < -1e-9 for v in values),
            "median_reduction_pct": statistics.median(values) if values else None,
            "min_reduction_pct": min(values) if values else None,
            "max_reduction_pct": max(values) if values else None,
        }
    return summary


def write_markdown(report, path, out, trace_path):
    uniform = [case for case in report["comparisons"] if case["factors"]["profile"] == "uniform"]
    valid = [case for case in uniform if all(r["status"] == "ok" for r in case["results"])]
    summary = report["summary"]
    context = report["experiment"]
    m = summary["mixed"]
    lines = [
        f"# {context['block_size']}×{context['num_blocks']} 块结构调度模拟实验", "",
        f"固定 block_size={context['block_size']}、block_num={context['num_blocks']}、"
        f"每轮 query token 预算={context['token_budget']}、max_model_len={context['max_model_len']}。"
        f"prefix cache={'开启' if context['prefix_cache'] else '关闭'}；"
        "均匀负载的 prompt 内容互不相同，不会因为长度相同就共享块。",
        "", "## 实验方法", "",
        f"主扫描：P={context['prompt_lengths']}，G={context['output_lengths']}，"
        f"请求数 N={context['request_counts']}，批次上限 S={context['seq_caps']}。"
        "主扫描全部第 0 轮到达；同一组负载分别运行三种策略。",
        "",
        "G 是 max_tokens，模拟所有请求生成到上限，没有提前 EOS。"
        "P+G−1 必须同时满足模型位置上限和单请求 KV 容量检查。"
        "本轮最后一个 prefill chunk 会产生第一个输出 token。",
        "",
        "KV 使用当前生产 BlockManager：真实物理块 ID、每请求块表、FIFO 空闲队列、"
        "ref_count、完整块哈希、冷缓存复用和覆盖淘汰。首次 prefill 预留完整当前历史；"
        "decode 跨块时扩容，空间不足从尚未选中的运行队尾抢占。共享块引用归零才进入空闲队列。",
        "",
        "三种策略的选择器遵循补充指南，包括无进展时继续已有块的部分 prefill。"
        "报告记录 reserved_prefill_rescues；若该值为 0，则本轮 separate 没有触发指南对极端无进展状态的修复。",
        "", "## 主扫描结果", "",
        f"主扫描共 {summary['comparisons']} 组配置，{summary['complete_comparisons']} 组全部完成。"
        f"mixed 比 separate 少跑轮次 {m['fewer_rounds']} 组、相同 {m['same_rounds']} 组、"
        f"更多 {m['more_rounds']} 组。",
    ]
    if m["median_reduction_pct"] is not None:
        lines += ["", f"mixed 轮次减少比例的中位数为 {m['median_reduction_pct']:.2f}%，"
                  f"范围 {m['min_reduction_pct']:.2f}% 至 {m['max_reduction_pct']:.2f}%。"
                  "这是对配置逐组统计的分布，不是运行时间加速比。"]
    if valid:
        best = max(valid, key=lambda case: case["results"][2]["round_reduction_pct"])
        worst = min(valid, key=lambda case: case["results"][2]["round_reduction_pct"])
        for label, case in (("mixed 轮次减少最多", best), ("mixed 轮次减少最少", worst)):
            s, i, mixed = case["results"]
            lines += ["", f"{label}：{describe(case)}；轮次 S/I/M="
                      f"{s['forward_rounds']}/{i['forward_rounds']}/{mixed['forward_rounds']}，"
                      f"mixed 减少 {mixed['round_reduction_pct']:.2f}%。"
                      f"实际 query 数 separate/mixed={s['query_tokens']}/{mixed['query_tokens']}，"
                      f"额外重算 token={s['extra_recompute_tokens']}/{mixed['extra_recompute_tokens']}。"]
    lines += ["", "下表摘取四个因素的长短/多少组合。P 是 prompt 长度，G 是输出上限，"
              "N 是请求数，S 是每批请求上限。S/I/M 列分别是 separate/interleaved/mixed 的 forward 轮次；"
              "抢占次数按 S/I/M 顺序列出。其余中间值配置在 CSV 中。", "",
              "| P | G | N | S | separate | interleaved | mixed | mixed轮次减少 | 抢占 S/I/M |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---|" ]
    p_values = {min(context["prompt_lengths"]), max(context["prompt_lengths"])}
    g_values = {min(context["output_lengths"]), max(context["output_lengths"])}
    n_values = {min(context["request_counts"]), max(context["request_counts"])}
    s_values = {min(context["seq_caps"]), 32 if 32 in context["seq_caps"] else max(context["seq_caps"])}
    lines += [markdown_row(case) for case in uniform if case["factors"]["prompt_tokens"] in p_values
              and case["factors"]["output_tokens"] in g_values
              and case["factors"]["request_count"] in n_values
              and case["limits"]["max_num_seqs"] in s_values]
    representative = next((c for c in uniform if c["factors"]["prompt_tokens"] == 768
                           and c["factors"]["output_tokens"] == 16
                           and c["factors"]["request_count"] == 128
                           and c["limits"]["max_num_seqs"] == 32
                           and all(r["status"] == "ok" for r in c["results"])), None)
    if representative and context["block_size"] == 128 and context["num_blocks"] == 78:
        separate, _, mixed = representative["results"]
        lines += ["", "P768/G16/N128/S32 体现了块边界和调度顺序的共同影响："
                  "每条 prompt 首次占 6 块，78 块最多容纳 13 条这样的请求；"
                  "第一次 decode 的输入长度变为 769，需要第 7 块，因而最多同时容纳 11 条。"
                  "separate 优先接纳 prefill，随后 decode 扩块时可能抢占；mixed 先为 decode 扩块，"
                  "再检查是否接纳新的 prefill。",
                  "", f"这组实测模拟中 separate/mixed 的抢占次数为 "
                  f"{separate['preemptions']}/{mixed['preemptions']}，额外重算 token 为 "
                  f"{separate['extra_recompute_tokens']}/{mixed['extra_recompute_tokens']}。"
                  "因此这一组的轮次改善同时包含同轮合批和减少抢占重算的影响。"]
    extras = [case for case in report["comparisons"] if case["factors"]["profile"] != "uniform"]
    lines += ["", "## 补充场景", "",
              "长度混合、先短后长/先长后短、明确共享 512-token 前缀、在线间隔到达，"
              "用于观察同一块预算下不同请求次序和共享关系的影响。", "",
              "| 场景 | N | S | separate | interleaved | mixed | mixed轮次减少 | prefix命中token S/I/M |",
              "|---|---:|---:|---:|---:|---:|---:|---|" ]
    for case in extras:
        f, limits = case["factors"], case["limits"]
        results = case["results"]
        counts = [str(r["forward_rounds"]) if r["status"] == "ok" else r["status"] for r in results]
        saving = results[2]["round_reduction_pct"]
        saving_text = f"{saving:+.2f}%" if saving is not None else "—"
        hits = "/".join(str(r["prefix_hit_tokens"]) for r in results)
        lines.append(f"| {f['profile']} | {f['request_count']} | {limits['max_num_seqs']} "
                     f"| {counts[0]} | {counts[1]} | {counts[2]} | {saving_text} | {hits} |")
    lines += ["", "mixed_alternating 为 (P64/G16、P768/G256) 交替；mixed_cross 为 "
              "(P64/G256、P768/G16) 交替；short_first/long_first 使用相同的一半短短请求和一半长长请求，"
              "只改变入队顺序；shared_prefix_512 为 P768/G128 且共享前512个token；"
              "online_mixed 使用 mixed_alternating 的长度，每隔4个模拟轮次到达一条。", "",
              "## 128-token 边界", "",
              "比较 P127、P128、P129，固定 G16/N128/S32。P129 首次就需要两块；"
              "P128 首次一块但第一次 decode 就跨块；P127 第一次 decode 仍写入原块。", "",
              "| P | G | N | S | separate | interleaved | mixed | mixed轮次减少 | 抢占 S/I/M |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---|" ]
    lines += [markdown_row(case) for case in report["comparisons"]
              if case["factors"]["profile"] in ("uniform", "boundary_127", "boundary_129")
              and case["factors"]["prompt_tokens"] in (127, 128, 129)
              and case["factors"]["output_tokens"] == 16
              and case["factors"]["request_count"] == 128 and case["limits"]["max_num_seqs"] == 32]
    max_rescues = max((r["reserved_prefill_rescues"] for c in report["comparisons"] for r in c["results"]), default=0)
    lines += ["", "## 阅读结果", "",
              f"本次所有结果中 reserved_prefill_rescues 的最大值为 {max_rescues}。"
              "peak_kv_blocks 统计实际被引用的物理块，peak_block_references 统计块引用数；"
              "共享场景中这两个数可以不同。",
              "",
              "看轮次时也要看 query_tokens、extra_recompute_tokens 和 prefix_hit_tokens。"
              "如果缓存驻留/抢占发生变化，轮次差异可能同时包含重算和冷前缀复用的变化。"
              "max_num_seqs 是单批上限，当前实现不会用它限制 running 队列总长度。",
              "",
              "每次 forward 的实际耗时还受输入量、历史长度、算子和设备影响。"
              "这些实验只执行真实块管理与调度状态推进，没有执行神经网络，不能把轮次比作为 GPU 时间加速比。",
              "", "## 复现与文件", "", "```powershell", "python experiments/simulation/sweep.py", "```", "",
              f"原始 JSON：`{out.as_posix()}`；CSV：`{out.with_suffix('.csv').as_posix()}`。"]
    if trace_path:
        lines += ["", f"逐轮块表示例：`{trace_path.as_posix()}`，其中保留每次选批的物理块 ID、"
                  "前缀命中长度、free_blocks 和提交后的 ref_count。"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    defaults = runtime_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-lengths", type=int_list, default=[64, 128, 512, 768])
    parser.add_argument("--output-lengths", type=int_list, default=[16, 128, 256])
    parser.add_argument("--requests", type=int_list, default=[8, 32, 128])
    parser.add_argument("--max-num-seqs", type=int_list, default=[4, 16, 32, 64])
    parser.add_argument("--token-budget", type=int, default=defaults["max_num_batched_tokens"])
    parser.add_argument("--block-size", type=int, default=defaults["kvcache_block_size"])
    parser.add_argument("--num-blocks", type=int, default=78)
    parser.add_argument("--max-model-len", type=int, default=defaults["max_model_len"])
    parser.add_argument("--prefix-cache", action=argparse.BooleanOptionalAction,
                        default=defaults["enable_prefix_cache"])
    parser.add_argument("--supplemental", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--record-example", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/scheduling-simulation/blocks-128x78/results.json"))
    parser.add_argument("--report", type=Path,
                        default=Path("doc/混合调度与三策略性能对照实验/128x78块结构模拟实验报告.md"))
    args = parser.parse_args(argv)
    if args.out.suffix.lower() != ".json":
        parser.error("--out must end in .json; a matching .csv is also written")
    configurations = [("uniform", n, p, g, s) for p, g, n, s in product(
        args.prompt_lengths, args.output_lengths, args.requests, args.max_num_seqs)]
    if args.supplemental:
        for profile, n, s in product(
                ("mixed_alternating", "mixed_cross", "short_first", "long_first", "shared_prefix_512"),
                (32, 128), (4, 32)):
            configurations.append((profile, n, 0, 0, s))
        configurations.extend(("online_mixed", n, 0, 0, 32) for n in (32, 128))
        configurations.extend((f"boundary_{p}", 128, p, 16, 32) for p in (127, 129))
    report = {
        "schema": 2, "assumptions": simulation_assumptions(args.prefix_cache),
        "experiment": {"runtime_defaults": defaults, "config_source_sha256": hashlib.sha256(CONFIG_SOURCE.read_bytes()).hexdigest(),
                       "block_size": args.block_size, "num_blocks": args.num_blocks,
                       "token_budget": args.token_budget, "max_model_len": args.max_model_len,
                       "prefix_cache": args.prefix_cache,
                       "prompt_lengths": args.prompt_lengths, "output_lengths": args.output_lengths,
                       "request_counts": args.requests, "seq_caps": args.max_num_seqs},
        "workloads": {}, "comparisons": [],
    }
    print(f"B={args.block_size}, blocks={args.num_blocks}, T={args.token_budget}, "
          f"max_model_len={args.max_model_len}, prefix_cache={args.prefix_cache}; "
          f"{len(configurations)} configurations x 3 policies", flush=True)
    try:
        for index, (profile, n, p, g, s) in enumerate(configurations, 1):
            requests = request_workload("uniform" if profile.startswith("boundary_") else profile, n, p, g)
            rows = [asdict(spec) for spec in requests]
            workload_id = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:16]
            report["workloads"][workload_id] = rows
            limits = Limits(s, args.token_budget, args.block_size, args.num_blocks,
                            args.max_model_len, args.prefix_cache)
            results = compare(requests, limits)
            report["comparisons"].append({
                "workload_id": workload_id, "limits": asdict(limits), "results": results,
                "factors": {"profile": profile, "request_count": n,
                            "prompt_tokens": p if p else "/".join(map(str, sorted({r.prompt_tokens for r in requests}))),
                            "output_tokens": g if g else "/".join(map(str, sorted({r.output_tokens for r in requests})))},
            })
            if index % 12 == 0 or index == len(configurations):
                print(f"Completed {index}/{len(configurations)}; last {profile} N={n} S={s}, "
                      f"rounds={[r['forward_rounds'] for r in results]}", flush=True)
        uniform = [c for c in report["comparisons"] if c["factors"]["profile"] == "uniform"]
        report["summary"] = summarize(uniform)
        trace_path = None
        if args.record_example:
            example = next((c for c in uniform if c["factors"]["prompt_tokens"] == 768
                            and c["factors"]["output_tokens"] == 256 and c["factors"]["request_count"] == 32
                            and c["limits"]["max_num_seqs"] == 32), None)
            if example:
                requests = [RequestSpec(**row) for row in report["workloads"][example["workload_id"]]]
                trace_path = args.out.parent / "block-trace-long-long.json"
                trace = {"schema": 2, "assumptions": report["assumptions"], "requests": [asdict(r) for r in requests],
                         "limits": example["limits"], "results": compare(requests, Limits(**example["limits"]), record_steps=True)}
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
        csv_path = write_report(report, args.out)
        write_markdown(report, args.report, args.out, trace_path)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"JSON: {args.out}\nCSV: {csv_path}\nReport: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
