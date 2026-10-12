"""Preserve every replay, summarize without pooling different repeats."""

import csv
import json

from experiments.common.repeat_stats import summarize_repeats


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        elif isinstance(item, (int, float)) or item is None:
            result[name] = item
    return result


def summarize_runs(runs, policies, trim_fraction):
    rows = []
    for policy in policies:
        selected = [r for r in runs if r["policy"] == policy]
        flattened = [flatten({"summary": r["summary"], "online": r["online"]}) for r in selected]
        keys = sorted({key for item in flattened for key in item})
        metrics = {key: summarize_repeats([item.get(key) for item in flattened], trim_fraction) for key in keys}
        rows.append({"policy": policy, "repeats": len(selected),
                     "output_stable_within_policy": len({r["output_sha256"] for r in selected}) == 1,
                     "metrics": metrics})
    return rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_report(out, payload):
    (out / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(out / "runs.csv", [
        {"round": r["round"], "policy_position": r["policy_position"], "policy": r["policy"],
         "output_sha256": r["output_sha256"], **flatten({"summary": r["summary"], "online": r["online"]})}
        for r in payload["runs"]])
    write_csv(out / "requests.csv", [
        {"round": r["round"], "policy": r["policy"], **request}
        for r in payload["runs"] for request in r["requests"]])
    write_csv(out / "summary.csv", [
        {"policy": row["policy"], "metric": name, **stats}
        for row in payload["summary"] for name, stats in row["metrics"].items()])

    def number(value):
        return "—" if value is None else f"{value:.3f}"

    lines = ["# 在线请求回放结果", "", "这是引擎级墙钟回放，延迟包含计划到达后的等待，不含 HTTP/网络。",
             "下表是各次 replay 指标的跨重复中位数；延迟单位 ms。没有合并各轮请求分布。", "",
             f"执行状态：{payload['status']}；输出一致性：{payload['matching_outputs']}。", "",
             "| 策略 | tok/s | 整体 TTFT p95 | 整体 ITL p95 | TPOT 均值 | E2E p95 | mixed 占比 | 抢占 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in payload["summary"]:
        keys = ("summary.output_tok_s", "online.groups.all.ttft_ms.p95", "online.groups.all.itl_ms.p95",
                "online.groups.all.tpot_ms.mean", "online.groups.all.e2e_ms.p95",
                "summary.mixed_step_fraction", "summary.counts.preemptions")
        lines.append("| " + " | ".join([row["policy"]] + [number(row["metrics"].get(k, {}).get("median")) for k in keys]) + " |")
    lines += ["", "## 整体延迟分布", "",
              "| 策略 | 指标 | 均值 | p50 | p95 | p99 | 最大值 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in payload["summary"]:
        for metric in ("ttft_ms", "tpot_ms", "itl_ms", "e2e_ms"):
            keys = [f"online.groups.all.{metric}.{stat}" for stat in ("mean", "p50", "p95", "p99", "max")]
            lines.append("| " + " | ".join([row["policy"], metric] + [number(row["metrics"][k]["median"]) for k in keys]) + " |")
    groups = sorted({r["group"] for r in payload["trace"]["requests"]})
    if len(groups) > 1:
        lines += ["", "| 策略 | 请求组 | TTFT p95 | ITL p95 | 最大 ITL（组内最坏值） | TPOT 均值 | E2E p95 |",
                  "|---|---|---:|---:|---:|---:|---:|"]
    for row in payload["summary"]:
        for group in groups if len(groups) > 1 else []:
            keys = [f"online.groups.{group}.{metric}" for metric in (
                "ttft_ms.p95", "itl_ms.p95", "max_itl_ms.max", "tpot_ms.mean", "e2e_ms.p95")]
            lines.append("| " + " | ".join([row["policy"], group] + [number(row["metrics"].get(k, {}).get("median")) for k in keys]) + " |")
    if {"old_decode", "late_prefill"} <= set(groups):
        lines += ["", "| 策略 | 干扰窗口老请求 ITL p95 | 窗口最大 ITL | 到达时仍在 decode 的老请求数 | 接纳时仍在 decode 的老请求数 |",
                  "|---|---:|---:|---:|---:|"]
        for row in payload["summary"]:
            keys = ("online.window_old_itl_ms.p95", "online.window_old_itl_ms.max",
                    "online.overlap.old_decoding_at_arrival", "online.overlap.old_decoding_at_admission")
            lines.append("| " + " | ".join([row["policy"]] + [number(row["metrics"][k]["median"]) for k in keys]) + " |")
    if payload["settings"]["ttft_slo_ms"] is not None or payload["settings"]["itl_slo_ms"] is not None:
        lines += ["", "| 策略 | SLO 达标比例 | SLO goodput（请求/s） |", "|---|---:|---:|"]
        for row in payload["summary"]:
            keys = ("online.groups.all.slo_pass_fraction", "online.groups.all.slo_goodput_requests_s")
            lines.append("| " + " | ".join([row["policy"]] + [number(row["metrics"][k]["median"]) for k in keys]) + " |")
    lines += ["", "## 结果判读", ""]
    lines += [f"- {warning}" for warning in payload["warnings"]] or ["- 未触发自动提示；仍需结合样本数和重复波动判断。"]
    lines += ["", "summary.csv 为逐指标的均值/中位数/截尾均值/std/CV/极值；runs.csv 保留各次计数和分布；",
              "requests.csv 保留每请求的首末 token 时刻及延迟。results.json 保存配置、负载、环境、源码指纹和全部结果。",
              "p99 和固定干扰窗口指标见 JSON/CSV；单 token 请求没有 TPOT/ITL，空值没有填为 0。",
              "SLO（若配置）要求 TTFT 和每请求最坏 ITL 满足相应阈值，goodput 按完整 replay 耗时计算。",
              "低输入速率下吞吐包含等待到达时间，不能直接当作 GPU 最大吞吐。"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
