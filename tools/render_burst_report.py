"""Build the 突发请求实验 reading guide from all published formal records (stdlib only)."""
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def main():
    formal = ROOT / "reports/data/burst_requests/formal"
    groups = defaultdict(lambda: defaultdict(list))
    for path in sorted(formal.rglob("results.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["status"] == "complete"
        for run in data["runs"]:
            groups[path.parent.name][run["policy"]].append(run)
    assert len(groups) == 12
    rows = []
    for case, policies in groups.items():
        assert all(len(runs) == 3 for runs in policies.values())
        def metric(policy, field):
            values = []
            for run in policies[policy]:
                item = run
                for key in field.split('.'):
                    item = item[key]
                values.append(item)
            return mean(values)
        fields = ('summary.output_tok_s', 'online.groups.late_prefill.ttft_ms.mean',
                  'online.groups.late_prefill.e2e_ms.mean', 'online.window_old_itl_ms.max')
        cells = []
        for field in fields:
            baseline, mixed = metric('interleaved', field), metric('mixed', field)
            cells.append(f'{baseline:.2f} → {mixed:.2f}（{100*(mixed/baseline-1):+.2f}%）')
        rows.append('| ' + ' | '.join([case, *cells]) + ' |')
    body = '''# 突发请求实验：受控突发请求确认

三个确认输入组、四档 token budget、三策略和三个执行 block，共 108 次正式策略回放。每个输入组有四个已有 Decode 请求（64-token 输入前缀、256-token 输出）和四个后到 Prefill 请求（完整长文本、32-token 输出）。以下每个值为三个执行 block 的均值。

## 相对 interleaved 的全部配置

箭头为 interleaved → mixed。百分比使用 `(mixed / interleaved - 1) × 100%`。吞吐正值更好，延迟负值更好。旧请求窗口最大 ITL 为每次回放中规定窗口内旧请求的最大 token 间隔，再对三个回放求均值；不是所有请求延迟的 p95。

| 输入组 / 预算 | 输出吞吐 token/s | 后到请求平均 TTFT ms | 后到请求平均 E2E ms | 旧请求窗口最大 ITL ms |
| --- | --- | --- | --- | --- |
''' + '\n'.join(rows) + '''

## 如何解释

T384 三个输入组中，相对 interleaved，后到请求 TTFT、完成时间以及旧请求的窗口最大间隔均改善。这个受控负载说明 mixed 能够同时推进两类请求，但不意味着所有场景、所有基线下 TTFT 都改善。

相对 separate 的收益与代价见 [完整配对分析](../data/burst_requests/formal/analysis.md)；该基线下后到请求 TTFT 更长。长输入在线实验 在持续到达负载下进一步观察到完成时间改善与首 token 等待增加的权衡。不能把两种负载的获益指标合并成“所有延迟均下降”。

## 记录和限制

- [全部正式测量](../data/burst_requests/formal/)，按 block0、block3、block4 和 confirm0/1/2_T* 定位；[公开数据导出说明](../data/README.md)。
- [reuse.json](../data/burst_requests/formal/reuse.json) 记录从先前中断批次复用的 12 个完整调用；这些记录只计入一次。
- 所有正式调用均无抢占，但输出哈希跨策略不一致。不能声称收益已在答案等价前提下得到证明。
- 每配置三个执行重复；配对置信区间只描述重复执行波动，未经多重比较校正，不是跨任意工作负载的保证。
- 重新生成本页：`python tools/render_burst_report.py`；全数据核验：`python tools/verify_results.py`。
'''
    output = ROOT / 'reports/burst_requests/README.md'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body, encoding='utf-8', newline='\n')
    print('突发请求实验 report generated from 36 files / 108 runs / all 12 configurations.')


if __name__ == '__main__':
    main()
