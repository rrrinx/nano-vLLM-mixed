"""Build the burst report from all published formal records (stdlib only)."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("en", "zh-CN"), default="en")
    args = parser.parse_args()
    base = ROOT if args.language == "en" else ROOT / "docs/zh-CN"
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
            cells.append(f'{baseline:.2f} → {mixed:.2f} ({100*(mixed/baseline-1):+.2f}%)')
        rows.append('| ' + ' | '.join([case, *cells]) + ' |')
    template = ROOT / "tools/templates" / f"burst_report.{args.language}.md"
    body = template.read_text(encoding="utf-8").replace("{rows}", "\n".join(rows))
    output = base / 'reports/burst_requests/README.md'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body, encoding='utf-8', newline='\n')
    print('Burst report generated from 36 files / 108 runs / all 12 configurations.')


if __name__ == '__main__':
    main()
