"""Run selected, stable block IDs; copy audited completed calls into a NEW matrix."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from audit import audit, read
from block_design import DEFAULT_BLOCK_IDS, ORDERS, POLICIES, parse_block_ids, position_balanced


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity(app):
    sources = sorted((app / 'nanovllm').rglob('*.py')) + sorted((app / 'experiments').rglob('*.py'))
    sources.append(app / 'experiments/benchmark.py')
    return {p.relative_to(app).as_posix(): sha256(p) for p in sources
            if 'results' not in p.relative_to(app).parts and '.ipynb_checkpoints' not in p.parts}


def checkpoint_identity(model):
    model = Path(model)
    found = {p.name: {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
             for p in sorted(model.glob('*.safetensors'))}
    index = model / 'model.safetensors.index.json'
    if index.exists():
        found[index.name] = {'sha256': sha256(index)}
    return found


def make_schedule(configs, out, block_ids):
    if not configs or len(configs) != len(set(configs)):
        raise ValueError('Supply distinct configurations')
    if len({p.stem for p in configs}) != len(configs):
        raise ValueError('Configuration names must be unique')
    schedule = []
    for block in block_ids:
        rotation = block % len(configs)
        for slot, path in enumerate(configs[rotation:] + configs[:rotation]):
            config = read(path)
            if (config['policies'] != POLICIES or config['warmup'] < 1
                    or config['save_details'] or config['diagnostics']
                    or not config['trace'] or config.get('source') or not config['ignore_eos']):
                raise ValueError(f'Invalid formal configuration: {path}')
            trace_path = Path(config['trace']).resolve()
            schedule.append({'block': block, 'slot': slot, 'config': str(path),
                             'config_sha256': sha256(path), 'trace_file_sha256': sha256(trace_path),
                             'set': trace_path.stem, 'budget': f"T{config['token_budget']}",
                             'token_budget': config['token_budget'], 'policy_order': list(ORDERS[block]),
                             'out': str(out / f'block{block}' / path.stem)})
    return schedule


def check_result(result, entry, config, expected_sources, mode, reference=None):
    audit(result, mode=mode)
    if [r['policy'] for r in result['runs']] != entry['policy_order']:
        raise ValueError('Result policy order differs from selected block')
    if any(r.get('policy_position') != i for i, r in enumerate(result['runs'], 1)):
        raise ValueError('Result policy positions differ from selected block')
    if result['trace'] != read(Path(config['trace'])):
        raise ValueError('Result workload differs from frozen trace')
    expected = dict(config, rounds=1, save_details=False, diagnostics=False)
    # Trace contents and actual execution order are checked separately from paths.
    for key, value in expected.items():
        if key not in ('out', 'trace', 'policy_orders') and result['settings'].get(key) != value:
            raise ValueError(f'Result setting differs: {key}')
    if result['source_sha256'] != expected_sources:
        raise ValueError('Inference/measurement source fingerprint differs; cannot reuse or pool')
    if result['checkpoint_files'] != checkpoint_identity(config['model']):
        raise ValueError('Checkpoint file identity differs; cannot reuse or pool')
    if reference is not None:
        if result['environment'] != reference['environment']:
            raise ValueError('Environment/GPU identity differs from the current pilot')
        if result['checkpoint_files'] != reference['checkpoint_files']:
            raise ValueError('Checkpoint identity differs from the current pilot')


def reuse_index(path):
    if path is None:
        return {}
    root = path.resolve()
    if not (root / 'schedule.json').is_file():
        root = root / 'formal'
    indexed, seen = {}, set()
    for e in read(root / 'schedule.json')['entries']:
        key = (e['block'], Path(e['config']).stem)
        if key in seen:
            raise ValueError('Duplicate calls in reuse source schedule')
        seen.add(key)
        result = root / f"block{e['block']}" / key[1] / 'results.json'
        if result.is_file():
            # A failed.json or partial per-policy output is deliberately not reused.
            indexed[key] = (e, result)
    return indexed


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configs', nargs='+', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--mode', choices=['burst', 'boundary', 'stream'], default='burst')
    p.add_argument('--block-ids', default=DEFAULT_BLOCK_IDS)
    p.add_argument('--reuse-from', type=Path, help='Stopped prior stage or formal directory; never modified')
    p.add_argument('--reference-result', type=Path, help='Current pilot for strict environment verification')
    a = p.parse_args(argv)
    block_ids = parse_block_ids(a.block_ids)
    root = Path.cwd()
    app = root
    runner = app / 'experiments/online/run_online.py'
    configs = [path.resolve() for path in a.configs]
    out = a.out.resolve()
    if out.exists():
        raise ValueError('Output must be a new directory')
    schedule = make_schedule(configs, out, block_ids)
    expected_sources = source_identity(app)
    reference = read(a.reference_result) if a.reference_result else None
    if a.execute and a.reuse_from and reference is None:
        raise ValueError('Reusing results requires a current --reference-result pilot')
    if reference and reference['source_sha256'] != expected_sources:
        raise ValueError('Current pilot source fingerprint differs')
    previous = reuse_index(a.reuse_from)
    reusable = {}
    for entry in schedule:
        key = (entry['block'], Path(entry['config']).stem)
        if key not in previous:
            continue
        old, result_path = previous[key]
        if old['policy_order'] != entry['policy_order'] or old['token_budget'] != entry['token_budget']:
            raise ValueError(f'Reuse schedule differs: {key}')
        check_result(read(result_path), entry, read(Path(entry['config'])), expected_sources, a.mode, reference)
        reusable[key] = (result_path, sha256(result_path))

    def command(entry, dry=False):
        cmd = [sys.executable, str(runner), '--config', entry['config'], '--out', entry['out'],
               '--rounds', '1', '--policy-orders', ','.join(entry['policy_order'])]
        return cmd + (['--dry-run'] if dry else [])

    for entry in schedule[:len(configs)]:
        subprocess.run(command(entry, dry=True), check=True, cwd=root)
    print('Block IDs:', block_ids, 'position-balanced:', position_balanced(block_ids), flush=True)
    print('Invocations:', len(schedule), 'Formal replays:', len(schedule) * len(POLICIES),
          'Reusable calls:', len(reusable), 'New calls:', len(schedule) - len(reusable), flush=True)
    if not a.execute:
        return
    out.mkdir(parents=True)

    def write(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    write('schedule.json', {'entries': schedule, 'blocks': len(block_ids), 'block_ids': block_ids,
                           'policies': POLICIES, 'position_balanced': position_balanced(block_ids),
                           'reuse_from': str(a.reuse_from.resolve()) if a.reuse_from else None,
                           'tool_sha256': sha256(__file__)})
    completed, reused = [], []
    try:
        for entry in schedule:
            path = Path(entry['config'])
            config = read(path)
            if sha256(path) != entry['config_sha256'] or sha256(config['trace']) != entry['trace_file_sha256']:
                raise ValueError('Configuration or trace changed during execution')
            if source_identity(app) != expected_sources:
                raise ValueError('Inference/measurement code changed during execution')
            key = (entry['block'], path.stem)
            if key in reusable:
                old_path, old_hash = reusable[key]
                if sha256(old_path) != old_hash:
                    raise ValueError('Reuse source changed; stop the old experiment before reusing it')
                print('REUSE', entry['block'], path.stem, old_path, flush=True)
                target = Path(entry['out'])
                target.mkdir(parents=True)
                for name in ('results.json', 'settings.json', 'workload.json', 'report.md',
                             'summary.csv', 'runs.csv', 'requests.csv'):
                    old_file = old_path.parent / name
                    if old_file.is_file():
                        shutil.copy2(old_file, target / name)
                if sha256(target / 'results.json') != old_hash:
                    raise ValueError('Copied result checksum differs')
                reused.append({'block': entry['block'], 'case': path.stem,
                               'source_record': str(old_path), 'sha256': old_hash})
                write('reuse.json', {'entries': reused})
            else:
                print('START', entry['block'], path.stem, entry['policy_order'], flush=True)
                subprocess.run(command(entry), check=True, cwd=root)
            result = read(Path(entry['out']) / 'results.json')
            check_result(result, entry, config, expected_sources, a.mode, reference)
            if reference is None:
                reference = result
            completed.append(entry)
            write('blocks.json', {'entries': completed})
    except BaseException as error:
        write('failure.json', {'completed_invocations': len(completed), 'error': repr(error)})
        raise
    print('COMPLETE', out)


if __name__ == '__main__':
    main()
