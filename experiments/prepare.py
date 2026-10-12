import json
import math
import os
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

root = Path.cwd()
app = root
sys.path.insert(0, str(app))
from experiments.benchmark import digest
from experiments.math500.run_math500 import encode_problem
from experiments.online.online_workload import file_hash, model_identity, validate_trace
from experiments.online.run_online import parse_options, preflight

model = str(Path(os.environ['MODEL']).resolve())
plan = Path(os.environ['EXPERIMENT_PLAN']).resolve()
delay = float(os.environ.get('BURST_DELAY', '1.0'))
blocks = int(os.environ.get('KV_BLOCKS', '565'))
assert math.isfinite(delay) and delay > 0 and blocks > 0
if plan.exists():
    raise ValueError('Use a new EXPERIMENT_PLAN; existing plans are never overwritten')
source = root / 'offline/SPEED-Bench/throughput_2k/test-00000-of-00001.parquet'
records = pq.read_table(source).to_pylist()
source_hash = file_hash(source)
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
p2 = read(app / 'experiments/metadata/exploration_samples.json')
abc = read(app / 'experiments/metadata/excluded_samples.json')
assert source_hash == p2['source_file_sha256'], 'Parquet identity changed'
explore = [r['dataset_index'] for r in p2['selected']['set0']]
for row in p2['selected']['set0']:
    assert records[row['dataset_index']]['question_id'] == row['question_id']
used = {r['index'] for r in abc['selected']}
used.update(r['dataset_index'] for group in p2['selected'].values() for r in group)
hf, identity = model_identity(model)
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
tokens, pools = {}, {'high_entropy': [], 'low_entropy': []}
for index, row in enumerate(records):
    turns = row.get('turns') or []
    if len(turns) != 1 or not isinstance(turns[0], str) or not turns[0].strip():
        continue
    if 'FULL BENCHMARK DATA SHOULD BE FETCHED' in turns[0]:
        continue
    if row.get('category') not in pools:
        continue
    ids = encode_problem(tokenizer, turns[0])
    tokens[index] = ids
    if index not in used and 1900 <= len(ids) < 2200:
        pools[row['category']].append(index)
assert len(explore) == 8 and all(i in tokens and len(tokens[i]) >= 64 for i in explore)
rng = random.Random(20261010)
sets = {'explore': explore}
for name in ['confirm0', 'confirm1', 'confirm2']:
    picks = {c: rng.sample([i for i in pool if i not in used], 4)
             for c, pool in sorted(pools.items())}
    order = [picks[c][j] for j in range(4) for c in sorted(picks)]
    used.update(order)
    sets[name] = order
stream_picks = {c: rng.sample([i for i in pool if i not in used], 50)
                for c, pool in sorted(pools.items())}
stream_ids = [stream_picks[c][j] for j in range(50) for c in sorted(stream_picks)]

def provenance(index):
    row = records[index]
    return {'dataset_index': index, 'question_id': row['question_id'],
            'category': row['category'], 'qwen_chat_tokens': len(tokens[index])}

def trace(requests, indexes, description):
    return {'schema': 'online-v1', 'synthetic': True, **identity,
            'requests': requests, 'workload_sha256': digest(requests),
            'source_file_sha256': source_hash,
            'provenance': [provenance(i) for i in indexes], 'description': description}

traces = {}
for name, indexes in sets.items():
    requests = []
    for position, index in enumerate(indexes):
        old = position < 4
        requests.append({'id': ('old_' if old else 'late_') + records[index]['question_id'],
                         'dataset_index': index, 'group': 'old_decode' if old else 'late_prefill',
                         'arrival_s': 0.0 if old else delay,
                         'prompt_token_ids': tokens[index][:64] if old else tokens[index],
                         'max_tokens': 256 if old else 32})
    traces[name] = trace(requests, indexes, '64-token old prefixes; complete late texts; burst')
stream = [{'id': records[i]['question_id'], 'dataset_index': i, 'group': 'requests',
           'arrival_s': 0.0, 'prompt_token_ids': tokens[i], 'max_tokens': 128}
          for i in stream_ids]
bank = trace(stream, stream_ids, 'Complete SPEED-Bench texts; source bank, NOT final arrivals')
all_requests = [r for t in traces.values() for r in t['requests']] + stream
max_len = max(len(r['prompt_token_ids']) + r['max_tokens'] - 1 for r in all_requests)
assert max_len <= hf['max_position_embeddings'] and max_len <= blocks * 128
required = {name: sum(math.ceil((len(r['prompt_token_ids']) + r['max_tokens'] - 1) / 128)
                     for r in t['requests']) for name, t in traces.items()}
assert max(required.values()) < blocks, 'Burst does not have sufficient fixed KV capacity'
common = {'model': model, 'dataset': None, 'source': None, 'profile': 'burst',
          'policies': ['separate', 'interleaved', 'mixed'], 'rounds': 1,
          'warmup': int(os.environ.get('EXPERIMENT_WARMUP', '1')),
          'seed': 114514, 'device': 'cuda:0', 'dtype': 'bfloat16', 'max_seqs': 16,
          'token_budget': 512, 'block_size': 128, 'blocks': blocks,
          'max_model_len': max_len, 'gpu_memory_utilization': 0.8,
          'prefix_cache': False, 'ignore_eos': True, 'cpu_threads': 1,
          'timeout_s': float(os.environ.get('EXPERIMENT_TIMEOUT', '3600')),
          'window_s': float(os.environ.get('EXPERIMENT_WINDOW', '5')), 'trim_fraction': 0.0,
          'save_details': False, 'diagnostics': False}
plan.mkdir(parents=True)
def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
for name, t in traces.items():
    validate_trace(t, vocab_size=hf['vocab_size'], max_model_len=max_len)
    trace_path = plan / f'trace_{name}.json'
    write(trace_path, t)
    for budget in (128, 256, 384, 512):
        path = plan / f'{name}_T{budget}.json'
        write(path, dict(common, trace=str(trace_path), token_budget=budget, out=None))
        preflight(parse_options(['--config', str(path)]))
validate_trace(bank, vocab_size=hf['vocab_size'], max_model_len=max_len)
write(plan / 'stream_bank.json', bank)
write(plan / 'manifest.json', {'model': model, **identity, 'source': str(source),
      'source_file_sha256': source_hash, 'seed': 20261010, 'length_band': [1900, 2200],
      'delay_s': delay, 'blocks': blocks, 'max_model_len': max_len,
      'required_burst_blocks': required,
      'sets': {k: [provenance(i) for i in v] for k, v in sets.items()},
      'stream': [provenance(i) for i in stream_ids], 'common': common})
print('PLAN', plan)
print('required_burst_blocks', required, 'max_model_len', max_len)
print('Explore late lengths', [len(r['prompt_token_ids']) for r in traces['explore']['requests'][4:]])
