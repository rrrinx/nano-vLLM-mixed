"""CPU/stdlib contracts for block selection, resume validation and paired statistics.

Can also run directly with Python (unittest), without loading model dependencies.
"""
import copy
import csv
import json
import math
import os
import statistics
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
import analyze
import block_design
import run_matrix
import upgrade_blocks
import launch


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class ExperimentBlocksTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / 'artifacts/experiment_blocks_tests'
        parent.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        app = self.root
        (app / 'experiments/online').mkdir(parents=True)
        (app / 'experiments/online/run_online.py').write_text('# fixture runner\n')
        (app / 'experiments/benchmark.py').write_text('# fixture source\n')
        self.sources = run_matrix.source_identity(app)
        self.model = self.root / 'model'
        self.model.mkdir()
        (self.model / 'weights.safetensors').write_bytes(b'fixture, not a checkpoint')
        self.trace = {'requests':[{'id':'r','max_tokens':2}], 'workload_sha256':'fixture'}
        trace_path = self.root / 'trace.json'
        write(trace_path, self.trace)
        self.configs=[]
        for budget in (384,512):
            path=self.root/f'case_T{budget}.json'
            write(path, {'model':str(self.model), 'trace':str(trace_path),'out':None,
                         'policies':block_design.POLICIES,'warmup':1,'rounds':1,
                         'save_details':False,'diagnostics':False,'ignore_eos':True,
                         'source':None,'token_budget':budget,'max_seqs':16})
            self.configs.append(path)

    def payload(self, entry):
        config=run_matrix.read(Path(entry['config']))
        runs=[]
        for position, policy in enumerate(entry['policy_order'],1):
            throughput=10+entry['block']*.1
            if policy=='mixed':
                throughput+=(entry['block']+1)*.2
            runs.append({'policy':policy,'policy_position':position,
                         'summary':{'output_tokens':2,'elapsed_s':2/throughput,
                                    'output_tok_s':throughput,'counts':{'preemptions':0}},
                         'requests':[{'id':'r','output_tokens':2,'ttft_ms':1,'tpot_ms':1,'e2e_ms':2}],
                         'online':{'overlap':{}}})
        return {'status':'complete','settings':dict(config,out=entry['out'],policy_orders=','.join(entry['policy_order'])),
                'trace':self.trace,'source_sha256':self.sources,'runs':runs,
                'environment':{'torch':'fixture','gpu':'fixture'},
                'checkpoint_files':run_matrix.checkpoint_identity(self.model)}

    def test_block_ids_preserve_orders_and_balance(self):
        self.assertEqual(block_design.parse_block_ids(block_design.DEFAULT_BLOCK_IDS),list(range(6)))
        self.assertTrue(block_design.position_balanced([0,3,4]))
        self.assertTrue(block_design.position_balanced([1,2,5]))
        self.assertFalse(block_design.position_balanced([0,1,2]))
        for bad in ('','0,0','-1','6','0,x','0,'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                block_design.parse_block_ids(bad)
        schedule=run_matrix.make_schedule(self.configs,self.root/'out',[0,3,4])
        self.assertEqual(len(schedule),6)
        self.assertEqual([e['block'] for e in schedule],[0,0,3,3,4,4])
        self.assertEqual(schedule[2]['policy_order'],['interleaved','mixed','separate'])
        self.assertIn('block4',schedule[-1]['out'])

    def test_reuse_rejects_incompatible_or_partial_results(self):
        entry=run_matrix.make_schedule(self.configs,self.root/'out',[0])[0]
        config=run_matrix.read(Path(entry['config']))
        original=self.payload(entry)
        run_matrix.check_result(original,entry,config,self.sources,'stream',original)
        def alter_settings(d): d['settings']['warmup']=2
        def alter_trace(d): d['trace']['workload_sha256']='changed'
        def alter_source(d): d['source_sha256']={}
        def alter_environment(d): d['environment']['gpu']='different'
        def alter_checkpoint(d): d['checkpoint_files']={}
        def alter_order(d): d['runs'].reverse()
        def partial(d): d['runs'].pop()
        for change in (alter_settings,alter_trace,alter_source,alter_environment,alter_checkpoint,alter_order,partial):
            candidate=copy.deepcopy(original)
            change(candidate)
            with self.subTest(change=change.__name__),self.assertRaises(ValueError):
                run_matrix.check_result(candidate,entry,config,self.sources,'stream',original)

    def test_copy_complete_calls_and_run_only_missing_with_provenance(self):
        old=self.root/'old/formal'
        old_schedule=run_matrix.make_schedule(self.configs,old,list(range(6)))
        write(old/'schedule.json',{'entries':old_schedule,'blocks':6})
        old_result=Path(old_schedule[0]['out'])/'results.json'
        write(old_result,self.payload(old_schedule[0]))
        old_hash=run_matrix.sha256(old_result)
        # Incomplete calls must be rerun, never included as partial policy samples.
        write(Path(old_schedule[6]['out'])/'failed.json',{'completed_runs':[]})
        reference=self.root/'pilot/results.json'
        write(reference,self.payload(old_schedule[0]))
        executed=[]
        def fake_run(command,**kwargs):
            if '--dry-run' in command:
                return
            config=Path(command[command.index('--config')+1])
            order=command[command.index('--policy-orders')+1].split(',')
            block=block_design.ORDERS.index(tuple(order))
            entry={'config':str(config),'out':command[command.index('--out')+1],
                   'block':block,'policy_order':order}
            executed.append((block,config.stem))
            write(Path(entry['out'])/'results.json',self.payload(entry))
        out=self.root/'new/formal'
        prior=Path.cwd()
        try:
            os.chdir(self.root)
            with patch.object(run_matrix.subprocess,'run',side_effect=fake_run):
                run_matrix.main(['--configs',*map(str,self.configs),'--out',str(out),'--block-ids','0,3,4',
                                 '--reuse-from',str(old.parent),'--reference-result',str(reference),
                                 '--mode','stream','--execute'])
        finally:
            os.chdir(prior)
        self.assertEqual(len(executed),5)
        self.assertNotIn((0,'case_T384'),executed)
        self.assertEqual(run_matrix.sha256(old_result),old_hash)
        self.assertEqual(run_matrix.sha256(out/'block0/case_T384/results.json'),old_hash)
        self.assertEqual(len(run_matrix.read(out/'reuse.json')['entries']),1)
        self.assertEqual(run_matrix.read(out/'schedule.json')['block_ids'],[0,3,4])
        rows=analyze.analyze(out,'stream')
        self.assertEqual(len(rows),18)
        self.assertIn('Reused completed calls: 1',(out/'analysis.md').read_text())

    def test_analysis_uses_actual_sample_size_and_refuses_incomplete_schedule(self):
        for ids in ([0],[0,3,4],list(range(6))):
            with self.subTest(ids=ids):
                out=self.root/f'analysis_{len(ids)}'
                schedule=run_matrix.make_schedule(self.configs[:1],out,ids)
                write(out/'schedule.json',{'entries':schedule,'blocks':len(ids),'block_ids':ids})
                write(out/'blocks.json',{'entries':schedule})
                for entry in schedule:
                    write(Path(entry['out'])/'results.json',self.payload(entry))
                analyze.analyze(out,'stream')
                with (out/'comparisons.csv').open(encoding='utf-8-sig',newline='') as f:
                    rows=list(csv.DictReader(f))
                r=next(r for r in rows if r['baseline']=='interleaved' and r['metric']=='throughput_tok_s')
                self.assertEqual(int(r['n']),len(ids))
                if len(ids)==1:
                    self.assertEqual(r['ci95_low'],'')
                    self.assertIn('n/a (one block)',(out/'analysis.md').read_text())
                else:
                    diffs=[(b+1)*.2 for b in ids]
                    critical=4.302652729911275 if len(ids)==3 else 2.570581835636305
                    half=critical*statistics.stdev(diffs)/math.sqrt(len(diffs))
                    self.assertAlmostEqual(float(r['ci95_high']),statistics.mean(diffs)+half)
                write(out/'blocks.json',{'entries':schedule[:-1]})
                with self.assertRaisesRegex(ValueError,'Not all planned'):
                    analyze.analyze(out,'stream')

    def test_upgrade_clones_immutable_files_without_touching_old_results(self):
        source=self.root/'package'
        original=source/'experiments/tool.py'
        original.parent.mkdir(parents=True)
        original.write_text('old\n')
        write(source/'PACKAGE_MANIFEST.json',{'sha256':{'experiments/tool.py':run_matrix.sha256(original)}})
        write(source/'runs/live.json',{'running':True})
        update=self.root/'update'
        replacement=update/'payload/experiments/tool.py'
        replacement.parent.mkdir(parents=True)
        replacement.write_text('new\n')
        write(update/'UPDATE_MANIFEST.json',{'sha256':{'experiments/tool.py':run_matrix.sha256(replacement)}})
        target=self.root/'package_new'
        upgrade_blocks.upgrade(source,target,update)
        self.assertEqual(original.read_text(),'old\n')
        self.assertEqual((target/'experiments/tool.py').read_text(),'new\n')
        self.assertTrue((source/'runs/live.json').exists())
        self.assertFalse((target/'runs').exists())
        with self.assertRaises(ValueError):
            upgrade_blocks.upgrade(source,target,update)
        with self.assertRaises(ValueError):
            upgrade_blocks.checked_files(source,{str(original.resolve()):run_matrix.sha256(original)})

    def test_launcher_forwards_selection_and_reuse_for_confirm_and_online(self):
        write(self.model/'config.json',{})
        write(self.model/'tokenizer_config.json',{})
        plan=self.root/'plan'
        base=run_matrix.read(self.configs[1])
        write(plan/'manifest.json',{'model':str(self.model),'delay_s':1,'blocks':565,
                                    'common':{'warmup':1,'window_s':5,'timeout_s':3600}})
        write(plan/'explore_T512.json',base)
        write(plan/'stream_bank.json',self.trace)
        fake_modules={'experiments.benchmark':SimpleNamespace(digest=lambda requests:'fixture'),
                      'experiments.online.online_workload':SimpleNamespace(validate_trace=lambda trace:None)}
        for stage in ('burst','online'):
            out=self.root/f'launch_{stage}'
            argv=['launch.py','--model',str(self.model),'--stage',stage,'--plan',str(plan),
                  '--budgets','384,512','--block-ids','0,3,4','--reuse-from',str(self.root/'old'),
                  '--out',str(out),'--dry-run','--no-install']
            if stage=='online':argv+=['--rates','0.25,0.5,1.0','--arrival-seeds','20261011']
            prior=Path.cwd()
            try:
                with patch.object(sys,'argv',argv), patch.object(launch,'ROOT',self.root), \
                        patch.object(launch,'EXPERIMENTS',self.root/'experiments'), \
                        patch.object(launch,'APP',self.root), \
                        patch.object(launch,'verify'),patch.object(launch,'dependencies'), \
                        patch.object(launch,'environment',return_value={}) as environment, \
                        patch.object(launch,'run') as run,patch.dict(sys.modules,fake_modules):
                    launch.main()
                    command=list(map(str,run.call_args.args[0]))
                    self.assertEqual(command[command.index('--block-ids')+1],'0,3,4')
                    self.assertEqual(command[command.index('--reuse-from')+1],str(self.root/'old'))
                    self.assertNotIn('--execute',command)
                    self.assertFalse(environment.call_args.kwargs['gpu'])
            finally:
                os.chdir(prior)
            self.assertEqual(run_matrix.read(out/'stage.json')['block_ids'],[0,3,4])
            if stage=='online':
                self.assertEqual(len(list((out/'configs').glob('E_*.json'))),6)


if __name__=='__main__':
    unittest.main()
