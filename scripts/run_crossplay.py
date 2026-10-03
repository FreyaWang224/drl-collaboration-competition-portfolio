"""Run frozen exploratory cross-play; retain failures and never retry by score."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
p=json.loads((root/'PROTOCOL_CROSSPLAY.json').read_text())
for source in p['sources'].values():
    for file_key,hash_key in [('checkpoint','sha256'),('diagonal_records','diagonal_sha256')]:
        if hashlib.sha256((root/source[file_key]).read_bytes()).hexdigest()!=source[hash_key]:
            raise ValueError('Frozen source changed')
python=str(root/'.venv/bin/python')
env=dict(os.environ,PYTHONPATH=str(root/'src')+':'+str(root/'environments/legacy-python'))
base=root/'runs/crossplay'
base.mkdir(exist_ok=False)
status=[]

def run_pair(index,a,b):
    pair=base/f'{a}_{b}'
    pair.mkdir()
    with (pair/'run.log').open('x') as log:
        for seed in p['evaluation_seeds']:
            cmd=[python,'-m','tennis.crossplay','--environment',str(root/'environments/Tennis.app'),
                 '--output',str(pair/f'seed_{seed}'),'--checkpoints',
                 str(root/p['sources'][str(a)]['checkpoint']),str(root/p['sources'][str(b)]['checkpoint']),
                 '--actor-seeds',str(a),str(b),'--evaluation-seed',str(seed),
                 '--worker-id',str(500+index),'--max-steps',str(p['external_episode_step_cap'])]
            code=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            if code!=0:return dict(actor_seeds=[a,b],exit_code=code,failed_evaluation_seed=seed)
    return dict(actor_seeds=[a,b],exit_code=0)

pairs=[(a,b) for a in p['training_seeds'] for b in p['training_seeds'] if a!=b]
with ThreadPoolExecutor(max_workers=p['parallel_pairs']) as pool:
    futures=[pool.submit(run_pair,index,a,b) for index,(a,b) in enumerate(pairs)]
    for future in as_completed(futures):
        result=future.result();status.append(result)
        temporary=base/'status.tmp'
        temporary.write_text(json.dumps(status,indent=2))
        temporary.replace(base/'status.json')
        print(json.dumps(result),flush=True)
if any(r['exit_code']!=0 for r in status):
    raise RuntimeError('Cross-play failed; retained records, no retry or replacement')
subprocess.run([python,str(root/'scripts/summarize_crossplay.py')],cwd=root,check=True)
paths=['CROSSPLAY_REPORT.md','README.md','artifacts/crossplay_results.json',
       'artifacts/crossplay_matrix.png','artifacts/crossplay_matrix.svg','artifacts/crossplay']
subprocess.run(['git','add',*paths],cwd=root,check=True)
subprocess.run(['git','commit','-m','Report real exploratory role-preserving MADDPG cross-play'],cwd=root,check=True)
subprocess.run(['git','push','origin','main'],cwd=root,check=True)
(base/'complete.json').write_text(json.dumps(dict(validated=True,published=True),indent=2))
print('Cross-play actual results validated and published.',flush=True)
