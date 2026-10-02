"""Resume infrastructure-interrupted evaluation, retaining completed records.

No checkpoint changes, score-based retries or evaluation-seed replacements.
Incomplete attempt directories are preserved for audit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--training-seed', type=int, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
protocol = json.loads((root/'PROTOCOL.json').read_text())
seed = args.training_seed
output = root/'runs'/f'baseline_eval_seed{seed}'
manifest = json.loads((output/'manifest.json').read_text())
checkpoint = Path(manifest['args']['checkpoint'])
if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != manifest['checkpoint_sha256']:
    raise RuntimeError('Checkpoint changed')
# Only resume after the old evaluator has exited. Do not duplicate live work.
found = subprocess.run(['pgrep', '-fl', f'tennis.cli evaluate.*baseline_eval_seed{seed}'], capture_output=True,text=True)
if found.returncode == 0:
    raise RuntimeError('Old evaluator still running')
env = dict(os.environ, PYTHONPATH=str(root/'src')+':'+str(root/'environments/legacy-python'))
records = []
interruptions = []
for index, evaluation_seed in enumerate(protocol['evaluation_seeds'], 1):
    parent_record = output/f'episode_{index:04d}.json'
    child = output/f'seed_{evaluation_seed}'
    if parent_record.exists():
        record = json.loads(parent_record.read_text())
        if record['evaluation_seed'] != evaluation_seed:
            raise RuntimeError('Existing record seed mismatch')
    else:
        if child.exists() and not (child/'summary.json').exists():
            saved = output/f'interrupted_seed_{evaluation_seed}'
            child.rename(saved)
            interruptions.append(dict(seed=evaluation_seed, saved_attempt=str(saved)))
        if not (child/'summary.json').exists():
            cmd = [sys.executable, '-m', 'tennis.cli', 'evaluate', '--environment', manifest['args']['environment'],
                   '--output', str(child), '--checkpoint', str(checkpoint), '--evaluation-seeds', str(evaluation_seed),
                   '--worker-id', str(manifest['args']['worker_id']), '--max-steps', str(manifest['args']['max_steps'])]
            subprocess.run(cmd, cwd=root, env=env, check=True)
        record = json.loads((child/'episode_0001.json').read_text())
        parent_record.write_text(json.dumps(record, indent=2)+'\n')
    records.append(record)
    print(f'restored/complete evaluation {index}/30', flush=True)
complete = all(r['complete'] for r in records)
child_summary = json.loads((output/f'seed_{protocol["evaluation_seeds"][0]}'/'summary.json').read_text())
summary = dict(training_seed=seed, checkpoint_metadata=child_summary['checkpoint_metadata'],
               requested_episodes=len(records), truncated_episodes=sum(r['truncated'] for r in records),
               mean_score=float(np.mean([r['score'] for r in records])) if complete else None,
               within_model_population_sd=float(np.std([r['score'] for r in records])) if complete else None,
               all_complete=complete, learning_rounds_during_evaluation=0,
               evaluation_resume='Infrastructure interruption only; completed episodes retained',
               interruptions=interruptions,
               elapsed_seconds=sum(json.loads((output/f'seed_{s}'/'summary.json').read_text())['elapsed_seconds']
                                   for s in protocol['evaluation_seeds']),
               elapsed_definition='sum of completed child run times; excludes interrupted attempt and parent overhead')
(output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
