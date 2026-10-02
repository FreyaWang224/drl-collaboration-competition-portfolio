"""Validate frozen protocol evidence, export models and aggregate real evals."""
import hashlib
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--protocol', type=Path, default=root/'PROTOCOL.json')
args = parser.parse_args()
p = json.loads(args.protocol.read_text())
prefix = 'baseline' if p['algorithm'] == 'independent_ddpg' else 'maddpg'
rows = []
for seed in p['training_seeds']:
    training = root/'runs'/f'{prefix}_seed{seed}'
    evaluation = root/'runs'/f'{prefix}_eval_seed{seed}'
    summary = json.loads((training/'summary.json').read_text())
    evaluation_summary = json.loads((evaluation/'summary.json').read_text())
    if summary.get('algorithm', 'independent_ddpg') != p['algorithm']:
        raise ValueError('Training algorithm mismatch')
    if {k: summary['config'][k] for k in p['config']} != p['config']:
        raise ValueError('Run differs from frozen hyperparameters')
    records = [json.loads(f.read_text()) for f in sorted(training.glob('episode_*.json'))]
    window = []
    first = None
    for r in records:
        if not np.isclose(r['score'], max(r['agent_returns']), atol=1e-9):
            raise ValueError('Invalid episode score')
        if r['complete']:
            window.append(r['score'])
            window = window[-100:]
        else:
            window = []
        rolling = float(np.mean(window)) if len(window) == 100 else None
        if (rolling is None) != (r['rolling_100'] is None) or (rolling is not None and not np.isclose(rolling, r['rolling_100'], atol=1e-9)):
            raise ValueError('Rolling-100 mismatch')
        if first is None and rolling is not None and rolling >= .5:
            first = r['episode']
    if first != summary['first_solved']:
        raise ValueError('First-solved selection mismatch')
    selected = training/('first_solved.pt' if first is not None else 'final.pt')
    manifest = json.loads((evaluation/'manifest.json').read_text())
    if manifest['algorithm'] != p['algorithm']:
        raise ValueError('Evaluation algorithm mismatch')
    if manifest['checkpoint_sha256'] != hashlib.sha256(selected.read_bytes()).hexdigest():
        raise ValueError('Wrong evaluated checkpoint')
    eval_records = [json.loads(f.read_text()) for f in sorted(evaluation.glob('episode_*.json'))]
    if [r['evaluation_seed'] for r in eval_records] != p['evaluation_seeds']:
        raise ValueError('Evaluation seed mismatch')
    if evaluation_summary['learning_rounds_during_evaluation'] != 0:
        raise ValueError('Evaluation learned')
    for evaluation_seed in p['evaluation_seeds']:
        child = evaluation/f'seed_{evaluation_seed}'
        child_summary = json.loads((child/'summary.json').read_text())
        child_manifest = json.loads((child/'manifest.json').read_text())
        if child_summary['learning_rounds_during_evaluation'] != 0:
            raise ValueError('Child evaluation learned')
        if child_manifest['checkpoint_sha256'] != manifest['checkpoint_sha256']:
            raise ValueError('Child evaluated a different checkpoint')
    scores = [r['score'] for r in eval_records]
    if evaluation_summary['all_complete']:
        if not np.isclose(np.mean(scores), evaluation_summary['mean_score'], atol=1e-9):
            raise ValueError('Evaluation mean mismatch')
    for run in [f'{prefix}_seed{seed}', f'{prefix}_eval_seed{seed}']:
        subprocess.run([sys.executable, str(root/'scripts/export_results.py'), run, '--kind', 'formal'], check=True)
    weights = root/'artifacts'/'models'
    weights.mkdir(exist_ok=True)
    shutil.copyfile(selected, weights/f'{prefix}_seed{seed}.pt')
    rows.append(dict(seed=seed, first_solved_episode=first,
                     first_solved_environment_step=records[first-1]['environment_steps'] if first is not None else None,
                     complete_training_episodes=sum(r['complete'] for r in records),
                     training_truncations=summary['truncated_episodes'],
                     environment_steps=summary['environment_steps'],
                     training_final_rolling100=records[-1]['rolling_100'],
                     training_highest_rolling100=max(r['rolling_100'] or 0 for r in records),
                     training_single_episode_max=max(r['score'] for r in records if r['complete']),
                     evaluation_episodes=len(eval_records),
                     evaluation_truncations=evaluation_summary['truncated_episodes'],
                     evaluation_mean=evaluation_summary['mean_score'],
                     evaluation_within_model_population_sd=evaluation_summary['within_model_population_sd'],
                     learning_rounds=summary['learning_rounds'], training_elapsed_seconds=summary['elapsed_seconds']))
means = [r['evaluation_mean'] for r in rows]
result = dict(protocol=p, seeds=rows, across_training_seed_evaluation_mean=float(np.mean(means)) if None not in means else None,
              across_training_seed_population_sd=float(np.std(means)) if None not in means else None,
              inference='Descriptive population SD of three model means, not CI or significance')
(root/'artifacts'/f'{prefix}_results.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
