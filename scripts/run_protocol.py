"""Run the predeclared baseline protocol; retain every seed and failure.

Separate workers avoid legacy class-level communicator reuse. This runner
never retries or replaces failed training seeds and never selects on eval.
"""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--protocol', type=Path, default=root/'PROTOCOL.json')
args = parser.parse_args()
protocol = json.loads(args.protocol.read_text())
algorithm = protocol['algorithm']
prefix = 'baseline' if algorithm == 'independent_ddpg' else 'maddpg'
worker_offset = 0 if prefix == 'baseline' else 50
env = dict(os.environ, PYTHONPATH=str(root/'src') + ':' + str(root/'environments/legacy-python'))
python = str(root / '.venv/bin/python')
binary = str(root / 'environments/Tennis.app')
processes = []
for seed in protocol['training_seeds']:
    output = root / 'runs' / f'{prefix}_seed{seed}'
    output.parent.mkdir(exist_ok=True)
    log = (output.parent / f'{prefix}_seed{seed}.log').open('x')
    cmd = [python, '-m', 'tennis.cli', 'train', '--environment', binary, '--output', str(output),
           '--algorithm', algorithm, '--seed', str(seed), '--worker-id', str(seed+worker_offset), '--episodes', str(protocol['max_episodes']),
           '--max-steps', str(protocol['external_episode_step_cap']),
           '--max-environment-steps', str(protocol['max_environment_steps']),
           '--warmup-steps', str(protocol['config']['warmup_steps'])]
    process = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    processes.append((seed, process, log, output))
results = []
ready = []
for seed, process, log, output in processes:
    code = process.wait()
    log.close()
    result = dict(seed=seed, training_exit_code=code)
    results.append(result)
    if code == 0:
        ready.append((seed, output, result))
    (root/'runs'/f'{prefix}_protocol_status.json').write_text(json.dumps(results, indent=2))
evaluation_processes = []
for seed, output, result in ready:
    summary = json.loads((output / 'summary.json').read_text())
    if {k: summary['config'][k] for k in protocol['config']} != protocol['config']:
        raise RuntimeError('Training configuration differs from frozen protocol')
    checkpoint = output / ('first_solved.pt' if summary['first_solved'] is not None else 'final.pt')
    evaluation = root / 'runs' / f'{prefix}_eval_seed{seed}'
    log = (root/'runs'/f'{prefix}_eval_seed{seed}.log').open('x')
    cmd = [python, '-m', 'tennis.cli', 'evaluate', '--environment', binary,
           '--output', str(evaluation), '--checkpoint', str(checkpoint), '--worker-id', str(seed+worker_offset+100),
           '--max-steps', str(protocol['external_episode_step_cap']), '--evaluation-seeds']
    cmd += [str(x) for x in protocol['evaluation_seeds']]
    process = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    evaluation_processes.append((process, log, result))
for process, log, result in evaluation_processes:
    result['evaluation_exit_code'] = process.wait()
    log.close()
    (root/'runs'/f'{prefix}_protocol_status.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(result), flush=True)
