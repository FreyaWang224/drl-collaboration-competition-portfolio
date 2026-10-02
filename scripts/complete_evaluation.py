"""Complete this run's evaluation in parallel without restarting seed 11.

Operational handoff only: data, checkpoints and protocol remain unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--coordinator-pid', type=int, required=True)
parser.add_argument('--existing-evaluator-pid', type=int, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
protocol = json.loads((root/'PROTOCOL.json').read_text())
for seed in protocol['training_seeds']:
    if not (root/'runs'/f'baseline_seed{seed}'/'summary.json').exists():
        raise RuntimeError('All training must be complete')
command = subprocess.check_output(['pgrep', '-fl', 'run_protocol.py'], text=True)
matching = [line for line in command.splitlines() if line.split()[0] == str(args.coordinator_pid)
            and line.endswith('scripts/run_protocol.py')]
if len(matching) != 1:
    raise RuntimeError('Coordinator identity mismatch')
os.kill(args.existing_evaluator_pid, 0)
for seed in [22, 33]:
    if (root/'runs'/f'baseline_eval_seed{seed}').exists():
        raise RuntimeError('Do not duplicate an existing evaluation')
os.kill(args.coordinator_pid, signal.SIGTERM)  # only coordinator, not evaluator
(root/'runs'/'orchestration_handoff.json').write_text(json.dumps(dict(
    reason='Parallelize evaluation of the three models; unchanged checkpoints, seeds and budgets',
    stopped_coordinator=args.coordinator_pid, preserved_seed11_evaluator=args.existing_evaluator_pid), indent=2))
env = dict(os.environ, PYTHONPATH=str(root/'src')+':'+str(root/'environments/legacy-python'))
children = []
for seed in [22, 33]:
    train = root/'runs'/f'baseline_seed{seed}'
    summary = json.loads((train/'summary.json').read_text())
    checkpoint = train/('first_solved.pt' if summary['first_solved'] is not None else 'final.pt')
    log = (root/'runs'/f'baseline_eval_seed{seed}.log').open('x')
    command = [str(root/'.venv/bin/python'), '-m', 'tennis.cli', 'evaluate',
               '--environment', str(root/'environments/Tennis.app'),
               '--output', str(root/'runs'/f'baseline_eval_seed{seed}'),
               '--checkpoint', str(checkpoint), '--worker-id', str(seed+100),
               '--max-steps', str(protocol['external_episode_step_cap']), '--evaluation-seeds']
    command += [str(s) for s in protocol['evaluation_seeds']]
    children.append((seed, subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT), log))
results = []
for seed, process, log in children:
    code = process.wait()
    log.close()
    results.append(dict(seed=seed, training_exit_code=0, evaluation_exit_code=code))
    print(json.dumps(results[-1]), flush=True)
while not (root/'runs'/'baseline_eval_seed11'/'summary.json').exists():
    try:
        os.kill(args.existing_evaluator_pid, 0)
    except ProcessLookupError:
        raise RuntimeError('Preserved seed11 evaluation exited without summary')
    time.sleep(5)
results.append(dict(seed=11, training_exit_code=0, evaluation_exit_code=0))
(root/'runs'/'protocol_status.json').write_text(json.dumps(sorted(results,key=lambda x:x['seed']),indent=2))
