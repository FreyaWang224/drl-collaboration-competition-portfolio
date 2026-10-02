"""Run the predeclared baseline protocol; retain every seed and failure.

Separate workers avoid legacy class-level communicator reuse. This runner
never retries or replaces failed training seeds and never selects on eval.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
protocol = json.loads((root / 'PROTOCOL.json').read_text())
env = dict(os.environ, PYTHONPATH=str(root/'src') + ':' + str(root/'environments/legacy-python'))
python = str(root / '.venv/bin/python')
binary = str(root / 'environments/Tennis.app')
processes = []
for seed in protocol['training_seeds']:
    output = root / 'runs' / f'baseline_seed{seed}'
    output.parent.mkdir(exist_ok=True)
    log = (output.parent / f'baseline_seed{seed}.log').open('x')
    cmd = [python, '-m', 'tennis.cli', 'train', '--environment', binary, '--output', str(output),
           '--seed', str(seed), '--worker-id', str(seed), '--episodes', str(protocol['max_episodes']),
           '--max-steps', str(protocol['external_episode_step_cap']),
           '--max-environment-steps', str(protocol['max_environment_steps']),
           '--warmup-steps', str(protocol['config']['warmup_steps'])]
    process = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    processes.append((seed, process, log, output))
results = []
for seed, process, log, output in processes:
    code = process.wait()
    log.close()
    result = dict(seed=seed, training_exit_code=code)
    if code == 0:
        summary = json.loads((output / 'summary.json').read_text())
        checkpoint = output / ('first_solved.pt' if summary['first_solved'] is not None else 'final.pt')
        evaluation = root / 'runs' / f'baseline_eval_seed{seed}'
        with (root/'runs'/f'baseline_eval_seed{seed}.log').open('x') as evaluation_log:
            cmd = [python, '-m', 'tennis.cli', 'evaluate', '--environment', binary,
                   '--output', str(evaluation), '--checkpoint', str(checkpoint), '--worker-id', str(seed+100),
                   '--max-steps', str(protocol['external_episode_step_cap']), '--evaluation-seeds']
            cmd += [str(x) for x in protocol['evaluation_seeds']]
            result['evaluation_exit_code'] = subprocess.run(cmd, cwd=root, env=env, stdout=evaluation_log,
                                                            stderr=subprocess.STDOUT).returncode
    results.append(result)
    (root/'runs'/'protocol_status.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(result), flush=True)
