"""Audit, train and independently evaluate. Output directories never overwrite."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import torch
from .core import Config, IndependentDDPG
from .environment import TennisEnvironment
from .metrics import ScoreWindow


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def episode(env, agent, training, cap, trace=False):
    step = env.reset(training=training)
    returns = np.zeros(2, np.float64)
    clipped = 0
    diagnostics = []
    trajectory = []
    for t in range(1, cap + 1):
        actions = agent.act(step.observations, explore=training,
                            random_action=training and agent.environment_steps < agent.config.warmup_steps)
        nxt = env.step(actions)
        if training:
            updates = agent.observe(step.observations, actions, nxt.rewards,
                                    nxt.observations, nxt.legacy_done)
            if updates:
                diagnostics = updates[-1]
        returns += nxt.rewards
        clipped += int((np.abs(actions) >= 1).sum())
        if trace:
            trajectory.append(dict(step=t, agent_ids=list(step.agent_ids),
                                   observations=step.observations.tolist(), actions=actions.tolist(),
                                   rewards=nxt.rewards.tolist(), next_observations=nxt.observations.tolist(),
                                   legacy_done=nxt.legacy_done.tolist()))
        step = nxt
        if step.legacy_done.any():
            break
    complete = bool(step.legacy_done.any())
    record = dict(agent_returns=returns.tolist(), score=float(returns.max()),
                  steps=t, complete=complete, truncated=not complete,
                  legacy_done=step.legacy_done.tolist(),
                  action_boundary_fraction=clipped / (t * 4), diagnostics=diagnostics)
    if trace:
        record['trajectory'] = trajectory
    return record


def binary_manifest(path):
    p = Path(path)
    candidates = sorted((p / 'Contents' / 'MacOS').glob('*')) if p.is_dir() else [p]
    result = []
    for f in candidates:
        if f.is_file():
            sha = hashlib.sha256()
            with f.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    sha.update(chunk)
            result.append(dict(path=str(f.resolve()), sha256=sha.hexdigest()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['smoke', 'train', 'evaluate'])
    parser.add_argument('--environment', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--worker-id', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=10000)
    parser.add_argument('--max-environment-steps', type=int, default=1000000)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--evaluation-seeds', type=int, nargs='+')
    parser.add_argument('--graphics', action='store_true')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--warmup-steps', type=int, default=1000)
    parser.add_argument('--noise-std', type=float, default=0.2)
    args = parser.parse_args()
    if min(args.episodes, args.max_steps, args.max_environment_steps) <= 0:
        parser.error('Budgets must be positive')
    if args.mode == 'evaluate' and (args.checkpoint is None or not args.evaluation_seeds):
        parser.error('Evaluation requires a checkpoint and explicit evaluation seeds')
    if args.mode == 'train' and args.checkpoint:
        parser.error('CLI resume is not supported: environment state is not restored')
    if args.mode == 'evaluate' and len(set(args.evaluation_seeds)) != len(args.evaluation_seeds):
        parser.error('Evaluation seeds must be distinct')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.monotonic()
    manifest = dict(mode=args.mode, args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    python=platform.python_version(), platform=platform.platform(), architecture=platform.machine(),
                    torch=torch.__version__, numpy=np.__version__, binaries=binary_manifest(args.environment),
                    algorithm='independent_ddpg',
                    source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
                    terminal_policy='Raw legacy local_done treated as terminal; internal timeout not distinguishable',
                    observation_roles='reset row order defines policy roles; intra-episode rows aligned by IDs')
    write_json(args.output / 'manifest.json', manifest)
    records = []
    if args.mode == 'evaluate':
        agent, metadata = IndependentDDPG.load(args.checkpoint)
        before = agent.learning_rounds
        for seed in args.evaluation_seeds:
            if len(args.evaluation_seeds) > 1:
                # unityagents 0.4 has class-level Pipe handles: closing the first
                # environment breaks reinitialization in the same interpreter.
                # Fresh processes preserve unmodified legacy API and seed isolation.
                child = args.output / f'seed_{seed}'
                command = [sys.executable, '-m', 'tennis.cli', 'evaluate',
                           '--environment', args.environment, '--output', str(child),
                           '--checkpoint', str(args.checkpoint), '--evaluation-seeds', str(seed),
                           '--worker-id', str(args.worker_id), '--max-steps', str(args.max_steps)]
                if args.graphics:
                    command.append('--graphics')
                subprocess.run(command, check=True)
                record = json.loads((child / 'episode_0001.json').read_text())
            else:
                with TennisEnvironment(args.environment, seed, args.worker_id, not args.graphics) as env:
                    if env.observation_size != agent.config.observation_size:
                        raise ValueError('Checkpoint/environment dimensions disagree')
                    record = episode(env, agent, False, args.max_steps)
                    record.update(evaluation_seed=seed, training_seed=agent.seed)
            records.append(record)
            write_json(args.output / f'episode_{len(records):04d}.json', record)
            print(f'evaluation {len(records)} score={record["score"]:.3f} complete={record["complete"]}', flush=True)
        if agent.learning_rounds != before:
            raise AssertionError('Evaluation modified training counter')
        complete = all(r['complete'] for r in records)
        summary = dict(training_seed=agent.seed, checkpoint_metadata=metadata,
                       requested_episodes=len(args.evaluation_seeds), truncated_episodes=sum(r['truncated'] for r in records),
                       mean_score=float(np.mean([r['score'] for r in records])) if complete else None,
                       within_model_population_sd=float(np.std([r['score'] for r in records])) if complete else None,
                       all_complete=complete, learning_rounds_during_evaluation=agent.learning_rounds-before)
    else:
        with TennisEnvironment(args.environment, args.seed, args.worker_id, not args.graphics) as env:
            manifest.update(base_observation_size=env.base_observation_size, stacks=env.stacks,
                            observation_size=env.observation_size, action_size=2)
            write_json(args.output / 'manifest.json', manifest)
            config = Config(env.observation_size, hidden=args.hidden, batch_size=args.batch_size,
                            warmup_steps=args.warmup_steps, noise_std=args.noise_std)
            agent = IndependentDDPG(config, args.seed)
            window = ScoreWindow()
            first_solved = None
            interaction_steps = 0
            for e in range(1, args.episodes + 1):
                remaining = args.max_environment_steps - interaction_steps
                if remaining <= 0:
                    break
                if args.mode == 'smoke':
                    # Random policy audit: never learn or claim algorithm performance.
                    original = agent.act
                    agent.act = lambda obs, **kw: agent.rng.uniform(-1, 1, (2, 2)).astype(np.float32)
                record = episode(env, agent, args.mode == 'train', min(args.max_steps, remaining),
                                 trace=args.mode == 'smoke')
                if args.mode == 'smoke':
                    agent.act = original
                interaction_steps += record['steps']
                score, rolling = window.add(record['agent_returns'], record['complete'])
                record.update(episode=e, rolling_100=rolling, environment_steps=interaction_steps,
                              agent_transitions=interaction_steps*2, learning_rounds=agent.learning_rounds,
                              optimizer_steps=agent.learning_rounds*4, elapsed_seconds=time.monotonic()-started)
                records.append(record)
                write_json(args.output / f'episode_{e:04d}.json', record)
                print(f'episode {e} score={score:.3f} rolling100={rolling} complete={record["complete"]}', flush=True)
                if args.mode == 'train' and first_solved is None and rolling is not None and rolling >= 0.5:
                    first_solved = e
                    agent.save(args.output / 'first_solved.pt', metadata=dict(selection='first_complete_rolling100_ge_0.5', episode=e, rolling100=rolling))
            if args.mode == 'train':
                agent.save(args.output / 'final.pt', metadata=dict(selection='final', episode=len(records), first_solved=first_solved))
                agent.save(args.output / 'training_state.pt', metadata=dict(episode=len(records), simulator_state_restored=False), full=True)
            summary = dict(config=asdict(config), seed=args.seed, episodes=len(records),
                           first_solved=first_solved, environment_steps=interaction_steps,
                           learning_rounds=agent.learning_rounds, truncated_episodes=sum(r['truncated'] for r in records),
                           random_policy_audit=args.mode == 'smoke')
    summary['elapsed_seconds'] = time.monotonic()-started
    write_json(args.output / 'summary.json', summary)


if __name__ == '__main__':
    main()
