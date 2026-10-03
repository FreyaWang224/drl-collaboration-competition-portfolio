"""Frozen actor-only cross-play with canonical roles; no critic or optimizer."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import subprocess
import numpy as np
import torch
from .core import Actor, Config
from .cli import episode, write_json, binary_manifest
from .environment import TennisEnvironment


class CrossPlayPolicy:
    def __init__(self, paths, seeds):
        if len(paths) != 2 or len(seeds) != 2:
            raise ValueError('Exactly two role-specific sources required')
        self.actors = []
        self.actor_training_seeds = list(seeds)
        self.hashes = []
        self.config = None
        for role, (path, seed) in enumerate(zip(paths, seeds)):
            state = torch.load(path, map_location='cpu', weights_only=False)
            if state.get('algorithm') != 'maddpg' or state.get('format_version') != 1:
                raise ValueError('Expected trusted MADDPG checkpoint')
            if state['seed'] != seed or len(state['actors']) != 2:
                raise ValueError('Source seed or actor count mismatch')
            config = Config(**state['config'])
            if self.config is not None and asdict(config) != asdict(self.config):
                raise ValueError('Incompatible source configurations')
            self.config = config
            actor = Actor(config.observation_size, config.hidden)
            actor.load_state_dict(state['actors'][role])
            actor.eval().requires_grad_(False)
            self.actors.append(actor)
            self.hashes.append(hashlib.sha256(Path(path).read_bytes()).hexdigest())

    @torch.no_grad()
    def act(self, observations, *, explore=False, random_action=False):
        if explore or random_action:
            raise ValueError('Cross-play evaluation must be deterministic')
        observations = np.asarray(observations, dtype=np.float32)
        if observations.shape != (2, self.config.observation_size) or not np.isfinite(observations).all():
            raise ValueError('Invalid canonical observation shape')
        return np.stack([actor(torch.from_numpy(observations[role])).numpy()
                         for role, actor in enumerate(self.actors)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoints', nargs=2, required=True)
    parser.add_argument('--actor-seeds', type=int, nargs=2, required=True)
    parser.add_argument('--evaluation-seed', type=int, required=True)
    parser.add_argument('--worker-id', type=int, required=True)
    parser.add_argument('--max-steps', type=int, default=10000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    started = time.monotonic()
    policy = CrossPlayPolicy(args.checkpoints, args.actor_seeds)
    before = [[p.clone() for p in actor.parameters()] for actor in policy.actors]
    manifest = dict(kind='exploratory_cross_play', actor_training_seeds=args.actor_seeds,
                    actor_roles=[0, 1], source_checkpoint_sha256=policy.hashes,
                    evaluation_seed=args.evaluation_seed, exploration=False, learning=False,
                    config=asdict(policy.config), binaries=binary_manifest(args.environment),
                    source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    external_step_cap=args.max_steps,
                    legacy_done_limit='Raw local_done treated as terminal; internal timeout ambiguous')
    write_json(args.output/'manifest.json', manifest)
    with TennisEnvironment(args.environment, args.evaluation_seed, args.worker_id, True) as env:
        if env.observation_size != policy.config.observation_size:
            raise ValueError('Environment/checkpoint observation mismatch')
        record = episode(env, policy, False, args.max_steps)
    if any(not torch.equal(p,q) for old,actor in zip(before,policy.actors)
           for p,q in zip(old,actor.parameters())):
        raise AssertionError('Evaluation changed actor parameters')
    record.update(actor_training_seeds=args.actor_seeds, evaluation_seed=args.evaluation_seed)
    write_json(args.output/'episode.json', record)
    write_json(args.output/'summary.json', dict(complete=record['complete'],
               optimizer_updates=0, actor_parameters_unchanged=True,
               elapsed_seconds=time.monotonic()-started))

if __name__ == '__main__':
    main()
