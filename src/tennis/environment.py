"""Legacy Unity bridge, preserving agent identity and raw local_done.

The API does not label internal timeout vs termination. Baseline treats raw
local_done as terminal (legacy convention), records this limitation, and keeps
our external step caps separate from terminal masks.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class Step:
    observations: np.ndarray
    rewards: np.ndarray
    legacy_done: np.ndarray
    agent_ids: tuple


class TennisEnvironment:
    def __init__(self, path, seed=0, worker_id=0, no_graphics=True, factory=None):
        self.closed = False
        self.ids = self.raw_ids = None
        self.needs_reset = True
        if factory is None:
            if not Path(path).exists():
                raise FileNotFoundError(path)
            if not hasattr(np, 'float_'):
                np.float_ = np.float64  # legacy API compatibility with NumPy 2
            from unityagents import UnityEnvironment
            factory = UnityEnvironment
        self.env = factory(file_name=str(path), seed=seed, worker_id=worker_id,
                           no_graphics=no_graphics)
        try:
            if len(self.env.brain_names) != 1:
                raise ValueError('Expected one brain')
            self.brain = self.env.brain_names[0]
            info = self.env.brains[self.brain]
            if int(info.vector_action_space_size) != 2 or info.vector_action_space_type != 'continuous':
                raise ValueError('Expected 2 continuous actions')
            self.base_observation_size = int(info.vector_observation_space_size)
            self.stacks = int(info.num_stacked_vector_observations)
            self.observation_size = self.base_observation_size * self.stacks
            if self.base_observation_size != 8:
                raise ValueError('Not the expected Udacity Tennis observation specification')
        except Exception:
            self.close()
            raise

    def decode(self, info, reset=False):
        raw = tuple(info.agents)
        if len(raw) != 2 or len(set(raw)) != 2:
            raise ValueError('Expected two distinct agents')
        if reset and self.ids is None:
            self.ids = raw
        if set(raw) != set(self.ids):
            raise ValueError('Agent identity changed within episode')
        order = [raw.index(i) for i in self.ids]
        obs = np.asarray(info.vector_observations, np.float32)
        rewards = np.asarray(info.rewards, np.float32)
        done = np.asarray(info.local_done, bool)
        if obs.shape != (2, self.observation_size) or rewards.shape != (2,) or done.shape != (2,):
            raise ValueError('Invalid environment shapes')
        if not np.isfinite(obs).all() or not np.isfinite(rewards).all():
            raise ValueError('Nonfinite environment output')
        self.raw_ids = raw
        self.needs_reset = bool(done.any())
        return Step(obs[order].copy(), rewards[order].copy(), done[order].copy(), self.ids)

    def reset(self, training=True):
        if self.closed:
            raise RuntimeError('Closed environment')
        return self.decode(self.env.reset(train_mode=training)[self.brain], reset=True)

    def step(self, actions):
        if self.closed or self.needs_reset:
            raise RuntimeError('Reset required or closed environment')
        actions = np.asarray(actions, np.float32)
        if actions.shape != (2, 2) or not np.isfinite(actions).all() or np.any(np.abs(actions) > 1):
            raise ValueError('Actions must be finite [2,2] within [-1,1]')
        order = [self.ids.index(i) for i in self.raw_ids]
        return self.decode(self.env.step(actions[order].copy())[self.brain])

    def close(self):
        if not self.closed:
            self.closed = True
            self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
