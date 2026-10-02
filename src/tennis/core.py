"""Networks, synchronized replay, and independent DDPG updates.

Two agents have separate actors, local critics and optimizers. Sharing joint
replay storage does not make this MADDPG. All computation defaults to CPU.
"""
from dataclasses import asdict, dataclass
import random
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    observation_size: int
    hidden: int = 128
    batch_size: int = 128
    capacity: int = 100000
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 0.0001
    critic_lr: float = 0.001
    noise_std: float = 0.2
    warmup_steps: int = 1000
    learn_every: int = 1
    updates_per_event: int = 1

    def __post_init__(self):
        for name in ('observation_size', 'hidden', 'batch_size', 'capacity',
                     'warmup_steps', 'learn_every', 'updates_per_event'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if self.capacity < self.batch_size or not 0 <= self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError('Invalid replay size, gamma or tau')
        if min(self.actor_lr, self.critic_lr) <= 0 or self.noise_std < 0:
            raise ValueError('Invalid learning rate or noise')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Actor(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h),
                                 nn.ReLU(), nn.Linear(h, 2), nn.Tanh())
        nn.init.uniform_(self.net[-2].weight, -0.003, 0.003)
        nn.init.uniform_(self.net[-2].bias, -0.003, 0.003)

    def forward(self, obs):
        return self.net(obs)


class Critic(nn.Module):
    def __init__(self, d, h, action_size=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + action_size, h), nn.ReLU(), nn.Linear(h, h),
                                 nn.ReLU(), nn.Linear(h, 1))
        nn.init.uniform_(self.net[-1].weight, -0.003, 0.003)
        nn.init.uniform_(self.net[-1].bias, -0.003, 0.003)

    def forward(self, obs, actions):
        return self.net(torch.cat((obs, actions), dim=-1))


@torch.no_grad()
def soft_update(target, online, tau):
    for dest, source in zip(target.parameters(), online.parameters()):
        dest.lerp_(source, tau)


def bellman_target(rewards, terminals, next_q, gamma):
    if rewards.shape != terminals.shape or rewards.shape != next_q.shape or rewards.ndim != 2 or rewards.shape[1] != 1:
        raise ValueError('Bellman inputs must all have shape [B,1]')
    return rewards + gamma * (1 - terminals) * next_q


class Replay:
    """Ring buffer: each row contains both agents from one environment step."""
    def __init__(self, capacity, d, seed):
        self.capacity, self.d = capacity, d
        self.rng = np.random.default_rng(seed)
        self.size = self.cursor = 0
        self.data = {
            'obs': np.empty((capacity, 2, d), np.float32),
            'actions': np.empty((capacity, 2, 2), np.float32),
            'rewards': np.empty((capacity, 2), np.float32),
            'next_obs': np.empty((capacity, 2, d), np.float32),
            'terminals': np.empty((capacity, 2), np.float32),
        }

    def add(self, obs, actions, rewards, next_obs, terminals):
        values = dict(obs=obs, actions=actions, rewards=rewards,
                      next_obs=next_obs, terminals=terminals)
        checked = {}
        for key, value in values.items():
            value = np.asarray(value, np.float32)
            if value.shape != self.data[key].shape[1:] or not np.isfinite(value).all():
                raise ValueError(f'Invalid {key}')
            checked[key] = value
        if np.any(np.abs(checked['actions']) > 1) or not np.isin(checked['terminals'], [0, 1]).all():
            raise ValueError('Invalid action bounds or terminal mask')
        for key, value in checked.items():
            self.data[key][self.cursor] = value
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n):
        if n > self.size:
            raise ValueError('Insufficient replay')
        indices = self.rng.choice(self.size, n, replace=False)
        return {k: torch.from_numpy(v[indices].copy()) for k, v in self.data.items()}

    def state(self):
        return dict(size=self.size, cursor=self.cursor, rng=self.rng.bit_generator.state,
                    data={k: v[:self.size].copy() for k, v in self.data.items()})

    def restore(self, state):
        self.size, self.cursor = state['size'], state['cursor']
        for k, v in state['data'].items():
            self.data[k][:self.size] = v
        self.rng.bit_generator.state = state['rng']


class IndependentDDPG:
    algorithm = 'independent_ddpg'

    @staticmethod
    def critic_dimensions(observation_size):
        return observation_size, 2

    def __init__(self, config, seed):
        self.config, self.seed = config, seed
        seed_all(seed)
        self.rng = np.random.default_rng(seed)
        self.actors = [Actor(config.observation_size, config.hidden) for _ in range(2)]
        critic_d, critic_a = self.critic_dimensions(config.observation_size)
        self.critics = [Critic(critic_d, config.hidden, critic_a) for _ in range(2)]
        self.target_actors = [Actor(config.observation_size, config.hidden) for _ in range(2)]
        self.target_critics = [Critic(critic_d, config.hidden, critic_a) for _ in range(2)]
        for targets, online in [(self.target_actors, self.actors), (self.target_critics, self.critics)]:
            for target, source in zip(targets, online):
                target.load_state_dict(source.state_dict())
                target.requires_grad_(False)
        self.actor_optimizers = [torch.optim.Adam(a.parameters(), lr=config.actor_lr) for a in self.actors]
        self.critic_optimizers = [torch.optim.Adam(q.parameters(), lr=config.critic_lr) for q in self.critics]
        self.replay = Replay(config.capacity, config.observation_size, seed)
        self.environment_steps = self.learning_rounds = 0

    def act(self, obs, *, explore=False, random_action=False):
        obs = np.asarray(obs, np.float32)
        if obs.shape != (2, self.config.observation_size) or not np.isfinite(obs).all():
            raise ValueError('Invalid actor observation')
        if random_action:
            if not explore:
                raise ValueError('Random actions are training-only')
            return self.rng.uniform(-1, 1, (2, 2)).astype(np.float32)
        with torch.no_grad():
            actions = np.stack([a(torch.from_numpy(obs[i])).numpy() for i, a in enumerate(self.actors)])
        if explore:
            actions += self.rng.normal(0, self.config.noise_std, actions.shape)
        return np.clip(actions, -1, 1).astype(np.float32)

    def observe(self, obs, actions, rewards, next_obs, terminals):
        self.replay.add(obs, actions, rewards, next_obs, terminals)
        self.environment_steps += 1
        c = self.config
        if self.environment_steps < c.warmup_steps or self.replay.size < c.batch_size or self.environment_steps % c.learn_every:
            return []
        return [self.learn(self.replay.sample(c.batch_size)) for _ in range(c.updates_per_event)]

    def learn(self, batch):
        diagnostics = []
        for i in range(2):
            obs, actions = batch['obs'][:, i], batch['actions'][:, i]
            with torch.no_grad():
                next_actions = self.target_actors[i](batch['next_obs'][:, i])
                next_q = self.target_critics[i](batch['next_obs'][:, i], next_actions)
                target = bellman_target(batch['rewards'][:, i:i+1],
                                        batch['terminals'][:, i:i+1], next_q, self.config.gamma)
            q = self.critics[i](obs, actions)
            critic_loss = nn.functional.mse_loss(q, target)
            if not torch.isfinite(critic_loss):
                raise FloatingPointError('Nonfinite critic loss')
            self.critic_optimizers[i].zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[i].parameters(), 1.0, error_if_nonfinite=True)
            self.critic_optimizers[i].step()
            self.critic_optimizers[i].zero_grad(set_to_none=True)
            self.critics[i].requires_grad_(False)
            try:
                actor_loss = -self.critics[i](obs, self.actors[i](obs)).mean()
                self.actor_optimizers[i].zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 1.0, error_if_nonfinite=True)
                self.actor_optimizers[i].step()
            finally:
                self.critics[i].requires_grad_(True)
            soft_update(self.target_actors[i], self.actors[i], self.config.tau)
            soft_update(self.target_critics[i], self.critics[i], self.config.tau)
            diagnostics.append(dict(agent=i, critic_loss=critic_loss.item(),
                                    actor_loss=actor_loss.item(), q_mean=q.detach().mean().item()))
        self.learning_rounds += 1
        return diagnostics

    def save(self, path, *, metadata, full=False):
        state = dict(format_version=1, algorithm=self.algorithm, config=asdict(self.config),
                     seed=self.seed, metadata=metadata, full=full,
                     environment_steps=self.environment_steps, learning_rounds=self.learning_rounds,
                     actors=[a.state_dict() for a in self.actors])
        if full:
            state.update(critics=[q.state_dict() for q in self.critics],
                         target_actors=[a.state_dict() for a in self.target_actors],
                         target_critics=[q.state_dict() for q in self.target_critics],
                         actor_optimizers=[o.state_dict() for o in self.actor_optimizers],
                         critic_optimizers=[o.state_dict() for o in self.critic_optimizers],
                         replay=self.replay.state(), rng=self.rng.bit_generator.state,
                         torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                         python_rng=random.getstate())
        from pathlib import Path
        temporary = Path(str(path) + '.tmp')
        torch.save(state, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, *, resume=False):
        # Only load trusted local files; full checkpoints contain Python objects.
        state = torch.load(path, map_location='cpu', weights_only=False)
        if state.get('format_version') != 1:
            raise ValueError('Unsupported checkpoint version')
        if state.get('algorithm') != cls.algorithm:
            if cls is IndependentDDPG and state.get('algorithm') == 'maddpg':
                from .maddpg import MADDPG
                return MADDPG.load(path, resume=resume)
            raise ValueError('Unsupported checkpoint algorithm')
        agent = cls(Config(**state['config']), state['seed'])
        for model, weights in zip(agent.actors, state['actors']):
            model.load_state_dict(weights)
        agent.environment_steps = state['environment_steps']
        agent.learning_rounds = state['learning_rounds']
        if resume:
            if not state['full']:
                raise ValueError('Inference checkpoint cannot resume learning')
            for name in ('critics', 'target_actors', 'target_critics', 'actor_optimizers', 'critic_optimizers'):
                for obj, weights in zip(getattr(agent, name), state[name]):
                    obj.load_state_dict(weights)
            agent.replay.restore(state['replay'])
            agent.rng.bit_generator.state = state['rng']
            torch.set_rng_state(state['torch_rng'])
            np.random.set_state(state['numpy_rng'])
            random.setstate(state['python_rng'])
        return agent, state['metadata']
