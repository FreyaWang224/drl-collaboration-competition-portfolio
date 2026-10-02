import copy
from types import SimpleNamespace as NS
import numpy as np
import pytest
import torch
from tennis.core import Config, IndependentDDPG, Replay, Actor, Critic, bellman_target, soft_update
from tennis.environment import TennisEnvironment
from tennis.metrics import ScoreWindow


@pytest.fixture
def agent():
    torch.set_num_threads(1)
    return IndependentDDPG(Config(3, hidden=16, batch_size=4, capacity=20, warmup_steps=1), 7)


def fill(agent):
    rng = np.random.default_rng(1)
    for _ in range(6):
        agent.replay.add(rng.normal(size=(2, 3)), rng.uniform(-1, 1, (2, 2)),
                         [0.1, -0.01], rng.normal(size=(2, 3)), [0, 1])


def test_actor_shape_bounds(agent):
    out = agent.act(np.ones((2, 3)), explore=True)
    assert out.shape == (2, 2) and np.max(np.abs(out)) <= 1
    assert torch.all(torch.abs(agent.actors[0](torch.randn(8, 3))) <= 1)


def test_separate_agents(agent):
    assert next(agent.actors[0].parameters()).data_ptr() != next(agent.actors[1].parameters()).data_ptr()
    assert not torch.equal(next(agent.actors[0].parameters()), next(agent.actors[1].parameters()))


def test_target_initialization(agent):
    for online, target in zip(agent.actors + agent.critics, agent.target_actors + agent.target_critics):
        assert all(torch.equal(a, b) for a, b in zip(online.parameters(), target.parameters()))
        assert all(not p.requires_grad for p in target.parameters())


def test_bellman_terminal_and_bootstrap():
    y = bellman_target(torch.tensor([[0.1], [0.1]]), torch.tensor([[0.], [1.]]), torch.full((2, 1), 0.8), 0.99)
    assert y[:, 0].tolist() == pytest.approx([0.892, 0.1])


def test_bellman_reject_broadcast():
    with pytest.raises(ValueError):
        bellman_target(torch.zeros(4), torch.zeros(4, 1), torch.zeros(4, 1), 0.99)


def test_soft_update():
    a, b = Actor(3, 8), Actor(3, 8)
    with torch.no_grad():
        for p in a.parameters(): p.fill_(0)
        for p in b.parameters(): p.fill_(2)
    soft_update(a, b, 0.25)
    assert all(torch.allclose(p, torch.full_like(p, 0.5)) for p in a.parameters())


def test_joint_alignment_and_ring():
    r = Replay(5, 3, 0)
    for t in range(8):
        r.add(np.full((2, 3), t), np.zeros((2, 2)), [t, -t], np.full((2, 3), t+1), [0, 1])
    b = r.sample(5)
    assert set(b['rewards'][:, 0].tolist()) == {3, 4, 5, 6, 7}
    assert torch.equal(b['obs'][:, 0, 0], b['rewards'][:, 0])
    assert torch.equal(b['next_obs'][:, 1, 0], b['rewards'][:, 0] + 1)


@pytest.mark.parametrize('bad', ['bounds', 'nan', 'shape', 'terminal'])
def test_replay_validation_atomic(bad):
    r = Replay(5, 3, 0)
    o, a, rew, term = np.zeros((2, 3)), np.zeros((2, 2)), np.zeros(2), np.zeros(2)
    if bad == 'bounds': a[0, 0] = 2
    if bad == 'nan': rew[0] = np.nan
    if bad == 'shape': o = np.zeros((3, 2))
    if bad == 'terminal': term[0] = 0.5
    with pytest.raises(ValueError): r.add(o, a, rew, o, term)
    assert r.size == 0


def test_actual_actor_gradient_with_frozen_critic(agent):
    critic = agent.critics[0]
    critic.requires_grad_(False)
    loss = -critic(torch.randn(4, 3), agent.actors[0](torch.randn(4, 3))).mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.actors[0].parameters())
    assert all(p.grad is None for p in critic.parameters())
    assert all(p.grad is None for p in agent.actors[1].parameters())


def test_learning_changes_online_and_not_target_grad(agent):
    fill(agent)
    before = [p.clone() for p in agent.actors[0].parameters()]
    diagnostics = agent.learn(agent.replay.sample(4))
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.actors[0].parameters()))
    assert len(diagnostics) == 2 and agent.learning_rounds == 1
    assert all(p.grad is None for n in agent.target_actors + agent.target_critics for p in n.parameters())
    assert all(p.grad is None for n in agent.critics for p in n.parameters())


def test_checkpoint_inference_and_resume(agent, tmp_path):
    fill(agent)
    agent.learn(agent.replay.sample(4))
    obs = np.ones((2, 3), np.float32)
    p = tmp_path / 'inference.pt'
    agent.save(p, metadata={'selection': 'test'})
    restored, metadata = IndependentDDPG.load(p)
    np.testing.assert_array_equal(agent.act(obs), restored.act(obs))
    assert metadata['selection'] == 'test'
    with pytest.raises(ValueError): IndependentDDPG.load(p, resume=True)
    p = tmp_path / 'full.pt'
    agent.save(p, metadata={}, full=True)
    expected = agent.act(obs, explore=True)
    expected_batch = agent.replay.sample(4)
    restored, _ = IndependentDDPG.load(p, resume=True)
    np.testing.assert_array_equal(expected, restored.act(obs, explore=True))
    actual_batch = restored.replay.sample(4)
    for k in expected_batch: assert torch.equal(expected_batch[k], actual_batch[k])
    a = agent.learn(expected_batch)
    b = restored.learn(actual_batch)
    assert a == b


def test_score_sum_then_max_and_truncation_gap():
    w = ScoreWindow(3)
    assert w.add([0.2, 0.3], True) == (0.3, None)
    w.add([0.8, 0.2], True)
    score, mean = w.add([0.1, 0.4], True)
    assert mean == pytest.approx(0.5)
    assert w.add([99, 99], False) == (99, None)
    assert not w.values


class Fake:
    def __init__(self, **kwargs):
        self.brain_names = ['TennisBrain']
        self.brains = {'TennisBrain': NS(vector_action_space_size=2, vector_action_space_type='continuous',
                                      vector_observation_space_size=8, num_stacked_vector_observations=3)}
        self.closed = False
    def info(self, ids, vals):
        return NS(agents=ids, vector_observations=np.repeat(np.array(vals)[:, None], 24, axis=1),
                  rewards=vals, local_done=[False, False])
    def reset(self, **kwargs): return {'TennisBrain': self.info([10, 20], [1, 2])}
    def step(self, actions):
        self.actions = actions
        return {'TennisBrain': self.info([20, 10], [2, 1])}
    def close(self): self.closed = True


def test_environment_bidirectional_identity_mapping():
    with TennisEnvironment('unused', factory=Fake) as e:
        e.reset()
        s = e.step([[0.1, 0.2], [0.3, 0.4]])
        np.testing.assert_array_equal(s.rewards, [1, 2])
        e.step([[0.1, 0.2], [0.3, 0.4]])
        np.testing.assert_allclose(e.env.actions, [[0.3, 0.4], [0.1, 0.2]])
        assert s.agent_ids == (10, 20) and e.observation_size == 24
    assert e.env.closed


def test_environment_reject_changed_ids_and_bad_actions():
    with TennisEnvironment('unused', factory=Fake) as e:
        e.reset()
        with pytest.raises(ValueError): e.step(np.full((2, 2), 2))
        with pytest.raises(ValueError): e.decode(e.env.info([10, 30], [1, 2]))


def test_evaluation_episode_no_learning(agent):
    from tennis.cli import episode
    class Env:
        def reset(self, training):
            assert not training
            return NS(observations=np.ones((2, 3)), agent_ids=(0, 1))
        def step(self, actions):
            return NS(observations=np.ones((2, 3)), rewards=np.array([0.1, 0.2]), legacy_done=np.ones(2, bool))
    actors_before = copy.deepcopy([a.state_dict() for a in agent.actors])
    r = episode(Env(), agent, False, 10)
    assert r['complete'] and r['score'] == 0.2
    assert agent.replay.size == agent.learning_rounds == agent.environment_steps == 0
    for a, old in zip(agent.actors, actors_before):
        assert all(torch.equal(v, old[k]) for k, v in a.state_dict().items())


def test_episode_score_not_stepwise_max(agent):
    from tennis.cli import episode
    class Env:
        def reset(self, training):
            self.t = 0
            return NS(observations=np.zeros((2, 3)), agent_ids=(0, 1))
        def step(self, actions):
            self.t += 1
            return NS(observations=np.zeros((2, 3)), rewards=np.array([1, 0] if self.t == 1 else [0, 1]),
                      legacy_done=np.array([self.t == 2]*2))
    r = episode(Env(), agent, False, 10)
    assert r['agent_returns'] == [1, 1] and r['score'] == 1  # stepwise max would be 2


def test_external_cap_does_not_become_terminal(agent):
    from tennis.cli import episode
    class Env:
        def reset(self, training):
            return NS(observations=np.zeros((2, 3)), agent_ids=(0, 1))
        def step(self, actions):
            return NS(observations=np.zeros((2, 3)), rewards=np.ones(2), legacy_done=np.zeros(2, bool))
    r = episode(Env(), agent, True, 2)
    assert r['truncated'] and not r['complete']
    assert not agent.replay.data['terminals'][:2].any()


def test_reset_order_preserves_policy_roles():
    with TennisEnvironment('unused', factory=Fake) as e:
        e.reset()
        result = e.decode(e.env.info([20, 10], [2, 1]), reset=True)
        assert result.agent_ids == (10, 20)
        np.testing.assert_array_equal(result.rewards, [1, 2])
