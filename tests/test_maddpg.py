import numpy as np
import pytest
import torch
from torch import nn
from tennis.core import Config, IndependentDDPG
from tennis.maddpg import MADDPG


@pytest.fixture
def agent():
    torch.set_num_threads(1)
    return MADDPG(Config(3, hidden=16, batch_size=4, capacity=20, warmup_steps=1), 7)


def batch(agent):
    rng = np.random.default_rng(17)
    for _ in range(6):
        agent.replay.add(rng.normal(size=(2,3)), rng.uniform(-1,1,(2,2)),
                         [0.1,-0.01],rng.normal(size=(2,3)),[0,1])
    return agent.replay.sample(4)


def test_local_actors_central_critics(agent):
    assert agent.actors[0].net[0].in_features == 3
    assert agent.critics[0].net[0].in_features == 10  # 2*3 + 2*2
    b = batch(agent)
    assert agent.critics[0](b['obs'].flatten(1),b['actions'].flatten(1)).shape == (4,1)
    with pytest.raises(RuntimeError):
        agent.critics[0](b['obs'][:,0],b['actions'][:,0])


class ConstantActor(nn.Module):
    def __init__(self,value):
        super().__init__()
        self.value=value
    def forward(self,obs):
        return torch.full((len(obs),2),self.value)


class SumActions(nn.Module):
    def __init__(self,scale):
        super().__init__()
        self.scale=scale
    def forward(self,obs,actions):
        assert obs.shape[1]==6 and actions.shape[1]==4
        return self.scale*actions.sum(dim=1,keepdim=True)


def test_joint_target_uses_both_target_actors_and_own_reward(agent):
    agent.target_actors=[ConstantActor(.2),ConstantActor(.4)]
    agent.target_critics=[SumActions(1),SumActions(2)]
    b=dict(next_obs=torch.zeros(2,2,3),rewards=torch.tensor([[.1,-.01],[.2,.1]]),
           terminals=torch.tensor([[0.,0.],[1.,1.]]))
    y=agent.critic_targets(b)
    assert y[0][:,0].tolist()==pytest.approx([1.288,.2])
    assert y[1][:,0].tolist()==pytest.approx([2.366,.1])
    assert all(not v.requires_grad for v in y)


@pytest.mark.parametrize('i',[0,1])
def test_actor_update_gradient_isolation(agent,i):
    b=batch(agent)
    agent.critics[i].requires_grad_(False)
    agent.actor_loss(b,i).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in agent.actors[i].parameters())
    assert all(p.grad is None for p in agent.actors[1-i].parameters())
    assert all(p.grad is None for p in agent.critics[i].parameters())


def test_actor_loss_preserves_partner_replay_actions(agent):
    b=batch(agent)
    seen=[]
    hook=agent.critics[0].register_forward_pre_hook(lambda model,inputs: seen.append(inputs[1].detach().clone()))
    agent.actor_loss(b,0)
    hook.remove()
    assert torch.equal(seen[0][:,2:],b['actions'][:,1])
    assert torch.allclose(seen[0][:,:2],agent.actors[0](b['obs'][:,0]))


def test_learn_updates_both_agents_without_target_grad(agent):
    b=batch(agent)
    before=[[p.clone() for p in a.parameters()] for a in agent.actors]
    result=agent.learn(b)
    for i in range(2):
        assert any(not torch.equal(p,q) for p,q in zip(before[i],agent.actors[i].parameters()))
    assert len(result)==2 and agent.learning_rounds==1
    assert all(p.grad is None for n in agent.target_actors+agent.target_critics for p in n.parameters())
    assert all(p.grad is None for n in agent.critics for p in n.parameters())


def test_checkpoint_dispatch_and_full_continuation(agent,tmp_path):
    b=batch(agent)
    agent.learn(b)
    path=tmp_path/'model.pt'
    agent.save(path,metadata={'algorithm':'maddpg'},full=True)
    expected_batch=agent.replay.sample(4)
    restored,meta=IndependentDDPG.load(path,resume=True)
    assert isinstance(restored,MADDPG) and meta['algorithm']=='maddpg'
    actual_batch=restored.replay.sample(4)
    for k in expected_batch:assert torch.equal(expected_batch[k],actual_batch[k])
    assert agent.learn(expected_batch)==restored.learn(actual_batch)
    np.testing.assert_array_equal(agent.act(np.ones((2,3))),restored.act(np.ones((2,3))))


def test_baseline_loader_rejects_algorithm_confusion(tmp_path):
    baseline=IndependentDDPG(Config(3,hidden=16,batch_size=4,capacity=20),0)
    path=tmp_path/'baseline.pt'
    baseline.save(path,metadata={})
    with pytest.raises(ValueError):MADDPG.load(path)
    loaded,_=IndependentDDPG.load(path)
    assert type(loaded) is IndependentDDPG
