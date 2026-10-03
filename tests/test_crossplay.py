import numpy as np
import pytest
import torch
from tennis.core import Config
from tennis.maddpg import MADDPG
from tennis.crossplay import CrossPlayPolicy


def sources(tmp_path):
    paths=[]
    for seed,values in [(11,(.1,.2)),(22,(.3,.4))]:
        agent=MADDPG(Config(3,hidden=8,batch_size=2,capacity=4),seed)
        with torch.no_grad():
            for actor,value in zip(agent.actors,values):
                for p in actor.parameters(): p.zero_()
                actor.net[-2].bias.fill_(value)
        path=tmp_path/f'{seed}.pt'
        agent.save(path,metadata={})
        paths.append(path)
    return paths


def test_role_correct_mixed_actors_without_training_objects(tmp_path):
    policy=CrossPlayPolicy(sources(tmp_path),[11,22])
    np.testing.assert_allclose(policy.act(np.zeros((2,3))),
                               [[np.tanh(.1)]*2,[np.tanh(.4)]*2],rtol=1e-6)
    assert not hasattr(policy,'critics') and not hasattr(policy,'actor_optimizers')
    assert all(not p.requires_grad for a in policy.actors for p in a.parameters())


def test_reverse_pair_is_distinct_role_combination(tmp_path):
    paths=sources(tmp_path)
    policy=CrossPlayPolicy(paths[::-1],[22,11])
    np.testing.assert_allclose(policy.act(np.zeros((2,3))),
                               [[np.tanh(.3)]*2,[np.tanh(.2)]*2],rtol=1e-6)


def test_diagonal_preserves_original_actor_outputs(tmp_path):
    path=sources(tmp_path)[0]
    policy=CrossPlayPolicy([path,path],[11,11])
    original,_=MADDPG.load(path)
    obs=np.ones((2,3),np.float32)
    np.testing.assert_array_equal(policy.act(obs),original.act(obs))
    with pytest.raises(ValueError):policy.act(obs,explore=True)
    with pytest.raises(ValueError):policy.act(obs,random_action=True)


def test_source_identity_and_shapes_rejected(tmp_path):
    paths=sources(tmp_path)
    with pytest.raises(ValueError):CrossPlayPolicy(paths,[22,11])
    policy=CrossPlayPolicy(paths,[11,22])
    with pytest.raises(ValueError):policy.act(np.zeros((3,2)))
