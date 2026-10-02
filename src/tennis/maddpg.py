"""MADDPG with local actors and agent-specific centralized critics.

Critics condition on joint observations/actions. Actor i replaces only its
own replay action; other replay actions stay fixed, matching the original
MADDPG policy-gradient form. No policy ensemble or learned partner model.
"""
import torch
from torch import nn
from .core import IndependentDDPG, bellman_target, soft_update


class MADDPG(IndependentDDPG):
    algorithm = 'maddpg'

    @staticmethod
    def critic_dimensions(observation_size):
        return 2 * observation_size, 4

    def actor_loss(self, batch, i):
        # Preserve the gradient only through the agent currently being updated.
        actions = [self.actors[j](batch['obs'][:, j]) if j == i
                   else batch['actions'][:, j].detach() for j in range(2)]
        return -self.critics[i](batch['obs'].flatten(start_dim=1),
                                torch.cat(actions, dim=-1)).mean()

    def critic_targets(self, batch):
        with torch.no_grad():
            next_actions = torch.cat([self.target_actors[j](batch['next_obs'][:, j])
                                      for j in range(2)], dim=-1)
            next_obs = batch['next_obs'].flatten(start_dim=1)
            return [bellman_target(batch['rewards'][:, i:i+1], batch['terminals'][:, i:i+1],
                                   self.target_critics[i](next_obs, next_actions), self.config.gamma)
                    for i in range(2)]

    def learn(self, batch):
        # Compute both targets before any online/target update for symmetry.
        targets = self.critic_targets(batch)
        observations = batch['obs'].flatten(start_dim=1)
        actions = batch['actions'].flatten(start_dim=1)
        diagnostics = []
        for i in range(2):
            critic = self.critics[i]
            q = critic(observations, actions)
            critic_loss = nn.functional.mse_loss(q, targets[i])
            if not torch.isfinite(critic_loss):
                raise FloatingPointError('Nonfinite centralized critic loss')
            self.critic_optimizers[i].zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0, error_if_nonfinite=True)
            self.critic_optimizers[i].step()
            self.critic_optimizers[i].zero_grad(set_to_none=True)
            critic.requires_grad_(False)
            try:
                # Clear both actor gradients so partner isolation is observable.
                for optimizer in self.actor_optimizers:
                    optimizer.zero_grad(set_to_none=True)
                actor_loss = self.actor_loss(batch, i)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 1.0, error_if_nonfinite=True)
                self.actor_optimizers[i].step()
            finally:
                critic.requires_grad_(True)
            soft_update(self.target_actors[i], self.actors[i], self.config.tau)
            soft_update(self.target_critics[i], critic, self.config.tau)
            diagnostics.append(dict(agent=i, critic_loss=critic_loss.item(),
                                    actor_loss=actor_loss.item(), q_mean=q.detach().mean().item()))
        self.learning_rounds += 1
        return diagnostics
