# Implementation reading route

Read modules in this order when returning to the project. Exercises and grading stay in chat.

1. `environment.py`: distinguish raw Unity row order from stable policy role order. `reset` and `step` return canonical observations, while actions are translated back into Unity order. The stack count is read from the actual brain.
2. `metrics.py`: accumulate agent returns in the episode loop, then take the maximum once. A truncated episode clears the consecutive qualifying window.
3. `core.py:Replay`: every stored row aligns both agents from the same environment step. Sampling indexes are shared across all fields. Executed actions are stored.
4. `core.py:IndependentDDPG.act`: uniform warm-up, then deterministic actors plus clipped Gaussian exploration; evaluation uses deterministic output.
5. `core.py:IndependentDDPG.learn`: take local slices for each agent; target branches use no_grad. Critic MSE updates the critic. Actor loss freezes critic parameters but retains the derivative with respect to the actor action. Targets receive soft updates.
6. `core.py:save/load`: paired actors for inference; full Python training state for continuation. Unity state is not serialized. CLI resume is deliberately unavailable.
7. `cli.py:episode`: distinguish raw legacy done from external cap; count joint steps separately from agent transitions and optimizer steps.
8. `cli.py:main`: predeclared selection and independent evaluation. Per-seed subprocesses avoid the original API's shared Pipe lifecycle problem.
9. `PROTOCOL.json` and `scripts/run_protocol.py`: the three final training seeds are distinct from pilot seed 0. Evaluation seeds are predeclared and not used for selection or tuning.
10. `scripts/summarize_protocol.py`: recompute scoring/windows from raw records, verify checkpoint hash and evaluation seeds, and aggregate per-model evaluation means using population SD.

## Interview explanations

**Why multi-agent?** The agents interact through the same ball; one policy changes the experiences collected by the other. Parallel independent tasks alone do not establish this interaction.

**Why start with independent DDPG?** It gives a simple local-critic reference. A centralized-critic comparison should answer a specific question after checking interface, gradient and exploration failures.

**Is shared replay MADDPG?** No. This baseline stores joint transitions but each critic only uses its own agent's observation/action. The implemented MADDPG conditions each critic on joint observations/actions while keeping execution local.

## Reading the MADDPG implementation

Read `maddpg.py` after the baseline learning update. `critic_dimensions` expands the critic inputs to joint observations `[B,48]` and joint actions `[B,4]`; the actors remain local `[B,24] -> [B,2]`. `critic_targets` uses both target actors, each agent's own reward, and that agent's terminal mask. Both targets are computed before any target-network update.

For agent i, the target is `y_i = r_i + gamma * (1-d_i) * Q_i_target(next_joint_obs, next_joint_target_actions)`. Here B is minibatch size; reward, terminal mask, Q and y all have shape `[B,1]`. The target branch uses `torch.no_grad()` because it supplies a regression label.

`actor_loss` replaces agent i's replay action with its current actor output and holds the partner's executed replay action fixed. Its scalar loss is `-mean(Q_i(joint_obs, own_current_action + partner_replay_action))`, where `+` denotes concatenation in canonical agent order. In `learn`, critic parameters are frozen while this loss backpropagates through the current action to actor i. Using `no_grad()` for this critic forward pass would destroy that actor gradient.

**How should the algorithms be compared?** Compare first qualification in joint environment steps as well as episodes, then compare separately measured frozen-model evaluations. An episode is longer when rallies last longer, so episode count alone is not a sampling budget. The same hidden widths do not imply equal parameter counts: the centralized critics add 6,656 input-layer parameters across the two critics. Three training seeds support descriptive results, not a statistical superiority claim.

**What is actually measured?** Training rolling-100 qualifies the course criterion. Frozen-policy evaluation measures a selected model separately. Population SD across independent training-model means describes between-run variation; it is not a confidence interval.

**What remains uncertain?** Legacy local_done cannot identify internal timeout causes; evaluation with original partners does not measure cross-play; three training seeds give limited stability evidence; Unity success is not real-world deployment evidence.
