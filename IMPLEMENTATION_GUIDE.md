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

**Is shared replay MADDPG?** No. This baseline stores joint transitions but each critic only uses its own agent's observation/action. MADDPG would condition critics on joint information while keeping execution local.

**What is actually measured?** Training rolling-100 qualifies the course criterion. Frozen-policy evaluation measures a selected model separately. Population SD across independent training-model means describes between-run variation; it is not a confidence interval.

**What remains uncertain?** Legacy local_done cannot identify internal timeout causes; evaluation with original partners does not measure cross-play; three training seeds give limited stability evidence; Unity success is not real-world deployment evidence.
