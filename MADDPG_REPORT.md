# MADDPG matched-budget comparison

Both methods use local actors. Independent DDPG has local critics; MADDPG critics condition on joint observations/actions. Hyperparameters, training seeds, environment-step budgets, selection rules and evaluation seeds are matched. Critic parameter counts and compute differ.

| Algorithm | Seed | First solved episode | Complete training episodes | Training truncations | Independent evaluation mean | Within-model population SD |
|---|---:|---:|---:|---:|---:|---:|
| Independent DDPG | 11 | 1282 | 1971 | 1 | 1.242667 | 1.057569 |
| Independent DDPG | 22 | 1473 | 2085 | 1 | 0.588333 | 0.916814 |
| Independent DDPG | 33 | 1222 | 1692 | 1 | 2.023000 | 0.991131 |
| MADDPG | 11 | 1267 | 1776 | 1 | 2.480000 | 0.449741 |
| MADDPG | 22 | 1526 | 2190 | 1 | 0.676667 | 0.861659 |
| MADDPG | 33 | 1176 | 1855 | 1 | 1.520333 | 1.241619 |

| Algorithm | Across-model evaluation mean | Population SD of three model means |
|---|---:|---:|
| Independent DDPG | 1.284667 | 0.586453 |
| MADDPG | 1.559000 | 0.736715 |

These are descriptive results from three training seeds, not confidence intervals or statistical significance. Unsolved training seeds remain in the table and use final checkpoints. No checkpoint was selected using evaluation scores. Pilot results are excluded.

Compute costs and single-episode training maxima are recorded separately in the result JSON; no runtime superiority is inferred from concurrently executed jobs.

![MADDPG training seed 11](artifacts/maddpg_seed11/training_curve.png)
![MADDPG training seed 22](artifacts/maddpg_seed22/training_curve.png)
![MADDPG training seed 33](artifacts/maddpg_seed33/training_curve.png)
