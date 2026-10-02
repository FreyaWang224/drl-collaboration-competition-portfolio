# Prospective experiment protocol

1. Audit supplied binary hash, platform/architecture, legacy API dependencies, agent IDs, observation/action dimensions, reset/step and terminal behavior. No environment downloaded yet.
2. Verify random-policy scoring. Build two independent DDPG agents, synchronized joint replay, local critics, separate actor/critic targets, training/evaluation CLI and meaningful numerical/gradient/score tests.
3. Run one debugging pilot; freeze hyperparameters, seed set and budget after pilot. Pilot is not final evidence.
4. Run three independently initialized training seeds. Preserve failed runs. Predeclare environment-step and episode limits before launching final runs. Log environment steps, agent transitions, gradient updates, elapsed time, complete episodes and truncations.
5. Save both actors at the first complete rolling-100 >= 0.5. If unsolved, report failure and retain final model. Do not select checkpoints using final evaluation results.
6. Evaluate each frozen candidate for 30 complete episodes with independent, predeclared evaluation seeds, no exploration and no learning. Do not tune on evaluation data.
7. Report episode max-agent score, rolling-100 and first solved endpoint separately from evaluation mean, single-episode maximum, per-agent returns and rally length. Aggregate per-training-seed evaluation means using population SD; do not describe it as CI or significance.
8. Diagnose actual failures before adding MADDPG or another enhancement. For comparisons use the same seed set and sampling budget; report compute separately.
9. Publish real JSON logs, plots, model weights, English Report, reproducibility instructions and demo. Resume sources remain local.

No measured Project 3 results exist at this stage.
