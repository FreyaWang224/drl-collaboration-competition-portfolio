# Collaboration and Competition — Independent DDPG

A from-scratch, two-agent continuous-control baseline for Udacity's supplied Unity Tennis environment. Each agent has its own actor and **local** critic. Synchronized replay storage is shared; this is independent DDPG, not MADDPG.

**Status:** implementation and 21 tests complete; real Unity smoke test complete; two development pilots retained. The longer pilot reached the course training threshold. Three predeclared independent training runs and 30 evaluation episodes per model are complete. See Report.md and artifacts/baseline_results.json for the verified measurements.

## Task and verified environment

Two agents keep a ball in play. Hitting it over the net gives +0.1; dropping it or hitting out gives -0.01. Runtime observation shape is `[2,24]`: 8 variables per agent, stacked over 3 observations. Each agent outputs two continuous action components (horizontal movement and jump), within `[-1,1]`; joint actions have shape `[2,2]`.

Episode score = maximum of the two **undiscounted episode returns**. Solved means average score >=0.5 over 100 consecutive complete training episodes. This differs from independent evaluation and from the single best episode.

The provided Mac binary is x86_64, Unity Player **2017.3.1f1 (fc1d3344e6ea)**. It ran successfully on the author's Apple Silicon Mac. Legacy API source is `unityagents 0.4.0`, pinned to Udacity commit `561eec3ae8678a23a4557f1a15414a9b076fdfff`. Do not substitute modern ML-Agents Tennis. No Unity Editor installation is needed.

## Setup (verified local stack)

Python 3.9.6, PyTorch 2.8.0, NumPy 2.0.2, protobuf 3.20.3 and grpcio 1.80.0; exact installed dependencies are in `requirements-lock.txt`. The Mac NumPy wheel requires a compatible macOS version. Binary execution requires x86_64 support on this machine. Other platforms have not been tested.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-lock.txt
.venv/bin/python scripts/download_environment.py
export PYTHONPATH="$PWD/src:$PWD/environments/legacy-python"
.venv/bin/python -m pytest -q
```

The download script retrieves the **course-supplied Mac binary** and official legacy Python sources into ignored `environments/`; it records archive SHA256 and sets executable permission. It requires network access. NumPy 2's removed `float_` alias is restored only when constructing the legacy adapter. The source remains separate from this project's own implementation.

## Run

Output directories must be new; existing runs are never silently overwritten.

```bash
# Interface audit only: random actions, no learning
.venv/bin/python -m tennis.cli smoke --environment environments/Tennis.app \
  --output runs/my_smoke --episodes 3 --max-steps 2000

# A single training run (full budget and configuration must be declared first)
.venv/bin/python -m tennis.cli train --environment environments/Tennis.app \
  --output runs/my_train --seed 11 --worker-id 11 --episodes 3000 \
  --max-environment-steps 200000 --max-steps 10000 --warmup-steps 10000

# Frozen checkpoint evaluation; supply all 30 seeds declared in PROTOCOL.json
.venv/bin/python -m tennis.cli evaluate --environment environments/Tennis.app \
  --output runs/my_eval --checkpoint runs/my_train/first_solved.pt \
  --worker-id 111 --evaluation-seeds 20001 20002 20003

# Complete predeclared 3-training-seed / 30-evaluation-episode protocol
# Runs three training processes on separate workers, then three evaluation processes.
# Each uses one Torch CPU thread.
.venv/bin/python scripts/run_protocol.py
```

The abbreviated evaluation command shows CLI usage, not a complete final evaluation. Use `PROTOCOL.json` for the full seed set. Add `--graphics` to view the simulator. A first-solved checkpoint saves both actors together. If a seed never solves, the runner evaluates `final.pt` and retains the failure label.

Evaluation uses a fresh Python/Unity process per seed because the legacy communicator has class-level Pipe handles that cannot be reused after close. No exploration or learning takes place. The CLI does not offer training resume: a full training checkpoint restores Python learning state, but not the Unity simulation state.

## Evidence and limitations

See `Report.md`, `PROTOCOL.json`, and `artifacts/`. Raw local run directories and binaries are ignored. Small exported JSON/plots retain the measured evidence. Development pilots are labelled and excluded from final three-seed statistics.

Legacy `local_done` does not distinguish true terminal states from internal Unity time limits. This baseline treats it as terminal, explicitly following the legacy convention. External script caps are logged separately and preserve bootstrapping; a truncated episode breaks the qualifying rolling window. Evaluation aggregates are withheld if any episode is truncated.

Resume material stays local under ignored `resume/`. Previous project repositories are not reused or modified. Public repository: https://github.com/FreyaWang224/drl-collaboration-competition-portfolio.

## Sources

- [Course index](https://learn-udacity.top/udrl597102/Deep%20Reinforcement%20Learning%20Nanodegree%20v5.0.0/index.html)
- [Official project and downloads](https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet)
- [Official API package version](https://github.com/udacity/deep-reinforcement-learning/blob/master/python/setup.py)
- [MADDPG paper, for the possible later centralized-critic comparison](https://arxiv.org/abs/1706.02275)
