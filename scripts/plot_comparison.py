"""Plot verified formal records without pooling training and evaluation scores."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
artifacts = root / 'artifacts'
methods = [('baseline', 'Independent DDPG', '#2466aa'),
           ('maddpg', 'MADDPG', '#7941a6')]
results = {prefix: json.loads((artifacts / f'{prefix}_results.json').read_text())
           for prefix, _, _ in methods}
seeds = results['baseline']['protocol']['training_seeds']
assert seeds == results['maddpg']['protocol']['training_seeds']
assert (results['baseline']['protocol']['max_environment_steps'] ==
        results['maddpg']['protocol']['max_environment_steps'])

fig, axes = plt.subplots(1, len(seeds), figsize=(13, 4), sharex=True, sharey=True)
for ax, seed in zip(axes, seeds):
    for prefix, label, color in methods:
        records = json.loads((artifacts / f'{prefix}_seed{seed}' / 'episodes.json').read_text())['records']
        # Retain gaps when external truncation breaks the consecutive window.
        ax.plot([r['environment_steps'] for r in records],
                [r['rolling_100'] if r['rolling_100'] is not None else np.nan for r in records],
                color=color, linewidth=1.1, label=label)
        row = next(r for r in results[prefix]['seeds'] if r['seed'] == seed)
        if row['first_solved_environment_step'] is not None:
            ax.scatter(row['first_solved_environment_step'], .5, color=color, s=24, zorder=3)
    ax.axhline(.5, color='#288254', linestyle='--', linewidth=1)
    ax.set(title=f'Training seed {seed}', xlabel='Joint environment steps')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.grid(alpha=.15)
axes[0].set_ylabel('Training rolling-100 episode score')
axes[0].legend(fontsize=8)
fig.suptitle('Matched sampling budget; dots mark first training qualification')
fig.tight_layout()
for ext in ('png', 'svg'):
    fig.savefig(artifacts / f'comparison_training_steps.{ext}', dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5))
for offset, (prefix, label, color) in zip((-.14, .14), methods):
    rows = results[prefix]['seeds']
    for index, row in enumerate(rows):
        if row['evaluation_truncations']:
            raise ValueError('Do not plot partial evaluations as complete model means')
        records = json.loads((artifacts / f'{prefix}_eval_seed{row["seed"]}' / 'episodes.json').read_text())['records']
        scores = np.array([r['score'] for r in records])
        assert np.isclose(scores.mean(), row['evaluation_mean'])
        # Deterministic visual jitter only; no synthetic outcomes or error bars.
        x = index + offset
        ax.scatter(x + np.linspace(-.06, .06, len(scores)), scores,
                   color=color, alpha=.25, s=14)
        ax.scatter(x, row['evaluation_mean'], color=color, marker='D', s=65,
                   edgecolors='white', zorder=3, label=label if index == 0 else None)
ax.set(xticks=range(len(seeds)), xticklabels=[str(s) for s in seeds],
       xlabel='Training seed of frozen paired model',
       ylabel='Independent evaluation episode score',
       title='30 fixed evaluation seeds per model; diamonds = model means')
ax.grid(axis='y', alpha=.15)
ax.legend()
fig.tight_layout()
for ext in ('png', 'svg'):
    fig.savefig(artifacts / f'comparison_evaluation.{ext}', dpi=180)
plt.close(fig)
print('Verified training-step and independent-evaluation comparisons exported.')
