"""Export real run records and plots; never substitute diagnostics for final eval."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('run')
parser.add_argument('--kind', choices=['pilot', 'formal', 'smoke'], required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
source = root / 'runs' / args.run
summary = json.loads((source / 'summary.json').read_text())
records = [json.loads(p.read_text()) for p in sorted(source.glob('episode_*.json'))]
for record in records:
    if not np.isclose(record['score'], max(record['agent_returns']), atol=1e-9):
        raise ValueError('Score aggregation mismatch')
output = root / 'artifacts' / args.run
output.mkdir(parents=True, exist_ok=True)
for filename in ('summary.json', 'manifest.json'):
    shutil.copyfile(source / filename, output / filename)
(output / 'episodes.json').write_text(json.dumps(dict(kind=args.kind, records=records), indent=2))
if 'first_solved' in summary and args.kind != 'smoke':
    fig, ax = plt.subplots(figsize=(9, 4.5))
    complete = [r for r in records if r['complete']]
    ax.plot([r['episode'] for r in complete], [r['score'] for r in complete],
            color='#8ca9ce', alpha=0.6, linewidth=0.65, label='Complete training episode score')
    rolling = [r for r in records if r['rolling_100'] is not None]
    ax.plot([r['episode'] for r in rolling], [r['rolling_100'] for r in rolling],
            color='#623eae', linewidth=1.8, label='Training rolling-100')
    ax.axhline(0.5, color='#288254', linestyle='--', label='Course threshold = 0.5')
    truncated = [r for r in records if r['truncated']]
    if truncated:
        ax.scatter([r['episode'] for r in truncated], [r['score'] for r in truncated],
                   marker='x', color='#b43d3d', label='Partial score at external truncation')
    ax.set(xlabel='Training episode', ylabel='Max of undiscounted agent returns',
           title=f'{args.kind.title()} — seed {summary["seed"]} — {summary.get("algorithm", "independent_ddpg")}')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(output / 'training_curve.png', dpi=170)
    fig.savefig(output / 'training_curve.svg')
    plt.close(fig)
print(output)
