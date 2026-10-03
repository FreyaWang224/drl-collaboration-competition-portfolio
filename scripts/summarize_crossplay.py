"""Validate actual cross-play evidence and export descriptive matrix/report."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
p=json.loads((root/'PROTOCOL_CROSSPLAY.json').read_text())
seeds=p['training_seeds'];cells=[]
for a in seeds:
    for b in seeds:
        if a==b:
            source=p['sources'][str(a)]
            path=root/source['diagonal_records']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=source['diagonal_sha256']:
                raise ValueError('Diagonal records changed')
            records=json.loads(path.read_text())['records']
            manifest=json.loads((path.parent/'manifest.json').read_text())
            summary=json.loads((path.parent/'summary.json').read_text())
            if manifest['checkpoint_sha256']!=source['sha256'] or summary['learning_rounds_during_evaluation']!=0:
                raise ValueError('Unverified diagonal model/evaluation')
        else:
            records=[]
            for evaluation_seed in p['evaluation_seeds']:
                directory=root/'runs/crossplay'/f'{a}_{b}'/f'seed_{evaluation_seed}'
                record=json.loads((directory/'episode.json').read_text())
                manifest=json.loads((directory/'manifest.json').read_text())
                summary=json.loads((directory/'summary.json').read_text())
                expected=[p['sources'][str(s)]['sha256'] for s in (a,b)]
                if manifest['source_checkpoint_sha256']!=expected or manifest['actor_training_seeds']!=[a,b] or manifest['actor_roles']!=[0,1]:
                    raise ValueError('Cross-play source identity mismatch')
                if manifest['exploration'] or manifest['learning'] or summary['optimizer_updates']!=0 or not summary['actor_parameters_unchanged']:
                    raise ValueError('Cross-play learned or explored')
                if record['actor_training_seeds']!=[a,b] or record['evaluation_seed']!=evaluation_seed:
                    raise ValueError('Record role/seed mismatch')
                if summary['complete']!=record['complete']:
                    raise ValueError('Completion mismatch')
                records.append(record)
        if [r['evaluation_seed'] for r in records]!=p['evaluation_seeds']:
            raise ValueError('Evaluation seed order mismatch')
        for record in records:
            if not np.isclose(record['score'],max(record['agent_returns']),atol=1e-9):
                raise ValueError('Score mismatch')
            if record['complete']==record['truncated']:
                raise ValueError('Invalid completion flags')
        complete=all(r['complete'] for r in records)
        scores=[r['score'] for r in records]
        cell=dict(actor0_seed=a,actor1_seed=b,diagonal_reused=a==b,episodes=len(records),
                  external_truncations=sum(r['truncated'] for r in records),
                  mean_score=float(np.mean(scores)) if complete else None,
                  within_cell_population_sd=float(np.std(scores)) if complete else None,
                  mean_agent_returns=np.mean([r['agent_returns'] for r in records],axis=0).tolist() if complete else None,
                  checkpoint_sha256=[p['sources'][str(s)]['sha256'] for s in (a,b)])
        cells.append(cell)
        out=root/'artifacts/crossplay'/f'{a}_{b}'
        out.mkdir(parents=True,exist_ok=True)
        (out/'episodes.json').write_text(json.dumps(dict(kind=p['kind'],records=records),indent=2)+'\n')
        (out/'summary.json').write_text(json.dumps(cell,indent=2)+'\n')
for cell in cells:
    diagonal=next(c for c in cells if c['actor0_seed']==cell['actor0_seed'] and c['actor1_seed']==cell['actor0_seed'])
    cell['signed_self_play_gap']=diagonal['mean_score']-cell['mean_score'] if diagonal['mean_score'] is not None and cell['mean_score'] is not None else None

def average(diagonal):
    values=[c['mean_score'] for c in cells if c['diagonal_reused']==diagonal]
    return float(np.mean(values)) if None not in values else None
result=dict(protocol=p,cells=cells,diagonal_mean=average(True),off_diagonal_mean=average(False),
            limitations=p['limits'],inference=p['aggregation'])
(root/'artifacts/crossplay_results.json').write_text(json.dumps(result,indent=2)+'\n')
matrix=np.array([[next(c['mean_score'] for c in cells if c['actor0_seed']==a and c['actor1_seed']==b) for b in seeds] for a in seeds],dtype=float)
fig,ax=plt.subplots(figsize=(6.5,5))
im=ax.imshow(np.ma.masked_invalid(matrix),cmap='Blues',vmin=0)
for i in range(len(seeds)):
    for j in range(len(seeds)):
        value=matrix[i,j];label=f'{value:.3f}' if np.isfinite(value) else 'Withheld'
        ax.text(j,i,label,ha='center',va='center',color='black',bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
ax.set(xticks=range(3),yticks=range(3),xticklabels=seeds,yticklabels=seeds,
       xlabel='Actor 1 training seed (partner)',ylabel='Actor 0 training seed',
       title='Exploratory MADDPG cross-play: 30 evaluation seeds per cell')
fig.colorbar(im,ax=ax,label='Mean max undiscounted agent return')
fig.tight_layout()
for extension in ('png','svg'):fig.savefig(root/f'artifacts/crossplay_matrix.{extension}',dpi=180)
plt.close(fig)
lines=['# Exploratory MADDPG cross-play','',
       'This is a post-training partner-compatibility analysis, separate from the original algorithm comparison. Rows supply actor 0 and columns supply actor 1; roles are never swapped. Three diagonal cells reuse verified original-pair evaluations; six ordered off-diagonal pairs each run 30 new episodes on the same fixed environment seed list. No critic, optimizer, exploration, learning, selection by evaluation, or score-based retries are used.', '',
       '![Cross-play matrix](artifacts/crossplay_matrix.png)', '',
       '| Actor 0 seed | Actor 1 seed | Mean score | Within-cell population SD | Mean actor 0 return | Mean actor 1 return | Signed self-play gap | Episodes / external truncations |',
       '|---|---|---:|---:|---:|---:|---:|---|']
fmt=lambda v:'Withheld' if v is None else f'{v:.6f}'
for c in cells:
    returns=c['mean_agent_returns'] or [None,None]
    lines.append(f'| {c["actor0_seed"]} | {c["actor1_seed"]} | {fmt(c["mean_score"])} | {fmt(c["within_cell_population_sd"])} | {fmt(returns[0])} | {fmt(returns[1])} | {fmt(c["signed_self_play_gap"])} | {c["episodes"]} / {c["external_truncations"]} |')
lines+=['',f'Diagonal mean: **{fmt(result["diagonal_mean"])}**. Off-diagonal mean: **{fmt(result["off_diagonal_mean"])}**. These average model-combination means, not independent training seeds.', '',
        'Signed self-play gap = row diagonal mean minus cell mean: positive means the fixed actor-0 policy performs worse in the joint score when paired with the new actor-1 partner. This is a descriptive joint outcome, not evidence assigning blame to a particular actor. Reverse pairings are different experiments.', '',
        'All cell-level episode records and source checkpoint hashes are retained. External truncation withholds a cell aggregate. Local_done still cannot distinguish internal timeout from true termination. The maximum-agent score can conceal individual return differences, hence both returns are reported.', '',
        'Six off-diagonal cells share the same three trained actor pairs and use aligned environment seeds; they are not six independent training runs. No confidence interval, statistical significance, arbitrary-partner generalization, or causal claim about centralized training is made. These data do not tune or select the original checkpoints.', '']
(root/'CROSSPLAY_REPORT.md').write_text('\n'.join(lines)+'\n')
readme=root/'README.md'
text=readme.read_text()
text+='\n## Exploratory cross-play\n\nSee [CROSSPLAY_REPORT.md](CROSSPLAY_REPORT.md) for the actual role-preserving 3 x 3 MADDPG partner matrix, and `PROTOCOL_CROSSPLAY.json` for the frozen protocol. This post-training analysis is separate from the baseline/MADDPG comparison; cross-play data do not select or tune checkpoints.\n'
readme.write_text(text)
print(json.dumps(result,indent=2))
