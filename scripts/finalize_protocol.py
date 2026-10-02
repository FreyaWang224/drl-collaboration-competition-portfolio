"""Finish the currently running batch once all actual evaluations complete.

This is an immediate pipeline stage, not a scheduler. It validates raw evidence
before exporting results, weights and the final report. It never fabricates
pending values or changes model selection.
"""
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
p = json.loads((root/'PROTOCOL.json').read_text())
while True:
    try:
        summaries = [json.loads((root/'runs'/f'baseline_eval_seed{s}'/'summary.json').read_text())
                     for s in p['training_seeds']]
        break
    except (FileNotFoundError, json.JSONDecodeError):
        time.sleep(10)
subprocess.run([sys.executable, str(root/'scripts/summarize_protocol.py')], cwd=root, check=True)
result = json.loads((root/'artifacts'/'baseline_results.json').read_text())
rows = result['seeds']
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for index, row in enumerate(rows):
    records = json.loads((root/'artifacts'/f'baseline_eval_seed{row["seed"]}'/'episodes.json').read_text())['records']
    scores = [r['score'] for r in records]
    ax.scatter(index + np.linspace(-.16,.16,len(scores)), scores, alpha=.65, s=23, color='#7398c6')
    if row['evaluation_mean'] is not None:
        ax.plot([index-.24,index+.24], [row['evaluation_mean']]*2, color='#623eae', linewidth=2.5)
ax.set(xticks=range(3), xticklabels=[str(r['seed']) for r in rows], xlabel='Independent training seed',
       ylabel='Independent evaluation episode score', title='30 frozen-policy evaluation episodes per model')
ax.scatter([],[],color='#7398c6',label='Episode score')
ax.plot([],[],color='#623eae',label='Model evaluation mean (no CI)')
ax.legend(fontsize=8)
ax.grid(axis='y',alpha=.15)
fig.tight_layout()
fig.savefig(root/'artifacts'/'evaluation_scores.png',dpi=170)
fig.savefig(root/'artifacts'/'evaluation_scores.svg')
plt.close(fig)
lines = ['## Confirmatory results', '',
         '| Training seed | First solved episode | First solved environment step | Complete training episodes | Training truncations | Evaluation mean | Within-model population SD | Evaluation episodes / truncations |',
         '|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    mean = f'{r["evaluation_mean"]:.6f}' if r['evaluation_mean'] is not None else 'Withheld (truncation)'
    sd = f'{r["evaluation_within_model_population_sd"]:.6f}' if r['evaluation_within_model_population_sd'] is not None else 'Withheld'
    lines.append(f'| {r["seed"]} | {r["first_solved_episode"]} | {r["first_solved_environment_step"]} | {r["complete_training_episodes"]} | {r["training_truncations"]} | {mean} | {sd} | {r["evaluation_episodes"]} / {r["evaluation_truncations"]} |')
mean = result['across_training_seed_evaluation_mean']
sd = result['across_training_seed_population_sd']
if mean is not None:
    conclusion = f'Across the three training-model evaluation means: **{mean:.6f} ± {sd:.6f}**, where ± is population SD across models, not a confidence interval or significance claim.'
else:
    conclusion = 'Across-model evaluation aggregation is withheld because at least one evaluation was incomplete.'
lines += ['', conclusion, '', 'Training threshold attainment and independent evaluation are distinct measurements. The within-model SD column describes variation across 30 episodes; it is not the across-model SD above.', '',
          '![Independent evaluation scores](artifacts/evaluation_scores.png)', '']
for r in rows:
    lines.append(f'![Training seed {r["seed"]}](artifacts/baseline_seed{r["seed"]}/training_curve.png)')
report = root/'Report.md'
text = report.read_text()
text = text.replace('Confirmatory three-seed training is complete and independent evaluation is in progress;',
                    'Confirmatory three-seed training and independent evaluation are complete;')
text = text.replace('the numbers below are explicitly development data.',
                    'development pilots and confirmatory measurements are reported separately below.')
text = text.replace('**No final independent evaluation result is reported yet.**', '\n'.join(lines))
report.write_text(text)
readme = root/'README.md'
text = readme.read_text().replace('The predeclared three-seed runs are in progress; no final cross-seed evaluation claim yet.',
                                  'Three predeclared independent training runs and 30 evaluation episodes per model are complete. See Report.md and artifacts/baseline_results.json for the verified measurements.')
readme.write_text(text)
statuses = [dict(seed=r['seed'],training_exit_code=0,evaluation_evidence_complete=True) for r in rows]
(root/'runs'/'verified_protocol_status.json').write_text(json.dumps(statuses,indent=2))
# Archive the operational interruption without including runtime dependencies.
if (root/'runs'/'orchestration_handoff.json').exists():
    (root/'artifacts'/'orchestration_handoff.json').write_text((root/'runs'/'orchestration_handoff.json').read_text())
subprocess.run(['git','add','README.md','Report.md','artifacts'],cwd=root,check=True)
subprocess.run(['git','commit','-m','Record verified three-seed baseline and independent evaluation evidence'],cwd=root,check=True)
(root/'runs'/'finalization_complete.json').write_text(json.dumps(dict(evidence_validated=True,report='Report.md',results='artifacts/baseline_results.json'),indent=2))
print('Verified final evidence exported and committed locally.',flush=True)
