"""Validate and publish the running MADDPG comparison when all runs finish."""
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1]
protocol=root/'PROTOCOL_MADDPG.json'
p=json.loads(protocol.read_text())
while True:
    status=root/'runs'/'maddpg_protocol_status.json'
    if status.exists():
        try:
            values=json.loads(status.read_text())
            if any(v.get('training_exit_code',0)!=0 or v.get('evaluation_exit_code',0)!=0 for v in values):
                raise RuntimeError('A formal run failed; retain evidence, do not fabricate results')
        except json.JSONDecodeError:
            pass
    try:
        for s in p['training_seeds']:
            json.loads((root/'runs'/f'maddpg_eval_seed{s}'/'summary.json').read_text())
        break
    except (FileNotFoundError,json.JSONDecodeError):
        time.sleep(10)
subprocess.run([sys.executable,str(root/'scripts/summarize_protocol.py'),'--protocol',str(protocol)],cwd=root,check=True)
m=json.loads((root/'artifacts'/'maddpg_results.json').read_text())
b=json.loads((root/'artifacts'/'baseline_results.json').read_text())
lines=['# MADDPG matched-budget comparison','',
       'Both methods use local actors. Independent DDPG has local critics; MADDPG critics condition on joint observations/actions. Hyperparameters, training seeds, environment-step budgets, selection rules and evaluation seeds are matched. Critic parameter counts and compute differ.', '',
       '| Algorithm | Seed | First solved episode | Complete training episodes | Training truncations | Independent evaluation mean | Within-model population SD |',
       '|---|---:|---:|---:|---:|---:|---:|']
for name,result in [('Independent DDPG',b),('MADDPG',m)]:
    for r in result['seeds']:
        mean='Withheld' if r['evaluation_mean'] is None else f'{r["evaluation_mean"]:.6f}'
        sd='Withheld' if r['evaluation_within_model_population_sd'] is None else f'{r["evaluation_within_model_population_sd"]:.6f}'
        lines.append(f'| {name} | {r["seed"]} | {r["first_solved_episode"]} | {r["complete_training_episodes"]} | {r["training_truncations"]} | {mean} | {sd} |')
lines += ['', '| Algorithm | Across-model evaluation mean | Population SD of three model means |', '|---|---:|---:|']
for name,result in [('Independent DDPG',b),('MADDPG',m)]:
    mean=result['across_training_seed_evaluation_mean'];sd=result['across_training_seed_population_sd']
    lines.append(f'| {name} | {mean:.6f} | {sd:.6f} |' if mean is not None else f'| {name} | Withheld | Withheld |')
lines += ['', 'These are descriptive results from three training seeds, not confidence intervals or statistical significance. Unsolved training seeds remain in the table and use final checkpoints. No checkpoint was selected using evaluation scores. Pilot results are excluded.', '',
          'Compute costs and single-episode training maxima are recorded separately in the result JSON; no runtime superiority is inferred from concurrently executed jobs.', '']
for s in p['training_seeds']:
    lines.append(f'![MADDPG training seed {s}](artifacts/maddpg_seed{s}/training_curve.png)')
(root/'MADDPG_REPORT.md').write_text('\n'.join(lines)+'\n')
report=root/'Report.md'
report.write_text(report.read_text().replace('Formal MADDPG results are not available yet.',
                                            'Formal MADDPG measurements are complete; see [MADDPG_REPORT.md](MADDPG_REPORT.md) and `artifacts/maddpg_results.json`.'))
readme=root/'README.md'
readme.write_text(readme.read_text().replace('MADDPG performance is not yet established. Its pilot is development evidence only.',
                                            'Formal MADDPG experiments are complete; see [MADDPG_REPORT.md](MADDPG_REPORT.md). Its pilot remains development evidence only.'))
paths=['README.md','Report.md','MADDPG_REPORT.md','artifacts/maddpg_results.json']
for s in p['training_seeds']:
    paths += [f'artifacts/maddpg_seed{s}',f'artifacts/maddpg_eval_seed{s}',f'artifacts/models/maddpg_seed{s}.pt']
subprocess.run(['git','add',*paths],cwd=root,check=True)
subprocess.run(['git','commit','-m','Report actual matched-budget MADDPG experiments'],cwd=root,check=True)
subprocess.run(['git','push','origin','main'],cwd=root,check=True)
(root/'runs'/'maddpg_finalization_complete.json').write_text(json.dumps(dict(validated=True,published=True),indent=2))
print('MADDPG actual results validated, documented and published.',flush=True)
