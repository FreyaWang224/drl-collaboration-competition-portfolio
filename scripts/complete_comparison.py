"""Finish figures and interpretation only after real results are validated/pushed.

This is an immediate pipeline stage, not a recurring scheduler. It does not
train, select models, retry evaluation seeds, or modify experiment records.
"""
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
flag = root / 'runs/maddpg_finalization_complete.json'
while not flag.exists():
    log = root / 'runs/maddpg_finalization.log'
    if log.exists() and 'Traceback (most recent call last)' in log.read_text():
        raise RuntimeError('Formal finalization failed; inspect retained evidence')
    time.sleep(10)
assert json.loads(flag.read_text()) == {'validated': True, 'published': True}
subprocess.run([sys.executable, str(root / 'scripts/plot_comparison.py')], cwd=root, check=True)
baseline = json.loads((root / 'artifacts/baseline_results.json').read_text())
maddpg = json.loads((root / 'artifacts/maddpg_results.json').read_text())
lines = ['', '## Sampling efficiency and interpretation', '',
         '| Training seed | Independent DDPG first qualifying environment step | MADDPG first qualifying environment step |',
         '|---|---:|---:|']
for b, m in zip(baseline['seeds'], maddpg['seeds']):
    assert b['seed'] == m['seed']
    lines.append(f'| {b["seed"]} | {b["first_solved_environment_step"]} | {m["first_solved_environment_step"]} |')
lines += ['', '![Training aligned by environment steps](artifacts/comparison_training_steps.png)', '',
          'Episode counts are not equal sampling budgets: better rallies can make episodes longer. These curves retain each run separately and show training rolling-100, not evaluation.', '',
          '![Frozen-model evaluation distribution](artifacts/comparison_evaluation.png)', '',
          'Faint dots are actual evaluation episodes; diamonds are per-model means. No confidence-interval error bars are shown. The same evaluation seed list is used across models; the 90 episodes for an algorithm must not be treated as 90 independent training runs.', '',
          '| Online network parameters (two agents) | Independent DDPG | MADDPG |',
          '|---|---:|---:|', '| Actors | 39940 | 39940 |', '| Critics | 40194 | 46850 |', '',
          'Target networks duplicate the corresponding online architectures. More centralized-critic parameters and concurrently executed jobs limit parameter/compute claims. Observed runtime is logged, not a speed superiority test.', '',
          'A complete evaluation episode here ends on legacy `local_done`; this API does not prove whether that signal represents true termination or an internal time limit. External caps are separately recorded. Cross-play and partner generalization have not been tested.', '']
bmean = baseline['across_training_seed_evaluation_mean']
mmean = maddpg['across_training_seed_evaluation_mean']
if bmean is not None and mmean is not None:
    direction = 'higher' if mmean > bmean else 'lower' if mmean < bmean else 'equal'
    lines += [f'Under this frozen protocol, the observed MADDPG across-model evaluation mean is {direction} than independent DDPG ({mmean:.6f} versus {bmean:.6f}). This describes the measured models; three training seeds and unequal critic parameter counts do not establish general or statistically significant superiority.', '']
else:
    lines += ['An incomplete evaluation prevents an across-model performance comparison; partial scores are retained without an aggregate claim.', '']
report = root / 'MADDPG_REPORT.md'
report.write_text(report.read_text() + '\n'.join(lines))
main_report = root / 'Report.md'
text = main_report.read_text().replace(
    'A controlled MADDPG comparison would change the critics to condition on joint information while keeping actors local. Cross-play between training seeds would test partner compatibility separately from original-pair evaluation. Neither experiment has been run yet.',
    'The completed MADDPG comparison changes critics to joint information while keeping actors local; see MADDPG_REPORT.md. Cross-play between training seeds would test partner compatibility separately from original-pair evaluation and has not been run.')
text = text.replace('## MADDPG extension: implementation, not yet a performance claim',
                    '## MADDPG extension: implementation and measured comparison')
text = text.replace('All failures will be retained; no superiority or significance is claimed before measurement.',
                    'All predeclared seeds are retained. The comparison report describes the actual measurements without a statistical significance claim.')
main_report.write_text(text)
readme = root / 'README.md'
text = readme.read_text()
start = text.index('**Status:**')
end = text.index('\n\n', start)
text = text[:start] + '**Status:** independent DDPG and MADDPG implementations and formal experiments complete; 29 tests pass. Each algorithm has three predeclared training seeds and 30 independent evaluation episodes per selected model. See [Report.md](Report.md) for baseline evidence and [MADDPG_REPORT.md](MADDPG_REPORT.md) for the measured comparison. Development pilots are excluded from formal statistics.' + text[end:]
text = text.replace('MADDPG paper, for the possible later centralized-critic comparison', 'MADDPG paper')
text = text.replace('# Planned matched-budget protocol;', '# Matched-budget protocol;')
text = text.replace('The planned comparison keeps hidden widths and sampling budgets the same;',
                    'The completed comparison keeps hidden widths and sampling budgets the same;')
readme.write_text(text)
paths = ['README.md', 'Report.md', 'MADDPG_REPORT.md', 'IMPLEMENTATION_GUIDE.md',
         'scripts/plot_comparison.py', 'scripts/complete_comparison.py']
paths += [f'artifacts/comparison_{kind}.{extension}'
          for kind in ('training_steps', 'evaluation') for extension in ('png', 'svg')]
subprocess.run(['git', 'diff', '--check'], cwd=root, check=True)
subprocess.run(['git', 'add', *paths], cwd=root, check=True)
subprocess.run(['git', 'commit', '-m', 'Compare real training sample efficiency and frozen-model evaluation'], cwd=root, check=True)
subprocess.run(['git', 'push', 'origin', 'main'], cwd=root, check=True)
(root / 'runs/comparison_complete.json').write_text(json.dumps({'published': True}, indent=2))
print('Actual comparison figures, interpretation and reading guide published.', flush=True)
