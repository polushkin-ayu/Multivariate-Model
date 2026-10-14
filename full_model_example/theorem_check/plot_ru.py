"""Plot convergence of the direct deterministic theorem trajectories."""
import json
import os
os.environ.setdefault('MPLCONFIGDIR', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.mplconfig')))
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
r = json.loads((OUT/'results.json').read_text())
Ns = r['design']['N']
errors = [r['summary']['max_abs_normalized_error_by_N'][str(n)] for n in Ns]

fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
ax.loglog(Ns, errors, 'o-', lw=2, label='Максимум по $h/s$ и $z$')
for n, err in zip(Ns, errors):
    ax.annotate(f'{err:.3g}', (n, err), xytext=(4, 7),
                textcoords='offset points', fontsize=8)
ax.set(xlabel='Размер калибровки $N$', ylabel='Максимальная абсолютная ошибка',
       title='Сходимость на детерминированных траекториях')
ax.grid(True, which='both', alpha=.25)
ax.legend(frameon=False)
fig.savefig(OUT/'convergence_ru.png', dpi=180)
fig.savefig(OUT/'convergence_ru.pdf')
