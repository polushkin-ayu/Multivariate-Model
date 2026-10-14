# Render English counterparts from saved experiment outputs. The N=64 histogram
# replays the original seeded calibration draws and does not rewrite any JSON.
import os
os.environ.setdefault('MPLCONFIGDIR', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.mplconfig')))
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter

OUT = Path(__file__).resolve().parent
r = json.loads((OUT / 'results.json').read_text())
flow = r['design']['flow']; p = r['primary']; base = r['baseline']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titleweight': 'bold', 'figure.facecolor': 'white', 'axes.grid': True,
    'grid.alpha': .17, 'savefig.bbox': 'tight'})
fmt = FuncFormatter(lambda x, pos: f'{x:,.0f}')

def save(fig, name):
    fig.savefig(OUT / f'{name}_en.png', dpi=180)
    fig.savefig(OUT / f'{name}_en.pdf')
    plt.close(fig)

a = r['curve']; t = np.array([q['fitted_tau'] for q in a])
g = flow * np.array([q['gain'] for q in a]); tau = r['design']['tau']
fig, (ax, bx) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
    gridspec_kw={'height_ratios': [2, 1]}, layout='constrained')
ax.plot(t, g, color='#245d87', lw=2.5, label='Full nonlinear model')
ax.fill_between(t, g, 0, where=g < 0, color='#c74d47', alpha=.15)
ax.fill_between(t, g, 0, where=g >= 0, color='#2a8c70', alpha=.25)
ax.axhline(0, color='#333', lw=.9)
ax.axvline(tau, color='#2a8c70', ls='--', label='True score strength, τ = 0.25')
ax.scatter([tau], [flow * r['oracle_gain']], color='#2a8c70', zorder=4)
ax.set_title('A useful score can hurt decisions when calibration is wrong', loc='left', fontsize=13)
ax.set_ylabel('Profit change, currency units\nper 100,000 applicants')
ax.yaxis.set_major_formatter(fmt); ax.legend(loc='lower right', frameon=False, fontsize=10)
h = r['pilot_distribution']; edges = np.array(h['edges']); counts = np.array(h['counts']) / sum(h['counts']) * 100
bx.bar((edges[:-1] + edges[1:]) / 2, counts, width=np.diff(edges) * .93, color='#8297aa')
bx.axvline(tau, color='#2a8c70', ls='--')
bx.set_ylabel('Share of training runs, %'); bx.set_xlabel('Estimated score strength, τ̂')
bx.set_title('Calibrating the same score from a 64-customer pilot', loc='left', fontsize=11)
save(fig, 'calibration_mechanism')

info = flow * r['oracle_gain']; cost = info - flow * p['mean_gain']; net = flow * p['mean_gain']
fig, ax = plt.subplots(figsize=(8, 4.8), layout='constrained')
ax.bar(0, info, color='#2a8c70', width=.6)
ax.bar(1, -cost, bottom=info, color='#c74d47', width=.6)
ax.bar(2, net, color='#245d87', width=.6)
ax.plot([.3, .7], [info, info], color='#777', lw=1, ls=':')
ax.plot([1.3, 1.7], [net, net], color='#777', lw=1, ls=':')
for x, y, label in [(0, info, f'+{info:,.0f}'), (1, net / 2, f'−{cost:,.0f}'), (2, net, f'{net:,.0f}')]:
    ax.annotate(label, (x, y), xytext=(0, 9 if x == 0 else -20 if x == 2 else 0),
        textcoords='offset points', ha='center', fontweight='bold', fontsize=14)
ax.axhline(0, color='#333', lw=.9)
ax.set_xticks([0, 1, 2], ['Value with\ncorrect calibration', 'Loss relative to\ncorrect calibration', 'Average result of\nusing the score'])
ax.set_ylabel('Currency units per 100,000 applicants'); ax.yaxis.set_major_formatter(fmt)
ax.set_title('The signal has value, but estimating it costs more', loc='left', fontsize=13)
ax.set_ylim(min(net * 1.45, -1100), info * 1.5)
save(fig, 'profit_decomposition')

rows = r['by_N']; ns = np.array([q['N'] for q in rows])
means = flow * np.array([q['mean_gain'] for q in rows]); ses = flow * np.array([q['mc_se'] for q in rows])
pw = np.array([q['fraction_worse'] for q in rows]); seprob = np.sqrt(pw * (1 - pw) / 200000)
fig, (ax, bx) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
    gridspec_kw={'height_ratios': [2, 1]}, layout='constrained')
ax.plot(ns, means, color='#245d87', lw=2, label='Mean effect of training')
ax.fill_between(ns, means - 1.96 * ses, means + 1.96 * ses, color='#245d87', alpha=.25,
    label='95% Monte Carlo interval for the mean')
ax.axhline(info, color='#2a8c70', ls='--', label='Value with known calibration')
ax.axhline(0, color='#333', lw=.9); ax.scatter([64], [net], color='#c74d47', zorder=4)
ax.set_ylabel('Profit change, currency units\nper 100,000 applicants'); ax.yaxis.set_major_formatter(fmt)
ax.set_title('More data on the same score changes the balance', loc='left', fontsize=13)
ax.legend(loc='lower right', frameon=False, fontsize=9)
bx.plot(ns, pw * 100, color='#c74d47', lw=2)
bx.fill_between(ns, 100 * (pw - 1.96 * seprob), 100 * (pw + 1.96 * seprob), color='#c74d47', alpha=.25)
bx.set_ylabel('Training runs with\nprofit loss, %')
bx.set_xlabel('Customers in the pilot used to calibrate the new score, N')
bx.set_ylim(0, 100); bx.set_xscale('log', base=2)
bx.set_xticks([32, 64, 128, 256, 512, 1024, 2048, 4096]); bx.xaxis.set_major_formatter(ScalarFormatter())
fig.get_layout_engine().set(h_pad=.12)
save(fig, 'learning_curve')

# Read the exact deterministic replay histogram saved during package preparation.
dist = json.loads((OUT / 'profit_histogram.json').read_text())
counts = np.asarray(dist['counts'], dtype=int)
edges = np.asarray(dist['edges'], dtype=float)
centers = (edges[1:] + edges[:-1]) / 2
fig, ax = plt.subplots(figsize=(9, 4.6), layout='constrained')
ax.bar(centers, counts / dist['reps'] * 100, width=np.diff(edges) * .95,
    color=np.where(centers < 0, '#c74d47', '#2a8c70'))
ax.axvline(0, color='#222'); ax.axvline(dist['mean'], color='#245d87', ls='--',
    label=f"Mean effect: {dist['mean']:,.0f}")
ax.set_title('The same score: distribution of financial outcomes after training',
    loc='left', fontsize=12, fontweight='bold')
ax.set_xlabel('Profit change, currency units per 100,000 applicants')
ax.set_ylabel('Share of calibrations, %'); ax.xaxis.set_major_formatter(fmt)
ax.text(.03, .9, f"Loss in {100 * dist['fraction_worse']:.1f}% of pilots\nN = {dist['N']}; {dist['reps']:,} calibrations",
    transform=ax.transAxes, va='top')
ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0, .7))
save(fig, 'profit_distribution')

# Reference-quality lines are fully cached in reference_quality.json; no simulation rerun.
rq = json.loads((OUT / 'reference_quality.json').read_text())
fig, ax = plt.subplots(figsize=(9, 4.6), layout='constrained')
for dh, color in [(1, '#2a8c70'), (4, '#245d87'), (16, '#c74d47')]:
    selected = [q for q in rq if q['D_H'] == dh]
    ax.plot([q['N'] for q in selected], [q['mean_gain_per_100k'] for q in selected],
        color=color, lw=2, label=f'Reference noise: σ = {int(np.sqrt(dh))}')
    ax.fill_between([q['N'] for q in selected],
        [q['mean_gain_per_100k'] - 1.96 * q['MCSE_per_100k'] for q in selected],
        [q['mean_gain_per_100k'] + 1.96 * q['MCSE_per_100k'] for q in selected], color=color, alpha=.15)
ax.axhline(0, color='#222', lw=.9); ax.set_xscale('log', base=2)
ax.set_xticks([32, 64, 128, 256, 512, 1024, 2048, 4096]); ax.xaxis.set_major_formatter(ScalarFormatter())
ax.set_xlabel('Customers used to calibrate the new score')
ax.set_ylabel('Mean effect, currency units\nper 100,000 applicants'); ax.yaxis.set_major_formatter(fmt)
ax.set_title('Reference-data quality changes the pilot size required', loc='left', fontsize=12, fontweight='bold')
ax.legend(frameon=False, loc='lower right')
save(fig, 'reference_quality')
