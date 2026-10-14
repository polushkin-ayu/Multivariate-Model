"""Higher-order whole-grid validation for the fixed sensitivity screen."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import json, time
from pathlib import Path
import sys
import numpy as np
from scipy.stats import norm
from scipy.interpolate import PchipInterpolator
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from verify_structural_weak_score import model as base_model
from cac_calibration import GaussianCAC
from structural_weak_score import WeakBoundaryEngine, predict
from cac_decision import BoundaryEngine

OUT = Path(__file__).resolve().parent
TAU = 0.25
REPS = 100_000
SEED = 2026101365
TAU_GRID = np.linspace(-1.0, 1.0, 81)
DA_GRID = (1, 4, 16)
PRICE_GRID = (25., 75., 150.)
GATE = (-3.200874778243636, 2.7545876756152166, 3.0)
N_GRID = (64, 512, 4096)
DH_GRID = (1, 4, 16)


def make_model(multiplier):
    b = base_model()
    d = b.noise.copy()
    d[1, 1] *= multiplier
    d[2, 2] *= multiplier
    return GaussianCAC(b.mu.copy(), b.sigma.copy(), d, b.cash)


def baseline_profit(m, price):
    e = BoundaryEngine(m, (1, 2), price, outer_order=40, inner_order=28)
    nodes, weights = leggauss(72)
    top = np.maximum(e.root, e.conditional_mean + 12 * e.conditional_sd)
    v = e.root[:, None] + (top - e.root)[:, None] * (nodes + 1) / 2
    w = np.repeat(e.w, 72, axis=0)
    w[:, e.vcol] = v.ravel()
    payoff = m.predict(w, (1, 2), 28).reshape(v.shape) - price
    density = norm.pdf((v - e.conditional_mean[:, None]) / e.conditional_sd) / e.conditional_sd
    profit = float(e.weights @ (np.sum(payoff * density * weights, axis=1) * (top - e.root) / 2))
    accept = float(e.weights @ norm.sf((e.root - e.conditional_mean) / e.conditional_sd))
    return profit, accept


def fitted_response(engine, m, fitted_tau, base_root, price):
    candidate = engine.root(m, float(fitted_tau), 1.0)
    nodes, weights = leggauss(16)
    v = candidate[:, None] + (base_root - candidate)[:, None] * (nodes + 1) / 2
    w = np.column_stack([np.repeat(engine.nuisance[:, 0], 16), v.ravel()])
    x = np.repeat(engine.nuisance[:, 1], 16)
    payoff = predict(m, w, x, TAU, engine.order).reshape(v.shape) - price
    density = norm.pdf((v - engine.vmean[:, None]) / engine.vsd) / engine.vsd
    integral = np.sum(payoff * density * weights, axis=1) * (base_root - candidate) / 2
    return float(engine.weights @ integral)


def exact_tau_draws(n, dh, reps, rng, m):
    df = n - 1
    vh = m.sigma[0, 0] + dh / 2.0
    vx = 1.0 + TAU**2 * m.sigma[0, 0]
    c = TAU * m.sigma[0, 0]
    a = rng.chisquare(df, reps)
    z = rng.normal(size=reps)
    raw = (c * a + np.sqrt(vh * vx - c * c) * np.sqrt(a) * z) / (df * m.sigma[0, 0])
    return raw, np.clip(raw, -1.0, 1.0)


def main():
    started = time.time()
    rng = np.random.default_rng(SEED)
    base = base_model()
    draws = {}
    for dh in DH_GRID:
        for n in N_GRID:
            raw, th = exact_tau_draws(n, dh, REPS, rng, base)
            draws[(n, dh)] = (raw, th)
    screen = json.loads((OUT / 'results.json').read_text())['rows']
    screen_idx = {(r['D_A_multiplier'], r['price'], r['N'], r['D_H_reference_noise_multiplier']): r
                  for r in screen}
    rows = []
    for da in DA_GRID:
        m = make_model(da)
        for price in PRICE_GRID:
            tag = f'DAV{da}_price{int(price)}'
            curve_path = OUT / f'{tag}_refined_curve.json'
            curve = json.loads(curve_path.read_text()) if curve_path.exists() else []
            engine = WeakBoundaryEngine(m, TAU, price, outer_order=14, inner_order=20, gate=GATE)
            base_root = engine.root(m, 0.0, 0.0)
            failures = []
            for j, t in enumerate(TAU_GRID):
                if j < len(curve):
                    if abs(curve[j]['fitted_tau'] - t) > 1e-12:
                        raise ValueError('refined checkpoint grid mismatch')
                    if curve[j].get('failure'):
                        failures.append({'tau': float(t), 'reason': curve[j]['failure']})
                    continue
                try:
                    gain = fitted_response(engine, m, t, base_root, price)
                    item = {'fitted_tau': float(t), 'gain': gain, 'failure': None}
                except (ValueError, ArithmeticError) as exc:
                    item = {'fitted_tau': float(t), 'gain': None, 'failure': str(exc)}
                    failures.append({'tau': float(t), 'reason': str(exc)})
                curve.append(item)
                curve_path.write_text(json.dumps(curve, indent=2) + '\n')
                if j % 20 == 0:
                    print(f'{tag}: tau {j+1}/81; elapsed {time.time()-started:.1f}s', flush=True)
            if any(q['gain'] is None for q in curve):
                interp = None
            else:
                interp = PchipInterpolator(TAU_GRID, [q['gain'] for q in curve])
            baseline, accept = baseline_profit(m, price)
            for n in N_GRID:
                for dh in DH_GRID:
                    raw, th = draws[(n, dh)]
                    if interp is None:
                        mean = mcse = relative = None
                    else:
                        increments = interp(th)
                        mean = float(increments.mean())
                        mcse = float(increments.std(ddof=1) / np.sqrt(REPS))
                        relative = 100 * mean / baseline
                    key = (da, price, n, dh)
                    old = screen_idx[key]
                    rows.append({
                        'D_A_multiplier': da, 'D_V_multiplier': da, 'price': price, 'true_tau': TAU,
                        'N': n, 'D_H_reference_noise_multiplier': dh, 'replications': REPS,
                        'mean_delta_per_100000': None if mean is None else mean * 100000,
                        'MCSE_delta_per_100000': None if mcse is None else mcse * 100000,
                        'relative_mean_profit_percent': relative,
                        'baseline_profit_per_100000': baseline * 100000,
                        'acquisition_probability': accept,
                        'clipping_fraction': float(np.mean(np.abs(raw) > 1.0)),
                        'certification_failure': interp is None,
                        'certification_failure_points': failures,
                        'screen_relative_mean_profit_percent': old['relative_mean_profit_percent'],
                        'relative_percent_point_change_vs_screen': (None if relative is None else relative - old['relative_mean_profit_percent'])
                    })
            print(f'{tag}: baseline={baseline*100000:.6f}; failures={len(failures)}; elapsed={time.time()-started:.1f}s', flush=True)
            (OUT / 'refined_results.json').write_text(json.dumps({
                'protocol': 'protocol.md', 'refinement': {'outer': 14, 'inner': 20, 'tau_grid_points': 81, 'vertical_gain': 16,
                    'baseline_outer': 40, 'baseline_inner': 28, 'baseline_vertical': 72},
                'seed': SEED, 'replications_per_N_DH': REPS, 'rows': rows,
                'elapsed_seconds': time.time()-started
            }, indent=2) + '\n')
    numeric = [r for r in rows if r['relative_percent_point_change_vs_screen'] is not None]
    diffs = np.array([abs(r['relative_percent_point_change_vs_screen']) for r in numeric])
    worst = numeric[int(np.argmax(diffs))]
    final = json.loads((OUT / 'refined_results.json').read_text())
    final['comparison'] = {
        'max_absolute_relative_profit_change_percentage_points': float(diffs.max()),
        'max_change_case': {k: worst[k] for k in ('D_A_multiplier','price','N','D_H_reference_noise_multiplier','screen_relative_mean_profit_percent','relative_mean_profit_percent','relative_percent_point_change_vs_screen')},
        'screening_rows_compared': len(numeric)
    }
    (OUT / 'refined_results.json').write_text(json.dumps(final, indent=2) + '\n')
    print(json.dumps({'rows': len(rows), 'elapsed_seconds': time.time()-started, **final['comparison']}, indent=2), flush=True)

if __name__ == '__main__':
    main()
