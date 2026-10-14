"""Fixed-grid low-order sensitivity screen; see protocol.md."""
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
from structural_weak_score import WeakBoundaryEngine
from structural_weak_score import predict
from cac_decision import BoundaryEngine

OUT = Path(__file__).resolve().parent
TAU = 0.25
REPS = 100_000
SEED = 2026101365
TAU_GRID = np.linspace(-1.0, 1.0, 41)
DA_GRID = (1, 4, 16)
PRICE_GRID = (25., 75., 150.)
GATE = (-3.200874778243636, 2.7545876756152166, 3.0)
N_GRID = (64, 512, 4096)
DH_GRID = (1, 4, 16)


def make_model(deployment_noise_multiplier):
    b = base_model()
    d = b.noise.copy()
    d[1, 1] *= deployment_noise_multiplier
    d[2, 2] *= deployment_noise_multiplier
    return GaussianCAC(b.mu.copy(), b.sigma.copy(), d, b.cash)


def baseline_profit(m, price):
    e = BoundaryEngine(m, (1, 2), price, outer_order=10, inner_order=14)
    nodes, weights = leggauss(10)
    top = np.maximum(e.root, e.conditional_mean + 12 * e.conditional_sd)
    v = e.root[:, None] + (top - e.root)[:, None] * (nodes + 1) / 2
    w = np.repeat(e.w, 10, axis=0)
    w[:, e.vcol] = v.ravel()
    payoff = m.predict(w, (1, 2), 14).reshape(v.shape) - price
    density = norm.pdf((v - e.conditional_mean[:, None]) / e.conditional_sd) / e.conditional_sd
    return float(e.weights @ (np.sum(payoff * density * weights, axis=1) * (top - e.root) / 2))


def fitted_response(engine, m, fitted_tau, base_root, price):
    candidate = engine.root(m, float(fitted_tau), 1.0)
    nodes, weights = leggauss(10)
    v = candidate[:, None] + (base_root - candidate)[:, None] * (nodes + 1) / 2
    w = np.column_stack([np.repeat(engine.nuisance[:, 0], 10), v.ravel()])
    x = np.repeat(engine.nuisance[:, 1], 10)
    payoff = predict(m, w, x, TAU, engine.order).reshape(v.shape) - price
    density = norm.pdf((v - engine.vmean[:, None]) / engine.vsd) / engine.vsd
    integral = np.sum(payoff * density * weights, axis=1) * (base_root - candidate) / 2
    return float(engine.weights @ integral)


def exact_tau_draws(n, dh, reps, rng, m):
    # Exact Bartlett representation for sample covariance of (X, mean H refs).
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
    draws = {}
    base = base_model()
    for dh in DH_GRID:
        for n in N_GRID:
            raw, clipped = exact_tau_draws(n, dh, REPS, rng, base)
            draws[(n, dh)] = (raw, clipped)
    rows = []
    for da in DA_GRID:
        m = make_model(da)
        for price in PRICE_GRID:
            tag = f'DAV{da}_price{int(price)}'
            baseline = baseline_profit(m, price)
            curve_path = OUT / f'{tag}_curve.json'
            curve = json.loads(curve_path.read_text()) if curve_path.exists() else []
            engine = WeakBoundaryEngine(m, TAU, price, outer_order=10, inner_order=14, gate=GATE)
            base_root = engine.root(m, 0.0, 0.0)
            failures = []
            for j, t in enumerate(TAU_GRID):
                if j < len(curve):
                    if abs(curve[j]['fitted_tau'] - t) > 1e-12:
                        raise ValueError('checkpoint grid mismatch')
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
                if j % 10 == 0:
                    print(f'{tag}: tau {j+1}/41; elapsed {time.time()-started:.1f}s', flush=True)
            valid = np.array([q['gain'] is not None for q in curve], dtype=bool)
            # Do not bridge certification failures; the complete scenario is marked and omitted from expectation.
            if valid.all():
                interp = PchipInterpolator(TAU_GRID, [q['gain'] for q in curve])
                for n in N_GRID:
                    for dh in DH_GRID:
                        raw, th = draws[(n, dh)]
                        increments = interp(th)
                        mean = float(increments.mean())
                        mcse = float(increments.std(ddof=1) / np.sqrt(REPS))
                        rows.append({
                            'D_A_multiplier': da, 'D_V_multiplier': da, 'price': price, 'true_tau': TAU,
                            'N': n, 'D_H_reference_noise_multiplier': dh,
                            'replications': REPS, 'mean_delta_per_100000': 100000*mean,
                            'MCSE_delta_per_100000': 100000*mcse,
                            'relative_mean_profit_percent': 100*mean/baseline,
                            'baseline_profit_per_100000': 100000*baseline,
                            'clipping_fraction': float(np.mean(np.abs(raw) > 1.0)),
                            'certification_failure': False,
                            'certification_failure_points': []})
            else:
                for n in N_GRID:
                    for dh in DH_GRID:
                        raw, _ = draws[(n, dh)]
                        rows.append({
                            'D_A_multiplier': da, 'D_V_multiplier': da, 'price': price, 'true_tau': TAU,
                            'N': n, 'D_H_reference_noise_multiplier': dh,
                            'replications': REPS, 'mean_delta_per_100000': None,
                            'MCSE_delta_per_100000': None,
                            'relative_mean_profit_percent': None,
                            'baseline_profit_per_100000': 100000*baseline,
                            'clipping_fraction': float(np.mean(np.abs(raw) > 1.0)),
                            'certification_failure': True,
                            'certification_failure_points': failures})
            print(f'{tag}: baseline={baseline*100000:.4f}; failures={len(failures)}; elapsed={time.time()-started:.1f}s', flush=True)
            (OUT / 'results.json').write_text(json.dumps({
                'protocol': 'protocol.md', 'seed': SEED,
                'screening_only': True, 'grid_fixed': True,
                'rows': rows, 'elapsed_seconds': time.time()-started
            }, indent=2) + '\n')
    print(json.dumps({'rows': len(rows), 'elapsed_seconds': time.time()-started,
                      'output': str(OUT/'results.json')}, indent=2), flush=True)

if __name__ == '__main__':
    main()
