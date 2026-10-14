"""Deterministic pathwise check of Theorem 1's leading term.

This is numerical integration over the full cash model, not a simulation over
calibration samples and not validation of the theorem's expectation claim.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from cac_decision import pack, regularity_certificate, unpack  # noqa: E402
from structural_weak_score import WeakBoundaryEngine, boundary_constants  # noqa: E402
from verify_structural_weak_score import model  # noqa: E402

PRICE = 25.0
GATE = (-3.200874778243636, 2.7545876756152166, 3.0)
N_VALUES = [1000, 3000, 10000, 30000, 100000]
RATIOS = [0.5, 2.0]
ZS = [-2, -1, 0, 1, 2]
U = np.array([.2, -.1, .15, .03, .01, -.01, .02, .01, -.02, .1, .05, -.03])


def main():
    start = time.time()
    m = model()
    cert = regularity_certificate(m, (1, 2))
    if not cert['certified_for_positive_prices']:
        raise RuntimeError(f'baseline regularity not certified: {cert}')
    s = float(np.sqrt((m.sigma[0, 0] + m.noise[0, 0] / 2) / m.sigma[0, 0]**2))
    coarse = boundary_constants(m, price=PRICE, outer_order=32, inner_order=32,
                                gate=GATE)['C']
    refined = boundary_constants(m, price=PRICE, outer_order=44, inner_order=40,
                                 gate=GATE)['C']
    C = float(refined)
    result = {
        'scope': 'deterministic numerical trajectory check; not random calibrated-sample mean validation',
        'model': {'source': 'verify_structural_weak_score.model()', 'price': PRICE,
                  'fixed_gate': GATE,
                  'regularity_certificate': cert},
        'design': {'N': N_VALUES, 'h_over_s': RATIOS, 'z': ZS,
                   'tau_true': 'h/sqrt(N)', 'tauhat': 'T/sqrt(N)',
                   'T': 'h+s*z', 'theta_hat': 'pack(theta)+U/sqrt(N)',
                   'U': U.tolist(), 'weight': 1.0},
        'constants': {'s': s, 'C': C, 'C_coarse': float(coarse),
                      'C_refinement_abs_difference': abs(C-float(coarse)),
                      'h_over_s': RATIOS,
                      'normalized_target': 'N_gain/(C*s^2) -> 2*(h/s)*(T/s)-(T/s)^2'},
        'paths': [], 'quadrature_checks': [], 'root_failures': [],
    }
    total = len(N_VALUES)*len(RATIOS)*len(ZS)
    count = 0
    for ratio in RATIOS:
        h = ratio*s
        for z in ZS:
            T = h+s*z
            leading_norm = 2*(h/s)*(T/s)-(T/s)**2
            for N in N_VALUES:
                count += 1
                try:
                    truth = m
                    tau_true = h/np.sqrt(N)
                    tau_hat = T/np.sqrt(N)
                    fitted = unpack(pack(m)+U/np.sqrt(N), m.cash)
                    engine = WeakBoundaryEngine(truth, tau_true, PRICE,
                                                outer_order=22, inner_order=28, gate=GATE)
                    gain = engine.gain(fitted, tau_hat, 1.0, legendre_order=14)
                    exact_norm = N*gain/(C*s*s)
                    row = {'N': N, 'h_over_s': ratio, 'z': z,
                           'T_over_s': T/s, 'N_gain': N*gain,
                           'exact_normalized': float(exact_norm),
                           'leading_normalized': float(leading_norm),
                           'normalized_error': float(exact_norm-leading_norm)}
                    result['paths'].append(row)
                    if N == N_VALUES[-1] and z in (-2, 0, 2):
                        # Refine all integration dimensions on selected limiting cases.
                        fine = WeakBoundaryEngine(truth, tau_true, PRICE,
                                                  outer_order=30, inner_order=36, gate=GATE)
                        fine_gain = fine.gain(fitted, tau_hat, 1.0, legendre_order=22)
                        result['quadrature_checks'].append({
                            'N': N, 'h_over_s': ratio, 'z': z,
                            'main_N_gain': N*gain, 'refined_N_gain': N*fine_gain,
                            'abs_difference_N_gain': abs(N*(fine_gain-gain)),
                            'refined_normalized': N*fine_gain/(C*s*s),
                        })
                except Exception as e:
                    result['root_failures'].append({'N': N, 'h_over_s': ratio,
                                                    'z': z, 'error': repr(e)})
                print(f'path {count}/{total} h/s={ratio:g} z={z:+d} N={N}: '
                      f'{"ok" if len(result["paths"]) and result["paths"][-1].get("N")==N and result["paths"][-1].get("z")==z and result["paths"][-1].get("h_over_s")==ratio else "FAILED"}',
                      flush=True)

    if result['root_failures']:
        print('Failed cases:', result['root_failures'], flush=True)
    result['summary'] = {
        'successful_paths': len(result['paths']), 'expected_paths': total,
        'max_abs_normalized_error_by_N': {
            str(N): max(abs(r['normalized_error']) for r in result['paths'] if r['N'] == N)
            for N in N_VALUES
        },
        'largest_N_error_by_h_over_s': {
            str(ratio): max(abs(r['normalized_error']) for r in result['paths']
                            if r['N'] == N_VALUES[-1] and r['h_over_s'] == ratio)
            for ratio in RATIOS
        },
        'largest_quadrature_abs_difference_N_gain': max(
            (q['abs_difference_N_gain'] for q in result['quadrature_checks']), default=None),
        'elapsed_seconds': time.time()-start,
    }
    (OUT/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['summary'], indent=2), flush=True)
    if result['root_failures'] or len(result['paths']) != total:
        raise RuntimeError('one or more trajectory integrations failed; see results.json')


if __name__ == '__main__':
    main()
