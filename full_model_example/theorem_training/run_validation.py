"""Finite-sample check of manuscript Theorem 1 under joint calibration."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
sys.path.insert(0, str(PROJECT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from structural_weak_score import fit, boundary_constants, WeakBoundaryEngine
from verify_structural_weak_score import model, sample


GATE = (-3.200874778243636, 2.7545876756152166, 3.0)
SEED = 2026101417
REPS = 128
OUTER, INNER = 14, 20
REFINED_OUTER, REFINED_INNER = 24, 32


def main():
    start = time.monotonic()
    rng = np.random.default_rng(SEED)
    truth = model()
    const = boundary_constants(truth, price=25.0, outer_order=32,
                               inner_order=32, gate=GATE)
    c_refined = boundary_constants(truth, price=25.0, outer_order=44,
                                   inner_order=40, gate=GATE)["C"]
    C = const["C"]
    s = np.sqrt((truth.sigma[0, 0] + truth.noise[0, 0] / 2)
                / truth.sigma[0, 0] ** 2)
    cases = []
    raw = {}
    for n in (2000, 20000, 100000):
        for ratio in (0.5, 2.0):
            h = ratio * s
            engine = WeakBoundaryEngine(truth, h / np.sqrt(n), price=25.0,
                                        outer_order=OUTER, inner_order=INNER,
                                        gate=GATE)
            exact, predicted, fitted_tau = [], [], []
            stabilizations = {"compact_model": 0, "tau_clip": 0,
                              "covariance_repair": 0}
            refinements = []
            case_start = time.monotonic()
            for rep in range(REPS):
                refs, x = sample(truth, n, h / np.sqrt(n), rng)
                fitted, tau, _, diag = fit(refs, x, truth.cash)
                stabilizations["compact_model"] += int(diag["compact_stabilization_active"])
                stabilizations["tau_clip"] += int(diag["tau_clipped"])
                stabilizations["covariance_repair"] += int(diag.get("sigma_repair_frobenius", 0) > 1e-6)
                T = np.sqrt(n) * tau
                base_root = engine.root(fitted, 0.0, 0.0)
                gain = n * engine.gain(fitted, tau, 1.0, base_root)
                pred = C * (2.0 * h * T - T * T)
                exact.append(gain)
                predicted.append(pred)
                fitted_tau.append(T)
                # Refine selected cases: first replication of every (N,h/s) cell.
                if rep == 0:
                    finer = WeakBoundaryEngine(truth, h / np.sqrt(n), price=25.0,
                                               outer_order=REFINED_OUTER,
                                               inner_order=REFINED_INNER, gate=GATE)
                    fine_gain = n * finer.gain(fitted, tau, 1.0)
                    refinements.append({"replication": rep, "coarse_exact": gain,
                                        "refined_exact": float(fine_gain),
                                        "absolute_difference": abs(float(fine_gain) - gain)})
            exact = np.asarray(exact)
            predicted = np.asarray(predicted)
            difference = exact - predicted
            normalized_abs_residual = np.abs(difference) / (C * s * s)
            limit = ratio * ratio - 1.0
            case = {
                "N": n, "h_over_s": ratio, "h": h, "replications": REPS,
                "mean_exact_scaled_gain": float(exact.mean()),
                "mean_theorem_prediction": float(predicted.mean()),
                "mean_paired_difference": float(difference.mean()),
                "paired_difference_SE": float(difference.std(ddof=1) / np.sqrt(REPS)),
                "mean_absolute_residual_normalized_Cs2": float(normalized_abs_residual.mean()),
                "median_absolute_residual_normalized_Cs2": float(np.median(normalized_abs_residual)),
                "mean_exact_normalized_Cs2": float(exact.mean() / (C * s * s)),
                "MC_SE_exact_normalized_Cs2": float(exact.std(ddof=1) / np.sqrt(REPS) / (C * s * s)),
                "mean_prediction_normalized_Cs2": float(predicted.mean() / (C * s * s)),
                "MC_SE_prediction_normalized_Cs2": float(predicted.std(ddof=1) / np.sqrt(REPS) / (C * s * s)),
                "mean_paired_difference_normalized_Cs2": float(difference.mean() / (C * s * s)),
                "paired_difference_SE_normalized_Cs2": float(difference.std(ddof=1) / np.sqrt(REPS) / (C * s * s)),
                "theorem_normalized_mean_limit": limit,
                "stabilizations": stabilizations,
                "refinements": refinements,
                "elapsed_seconds": time.monotonic() - case_start,
            }
            cases.append(case)
            raw[f"N{n}_h_over_s{ratio:g}_exact"] = exact
            raw[f"N{n}_h_over_s{ratio:g}_predicted"] = predicted
            raw[f"N{n}_h_over_s{ratio:g}_difference"] = difference
            raw[f"N{n}_h_over_s{ratio:g}_T"] = np.asarray(fitted_tau)
            np.savez_compressed(ROOT / "paired_gains.npz", **raw)
            checkpoint = {
                "experiment": "joint calibration theorem validation; synthetic original CAC model",
                "seed": SEED, "replications_per_case": REPS,
                "N_values": [2000, 20000, 100000], "h_over_s_values": [0.5, 2.0],
                "CAC": 25.0, "gate": GATE, "C_gate": C,
                "C_gate_refined_44x40": c_refined,
                "C_gate_quadrature_absolute_difference": abs(C - c_refined),
                "s": float(s), "engine_quadrature": {"outer": OUTER, "inner": INNER},
                "selected_case_refinement_quadrature": {"outer": REFINED_OUTER, "inner": REFINED_INNER},
                "elapsed_seconds": time.monotonic() - start, "cases": cases,
            }
            (ROOT / "summary.json").write_text(json.dumps(checkpoint, indent=2) + "\n")
            print(json.dumps(case), flush=True)

    np.savez_compressed(ROOT / "paired_gains.npz", **raw)
    summary = {
        "experiment": "joint calibration theorem validation; synthetic original CAC model",
        "seed": SEED, "replications_per_case": REPS,
        "N_values": [2000, 20000, 100000], "h_over_s_values": [0.5, 2.0],
        "CAC": 25.0, "gate": GATE, "C_gate": C,
        "C_gate_refined_44x40": c_refined,
        "C_gate_quadrature_absolute_difference": abs(C - c_refined),
        "s": float(s), "engine_quadrature": {"outer": OUTER, "inner": INNER},
        "selected_case_refinement_quadrature": {"outer": REFINED_OUTER, "inner": REFINED_INNER},
        "elapsed_seconds": time.monotonic() - start, "cases": cases,
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    colors = {2000: "#1f77b4", 20000: "#ff7f0e", 100000: "#2ca02c"}
    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)
    for case in cases:
        key = f"N{case['N']}_h_over_s{case['h_over_s']:g}"
        ax.scatter(raw[key + "_predicted"], raw[key + "_exact"], s=17,
                   alpha=0.48, color=colors[case["N"]],
                   label=f"N={case['N']}" if case["h_over_s"] == 0.5 else None)
    all_values = np.concatenate([raw[k] for k in raw if k.endswith("_exact") or k.endswith("_predicted")])
    low, high = float(all_values.min()), float(all_values.max())
    pad = max((high - low) * 0.04, 0.05)
    ax.plot([low-pad, high+pad], [low-pad, high+pad], color="black", lw=1, ls="--")
    ax.set(xlabel="Theorem prediction  C(2hT − T²)", ylabel="Exact scaled gain  N × gain",
           title="Paired replication gains: exact vs theorem")
    ax.legend(title="Calibration size", frameon=False)
    ax.grid(alpha=0.2)
    fig.savefig(ROOT / "exact_vs_theorem.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for ratio, marker in ((0.5, "o"), (2.0, "s")):
        selected = [case for case in cases if case["h_over_s"] == ratio]
        ax.plot([case["N"] for case in selected],
                [case["mean_absolute_residual_normalized_Cs2"] for case in selected],
                marker=marker, lw=1.8, label=f"h/s={ratio:g}")
    ax.set(xscale="log", xlabel="Calibration size N",
           ylabel="Mean |exact − prediction| / (C s²)",
           title="Normalized absolute theorem residual")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False)
    fig.savefig(ROOT / "normalized_residual_vs_N.png", dpi=180)
    plt.close(fig)
    print(f"Saved outputs under {ROOT}; total seconds={summary['elapsed_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
