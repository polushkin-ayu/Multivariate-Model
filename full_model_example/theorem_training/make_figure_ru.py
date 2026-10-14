"""Render manuscript-ready Russian two-panel theorem-validation figure."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
summary = json.loads((ROOT / "summary.json").read_text())
raw = np.load(ROOT / "paired_gains.npz")
C_s2 = summary["C_gate"] * summary["s"] ** 2
colors = {2000: "#2878B5", 20000: "#E1812C", 100000: "#3A923A"}
markers = {0.5: "o", 2.0: "s"}

fig, (ax, residual_ax) = plt.subplots(1, 2, figsize=(12.2, 5.3), constrained_layout=True)
for case in summary["cases"]:
    n, ratio = case["N"], case["h_over_s"]
    prefix = f"N{n}_h_over_s{ratio:g}"
    predicted = raw[prefix + "_predicted"] / C_s2
    exact = raw[prefix + "_exact"] / C_s2
    ax.scatter(predicted, exact, s=18, alpha=.52, color=colors[n],
               marker=markers[ratio], edgecolors="none")
all_x = np.concatenate([raw[f"N{n}_h_over_s{r:g}_predicted"] / C_s2
                        for n in (2000, 20000, 100000) for r in (.5, 2.)])
all_y = np.concatenate([raw[f"N{n}_h_over_s{r:g}_exact"] / C_s2
                        for n in (2000, 20000, 100000) for r in (.5, 2.)])
lo, hi = min(all_x.min(), all_y.min()), max(all_x.max(), all_y.max())
pad = .04 * max(hi - lo, 1.0)
ax.plot([lo-pad, hi+pad], [lo-pad, hi+pad], color="black", ls="--", lw=1.2)
ax.set(xlabel="Предсказание теоремы: $C(2hT-T^2)/(Cs^2)$",
       ylabel="Полная модель: $N\\,\\Delta\\Pi/(Cs^2)$",
       title="Парные эффекты")
ax.grid(alpha=.2)

for ratio, marker in markers.items():
    cases = [case for case in summary["cases"] if case["h_over_s"] == ratio]
    residual_ax.plot([case["N"] for case in cases],
                     [case["mean_absolute_residual_normalized_Cs2"] for case in cases],
                     marker=marker, color={0.5: "#555555", 2.0: "#111111"}[ratio],
                     lw=1.8, label=f"$h/s={ratio:g}$".replace(".", ","))
residual_ax.set(xscale="log", yscale="log", xlabel="Размер калибровки $N$",
                ylabel="Среднее $|N\\Delta\\Pi-C(2hT-T^2)|/(Cs^2)$",
                title="Расхождение с локальным приближением")
residual_ax.grid(alpha=.23, which="both")

from matplotlib.lines import Line2D
legend = [Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[n],
                 markeredgecolor="none", label=f"N={n:,}".replace(",", " "), markersize=7)
          for n in (2000, 20000, 100000)]
legend += [Line2D([0], [0], marker=markers[r], color="#555555", label=f"h/s={r:g}".replace(".", ","),
                  linestyle="none", markersize=6) for r in (.5, 2.)]
fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False,
           bbox_to_anchor=(.5, 1.035))
fig.savefig(ROOT / "theorem_validation_ru.png", dpi=200, bbox_inches="tight")
fig.savefig(ROOT / "theorem_validation_ru.pdf", bbox_inches="tight")
plt.close(fig)
