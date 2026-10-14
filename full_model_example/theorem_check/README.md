# Direct numerical check of Theorem 1

Run `python3 full_model_example/theorem_check/run.py` from the project root to
recompute `results.json`, then run
`MPLBACKEND=Agg MPLCONFIGDIR=.mplconfig python3 full_model_example/theorem_check/plot.py` to write `convergence.png`
and `convergence.pdf`. The calculation reuses the original full cash model at
price 25 from `verify_structural_weak_score.py`, with the fixed gate from
`verify_gated_weak_score.py`.

For each `h/s` in `{0.5, 2}`, each `z` in `{-2,-1,0,1,2}`, and each
`N` in `{1000,3000,10000,30000,100000}`, the script uses the deterministic
path `tau_true=h/sqrt(N)`, `tauhat=T/sqrt(N)`, `T=h+s*z`, and the shared fitted
parameter perturbation `pack(theta_hat)=pack(theta)+U/sqrt(N)`. It numerically
integrates the policy profit difference under the true model and compares
`N*gain/(C*s^2)` with `2*(h/s)*(T/s)-(T/s)^2`. Here
`s=sqrt((sigma_HH+noise_HH/2)/sigma_HH^2)` and `C` is the gated coefficient
from `boundary_constants`.

The quadrature refinement selected cases increase the gated Legendre orders,
inner cash order, and root-interval Legendre order. Every trajectory failure is retained
in `root_failures` and causes a nonzero program exit; none occurred in the
recorded run. `results.json` contains all 50 path values and all refinement
checks.

This is a deterministic numerical trajectory check, not a Monte Carlo
calibrated-sample experiment. It illustrates convergence for the specified
shared perturbation and finite set of `z` values. It does not estimate the
expectation over fitted models in the theorem or independently establish the
theorem's uniform-integrability claim.
