# fast-ap-solver

Fast, differentiable, fixed-step JAX solvers for the 2D Aliev-Panfilov system (explicit RK in a `lax.scan`, and
exponential/Lawson/ETD schemes in the DCT eigenbasis), with the benchmarks, tests and results that compare them with the
production diffrax solver (`F02`). Full evaluation: [REPORT.md](REPORT.md).

## Takeaway

The backward pass was slow for two removable reasons:

1. **diffrax overhead (~6x).** The same Tsit5 scheme at dt 0.02 in a plain `lax.scan` with per-step
   `jax.checkpoint` reproduces F02 to round-off and is 5.9x faster per VJP (83.5 vs 488 ms).
2. **Explicit diffusion forces dt ≈ 0.02 (~8x more).** The largest Laplacian eigenvalue is 8D/dx² ≈ 99, so explicit
   Tsit5 is unstable above dt ≈ 0.035. The edge-replicate (Neumann) 5-point Laplacian is diagonalised exactly by the DCT-II,
   and exp(hD∇²) is separable: two 224×224 matmuls per application. Integrating-factor RK4 (Lawson4) then runs at
   dt 0.2 (50 steps instead of 500).

| use | recommended | speedup vs F02 | evidence of equivalence |
|---|---|---|---|
| IC fitting / DPS guidance | `lawson4_d0.2` (unroll 4) | **≈50x per VJP**; ≈6 ms/case at B = 16 (81x vs F02 B=1, 42x vs F02's best batch) | 16/16 equal-VJP Adam fits identical to 3 decimals; rollouts to τ 30 overlap |
| exact F02 behaviour | `tsit5_d0.02` in scan | ≈6x | same scheme, matches to round-off |
| parameter fitting | `rk4_d0.025`, or lawson4_d0.2 then rk4 to polish | ≈7.5x (hybrid: most iterations at ≈30x) | θ-gradients match REF to ~1e-6 |

## How to run

The code is a verbatim copy of files from [cardiac-guided-diffusion](https://github.com/Alx-Ho/cardiac-guided-diffusion)
and is not a standalone package: it imports `cardiac.*` siblings relatively. To run it, copy the files under `cardiac/` of
this directory into the `cardiac/` package (and `cardiac/tests/`) of a cardiac-guided-diffusion checkout, and run from that
checkout's root. It needs `ap_simulator` via `AP_SIMULATOR_PATH` (e.g. `/home/alex/diff_sim_latent/ap_simulator`).
The commands from REPORT.md:

```
cd /home/alex/cardiac-guided-diffusion
export AP_SIMULATOR_PATH=/home/alex/diff_sim_latent/ap_simulator MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false
python -m cardiac.bench_fastsim --phase accuracy --out cardiac/runs/fastsim_gpu_acc --reps 5 --force
python -m cardiac.bench_fastsim --phase timing --from-accuracy cardiac/runs/fastsim_gpu_acc --out cardiac/runs/fastsim_gpu \
  --remat step,none,segment --batches 4,16,32 --reps 5 --force
python -m cardiac.fastsim_fit_compare --force --no-render && JAX_PLATFORMS=cpu python -m cardiac.fastsim_fit_compare --plot-only
python -m cardiac.fastsim_param_check all --force
# tests (CPU): JAX_PLATFORMS=cpu python -m pytest cardiac/tests -q -m "not slow"   -> 332 passed
```

## Layout

```
cardiac/fastsim.py                  the solvers
cardiac/bench_fastsim.py            accuracy / timing benchmark vs diffrax
cardiac/fastsim_fit_compare.py      equal-VJP Adam IC fits, F02 vs lawson4_d0.2
cardiac/fastsim_param_check.py      parameter gradients / recovery
cardiac/tests/test_fastsim*.py      tests (test_fastsim.py, test_fastsim_fit_compare.py, test_fastsim_params.py)
REPORT.md                           evaluation report
results/                            final outputs (no caches): solver_accuracy/, solver_timing/, fit_compare/, param_check/
docs/plans/                         the two plan files these results came from
LICENSE                             GPL-3.0 (same as cardiac-guided-diffusion)
```

Note: image and plan links inside REPORT.md and the plan files still use the original paths of the source repo
(e.g. `../fastsim-fit-compare-media/`); the same files are under `results/` here.
