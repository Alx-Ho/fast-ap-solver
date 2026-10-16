# Faster differentiable Aliev–Panfilov solver: evaluation report

Date: 2026-10-16. Hardware: RTX A6000 (training job paused during the GPU runs). Data: chaotic dataset, n = 224,
grid [-1, 1]², θ* = (D 0.001, eps0 0.05, a 0.09, k 8.2, mu1 0.07, mu2 0.3). Task: recover the state at frame 128
from noisy u at τ = 1..10.

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

## 1. Solver accuracy and cost (GPU, eval_val4, 4 cases × 4 points)

Errors vs REF (diffrax Tsit5 rtol = atol = 1e-6, dtmax 0.0025); acceptance as in `solver_sweep` (U ≤ 0.002, V ≤ 0.005,
grad cos ≥ 0.99, rel ≤ 0.1). Full table: `accuracy_summary.md`; plots `error_vs_cost.png`, `convergence.png`.

| setting | dt | steps | ms/VJP | speedup | max U RMSE | min grad cos | accepted |
|---|---|---|---|---|---|---|---|
| F02 (diffrax) | 0.02 | 500 | 486 | 1x | 3.7e-7 | 1.00000 | yes |
| tsit5 (scan) | 0.02 | 500 | 84 | 5.8x | 3.8e-7 | 1.00000 | yes |
| rk4 | 0.025 | 400 | 44 | 11x | 9.4e-6 | 0.99999 | yes |
| etdrk4 | 0.2 | 50 | 19 | 26x | 2.1e-4 | 0.99971 | yes |
| strang4 | 0.05 | 200 | 15 | 33x | 3.2e-4 | 0.99925 | yes |
| lawson4 | 0.1 | 100 | 20 | 24x | 1.2e-4 | 0.99999 | yes |
| lawson4 | 0.2 | 50 | 10 | 48x | 5.2e-4 | 0.99996 | yes |

Rejected: euler/heun at dt 0.02 (at the stability limit, blow up), imex_euler (first order), strang2/etd2 at dt ≥ 0.05
(second order, too inaccurate).

## 2. Timing variants (GPU; `timing_summary.md`, `timing_gpu.png`, `throughput.png`)

- **Backward strategy:** `jax.checkpoint` per step is best: VJP/fwd ≈ 3. Without remat it is 13–21 and uses up to 8 GB;
  segment remat sits in between. A hand-written adjoint would add little.
- **Diffusion operator:** separable exp (2 matmuls) beats the eigenbasis (4 matmuls) by 1.8x and `jax.scipy.fft.dct`
  by 2.7x. All matmuls use `precision=HIGHEST` (TF32 would break accuracy).
- **Unroll 4:** lawson4_d0.2 goes from 10.6 to 9.4 ms/VJP (52x).
- **Batching:** lawson4_d0.2 costs 7.6 / 6.0 / 6.0 ms per case at B = 4 / 16 / 32, peak 1.2 GB at B = 32. F02 costs
  258 ms/case at B = 4 and runs out of memory at B ≥ 16.

## 3. End-to-end check: equal-VJP Adam IC fits (16 headline test cases)

Identical y, init, Adam (lr 0.01, 200 iterations, early stop off); only the forward differs. All rollouts use the same A01
judge to τ 30. Media: `../fastsim-fit-compare-media/fit_compare_*` (figure, summary, montage + per-case videos);
W&B runs `fastsim_fit_compare`, `fastsim_fit_compare_plots`.

| | F02 | lawson4_d0.2 |
|---|---|---|
| wall per fit (excl. compile) | 97.0 s | 1.9 s (51.7x) |
| u0 / v0 RMSE (mean) | 0.158 / 0.2598 | 0.158 / 0.2599 |
| rollout u / v RMSE at τ 30 | 0.0564 / 0.0512 | 0.0563 / 0.0511 |
| final obs loss | — | within 0.05% of F02 in every case |
| failed steps | 0 | 0 |

## 4. Parameter gradients and recovery (`../fastsim-fit-compare-media/param_*`)

- `fastsim.make_forward_fast_p(params, u0, v0)` keeps all 6 parameters traced. CPU finite-difference and JVP checks pass
  for every parameter (worst relative error 2e-6). ap_simulator's `BoundedLogRDParams` drops in unchanged.
- **Away from the optimum** (perturbed θ, joint point), every scheme's 6-vector gradient matches REF (1 − cos ≤ 1e-8).
- **At the true θ**, where the true gradient is small and noise-driven, solver bias dominates. 1 − cos is: F02, tsit5-scan,
  rk4 ≤ 2e-6; lawson4_d0.05 0.024; lawson4_d0.1 and etdrk4_d0.1 1.2–1.6; lawson4/etdrk4 at dt 0.2 and strang4 ≈ 1.9.
  So a converged θ-fit with dt ≥ 0.1 lands on a slightly different optimum than F02.
- **Recovery** (true IC known; 2 cases × 3 seeds; ±20–40% θ starts; decayed Adam lr 0.02; 200 iterations): F02, lawson4_d0.1
  and lawson4_d0.2 are indistinguishable (mean θ error 12.1% each, loss ratio 1.000–1.001), at 14x / 25x speedup. No fit
  converged in 200 iterations, so the near-optimum bias was not exercised.

## Not implemented, and when it would be worth it

- **3D (`RDSystem3D`).** Next step. Separable exp extends to 3 axes; the user's training job is 3D.
- **Wiring into `physics.py` / `SolverCfg`.** A one-line dispatch on the factory, once a scheme is adopted.
- **Converged parameter fits** (more iterations or a higher lr) to measure the lawson4 bias on recovered θ, and the hybrid schedule.
- **Pallas fused multi-step kernels.** The state (400 KB) exceeds SM shared memory and needs halo tiling. Revisit if
  per-step launch overhead dominates after batching.
- **Custom adjoint.** Unnecessary: remat-per-step VJP/fwd ≈ 3.
- **Coarse-grid gradients, bf16, parareal.** These change the math; risky for chaotic dynamics at these thresholds.

## Reproduce

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
`--force` overrides the guard that refuses to use the GPU while training PID 1657544 exists. Pause that job first
(`kill -STOP`), and always resume it afterwards (`kill -CONT`).

Code (uncommitted): `cardiac/fastsim.py`, `cardiac/bench_fastsim.py`, `cardiac/fastsim_fit_compare.py`,
`cardiac/fastsim_param_check.py`, and tests `cardiac/tests/test_fastsim*.py`. Plans: `../fast-ap-solver.md`,
`../fastsim-fit-compare.md`.
