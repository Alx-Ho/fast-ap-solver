# Plan: fast-ap-solver (prototype + benchmark of faster differentiable Aliev-Panfilov solvers)

## Goal
Prototype fixed-step plain-JAX (`lax.scan`) forward+VJP schemes for the 2D Aliev-Panfilov system. The schemes cover two
approaches: explicit RK inside the scan, and exact/implicit diffusion in the DCT eigenbasis, which removes the dt~0.02
stability limit. Benchmark them against the production diffrax F02 solver: accuracy on CPU now, speed on GPU in a
separate step the user approves. New files only.

## Project commands
- Python: `/home/alex/conda/envs/diff/bin/python`. Run from `/home/alex/cardiac-guided-diffusion` with
  `JAX_PLATFORMS=cpu MPLBACKEND=Agg`. The implementer never uses the GPU and never signals PID 1657544.
- New tests: `JAX_PLATFORMS=cpu python -m pytest cardiac/tests/test_fastsim.py -q`
- Regression: `JAX_PLATFORMS=cpu python -m pytest cardiac/tests/test_physics.py -q -m "not slow"`
- Lint: `python -m ruff check cardiac/fastsim.py cardiac/bench_fastsim.py cardiac/tests/test_fastsim.py --select E9,F,E7 --ignore E731`
- Full fast suite (optional, ~12 min): `JAX_PLATFORMS=cpu python -m pytest cardiac/tests -q -m "not slow"`
- There is no build step.
- W&B: `cardiac.progress.make_wb(name, project="cardiac-guided-diffusion", entity="alx-ho-baylor-college-of-medicine")`.

## Key facts (do not re-derive)
- **Target ODE.** Reproduce exactly the semi-discrete ODE of `physics.make_forward_batched`:
  - `du = D*lap5(u) - k u (u-a)(u-1) - u v`
  - `dv = (eps0 + mu1 v/(u+mu2)) (-v - k u (u-a-1))`
  - `lap5` is the 5-point stencil with `jnp.pad(mode="edge")`, dx = (x1-x0)/(n-1), and dy likewise.
- **Params are traced.** Params (`physics.FixedAPParams`, an eqx pytree) stay traced so that gradients flow to all 6
  params. diff_sim_latent trains them. The benchmark itself only measures gradients wrt the initial condition.
- **Fixed constants** (float64 numpy, cast to the input dtype):
  - the 1D DCT-II matrix `C`: `C[k,j] = s_k cos(pi k (2j+1)/(2n))`, with `s_0 = sqrt(1/n)` and `s_k = sqrt(2/n)`
  - the eigenvalues `lam_k = -4 sin^2(pi k/(2n))` of the edge-replicate second-difference matrix T, which C diagonalises exactly
- **Built inside JAX, once per call, outside the scan:**
  - `Lhat = D*(lam_x[:,None]/dx^2 + lam_y[None,:]/dy^2)`
  - `E = exp(h Lhat)`
  - the phi / ETDRK4 coefficients
  - the separable propagator `Px = C^T diag(exp(h D lam_x/dx^2)) C`, and `Py` likewise
- **Transforms and propagators.**
  - Forward transform `uhat = C u C^T`, inverse `u = C^T uhat C`, batched over the leading dims.
  - Every matmul uses `precision=HIGHEST`. Default TF32 on the A6000 would ruin accuracy.
  - `exp(hDL) u = Px u Py^T` is exact and costs 2 matmuls.
  - Resolvents and phi functions are not separable. They need the eigenbasis (4 matmuls) or dctn.
- **phi functions.** Use the Kassam-Trefethen contour mean for every z, including z = 0, with no series and no
  branching. The contour has 32-64 points, radius 1, and is complex64-safe:
  `phi_k(z) = mean_j phi_k_direct(z + exp(i theta_j))`.
- **Stiffness.**
  - `max|eig| ~ 8D/dx^2 ~ 99.4`, so explicit Euler/Heun are stable for dt <~ 0.020 and RK4 for dt <~ 0.028.
  - The reaction is O(10).
  - ETDRK4 and Lawson4 can show stiff order reduction.
- **Eval data** is read-only, from the old repo: `/home/alex/physics-guidance/cardiac/runs/eval_val4` (4 cases).
  - npz keys: `y`, `u_true`, `v_true`, `guided_warm_seed0_u0/v0`, `adam_final_u0/v0`.
  - The y hash is `metrics.json[name]["y_sha1"]`.
  - `config.json["args"]` holds tau1, spacing, M, noise_std, loss, and stats.
  - The prior sweep is `/home/alex/physics-guidance/cardiac/runs/solver_sweep/summary.md`.
- **Reuse from `cardiac/solver_sweep.py`** by importing it, not by copying or modifying it:
  - `THRESH`, `TIMING_POINT`, `_flat`, `write_json`, `fmt`, `analyze`, `write_plots`.
  - Importing it also loads the model/guidance code. That is fine: no UNet is initialised.
  - Keep its results schema `{"meta", "settings", "rows"}`, adding extra fields only:
    - setting record: `status, error, compile_s, cfg`
    - row: `setting, case, point, failed, error, num_steps, ok_flag, U_rmse_max/mean, V_rmse_max/mean, U_rmse_tau, V_rmse_tau, grad_cos, grad_rel, t_fwd_ms, t_vjp_ms`
- **Baselines** are diffrax `physics.make_forward` / `make_forward_batched` with `solver_sweep.settings_table()["REF"]`
  and `["F02"]`, checkpoints=1024.

## Files to create (no existing file is modified)
1. **`cardiac/fastsim.py`**
   - **`FastCfg`** (subclass of `cardiac.config._Cfg`) has these fields:
     - `scheme`, `dt`
     - `diff_op`: one of `auto|lap5|sep|eig|dct`. `auto` resolves to `sep` for strang/lawson4, `eig` for etd2/etdrk4/imex_euler, and `lap5` for the explicit schemes.
     - `react`: `rk2|rk4`, used by strang only
     - `react_sub`: number of reaction substeps
     - `remat`: `step|segment|none`
     - `unroll`: int
   - **Operators** act on `[..., n, n]` arrays. Their dtype follows the input, so x64 works in tests.
     - `lap5`
     - `dct_matrix(n)` and `eig_lam(n)` (numpy constants)
     - `to_eig`/`from_eig` (matmul), `to_dct`/`from_dct` (`jax.scipy.fft.dctn`/`idctn`, type 2, ortho, axes (-2,-1))
     - `sep_exp(h, D)`
     - `phi_contour(z, k)`
     - the ETDRK4 coefficients
     - the resolvent `1/(1 - h Lhat)`
   - **`reaction(u, v, P)`** returns `(Nu, Nv)`.
   - **Steppers.** v never diffuses.
     - `euler`, `heun`, `rk4`, `tsit5`, all using lap5. tsit5 runs 6 stages with b_7 = 0 and no error estimate. Its
       tableau comes from `diffrax._solver.tsit5._tsit5_tableau`, with a hard-coded fallback.
     - `strang`: reaction half-step (RK2/RK4, `react_sub` substeps), then exact diffusion, then a reaction half-step.
     - `imex_euler`: implicit u via the resolvent, explicit v.
     - `etd2` (Cox-Matthews) and `etdrk4` (Cox-Matthews). Their v-part reduces to Heun / RK4.
     - `lawson4`: integrating-factor RK4 using only separable exponentials.
   - **Driver.**
     - Segments are `diff([0] + save_taus)`, with `steps = round(seg/dt)`. Raise `ValueError` unless every segment
       satisfies `|seg/dt - k| < 1e-4`.
     - Equal segments: an outer `lax.scan` over M segments emits `(u, v)`, wrapping an inner scan over steps.
     - Unequal segments: a Python loop over segments.
     - `remat="step"` checkpoints each step; `remat="segment"` checkpoints the inner scan.
   - **Factories.**
     - `make_forward_fast_p(n, grid, save_taus, cfg) -> fn(params, u0, v0)`
     - `make_forward_fast(n, grid, save_taus, cfg, params) -> fn(u0[n,n], v0[n,n]) -> (U[M,n,n], V[M,n,n], info)`
     - `make_forward_fast_batched(...)`: B-leading shapes, `[B,M,n,n]` outputs
     - `info = {"num_steps": static int, "ok": all-finite jnp bool}`
     - There is one implementation on `[..., n, n]`.
   - **Name helpers.** `parse_setting(name)` and `setting_name(cfg)`, for names like `strang2_d0.1`, `etd2_d0.05`, `strang2_d0.1_dct`.

2. **`cardiac/bench_fastsim.py`**
   - **Module top:** `os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")`.
   - **GPU guard.** `gpu_guard(env, pid, force)` is a pure function that runs in `main()` before any jax call.
     - If `JAX_PLATFORMS` is not cpu-only and `/proc/<pid>` exists, print a warning and exit 2 unless `--force`.
     - It only checks the path. It never signals the process.
     - Default `--guard-pid` is 1657544.
     - Timing-phase children get an internal flag that skips the guard.
   - **Common flags.**
     - `--eval` (default: the absolute old-repo path above)
     - `--stats` (default: the eval config's `args.stats`, resolved relative to the eval dir's repo root, then the cwd; falls back to `cardiac/artifacts/stats.json`)
     - `--out`, `--settings`, `--cases N`
     - `--points` (default `truth,z_init,warm_seed0,adam_final`)
     - `--reps` (default 3; values below 3 are flagged "indicative" in the summary)
     - `--no-wandb`, `--plot-only`
     - `--redo` (without it, the run resumes and skips settings already in results.json)
     - `--force`, `--guard-pid`
   - **Timing protocol** (both phases):
     1. Lower and compile each function once: `c = jax.jit(f).lower(*args).compile()`.
     2. Record `compile_s` and `c.memory_analysis().temp_size_in_bytes` from that same object.
     3. Make one explicit warm-up call each for fwd and VJP.
     4. Report the median of `--reps` timed calls.
   - **`--phase accuracy`:**
     - Uses a lightweight loader: no UNet, no A01 forward. It verifies the y sha1 and skips missing points.
     - No tweedie points. The summary must say that the point set is truth/z_init/warm/adam, so acceptance is not
       identical to the old sweep.
     - REF runs first and is cached to `<out>/ref_cache.npz`.
     - For each setting:
       - jit the forward, and jit value_and_grad of the gaussian `obs_loss` (sigma = noise_std) wrt (u0, v0)
       - for each point: U/V RMSE per tau, plus grad cos and grad rel L2 vs REF
       - at `TIMING_POINT`: fwd and VJP ms, `vjp_over_fwd`, `temp_MB`
     - After each setting: write results.json atomically, print a progress line, and log the aggregates to W&B.
   - **`--phase timing`** (the GPU step; only smoke-tested on CPU here):
     - Every config runs in a subprocess, following the bench_batch pattern with a `BENCH {json}` line.
       `peak_bytes_in_use` is recorded, or `temp_bytes` on CPU.
     - Settings come from `--settings`, or from `--from-accuracy <dir>`: F02, tsit5_d0.02, the cheapest accepted
       setting per family (at most 6), and the operator variants (`strang2_d0.1` with sep/eig/dct, `etd2_d0.1` with eig/dct).
     - Remat `step,none` runs at B=1. Remat `segment` and `--unroll 4` run for the top 2 only.
     - `--batches 4,16,32` runs for F02 (baseline: `physics.make_forward_batched`) and the top 2. Lanes tile the
       cases' warm points.
     - Configs run in priority order with time-capped reps.
     - `--max-minutes` (default 20) stops launching new configs; the rest are recorded as `skipped_budget`. OOMs are recorded, not fatal.
     - Records per config: compile_s, fwd/VJP ms, VJP/fwd, ms per case, peak memory, speedup vs F02 at the same B, and lane-0 batched-vs-unbatched diff.
   - **Outputs** go under `--out`: `results.json`, `summary.md`, and the plots.
     - Summary columns: the `analyze` columns plus scheme, dt, steps, RHS/step, temp MB, VJP/fwd, speedup vs F02 (same
       run and backend), and accepted.
     - The summary also gives the best accepted setting, the backend, and the point-set note.
     - Plots: `error_vs_cost.png` and `gradient_agreement.png` (both via `write_plots`), `convergence.png` (error vs dt
       per family, log-log, with threshold lines), `timing.png`, and `throughput.png`.

3. **`cardiac/tests/test_fastsim.py`**: CPU only, fast, not marked slow, target under 3 min.

**DEFAULT_ACCURACY_SETTINGS:**
- REF, F02
- euler_d{0.01,0.02}, heun_d{0.01,0.02}, rk4_d{0.02,0.025}, tsit5_d0.02
- strang2_d{0.02,0.05,0.1,0.2}, strang4_d{0.05,0.1,0.2}
- imex_euler_d{0.02,0.05}
- etd2_d{0.02,0.05,0.1,0.2}, etdrk4_d{0.05,0.1,0.2}, lawson4_d{0.05,0.1,0.2}

If the CPU budget is tight, drop these first: euler_d0.01, heun_d0.01, strang2_d0.02, etd2_d0.02. Always keep dt
0.05/0.1/0.2 for the exponential families. The remat, unroll and operator variants do not change the math, so they are
swept in the timing phase only.

## Ordered tasks (single track)

**T1. fastsim part 1:** FastCfg, operators, reaction, explicit steppers, driver, factories, name helpers.

Tests:
- (a) x64: `C^T diag(lam) C == T` to 1e-10, and the spectrum equals `eigh(T)`.
- (b) Random non-smooth u: the eigenbasis Laplacian equals `D*lap5` to 1e-10 under x64 and to 1e-4 relative in
  float32. The dct path equals the eig path to the same tolerances.
- (c) `sep_exp` equals the eigenbasis exp to 1e-5 (float32).
- (d) tsit5_d0.02 vs diffrax F02, n=32, taus [0.5, 1.0]: forward max diff <= 1e-5, grad rel <= 1e-4.
- (e) batched == per-lane unbatched == `vmap(unbatched)`, taus [0.1, 0.2], tolerance 1e-5.
- (f) save_taus off the dt grid raise `ValueError`.
- (g) remat step/none/segment give the same grads, taus [0.1, 0.2], rel 1e-5.

**T2. fastsim part 2:** strang, imex_euler, etd2, etdrk4, lawson4, phi_contour, ETDRK4 coefficients.

Tests:
- (p) phi_contour:
  - At z = 0 it gives 1, 1/2, 1/6 to 1e-6.
  - At moderate |z| (0.5-20, negative real) it matches the float64 direct formulas to 1e-5 relative.
  - It is differentiable wrt D.
- (h) Convergence, run under `jax.experimental.enable_x64()`. The reference is diffrax Tsit5 with rtol = atol = 1e-10,
  dtmax 1e-3, n=32, smooth random IC, taus [0.5, 1.0].
  - Measure the order only in the non-stiff regime: choose D so that `h*max|Lhat| <~ 1` for the tested dts.
  - Each scheme has explicit dt pairs. Both errors must lie in [1e-11, 1e-2]; otherwise the test fails with a message
    and is never skipped.
  - The observed order must be >= p - 0.3, where p = 1 for euler and imex_euler, p = 2 for heun, strang2, strang4 and
    etd2, and p = 4 for rk4, tsit5, etdrk4 and lawson4.
  - The error at the finest dt must be < 1e-4.
- (i) Stiff check, with D stiffness-matched (`8D/dx^2 ~ 100`):
  - At dt = 0.05 and 0.1, strang, etd2, etdrk4 and lawson4 stay finite and have max U error < 1e-2 vs the reference.
    The implementer may calibrate this tolerance once; record it in the test.
  - euler at dt = 0.05 goes non-finite or has a large error.
  - No order is asserted, because stiff order reduction is expected.
- (j) IC gradients for strang2, etd2 and lawson4: grad·d agrees with `jax.jvp` (rel <= 1e-4) and with central FD (h = 1e-2, rel <= 5e-2).
- (o) Parameter gradients for strang2, etd2 and lawson4, n=16, taus [0.2, 0.4]:
  - grad wrt all 6 `FixedAPParams` leaves of a loss on U, vs a local diffrax reference (RDSystem with params passed as
    args, Tsit5, fine fixed dt). The scheme runs at small dt; rel <= 1e-3.
  - The same grad agrees with the JVP directional derivative (rel <= 1e-4).
- (k) The sep/eig/dct variants agree after a short tau, tolerance 1e-4.

**T3. bench accuracy phase:** guard, loader, stats resolution, REF cache, resume, timing protocol, summary/plots, W&B.

Tests:
- (l) CLI smoke on the `run_eval_cli` fixture (n=16, taus 0.2/0.4) with `--settings REF,F02,rk4_d0.02,strang2_d0.1 --no-wandb --reps 1`:
  - the outputs exist, REF is cached, and a rerun resumes
  - the summary flags reps < 3
- (m) `gpu_guard(env, pid, force)` returns refuse, allow, allow for: a fake existing PID with a GPU env; a cpu env;
  `force`. No jax GPU backend is initialised and no signal is sent.

**T4. bench timing phase.** CPU smoke run:
`JAX_PLATFORMS=cpu python -m cardiac.bench_fastsim --phase timing --settings F02,strang2_d0.1 --remat step,none --batches 1,2 --reps 1 --cases 1 --no-wandb --max-minutes 5 --out cardiac/runs/fastsim_timing_smoke`

**T5. CPU accuracy run (implementer).**
1. Smoke run: `--cases 1 --points truth,warm_seed0`.
2. Full run: `JAX_PLATFORMS=cpu MPLBACKEND=Agg python -m cardiac.bench_fastsim --phase accuracy --out cardiac/runs/fastsim_cpu --reps 3`
   - Estimate: REF ~10-15 min (cached), F02 ~3 min, fast settings ~15-25 min; ~30-45 min in total.
   - If the projection exceeds 90 min, trim the dt grid as above and/or use `--cases 2`, and say so.
3. Copy summary.md and the PNGs to `/home/alex/cardiac-guided-diffusion/.claude/plans/fast-ap-solver-media/`.
4. Write `REPORT.md` there, with:
   - findings: best accepted setting per family, VJP/fwd ratios, CPU speedup vs F02
   - the GPU command
   - a "not implemented, why/when" section, which includes the 3D next step: RDSystem3D. The DCT eigenbasis and the
     separable exp extend directly to 3 axes (3 matmuls). This matters because the user's running training job is
     `trainable_AP_multi_horizon_3d.py`.
5. Report the W&B URL.

**T6. Final checks:** lint is clean; `test_fastsim.py` and `test_physics.py` (not slow) pass; `git status` shows only the 3 new files.

**GPU timing step** (not run by the implementer). It needs the user's approval and the GPU free of the training job.
The guard exits 2 while PID 1657544 exists, unless `--force`. Estimated 10-15 min, hard cap `--max-minutes 20`:
```
cd /home/alex/cardiac-guided-diffusion && MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/alex/conda/envs/diff/bin/python -m cardiac.bench_fastsim --phase timing \
  --from-accuracy cardiac/runs/fastsim_cpu --out cardiac/runs/fastsim_gpu --max-minutes 20 --reps 5
```

## Parallel tracks
Single track (T1 -> T6), because the bench depends on the fastsim API. No setup step and no new dependencies.

## W&B
- Project: `cardiac-guided-diffusion`, via `make_wb`.
- Runs: `fastsim-accuracy-<backend>` and `fastsim-timing-<backend>`.
- Logged live, per setting/config: U_max, V_max, cos_min, rel_max, t_fwd_ms, t_vjp_ms, vjp_over_fwd, speedup_vs_F02,
  compile_s, temp_MB/peak_MB, accepted. A table grows as settings finish.
- At the end: the summary table, the PNGs, and a results.json artifact.
- `--no-wandb` switches to `NullWB`. W&B failures are non-fatal.

## Acceptance criteria
- [ ] The only new tracked files are `cardiac/fastsim.py`, `cardiac/bench_fastsim.py` and `cardiac/tests/test_fastsim.py`. No existing file is modified.
- [ ] All 9 schemes follow the `physics.make_forward` contract in both factories.
- [ ] All schemes are jit-able, vmap-able, and differentiable wrt (u0, v0) and all 6 params.
- [ ] Tests (a)-(p) pass on CPU in under 3 min. `test_physics.py` (not slow) passes. ruff is clean.
- [ ] tsit5_d0.02 in the scan reproduces F02: forward <= 1e-5, grad rel <= 1e-4.
- [ ] The CPU accuracy run is complete; any reduction is stated.
- [ ] `cardiac/runs/fastsim_cpu` has results.json, summary.md, ref_cache.npz and the 3 PNGs.
- [ ] Accepted flags use the existing thresholds, and speedup vs F02 comes from the same run.
- [ ] The media dir has summary.md, the PNGs and REPORT.md (including the not-implemented/3D section). The W&B URL is reported.
- [ ] The timing phase is smoke-tested on CPU and bounded by `--max-minutes`.
- [ ] The guard refuses without initialising the GPU. PID 1657544 is never signalled, and no GPU work is run.

## Scope notes
- **Added:**
  - lawson4
  - the separable `Px u Py^T` diffusion step
  - param-gradient support (the training repo trains theta)
  - remat/unroll sweeps, as a proxy for what a hand-written adjoint could save
  - compiled memory numbers, the time budget, and REPORT.md
- **Left out** (each is covered in REPORT.md under "not implemented, why/when"):
  - a custom_vjp adjoint: `remat="none"` already bounds its gain
  - Rush-Larsen for v
  - Pallas fused multi-step kernels: the 400 KB state needs halo tiling
  - XLA command buffers
  - RKC
  - coarse-grid gradients, bf16, parareal
  - tweedie points
  - wiring fastsim into physics/SolverCfg
  - the 3D extension (a next step)

## Risks and open questions
- **Safety and API.** No data deletion, migrations, new dependencies, or public API changes. One private diffrax
  symbol is used, with a fallback.
- **Accuracy.** Chaotic dynamics amplify errors over 10 tau, so the large-dt schemes may fail the 0.002 threshold.
  That would be a result, not a bug. Stiff order reduction is expected for ETDRK4 and Lawson4.
- **Timings.** CPU rankings will not carry over to the GPU: the GPU is launch-bound, while the CPU pays for the 224^3
  matmuls.
- **Inadmissible inputs** (the u = -mu2 singularity) return `ok=False`, the same as diffrax.
- **OPEN: pause/resume wrapper.** The coordinator asked for a wrapper that pauses and resumes the training job (PID
  1657544) around the GPU run. Writing that into the plan was BLOCKED by the permission system (interference with a
  running workload), so it is not included. It needs the user's direct approval and permission. Until then, the GPU
  step runs only when the GPU is free.
