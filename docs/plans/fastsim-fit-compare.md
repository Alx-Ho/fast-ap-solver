# fastsim-fit-compare: fast-solver follow-up (A: equal-VJP Adam IC fits + rollout videos, B: parameter gradients + recovery)

(Revision 1: applied orchestrator decisions on B2 optimiser/mapping, F02 stability, B1 timing/metrics, B3 FD tests,
A caches and A Adam stop/labels.)

## Goal
One GPU pause, two independent parts. (A) For the 16 headline test cases, fit (u0, v0) at frame 128 with the repo's basic
Adam (200 it. = 200 VJPs) twice, identical except the forward operator: original F02 (diffrax tsit5, fixed dt 0.02) vs the
fastest accepted fast solver `lawson4_d0.2_unroll4`. Roll the truth and both fits out with the same A01 judge solver and
show side-by-side videos, an RMSE-over-tau figure and a table with fit wall times. (B) Check the fast solvers for parameter
(theta) gradients vs REF, and for theta recovery with the user's real `BoundedLogRDParams` class, F02 vs lawson4.

## Project commands
- Run everything from the repo root `/home/alex/cardiac-guided-diffusion`.
- CPU tests: `JAX_PLATFORMS=cpu AP_SIMULATOR_PATH=/home/alex/diff_sim_latent/ap_simulator python -m pytest cardiac/tests/test_fastsim_fit_compare.py cardiac/tests/test_fastsim_params.py -q`
  (full suite: `... python -m pytest cardiac/tests -q`).
- Lint: `ruff check <new files>` if installed, else `python -m pyflakes <new files>`. No build step. No new dependencies.

## Shared facts (both tracks; each track has its own small loaders, no shared new module)
- Cases: `final_figs.load_eval(eval_dir, headline=True)` returns the 16 headline cases (= `split["test"][4:20]`). Each case's
  chunk dir holds `estimates_<case>.npz` (`y`, `u_true`, `v_true`, `adam_final_{u0,v0}`); the third return value is the run
  args `ra`. Verify `sha1(y) == metrics[case]["y_sha1"]` (bench_fastsim.load_eval pattern). Default `--eval
  cardiac/runs/eval_test_n20`; if absent, fall back to `/home/alex/physics-guidance/cardiac/runs/eval_test_n20` and print
  which was used.
- Stats: `bench_fastsim.resolve_stats(None, ra, eval_dir)`. Data config `ra["data_root"]/config.json` gives n (224), grid,
  dx and the true theta via `physics.FixedAPParams.from_config`. Obs taus: `physics.observation_taus(ra["tau1"],
  ra["spacing"], ra["M"])` (1..10). Loss: `physics.obs_loss(ra["loss"], ra["noise_std"], FullField(), n)` (gaussian).
  Normaliser: `data.Normalizer.from_stats`.
- Diffrax settings: `solver_sweep.settings_table()`. REF = rtol=atol=1e-6, dtmax 0.0025, max_steps 40000; F02 = fixed dt 0.02;
  A01 = rtol=atol=1e-3, dtmax 0.01.
- Fast settings:
  - `fastsim.parse_setting(name)` gives the config.
  - `fastsim.make_forward_fast(n, grid, taus, cfg, params)` is the fixed-theta forward.
  - `fastsim.make_forward_fast_p(n, grid, taus, cfg)` returns `fn(params, u0, v0)` with theta traced. `params` can be any
    object with attributes D, eps0, a, k, mu1, mu2; `BoundedLogRDParams` qualifies.
- Traced theta for diffrax: call `physics.make_forward(n, grid, taus, cfg, params)(u0, v0)` INSIDE the traced function, with
  `params` a `FixedAPParams` whose leaves are tracers (it is an eqx.Module, differentiated as a pytree) or a
  `BoundedLogRDParams`. B3's unit test verifies this works. If it does not, write a thin wrapper in the track B file; do not
  edit physics.py.
- GPU guard: call `bench_fastsim.gpu_guard(os.environ, a.train_pid, a.force)` before importing jax, and reuse bench_fastsim's
  `--train-pid` default. W&B: `progress.make_wb(name, config=...)`, or `progress.NullWB` with `--no-wandb`.
- Colours (all figures and video titles): truth black, F02 `#1f77b4` (solid), lawson4_d0.2 `#d62728` (dashed),
  lawson4_d0.1 `#ff7f0e`. Other settings take colours from a fixed qualitative palette keyed by setting name, so the same
  name always gets the same colour.

## Parallel tracks (no setup step: no installs or lockfile changes)
- Track A owns `cardiac/fastsim_fit_compare.py`, `cardiac/tests/test_fastsim_fit_compare.py`, the outputs in
  `cardiac/runs/fastsim_fit_compare/` and the media files `.claude/plans/fastsim-fit-compare-media/fit_compare_*`.
- Track B owns `cardiac/fastsim_param_check.py`, `cardiac/tests/test_fastsim_params.py`, the outputs in
  `cardiac/runs/fastsim_param_check/` and the media files `.claude/plans/fastsim-fit-compare-media/param_*`.
- Neither track modifies existing files or imports the other's new module.

---------------------------------------------------------------------------------------------------------------------------
## Track A: `cardiac/fastsim_fit_compare.py` (equal-VJP IC fit = the end-to-end Adam check)

### Method
- Init (= run_eval.run_adam with v_init "mean" and no mask): `u = clip(y[0],0,1)`, `v = full_like(u, stats["v"]["mean"])`.
  Assert `ra.get("v_init","mean")=="mean"`, `ra.get("u_init","mask_mean")=="mask_mean"` and no mask/sparse (as
  rollout_compare does).
- Adam: `config.AdamCfg(lr=ra["adam_lr"], iters=a.iters or ra["adam_iters"], snapshot_iter=0, stop=<off>)`, with
  early stopping explicitly disabled (use AdamCfg's "no stop" value, e.g. `stop="none"`; check `config.py`). Expected
  values: lr 0.01, 200 iterations.
  - Build `baseline.AdamFitter(forward, loss_fn, norm, cfg, n)` ONCE per method and call `.fit(y, iu, iv)` per case. It
    returns u0, v0, loss_hist, final_L_obs, wall_time_s (excluding compile), compile_time_s, n_vjp and n_failed.
  - Assert `n_vjp == iters` for both methods, and record/report `n_failed`.
  - Use B=1 for both methods (clean per-fit wall time).
- Original forward: `physics.make_forward(n, grid, obs_taus, replace(settings_table()["F02"], checkpoints=ra.get("checkpoints", 1024)), fp)`.
- Fast forward: `fastsim.make_forward_fast(n, grid, obs_taus, parse_setting(a.fast), fp)` with `--fast lawson4_d0.2_unroll4`.
- Judge rollouts: `jax.jit(physics.make_forward(n, grid, roll_taus, SolverCfg(rtol=1e-3, atol=1e-3, dtmax=0.01,
  stepping="adaptive", checkpoints=None), fp))` (= rollout_compare's A01), with `roll_taus = arange(dt, tmax+1e-6, dt)`,
  `--dt 0.25 --tmax 30`, and the tau=0 state prepended. Per-tau RMSE via `rollout_compare.rmse_curve`.

### CLI
`--eval`, `--cases N` (first N headline cases; default 16), `--fast`, `--iters`, `--tmax 30 --dt 0.25`, `--out
cardiac/runs/fastsim_fit_compare`, `--media-dir .claude/plans/fastsim-fit-compare-media`, `--video-cases 4` (first K cases
get per-case videos and form the montage), `--no-render`, `--plot-only`, `--redo`, `--force`, `--train-pid`,
`--no-wandb`, `--wandb-name fastsim_fit_compare`.

### Caches (resumable; write to tmp then rename; skip existing unless --redo)
- `meta.json`: cases, global idx, Adam cfg, F02/fast/judge cfg dicts, device, tmax/dt. Refuse to run on a mismatch unless
  --redo.
- `fit_{f02,fast}_<case>.npz` (small): the IC pair u0, v0 plus metrics: loss_hist, final_L_obs, wall_time_s,
  compile_time_s, n_vjp, n_failed. The f02 file also stores `max|u0/v0 - stored adam_final|` (informational).
- `roll_<case>.npz` (small): taus (including 0) and rmse_{u,v}_{f02,fast} curves, computed in the GPU run.
- Frames: only for the `--video-cases` cases, as `frames_<case>.npz` (float16, savez_compressed). Rollouts for all other
  cases are not cached as frames.
- `--plot-only` re-renders from the frame caches. If a video case lacks frames, it recomputes those rollouts from the cached
  IC pairs on whatever device is available (CPU is slower, but there are only K cases). It never refits.

### Per-case loop
Fit f02, fit fast, run the 3 judge rollouts, write the caches, then print one line, e.g.
`[fit_compare] case 3/16 <case>: F02 200 VJP 97.4s L=.. u0 0.071 v0 0.262 | fast 200 VJP 1.9s L=.. u0 .. v0 .. | 51x | v@30 0.41/0.41 | elapsed 5.6 min`
The same numbers go to `wb.log`:
- per method: wall_s, compile_s, n_vjp, n_failed, final_L, rmse_u0, rmse_v0, rollout RMSE u/v @10/@30/@end;
- speedup;
- max|u0_f02 - u0_fast| and max|v0_f02 - v0_fast|.

### Outputs (rendered inline unless --no-render; or later with --plot-only)
- `videos/fit_compare_<case>.mp4` for each video case, in the coevolve_video layout:
  - rows: u, u residual, v, v residual; colormaps Reds/Blues/RdBu_r; residual limits ±0.5/±1.0;
  - per-panel RMSE and a phase caption; fps 15, dpi 90;
  - columns: Truth | Original F02 | Fast lawson4;
  - column titles show that case's measured wall time and the measured speedup (F02 wall / fast wall), e.g.
    "Original: diffrax Tsit5, dt 0.02 (F02)\n200 VJPs, fit 97.4 s" and "Fast: Lawson4, dt 0.2\n200 VJPs, fit 1.9 s
    (51x faster, measured)".
  - Reimplement the render loop as a function (coevolve_video's loop lives inside main() and cannot be imported).
- `videos/fit_compare_montage.mp4`:
  - rows: the video cases;
  - columns: Truth v | F02 v | fast v | F02 v residual | fast v residual;
  - one shared tau clock;
  - the title shows the mean measured fit time and speedup per method.
- `fit_compare.{png,pdf}` (from the cached RMSE curves and loss_hist; no recompute), with panels:
  - rollout RMSE u and v vs tau over all 16 cases (mean + 25-75% band, obs window shaded);
  - obs loss vs Adam iteration (mean, log y);
  - obs loss vs fit wall time (log x).
- `summary.md` / `summary.json`:
  - Rows: per case, plus mean ± std.
  - Columns: u0/v0 RMSE, rollout RMSE u/v @10/@30/@end, final loss, VJPs, n_failed, wall s, compile s, measured speedup,
    max |F02 - fast| in u0/v0, and the F02-vs-stored diff.
  - Header: device, settings, identical y/init/lr/iters/no early stop, and that timing excludes compile.
- At the end:
  - W&B: `wandb.Video` (montage and per-case videos), `wandb.Image` (figure), `wandb.Table` (summary); print the run URL.
  - Copy the montage, the figure, summary.md and the first 3 case videos to the media dir (`fit_compare_*` names).

### Tasks A
- **A1.** CLI, guard, case loading, caches, both fits (asserts on n_vjp), judge rollouts (RMSE for all cases, frames for
  the video cases), progress line and W&B logging. CPU smoke: `--cases 1 --video-cases 1 --iters 3 --tmax 2 --dt 0.5
  --no-render --no-wandb`.
- **A2.** Rendering (per case and montage), figure, summary, W&B uploads, media copy, and `--plot-only`, including the
  rollout recompute when frames are missing. CPU smoke on A1's cache.
- **A3.** `cardiac/tests/test_fastsim_fit_compare.py`:
  - gpu_guard wiring: refuses without --force (monkeypatch the pid).
  - Tiny end-to-end CPU run in tmp_path (1 case, 3 iterations, tmax 1, dt 0.5, no wandb). Check that the caches exist,
    n_vjp == 3 for both methods, n_failed is reported, the RMSEs are finite, and the mp4, png and summary.md exist.
  - A re-run resumes (fit cache mtime unchanged).
  - Deleting `frames_<case>.npz` and running `--plot-only` regenerates the video without refitting.
  - Skip if the eval dir, the simulator or ffmpeg is missing.

---------------------------------------------------------------------------------------------------------------------------
## Track B: `cardiac/fastsim_param_check.py` (subcommands `grad`, `recover`, `report`; `all` runs grad, recover, report)

### Common
- Cases: the first `--cases 4` headline cases (same loader as A; truth state and y come from the eval npz). Obs taus 1..10,
  gaussian loss.
- Settings: `--settings F02,tsit5_d0.02,rk4_d0.025,lawson4_d0.05,lawson4_d0.1,lawson4_d0.2,lawson4_d0.2_unroll4,etdrk4_d0.1,etdrk4_d0.2,strang4_d0.05`.
  Reference: `--ref REF`.
- `make_fwd_p(name)` returns `fn(params, u0, v0)`:
  - diffrax names: `make_forward` called inside the traced function (checkpoints 1024, as in bench_fastsim.make_setting);
  - fast names: `make_forward_fast_p`.
- Perturbed theta (deterministic): for seed s, `rng = default_rng(s)`. Each of the 6 params, D included, gets
  `factor = exp(sign * U(log 1.2, log 1.4))` with a random sign. Seeds: `--seeds 0,1,2`.
- F02 stability limit, computed rather than hardcoded:
  - `D_max_F02 = 3.5 * dx**2 / (8 * 0.02)`, where dx is the grid spacing from the data config and 3.5 is Tsit5's real-axis
    stability bound for the 5-point Laplacian (eigenvalue up to 8D/dx²). This is about 0.00176 at n=224.
  - Store it in meta and in summary.md.
  - Assert that every perturbed start (including D × 1.4) has D below it; the test checks this for seeds 0-2.

### B1 `grad`: parameter-gradient accuracy at full scale
- Points per case, each evaluated with ONE VJP of the loss w.r.t. (u0, v0, theta), theta being a traced `FixedAPParams`:
  - `truth`: true state, true theta;
  - `theta_pert`: true state, seed-0 perturbed theta;
  - `joint`: the stored `adam_final` state, seed-0 perturbed theta.
- Metrics vs REF, at every point:
  - Per-param normalised error `|g_k - g_ref_k| / ||g_ref||`, computed in log-parameter scale (theta_k * g_k) so the 6
    entries are comparable. This feeds the bar chart.
  - Per-param relative error `|g_k - g_ref_k| / |g_ref_k|`, reported only where `|g_ref_k| >= 1e-3 * ||g_ref||`
    (otherwise "n/a (tiny ref)"). At the `truth` point the reference gradient is noise-driven, so expect many n/a.
  - Sign agreement per param (same tiny-ref exclusion).
  - Cosine of the 6-vector (log scale), for all points.
  - The (u0, v0) block of the joint gradient: cosine and relative L2 vs REF. This is the state-VJP accuracy; there is no
    separate state-VJP run.
  - Forward U/V RMSE vs REF.
- Timing: only the theta-only VJP and the joint VJP, each the median of 3 reps after warm-up (`bench_fastsim.med_time_ms`
  with `cap_s`), plus compile seconds.
  - Per-setting cap `--setting-cap-s 240` covers compile plus all points plus timing. On overrun, record
    `status="capped"` with whatever finished and move on.
  - REF is exempt from the cap but is computed only once and cached.
- Caching and logging:
  - REF gradients go in `ref_grads.npz`, keyed by case/point and validated like bench's ref_cache.
  - Per-setting rows go in `grad_results.json` (resume per setting).
  - A failure (non-finite, ok=False, compile error) is recorded, not fatal.
  - Print one line per (setting, case). `wb.log` per-setting aggregates: median/max normalised error per param, min
    cosine, state-block cosine, ms per theta/joint VJP.

### B2 `recover`: parameter recovery with the true (u0, v0) known
- Model: `BoundedLogRDParams(**theta_init, bounds=bounds, mapping_mode="direct")`, imported as in physics.py
  (`ap_simulator.params`).
  - Bounds: `DEFAULT_BOUNDS`, widened to include 0.5x-2x truth (the physics.load_dataset_rd pattern).
  - With the direct mapping raw = log(param), read back as exp and then clipped to the bounds. Outside the bounds the
    gradient is zero, so a param pinned at a bound stays there. Detect this and report it per fit (`at_bound` flags).
  - The raw leaves are the optimised pytree (`eqx.partition(m, eqx.is_inexact_array)`).
  - Pass the module directly as `params` to both forwards (the unit test confirms this works; add a thin adapter in this
    file only if needed).
- Loss: obs loss of `fwd(params, u_true, v_true)` vs the stored y.
- Optimiser: identical for every method:
  `optax.adam(optax.exponential_decay(0.02, transition_steps=iters // 3, decay_rate=0.5, staircase=True))` with
  `--iters 200` (200 VJPs).
- Methods: `--methods F02,lawson4_d0.2,lawson4_d0.1`. Fits: `--recover-cases 2` × seeds 0, 1, 2.
- Recorded every iteration: loss, the 6 physical params and a finite/ok flag.
- Two failure kinds, kept in separate columns (never crash):
  - `instability`: the loss or gradient is non-finite, or the solver reports ok=False. Record `failed_at_it`, keep the last
    finite params and stop that fit. For F02, also record whether D exceeded `D_max_F02` at that point. The widened bounds
    allow D up to 2x truth, so F02 can drift into its explicit stability limit. That is a legitimate finding to report, not
    hide.
  - `not_converged`: the fit finished, but some param's final relative error is above 10% (threshold `--conv-tol 0.1`),
    reported per param.
- Per-fit cache `rec_<method>_<case>_s<seed>.npz`: loss_hist, param_hist, wall_s (excluding compile), compile_s, n_vjp,
  instability info (failed_at_it, reason, D at failure), at_bound, and final relative errors per param.
- Print one line per fit. `wb.log` per fit: final relative error per param, mean |relative error|, final loss, wall s,
  failure kind.

### `report`
- `summary.md` contains:
  - The B1 table: setting × param, with median/max normalised error, relative error where defined, sign agreement %, min
    cosine, state-block cosine, ms per theta/joint VJP, speedup vs F02 for the joint VJP, status (ok/capped/failed).
  - The B2 table: method × param, with mean ± std relative error over case × seed, an instability-failures column, a
    not-converged column, an at-bound column, wall s, and speedup.
  - A final-loss comparison.
  - D_max_F02 and the starting D values.
- `param_grad_error.{png,pdf}`: grouped bars, x = the 6 params, one bar per setting (log y, median normalised error with a
  max whisker), dashed line at 1e-2. Second panel: 1 - cosine (log) per setting, by point.
- `param_recovery.{png,pdf}`: 6 panels, one per param, plotting param/truth ratio vs iteration, coloured by method, with
  thin lines per case × seed and a truth line at 1. Failures are marked with an x at failed_at_it; the D panel shows
  D_max_F02/truth as a dotted line. Extra panel: loss vs iteration and vs wall time.
- W&B images and tables. Copy the summary (as `param_summary.md`) and the pngs/pdfs to the media dir with a `param_`
  prefix.

### CLI
`--eval`, `--cases`, `--recover-cases`, `--settings`, `--ref`, `--methods`, `--seeds`, `--iters`, `--conv-tol`,
`--setting-cap-s`, `--taus-max` (use only obs taus <= this; for CPU smoke), `--out cardiac/runs/fastsim_param_check`,
`--media-dir`, `--redo`, `--force`, `--train-pid`, `--no-wandb`, `--wandb-name fastsim_param_check`.

### Tasks B
- **B0** (do FIRST; CPU only; blocks every GPU step): write the core of `cardiac/tests/test_fastsim_params.py`, test (a),
  which catches operators precomputed from a static D/theta:
  - Tiny synthetic grid (n=16, taus [0.2, 0.4]).
  - For each param (D, eps0, a, k, mu1, mu2), compare the theta-gradient (and a JVP) of the obs loss against central finite
    differences in log-theta (rel tol 1e-2, step 1e-3), explicitly for `lawson4_d0.1`, `etdrk4_d0.1`, `strang4_d0.05` and
    `rk4_d0.025` via `make_forward_fast_p`, and for `physics.make_forward` (F02) with traced FixedAPParams.
  - Also check the fast gradients agree with F02's.
  - If a scheme fails for D, that scheme is excluded from B1/B2 and flagged.
- **B1-task.** Loader, guard, D_max_F02, make_fwd_p for diffrax and fast names with traced theta, and the `grad`
  subcommand (metrics, cap, caches, logging). CPU smoke: `grad --cases 1 --settings lawson4_d0.2,tsit5_d0.02 --ref F02
  --taus-max 1 --no-wandb`.
- **B2-task.** The `recover` subcommand (direct-mapping BoundedLogRDParams, decayed Adam, failure kinds, at-bound
  detection, caches, logging). CPU smoke: `recover --recover-cases 1 --seeds 0 --methods lawson4_d0.2 --iters 3
  --taus-max 1 --no-wandb`.
- **B3-task.** `report` (tables, the two figures, W&B, media copy), plus the remaining tests:
  - (b) BoundedLogRDParams(mapping_mode="direct") passes into both forwards, and gradients w.r.t. its raw leaves are finite
    and equal param × d/dparam inside the bounds;
  - (c) the perturbation is deterministic, within ±20-40%, and all starts satisfy D < D_max_F02;
  - (d) a tiny `all` run in tmp_path (CPU, 1 case, settings lawson4_d0.2, ref F02, taus-max 1, 3 iterations) produces
    summary.md and both pngs. Skip (d) if the eval dir or simulator is missing.

---------------------------------------------------------------------------------------------------------------------------
## Combined GPU run (orchestrator; training job paused; A then B)
Precondition: both test files pass on CPU (in particular B0's FD checks).
```
cd /home/alex/cardiac-guided-diffusion && \
python -m cardiac.fastsim_fit_compare --force --no-render 2>&1 | tee cardiac/runs/fastsim_fit_compare.log && \
python -m cardiac.fastsim_param_check all --force 2>&1 | tee cardiac/runs/fastsim_param_check.log
# After the GPU is released: render A's videos/figure/table from caches (frames for the 4 video cases are cached).
JAX_PLATFORMS=cpu python -m cardiac.fastsim_fit_compare --plot-only
```
B's `report` makes only light plots, so it runs inside `all`. To render on the GPU run instead, drop A's `--no-render`
(adds ~5 min).

GPU estimate:
- A: F02 200 VJPs × 0.49 s ≈ 98 s/case, so ~26 min for 16 cases plus ~2 min compile; fast ~2 s/case (~1 min); 48 A01
  rollouts to tau 30 ~3-8 min. Total ~35 min (upper ~45).
- B1: 12 REF joint VJPs (~5-15 s each, ~3 min) plus 10 settings × (12 joint VJPs + timing of theta/joint VJPs at 3 reps +
  compile ~0.5-1.5 min), capped at 4 min per setting. ~12-25 min.
- B2: F02 6 fits × 200 theta-VJPs × ~0.5-0.6 s ≈ 10-12 min, plus lawson4 d0.2/d0.1 12 fits ≈ 2-3 min. ~13-15 min.
- B total ~25-40 min. Overall ~60-85 min of GPU, upper bound ~100 min.

## W&B
Project via `progress.make_wb` (repo default project `cardiac-guided-diffusion`). Runs:
- `fastsim_fit_compare`: per-case fit metrics live; videos, figure and table at the end. The plot-only run is logged as
  `fastsim_fit_compare_plots`.
- `fastsim_param_check`: per setting and per fit live; figures and tables at the end.

Print the run URLs.

## Acceptance criteria
- [ ] Both scripts refuse to start on GPU while the train pid exists unless `--force`; `JAX_PLATFORMS=cpu` is always allowed.
- [ ] A: both methods use the same y, init and AdamCfg (early stop disabled) at B=1; only the forward differs. `n_vjp == iters`
      is asserted for both methods and n_failed is reported. Wall time excludes compile, which is reported separately.
- [ ] A: truth and both fits are rolled out with the same A01 judge from tau 0 to 30 at dt 0.25. Per-tau RMSE u/v is cached
      for every case; frames are cached only for the video cases.
- [ ] A: per-case mp4s for the video cases (Truth | F02 | fast; u, u residual, v, v residual; per-panel RMSE; VJPs, measured
      fit time and measured speedup in the titles), a montage mp4, the figure (RMSE vs tau, loss vs iteration and vs wall
      time), and a summary table with the measured speedup.
- [ ] A: `--plot-only` regenerates everything without refitting, recomputing rollouts only when frames are missing.
- [ ] B0 FD/JVP checks for all 6 params pass on CPU for lawson4, etdrk4, strang4, rk4 and F02 before any GPU step.
- [ ] B1: for all listed settings and points, normalised per-param errors, relative errors where the reference is not tiny,
      sign agreement, 6-vector cosine, and the state-block cosine vs REF. ms per theta/joint VJP is reported, the
      per-setting cap is enforced, and failures are recorded, not fatal.
- [ ] B2: D_max_F02 is computed from dx and all starts are asserted below it. Theta is recovered from 3 perturbed starts per
      case with BoundedLogRDParams(direct) and decayed Adam for F02, lawson4_d0.2 and lawson4_d0.1 at the same VJP count.
      Per-param relative errors, loss curves, wall time, and separate instability / not-converged / at-bound columns are
      reported.
- [ ] B: summary.md, the per-param error bar chart, and the recovery trajectory plot (truth line, failure markers, D_max
      line) are written and logged to W&B.
- [ ] Both scripts resume from per-unit caches; per-case, per-setting and per-fit progress lines are printed and logged live.
- [ ] Media copied to `.claude/plans/fastsim-fit-compare-media/` (`fit_compare_*`, `param_*`).
- [ ] Both new test files pass on CPU and the full `cardiac/tests` suite still passes. No existing file is modified and no
      dependency is added.

## Scope notes
- Added beyond the request:
  - A: the loss-vs-iteration and loss-vs-wall-time panels, and the F02-vs-stored-adam and F02-vs-fast IC diffs.
  - B1 includes `lawson4_d0.2_unroll4`, so A's exact setting is covered.
  - `--no-render`/`--plot-only`, to shorten the GPU pause.
- A renders per-case videos for 4 cases by default (`--video-cases`) to keep caches small. RMSE curves, the figure and the
  table cover all 16 cases. `--video-cases 16` renders all of them.
- Case loading is duplicated (it is small) in both scripts so the tracks share no files.
- Not done:
  - Batched fits: B=1 gives clean per-fit times.
  - Joint (theta + v0) recovery: it would double B2's F02 time, so it is left as a possible follow-up flag.
- Next: 3D (RDSystem3D) is explicitly out of scope and planned after this follow-up finishes. It would reuse B's traced-theta
  harness and B0's FD tests, and A's compare/video pipeline, on a 3D forward.

## Risks / open questions
- There are no data deletions, migrations, new dependencies or public API changes (new files only).
- Eval dir location (repo-relative vs `/home/alex/physics-guidance/...`): the fallback is coded, and implementers verify
  which one exists.
- Passing a traced theta through `physics.make_forward`/`RDSystem` is unverified; B0 checks it first. If RDSystem casts
  params to Python floats, B needs a wrapper (possible extra work).
- B0 may reveal a fast scheme whose operators are built from a static D (giving a wrong d/dD). That scheme is then excluded
  from B1/B2 and flagged. It would also be a correctness issue for any future theta fitting with fastsim.
- The direct mapping clips at the bounds, which gives a zero gradient there. A param that hits a widened bound freezes;
  this is reported as at_bound.
- F02 can drift past D_max_F02 during recovery because the widened bounds allow up to 2x truth. That is a reported finding
  (instability column), not a bug.
- The REF param-VJP cost at n=224 is unmeasured. If it exceeds ~30 s each, rerun B1 with `--cases 2`.
- If eval_test_n20's Adam did not use F02, A's F02 fit will not match the stored `adam_final_*`; the diff is reported.
- Timings require an otherwise idle GPU (training paused).
