"""Equal-VJP initial-condition fit, original F02 forward vs a fast fixed-step solver, judged by one common rollout.

For the headline test cases (frame 128, u observed at tau = 1..10) the state (u0, v0) is fitted with the repo's basic Adam
(`baseline.AdamFitter`, 200 iterations = 200 VJPs, early stopping off, B=1) twice, identical except for the forward operator:
  F02   diffrax Tsit5, fixed dt 0.02, `physics.make_forward`
  fast  `fastsim.make_forward_fast` (default lawson4_d0.2_unroll4)
The truth and both fits are then rolled out from tau = 0 to --tmax with the same A01 judge solver (rtol/atol 1e-3, dtmax 0.01,
saved every --dt); per-tau RMSE of u and v is cached for every case, frames only for the first --video-cases cases.

  python -m cardiac.fastsim_fit_compare --force --no-render        # real job (GPU): fits, rollouts, caches, table, figure
  JAX_PLATFORMS=cpu python -m cardiac.fastsim_fit_compare --plot-only   # videos / montage / figure / table from the caches

Caches in --out (resumable, atomic writes, existing files are skipped unless --redo):
  meta.json, fit_{f02,fast}_<case>.npz, roll_<case>.npz, frames_<case>.npz (float16; video cases only).
Outputs: <out>/videos/fit_compare_<case>.mp4, fit_compare_montage.mp4, <out>/fit_compare.{png,pdf}, summary.{md,json};
the montage, figure, summary and the first 3 case videos are copied to --media-dir as fit_compare_*.
The GPU guard (bench_fastsim.gpu_guard) refuses to run on the GPU while the training job's PID exists unless --force.
"""
import argparse
import dataclasses
import hashlib
import json
import os
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np

from . import bench_fastsim as bf
from .rollout_compare import rmse_curve

DEFAULT_EVAL = "cardiac/runs/eval_test_n20"
FALLBACK_EVAL = "/home/alex/physics-guidance/cardiac/runs/eval_test_n20"
FIXED_COLORS = {"truth": "#000000", "F02": "#1f77b4", "lawson4_d0.2": "#d62728", "lawson4_d0.1": "#ff7f0e"}
PALETTE = ["#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#98df8a", "#c5b0d5"]
TAG = "[fit_compare]"


def say(s):
    print(f"{TAG} {s}", flush=True)


def base_name(name):
    return name.split("_unroll")[0].split("_remat")[0]


def color_for(name):
    """Fixed colours for truth / F02 / lawson4_d0.2 / lawson4_d0.1; any other setting a palette colour keyed by its name."""
    b = base_name(name)
    if b in FIXED_COLORS:
        return FIXED_COLORS[b]
    return PALETTE[int(hashlib.sha1(b.encode()).hexdigest(), 16) % len(PALETTE)]


def linestyle_for(name):
    return "-" if name in ("F02", "truth") else "--"


def fast_label(cfg):
    return f"{cfg.scheme.capitalize()}, dt {cfg.dt:g}"


def fmt_speedup(sp):
    return f"{sp:.0f}x" if sp >= 10 else f"{sp:.1f}x"


def case_system(nm):
    return nm.rsplit("_f", 1)[0]


# ---------------------------------------------------------------------------------------------------------------- CLI
def make_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval", default=DEFAULT_EVAL, help=f"expand_eval dir; falls back to {FALLBACK_EVAL} if absent")
    p.add_argument("--cases", type=int, default=16, help="first N headline cases")
    p.add_argument("--fast", default="lawson4_d0.2_unroll4", help="fastsim setting name")
    p.add_argument("--iters", type=int, default=None, help="Adam iterations (default: the eval's adam_iters, 200)")
    p.add_argument("--tmax", type=float, default=30.0)
    p.add_argument("--dt", type=float, default=0.25)
    p.add_argument("--taus-max", type=float, default=None,
                   help="use only the observation taus <= this (CPU smoke / tests; default: all, 1..10)")
    p.add_argument("--out", default="cardiac/runs/fastsim_fit_compare")
    p.add_argument("--media-dir", default=".claude/plans/fastsim-fit-compare-media")
    p.add_argument("--video-cases", type=int, default=4, help="first K cases get per-case videos and form the montage")
    p.add_argument("--no-render", action="store_true", help="do not write the mp4 files (figure and table are still written)")
    p.add_argument("--plot-only", action="store_true", help="re-render from the caches; never refits")
    p.add_argument("--redo", action="store_true", help="ignore the caches (and a meta.json mismatch)")
    p.add_argument("--force", action="store_true", help="run on the GPU even while the guarded PID exists")
    p.add_argument("--train-pid", type=int, default=bf.GUARD_PID)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-name", default="fastsim_fit_compare")
    return p


def guard(a):
    if bf.gpu_guard(os.environ, a.train_pid, a.force) == "refuse":
        print(f"REFUSING to run: JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS')!r} is not CPU-only and PID {a.train_pid} "
              f"(the training job) exists, so the GPU is in use. Run with JAX_PLATFORMS=cpu, wait for the job, or pass "
              f"--force.", file=sys.stderr, flush=True)
        sys.exit(2)


def make_wandb(a, config):
    if a.no_wandb:
        from .progress import NullWB
        return NullWB()
    from .progress import make_wb
    return make_wb(a.wandb_name + ("_plots" if a.plot_only else ""), config=config)


# ---------------------------------------------------------------------------------------------------------------- atomics
def save_npz(path, **kw):
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, **kw)
    os.replace(tmp, path)


def save_npz_compressed(path, **kw):
    tmp = path[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **kw)
    os.replace(tmp, path)


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def load_npz(path):
    with np.load(path) as z:
        return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


# ---------------------------------------------------------------------------------------------------------------- loading
def resolve_eval(path):
    if os.path.isdir(path):
        say(f"using eval dir {path}")
        return path
    if os.path.isdir(FALLBACK_EVAL):
        say(f"eval dir {path} not found; using fallback {FALLBACK_EVAL}")
        return FALLBACK_EVAL
    raise FileNotFoundError(f"eval dir not found: {path} (fallback {FALLBACK_EVAL} also missing)")


def load_ctx(a):
    """Everything the fits / rollouts need (imports jax). Cases are the first --cases headline cases of the eval."""
    from . import physics
    from .data import Normalizer
    from .final_figs import load_eval
    ev = resolve_eval(a.eval)
    M, where, ra = load_eval(ev, headline=True)
    names = list(M)[:a.cases]
    assert names, f"no headline cases in {ev}"
    assert (ra.get("v_init") or "mean") == "mean" and (ra.get("u_init") or "mask_mean") == "mask_mean", \
        "eval used a non-default init; this script reproduces run_eval.run_adam with the mean-v / clip(y1) init only"
    assert not ra.get("obs_mask") and not ra.get("sparse") and (ra.get("ic_time") or "frame") == "frame"
    with open(bf.resolve_stats(None, ra, ev)) as f:
        stats = json.load(f)
    dpath = os.path.join(ra["data_root"], "config.json")
    with open(dpath) as f:
        dcfg = json.load(f)
    n, grid = int(dcfg["n"]), tuple(float(g) for g in dcfg["grid"])
    obs_taus = np.asarray(physics.observation_taus(ra["tau1"], ra["spacing"], ra["M"]), np.float32)
    if a.taus_max is not None:
        obs_taus = obs_taus[obs_taus <= a.taus_max + 1e-6]
    k = len(obs_taus)
    assert k >= 1
    sigma = ra["noise_std"] if ra["noise_std"] > 0 else 1.0
    split = json.load(open(ra.get("split") or "cardiac/artifacts/split.json"))[ra.get("subset", "test")]
    data = {}
    for nm in names:
        z = np.load(os.path.join(where[nm], f"estimates_{nm}.npz"))
        y = z["y"]
        if hashlib.sha1(y.tobytes()).hexdigest() != M[nm]["y_sha1"]:
            raise RuntimeError(f"y hash mismatch for {nm}")
        data[nm] = SimpleNamespace(y=np.asarray(y[:k], np.float32), u_true=np.asarray(z["u_true"], np.float32),
                                   v_true=np.asarray(z["v_true"], np.float32),
                                   stored_u0=np.asarray(z["adam_final_u0"], np.float32),
                                   stored_v0=np.asarray(z["adam_final_v0"], np.float32),
                                   gidx=int(split.index(M[nm].get("system", case_system(nm)))))
    return SimpleNamespace(
        eval=ev, ra=ra, names=names, data=data, n=n, grid=grid, obs_taus=obs_taus, stats=stats,
        norm=Normalizer.from_stats(stats), fp=physics.FixedAPParams.from_config(dpath),
        loss_fn=physics.obs_loss(ra["loss"], sigma, physics.FullField(), n), n_obs=k * n * n)


def build_meta(a, ctx, iters):
    import jax
    from .config import SolverCfg
    from .fastsim import parse_setting
    from .solver_sweep import settings_table
    ckpt = ctx.ra.get("checkpoints") or 1024
    f02 = dataclasses.replace(settings_table()["F02"], checkpoints=ckpt)
    fast = parse_setting(a.fast)
    judge = SolverCfg(rtol=1e-3, atol=1e-3, dtmax=0.01, stepping="adaptive", checkpoints=None)
    dev = jax.devices()[0]
    return {"cases": ctx.names, "global_idx": {nm: ctx.data[nm].gidx for nm in ctx.names},
            "adam": {"lr": float(ctx.ra["adam_lr"]), "iters": int(iters), "stop": "none", "snapshot_iter": 0},
            "f02": f02.to_dict(), "fast_name": a.fast, "fast": fast.to_dict(), "judge": judge.to_dict(),
            "tmax": float(a.tmax), "dt": float(a.dt), "obs_taus": [float(t) for t in ctx.obs_taus],
            "obs_end": float(ctx.obs_taus.max()), "n": ctx.n, "n_obs": int(ctx.n_obs),
            "loss": ctx.ra["loss"], "noise_std": float(ctx.ra["noise_std"]), "eval": ctx.eval,
            "device": f"{dev.platform}:{dev.device_kind}", "volatile": ["eval", "device"]}


def check_meta(a, meta):
    """Write meta.json, or refuse if an existing one differs (ignoring the volatile keys) unless --redo."""
    path = os.path.join(a.out, "meta.json")
    if os.path.exists(path) and not a.redo:
        old = json.load(open(path))
        vol = set(meta["volatile"])
        diff = sorted(k for k in set(old) | set(meta) if k not in vol and old.get(k) != meta.get(k))
        if diff:
            raise SystemExit(f"{path} differs from this run in {diff}; use --redo or another --out")
    save_json(path, meta)


# ---------------------------------------------------------------------------------------------------------------- fits
def make_fitters(a, ctx, meta):
    from .baseline import AdamFitter
    from .config import AdamCfg, SolverCfg
    from .fastsim import make_forward_fast, parse_setting
    from .physics import make_forward
    cfg = AdamCfg(lr=meta["adam"]["lr"], iters=meta["adam"]["iters"], snapshot_iter=0, stop="none")
    f02_fwd = make_forward(ctx.n, ctx.grid, ctx.obs_taus, SolverCfg.from_dict(meta["f02"]), ctx.fp)
    fast_fwd = make_forward_fast(ctx.n, ctx.grid, ctx.obs_taus, parse_setting(a.fast), ctx.fp)
    return {"f02": AdamFitter(f02_fwd, ctx.loss_fn, ctx.norm, cfg, ctx.n),
            "fast": AdamFitter(fast_fwd, ctx.loss_fn, ctx.norm, cfg, ctx.n)}


def rmse(a_, b_):
    return float(np.sqrt(np.mean((np.asarray(a_, np.float64) - np.asarray(b_, np.float64)) ** 2)))


def do_fit(fitter, d, iters, with_stored):
    iu = np.clip(d.y[0], 0.0, 1.0).astype(np.float32)  # = run_eval.init_u / init_v ("mean")
    iv = np.full_like(iu, d.v_mean)
    r = fitter.fit(d.y, iu, iv)
    assert int(r["n_vjp"]) == iters, f"n_vjp {r['n_vjp']} != iters {iters}"
    out = {"u0": np.asarray(r["u0"], np.float32), "v0": np.asarray(r["v0"], np.float32),
           "loss_hist": np.asarray(r["loss_hist"], np.float64), "final_L_obs": float(r["final_L_obs"]),
           "wall_time_s": float(r["wall_time_s"]), "compile_time_s": float(r["compile_time_s"]),
           "n_vjp": int(r["n_vjp"]), "n_failed": int(r["n_failed"]),
           "rmse_u0": rmse(r["u0"], d.u_true), "rmse_v0": rmse(r["v0"], d.v_true)}
    if with_stored:
        out["diff_stored_u0"] = float(np.max(np.abs(out["u0"] - d.stored_u0)))
        out["diff_stored_v0"] = float(np.max(np.abs(out["v0"] - d.stored_v0)))
    return out


# ---------------------------------------------------------------------------------------------------------------- rollouts
def make_judge(ctx, meta):
    import jax
    from .config import SolverCfg
    from .physics import make_forward
    taus = np.arange(meta["dt"], meta["tmax"] + 1e-6, meta["dt"], dtype=np.float32)
    fwd = jax.jit(make_forward(ctx.n, ctx.grid, taus, SolverCfg(**meta["judge"]), ctx.fp))
    return taus, fwd


def rollout_case(judge, d, ics):
    """-> trajectories {truth,f02,fast: (u[T+1,n,n], v[T+1,n,n])} (float32, tau=0 state prepended) and ok flags."""
    import jax.numpy as jnp
    _, fwd = judge
    traj, ok = {}, {}
    for key, (u0, v0) in (("truth", (d.u_true, d.v_true)), *ics.items()):
        U, V, info = fwd(jnp.asarray(u0, jnp.float32), jnp.asarray(v0, jnp.float32))
        traj[key] = (np.concatenate([np.asarray(u0, np.float32)[None], np.asarray(U)]),
                     np.concatenate([np.asarray(v0, np.float32)[None], np.asarray(V)]))
        ok[key] = bool(info["ok"])
    return traj, ok


def curves_of(traj):
    out = {}
    for key in ("f02", "fast"):
        for j, var in enumerate("uv"):
            out[f"rmse_{var}_{key}"] = rmse_curve(traj[key][j], traj["truth"][j])
    return out


def save_frames(path, traj):
    kw = {f"{var}_{key}": traj[key][j].astype(np.float16) for key in ("truth", "f02", "fast") for j, var in enumerate("uv")}
    save_npz_compressed(path, **kw)


def load_frames(path):
    z = load_npz(path)
    return {key: (z[f"u_{key}"].astype(np.float32), z[f"v_{key}"].astype(np.float32)) for key in ("truth", "f02", "fast")}


# ---------------------------------------------------------------------------------------------------------------- results
def at_tau(taus, t):
    return int(np.argmin(np.abs(np.asarray(taus) - t)))


def case_log(res, taus):
    """Flat dict of the per-case metrics (W&B and progress line)."""
    f2, fa = res["f02"], res["fast"]
    d = {"speedup": f2["wall_time_s"] / max(fa["wall_time_s"], 1e-9),
         "max_abs_diff_u0_f02_fast": float(np.max(np.abs(f2["u0"] - fa["u0"]))),
         "max_abs_diff_v0_f02_fast": float(np.max(np.abs(f2["v0"] - fa["v0"])))}
    for key, r in (("f02", f2), ("fast", fa)):
        for k in ("wall_time_s", "compile_time_s", "n_vjp", "n_failed", "final_L_obs", "rmse_u0", "rmse_v0"):
            d[f"{key}/{k}"] = float(r[k])
        for var in "uv":
            c = res["roll"][f"rmse_{var}_{key}"]
            for lab, t in (("10", 10.0), ("30", 30.0)):
                d[f"{key}/rmse_{var}@{lab}"] = float(c[at_tau(taus, t)])
            d[f"{key}/rmse_{var}@end"] = float(c[-1])
    return d


def progress_line(i, N, nm, res, d, elapsed, cached):
    f2, fa = res["f02"], res["fast"]
    return (f"case {i + 1}/{N} {nm}{' (cached)' if cached else ''}: F02 {f2['n_vjp']} VJP {f2['wall_time_s']:.1f}s "
            f"L={f2['final_L_obs']:.4g} u0 {f2['rmse_u0']:.3f} v0 {f2['rmse_v0']:.3f} nfail {f2['n_failed']} | "
            f"fast {fa['n_vjp']} VJP {fa['wall_time_s']:.1f}s L={fa['final_L_obs']:.4g} u0 {fa['rmse_u0']:.3f} "
            f"v0 {fa['rmse_v0']:.3f} nfail {fa['n_failed']} | {fmt_speedup(d['speedup'])} | "
            f"v@end {d['f02/rmse_v@end']:.3f}/{d['fast/rmse_v@end']:.3f} | elapsed {elapsed / 60:.1f} min")


def load_results(out, meta):
    """Complete cases (both fits + roll caches) -> {case: {f02, fast, roll}}, taus (with tau=0)."""
    res, taus = {}, None
    for nm in meta["cases"]:
        ps = {"f02": f"fit_f02_{nm}.npz", "fast": f"fit_fast_{nm}.npz", "roll": f"roll_{nm}.npz"}
        if not all(os.path.exists(os.path.join(out, p)) for p in ps.values()):
            continue
        r = {k: load_npz(os.path.join(out, p)) for k, p in ps.items()}
        taus = r["roll"]["taus"]
        res[nm] = r
    return res, taus


# ---------------------------------------------------------------------------------------------------------------- the run
def run_fits(a, wb, ctx, meta):
    iters = meta["adam"]["iters"]
    path = lambda kind, nm: os.path.join(a.out, f"{kind}_{nm}.npz")  # noqa: E731
    have = lambda p: os.path.exists(p) and not a.redo  # noqa: E731
    fitters = judge = None
    t0, N = time.time(), len(ctx.names)
    for i, nm in enumerate(ctx.names):
        d = ctx.data[nm]
        d.v_mean = float(ctx.stats["v"]["mean"])
        fresh = False
        res = {}
        for key in ("f02", "fast"):
            p = path(f"fit_{key}", nm)
            if have(p):
                res[key] = load_npz(p)
                continue
            fitters = fitters or make_fitters(a, ctx, meta)
            r = do_fit(fitters[key], d, iters, key == "f02")
            save_npz(p, **r)
            res[key] = r
            fresh = True
        need_frames = i < a.video_cases and not have(path("frames", nm))
        if fresh or not have(path("roll", nm)) or need_frames:
            judge = judge or make_judge(ctx, meta)
            traj, ok = rollout_case(judge, d, {k: (res[k]["u0"], res[k]["v0"]) for k in ("f02", "fast")})
            roll = {"taus": np.concatenate([[0.0], judge[0]]).astype(np.float32), **curves_of(traj),
                    **{f"ok_{k}": np.array(v) for k, v in ok.items()}}
            save_npz(path("roll", nm), **roll)
            if i < a.video_cases:
                save_frames(path("frames", nm), traj)
            res["roll"] = roll
        else:
            res["roll"] = load_npz(path("roll", nm))
        dd = case_log(res, res["roll"]["taus"])
        wb.log({"case_idx": i, **{f"case/{k}": v for k, v in dd.items()}})
        say(progress_line(i, N, nm, res, dd, time.time() - t0, not fresh))


# ---------------------------------------------------------------------------------------------------------------- summary
def summarize(meta, res, taus):
    """Rows per case + mean/std; values in plain floats (json) -> (columns, rows, json dict)."""
    T = float(taus[-1])
    ts = [("10", at_tau(taus, 10.0)), ("30", at_tau(taus, 30.0)), ("end", len(taus) - 1)]
    cols = ["u0 RMSE", "v0 RMSE"] + [f"{v} RMSE @{lab}" for lab, _ in ts for v in "uv"] + [
        "final L_obs", "VJPs", "n_failed", "wall s", "compile s", "speedup", "max|dU0| F02-fast", "max|dV0| F02-fast",
        "max|dU0| F02-stored", "max|dV0| F02-stored"]

    def row(r):
        out = []
        for key in ("f02", "fast"):
            f = r[key]
            vals = [f["rmse_u0"], f["rmse_v0"]]
            for _, j in ts:
                for var in "uv":
                    vals.append(float(r["roll"][f"rmse_{var}_{key}"][j]))
            vals += [f["final_L_obs"], f["n_vjp"], f["n_failed"], f["wall_time_s"], f["compile_time_s"]]
            out.append(vals)
        sp = r["f02"]["wall_time_s"] / max(r["fast"]["wall_time_s"], 1e-9)
        extra = [float(np.max(np.abs(r["f02"]["u0"] - r["fast"]["u0"]))), float(np.max(np.abs(r["f02"]["v0"] - r["fast"]["v0"]))),
                 float(r["f02"].get("diff_stored_u0", np.nan)), float(r["f02"].get("diff_stored_v0", np.nan))]
        return out[0], out[1], sp, extra

    per = {nm: row(r) for nm, r in res.items()}
    js = {"columns": cols, "tau_end": T, "cases": {}}
    for nm, (a_, b_, sp, ex) in per.items():
        js["cases"][nm] = {"f02": dict(zip(cols, a_)), "fast": dict(zip(cols, b_)), "speedup": sp,
                           "diff": dict(zip(cols[-4:], ex))}
    A = np.array([p[0] for p in per.values()], float)
    B = np.array([p[1] for p in per.values()], float)
    S = np.array([p[2] for p in per.values()], float)
    X = np.array([p[3] for p in per.values()], float)
    js["mean"] = {"f02": dict(zip(cols, A.mean(0))), "fast": dict(zip(cols, B.mean(0))), "speedup": float(S.mean())}
    js["std"] = {"f02": dict(zip(cols, A.std(0))), "fast": dict(zip(cols, B.std(0))), "speedup": float(S.std())}
    js["speedup_of_mean_wall"] = float(A[:, cols.index("wall s")].mean() / max(B[:, cols.index("wall s")].mean(), 1e-9))
    return cols, per, (A, B, S, X), js


def write_summary(a, meta, res, taus, path_md, path_json):
    cols, per, (A, B, S, X), js = summarize(meta, res, taus)
    f = lambda x: "nan" if not np.isfinite(x) else (f"{x:.4g}")  # noqa: E731
    ncol = len(cols) - 5  # per-method columns; then speedup; then the 4 diff columns
    sel = list(range(ncol))
    hdr = ["case"] + cols[:ncol] + ["speedup (F02/fast wall)"] + cols[ncol + 1:]
    # F02 / fast in one cell; integer columns (VJPs, n_failed) shown as ints
    ints = {cols.index("VJPs"), cols.index("n_failed")}

    def cell(i, x, y):
        if i in ints:
            return f"{int(x)} / {int(y)}"
        return f"{f(x)} / {f(y)}"

    def mrow(label, A_, B_, sp, ex):
        return ([label] + [cell(i, A_[i], B_[i]) for i in sel] + [sp] + [f(x) for x in ex])

    lines = [f"| {' | '.join(hdr)} |", "|" + "---|" * len(hdr)]
    for k, (nm, (a_, b_, sp, ex)) in enumerate(per.items()):
        lines.append("| " + " | ".join(mrow(nm, a_, b_, f"{sp:.1f}x", ex)) + " |")
    ms = [f"{f(A[:, i].mean())}±{f(A[:, i].std())} / {f(B[:, i].mean())}±{f(B[:, i].std())}" for i in sel]
    lines.append("| " + " | ".join(["mean ± std"] + ms + [f"{S.mean():.1f}±{S.std():.1f}x"] +
                                    [f"{f(np.nanmean(X[:, j]))}±{f(np.nanstd(X[:, j]))}" for j in range(4)]) + " |")
    fa = meta["fast_name"]
    md = [f"# Equal-VJP IC fit: original F02 vs {fa} ({len(per)} cases)", "",
          f"- Device: {meta['device']}. Judge: A01 (rtol/atol 1e-3, dtmax 0.01), saved every {meta['dt']:g} from tau 0 to "
          f"{meta['tmax']:g}; RMSE is vs the same rollout from the true state.",
          f"- F02: diffrax Tsit5 fixed dt 0.02 (checkpoints {meta['f02']['checkpoints']}); fast: `{fa}`.",
          f"- Identical y (obs taus {meta['obs_taus'][0]:g}..{meta['obs_taus'][-1]:g}), init (clip(y1,0,1), train-mean v), "
          f"Adam lr {meta['adam']['lr']:g}, {meta['adam']['iters']} iterations, early stop off, B=1. Only the forward differs.",
          "- Wall time excludes compile (compile is paid by the first fit of each method in the process, later fits report 0).",
          "- Cells are `F02 / fast`; the speedup is F02 wall / fast wall of the same case (measured). "
          f"Mean wall: F02 {A[:, cols.index('wall s')].mean():.1f} s, fast {B[:, cols.index('wall s')].mean():.1f} s "
          f"({js['speedup_of_mean_wall']:.1f}x of the means).",
          f"- RMSE @10 / @30 are the rollout RMSE at tau 10 / 30 (nearest saved tau); @end is tau = {js['tau_end']:g}.", "",
          *lines]
    open(path_md, "w").write("\n".join(md) + "\n")
    save_json(path_json, js)
    rows = [[nm, *[float(x) for x in a_], *[float(x) for x in b_], float(sp), *ex] for nm, (a_, b_, sp, ex) in per.items()]
    wb_cols = ["case"] + [f"f02 {c}" for c in cols[:ncol]] + [f"fast {c}" for c in cols[:ncol]] + ["speedup"] + cols[ncol + 1:]
    return wb_cols, rows


# ---------------------------------------------------------------------------------------------------------------- figure
def make_figure(a, meta, res, taus, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    names = [("f02", "Original F02 (diffrax Tsit5, dt 0.02)"), ("fast", f"Fast {fast_label(_fast_cfg(meta))}")]
    cols = {"f02": color_for("F02"), "fast": color_for(meta["fast_name"])}
    lss = {"f02": "-", "fast": "--"}
    fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
    for axx, var, ttl in ((ax[0, 0], "u", "Voltage u"), (ax[0, 1], "v", "Recovery v")):
        for key, lab in names:
            Y = np.stack([r["roll"][f"rmse_{var}_{key}"] for r in res.values()])
            axx.fill_between(taus, np.percentile(Y, 25, 0), np.percentile(Y, 75, 0), color=cols[key], alpha=0.15, lw=0)
            axx.plot(taus, Y.mean(0), color=cols[key], ls=lss[key], lw=2, label=lab)
        axx.axvspan(0, meta["obs_end"], color="0.9", zorder=0, label="Observation window")
        axx.set_yscale("log")
        axx.set_xlabel(r"time $\tau$ after the estimated state")
        axx.set_ylabel(f"RMSE of {var}($\\tau$) vs true rollout")
        axx.set_title(f"{ttl}: mean and 25-75% over {len(res)} cases")
        axx.grid(alpha=0.25)
    ax[0, 0].legend(frameon=False, fontsize=9)
    half = meta["n_obs"] / 2.0
    for key, lab in names:
        H = np.stack([r[key]["loss_hist"] for r in res.values()]) / half
        ax[1, 0].plot(np.arange(1, H.shape[1] + 1), H.mean(0), color=cols[key], ls=lss[key], lw=2, label=lab)
        W = np.stack([np.linspace(0, r[key]["wall_time_s"], H.shape[1] + 1)[1:] for r in res.values()])
        ax[1, 1].plot(W.mean(0), H.mean(0), color=cols[key], ls=lss[key], lw=2, label=lab)
    ax[1, 0].set_xlabel("Adam iteration (1 VJP each)")
    ax[1, 1].set_xlabel("fit wall time [s], excluding compile")
    ax[1, 1].set_xscale("log")
    for axx in ax[1]:
        axx.set_yscale("log")
        axx.set_ylabel(r"observation loss / (N$_{obs}$/2)  (1 = noise level)")
        axx.grid(alpha=0.25, which="both")
    ax[1, 0].set_title("Observation loss vs iteration (mean over cases)")
    ax[1, 1].set_title("Loss vs wall time (uniform cost per iteration)")
    sp = np.mean([r["f02"]["wall_time_s"] / max(r["fast"]["wall_time_s"], 1e-9) for r in res.values()])
    fig.suptitle(f"Equal-VJP Adam fit of (u0, v0): F02 vs {meta['fast_name']}; mean measured fit speedup {fmt_speedup(sp)}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(png, dpi=160)
    fig.savefig(png[:-4] + ".pdf")
    plt.close(fig)


def _fast_cfg(meta):
    from .fastsim import parse_setting
    return parse_setting(meta["fast_name"])


# ---------------------------------------------------------------------------------------------------------------- videos
def even_fig(plt, w_in, h_in, dpi, nrows, ncols, **kw):
    """Figure with even pixel dimensions (libx264 / yuv420p)."""
    w = int(round(w_in * dpi / 2)) * 2 / dpi
    h = int(round(h_in * dpi / 2)) * 2 / dpi
    return plt.subplots(nrows, ncols, figsize=(w, h), dpi=dpi, **kw)


def phase_caption(tau, obs_end):
    if tau == 0:
        return "initial state (estimated)", "k"
    if tau <= obs_end + 1e-6:
        return f"inside observation window (u observed at tau=1..{obs_end:g})", "k"
    return "prediction beyond the window (no correction)", "#b03a2e"


def render_case_video(path, system, T, traj, titles, tcolors, obs_end, fps=15, dpi=90, res_u=0.5, res_v=1.0):
    """Columns Truth | F02 | fast; rows u, u residual, v, v residual (coevolve_video layout)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    keys = ["truth", "f02", "fast"]
    K = len(keys)
    fig, ax = even_fig(plt, 2.9 * K + 0.6, 12.9, dpi, 4, K)
    ut, vt = traj["truth"]
    ims, labs = [], []
    for j, key in enumerate(keys):
        U, V = traj[key]
        row = [ax[0, j].imshow(U[0], cmap="Reds", vmin=0, vmax=1), None, ax[2, j].imshow(V[0], cmap="Blues", vmin=0, vmax=2.2), None]
        if key != "truth":
            row[1] = ax[1, j].imshow(U[0] - ut[0], cmap="RdBu_r", vmin=-res_u, vmax=res_u)
            row[3] = ax[3, j].imshow(V[0] - vt[0], cmap="RdBu_r", vmin=-res_v, vmax=res_v)
        else:
            for r in (1, 3):
                ax[r, j].set_facecolor("0.93")
                ax[r, j].text(0.5, 0.5, "residual =\nestimate - truth\n\nred: estimate too high\nblue: estimate too low",
                              ha="center", va="center", transform=ax[r, j].transAxes, fontsize=9, color="0.35")
        ims.append(row)
        ax[0, j].set_title(titles[j], fontsize=10, color=tcolors[j])
        labs.append([ax[r, j].text(0.03, 0.03, "", fontsize=8, color="k", backgroundcolor="w", transform=ax[r, j].transAxes)
                     for r in range(4)])
        for r in range(4):
            ax[r, j].set_xticks([]), ax[r, j].set_yticks([])
    ax[0, 0].set_ylabel("u (voltage)"), ax[1, 0].set_ylabel(f"u residual (+-{res_u})")
    ax[2, 0].set_ylabel("v (recovery, never observed)"), ax[3, 0].set_ylabel(f"v residual (+-{res_v})")
    sup = fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 0.95, 0.965))
    for r, lim in ((1, res_u), (3, res_v)):
        pos = ax[r, -1].get_position()
        cb = fig.colorbar(ims[-1][r], cax=fig.add_axes([pos.x1 + 0.008, pos.y0, 0.008, pos.height]))
        cb.set_ticks([-lim, 0, lim])
        cb.ax.tick_params(labelsize=8)
    writer = FFMpegWriter(fps=fps, bitrate=4000)
    with writer.saving(fig, path, dpi=dpi):
        for i, tau in enumerate(T):
            for j, key in enumerate(keys):
                U, V = traj[key]
                ims[j][0].set_data(U[i]), ims[j][2].set_data(V[i])
                if key != "truth":
                    du, dv = U[i] - ut[i], V[i] - vt[i]
                    ims[j][1].set_data(du), ims[j][3].set_data(dv)
                    labs[j][1].set_text(f"RMSE {np.sqrt(np.mean(du ** 2)):.3f}")
                    labs[j][3].set_text(f"RMSE {np.sqrt(np.mean(dv ** 2)):.3f}")
            ph, col = phase_caption(tau, obs_end)
            sup.set_text(f"{system}   tau = {tau:5.2f}   -   {ph}")
            sup.set_color(col)
            writer.grab_frame()
    plt.close(fig)


def render_montage(path, systems, T, trajs, title, obs_end, fps=15, dpi=90, res_v=1.0, col_titles=None, tcolors=None):
    """Rows = cases; columns Truth v | F02 v | fast v | F02 v residual | fast v residual; one shared tau clock."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    R = len(systems)
    fig, ax = even_fig(plt, 2.4 * 5 + 0.8, 2.4 * R + 1.5, dpi, R, 5, squeeze=False)
    ims, labs = [], []
    for r in range(R):
        tr = trajs[r]
        vt = tr["truth"][1]
        row, lab = [], []
        for j, key in enumerate(("truth", "f02", "fast")):
            row.append(ax[r, j].imshow(tr[key][1][0], cmap="Blues", vmin=0, vmax=2.2))
        for j, key in ((3, "f02"), (4, "fast")):
            row.append(ax[r, j].imshow(tr[key][1][0] - vt[0], cmap="RdBu_r", vmin=-res_v, vmax=res_v))
            lab.append(ax[r, j].text(0.03, 0.03, "", fontsize=8, backgroundcolor="w", transform=ax[r, j].transAxes))
        ims.append(row)
        labs.append(lab)
        ax[r, 0].set_ylabel(systems[r], fontsize=8)
        for j in range(5):
            ax[r, j].set_xticks([]), ax[r, j].set_yticks([])
    for j in range(5):
        ax[0, j].set_title(col_titles[j], fontsize=9, color=tcolors[j])
    sup = fig.suptitle("")
    fig.tight_layout(rect=(0, 0.02, 1, 0.9))
    fig.text(0.01, 0.005, f"v: Blues 0-2.2 (never observed); residual = estimate - truth, +-{res_v:g}; panel text = RMSE",
             fontsize=7, ha="left", va="bottom")
    writer = FFMpegWriter(fps=fps, bitrate=4000)
    with writer.saving(fig, path, dpi=dpi):
        for i, tau in enumerate(T):
            for r in range(R):
                tr = trajs[r]
                vt = tr["truth"][1]
                for j, key in ((0, "truth"), (1, "f02"), (2, "fast")):
                    ims[r][j].set_data(tr[key][1][i])
                for q, (j, key) in enumerate(((3, "f02"), (4, "fast"))):
                    dv = tr[key][1][i] - vt[i]
                    ims[r][j].set_data(dv)
                    labs[r][q].set_text(f"RMSE {np.sqrt(np.mean(dv ** 2)):.3f}")
            ph, col = phase_caption(tau, obs_end)
            sup.set_text(f"{title}\ntau = {tau:5.2f}   -   {ph}")
            sup.set_color(col)
            writer.grab_frame()
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------- outputs
def make_outputs(a, wb, meta, ctx=None):
    """Figure + table (always) and the videos (unless --no-render) from the caches; recomputes frames only if missing."""
    out = a.out
    res, taus = load_results(out, meta)
    if not res:
        raise SystemExit(f"no complete case in {out} (need fit_f02/fit_fast/roll caches)")
    if len(res) < len(meta["cases"]):
        say(f"WARNING: only {len(res)}/{len(meta['cases'])} cases complete; outputs cover those")
    files = {}
    png = os.path.join(out, "fit_compare.png")
    make_figure(a, meta, res, taus, png)
    wb_cols, rows = write_summary(a, meta, res, taus, os.path.join(out, "summary.md"), os.path.join(out, "summary.json"))
    say(f"wrote {png}, fit_compare.pdf, summary.md, summary.json")
    files.update(figure=png, summary=os.path.join(out, "summary.md"))
    videos = {}
    if not a.no_render:
        vdir = os.path.join(out, "videos")
        os.makedirs(vdir, exist_ok=True)
        vcases = [nm for nm in meta["cases"][:a.video_cases] if nm in res]
        trajs, judge = {}, None
        fcfg = _fast_cfg(meta)
        for nm in vcases:
            fp_ = os.path.join(out, f"frames_{nm}.npz")
            if os.path.exists(fp_):
                trajs[nm] = load_frames(fp_)
            else:
                say(f"frames_{nm}.npz missing: recomputing the 3 judge rollouts from the cached IC pairs")
                guard(a)
                if ctx is None:
                    a.cases = len(meta["cases"])
                    ctx = load_ctx(a)
                judge = judge or make_judge(ctx, meta)
                d = ctx.data[nm]
                trajs[nm], _ = rollout_case(judge, d, {k: (res[nm][k]["u0"], res[nm][k]["v0"]) for k in ("f02", "fast")})
                save_frames(fp_, trajs[nm])
        T = np.asarray(taus)
        for nm in vcases:
            f2, fa = res[nm]["f02"], res[nm]["fast"]
            sp = f2["wall_time_s"] / max(fa["wall_time_s"], 1e-9)
            titles = ["Ground truth", f"Original: diffrax Tsit5, dt 0.02 (F02)\n{int(f2['n_vjp'])} VJPs, fit {f2['wall_time_s']:.1f} s",
                      f"Fast: {fast_label(fcfg)}\n{int(fa['n_vjp'])} VJPs, fit {fa['wall_time_s']:.1f} s "
                      f"({fmt_speedup(sp)} {'faster' if sp >= 1 else 'speed'}, measured)"]
            p = os.path.join(vdir, f"fit_compare_{nm}.mp4")
            render_case_video(p, case_system(nm), T, trajs[nm], titles,
                              ["k", color_for("F02"), color_for(meta["fast_name"])], meta["obs_end"])
            videos[nm] = p
            say(f"wrote {p}")
        if vcases:
            mw = np.mean([res[nm]["f02"]["wall_time_s"] for nm in vcases])
            mf = np.mean([res[nm]["fast"]["wall_time_s"] for nm in vcases])
            sp = mw / max(mf, 1e-9)
            ct = ["Truth v", f"F02 v ({mw:.1f} s)", f"{fast_label(fcfg)} v ({mf:.1f} s)", "F02 v residual", "fast v residual"]
            tc = ["k", color_for("F02"), color_for(meta["fast_name"]), color_for("F02"), color_for(meta["fast_name"])]
            title = (f"Equal-VJP IC fit ({meta['adam']['iters']} VJPs): mean measured fit time F02 {mw:.1f} s vs "
                     f"{meta['fast_name']} {mf:.1f} s ({fmt_speedup(sp)} faster)")
            p = os.path.join(vdir, "fit_compare_montage.mp4")
            render_montage(p, [case_system(nm) for nm in vcases], T, [trajs[nm] for nm in vcases], title, meta["obs_end"],
                           col_titles=ct, tcolors=tc)
            videos["montage"] = p
            say(f"wrote {p}")
    # W&B (never fatal; NullWB has no wandb attribute) and media copy
    try:
        wb.table("table/fit_compare", wb_cols, rows)
        if hasattr(wb, "wandb"):
            wb.log({"figure/fit_compare": wb.wandb.Image(png),
                    **{f"video/{k}": wb.wandb.Video(p, format="mp4") for k, p in videos.items()}})
    except Exception as e:
        say(f"WARNING: W&B final logging failed ({e!r})")
    if a.media_dir:
        os.makedirs(a.media_dir, exist_ok=True)
        cp = lambda src, dst: shutil.copyfile(src, os.path.join(a.media_dir, dst))  # noqa: E731
        cp(png, "fit_compare_figure.png")
        cp(png[:-4] + ".pdf", "fit_compare_figure.pdf")
        cp(files["summary"], "fit_compare_summary.md")
        if "montage" in videos:
            cp(videos["montage"], "fit_compare_montage.mp4")
        for nm in [k for k in videos if k != "montage"][:3]:
            cp(videos[nm], f"fit_compare_{nm}.mp4")
        say(f"copied media to {a.media_dir}")


def main(argv=None):
    a = make_parser().parse_args(argv)
    if not a.plot_only:
        guard(a)   # before anything touches jax or the disk
    os.makedirs(a.out, exist_ok=True)
    if a.plot_only:
        mp = os.path.join(a.out, "meta.json")
        if not os.path.exists(mp):
            raise SystemExit(f"--plot-only needs {mp} (run the fits first)")
        meta = json.load(open(mp))
        wb = make_wandb(a, {"plot_only": True, "meta": meta})
        make_outputs(a, wb, meta)
        wb.finish()
        return
    ctx = load_ctx(a)
    ra = ctx.ra
    iters = a.iters or int(ra["adam_iters"])
    meta = build_meta(a, ctx, iters)
    check_meta(a, meta)
    say(f"{len(ctx.names)} cases, {iters} Adam iterations (lr {ra['adam_lr']}), obs taus {meta['obs_taus']}, "
        f"fast={a.fast}, device {meta['device']}")
    wb = make_wandb(a, {**{k: v for k, v in meta.items() if k != "volatile"}, "args": vars(a)})
    run_fits(a, wb, ctx, meta)
    make_outputs(a, wb, meta, ctx)
    wb.finish()


if __name__ == "__main__":
    main()
