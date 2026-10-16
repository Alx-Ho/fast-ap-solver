"""Parameter (theta) gradients and theta recovery of the fast solvers (cardiac.fastsim) against the diffrax solvers.

  python -m cardiac.fastsim_param_check grad    # B1: theta/state gradient accuracy vs a reference solver at full scale
  python -m cardiac.fastsim_param_check recover # B2: theta recovery with BoundedLogRDParams(direct) + decayed Adam
  python -m cardiac.fastsim_param_check report  # summary.md + param_grad_error / param_recovery figures (from caches)
  python -m cardiac.fastsim_param_check all     # grad, recover, report

Cases: the first --cases headline cases of --eval (final_figs.load_eval(headline=True)); obs taus 1..10, gaussian loss.
Everything is cached per unit under --out and resumed (grad_results.json per setting, ref_grads.npz, rec_*.npz per fit).
The GPU guard refuses to start on a non-CPU JAX_PLATFORMS while the training job's PID exists unless --force (it only
checks that /proc/<pid> exists; this module never signals any process).
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

from . import bench_fastsim as bf  # noqa: E402  (numpy-only at import time; jax is imported lazily)

PARAM_NAMES = ("D", "eps0", "a", "k", "mu1", "mu2")
FALLBACK_EVAL = "/home/alex/physics-guidance/cardiac/runs/eval_test_n20"
DEFAULT_SETTINGS = ("F02,tsit5_d0.02,rk4_d0.025,lawson4_d0.05,lawson4_d0.1,lawson4_d0.2,lawson4_d0.2_unroll4,"
                    "etdrk4_d0.1,etdrk4_d0.2,strang4_d0.05")
POINTS = ("truth", "theta_pert", "joint")
TINY_REF = 1e-3          # relative errors / signs only where |g_ref_k| >= TINY_REF * ||g_ref|| (log-theta scale)
F02_DT = 0.02
TSIT5_REAL_AXIS = 3.5    # Tsit5 real-axis stability bound; 5-point Laplacian eigenvalues reach 8 D / dx^2

# colours: truth black, F02 / lawson4_d0.2 / lawson4_d0.1 as in the plan; the rest from a fixed palette keyed by name
COLORS = {"truth": "k", "REF": "#7f7f7f", "F02": "#1f77b4", "lawson4_d0.2": "#d62728", "lawson4_d0.1": "#ff7f0e",
          "tsit5_d0.02": "#2ca02c", "rk4_d0.025": "#9467bd", "lawson4_d0.05": "#8c564b", "lawson4_d0.2_unroll4": "#e377c2",
          "etdrk4_d0.1": "#bcbd22", "etdrk4_d0.2": "#17becf", "strang4_d0.05": "#393b79"}
EXTRA = ["#637939", "#8c6d31", "#843c39", "#7b4173", "#3182bd", "#e6550d", "#31a354", "#756bb1"]
LINESTYLES = {"lawson4_d0.2": "--"}


def color_of(name):
    return COLORS.get(name) or EXTRA[int(hashlib.sha1(name.encode()).hexdigest(), 16) % len(EXTRA)]


# ----------------------------------------------------------------------------------------------------------------
# pure helpers (no jax)
# ----------------------------------------------------------------------------------------------------------------
def d_max_f02(dx):
    """Explicit stability limit of F02 (diffrax Tsit5, fixed dt 0.02): lambda_max = 8 D / dx^2 <= 3.5 / dt."""
    return TSIT5_REAL_AXIS * dx ** 2 / (8.0 * F02_DT)


def perturb_factors(seed):
    """Deterministic: each of the 6 params gets factor exp(sign * U(log 1.2, log 1.4)) with a random sign."""
    rng = np.random.default_rng(seed)
    out = {}
    for k in PARAM_NAMES:
        sign = 1.0 if rng.random() < 0.5 else -1.0
        out[k] = float(np.exp(sign * rng.uniform(np.log(1.2), np.log(1.4))))
    return out


def perturbed_theta(true, seed):
    f = perturb_factors(seed)
    return {k: float(true[k]) * f[k] for k in PARAM_NAMES}


def check_starts(true, seeds, dmax):
    """Assert that every perturbed start (even D x 1.4) lies below the F02 stability limit."""
    assert 1.4 * float(true["D"]) < dmax, f"true D x 1.4 = {1.4 * float(true['D']):.3e} >= D_max_F02 = {dmax:.3e}"
    for s in seeds:
        d = perturbed_theta(true, s)["D"]
        assert d < dmax, f"seed {s}: perturbed D {d:.3e} >= D_max_F02 {dmax:.3e}"


def widened_bounds(true):
    """DEFAULT_BOUNDS widened to include 0.5x..2x truth (the physics.load_dataset_rd pattern)."""
    from ap_simulator.params import DEFAULT_BOUNDS
    b = dict(DEFAULT_BOUNDS)
    for k in PARAM_NAMES:
        lo, hi = b[k]
        b[k] = (min(lo, 0.5 * float(true[k])), max(hi, 2.0 * float(true[k])))
    return b


def grad_metrics(theta, g6, gs, ref6, refs):
    """Per-point metrics of a gradient (theta-block g6 [6], state block gs [2n^2]) vs the reference, in log-theta scale."""
    theta, g6, ref6 = (np.asarray(x, np.float64) for x in (theta, g6, ref6))
    lg, lr = theta * g6, theta * ref6
    nr = np.linalg.norm(lr)
    nerr = np.abs(lg - lr) / max(nr, 1e-300)
    valid = np.abs(lr) >= TINY_REF * nr
    rel = np.where(valid, np.abs(lg - lr) / np.maximum(np.abs(lr), 1e-300), np.nan)
    sign = np.where(valid, (np.sign(lg) == np.sign(lr)).astype(float), np.nan)
    cos6 = float(lg @ lr / max(np.linalg.norm(lg) * nr, 1e-300))
    gs, refs = np.asarray(gs, np.float64), np.asarray(refs, np.float64)
    ns = np.linalg.norm(refs)
    return dict(nerr=nerr.tolist(), rel=rel.tolist(), sign=sign.tolist(), cos6=cos6,
                state_cos=float(gs @ refs / max(np.linalg.norm(gs) * ns, 1e-300)),
                state_rel=float(np.linalg.norm(gs - refs) / max(ns, 1e-300)))


def agg_rows(rows):
    """Aggregate the finished rows of one setting: per-param median/max normalised error etc. (None if no good row)."""
    good = [r for r in rows if not r.get("failed")]
    if not good:
        return None
    ne = np.array([r["nerr"] for r in good], float)
    rel = np.array([r["rel"] for r in good], float)
    sg = np.array([r["sign"] for r in good], float)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = dict(n_good=len(good), n_rows=len(rows), nerr_med=np.median(ne, 0).tolist(), nerr_max=ne.max(0).tolist(),
                 rel_med=np.nanmedian(rel, 0).tolist(), rel_n=np.sum(np.isfinite(rel), 0).tolist(),
                 sign_pct=(100 * np.nanmean(sg, 0)).tolist(), cos6_min=min(r["cos6"] for r in good),
                 state_cos_min=min(r["state_cos"] for r in good), state_rel_max=max(r["state_rel"] for r in good),
                 U_rmse_max=max(r["U_rmse"] for r in good), V_rmse_max=max(r["V_rmse"] for r in good))
        d["by_point"] = {p: dict(cos6_min=min(r["cos6"] for r in good if r["point"] == p),
                                 nerr_max=np.max([r["nerr"] for r in good if r["point"] == p], 0).tolist())
                         for p in POINTS if any(r["point"] == p for r in good)}
    return d


def last_finite(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x[-1]) if len(x) else float("nan")


def fnum(x, f="{:.3g}"):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f.format(x)


# ----------------------------------------------------------------------------------------------------------------
# context + forwards (jax imported here)
# ----------------------------------------------------------------------------------------------------------------
def resolve_eval(path):
    if os.path.isdir(path):
        return path
    if os.path.isdir(FALLBACK_EVAL):
        print(f"eval dir {path} not found; using the fallback {FALLBACK_EVAL}", flush=True)
        return FALLBACK_EVAL
    raise FileNotFoundError(f"neither {path} nor {FALLBACK_EVAL} exists")


def load_ctx(a):
    """Cases (first max(--cases, --recover-cases) headline cases), grid, true theta, loss, obs taus (<= --taus-max)."""
    from . import final_figs
    from .physics import FixedAPParams, FullField, obs_loss, observation_taus
    ev = resolve_eval(a.eval)
    M, where, ra = final_figs.load_eval(ev, headline=True)
    names = list(M)[:max(a.cases, a.recover_cases)]
    dpath = os.path.join(ra["data_root"], "config.json")
    with open(dpath) as f:
        dcfg = json.load(f)
    n, grid = int(dcfg["n"]), tuple(float(g) for g in dcfg["grid"])
    dx = (grid[1] - grid[0]) / (n - 1)
    fp = FixedAPParams.from_config(dpath)
    true = {k: float(getattr(fp, k)) for k in PARAM_NAMES}
    taus = np.asarray(observation_taus(ra["tau1"], ra["spacing"], ra["M"]), np.float32)
    m = len(taus) if not a.taus_max else int(np.sum(taus <= a.taus_max + 1e-6))
    if m < 1:
        raise ValueError(f"--taus-max {a.taus_max} leaves no obs tau")
    cases = []
    for c in names:
        z = np.load(os.path.join(where[c], f"estimates_{c}.npz"))
        if hashlib.sha1(z["y"].tobytes()).hexdigest() != M[c]["y_sha1"]:
            raise RuntimeError(f"y hash mismatch for {c}")
        adam = (np.asarray(z["adam_final_u0"], np.float32), np.asarray(z["adam_final_v0"], np.float32)) \
            if "adam_final_u0" in z.files else None
        cases.append(SimpleNamespace(name=c, y=np.asarray(z["y"][:m], np.float32), u=np.asarray(z["u_true"], np.float32),
                                     v=np.asarray(z["v_true"], np.float32), adam=adam))
    return SimpleNamespace(n=n, grid=grid, dx=dx, taus=taus[:m], loss_fn=obs_loss(ra["loss"], ra["noise_std"], FullField(), n),
                           true=true, fp=fp, ra=ra, eval=ev, cases=cases, dmax=d_max_f02(dx))


def make_fwd_p(name, ctx):
    """fn(params, u0, v0) -> (U, V, info) with theta traced: diffrax names call make_forward inside the traced fn."""
    from . import fastsim
    from .physics import make_forward
    from .solver_sweep import settings_table
    table = settings_table()
    if name in table:
        cfg = dataclasses.replace(table[name], checkpoints=1024)

        def fn(params, u0, v0):
            return make_forward(ctx.n, ctx.grid, ctx.taus, cfg, params)(u0, v0)
        return fn
    return fastsim.make_forward_fast_p(ctx.n, ctx.grid, ctx.taus, fastsim.parse_setting(name))


def fixed_params(vals):
    from .physics import FixedAPParams
    return FixedAPParams.from_values(vals)


def make_grad_fns(fwd_p, loss_fn):
    """(joint, theta_only): fn(params, u, v, y) -> ((L, (U, V, ok)), grads) for argnums (0, 1, 2) resp. 0."""
    import jax
    import jax.numpy as jnp

    def f(params, u, v, y):
        U, V, info = fwd_p(params, u, v)
        return loss_fn(U, y), (U, V, jnp.asarray(info.get("ok", True)))

    return (lambda p, u, v, y: jax.value_and_grad(f, argnums=(0, 1, 2), has_aux=True)(p, u, v, y),
            lambda p, u, v, y: jax.value_and_grad(f, argnums=0, has_aux=True)(p, u, v, y))


def theta_vec(p):
    return np.array([float(getattr(p, k)) for k in PARAM_NAMES], np.float64)


def points_of(ctx, case):
    """name -> (u, v, theta dict). joint = stored Adam state at the seed-0 perturbed theta."""
    pert = perturbed_theta(ctx.true, 0)
    pts = {"truth": (case.u, case.v, ctx.true), "theta_pert": (case.u, case.v, pert)}
    if case.adam is not None:
        pts["joint"] = (case.adam[0], case.adam[1], pert)
    return pts


def eval_point(cj, ctx, case, pname):
    import jax
    import jax.numpy as jnp
    u, v, th = points_of(ctx, case)[pname]
    fp = fixed_params(th)
    (L, (U, V, ok)), (gp, gu, gv) = jax.block_until_ready(cj(fp, jnp.asarray(u), jnp.asarray(v), jnp.asarray(case.y)))
    U, V = np.asarray(U), np.asarray(V)
    g6 = theta_vec(gp)
    gs = np.concatenate([np.asarray(gu).ravel(), np.asarray(gv).ravel()])
    good = bool(ok) and np.isfinite(U).all() and np.isfinite(V).all() and np.isfinite(g6).all() and np.isfinite(gs).all()
    return dict(L=float(L), U=U, V=V, g6=g6, gs=gs, theta=theta_vec(fp), good=good, ok=bool(ok))


# ----------------------------------------------------------------------------------------------------------------
# B1 grad
# ----------------------------------------------------------------------------------------------------------------
def grad_meta(ctx, a):
    return json.dumps({"eval": os.path.abspath(ctx.eval), "cases": [c.name for c in ctx.cases[:a.cases]],
                       "points": {c.name: sorted(points_of(ctx, c)) for c in ctx.cases[:a.cases]},
                       "taus": [float(t) for t in ctx.taus], "ref": a.ref,
                       "pert0": perturbed_theta(ctx.true, 0)}, sort_keys=True)


def load_ref(path, meta):
    if not os.path.exists(path):
        return {}
    z = np.load(path, allow_pickle=False)
    if str(z["meta"]) != meta:
        print("ref_grads.npz does not match this run (cases/points/taus/ref changed): ignoring it", flush=True)
        return {}
    ref = {}
    for k in z.files:
        if k.endswith("|g6"):
            c, p = k[:-3].split("|")
            ref[(c, p)] = tuple(z[f"{c}|{p}|{s}"] for s in ("U", "V", "g6", "gs"))
    return ref


def save_ref(path, meta, ref):
    d = {"meta": np.array(meta)}
    for (c, p), (U, V, g6, gs) in ref.items():
        d[f"{c}|{p}|U"], d[f"{c}|{p}|V"], d[f"{c}|{p}|g6"], d[f"{c}|{p}|gs"] = U, V, g6, gs
    tmp = path + ".tmp.npz"
    np.savez(tmp, **d)
    os.replace(tmp, path)


def ensure_ref(ctx, a, out):
    """Reference gradients (a.ref) at every (case, point); computed once, cached, exempt from the per-setting cap."""
    import jax
    import jax.numpy as jnp
    path, meta = os.path.join(out, "ref_grads.npz"), grad_meta(ctx, a)
    ref = load_ref(path, meta)
    cases = ctx.cases[:a.cases]
    todo = [(c, p) for c in cases for p in points_of(ctx, c) if (c.name, p) not in ref]
    if not todo:
        print(f"[grad] reference {a.ref}: all {len(ref)} points cached", flush=True)
        return ref
    cj, _ = make_grad_fns(make_fwd_p(a.ref, ctx), ctx.loss_fn)
    c0 = cases[0]
    u, v, th = next(iter(points_of(ctx, c0).values()))
    t0 = time.time()
    cj, cs, _ = bf.compile_fn(cj, (fixed_params(th), jnp.asarray(u), jnp.asarray(v), jnp.asarray(c0.y)))
    print(f"[grad] reference {a.ref}: compile {cs:.1f}s, {len(todo)} points to run", flush=True)
    for i, (c, p) in enumerate(todo):
        t1 = time.time()
        r = eval_point(cj, ctx, c, p)
        if not r["good"]:
            raise RuntimeError(f"reference {a.ref} failed (non-finite / not ok) at {c.name}/{p}")
        ref[(c.name, p)] = (r["U"].astype(np.float32), r["V"].astype(np.float32), r["g6"], r["gs"].astype(np.float32))
        save_ref(path, meta, ref)
        print(f"[grad] ref {a.ref} {c.name}/{p} ({i + 1}/{len(todo)}): {time.time() - t1:.1f}s, elapsed {time.time() - t0:.0f}s",
              flush=True)
    jax.clear_caches()
    return ref


def run_grad_setting(name, ctx, a, ref):
    """One setting at every (case, point); returns the record (status ok | capped | failed, rows, timing)."""
    import jax
    import jax.numpy as jnp
    cases = ctx.cases[:a.cases]
    t_start = time.time()
    rec = dict(status="ok", error=None, compile_s=None, ms_theta=None, ms_joint=None, reps=0, rows=[])
    try:
        cj, ct = make_grad_fns(make_fwd_p(name, ctx), ctx.loss_fn)
        c0 = cases[0]
        u, v, th = next(iter(points_of(ctx, c0).values()))
        args = (fixed_params(th), jnp.asarray(u), jnp.asarray(v), jnp.asarray(c0.y))
        cj, cs1, _ = bf.compile_fn(cj, args)
        ct, cs2, _ = bf.compile_fn(ct, args)
        rec["compile_s"] = cs1 + cs2
    except Exception as e:  # noqa: BLE001 - failures are data
        rec.update(status="failed", error=f"{type(e).__name__}: {str(e)[:300]}")
        return rec
    left = lambda: a.setting_cap_s - (time.time() - t_start)  # noqa: E731
    for ci, case in enumerate(cases):
        for pname in points_of(ctx, case):
            if left() <= 0:
                rec["status"] = "capped"
                break
            row = dict(setting=name, case=case.name, point=pname, failed=False, error=None)
            try:
                r = eval_point(cj, ctx, case, pname)
                if not r["good"]:
                    row.update(failed=True, error="solver not ok or non-finite output/gradient")
                else:
                    rU, rV, rg, rgs = ref[(case.name, pname)]
                    row.update(grad_metrics(r["theta"], r["g6"], r["gs"], rg, rgs), loss=r["L"],
                               U_rmse=float(np.sqrt(np.mean((r["U"] - rU) ** 2, axis=(1, 2))).max()),
                               V_rmse=float(np.sqrt(np.mean((r["V"] - rV) ** 2, axis=(1, 2))).max()))
            except Exception as e:  # noqa: BLE001
                row.update(failed=True, error=f"{type(e).__name__}: {str(e)[:200]}")
            rec["rows"].append(row)
        rr = [r for r in rec["rows"] if r["case"] == case.name]
        good = [r for r in rr if not r["failed"]]
        if good:
            print(f"[grad] {name} case {ci + 1}/{len(cases)} {case.name}: nerr max {max(max(r['nerr']) for r in good):.2e} "
                  f"cos6 min {min(r['cos6'] for r in good):.6f} state cos min {min(r['state_cos'] for r in good):.6f} "
                  f"U/V rmse max {max(r['U_rmse'] for r in good):.1e}/{max(r['V_rmse'] for r in good):.1e} "
                  f"({len(good)}/{len(rr)} ok)", flush=True)
        else:
            print(f"[grad] {name} case {ci + 1}/{len(cases)} {case.name}: all {len(rr)} points failed "
                  f"({rr[0]['error'] if rr else rec['status']})", flush=True)
        if rec["status"] == "capped":
            break
    if rec["status"] == "ok" and left() > 0 and any(not r["failed"] for r in rec["rows"]):
        try:     # timing at the first (case, point) of the joint kind: theta-only and joint VJP, median of 3 after warm-up
            case = cases[0]
            pn = "joint" if "joint" in points_of(ctx, case) else next(iter(points_of(ctx, case)))
            u, v, th = points_of(ctx, case)[pn]
            targs = (fixed_params(th), jnp.asarray(u), jnp.asarray(v), jnp.asarray(case.y))
            jax.block_until_ready(ct(*targs))
            jax.block_until_ready(cj(*targs))
            rec["ms_theta"], k1 = bf.med_time_ms(lambda: ct(*targs), 3, cap_s=max(left(), 0.0))
            rec["ms_joint"], k2 = bf.med_time_ms(lambda: cj(*targs), 3, cap_s=max(left(), 0.0))
            rec["reps"] = min(k1, k2)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"timing: {type(e).__name__}: {str(e)[:200]}"
    elif rec["status"] == "ok" and left() <= 0:
        rec["status"] = "capped"
    if not any(not r["failed"] for r in rec["rows"]) and rec["status"] == "ok":
        rec["status"] = "failed"
    rec["wall_s"] = time.time() - t_start
    return rec


def load_json(path):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except ValueError:
            pass
    return None


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def run_grad(ctx, a, wb, out):
    settings = [s for s in a.settings.split(",") if s]
    meta = grad_meta(ctx, a)
    path = os.path.join(out, "grad_results.json")
    res = load_json(path)
    if res is None or res.get("meta") != meta or a.redo:
        res = {"meta": meta, "dmax_f02": ctx.dmax, "settings": {}}
    ref = ensure_ref(ctx, a, out)
    wb.define_metric("grad/*", "grad/setting_idx")
    t0 = time.time()
    for i, s in enumerate(settings):
        if s in res["settings"] and not a.redo:
            print(f"[grad] {s}: cached ({res['settings'][s]['status']})", flush=True)
            continue
        rec = run_grad_setting(s, ctx, a, ref)
        res["settings"][s] = rec
        save_json(path, res)
        ag = agg_rows(rec["rows"])
        print(f"[grad] {s} ({i + 1}/{len(settings)}): {rec['status']}"
              + (f" err={rec['error']}" if rec["error"] else "")
              + (f", ms theta-VJP {fnum(rec['ms_theta'], '{:.1f}')} joint-VJP {fnum(rec['ms_joint'], '{:.1f}')}, "
                 f"compile {fnum(rec['compile_s'], '{:.1f}')}s" if rec["status"] != "failed" else "")
              + f", elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
        d = {"grad/setting_idx": i + 1, f"grad/{s}/status_ok": float(rec["status"] == "ok")}
        if ag:
            for k, nm in enumerate(PARAM_NAMES):
                d[f"grad/{s}/nerr_med/{nm}"] = ag["nerr_med"][k]
                d[f"grad/{s}/nerr_max/{nm}"] = ag["nerr_max"][k]
            d.update({f"grad/{s}/cos6_min": ag["cos6_min"], f"grad/{s}/state_cos_min": ag["state_cos_min"]})
        for k in ("ms_theta", "ms_joint", "compile_s"):
            if rec.get(k) is not None:
                d[f"grad/{s}/{k}"] = rec[k]
        wb.log(d)
    return res


# ----------------------------------------------------------------------------------------------------------------
# B2 recover
# ----------------------------------------------------------------------------------------------------------------
def make_recover_step(method, ctx):
    """Compiled value-and-grad of the obs loss wrt the raw leaves of a BoundedLogRDParams(direct) module.
    The jitted function takes the list of raw leaves (the module is rebuilt inside, the static bounds stay in the closure)."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from ap_simulator.params import BoundedLogRDParams
    bounds = widened_bounds(ctx.true)
    fwd_p = make_fwd_p(method, ctx)
    model0 = BoundedLogRDParams(**{k: jnp.asarray(ctx.true[k], jnp.float32) for k in PARAM_NAMES}, bounds=bounds,
                                mapping_mode="direct")
    diff0, static = eqx.partition(model0, eqx.is_inexact_array)
    treedef = jax.tree_util.tree_structure(diff0)

    def f(leaves, u, v, y):
        m = eqx.combine(jax.tree_util.tree_unflatten(treedef, leaves), static)
        U, V, info = fwd_p(m, u, v)
        th = jnp.stack([getattr(m, k) for k in PARAM_NAMES])
        return ctx.loss_fn(U, y), (jnp.asarray(info.get("ok", True)), th)

    def leaves_of(theta):
        m = BoundedLogRDParams(**{k: jnp.asarray(theta[k], jnp.float32) for k in PARAM_NAMES}, bounds=bounds,
                               mapping_mode="direct")
        return jax.tree_util.tree_leaves(eqx.partition(m, eqx.is_inexact_array)[0])

    return SimpleNamespace(vg=jax.value_and_grad(f, has_aux=True), leaves_of=leaves_of, bounds=bounds, compiled=None,
                           compile_s=0.0)


def fit_one(step, ctx, case, theta0, iters, tol, lr=0.02):
    """One recovery fit (true u0, v0 known). Returns the dict stored in rec_*.npz."""
    import jax.numpy as jnp
    import optax
    u, v, y = jnp.asarray(case.u), jnp.asarray(case.v), jnp.asarray(case.y)
    leaves = step.leaves_of(theta0)
    compile_s = 0.0
    if step.compiled is None:
        step.compiled, step.compile_s, _ = bf.compile_fn(step.vg, (leaves, u, v, y))
        compile_s = step.compile_s
    cvg = step.compiled
    opt = optax.adam(optax.exponential_decay(lr, transition_steps=max(1, iters // 3), decay_rate=0.5, staircase=True))
    ost = opt.init(leaves)
    loss_h, par_h, t_h = np.full(iters + 1, np.nan), np.full((iters + 1, 6), np.nan), np.full(iters + 1, np.nan)
    failed_at, reason, d_fail, th_fail = -1, "", np.nan, np.full(6, np.nan)
    cum, n_vjp = 0.0, 0
    t_h[0] = 0.0
    for it in range(iters):
        t0 = time.time()
        (L, (ok, th)), g = cvg(leaves, u, v, y)
        L, th = float(L), np.asarray(th, np.float64)
        gfin = all(bool(np.isfinite(np.asarray(x)).all()) for x in g)
        n_vjp += 1
        if not (np.isfinite(L) and gfin and bool(ok)):
            failed_at, d_fail, th_fail = it, float(th[0]), th
            reason = ("loss non-finite " if not np.isfinite(L) else "") + ("grad non-finite " if not gfin else "") \
                + ("solver not ok" if not bool(ok) else "")
            break
        updates, ost = opt.update(g, ost)
        leaves = optax.apply_updates(leaves, updates)
        cum += time.time() - t0
        loss_h[it], par_h[it], t_h[it + 1] = L, th, cum
    wall = cum
    if failed_at < 0:
        (L, (_, th)), _ = cvg(leaves, u, v, y)      # evaluation at the final params (not counted as a VJP of the fit)
        loss_h[iters], par_h[iters] = float(L), np.asarray(th, np.float64)
    fin = par_h[~np.isnan(par_h[:, 0])][-1] if np.any(~np.isnan(par_h[:, 0])) else np.array([theta0[k] for k in PARAM_NAMES])
    true = np.array([ctx.true[k] for k in PARAM_NAMES])
    rel = np.abs(fin - true) / true
    b = np.array([step.bounds[k] for k in PARAM_NAMES])
    at_bound = (np.abs(fin - b[:, 0]) <= 1e-5 * b[:, 0]) | (np.abs(fin - b[:, 1]) <= 1e-5 * b[:, 1])
    return dict(loss_hist=loss_h, param_hist=par_h, t_hist=t_h, wall_s=wall, compile_s=compile_s, n_vjp=n_vjp,
                failed_at_it=failed_at, reason=reason, D_at_failure=d_fail, param_fail=th_fail,
                D_exceeded=float(np.isfinite(d_fail) and d_fail > ctx.dmax), at_bound=at_bound, rel_err=rel,
                not_conv=rel > tol, final=fin, start=np.array([theta0[k] for k in PARAM_NAMES]), true=true,
                dmax=ctx.dmax)


def run_recover(ctx, a, wb, out):
    methods = [m for m in a.methods.split(",") if m]
    seeds = [int(s) for s in a.seeds.split(",") if s]
    cases = ctx.cases[:a.recover_cases]
    check_starts(ctx.true, seeds, ctx.dmax)
    wb.define_metric("recover/*", "recover/fit_idx")
    total, idx, t0 = len(methods) * len(cases) * len(seeds), 0, time.time()
    for m in methods:
        step = None
        for case in cases:
            for s in seeds:
                idx += 1
                fpath = os.path.join(out, f"rec_{m}_{case.name}_s{s}.npz")
                key = json.dumps({"eval": os.path.abspath(ctx.eval), "method": m, "case": case.name, "seed": s,
                                  "iters": a.iters, "taus": [float(t) for t in ctx.taus], "tol": a.conv_tol,
                                  "start": perturbed_theta(ctx.true, s)}, sort_keys=True)
                if os.path.exists(fpath) and not a.redo:
                    try:
                        if str(np.load(fpath)["key"]) == key:
                            print(f"[recover] {idx}/{total} {m} {case.name} s{s}: cached", flush=True)
                            continue
                    except Exception:  # noqa: BLE001 - corrupt cache: recompute
                        pass
                step = step or make_recover_step(m, ctx)
                try:
                    r = fit_one(step, ctx, case, perturbed_theta(ctx.true, s), a.iters, a.conv_tol)
                except Exception as e:  # noqa: BLE001 - never crash the whole queue
                    print(f"[recover] {idx}/{total} {m} {case.name} s{s}: ERROR {type(e).__name__}: {str(e)[:200]}", flush=True)
                    continue
                tmp = fpath + ".tmp.npz"
                np.savez(tmp, key=np.array(key), reason=np.array(r.pop("reason")), **r)
                os.replace(tmp, fpath)
                r["reason"] = ""
                kind = "instability" if r["failed_at_it"] >= 0 else ("not_converged" if r["not_conv"].any() else "ok")
                print(f"[recover] {idx}/{total} {m} {case.name} s{s}: {r['n_vjp']} VJP {r['wall_s']:.1f}s "
                      f"(compile {r['compile_s']:.1f}s) L={fnum(last_finite(r['loss_hist']))} "
                      f"rel err " + " ".join(f"{k} {e:.3f}" for k, e in zip(PARAM_NAMES, r["rel_err"]))
                      + f" mean {r['rel_err'].mean():.3f} -> {kind}"
                      + (f" at it {r['failed_at_it']} (D {r['D_at_failure']:.2e} {'>' if r['D_exceeded'] else '<='} D_max "
                         f"{ctx.dmax:.2e})" if kind == "instability" else "")
                      + (" at_bound: " + ",".join(k for k, b in zip(PARAM_NAMES, r["at_bound"]) if b) if r["at_bound"].any() else "")
                      + f", elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
                d = {"recover/fit_idx": idx, f"recover/{m}/wall_s": r["wall_s"], f"recover/{m}/rel_err_mean": float(r["rel_err"].mean()),
                     f"recover/{m}/final_loss": last_finite(r["loss_hist"]),
                     f"recover/{m}/failed": float(r["failed_at_it"] >= 0), f"recover/{m}/kind": ["ok", "not_converged", "instability"].index(kind)}
                d.update({f"recover/{m}/rel_err/{k}": float(e) for k, e in zip(PARAM_NAMES, r["rel_err"])})
                wb.log(d)


# ----------------------------------------------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------------------------------------------
def load_fits(out):
    fits = []
    for f in sorted(os.listdir(out)) if os.path.isdir(out) else []:
        if f.startswith("rec_") and f.endswith(".npz") and not f.endswith(".tmp.npz"):
            z = np.load(os.path.join(out, f), allow_pickle=False)
            d = {k: z[k] for k in z.files}
            m = json.loads(str(d["key"]))
            d.update(method=m["method"], case=m["case"], seed=m["seed"])
            d["reason"] = str(d["reason"])
            fits.append(d)
    return fits


def md_table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def write_summary(res, fits, path, ctx_info):
    L = ["# fastsim parameter-gradient / parameter-recovery check", ""]
    L += [f"Eval dir: `{ctx_info.get('eval')}`; data n={ctx_info.get('n')}, dx={ctx_info.get('dx'):.5g}; "
          f"true theta: " + ", ".join(f"{k}={v:.4g}" for k, v in ctx_info["true"].items()) + ".", "",
          f"**D_max_F02 = 3.5 dx^2 / (8 * 0.02) = {ctx_info['dmax']:.4e}** (Tsit5 real-axis stability bound 3.5 for the 5-point "
          f"Laplacian, eigenvalues up to 8D/dx^2, fixed dt 0.02; true D = {ctx_info['true']['D']:.4g}, i.e. "
          f"{ctx_info['dmax'] / ctx_info['true']['D']:.2f}x truth). Perturbed starts (factor exp(+-U(log1.2, log1.4)) per "
          "param, seeds below): " + "; ".join(
              f"seed {s}: D={perturbed_theta(ctx_info['true'], s)['D']:.3e} (x{perturb_factors(s)['D']:.3f})"
              for s in ctx_info.get("seeds", [0])) + ". All starts are below D_max_F02 (asserted).", ""]
    if res and res.get("settings"):
        S = res["settings"]
        aggs = {s: agg_rows(r["rows"]) for s, r in S.items()}
        L += ["## B1: parameter-gradient accuracy vs the reference solver", "",
              f"Reference: **{ctx_info.get('ref')}**. One VJP of the obs loss wrt (u0, v0, theta) with traced theta at three points per "
              "case: truth (true state, true theta), theta_pert (true state, seed-0 perturbed theta), joint (stored Adam state, "
              "seed-0 perturbed theta). Errors are in log-theta scale (theta_k * g_k) so the six entries are comparable. "
              "nerr = |g_k - g_ref_k| / ||g_ref|| (median / max over cases x points). Relative errors and sign agreement only where "
              f"|g_ref_k| >= {TINY_REF:g} ||g_ref|| (otherwise n/a; at the truth point the reference gradient is noise driven).", "",
              "### Normalised error nerr, median / max", ""]
        L += md_table(["setting", "status"] + list(PARAM_NAMES),
                      [[s, r["status"]] + ([f"{fnum(ag['nerr_med'][k], '{:.1e}')} / {fnum(ag['nerr_max'][k], '{:.1e}')}"
                                           for k in range(6)] if ag else ["-"] * 6) for (s, r), ag in zip(S.items(), aggs.values())])
        L += ["", "### Relative error (median, n valid) and sign agreement %", ""]
        L += md_table(["setting"] + list(PARAM_NAMES),
                      [[s] + ([f"{fnum(ag['rel_med'][k], '{:.1e}')} (n={ag['rel_n'][k]}) / {fnum(ag['sign_pct'][k], '{:.0f}')}%"
                               for k in range(6)] if ag else ["-"] * 6) for s, ag in aggs.items()])
        base = S.get("F02", {}).get("ms_joint")
        L += ["", "### Cosines, state block, forward error and cost", ""]
        rows = []
        for s, r in S.items():
            ag = aggs[s]
            sp = fnum(base / r["ms_joint"], "{:.2f}x") if base and r.get("ms_joint") else "n/a"
            rows.append([s, r["status"] + (f" ({r['error'][:60]})" if r.get("error") else ""),
                         fnum(ag["cos6_min"], "{:.6f}") if ag else "-", fnum(ag["state_cos_min"], "{:.6f}") if ag else "-",
                         fnum(ag["state_rel_max"], "{:.1e}") if ag else "-",
                         f"{fnum(ag['U_rmse_max'], '{:.1e}')}/{fnum(ag['V_rmse_max'], '{:.1e}')}" if ag else "-",
                         fnum(r.get("ms_theta"), "{:.1f}"), fnum(r.get("ms_joint"), "{:.1f}"), sp,
                         fnum(r.get("compile_s"), "{:.1f}"), f"{ag['n_good']}/{ag['n_rows']}" if ag else "0"])
        L += md_table(["setting", "status", "min cos (6-vec)", "min cos (state block)", "max rel L2 (state block)",
                       "max U/V RMSE vs ref", "ms theta-VJP", "ms joint VJP", "joint-VJP speedup vs F02", "compile s", "ok points"], rows)
        L += ["", "### 1 - cosine of the 6-vector, max over cases, by point", ""]
        L += md_table(["setting"] + list(POINTS), [[s] + [fnum(1 - ag["by_point"][p]["cos6_min"], "{:.1e}") if ag and p in ag["by_point"]
                                                          else "-" for p in POINTS] for s, ag in aggs.items()])
        L.append("")
    fm = {}
    for f in fits:
        fm.setdefault(f["method"], []).append(f)
    if fm:
        order = sorted(fm, key=lambda m: (m != "F02", m))
        wall = {m: float(np.mean([f["wall_s"] for f in fm[m]])) for m in order}
        L += ["## B2: parameter recovery (true u0, v0 known; BoundedLogRDParams direct mapping, decayed Adam, same VJP count)", "",
              "Final relative error |theta - theta*| / theta* in %, mean +- std over case x seed fits (an unstable fit contributes its "
              f"last finite params). instability = non-finite loss/gradient or solver not ok (fit stopped); not_converged = a finished "
              f"fit with some param above {100 * ctx_info.get('tol', 0.1):.0f}% relative error (counted among the finished fits "
              "only); at_bound = a param pinned at its (widened) bound at the end (zero gradient there).", ""]
        L += md_table(["method", "fits"] + list(PARAM_NAMES) + ["mean |rel err| %", "instability", "not_converged", "at_bound",
                                                               "wall s (excl. compile)", "compile s", "speedup vs F02"],
                      [[m, len(fm[m])]
                       + [f"{100 * np.mean([f['rel_err'][k] for f in fm[m]]):.1f} +- {100 * np.std([f['rel_err'][k] for f in fm[m]]):.1f}"
                          for k in range(6)]
                       + [f"{100 * np.mean([f['rel_err'].mean() for f in fm[m]]):.1f}",
                          f"{sum(f['failed_at_it'] >= 0 for f in fm[m])}/{len(fm[m])}",
                          f"{sum((f['failed_at_it'] < 0) and bool(f['not_conv'].any()) for f in fm[m])}/{sum(f['failed_at_it'] < 0 for f in fm[m])}",
                          ", ".join(f"{k}:{sum(bool(f['at_bound'][i]) for f in fm[m])}" for i, k in enumerate(PARAM_NAMES)
                                    if any(f['at_bound'][i] for f in fm[m])) or "none",
                          f"{wall[m]:.1f}", f"{max(float(f['compile_s']) for f in fm[m]):.1f}",
                          fnum(wall.get("F02", np.nan) / wall[m], "{:.1f}x") if "F02" in wall else "n/a"] for m in order])
        L += ["", "### Failures and not-converged params per fit", ""]
        rows = []
        for m in order:
            for f in sorted(fm[m], key=lambda f: (f["case"], f["seed"])):
                if f["failed_at_it"] >= 0:
                    note = (f"instability at it {int(f['failed_at_it'])} ({f['reason'].strip()}), D at failure {float(f['D_at_failure']):.3e} "
                            f"{'EXCEEDS' if f['D_exceeded'] else 'below'} D_max_F02 {float(f['dmax']):.3e}")
                elif f["not_conv"].any():
                    note = "not converged: " + ", ".join(k for k, b in zip(PARAM_NAMES, f["not_conv"]) if b)
                else:
                    note = "ok"
                rows.append([m, f["case"], f["seed"], f"{100 * float(f['rel_err'].mean()):.1f}", note])
        L += md_table(["method", "case", "seed", "mean |rel err| %", "outcome"], rows)
        L += ["", "### Final loss", ""]
        fl = lambda f: last_finite(f["loss_hist"])  # noqa: E731
        base_l = {(f["case"], f["seed"]): fl(f) for f in fm.get("F02", [])}
        L += md_table(["method", "mean final loss", "median final loss", "median ratio to F02 (same case/seed)"],
                      [[m, fnum(float(np.nanmean([fl(f) for f in fm[m]]))), fnum(float(np.nanmedian([fl(f) for f in fm[m]]))),
                        fnum(float(np.nanmedian([fl(f) / base_l[(f["case"], f["seed"])] for f in fm[m] if (f["case"], f["seed"]) in base_l]
                                                or [np.nan])), "{:.3f}")] for m in order])
        L.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def plot_grad(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = {s: r for s, r in res["settings"].items() if agg_rows(r["rows"])}
    aggs = {s: agg_rows(r["rows"]) for s, r in S.items()}
    names = list(S)
    fig, ax = plt.subplots(1, 2, figsize=(17, 5.5), gridspec_kw=dict(width_ratios=[3, 2]))
    w = 0.8 / max(len(names), 1)
    for j, s in enumerate(names):
        ag = aggs[s]
        med = np.maximum(ag["nerr_med"], 1e-9)
        up = np.maximum(np.array(ag["nerr_max"]) - med, 0)
        ax[0].bar(np.arange(6) + (j - len(names) / 2 + 0.5) * w, med, w, yerr=[np.zeros(6), up], color=color_of(s), label=s,
                  capsize=1.5, error_kw=dict(lw=0.8))
    ax[0].axhline(1e-2, color="k", ls="--", lw=1)
    ax[0].set_yscale("log"), ax[0].set_xticks(range(6)), ax[0].set_xticklabels(PARAM_NAMES)
    ax[0].set_ylabel("|g_k - g_ref_k| / ||g_ref||  (log-theta scale)\nmedian over cases x points, whisker = max")
    ax[0].set_title(f"per-param gradient error vs {res.get('ref', 'reference')} (dashed: 1e-2)")
    ax[0].grid(alpha=.3, which="both"), ax[0].legend(fontsize=7, ncol=2)
    pts = [p for p in POINTS if any(p in aggs[s]["by_point"] for s in names)]
    w2 = 0.8 / max(len(pts), 1)
    for j, p in enumerate(pts):
        v = [max(1 - aggs[s]["by_point"][p]["cos6_min"], 1e-12) if p in aggs[s]["by_point"] else np.nan for s in names]
        ax[1].bar(np.arange(len(names)) + (j - len(pts) / 2 + 0.5) * w2, v, w2, label=p, color=["#4c72b0", "#dd8452", "#55a868"][j % 3])
    ax[1].set_yscale("log"), ax[1].set_xticks(range(len(names))), ax[1].set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax[1].set_ylabel("1 - cosine of the 6-vector (log-theta), max over cases")
    ax[1].set_title("direction error by point"), ax[1].grid(alpha=.3, which="both"), ax[1].legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"param_grad_error.{ext}"), dpi=110)
    plt.close(fig)


def plot_recovery(fits, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = sorted({f["method"] for f in fits}, key=lambda m: (m != "F02", m))
    fig, axs = plt.subplots(2, 4, figsize=(21, 9))
    ax = axs.ravel()
    for f in fits:
        c, ls = color_of(f["method"]), LINESTYLES.get(f["method"], "-")
        true = f["true"]
        it = np.arange(f["param_hist"].shape[0])
        for k in range(6):
            ax[k].plot(it, f["param_hist"][:, k] / true[k], color=c, ls=ls, lw=0.8, alpha=0.7)
            if f["failed_at_it"] >= 0:
                ax[k].plot([f["failed_at_it"]], [f["param_fail"][k] / true[k]], "x", color=c, ms=8, mew=2)
        ax[6].plot(it, f["loss_hist"], color=c, ls=ls, lw=0.8, alpha=0.7)
        ax[7].plot(f["t_hist"], f["loss_hist"], color=c, ls=ls, lw=0.8, alpha=0.7)
    for k, nm in enumerate(PARAM_NAMES):
        ax[k].axhline(1.0, color="k", lw=1)
        ax[k].set_title(f"{nm} / truth"), ax[k].set_xlabel("Adam iteration"), ax[k].grid(alpha=.3)
    dm = fits[0]["dmax"] / fits[0]["true"][0]
    ax[0].axhline(dm, color="k", ls=":", lw=1.2)
    ax[0].text(0.02, 0.97, f"dotted: D_max_F02 / truth = {dm:.2f}", transform=ax[0].transAxes, va="top", fontsize=8)
    ax[6].set_yscale("log"), ax[6].set_title("obs loss vs iteration"), ax[6].set_xlabel("Adam iteration")
    ax[7].set_yscale("log"), ax[7].set_title("obs loss vs fit wall time (excl. compile)"), ax[7].set_xlabel("seconds")
    ax[6].grid(alpha=.3), ax[7].grid(alpha=.3)
    from matplotlib.lines import Line2D
    ax[3].legend(handles=[Line2D([0], [0], color=color_of(m), ls=LINESTYLES.get(m, "-"), label=m) for m in methods]
                 + [Line2D([0], [0], color="k", marker="x", ls="", label="instability (failed_at_it)")], fontsize=8)
    fig.suptitle("parameter recovery: thin lines = case x seed fits; black line = truth")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"param_recovery.{ext}"), dpi=110)
    plt.close(fig)


def run_report(ctx, a, wb, out):
    res = load_json(os.path.join(out, "grad_results.json"))
    fits = load_fits(out)
    info = dict(eval=ctx.eval, n=ctx.n, dx=ctx.dx, true=ctx.true, dmax=ctx.dmax, ref=a.ref,
                seeds=[int(s) for s in a.seeds.split(",") if s], tol=a.conv_tol)
    if res:
        res["ref"] = a.ref
    spath = os.path.join(out, "summary.md")
    write_summary(res, fits, spath, info)
    made = [spath]
    if res and any(agg_rows(r["rows"]) for r in res["settings"].values()):
        plot_grad(res, out)
        made += [os.path.join(out, f"param_grad_error.{e}") for e in ("png", "pdf")]
    if fits:
        plot_recovery(fits, out)
        made += [os.path.join(out, f"param_recovery.{e}") for e in ("png", "pdf")]
    os.makedirs(a.media_dir, exist_ok=True)
    for p in made:
        b = os.path.basename(p)
        shutil.copy(p, os.path.join(a.media_dir, "param_summary.md" if b == "summary.md" else b))
    if getattr(wb, "wandb", None) is not None:
        d = {f"fig/{os.path.basename(p)[:-4]}": wb.wandb.Image(p) for p in made if p.endswith(".png")}
        wb.log(d)
        if res:
            wb.table("table/grad", ["setting", "status", "ms_theta", "ms_joint", "compile_s", "cos6_min", "state_cos_min"],
                     [[s, r["status"], r.get("ms_theta"), r.get("ms_joint"), r.get("compile_s"),
                       (agg_rows(r["rows"]) or {}).get("cos6_min"), (agg_rows(r["rows"]) or {}).get("state_cos_min")]
                      for s, r in res["settings"].items()])
    print("report written:", ", ".join(made), flush=True)
    print("media copied to", a.media_dir, flush=True)


# ----------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["grad", "recover", "report", "all"])
    p.add_argument("--eval", default="cardiac/runs/eval_test_n20")
    p.add_argument("--cases", type=int, default=4, help="B1: first N headline cases")
    p.add_argument("--recover-cases", type=int, default=2, help="B2: first N headline cases")
    p.add_argument("--settings", default=DEFAULT_SETTINGS)
    p.add_argument("--ref", default="REF")
    p.add_argument("--methods", default="F02,lawson4_d0.2,lawson4_d0.1")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--conv-tol", type=float, default=0.1)
    p.add_argument("--setting-cap-s", type=float, default=240.0)
    p.add_argument("--taus-max", type=float, default=None, help="use only obs taus <= this (CPU smoke)")
    p.add_argument("--out", default="cardiac/runs/fastsim_param_check")
    p.add_argument("--media-dir", default=".claude/plans/fastsim-fit-compare-media")
    p.add_argument("--redo", action="store_true")
    p.add_argument("--force", action="store_true", help="run on the GPU even while the guarded PID exists")
    p.add_argument("--train-pid", type=int, default=bf.GUARD_PID)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-name", default="fastsim_param_check")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if bf.gpu_guard(os.environ, a.train_pid, a.force) == "refuse":      # before any jax import
        print(f"refusing to start: the training job (pid {a.train_pid}) is running and JAX_PLATFORMS is not cpu. Use "
              "JAX_PLATFORMS=cpu or --force.", file=sys.stderr, flush=True)
        return 2
    import jax
    from . import progress
    os.makedirs(a.out, exist_ok=True)
    ctx = load_ctx(a)
    print(f"[param_check] device {jax.devices()[0]}, eval {ctx.eval}, n={ctx.n}, cases {[c.name for c in ctx.cases]}, "
          f"taus {ctx.taus.tolist()}, D_max_F02 {ctx.dmax:.4e}", flush=True)
    seeds = [int(s) for s in a.seeds.split(",") if s]
    check_starts(ctx.true, seeds, ctx.dmax)
    save_json(os.path.join(a.out, "meta.json"), dict(args=vars(a), eval=ctx.eval, n=ctx.n, dx=ctx.dx, dmax_f02=ctx.dmax,
                                                     true=ctx.true, cases=[c.name for c in ctx.cases], taus=ctx.taus.tolist(),
                                                     device=str(jax.devices()[0]),
                                                     starts={str(s): perturbed_theta(ctx.true, s) for s in seeds}))
    wb = progress.NullWB() if a.no_wandb else progress.make_wb(a.wandb_name, config=vars(a))
    try:
        if a.cmd in ("grad", "all"):
            run_grad(ctx, a, wb, a.out)
        if a.cmd in ("recover", "all"):
            run_recover(ctx, a, wb, a.out)
        if a.cmd in ("report", "all"):
            run_report(ctx, a, wb, a.out)
    finally:
        wb.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
