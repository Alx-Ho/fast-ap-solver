"""Benchmark of the fixed-step JAX solvers of cardiac.fastsim against the production diffrax solvers.

  # accuracy (CPU is enough): error vs REF on the points of an existing eval dir, CPU/GPU timing at one point
  JAX_PLATFORMS=cpu python -m cardiac.bench_fastsim --phase accuracy --out cardiac/runs/fastsim_cpu --reps 3
  # timing (GPU step; one subprocess per config, optional wall-clock budget)
  python -m cardiac.bench_fastsim --phase timing --from-accuracy cardiac/runs/fastsim_cpu --out cardiac/runs/fastsim_gpu

Scoring points are truth / z_init / warm_seed0 / adam_final of the eval dir (NO tweedie points, no UNet, no A01 forward), so
ACCEPTED here is not identical to cardiac.solver_sweep's. The thresholds are the same (solver_sweep.THRESH).
The GPU guard refuses to run on a non-CPU-only JAX_PLATFORMS while the training job's PID exists (path check only; this
module never signals any process).
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

GUARD_PID = 1657544
DEFAULT_EVAL = "/home/alex/physics-guidance/cardiac/runs/eval_val4"
DEFAULT_POINTS = "truth,z_init,warm_seed0,adam_final"
DEFAULT_ACCURACY_SETTINGS = (
    ["REF", "F02", "euler_d0.01", "euler_d0.02", "heun_d0.01", "heun_d0.02", "rk4_d0.02", "rk4_d0.025", "tsit5_d0.02"]
    + [f"strang2_d{d}" for d in (0.02, 0.05, 0.1, 0.2)] + [f"strang4_d{d}" for d in (0.05, 0.1, 0.2)]
    + [f"imex_euler_d{d}" for d in (0.02, 0.05)] + [f"etd2_d{d}" for d in (0.02, 0.05, 0.1, 0.2)]
    + [f"etdrk4_d{d}" for d in (0.05, 0.1, 0.2)] + [f"lawson4_d{d}" for d in (0.05, 0.1, 0.2)])
POINT_NOTE = ("Point set: truth / z_init / warm_seed0 / adam_final of the eval dir (no tweedie points, no UNet, no A01 "
              "forward), so ACCEPTED here is not identical to the acceptance of cardiac.solver_sweep.")


# ----------------------------------------------------------------------------------------------------------------
# GPU guard (pure; runs in main() before any jax call)
# ----------------------------------------------------------------------------------------------------------------
def gpu_guard(env, pid, force=False):
    """'refuse' if JAX is not restricted to CPU and /proc/<pid> exists (the training job holds the GPU), else 'allow'.
    Only checks that the path exists; never signals the process and never touches jax."""
    if force:
        return "allow"
    plats = (env.get("JAX_PLATFORMS") or env.get("JAX_PLATFORM_NAME") or "").lower()
    names = [p.strip() for p in plats.split(",") if p.strip()]
    if names and all(p == "cpu" for p in names):
        return "allow"
    return "refuse" if os.path.exists(f"/proc/{int(pid)}") else "allow"


# ----------------------------------------------------------------------------------------------------------------
# lightweight eval loader (no UNet, no forward)
# ----------------------------------------------------------------------------------------------------------------
def resolve_stats(explicit, ea, eval_dir):
    """--stats, else the eval config's args.stats resolved against the eval dir's repo root (eval_dir/../../..), then the
    cwd, then this repo's cardiac/artifacts/stats.json."""
    if explicit:
        return explicit
    s = ea.get("stats") or "cardiac/artifacts/stats.json"
    root = os.path.abspath(os.path.join(eval_dir, "..", "..", ".."))
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cands = [s] if os.path.isabs(s) else [os.path.join(root, s), os.path.join(os.getcwd(), s)]
    cands.append(os.path.join(here, "cardiac", "artifacts", "stats.json"))
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"stats not found; tried {cands}")


def load_eval(a):
    from .physics import FixedAPParams, FullField, obs_loss, observation_taus
    with open(os.path.join(a.eval, "config.json")) as f:
        ea = json.load(f)["args"]
    with open(os.path.join(a.eval, "metrics.json")) as f:
        metrics = json.load(f)
    with open(resolve_stats(a.stats, ea, a.eval)) as f:
        stats = json.load(f)
    dpath = os.path.join(ea["data_root"], "config.json")
    with open(dpath) as f:
        dcfg = json.load(f)
    n, grid = int(dcfg["n"]), tuple(float(g) for g in dcfg["grid"])
    taus = observation_taus(ea["tau1"], ea["spacing"], ea["M"])
    sigma = ea["noise_std"] if ea["noise_std"] > 0 else 1.0
    want = [p for p in a.points.split(",") if p]
    cases = []
    for i, (name, meta) in enumerate(metrics.items()):
        if a.cases and i >= a.cases:
            break
        npz = np.load(os.path.join(a.eval, f"estimates_{name}.npz"))
        y = npz["y"]
        if hashlib.sha1(y.tobytes()).hexdigest() != meta["y_sha1"]:
            raise RuntimeError(f"y hash mismatch for {name}")
        pts = {}
        if "truth" in want:
            pts["truth"] = (npz["u_true"], npz["v_true"])
        if "z_init" in want:
            u = np.clip(y[0], 0.0, 1.0)
            pts["z_init"] = (u, np.full_like(u, stats["v"]["mean"]))
        for pn, key in (("warm_seed0", "guided_warm_seed0"), ("adam_final", "adam_final")):
            if pn in want and f"{key}_u0" in npz.files:
                pts[pn] = (npz[f"{key}_u0"], npz[f"{key}_v0"])
        pts = {k: (np.asarray(u, np.float32), np.asarray(v, np.float32)) for k, (u, v) in pts.items()}
        cases.append(SimpleNamespace(name=name, y=np.asarray(y, np.float32), points=pts))
    return SimpleNamespace(n=n, grid=grid, taus=taus, sigma=sigma, ea=ea, fp=FixedAPParams.from_config(dpath),
                           loss_fn=obs_loss(ea["loss"], sigma, FullField(), n), cases=cases)


# ----------------------------------------------------------------------------------------------------------------
# settings
# ----------------------------------------------------------------------------------------------------------------
def make_setting(name, ctx):
    """-> (fwd fn(u0, v0) -> (U, V, info), record dict) for REF/F02-style diffrax names or fastsim names."""
    from . import fastsim
    from .physics import make_forward
    from .solver_sweep import settings_table
    table = settings_table()
    if name in table:
        cfg = dataclasses.replace(table[name], checkpoints=1024)
        return make_forward(ctx.n, ctx.grid, ctx.taus, cfg, ctx.fp), {"kind": "diffrax", "family": "diffrax",
                                                                       "cfg": cfg.to_dict()}
    cfg = fastsim.parse_setting(name)
    seg, ks = fastsim.plan_segments(ctx.taus, cfg.dt)
    fam = fastsim.setting_name(dataclasses.replace(cfg, dt=1.0, diff_op="auto", react_sub=1, remat="none",
                                                   unroll=1)).split("_d")[0]
    return fastsim.make_forward_fast(ctx.n, ctx.grid, ctx.taus, cfg, ctx.fp), {
        "kind": "fast", "family": fam, "scheme": cfg.scheme, "dt": cfg.dt, "steps": int(ks.sum()),
        "rhs_per_step": fastsim.rhs_per_step(cfg), "cfg": cfg.to_dict()}


def compile_fn(f, args):
    """Lower + compile once; returns (callable, compile_s, temp_MB) from that same object."""
    import jax
    t0 = time.time()
    c = jax.jit(f).lower(*args).compile()
    dt = time.time() - t0
    try:
        ma = c.memory_analysis()
        tmp = float(ma.temp_size_in_bytes) / 2 ** 20 if ma is not None else None
    except Exception:  # noqa: BLE001 - not available on every backend
        tmp = None
    return c, dt, tmp


def med_time_ms(fn, reps, cap_s=None):
    """median wall ms of `reps` calls (after the caller's warm-up); stops early once cap_s is exceeded."""
    import jax
    ts, t_start = [], time.time()
    for _ in range(reps):
        t0 = time.time()
        jax.block_until_ready(fn())
        ts.append(time.time() - t0)
        if cap_s is not None and time.time() - t_start > cap_s:
            break
    return float(np.median(ts)) * 1e3, len(ts)


def make_vg(fwd, loss_fn):
    import jax
    import jax.numpy as jnp

    def vg(u, v, y):
        def f(u, v):
            U, V, info = fwd(u, v)
            return loss_fn(U, y), (U, V, jnp.asarray(info.get("ok", True)), jnp.asarray(info.get("num_steps", 0)))
        (L, aux), g = jax.value_and_grad(f, argnums=(0, 1), has_aux=True)(u, v)
        return L, aux, g
    return vg


# ----------------------------------------------------------------------------------------------------------------
# accuracy phase
# ----------------------------------------------------------------------------------------------------------------
def _flat(gu, gv):
    return np.concatenate([np.asarray(gu, np.float64).ravel(), np.asarray(gv, np.float64).ravel()])


def _ref_meta(ctx, a):
    return json.dumps({"eval": os.path.abspath(a.eval), "cases": [c.name for c in ctx.cases],
                       "points": {c.name: sorted(c.points) for c in ctx.cases}, "taus": [float(t) for t in ctx.taus]})


def load_ref_cache(path, meta):
    if not os.path.exists(path):
        return {}
    z = np.load(path, allow_pickle=False)
    if str(z["meta"]) != meta:
        print("ref_cache.npz does not match this run (cases/points/taus changed): ignoring it", flush=True)
        return {}
    ref = {}
    for k in z.files:
        if k.endswith("|U"):
            c, p = k[:-2].split("|")
            ref[(c, p)] = (z[k], z[f"{c}|{p}|V"], z[f"{c}|{p}|g"])
    return ref


def save_ref_cache(path, meta, ref):
    d = {"meta": np.array(meta)}
    for (c, p), (U, V, g) in ref.items():
        d[f"{c}|{p}|U"], d[f"{c}|{p}|V"], d[f"{c}|{p}|g"] = U, V, g
    tmp = path + ".tmp.npz"
    np.savez(tmp, **d)
    os.replace(tmp, path)


def run_setting(name, ctx, a, results, ref, out, on_point=None, save=None):
    """Evaluate one setting on every (case, point); appends rows to results (REF fills `ref`)."""
    import jax
    import jax.numpy as jnp
    fwd, rec = make_setting(name, ctx)
    rec.update(status="ok", error=None, compile_s=None, temp_MB=None, fwd_temp_MB=None)
    results["settings"][name] = rec
    vg = make_vg(fwd, ctx.loss_fn)
    cvg = cfwd = None
    is_ref = name == "REF"
    if is_ref:      # keep the finished rows of a partially finished REF (they are in the cache), drop the rest
        results["rows"] = [r for r in results["rows"] if r["setting"] != "REF" or (r["case"], r["point"]) in ref]
    else:
        results["rows"] = [r for r in results["rows"] if r["setting"] != name]
    have = {(r["case"], r["point"]) for r in results["rows"] if r["setting"] == name}
    try:
        c0 = ctx.cases[0]
        u0, v0 = next(iter(c0.points.values()))
        args = (jnp.asarray(u0), jnp.asarray(v0), jnp.asarray(c0.y))
        cvg, rec["compile_s"], rec["temp_MB"] = compile_fn(vg, args)
        if not is_ref:
            cfwd, cs, rec["fwd_temp_MB"] = compile_fn(lambda u, v: fwd(u, v), args[:2])
            rec["compile_s"] += cs
    except Exception as e:  # noqa: BLE001 - failures are data
        rec.update(status="failed", error=f"{type(e).__name__}: {str(e)[:300]}")
        if is_ref:
            raise
    for case in ctx.cases:
        for pname, (u, v) in case.points.items():
            if is_ref and (case.name, pname) in have:
                continue
            row = {"setting": name, "case": case.name, "point": pname, "failed": False, "error": None}
            results["rows"].append(row)
            if rec["status"] != "ok":
                row.update(failed=True, error="setting " + rec["status"])
                continue
            uj, vj, yj = jnp.asarray(u), jnp.asarray(v), jnp.asarray(case.y)
            try:
                L, (U, V, ok, ns), (gu, gv) = jax.block_until_ready(cvg(uj, vj, yj))
                U, V, g = np.asarray(U), np.asarray(V), _flat(gu, gv)
                row["num_steps"], row["ok_flag"] = int(ns), bool(ok)
                if not (bool(ok) and np.isfinite(U).all() and np.isfinite(V).all()):
                    row.update(failed=True, error="solver not ok or non-finite output")
                elif pname != "truth" and not np.isfinite(g).all():
                    row.update(failed=True, error="non-finite gradient")
                if is_ref:
                    if row["failed"]:
                        raise RuntimeError(f"REF failed at {case.name}/{pname}: {row['error']}")
                    ref[(case.name, pname)] = (U.astype(np.float32), V.astype(np.float32), g)
                rU, rV, rg = ref[(case.name, pname)]
                eU = np.sqrt(np.mean((U - rU) ** 2, axis=(1, 2)))
                eV = np.sqrt(np.mean((V - rV) ** 2, axis=(1, 2)))
                row.update(U_rmse_max=float(eU.max()), U_rmse_mean=float(eU.mean()), V_rmse_max=float(eV.max()),
                           V_rmse_mean=float(eV.mean()), U_rmse_tau=eU, V_rmse_tau=eV)
                if pname != "truth":
                    den = np.linalg.norm(g) * np.linalg.norm(rg)
                    row["grad_cos"] = float(g @ rg / den) if den > 0 else float("nan")
                    row["grad_rel"] = float(np.linalg.norm(g - rg) / max(np.linalg.norm(rg), 1e-30))
                from .solver_sweep import TIMING_POINT
                if pname == TIMING_POINT and not row["failed"] and not is_ref:
                    jax.block_until_ready(cfwd(uj, vj))                  # explicit warm-ups
                    jax.block_until_ready(cvg(uj, vj, yj))
                    row["t_fwd_ms"], _ = med_time_ms(lambda: cfwd(uj, vj), a.reps)
                    row["t_vjp_ms"], _ = med_time_ms(lambda: cvg(uj, vj, yj), a.reps)
                    row["vjp_over_fwd"] = row["t_vjp_ms"] / row["t_fwd_ms"]
                    row["temp_MB"] = rec["temp_MB"]
            except Exception as e:  # noqa: BLE001 - failures are data
                if is_ref:
                    raise
                row.update(failed=True, error=f"{type(e).__name__}: {str(e)[:200]}")
            if on_point:
                on_point(name, row)
            if is_ref and save:
                save()
    return rec


def family_of(name, rec):
    return rec.get("family") or ("diffrax" if name in ("REF", "F02") else name.split("_d")[0])


def enriched_agg(results):
    """solver_sweep.analyze + scheme/dt/steps/rhs/temp MB/VJP-over-fwd/speedup vs F02 (same run, same backend)."""
    from .solver_sweep import analyze
    agg = analyze(results)
    base = agg.get("F02", {})
    for n, a in agg.items():
        rec = results["settings"][n]
        a.update(family=family_of(n, rec), scheme=rec.get("scheme", "diffrax"), dt=rec.get("dt"), steps=rec.get("steps"),
                 rhs_per_step=rec.get("rhs_per_step"), temp_MB=rec.get("temp_MB"),
                 vjp_over_fwd=(a["t_vjp_ms"] / a["t_fwd_ms"] if a["t_vjp_ms"] and a["t_fwd_ms"] else None),
                 speedup_vjp=(base["t_vjp_ms"] / a["t_vjp_ms"] if base.get("t_vjp_ms") and a["t_vjp_ms"] else None),
                 speedup_fwd=(base["t_fwd_ms"] / a["t_fwd_ms"] if base.get("t_fwd_ms") and a["t_fwd_ms"] else None))
    return agg


def best_per_family(agg):
    """{family: setting name} = the accepted setting with the lowest median VJP time of each family (excluding REF)."""
    best = {}
    for n, a in agg.items():
        if n == "REF" or not a["accepted"] or a["t_vjp_ms"] is None:
            continue
        f = a["family"]
        if f not in best or a["t_vjp_ms"] < agg[best[f]]["t_vjp_ms"]:
            best[f] = n
    return best


def write_summary_fast(results, path, extra_note=""):
    agg = enriched_agg(results)
    meta = results["meta"]
    best = best_per_family(agg)
    reps = meta.get("reps")
    L = ["# fastsim accuracy / cost benchmark", "",
         f"Backend: **{meta.get('backend')}**; eval dir: `{meta.get('eval')}`; cases: {len(meta.get('cases', []))}; points: "
         f"{', '.join(meta.get('points', []))}; timing reps: {reps}"
         + (" (**indicative**: fewer than 3 reps)" if reps is not None and reps < 3 else "") + ".", "",
         POINT_NOTE, "",
         "ACCEPTED (thresholds as in solver_sweep, at all points and cases): max-over-tau U RMSE <= 0.002, V RMSE <= 0.005, "
         "gradient cosine >= 0.99 and rel. L2 error <= 0.1, no failures; errors vs REF (diffrax tsit5, rtol=atol=1e-6, "
         "dtmax 0.0025). Speedup = F02 time / setting time, from this run on this backend (median over cases at the "
         "warm_seed0 point).", "",
         "| setting | scheme | dt | steps | RHS/step | status | fails/rows | max U RMSE | max V RMSE | min grad cos | "
         "max grad rel | ms/fwd | ms/VJP | VJP/fwd | VJP speedup vs F02 | temp MB | compile s | accepted |",
         "|" + "---|" * 18]
    for n, a in agg.items():
        L.append(f"| {n} | {a['scheme']} | {fmt(a['dt'])} | {fmt(a['steps'] or a['num_steps_med'], '{:.0f}')} | "
                 f"{fmt(a['rhs_per_step'], '{:.0f}')} | {a['status']} | {a['n_fail']}/{a['n_rows']} | {fmt(a['U_max'])} | "
                 f"{fmt(a['V_max'])} | {fmt(a['cos_min'], '{:.5f}')} | {fmt(a['rel_max'])} | "
                 f"{fmt(a['t_fwd_ms'], '{:.1f}')} | {fmt(a['t_vjp_ms'], '{:.1f}')} | {fmt(a['vjp_over_fwd'], '{:.2f}')} | "
                 f"{fmt(a['speedup_vjp'], '{:.2f}x')} | {fmt(a['temp_MB'], '{:.0f}')} | {fmt(a['compile_s'], '{:.1f}')} | "
                 f"{'yes' if a['accepted'] else 'no'} |")
    L += ["", "## Best accepted setting per family (lowest VJP time)", ""]
    if best:
        L += ["| family | setting | dt | ms/VJP | VJP/fwd | VJP speedup vs F02 | fwd speedup vs F02 |", "|---|---|---|---|---|---|---|"]
        for f, n in sorted(best.items(), key=lambda kv: agg[kv[1]]["t_vjp_ms"]):
            a = agg[n]
            L.append(f"| {f} | {n} | {fmt(a['dt'])} | {fmt(a['t_vjp_ms'], '{:.1f}')} | {fmt(a['vjp_over_fwd'], '{:.2f}')} | "
                     f"{fmt(a['speedup_vjp'], '{:.2f}x')} | {fmt(a['speedup_fwd'], '{:.2f}x')} |")
        top = min(best.values(), key=lambda n: agg[n]["t_vjp_ms"])
        L += ["", f"Best accepted setting overall: **{top}** ({fmt(agg[top]['speedup_vjp'], '{:.2f}x')} faster per VJP than "
              f"F02 on {meta.get('backend')})."]
    else:
        L.append("No setting is ACCEPTED.")
    if extra_note:
        L += ["", extra_note]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return agg, best


def fmt(x, f="{:.4g}"):
    from .solver_sweep import fmt as _fmt
    return _fmt(x, f)


def write_plots_fast(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .solver_sweep import THRESH, write_plots
    write_plots(results, out)                       # error_vs_cost.png, gradient_agreement.png
    agg = enriched_agg(results)
    fams = {}
    for n, a in agg.items():
        if n != "REF" and a["dt"] is not None and a["U_max"] is not None:
            fams.setdefault(a["family"], []).append((a["dt"], a))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for a_, (k, title, thr) in zip(ax, [("U_max", "max U RMSE vs REF", THRESH["U"]), ("V_max", "max V RMSE vs REF", THRESH["V"]),
                                        ("rel_max", "gradient rel. L2 vs REF", THRESH["rel"])]):
        for f, pts in sorted(fams.items()):
            pts = sorted(pts, key=lambda t: t[0])
            a_.plot([p[0] for p in pts], [max(p[1][k], 1e-12) for p in pts], "o-", label=f)
        a_.axhline(thr, color="k", ls="--", lw=1)
        a_.set_xscale("log"), a_.set_yscale("log"), a_.set_xlabel("dt"), a_.set_ylabel(title), a_.grid(alpha=.3, which="both")
    ax[0].legend(fontsize=7)
    fig.suptitle("convergence: worst-case error vs dt per family (dashed = acceptance threshold)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "convergence.png"), dpi=100)
    plt.close(fig)
    rows = sorted([(n, a) for n, a in agg.items() if a["t_vjp_ms"]], key=lambda t: t[1]["t_vjp_ms"])
    if rows:
        fig, ax = plt.subplots(figsize=(11, 0.32 * len(rows) + 2))
        y = np.arange(len(rows))
        ax.barh(y + 0.2, [a["t_vjp_ms"] for _, a in rows], 0.4, label="forward+VJP",
                color=["tab:green" if a["accepted"] else "tab:red" for _, a in rows])
        ax.barh(y - 0.2, [a["t_fwd_ms"] for _, a in rows], 0.4, label="forward", color="tab:gray")
        ax.set_yticks(y)
        ax.set_yticklabels([n for n, _ in rows], fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel(f"median ms ({results['meta'].get('backend')}); green = accepted, red = rejected")
        ax.legend()
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "timing.png"), dpi=100)
        plt.close(fig)


def wandb_setting_log(wb, results, name, idx, total):
    agg = enriched_agg(results)
    a = agg[name]
    d = {"setting_idx": idx, "n_settings": total}
    for k in ("U_max", "V_max", "cos_min", "rel_max", "t_fwd_ms", "t_vjp_ms", "vjp_over_fwd", "compile_s", "temp_MB",
              "speedup_vjp", "speedup_fwd", "n_fail"):
        if a.get(k) is not None:
            d[f"setting/{k}"] = float(a[k])
    d["setting/accepted"] = float(a["accepted"])
    wb.log(d)
    cols = ["setting", "status", "dt", "U_max", "V_max", "cos_min", "rel_max", "ms_fwd", "ms_vjp", "vjp_over_fwd",
            "speedup_vjp", "temp_MB", "compile_s", "accepted"]
    rows = [[n, x["status"], x["dt"], x["U_max"], x["V_max"], x["cos_min"], x["rel_max"], x["t_fwd_ms"], x["t_vjp_ms"],
             x["vjp_over_fwd"], x["speedup_vjp"], x["temp_MB"], x["compile_s"], x["accepted"]] for n, x in agg.items()]
    wb.table("table/settings", cols, rows)


def wandb_finish(wb, out, files):
    if getattr(wb, "run", None) is None:
        wb.finish()
        return
    try:
        for f in files:
            p = os.path.join(out, f)
            if f.endswith(".png") and os.path.exists(p):
                wb._safe(lambda p=p, f=f: wb.wandb.log({f"plots/{f[:-4]}": wb.wandb.Image(p)}))
        sm = os.path.join(out, "summary.md")
        if os.path.exists(sm):
            wb._safe(lambda: wb.wandb.log({"summary_md": wb.wandb.Html("<pre>" + open(sm).read() + "</pre>")}))

        def art():
            art_ = wb.wandb.Artifact("fastsim-results", type="results")
            art_.add_file(os.path.join(out, "results.json"))
            wb.run.log_artifact(art_)
        wb._safe(art)
    finally:
        wb.finish()


def run_accuracy(a):
    import jax
    from .progress import NullWB, make_wb
    from .solver_sweep import THRESH, write_json
    os.makedirs(a.out, exist_ok=True)
    rpath = os.path.join(a.out, "results.json")
    backend = jax.default_backend()
    if a.plot_only:
        results = json.load(open(rpath))
    else:
        ctx = load_eval(a)
        if not ctx.cases or not all(c.points for c in ctx.cases):
            raise RuntimeError("no evaluation points found")
        names = [s for s in a.settings.split(",") if s] or list(DEFAULT_ACCURACY_SETTINGS)
        names = ["REF"] + [s for s in names if s != "REF"]
        results = None
        if os.path.exists(rpath) and not a.redo:
            results = json.load(open(rpath))
            if results["meta"].get("cases") != [c.name for c in ctx.cases]:
                print("results.json is for other cases: starting fresh", flush=True)
                results = None
        meta = {"eval": a.eval, "cases": [c.name for c in ctx.cases], "points": sorted({p for c in ctx.cases for p in c.points}),
                "args": vars(a), "thresholds": THRESH,
                "backend": backend, "reps": a.reps, "n": ctx.n, "taus": [float(t) for t in ctx.taus], "note": POINT_NOTE}
        if results is None:
            results = {"meta": meta, "settings": {}, "rows": []}
        results["meta"].update(meta)
        cache_path = os.path.join(a.out, "ref_cache.npz")
        ref_meta = _ref_meta(ctx, a)
        ref = {} if a.redo else load_ref_cache(cache_path, ref_meta)
        wb = NullWB() if a.no_wandb else make_wb(f"fastsim-accuracy-{backend}", config={"args": vars(a), "settings": names})
        todo = [s for s in names if s not in results["settings"] or (s == "REF" and len(ref) < sum(len(c.points) for c in ctx.cases))]
        print(f"fastsim accuracy: backend={backend}, {len(ctx.cases)} cases, points={meta['points']}, "
              f"{len(todo)}/{len(names)} settings to run, reps={a.reps}", flush=True)

        def flush():
            write_json(rpath, results)

        t_all = time.time()
        for i, nm in enumerate(names):
            if nm not in todo:
                continue
            t0 = time.time()

            def on_point(n_, row, nm=nm):
                tail = "" if row["failed"] else (f"U {row['U_rmse_max']:.3g} V {row['V_rmse_max']:.3g}"
                                                  + (f" cos {row['grad_cos']:.5f} rel {row['grad_rel']:.3g}" if "grad_cos" in row else "")
                                                  + (f" fwd {row['t_fwd_ms']:.0f}ms vjp {row['t_vjp_ms']:.0f}ms" if "t_vjp_ms" in row else ""))
                print(f"  [{nm}] {row['case']}/{row['point']} " + (f"FAILED {row['error']}" if row["failed"] else tail),
                      f"(+{time.time() - t_all:.0f}s)", flush=True)
                flush()

            rec = run_setting(nm, ctx, a, results, ref, a.out, on_point,
                              save=lambda: save_ref_cache(cache_path, ref_meta, ref))
            if nm == "REF":
                save_ref_cache(cache_path, ref_meta, ref)
            flush()
            agg = enriched_agg(results)[nm]
            print(f"[{i + 1}/{len(names)}] {nm}: {rec['status']} in {time.time() - t0:.0f}s | U {fmt(agg['U_max'])} "
                  f"V {fmt(agg['V_max'])} cos {fmt(agg['cos_min'], '{:.5f}')} rel {fmt(agg['rel_max'])} | fwd "
                  f"{fmt(agg['t_fwd_ms'], '{:.0f}')}ms vjp {fmt(agg['t_vjp_ms'], '{:.0f}')}ms (x{fmt(agg['vjp_over_fwd'], '{:.2f}')}) "
                  f"speedup {fmt(agg['speedup_vjp'], '{:.2f}')} | accepted {agg['accepted']}", flush=True)
            wandb_setting_log(wb, results, nm, i, len(names))
        flush()
    agg, best = write_summary_fast(results, os.path.join(a.out, "summary.md"))
    write_plots_fast(results, a.out)
    if not a.plot_only:
        wandb_finish(wb, a.out, ["error_vs_cost.png", "gradient_agreement.png", "convergence.png", "timing.png"])
    print("best accepted per family:", {f: n for f, n in best.items()}, flush=True)
    print("wrote", a.out, flush=True)
    return results


# ----------------------------------------------------------------------------------------------------------------
# timing phase (GPU step): one subprocess per config
# ----------------------------------------------------------------------------------------------------------------
def timing_child(a):
    """One config: compile, warm up, time fwd and VJP of a B-lane problem; print a `BENCH {json}` line."""
    import jax
    import jax.numpy as jnp
    from . import fastsim
    from .physics import make_forward, make_forward_batched
    from .solver_sweep import TIMING_POINT, settings_table
    spec = json.loads(a.internal_one)
    ctx = load_eval(a)
    B = int(spec["B"])
    name = spec["name"]
    res = {"name": name, "B": B, "remat": spec.get("remat"), "unroll": spec.get("unroll", 1), "status": "ok",
           "backend": jax.default_backend()}
    pts = [(c, (c.points.get(TIMING_POINT) or next(v for k, v in c.points.items() if k != "truth"))) for c in ctx.cases]
    lanes = [pts[b % len(pts)] for b in range(B)]
    u0 = np.stack([p[1][0] for p in lanes])
    v0 = np.stack([p[1][1] for p in lanes])
    y = np.stack([p[0].y for p in lanes])
    table = settings_table()
    if name in table:
        cfg = dataclasses.replace(table[name], checkpoints=1024)
        f1 = make_forward(ctx.n, ctx.grid, ctx.taus, cfg, ctx.fp)
        fB = make_forward_batched(ctx.n, ctx.grid, ctx.taus, cfg, ctx.fp)
    else:
        cfg = fastsim.parse_setting(name)
        if spec.get("remat"):
            cfg.remat = spec["remat"]
        cfg.unroll = int(spec.get("unroll", 1))
        res["setting"] = fastsim.setting_name(cfg)
        f1 = fastsim.make_forward_fast(ctx.n, ctx.grid, ctx.taus, cfg, ctx.fp)
        fB = fastsim.make_forward_fast_batched(ctx.n, ctx.grid, ctx.taus, cfg, ctx.fp)
    loss1 = ctx.loss_fn
    if B == 1:
        fwd, loss, args = f1, loss1, (jnp.asarray(u0[0]), jnp.asarray(v0[0]), jnp.asarray(y[0]))
    else:
        fwd = fB
        loss = lambda U, yy: jnp.sum(jax.vmap(loss1)(U, yy))
        args = (jnp.asarray(u0), jnp.asarray(v0), jnp.asarray(y))
    vg = make_vg(fwd, loss)
    cvg, res["compile_s_vjp"], res["temp_MB_vjp"] = compile_fn(vg, args)
    cfwd, res["compile_s_fwd"], res["temp_MB_fwd"] = compile_fn(lambda u, v: fwd(u, v), args[:2])
    res["compile_s"] = res["compile_s_vjp"] + res["compile_s_fwd"]
    Ub = jax.block_until_ready(cfwd(args[0], args[1]))                      # warm-ups
    res["num_steps"] = int(jax.block_until_ready(cvg(*args))[1][3])
    res["t_fwd_ms"], r1 = med_time_ms(lambda: cfwd(args[0], args[1]), a.reps, a.time_cap)
    res["t_vjp_ms"], r2 = med_time_ms(lambda: cvg(*args), a.reps, a.time_cap)
    res["reps_done"] = min(r1, r2)
    res["vjp_over_fwd"] = res["t_vjp_ms"] / res["t_fwd_ms"]
    res["ms_per_case"] = res["t_vjp_ms"] / B
    if B > 1:                                                             # lane 0: batched vs unbatched
        U1 = jax.jit(f1)(args[0][0], args[1][0])[0]
        res["lane0_maxdiff"] = float(jnp.max(jnp.abs(Ub[0][0] - U1)))
    mem = None
    try:
        mem = jax.devices()[0].memory_stats()
    except Exception:  # noqa: BLE001
        pass
    if mem and "peak_bytes_in_use" in mem:
        res["peak_MB"], res["peak_src"] = mem["peak_bytes_in_use"] / 2 ** 20, "peak_bytes_in_use"
    else:
        res["peak_MB"], res["peak_src"] = res["temp_MB_vjp"], "temp_bytes"
    print("BENCH " + json.dumps(res), flush=True)


def is_diffrax(name):
    from .solver_sweep import settings_table
    return name in settings_table()


def select_timing_settings(a):
    """(names, top2): from --settings, or from --from-accuracy."""
    if a.settings:
        names = [s for s in a.settings.split(",") if s]
        fast = [s for s in names if not is_diffrax(s)]
        return names, fast[:2]
    results = json.load(open(os.path.join(a.from_accuracy, "results.json")))
    agg = enriched_agg(results)
    best = best_per_family(agg)
    ranked = sorted(best.values(), key=lambda n: agg[n]["t_vjp_ms"])
    names = ["F02", "tsit5_d0.02"] + [n for n in ranked if n != "tsit5_d0.02"][:6]
    top2 = [n for n in ranked if n != "tsit5_d0.02"][:2] or ["tsit5_d0.02"]
    for nm, default, ops in (("strang2_d0.1", "sep", ("sep", "eig", "dct")), ("etd2_d0.1", "eig", ("eig", "dct"))):
        names += [nm if op == default else f"{nm}_{op}" for op in ops]
    seen, out = set(), []
    for n in names:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out, top2


def build_timing_configs(a, names, top2):
    remats = [r for r in a.remat.split(",") if r]
    batches = [int(b) for b in a.batches.split(",") if b]
    cfgs = []

    def add(**kw):
        cfgs.append(kw)
    is_fast = lambda n: not is_diffrax(n)
    for n in names:                                      # B=1, default remat then the other remat modes
        if not is_fast(n):
            add(name=n, B=1)
        else:
            add(name=n, B=1, remat=remats[0])
    for n in names:
        if is_fast(n):
            for r in remats[1:]:
                add(name=n, B=1, remat=r)
    for B in sorted(b for b in batches if b > 1):
        for n in (["F02"] if "F02" in names else []) + top2:
            add(name=n, B=B, remat=remats[0] if is_fast(n) else None)
    for n in top2:
        if "segment" not in remats:
            add(name=n, B=1, remat="segment")
        add(name=n, B=1, remat=remats[0], unroll=4)
    seen, out = set(), []
    for c in cfgs:
        k = json.dumps(c, sort_keys=True)
        if k not in seen:
            out.append(c)
            seen.add(k)
    return out


def cfg_label(c):
    s = c["name"]
    if c.get("remat"):
        s += f"|{c['remat']}"
    if c.get("unroll", 1) != 1:
        s += f"|unroll{c['unroll']}"
    return f"{s}|B{c['B']}"


def run_timing(a):
    from .progress import NullWB, make_wb
    from .solver_sweep import write_json
    os.makedirs(a.out, exist_ok=True)
    rpath = os.path.join(a.out, "results.json")
    if a.plot_only:
        results = json.load(open(rpath))
        wb = NullWB()
    else:
        names, top2 = select_timing_settings(a)
        cfgs = build_timing_configs(a, names, top2)
        results = {"meta": {"eval": a.eval, "args": vars(a), "reps": a.reps, "max_minutes": a.max_minutes,
                            "settings": names, "top2": top2, "note": POINT_NOTE}, "configs": []}
        wb = None
        t_start = time.time()
        print(f"fastsim timing: {len(cfgs)} configs, reps={a.reps}, max_minutes={a.max_minutes}, top2={top2}", flush=True)
        for i, c in enumerate(cfgs):
            lab = cfg_label(c)
            if a.max_minutes is not None and (time.time() - t_start) / 60 > a.max_minutes:
                results["configs"].append({**c, "label": lab, "status": "skipped_budget"})
                print(f"[{i + 1}/{len(cfgs)}] {lab}: skipped_budget", flush=True)
                continue
            cmd = [sys.executable, "-m", "cardiac.bench_fastsim", "--phase", "timing", "--internal-one", json.dumps(c),
                   "--eval", a.eval, "--cases", str(a.cases), "--reps", str(a.reps), "--points", a.points,
                   "--time-cap", str(a.time_cap), "--out", a.out] + (["--stats", a.stats] if a.stats else [])
            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True)
            line = [ln for ln in r.stdout.splitlines() if ln.startswith("BENCH ")]
            if line:
                rec = json.loads(line[0][6:])
            else:
                err = (r.stderr or "")[-600:]
                rec = {**c, "status": "oom" if ("RESOURCE_EXHAUSTED" in err or "out of memory" in err.lower()) else "failed",
                       "error": err}
            rec["label"], rec["wall_s"] = lab, time.time() - t0
            results["configs"].append(rec)
            write_json(rpath, results)
            if wb is None:
                wb = NullWB() if a.no_wandb else make_wb(f"fastsim-timing-{rec.get('backend', 'unknown')}",
                                                         config={"args": vars(a)})
            add_speedups(results)
            print(f"[{i + 1}/{len(cfgs)}] {lab}: {rec['status']} fwd {fmt(rec.get('t_fwd_ms'), '{:.1f}')}ms vjp "
                  f"{fmt(rec.get('t_vjp_ms'), '{:.1f}')}ms vjp/fwd {fmt(rec.get('vjp_over_fwd'), '{:.2f}')} peak "
                  f"{fmt(rec.get('peak_MB'), '{:.0f}')}MB speedup {fmt(rec.get('speedup_vs_F02'), '{:.2f}')} "
                  f"compile {fmt(rec.get('compile_s'), '{:.1f}')}s ({rec['wall_s']:.0f}s wall)", flush=True)
            if rec["status"] == "ok":
                wb.log({"config_idx": i, **{f"config/{k}": float(rec[k]) for k in
                        ("t_fwd_ms", "t_vjp_ms", "vjp_over_fwd", "compile_s", "peak_MB", "ms_per_case", "speedup_vs_F02")
                        if rec.get(k) is not None}})
        write_json(rpath, results)
    add_speedups(results)
    write_json(rpath, results)
    write_timing_outputs(results, a.out)
    if not a.plot_only:
        wandb_finish(wb or NullWB(), a.out, ["timing.png", "throughput.png"])
    print("wrote", a.out, flush=True)
    return results


def add_speedups(results):
    base = {c["B"]: c for c in results["configs"] if c["name"] == "F02" and c.get("status") == "ok"}
    for c in results["configs"]:
        b = base.get(c["B"])
        c["speedup_vs_F02"] = (b["t_vjp_ms"] / c["t_vjp_ms"]) if b and c.get("status") == "ok" and c.get("t_vjp_ms") else None
        c["speedup_fwd_vs_F02"] = (b["t_fwd_ms"] / c["t_fwd_ms"]) if b and c.get("status") == "ok" and c.get("t_fwd_ms") else None
    return base


def write_timing_outputs(results, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cf = results["configs"]
    be = next((c.get("backend") for c in cf if c.get("backend")), "unknown")
    reps = results["meta"].get("reps")
    L = ["# fastsim timing", "", f"Backend: **{be}**; reps: {reps}" + (" (**indicative**: fewer than 3 reps)"
                                                                         if reps is not None and reps < 3 else "")
         + f"; wall budget: {results['meta'].get('max_minutes')} min (None = no cap).", "", POINT_NOTE, "",
         "Lanes tile the cases' warm_seed0 points; VJP = forward + reverse of the summed gaussian obs loss wrt (u0, v0). "
         "Speedup = F02 VJP time / config VJP time at the same B (F02 at B>1 = physics.make_forward_batched).", "",
         "| config | status | steps | compile s | ms/fwd | ms/VJP | VJP/fwd | ms/case (VJP) | peak MB | peak src | "
         "speedup vs F02 | lane-0 diff |", "|" + "---|" * 12]
    for c in cf:
        L.append(f"| {c['label']} | {c['status']} | {fmt(c.get('num_steps'), '{:.0f}')} | {fmt(c.get('compile_s'), '{:.1f}')} | "
                 f"{fmt(c.get('t_fwd_ms'), '{:.1f}')} | {fmt(c.get('t_vjp_ms'), '{:.1f}')} | "
                 f"{fmt(c.get('vjp_over_fwd'), '{:.2f}')} | {fmt(c.get('ms_per_case'), '{:.1f}')} | "
                 f"{fmt(c.get('peak_MB'), '{:.0f}')} | {c.get('peak_src', '-')} | {fmt(c.get('speedup_vs_F02'), '{:.2f}x')} | "
                 f"{fmt(c.get('lane0_maxdiff'), '{:.2e}')} |")
    bad = [c for c in cf if c["status"] not in ("ok",)]
    if bad:
        L += ["", "Not run / failed: " + "; ".join(f"{c['label']} ({c['status']})" for c in bad)]
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    ok = [c for c in cf if c["status"] == "ok" and c["B"] == 1]
    if ok:
        ok = sorted(ok, key=lambda c: c["t_vjp_ms"])
        fig, ax = plt.subplots(figsize=(11, 0.34 * len(ok) + 2))
        y = np.arange(len(ok))
        ax.barh(y + 0.2, [c["t_vjp_ms"] for c in ok], 0.4, label="forward+VJP")
        ax.barh(y - 0.2, [c["t_fwd_ms"] for c in ok], 0.4, label="forward", color="tab:gray")
        ax.set_yticks(y), ax.set_yticklabels([c["label"] for c in ok], fontsize=7), ax.set_xscale("log")
        ax.set_xlabel(f"median ms ({be}), B=1"), ax.legend(), ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "timing.png"), dpi=100)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    names = sorted({c["name"] for c in cf if c["status"] == "ok" and c["B"] > 1})
    for n in names:
        pts = sorted([(c["B"], c["ms_per_case"]) for c in cf if c["name"] == n and c["status"] == "ok"
                      and c.get("remat") in (None, next((x.get("remat") for x in cf if x["name"] == n and x["B"] > 1), None))
                      and c.get("unroll", 1) == 1])
        if pts:
            ax.plot(*zip(*pts), "o-", label=n)
    ax.set_xscale("log", base=2), ax.set_yscale("log"), ax.set_xlabel("batch B"), ax.set_ylabel(f"VJP ms per case ({be})")
    ax.grid(alpha=.3, which="both")
    if names:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "throughput.png"), dpi=100)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------------------------
def make_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=["accuracy", "timing"], default="accuracy")
    p.add_argument("--eval", default=DEFAULT_EVAL)
    p.add_argument("--stats", default=None, help="default: the eval config's args.stats (resolved), else cardiac/artifacts")
    p.add_argument("--out", default="cardiac/runs/fastsim_cpu")
    p.add_argument("--settings", default="", help="comma list (accuracy default: DEFAULT_ACCURACY_SETTINGS; REF first)")
    p.add_argument("--cases", type=int, default=0, help="first N cases (0 = all)")
    p.add_argument("--points", default=DEFAULT_POINTS)
    p.add_argument("--reps", type=int, default=3, help="timed repetitions (< 3 is flagged indicative)")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--redo", action="store_true", help="ignore results.json / ref_cache.npz (default: resume)")
    p.add_argument("--force", action="store_true", help="run on the GPU even while the guarded PID exists")
    p.add_argument("--guard-pid", type=int, default=GUARD_PID)
    # timing phase
    p.add_argument("--from-accuracy", default=None, help="accuracy output dir to pick the timing settings from")
    p.add_argument("--batches", default="4,16,32", help="extra batch sizes (F02 and the top 2); B=1 always runs")
    p.add_argument("--remat", default="step,none", help="remat modes at B=1 (first one is used for B>1)")
    p.add_argument("--max-minutes", type=float, default=None, help="optional wall budget; default: no cap")
    p.add_argument("--time-cap", type=float, default=120.0, help="seconds per timed phase of one config")
    p.add_argument("--internal-one", default=None, help=argparse.SUPPRESS)
    return p


def main(argv=None):
    a = make_parser().parse_args(argv)
    if a.internal_one is None and gpu_guard(os.environ, a.guard_pid, a.force) == "refuse":
        print(f"REFUSING to run: JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS')!r} is not CPU-only and PID {a.guard_pid} "
              f"(the training job) exists, so the GPU is in use. Run with JAX_PLATFORMS=cpu, wait for the job, or pass "
              f"--force.", file=sys.stderr, flush=True)
        sys.exit(2)
    if a.phase == "accuracy":
        return run_accuracy(a)
    if a.internal_one is not None:
        return timing_child(a)
    if not a.settings and not a.from_accuracy and not a.plot_only:
        raise SystemExit("--phase timing needs --settings or --from-accuracy")
    return run_timing(a)


if __name__ == "__main__":
    main()
