"""Tests of cardiac.fastsim_param_check (parameter gradients / recovery of the fast solvers). CPU only, fast.

(b) BoundedLogRDParams(direct) goes into both forwards; raw-leaf gradients = param * d/dparam inside the bounds.
(c) the theta perturbation is deterministic, within +-20-40%, and every start has D < D_max_F02.
(d) a tiny CPU `all` run (1 case, lawson4_d0.2 vs F02 ref, taus <= 1, 3 iterations) writes summary.md and both pngs; the GPU
    guard refuses while the guarded pid exists.
(a) B0: finite-difference / JVP checks of the theta-gradient of fastsim (traced theta) and of physics.make_forward (F02,
    traced FixedAPParams) on a tiny n=16 grid in float64. Catches operators precomputed from a static D/theta.
"""
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")   # before the first jax call: these tests never touch the GPU

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64

from cardiac import fastsim as fs
from cardiac.config import PARAM_NAMES, SolverCfg
from cardiac import fastsim_param_check as pc
from cardiac.physics import FixedAPParams, make_forward

GRID = (-1.0, 1.0, -1.0, 1.0)
VALUES = {"D": 0.005, "eps0": 0.05, "a": 0.09, "k": 8.2, "mu1": 0.07, "mu2": 0.3}
TAUS = [0.2, 0.4]
FAST = ["lawson4_d0.1", "etdrk4_d0.1", "strang4_d0.05", "rk4_d0.025", "lawson4_d0.2_unroll4"]
N = 16


class _P:
    def __init__(self, vals):
        for k, v in zip(PARAM_NAMES, vals):
            setattr(self, k, v)


def _ic(seed=0):
    rng = np.random.default_rng(seed)
    C = fs.dct_matrix(N)
    out = []
    for off, amp in ((0.4, 0.12), (0.1, 0.03)):
        c = np.zeros((N, N))
        c[:4, :4] = rng.normal(size=(4, 4))
        out.append(jnp.asarray(off + amp * (C.T @ c @ C) * N / 4 + 0.02 * rng.normal(size=(N, N))))
    return out


def _fwd_fast(name):
    return fs.make_forward_fast_p(N, GRID, TAUS, fs.parse_setting(name))


def _fwd_f02():
    cfg = SolverCfg(stepping="fixed", dtmax=0.02, checkpoints=1024)

    def fn(p, u0, v0):
        return make_forward(N, GRID, TAUS, cfg, FixedAPParams(**{k: getattr(p, k) for k in PARAM_NAMES}))(u0, v0)
    return fn


def _loss_of_logtheta(fwd, u0, v0, Y):
    def loss(lt):
        U, _, _ = fwd(_P([jnp.exp(lt[i]) for i in range(6)]), u0, v0)
        return 0.5 * jnp.sum((U - Y) ** 2)
    return loss


def _setup():
    u0, v0 = _ic()
    lt0 = jnp.log(jnp.asarray([VALUES[k] for k in PARAM_NAMES], jnp.float64))
    Uref, _, _ = _fwd_fast("lawson4_d0.05")(_P(list(jnp.exp(lt0))), u0, v0)
    Y = Uref + 0.05 * jnp.asarray(np.random.default_rng(1).normal(size=Uref.shape))
    lt = lt0 + jnp.log(jnp.asarray([1.3, 0.75, 1.25, 1.2, 0.8, 1.3]))   # evaluate away from the data-generating theta
    return u0, v0, Y, lt


def _fd_grad(loss, lt, h=1e-3):
    g = []
    for i in range(6):
        e = jnp.zeros(6).at[i].set(h)
        g.append(float((loss(lt + e) - loss(lt - e)) / (2 * h)))
    return np.array(g)


def _check(name, fwd, store, forward_mode=True):
    with enable_x64():
        u0, v0, Y, lt = _setup()
        loss = _loss_of_logtheta(fwd, u0, v0, Y)
        g_ad = np.asarray(jax.grad(loss)(lt))
        g_fd = _fd_grad(loss, lt)
        assert np.all(np.isfinite(g_ad)), (name, g_ad)
        scale = np.linalg.norm(g_fd)
        for i, k in enumerate(PARAM_NAMES):
            assert abs(g_fd[i]) > 1e-4 * scale, f"{name}: FD sensitivity to {k} is ~0, check is meaningless ({g_fd})"
            assert abs(g_ad[i] - g_fd[i]) <= 1e-2 * abs(g_fd[i]) + 1e-6 * scale, (name, k, g_ad[i], g_fd[i])
        # JVP along a random direction in log-theta vs central difference along it
        d = jnp.asarray(np.random.default_rng(3).normal(size=6))
        # diffrax's RecursiveCheckpointAdjoint is reverse-only (custom_vjp): F02 uses the VJP contracted with d instead
        jvp = float(jax.jvp(loss, (lt,), (d,))[1]) if forward_mode else float(g_ad @ np.asarray(d))
        h = 1e-3
        fd = float((loss(lt + h * d) - loss(lt - h * d)) / (2 * h))
        assert abs(jvp - fd) <= 1e-2 * abs(fd), (name, jvp, fd)
        store[name] = g_ad


GRADS = {}


@pytest.mark.parametrize("name", FAST)
def test_a_fast_theta_grad_vs_fd(name):
    _check(name, _fwd_fast(name), GRADS)


def test_a_f02_theta_grad_vs_fd():
    _check("F02", _fwd_f02(), GRADS, forward_mode=False)


def test_a_fast_grads_agree_with_f02():
    for nm in ["F02"] + FAST:
        if nm not in GRADS:
            _check(nm, _fwd_f02() if nm == "F02" else _fwd_fast(nm), GRADS, forward_mode=nm != "F02")
    g0 = GRADS["F02"]
    for nm in FAST:
        g = GRADS[nm]
        cos = g @ g0 / (np.linalg.norm(g) * np.linalg.norm(g0))
        assert cos > 0.99 and np.linalg.norm(g - g0) <= 0.1 * np.linalg.norm(g0), (nm, cos, g, g0)


# ---------------------------------------------------------------------------------------------------------------
# (b) BoundedLogRDParams(direct) as `params`
# ---------------------------------------------------------------------------------------------------------------
def _bounded(vals):
    import equinox as eqx
    from ap_simulator.params import BoundedLogRDParams
    m = BoundedLogRDParams(**{k: jnp.asarray(vals[k]) for k in PARAM_NAMES}, bounds=pc.widened_bounds(VALUES),
                           mapping_mode="direct")
    return m, eqx


@pytest.mark.parametrize("which", ["lawson4_d0.1", "F02"])
def test_b_bounded_log_params_pass_through_both_forwards(which):
    with enable_x64():
        u0, v0 = _ic()
        fwd = _fwd_f02() if which == "F02" else _fwd_fast(which)
        model, eqx = _bounded(VALUES)
        diff, static = eqx.partition(model, eqx.is_inexact_array)
        Y = fwd(_P([jnp.asarray(VALUES[k]) for k in PARAM_NAMES]), u0, v0)[0] + 0.03

        def loss_raw(d):
            return 0.5 * jnp.sum((fwd(eqx.combine(d, static), u0, v0)[0] - Y) ** 2)

        def loss_phys(vec):
            return 0.5 * jnp.sum((fwd(_P([vec[i] for i in range(6)]), u0, v0)[0] - Y) ** 2)

        g_raw = jax.grad(loss_raw)(diff)
        g_phys = np.asarray(jax.grad(loss_phys)(jnp.asarray([VALUES[k] for k in PARAM_NAMES])))
        g_raw = np.array([float(getattr(g_raw, "raw_" + k)) for k in PARAM_NAMES])
        assert np.all(np.isfinite(g_raw))
        np.testing.assert_allclose(g_raw, np.array([VALUES[k] for k in PARAM_NAMES]) * g_phys, rtol=1e-6, atol=1e-9 * np.abs(g_raw).max())


# ---------------------------------------------------------------------------------------------------------------
# (c) perturbation + D_max_F02
# ---------------------------------------------------------------------------------------------------------------
def test_c_perturbation_deterministic_and_stable_starts():
    true = {"D": 0.001, "eps0": 0.05, "a": 0.09, "k": 8.2, "mu1": 0.07, "mu2": 0.3}
    dx = 2.0 / 223
    dmax = pc.d_max_f02(dx)
    assert abs(dmax - 1.76e-3) < 2e-5
    for s in (0, 1, 2):
        f = pc.perturb_factors(s)
        assert f == pc.perturb_factors(s)
        for k, x in f.items():
            assert 1.2 - 1e-9 <= max(x, 1 / x) <= 1.4 + 1e-9, (s, k, x)
        assert pc.perturbed_theta(true, s)["D"] < dmax
    assert pc.perturb_factors(0) != pc.perturb_factors(1)
    pc.check_starts(true, (0, 1, 2), dmax)
    with pytest.raises(AssertionError):
        pc.check_starts({**true, "D": 0.002}, (0,), dmax)


# ---------------------------------------------------------------------------------------------------------------
# (d) tiny end-to-end run + guard
# ---------------------------------------------------------------------------------------------------------------
def test_d_gpu_guard_refuses(monkeypatch, tmp_path):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    assert pc.main(["grad", "--train-pid", str(os.getpid()), "--out", str(tmp_path)]) == 2
    assert pc.bf.gpu_guard({"JAX_PLATFORMS": "cpu"}, os.getpid()) == "allow"
    assert pc.bf.gpu_guard({}, os.getpid(), force=True) == "allow"


def test_d_tiny_all_run(tmp_path):
    ev = next((e for e in ("cardiac/runs/eval_test_n20", pc.FALLBACK_EVAL) if os.path.isdir(e)), None)
    if ev is None or not os.path.exists("cardiac/artifacts/split.json"):
        pytest.skip("eval dir missing")
    try:
        with open(os.path.join(ev, "chunk_0", "config.json")) as f:
            root = json.load(f)["args"]["data_root"]
        if not os.path.exists(os.path.join(root, "config.json")):
            pytest.skip("data root missing")
    except OSError:
        pytest.skip("eval config missing")
    out = tmp_path / "pc"
    rc = pc.main(["all", "--eval", ev, "--cases", "1", "--recover-cases", "1", "--seeds", "0", "--methods", "lawson4_d0.2",
                  "--settings", "lawson4_d0.2", "--ref", "F02", "--iters", "3", "--taus-max", "1", "--no-wandb",
                  "--out", str(out), "--media-dir", str(tmp_path / "media")])
    assert rc == 0
    for f in ("summary.md", "param_grad_error.png", "param_recovery.png", "grad_results.json", "ref_grads.npz"):
        assert (out / f).exists(), f
    res = json.load(open(out / "grad_results.json"))
    assert res["settings"]["lawson4_d0.2"]["status"] == "ok"
    assert (tmp_path / "media" / "param_summary.md").exists()
    assert "D_max_F02" in (out / "summary.md").read_text()
