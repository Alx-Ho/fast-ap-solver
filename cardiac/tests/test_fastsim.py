"""Tests of cardiac.fastsim / cardiac.bench_fastsim. CPU only (JAX_PLATFORMS=cpu), fast, not marked slow."""
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")   # before the first jax call: these tests never touch the GPU

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64

from cardiac import fastsim as fs
from cardiac.config import SolverCfg, PARAM_NAMES
from cardiac.physics import FixedAPParams, make_forward

GRID = (-1.0, 1.0, -1.0, 1.0)
VALUES = {"D": 0.001, "eps0": 0.05, "a": 0.09, "k": 8.2, "mu1": 0.07, "mu2": 0.3}
ALL_SCHEMES = ["euler_d0.01", "heun_d0.02", "rk4_d0.05", "tsit5_d0.05", "strang2_d0.05", "strang4_d0.05",
               "imex_euler_d0.05", "etd2_d0.05", "etdrk4_d0.05", "lawson4_d0.05"]


def _params():
    return FixedAPParams.from_values(VALUES)


def _p64(**over):
    """float64 FixedAPParams (leaves are float64 arrays). Values are rounded through float32 so that
    they equal FixedAPParams.from_values(...) (the diffrax reference) exactly."""
    return FixedAPParams(**{k: jnp.asarray(np.float32(over.get(k, VALUES[k])).astype(np.float64)) for k in PARAM_NAMES})


def _wave_ic(n):
    x = np.linspace(-1, 1, n)
    X, _ = np.meshgrid(x, x, indexing="ij")
    return (jnp.asarray((0.8 * (X < -0.6)).astype(np.float32)),
            jnp.asarray((0.1 + 0.4 * (X < -0.2) * (X > -0.6)).astype(np.float32)))


def _rand_ic(n, seed=0, kmax=4, noise=0.0, dtype=np.float64):
    """smooth random IC (low DCT modes) plus optional white noise; u in ~[0.1,0.8], v in ~[0.05,0.16]"""
    rng = np.random.default_rng(seed)
    out = []
    for off, amp in ((0.4, 0.12), (0.1, 0.03)):
        c = np.zeros((n, n))
        c[:kmax, :kmax] = rng.normal(size=(kmax, kmax))
        C = fs.dct_matrix(n)
        f = off + amp * (C.T @ c @ C) * n / kmax + noise * rng.normal(size=(n, n))
        out.append(jnp.asarray(f.astype(dtype)))
    return tuple(out)


def _loss(U, Y):
    return 0.5 * jnp.sum((U - Y) ** 2)


# ---------------------------------------------------------------------------------------------------------------
# (a) (b) (c): DCT eigenbasis, operators
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("n", [8, 17, 32])
def test_a_dct_diagonalises_T(n):
    C, lam, T = fs.dct_matrix(n), fs.eig_lam(n), fs.second_diff_matrix(n)
    np.testing.assert_allclose(C @ C.T, np.eye(n), atol=1e-12)
    np.testing.assert_allclose(C.T @ np.diag(lam) @ C, T, atol=1e-10)
    np.testing.assert_allclose(np.sort(lam), np.linalg.eigvalsh(T), atol=1e-10)


@pytest.mark.parametrize("x64", [True, False])
def test_b_eig_and_dct_laplacian_equal_lap5(x64):
    n, D = 24, 0.37
    rng = np.random.default_rng(1)
    dt = np.float64 if x64 else np.float32
    u_np = rng.normal(size=(3, n, n)).astype(dt)               # non-smooth, includes the boundary
    tol = 1e-10 if x64 else 1e-4

    def run():
        u = jnp.asarray(u_np)
        ops = fs.make_ops(n, GRID, u.dtype)
        ref = D * fs.lap5(u, ops.dx, ops.dy)
        eig = D * fs.from_eig(ops.lhat0 * fs.to_eig(u, ops.C), ops.C)
        dct = D * fs.from_dct(ops.lhat0 * fs.to_dct(u))
        return [np.asarray(a, np.float64) for a in (ref, eig, dct)]

    if x64:
        with enable_x64():
            ref, eig, dct = run()
    else:
        ref, eig, dct = run()
    scale = np.abs(ref).max()
    assert np.abs(eig - ref).max() / scale < tol
    assert np.abs(dct - ref).max() / scale < tol
    assert np.abs(dct - eig).max() / scale < tol


def test_c_sep_exp_equals_eigenbasis_exp():
    n, h, D = 32, 0.1, 0.05
    u = jnp.asarray(np.random.default_rng(2).normal(size=(2, n, n)).astype(np.float32))
    ops = fs.make_ops(n, GRID, jnp.float32)
    Px, Py = fs.sep_exp(h, D, ops)
    a = fs._mm(fs._mm(Px, u), Py.T)
    b = fs.from_eig(jnp.exp(h * D * ops.lhat0) * fs.to_eig(u, ops.C), ops.C)
    c = fs.make_diffuser("dct", h, D, ops)(u)
    s = float(jnp.abs(b).max())
    assert float(jnp.abs(a - b).max()) / s < 1e-5
    assert float(jnp.abs(c - b).max()) / s < 1e-5


# ---------------------------------------------------------------------------------------------------------------
# (p) phi functions
# ---------------------------------------------------------------------------------------------------------------
def test_p_phi_contour():
    z0 = fs.phi_contour_all(jnp.zeros(()), (1, 2, 3))
    np.testing.assert_allclose([float(z0[1]), float(z0[2]), float(z0[3])], [1.0, 0.5, 1 / 6], atol=1e-6)
    zs = -np.concatenate([np.linspace(0.5, 3, 100), np.linspace(3, 20, 100)])
    got = fs.phi_contour_all(jnp.asarray(zs, jnp.float32), (1, 2, 3))
    with enable_x64():
        zz = jnp.asarray(zs)
        ref = {1: (jnp.exp(zz) - 1) / zz, 2: (jnp.exp(zz) - 1 - zz) / zz ** 2,
               3: (jnp.exp(zz) - 1 - zz - zz ** 2 / 2) / zz ** 3}
        ref = {k: np.asarray(v) for k, v in ref.items()}
        got64 = {k: np.asarray(v) for k, v in fs.phi_contour_all(zz, (1, 2, 3)).items()}
    for k in (1, 2, 3):
        # float32 contour: 1e-5 relative (+ 1e-6 absolute: phi_3 loses digits when the unit circle passes near 0, |z|~1)
        np.testing.assert_allclose(np.asarray(got[k]), ref[k], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(got64[k], ref[k], rtol=1e-10)
    # differentiable wrt D (z = h D lam): vs the closed-form derivative phi_1'(z) = (phi_0 - phi_1)/z ... and FD
    lam = jnp.asarray(-np.linspace(0.3, 30, 7), jnp.float32)

    def f(D):
        return jnp.sum(fs.phi_contour(0.1 * D * lam, 2))
    g = jax.grad(f)(jnp.float32(0.5))
    fd = (f(jnp.float32(0.5 + 1e-2)) - f(jnp.float32(0.5 - 1e-2))) / 2e-2
    assert np.isfinite(float(g)) and abs(float(g) - float(fd)) < 2e-3 * max(1.0, abs(float(fd)))
    # phi_k'(z) = (phi_{k-1}(z) - k phi_k(z)) / z for z != 0, checked on phi_2
    z = jnp.asarray(-1.7, jnp.float32)
    d = jax.grad(lambda z: fs.phi_contour(z, 2))(z)
    p1, p2 = fs.phi_contour(z, 1), fs.phi_contour(z, 2)
    assert abs(float(d) - float((p1 - 2 * p2) / z)) < 1e-4


# ---------------------------------------------------------------------------------------------------------------
# (d) tsit5 in the scan reproduces diffrax F02
# ---------------------------------------------------------------------------------------------------------------
def test_d_tsit5_reproduces_f02():
    n, taus = 32, [0.5, 1.0]
    P = _params()
    u0, v0 = _wave_ic(n)
    f02 = make_forward(n, GRID, taus, SolverCfg(stepping="fixed", dtmax=0.02, checkpoints=1024), P)
    fast = fs.make_forward_fast(n, GRID, taus, fs.parse_setting("tsit5_d0.02"), P)
    Ur, Vr, ir = jax.jit(f02)(u0, v0)
    U, V, info = jax.jit(fast)(u0, v0)
    assert U.shape == (2, n, n) and bool(info["ok"]) and info["num_steps"] == 50
    assert float(jnp.abs(U - Ur).max()) <= 1e-5 and float(jnp.abs(V - Vr).max()) <= 1e-5
    Y = Ur + 0.05 * jnp.cos(7 * Ur)

    def L(fwd):
        return jax.jit(jax.grad(lambda u, v: _loss(fwd(u, v)[0], Y), argnums=(0, 1)))
    gr, gf = L(f02)(u0 * 0.9, v0 * 1.1 + 0.01), L(fast)(u0 * 0.9, v0 * 1.1 + 0.01)
    a = np.concatenate([np.asarray(x).ravel() for x in gr]).astype(np.float64)
    b = np.concatenate([np.asarray(x).ravel() for x in gf]).astype(np.float64)
    assert np.linalg.norm(a - b) / np.linalg.norm(a) <= 1e-4


# ---------------------------------------------------------------------------------------------------------------
# (e) (f) (g): batching, segments, remat
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["tsit5_d0.05", "strang2_d0.05", "etd2_d0.05", "lawson4_d0.05"])
def test_e_batched_equals_unbatched_equals_vmap(name):
    n, taus = 16, [0.1, 0.2]
    P = _params()
    cfg = fs.parse_setting(name)
    u1, v1 = _wave_ic(n)
    rng = np.random.default_rng(3)
    ub = jnp.stack([u1, jnp.clip(u1 * 0.7 + 0.1, 0, 1), jnp.asarray(rng.uniform(0.0, 0.6, (n, n)), jnp.float32)])
    vb = jnp.stack([v1, v1 * 0.5, jnp.asarray(rng.uniform(0.0, 0.3, (n, n)), jnp.float32)])
    one = jax.jit(fs.make_forward_fast(n, GRID, taus, cfg, P))
    bat = jax.jit(fs.make_forward_fast_batched(n, GRID, taus, cfg, P))
    Ub, Vb, info = bat(ub, vb)
    assert Ub.shape == (3, 2, n, n) and Vb.shape == (3, 2, n, n) and bool(info["ok"])
    Uv, Vv, _ = jax.jit(jax.vmap(fs.make_forward_fast(n, GRID, taus, cfg, P)))(ub, vb)
    for b in range(3):
        U1, V1, _ = one(ub[b], vb[b])
        assert float(jnp.abs(Ub[b] - U1).max()) < 1e-5 and float(jnp.abs(Vb[b] - V1).max()) < 1e-5
        assert float(jnp.abs(Uv[b] - U1).max()) < 1e-5 and float(jnp.abs(Vv[b] - V1).max()) < 1e-5


def test_f_off_grid_taus_raise():
    cfg = fs.parse_setting("rk4_d0.05")
    for taus in ([0.1, 0.17], [0.03, 0.1], [0.1, 0.1], [0.0, 0.1]):
        with pytest.raises(ValueError):
            fs.make_forward_fast(8, GRID, taus, cfg, _params())
    with pytest.raises(ValueError):
        fs.make_forward_fast(8, GRID, [0.1, 0.2], fs.FastCfg(scheme="euler", dt=0.01, diff_op="sep"), _params())
    with pytest.raises(ValueError):
        fs.make_forward_fast(8, GRID, [0.1, 0.2], fs.FastCfg(scheme="etd2", dt=0.05, diff_op="sep"), _params())
    fs.make_forward_fast(8, GRID, [0.1, 0.2], fs.FastCfg(scheme="euler", dt=0.01), _params())   # on grid: fine


@pytest.mark.parametrize("name", ["tsit5_d0.05", "etd2_d0.05", "lawson4_d0.05"])
def test_g_remat_modes_agree(name):
    n, taus = 16, [0.1, 0.2]
    u0, v0 = _wave_ic(n)
    Y = jnp.asarray(np.random.default_rng(4).uniform(0, 0.5, (2, n, n)), jnp.float32)
    gs = {}
    for rm in ("none", "step", "segment"):
        cfg = fs.parse_setting(name)
        cfg.remat = rm
        fp = fs.make_forward_fast_p(n, GRID, taus, cfg)

        def loss(u, v, p):
            return _loss(fp(p, u, v)[0], Y)
        gs[rm] = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(u0, v0, _params())
    ref = jax.tree_util.tree_leaves(gs["none"])
    for rm in ("step", "segment"):
        for a, b in zip(ref, jax.tree_util.tree_leaves(gs[rm])):
            assert float(jnp.linalg.norm(a - b)) <= 1e-5 * max(float(jnp.linalg.norm(a)), 1e-12) + 1e-30


def test_unequal_segments_python_loop_matches_scan_path():
    n = 16
    u0, v0 = _wave_ic(n)
    cfg = fs.parse_setting("etdrk4_d0.05")
    Ua, Va, ia = jax.jit(fs.make_forward_fast(n, GRID, [0.1, 0.3], cfg, _params()))(u0, v0)       # unequal -> loop
    Ub, Vb, ib = jax.jit(fs.make_forward_fast(n, GRID, [0.1, 0.2, 0.3], cfg, _params()))(u0, v0)  # equal -> scan
    assert ia["num_steps"] == ib["num_steps"] == 6
    assert float(jnp.abs(Ua[0] - Ub[0]).max()) < 1e-6 and float(jnp.abs(Ua[1] - Ub[2]).max()) < 1e-6
    cfg.remat = "segment"
    Uc, _, _ = jax.jit(fs.make_forward_fast(n, GRID, [0.1, 0.3], cfg, _params()))(u0, v0)
    assert float(jnp.abs(Uc - Ua).max()) < 1e-6


@pytest.mark.parametrize("name", ALL_SCHEMES)
def test_all_schemes_jit_vmap_grad_all_params(name):
    n, taus = 16, [0.1, 0.2]
    cfg = fs.parse_setting(name)
    u0, v0 = _wave_ic(n)
    fp = fs.make_forward_fast_p(n, GRID, taus, cfg)
    U, V, info = jax.jit(fp)(_params(), u0, v0)
    assert U.shape == (2, n, n) and bool(info["ok"])
    Y = U + 0.03

    def loss(p, u, v):
        return _loss(fp(p, u, v)[0], Y)
    gp, gu, gv = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(_params(), u0 * 0.9, v0 * 1.1)
    for k in PARAM_NAMES:
        g = float(getattr(gp, k))
        assert np.isfinite(g) and g != 0.0, (name, k, g)
    assert np.isfinite(np.asarray(gu)).all() and np.isfinite(np.asarray(gv)).all()


def test_setting_name_roundtrip():
    for nm in ["strang2_d0.1", "strang4_d0.05", "etd2_d0.05", "imex_euler_d0.02", "tsit5_d0.02",
               "strang2_d0.1_dct", "etd2_d0.1_eig", "lawson4_d0.1_sep_rs2_rematnone_unroll4", "euler_d0.01"]:
        assert fs.setting_name(fs.parse_setting(nm)) == nm
    assert fs.parse_setting("strang4_d0.2").react == "rk4" and fs.parse_setting("strang2_d0.2").react == "rk2"
    for bad in ["foo_d0.1", "strang2_0.1", "etd2_dxx", "etd2_d0.1_bogus"]:
        with pytest.raises(ValueError):
            fs.parse_setting(bad)


# ---------------------------------------------------------------------------------------------------------------
# (h) (i): convergence and stiffness vs a tight diffrax reference (x64)
# ---------------------------------------------------------------------------------------------------------------
# (scheme, (dt_coarse, dt_fine), formal order p). The step pairs are chosen so that both errors lie in [1e-11, 1e-2]
# and the finest error is < 1e-4; the first-order schemes need dt ~ 1e-4 for that.
CONV = [("euler", (1e-4, 5e-5), 1), ("imex_euler", (1e-4, 5e-5), 1), ("heun", (0.01, 0.005), 2),
        ("strang2", (0.02, 0.01), 2), ("strang4", (0.02, 0.01), 2), ("etd2", (0.01, 0.005), 2),
        ("rk4", (0.1, 0.05), 4), ("tsit5", (0.1, 0.05), 4), ("etdrk4", (0.1, 0.05), 4), ("lawson4", (0.1, 0.05), 4)]


def _reference(n, taus, values, u0, v0):
    fwd = make_forward(n, GRID, taus, SolverCfg(rtol=1e-10, atol=1e-10, dtmax=1e-3, checkpoints=None,
                                                max_steps=20000), FixedAPParams.from_values(values))
    U, V, info = jax.jit(fwd)(u0, v0)
    assert bool(info["ok"])
    return U, V


@pytest.fixture(scope="module")
def conv_ref():
    n, taus = 32, [0.5, 1.0]
    values = dict(VALUES, D=5e-4)          # h * max|Lhat| = h * 8D/dx^2 <~ 1 for h <= 0.1
    with enable_x64():
        u0, v0 = _rand_ic(n, 0)
        Ur, Vr = _reference(n, taus, values, u0, v0)
        return n, taus, values, np.asarray(u0), np.asarray(v0), np.asarray(Ur), np.asarray(Vr)


@pytest.mark.parametrize("scheme,dts,p", CONV)
def test_h_convergence_order(conv_ref, scheme, dts, p):
    n, taus, values, u0, v0, Ur, Vr = conv_ref
    errs = []
    with enable_x64():
        for dt in dts:
            f = jax.jit(fs.make_forward_fast(n, GRID, taus, fs.parse_setting(f"{scheme}_d{dt}"), _p64(D=5e-4)))
            U, V, info = f(jnp.asarray(u0), jnp.asarray(v0))
            assert U.dtype == jnp.float64
            errs.append(max(float(np.abs(np.asarray(U) - Ur).max()), float(np.abs(np.asarray(V) - Vr).max())))
    for e in errs:
        assert 1e-11 <= e <= 1e-2, f"{scheme}: error {e:.3g} outside [1e-11, 1e-2] for dts {dts}; fix the dt pair"
    order = np.log(errs[0] / errs[1]) / np.log(dts[0] / dts[1])
    assert order >= p - 0.3, f"{scheme}: observed order {order:.2f} < {p - 0.3} (errors {errs})"
    assert errs[1] < 1e-4, f"{scheme}: finest error {errs[1]:.3g} >= 1e-4"


STIFF_TOL = 1e-2     # calibrated once (n=32, noisy IC, 8D/dx^2 ~ 100): the worst of the four schemes is < 1e-2


def test_i_stiff_regime():
    n, taus = 32, [0.5, 1.0]
    dx = 2.0 / (n - 1)
    D = 100.0 * dx ** 2 / 8.0
    values = dict(VALUES, D=D)
    with enable_x64():
        u0, v0 = _rand_ic(n, 5, kmax=8, noise=0.01)
        Ur, Vr = _reference(n, taus, values, u0, v0)
        P = _p64(D=D)
        for dt in (0.05, 0.1):
            for name in ("strang2", "strang4", "etd2", "etdrk4", "lawson4"):
                for op in ("auto", "dct"):
                    cfg = fs.parse_setting(f"{name}_d{dt}")
                    cfg.diff_op = op
                    U, V, info = jax.jit(fs.make_forward_fast(n, GRID, taus, cfg, P))(u0, v0)
                    err = float(jnp.abs(U - Ur).max())
                    assert bool(info["ok"]) and np.isfinite(err), (name, dt, op)
                    assert err < STIFF_TOL, f"{name} dt={dt} {op}: max U error {err:.3g}"
        # explicit Euler at dt = 0.05 (h * 8D/dx^2 = 5) is unstable
        U, V, info = jax.jit(fs.make_forward_fast(n, GRID, taus, fs.parse_setting("euler_d0.05"), P))(u0, v0)
        err = float(jnp.abs(U - Ur).max())
        assert (not bool(info["ok"])) or (not np.isfinite(err)) or err > 1.0, f"euler unexpectedly stable: {err}"


# ---------------------------------------------------------------------------------------------------------------
# (j) (o) (k): gradients (x64)
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["strang2_d0.05", "etd2_d0.05", "lawson4_d0.05"])
def test_j_ic_gradient_vs_jvp_and_fd(name):
    n, taus = 16, [0.2, 0.4]
    with enable_x64():
        u0, v0 = _rand_ic(n, 7, kmax=5)
        Y = jnp.asarray(np.random.default_rng(8).uniform(0.1, 0.6, (2, n, n)))
        fwd = fs.make_forward_fast(n, GRID, taus, fs.parse_setting(name), _p64())

        def L(u, v):
            return _loss(fwd(u, v)[0], Y)
        gu, gv = jax.jit(jax.grad(L, argnums=(0, 1)))(u0, v0)
        rng = np.random.default_rng(9)
        du, dv = jnp.asarray(rng.normal(size=(n, n))), jnp.asarray(rng.normal(size=(n, n)))
        gd = float(jnp.vdot(gu, du) + jnp.vdot(gv, dv))
        jv = float(jax.jvp(L, (u0, v0), (du, dv))[1])
        assert abs(gd - jv) <= 1e-4 * abs(jv)
        h = 1e-2
        fd = float(L(u0 + h * du, v0 + h * dv) - L(u0 - h * du, v0 - h * dv)) / (2 * h)
        assert abs(gd - fd) <= 5e-2 * abs(fd), (gd, fd)


@pytest.mark.parametrize("name", ["strang2_d0.005", "etd2_d0.005", "lawson4_d0.05"])
def test_o_param_gradients_vs_diffrax(name):
    n, taus = 16, [0.2, 0.4]
    with enable_x64():
        u0, v0 = _rand_ic(n, 11, kmax=5)
        Y = jnp.asarray(np.random.default_rng(12).uniform(0.1, 0.6, (2, n, n)))
        P0 = _p64(D=0.01)
        fp = fs.make_forward_fast_p(n, GRID, taus, fs.parse_setting(name))
        ref_cfg = SolverCfg(stepping="fixed", dtmax=1e-3, checkpoints=None, max_steps=2000)

        def L_fast(P):
            return _loss(fp(P, u0, v0)[0], Y)

        def L_ref(P):    # local diffrax reference: make_forward passes the params through diffeqsolve's `args`
            return _loss(make_forward(n, GRID, taus, ref_cfg, P)(u0, v0)[0], Y)
        gf = jax.jit(jax.grad(L_fast))(P0)
        gr = jax.jit(jax.grad(L_ref))(P0)
        for k in PARAM_NAMES:
            a, b = float(getattr(gf, k)), float(getattr(gr, k))
            assert abs(b) > 0 and abs(a - b) <= 1e-3 * abs(b), (name, k, a, b)
        t = FixedAPParams(**{k: jnp.asarray(c, jnp.float64) for k, c in zip(PARAM_NAMES, (0.3, -0.5, 0.2, 1.0, -0.7, 0.4))})
        jv = float(jax.jvp(L_fast, (P0,), (t,))[1])
        gd = sum(float(getattr(gf, k)) * float(getattr(t, k)) for k in PARAM_NAMES)
        assert abs(gd - jv) <= 1e-4 * abs(jv)


@pytest.mark.parametrize("name,ops", [("strang2_d0.05", ("sep", "eig", "dct")), ("lawson4_d0.05", ("sep", "eig", "dct")),
                                      ("etd2_d0.05", ("eig", "dct")), ("imex_euler_d0.05", ("eig", "dct")),
                                      ("etdrk4_d0.05", ("eig", "dct"))])
def test_k_operator_variants_agree(name, ops):
    n, taus = 32, [0.2, 0.4]
    u0, v0 = _wave_ic(n)
    outs = []
    for op in ops:
        cfg = fs.parse_setting(name)
        cfg.diff_op = op
        outs.append(jax.jit(fs.make_forward_fast(n, GRID, taus, cfg, _params()))(u0, v0)[0])
    for o in outs[1:]:
        assert float(jnp.abs(o - outs[0]).max()) < 1e-4


# ---------------------------------------------------------------------------------------------------------------
# bench_fastsim: (l) accuracy-phase CLI smoke on the synthetic eval fixture, (m) GPU guard, timing child
# ---------------------------------------------------------------------------------------------------------------
def test_l_bench_cli_smoke(run_eval_cli, tmp_path, capsys):
    from cardiac.bench_fastsim import main
    run_eval_cli("ev")
    ev, out = run_eval_cli.out("ev"), str(tmp_path / "fs")
    argv = ["--phase", "accuracy", "--eval", ev, "--out", out, "--settings", "REF,F02,rk4_d0.02,strang2_d0.1",
            "--no-wandb", "--reps", "1"]
    res = main(argv)
    for f in ("results.json", "summary.md", "ref_cache.npz", "error_vs_cost.png", "gradient_agreement.png",
              "convergence.png"):
        assert os.path.exists(os.path.join(out, f)), f
    assert set(res["settings"]) == {"REF", "F02", "rk4_d0.02", "strang2_d0.1"}
    assert all(r["failed"] is False for r in res["rows"]), [r for r in res["rows"] if r["failed"]]
    pts = {r["point"] for r in res["rows"]}
    assert {"truth", "z_init", "warm_seed0", "adam_final"} <= pts
    cache = np.load(os.path.join(out, "ref_cache.npz"))
    assert any(k.endswith("|g") for k in cache.files)
    summ = open(os.path.join(out, "summary.md")).read()
    assert "indicative" in summ and "no tweedie points" in summ and "speedup" in summ.lower()
    row = next(r for r in res["rows"] if r["setting"] == "rk4_d0.02" and r["point"] == "warm_seed0")
    for k in ("U_rmse_max", "V_rmse_max", "grad_cos", "grad_rel", "t_fwd_ms", "t_vjp_ms", "vjp_over_fwd", "num_steps"):
        assert row[k] is not None and np.isfinite(row[k]), k
    assert res["settings"]["strang2_d0.1"]["cfg"]["scheme"] == "strang"
    assert res["settings"]["F02"]["status"] == "ok" and res["settings"]["F02"]["compile_s"] > 0
    # rerun: everything is skipped (resume); a new setting runs alone and REF comes from the cache
    mt = os.path.getmtime(os.path.join(out, "ref_cache.npz"))
    capsys.readouterr()
    main(argv)
    assert "0/4 settings to run" in capsys.readouterr().out
    main(argv[:-4] + ["REF,F02,rk4_d0.02,strang2_d0.1,etd2_d0.1", "--no-wandb", "--reps", "1"])
    assert "1/5 settings to run" in capsys.readouterr().out
    assert os.path.getmtime(os.path.join(out, "ref_cache.npz")) == mt
    assert "etd2_d0.1" in json.load(open(os.path.join(out, "results.json")))["settings"]
    # plot-only regenerates the report from results.json
    os.remove(os.path.join(out, "summary.md"))
    main(["--phase", "accuracy", "--eval", ev, "--out", out, "--plot-only"])
    assert os.path.exists(os.path.join(out, "summary.md"))


def test_m_gpu_guard_pure(monkeypatch):
    from cardiac.bench_fastsim import gpu_guard, main
    def no_kill(*a, **k):
        raise AssertionError("no signal may be sent")
    monkeypatch.setattr(os, "kill", no_kill)
    pid = os.getpid()                      # a PID that exists
    assert gpu_guard({"JAX_PLATFORMS": "cuda"}, pid, False) == "refuse"
    assert gpu_guard({}, pid, False) == "refuse"
    assert gpu_guard({"JAX_PLATFORMS": "cpu"}, pid, False) == "allow"
    assert gpu_guard({"JAX_PLATFORMS": "cuda"}, pid, True) == "allow"
    assert gpu_guard({"JAX_PLATFORMS": "cuda"}, 2 ** 22 + 12345, False) == "allow"      # no such /proc entry
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    with pytest.raises(SystemExit) as e:
        main(["--phase", "accuracy", "--guard-pid", str(pid), "--out", "/nonexistent/never_created"])
    assert e.value.code == 2 and not os.path.exists("/nonexistent")
    from jax._src import xla_bridge
    assert set(getattr(xla_bridge, "_backends", {})) <= {"cpu"}          # no GPU backend was initialised


def test_timing_child_and_config_plan(run_eval_cli, tmp_path, capsys):
    from cardiac.bench_fastsim import build_timing_configs, make_parser, timing_child, add_speedups
    run_eval_cli("ev")
    ev = run_eval_cli.out("ev")
    base = ["--phase", "timing", "--eval", ev, "--out", str(tmp_path / "t"), "--reps", "1"]
    recs = []
    for spec in ({"name": "F02", "B": 1}, {"name": "F02", "B": 2}, {"name": "strang2_d0.1", "B": 1, "remat": "step"},
                 {"name": "etd2_d0.1_dct", "B": 3, "remat": "none", "unroll": 2}):
        capsys.readouterr()
        timing_child(make_parser().parse_args(base + ["--internal-one", json.dumps(spec)]))
        line = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("BENCH ")][0]
        recs.append(json.loads(line[6:]))
    for r in recs:
        assert r["status"] == "ok" and r["t_vjp_ms"] > 0 and r["t_fwd_ms"] > 0 and r["peak_MB"] is not None
        assert r["backend"] == "cpu"
        if r["B"] > 1:
            assert r["lane0_maxdiff"] < 1e-3
    assert recs[3]["unroll"] == 2 and recs[1]["B"] == 2
    a = make_parser().parse_args(base + ["--settings", "F02,strang2_d0.1,etd2_d0.1", "--remat", "step,none",
                                         "--batches", "1,2"])
    cfgs = build_timing_configs(a, ["F02", "strang2_d0.1", "etd2_d0.1"], ["strang2_d0.1", "etd2_d0.1"])
    labels = [(c["name"], c["B"], c.get("remat"), c.get("unroll", 1)) for c in cfgs]
    assert labels[0] == ("F02", 1, None, 1) and ("strang2_d0.1", 1, "none", 1) in labels
    assert ("F02", 2, None, 1) in labels and ("strang2_d0.1", 2, "step", 1) in labels
    assert ("strang2_d0.1", 1, "segment", 1) in labels and ("etd2_d0.1", 1, "step", 4) in labels
    res = {"configs": [dict(r, label="x") for r in recs]}
    add_speedups(res)
    assert res["configs"][0]["speedup_vs_F02"] == pytest.approx(1.0)
