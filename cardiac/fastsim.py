"""Fixed-step, plain-JAX (`lax.scan`) forward solvers for the 2D Aliev-Panfilov system (prototype, benchmark only).

Same semi-discrete ODE as `physics.make_forward_batched` / ap_simulator.RDSystem:

    du = D*lap5(u) - k u (u-a)(u-1) - u v          (lap5: 5-point stencil, edge-replicate boundary)
    dv = (eps0 + mu1 v/(u+mu2)) (-v - k u (u-a-1))

Two families of schemes:
  * explicit RK inside the scan (euler, heun, rk4, tsit5), stability-limited to dt <~ 0.02-0.03;
  * exact diffusion in the DCT eigenbasis (strang, imex_euler, etd2, etdrk4, lawson4), no diffusion stability limit.
    The edge-replicate second-difference matrix T is diagonalised exactly by the orthonormal DCT-II matrix C:
    C T C^T = diag(lam), lam_k = -4 sin^2(pi k / 2n). exp(h D Lap) = Px (x) Py is separable (2 matmuls).

All operators act on [..., n, n] arrays and follow the input dtype. Params (`physics.FixedAPParams`-like objects with
D, eps0, a, k, mu1, mu2) stay traced, so gradients flow to u0, v0 and all 6 params.

Factories (one implementation, works on any leading dims):
  make_forward_fast_p(n, grid, save_taus, cfg)           -> fn(params, u0, v0)
  make_forward_fast(n, grid, save_taus, cfg, params)     -> fn(u0[n,n], v0[n,n])     -> (U[M,n,n], V[M,n,n], info)
  make_forward_fast_batched(n, grid, save_taus, cfg, p)  -> fn(u0[B,n,n], v0[B,n,n]) -> (U[B,M,n,n], V[B,M,n,n], info)
"""
import re
from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from .config import PARAM_NAMES, _Cfg

HI = jax.lax.Precision.HIGHEST
SCHEMES = ("euler", "heun", "rk4", "tsit5", "strang", "imex_euler", "etd2", "etdrk4", "lawson4")
EXPLICIT = ("euler", "heun", "rk4", "tsit5")
DIFF_OPS = ("auto", "lap5", "sep", "eig", "dct")


@dataclass
class FastCfg(_Cfg):
    scheme: str = "tsit5"
    dt: float = 0.02
    diff_op: str = "auto"      # auto | lap5 | sep | eig | dct
    react: str = "rk2"         # strang only: rk2 (Heun) | rk4
    react_sub: int = 1         # strang only: reaction substeps per half step
    remat: str = "step"        # step | segment | none  (CPU: step is also the fastest, see REPORT)
    unroll: int = 1


def resolve_diff_op(cfg):
    """`auto` -> sep (strang, lawson4), eig (etd2, etdrk4, imex_euler), lap5 (explicit). Validates the combination."""
    if cfg.scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {cfg.scheme!r}")
    if cfg.diff_op not in DIFF_OPS:
        raise ValueError(f"unknown diff_op {cfg.diff_op!r}")
    if cfg.scheme in EXPLICIT:
        allowed, default = ("lap5",), "lap5"
    elif cfg.scheme in ("strang", "lawson4"):
        allowed, default = ("sep", "eig", "dct"), "sep"
    else:
        allowed, default = ("eig", "dct"), "eig"
    op = default if cfg.diff_op == "auto" else cfg.diff_op
    if op not in allowed:
        raise ValueError(f"diff_op {op!r} not valid for scheme {cfg.scheme!r} (allowed: {allowed})")
    if cfg.remat not in ("step", "segment", "none"):
        raise ValueError(f"unknown remat {cfg.remat!r}")
    if cfg.scheme == "strang" and cfg.react not in ("rk2", "rk4"):
        raise ValueError(f"unknown react {cfg.react!r}")
    return op


def rhs_per_step(cfg):
    """Nominal number of reaction/RHS evaluations per step."""
    s = cfg.scheme
    if s == "strang":
        return 2 * cfg.react_sub * (2 if cfg.react == "rk2" else 4)
    return {"euler": 1, "heun": 2, "rk4": 4, "tsit5": 6, "imex_euler": 1, "etd2": 2, "etdrk4": 4, "lawson4": 4}[s]


# ----------------------------------------------------------------------------------------------------------------
# operators on [..., n, n]
# ----------------------------------------------------------------------------------------------------------------
def _mm(a, b):
    return jnp.matmul(a, b, precision=HI)


def lap5(u, dx, dy):
    """5-point Laplacian, edge-replicate boundary (identical to RDSystem.laplacian), on the last two axes."""
    pw = [(0, 0)] * (u.ndim - 2) + [(1, 1), (1, 1)]
    up = jnp.pad(u, pw, mode="edge")
    return ((up[..., 2:, 1:-1] - 2 * up[..., 1:-1, 1:-1] + up[..., :-2, 1:-1]) / dx ** 2 +
            (up[..., 1:-1, 2:] - 2 * up[..., 1:-1, 1:-1] + up[..., 1:-1, :-2]) / dy ** 2)


def dct_matrix(n):
    """Orthonormal 1D DCT-II matrix, float64 numpy: C[k,j] = s_k cos(pi k (2j+1)/(2n)), s_0 = sqrt(1/n), else sqrt(2/n)."""
    k = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    C = np.cos(np.pi * k * (2 * j + 1) / (2 * n)) * np.sqrt(2.0 / n)
    C[0, :] = np.sqrt(1.0 / n)
    return C


def eig_lam(n):
    """Eigenvalues -4 sin^2(pi k/(2n)) of the edge-replicate second-difference matrix T (float64 numpy)."""
    return -4.0 * np.sin(np.pi * np.arange(n) / (2.0 * n)) ** 2


def second_diff_matrix(n):
    """T (float64 numpy): tridiagonal (1,-2,1) with -1 in both corners (edge-replicate boundary)."""
    T = (np.diag(-2.0 * np.ones(n)) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1))
    T[0, 0] = T[-1, -1] = -1.0
    return T


def to_eig(u, C):
    """uhat = C u C^T (matmul DCT-II), batched over leading dims."""
    return _mm(_mm(C, u), C.T)


def from_eig(uh, C):
    return _mm(_mm(C.T, uh), C)


def to_dct(u):
    return jax.scipy.fft.dctn(u, type=2, norm="ortho", axes=(-2, -1))


def _idct_last(X):
    """Orthonormal inverse DCT-II (DCT-III) along the last axis via one FFT (Makhoul), dtype-preserving.
    (jax 0.4.30's idctn silently casts to float32, which would defeat the x64 checks.)"""
    N = X.shape[-1]
    cdt = jnp.complex128 if X.dtype == jnp.float64 else jnp.complex64
    sc = np.full(N, np.sqrt(2.0 * N))
    sc[0] = np.sqrt(4.0 * N)
    Y = X * jnp.asarray(sc, X.dtype)
    Yr = jnp.concatenate([jnp.zeros_like(Y[..., :1]), Y[..., :0:-1]], -1)      # Y[N-k], Y[N] = 0
    V = (Y - 1j * Yr) * jnp.asarray(np.exp(1j * np.pi * np.arange(N) / (2 * N)) / 2, cdt)
    v = jnp.real(jnp.fft.ifft(V))
    return jnp.zeros_like(X).at[..., 0::2].set(v[..., :(N + 1) // 2]).at[..., 1::2].set(v[..., ::-1][..., :N // 2])


def from_dct(uh):
    """Inverse of to_dct (type 2, ortho, axes (-2,-1))."""
    return jnp.swapaxes(_idct_last(jnp.swapaxes(_idct_last(uh), -1, -2)), -1, -2)


def make_ops(n, grid, dtype):
    """Constants of one problem size (numpy float64 -> `dtype`)."""
    x0, x1, y0, y1 = grid
    dx, dy = (x1 - x0) / (n - 1), (y1 - y0) / (n - 1)
    lam = eig_lam(n)
    return SimpleNamespace(
        n=n, dx=dx, dy=dy, dtype=dtype, C=jnp.asarray(dct_matrix(n), dtype),
        lamx=jnp.asarray(lam / dx ** 2, dtype), lamy=jnp.asarray(lam / dy ** 2, dtype),
        lhat0=jnp.asarray(lam[:, None] / dx ** 2 + lam[None, :] / dy ** 2, dtype))


def sep_exp(h, D, ops):
    """Separable propagator of exp(h D Lap): returns (Px, Py) with exp(hDL) u = Px u Py^T (exact, 2 matmuls)."""
    C = ops.C
    Px = _mm(C.T, jnp.exp(h * D * ops.lamx)[:, None] * C)
    Py = _mm(C.T, jnp.exp(h * D * ops.lamy)[:, None] * C)
    return Px, Py


def make_diffuser(kind, h, D, ops):
    """u -> exp(h D Lap) u via the separable (sep), matmul-eigenbasis (eig) or dctn (dct) path."""
    if kind == "sep":
        Px, Py = sep_exp(h, D, ops)
        return lambda u: _mm(_mm(Px, u), Py.T)
    E = jnp.exp(h * D * ops.lhat0)
    if kind == "eig":
        return lambda u: from_eig(E * to_eig(u, ops.C), ops.C)
    if kind == "dct":
        return lambda u: from_dct(E * to_dct(u))
    raise ValueError(kind)


def _transforms(kind, ops):
    if kind == "eig":
        return (lambda u: to_eig(u, ops.C)), (lambda uh: from_eig(uh, ops.C))
    if kind == "dct":
        return to_dct, from_dct
    raise ValueError(kind)


def phi_contour_all(z, ks=(1, 2, 3), M=64):
    """phi_k(z), k in ks (phi_0 = exp), for REAL z via the Kassam-Trefethen contour mean:
    phi_k(z) = mean_j Re phi_k_direct(z + R exp(i theta_j)), theta_j on the upper half circle (conjugate symmetry).
    No series and no branching, so it is valid at z = 0; complex64 for float32 input. Differentiable.
    DEVIATION from the plan's fixed radius 1: R = 1 + |z| (constant under autodiff). With R = 1 the circle passes within
    ~0.05 of the removable singularity w = 0 when z ~ -1 and phi_3 loses ~4 digits in complex64 (measured: 2e-5 relative,
    plan test needs 1e-5); with R = 1 + |z| the origin is always 1 away from the contour (|w| >= 1). M = 64 half-circle
    points keep the aliasing error below float32 roundoff for |z| <= 60 (measured <= 3e-6 relative). Returns a dict."""
    z = jnp.asarray(z)
    cdt = jnp.complex128 if z.dtype == jnp.float64 else jnp.complex64
    th = np.pi * (np.arange(M) + 0.5) / M
    R = jax.lax.stop_gradient(1.0 + jnp.abs(z))[..., None]
    w = z[..., None].astype(cdt) + R.astype(cdt) * jnp.asarray(np.exp(1j * th), cdt)
    e = jnp.exp(w)
    fs = {0: e, 1: (e - 1) / w, 2: (e - 1 - w) / w ** 2, 3: (e - 1 - w - w ** 2 / 2) / w ** 3}
    return {k: jnp.mean(jnp.real(fs[k]), axis=-1) for k in ks}


def phi_contour(z, k, M=64):
    return phi_contour_all(z, (k,), M)[k]


def etdrk4_coeffs(z, M=64):
    """Cox-Matthews ETDRK4 coefficients at z = h*Lhat (all elementwise, via phi functions):
    E = e^z, E2 = e^{z/2}, Q = phi1(z/2) (stage weight, times h/2), f1 = p1-3p2+4p3, f2 = p2-2p3, f3 = -p2+4p3."""
    p = phi_contour_all(z, (1, 2, 3), M)
    q = phi_contour_all(z / 2, (1,), M)[1]
    return SimpleNamespace(E=jnp.exp(z), E2=jnp.exp(z / 2), Q=q, f1=p[1] - 3 * p[2] + 4 * p[3],
                           f2=p[2] - 2 * p[3], f3=-p[2] + 4 * p[3])


def resolvent(h, D, ops):
    """(1 - h D Lhat)^-1 in the eigenbasis."""
    return 1.0 / (1.0 - h * D * ops.lhat0)


# ----------------------------------------------------------------------------------------------------------------
# reaction + steppers
# ----------------------------------------------------------------------------------------------------------------
def reaction(u, v, P):
    """Pointwise Aliev-Panfilov reaction (Nu, Nv); v never diffuses."""
    eps = P.eps0 + (P.mu1 * v) / (u + P.mu2)
    du = -P.k * u * (u - P.a) * (u - 1) - u * v
    dv = eps * (-v - P.k * u * (u - P.a - 1))
    return du, dv


_TSIT5 = dict(
    A=[[0.161], [-0.008480655492356989, 0.335480655492357],
       [2.8971530571054935, -6.359448489975075, 4.3622954328695815],
       [5.325864828439257, -11.748883564062828, 7.4955393428898365, -0.09249506636175525],
       [5.86145544294642, -12.92096931784711, 8.159367898576159, -0.071584973281401, -0.028269050394068383]],
    b=[0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774])


def tsit5_tableau():
    """(A rows 2..6, b[0..5]) of Tsitouras 5(4); from diffrax's private tableau, with a hard-coded fallback."""
    try:
        from diffrax._solver.tsit5 import _tsit5_tableau as t
        A = [[float(x) for x in r] for r in t.a_lower[:5]]
        b = [float(x) for x in t.b_sol[:6]]
        assert len(A) == 5 and len(b) == 6 and float(t.b_sol[6]) == 0.0
        return A, b
    except Exception:  # noqa: BLE001 - private symbol, fall back to the constants above
        return _TSIT5["A"], _TSIT5["b"]


_RK = {"euler": ([], [1.0]),
       "heun": ([[1.0]], [0.5, 0.5]),
       "rk4": ([[0.5], [0.0, 0.5], [0.0, 0.0, 1.0]], [1 / 6, 1 / 3, 1 / 3, 1 / 6])}


def _rk_tableau(name):
    return tsit5_tableau() if name == "tsit5" else _RK[name]


def _erk_step(f, y, h, A, b):
    """Explicit RK step for y = (u, v) with rhs f(u, v) -> (du, dv); A = strictly lower rows (stage 2..), b weights."""
    ks = [f(*y)]
    for row in A:
        yi = tuple(c + h * sum(a * k[i] for a, k in zip(row, ks) if a != 0.0) for i, c in enumerate(y))
        ks.append(f(*yi))
    return tuple(c + h * sum(bj * k[i] for bj, k in zip(b, ks) if bj != 0.0) for i, c in enumerate(y))


def _react_substeps(u, v, tau, sub, P, kind):
    """Integrate (u,v)' = N(u,v) over time tau with `sub` RK2 (Heun) / RK4 substeps."""
    A, b = _RK["heun" if kind == "rk2" else "rk4"]
    f = lambda a, c: reaction(a, c, P)
    hs = tau / sub
    for _ in range(sub):
        u, v = _erk_step(f, (u, v), hs, A, b)
    return u, v


def build_step(cfg, ops, P, h, kind):
    """Return step(u, v) -> (u, v) for one step of size h. Coefficients are built here (outside the scan)."""
    s = cfg.scheme
    D = P.D
    N = lambda a, c: reaction(a, c, P)

    if s in EXPLICIT:
        A, b = _rk_tableau(s)

        def f2(a, c):
            nu, nv = reaction(a, c, P)
            return D * lap5(a, ops.dx, ops.dy) + nu, nv
        return lambda u, v: _erk_step(f2, (u, v), h, A, b)

    if s == "strang":
        diff = make_diffuser(kind, h, D, ops)

        def step(u, v):
            u, v = _react_substeps(u, v, h / 2, cfg.react_sub, P, cfg.react)
            u = diff(u)
            return _react_substeps(u, v, h / 2, cfg.react_sub, P, cfg.react)
        return step

    if s == "lawson4":
        Dh, Dh2 = make_diffuser(kind, h, D, ops), make_diffuser(kind, h / 2, D, ops)

        def step(u, v):
            k1u, k1v = N(u, v)
            au, av = Dh2(u + h / 2 * k1u), v + h / 2 * k1v
            k2u, k2v = N(au, av)
            Dh2u = Dh2(u)
            bu, bv = Dh2u + h / 2 * k2u, v + h / 2 * k2v
            k3u, k3v = N(bu, bv)
            Dhu = Dh(u)
            cu, cv = Dhu + h * Dh2(k3u), v + h * k3v
            k4u, k4v = N(cu, cv)
            u1 = Dhu + h / 6 * (Dh(k1u) + 2 * Dh2(k2u + k3u) + k4u)
            v1 = v + h / 6 * (k1v + 2 * (k2v + k3v) + k4v)
            return u1, v1
        return step

    fwd, inv = _transforms(kind, ops)
    z = h * D * ops.lhat0

    if s == "imex_euler":
        R = resolvent(h, D, ops)

        def step(u, v):
            nu, nv = N(u, v)
            return inv(R * fwd(u + h * nu)), v + h * nv
        return step

    if s == "etd2":
        p = phi_contour_all(z, (1, 2))
        E, p1, p2 = jnp.exp(z), p[1], p[2]

        def step(u, v):
            n0u, n0v = N(u, v)
            n0h = fwd(n0u)
            ah = E * fwd(u) + h * p1 * n0h
            au, av = inv(ah), v + h * n0v
            nau, nav = N(au, av)
            u1 = inv(ah + h * p2 * (fwd(nau) - n0h))
            v1 = av + h / 2 * (nav - n0v)
            return u1, v1
        return step

    if s == "etdrk4":
        c = etdrk4_coeffs(z)

        def step(u, v):
            uh = fwd(u)
            n0u, n0v = N(u, v)
            n0h = fwd(n0u)
            ah = c.E2 * uh + (h / 2) * c.Q * n0h
            au, av = inv(ah), v + h / 2 * n0v
            nau, nav = N(au, av)
            nah = fwd(nau)
            bh = c.E2 * uh + (h / 2) * c.Q * nah
            bu, bv = inv(bh), v + h / 2 * nav
            nbu, nbv = N(bu, bv)
            nbh = fwd(nbu)
            ch = c.E2 * ah + (h / 2) * c.Q * (2 * nbh - n0h)
            cu, cv = inv(ch), av + h / 2 * (2 * nbv - n0v)
            ncu, ncv = N(cu, cv)
            nch = fwd(ncu)
            u1 = inv(c.E * uh + h * (c.f1 * n0h + 2 * c.f2 * (nah + nbh) + c.f3 * nch))
            v1 = v + h * (n0v / 6 + (nav + nbv) / 3 + ncv / 6)
            return u1, v1
        return step

    raise ValueError(s)


# ----------------------------------------------------------------------------------------------------------------
# driver + factories
# ----------------------------------------------------------------------------------------------------------------
def plan_segments(save_taus, dt):
    """(seg lengths float64, steps per segment int). ValueError unless every segment is an integer number of dt."""
    taus = np.asarray(save_taus, np.float64)
    seg = np.diff(np.concatenate([[0.0], taus]))
    if np.any(seg <= 0):
        raise ValueError("save_taus must be strictly increasing and > 0")
    r = seg / dt
    k = np.rint(r).astype(int)
    if np.any(k < 1) or np.any(np.abs(r - k) >= 1e-4):
        raise ValueError(f"save_taus segments {seg.tolist()} are not integer multiples of dt={dt}")
    return seg, k


def _cast_params(params, dtype):
    return SimpleNamespace(**{k: jnp.asarray(getattr(params, k), dtype) for k in PARAM_NAMES})


def make_forward_fast_p(n, grid, save_taus, cfg):
    """fn(params, u0[..., n, n], v0) -> (U[..., M, n, n], V[..., M, n, n], info{num_steps, ok})."""
    kind = resolve_diff_op(cfg)
    seg, ks = plan_segments(save_taus, cfg.dt)
    hs = seg / ks
    M = len(seg)
    equal = bool(np.all(ks == ks[0]) and np.allclose(hs, hs[0], rtol=1e-5, atol=0.0))
    num_steps = int(ks.sum())

    def forward(params, u0, v0):
        dt_ = u0.dtype
        P = _cast_params(params, dt_)
        ops = make_ops(n, grid, dt_)

        def run_steps(step, k, carry):
            body = jax.checkpoint(lambda c, _: (step(*c), None)) if cfg.remat == "step" else (
                lambda c, _: (step(*c), None))
            return jax.lax.scan(body, carry, None, length=int(k), unroll=cfg.unroll)[0]

        if equal:
            step = build_step(cfg, ops, P, float(hs[0]), kind)

            def seg_fn(c):
                return run_steps(step, ks[0], c)
            if cfg.remat == "segment":
                seg_fn = jax.checkpoint(seg_fn)

            def outer(c, _):
                c = seg_fn(c)
                return c, c
            _, (Us, Vs) = jax.lax.scan(outer, (u0, v0), None, length=M)
        else:
            carry, Ul, Vl = (u0, v0), [], []
            cache = {}
            for h, k in zip(hs, ks):
                key = (round(float(h), 12), int(k))
                if key not in cache:
                    cache[key] = build_step(cfg, ops, P, float(h), kind)
                f = (lambda c, st=cache[key], kk=k: run_steps(st, kk, c))
                if cfg.remat == "segment":
                    f = jax.checkpoint(f)
                carry = f(carry)
                Ul.append(carry[0])
                Vl.append(carry[1])
            Us, Vs = jnp.stack(Ul), jnp.stack(Vl)
        U, V = jnp.moveaxis(Us, 0, -3), jnp.moveaxis(Vs, 0, -3)
        ok = jnp.all(jnp.isfinite(U)) & jnp.all(jnp.isfinite(V))
        return U, V, {"num_steps": num_steps, "ok": ok}

    return forward


def make_forward_fast(n, grid, save_taus, cfg, params):
    """physics.make_forward contract: fn(u0[n,n], v0[n,n]) -> (U[M,n,n], V[M,n,n], info)."""
    fp = make_forward_fast_p(n, grid, save_taus, cfg)
    return lambda u0, v0: fp(params, u0, v0)


def make_forward_fast_batched(n, grid, save_taus, cfg, params):
    """physics.make_forward_batched contract: fn(u0[B,n,n], v0[B,n,n]) -> (U[B,M,n,n], V[B,M,n,n], info).
    Same implementation as make_forward_fast (everything acts on [..., n, n]); `ok` is shared over lanes."""
    return make_forward_fast(n, grid, save_taus, cfg, params)


# ----------------------------------------------------------------------------------------------------------------
# setting names: strang2_d0.1, etd2_d0.05, strang2_d0.1_dct, etdrk4_d0.1_rs2_rematstep_unroll4
# ----------------------------------------------------------------------------------------------------------------
_PREFIXES = ("imex_euler", "strang2", "strang4", "lawson4", "etdrk4", "etd2", "tsit5", "euler", "heun", "rk4")


def parse_setting(name):
    """`<scheme>_d<dt>[_<diff_op>][_rs<k>][_remat<mode>][_unroll<k>]` -> FastCfg (strang2/strang4: react rk2/rk4)."""
    pre = next((p for p in _PREFIXES if name.startswith(p + "_d")), None)
    if pre is None:
        raise ValueError(f"cannot parse setting {name!r}")
    kw = {"scheme": "strang" if pre.startswith("strang") else pre}
    if pre.startswith("strang"):
        kw["react"] = "rk2" if pre == "strang2" else "rk4"
    toks = name[len(pre) + 2:].split("_")
    try:
        kw["dt"] = float(toks[0])
    except ValueError:
        raise ValueError(f"cannot parse dt in setting {name!r}") from None
    for t in toks[1:]:
        if t in ("lap5", "sep", "eig", "dct"):
            kw["diff_op"] = t
        elif re.fullmatch(r"rs\d+", t):
            kw["react_sub"] = int(t[2:])
        elif re.fullmatch(r"remat(step|segment|none)", t):
            kw["remat"] = t[5:]
        elif re.fullmatch(r"unroll\d+", t):
            kw["unroll"] = int(t[6:])
        else:
            raise ValueError(f"cannot parse token {t!r} in setting {name!r}")
    return FastCfg(**kw)


def setting_name(cfg):
    base = {"strang": "strang2" if cfg.react == "rk2" else "strang4"}.get(cfg.scheme, cfg.scheme)
    d = FastCfg()
    s = f"{base}_d{cfg.dt:g}"
    if cfg.diff_op != "auto":
        s += f"_{cfg.diff_op}"
    if cfg.react_sub != d.react_sub:
        s += f"_rs{cfg.react_sub}"
    if cfg.remat != d.remat:
        s += f"_remat{cfg.remat}"
    if cfg.unroll != d.unroll:
        s += f"_unroll{cfg.unroll}"
    return s
