"""Tests of cardiac.fastsim_fit_compare. CPU only (JAX_PLATFORMS=cpu), not marked slow.

The end-to-end test runs the real pipeline on the real eval dir with tiny settings (1 case, 3 Adam iterations, observation
taus 1..2, rollout to tau 1); it is skipped if the eval dir, its data, the simulator or ffmpeg is missing.
"""
import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")   # before the first jax call: these tests never touch the GPU
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest

from cardiac import fastsim_fit_compare as fc


def _eval_dir():
    for p in (fc.DEFAULT_EVAL, fc.FALLBACK_EVAL):
        if os.path.isdir(p):
            return p
    return None


def test_gpu_guard_refuses_without_force(monkeypatch, capsys):
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    with pytest.raises(SystemExit) as e:  # this very process stands in for the training job's PID
        fc.main(["--train-pid", str(os.getpid()), "--no-wandb", "--out", "/nonexistent/never_created"])
    assert e.value.code == 2
    assert "REFUSING" in capsys.readouterr().err
    assert not os.path.exists("/nonexistent/never_created")


def test_gpu_guard_wiring_cpu_and_force_allowed(monkeypatch):
    a = fc.make_parser().parse_args(["--train-pid", str(os.getpid())])
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    fc.guard(a)   # CPU-only is always allowed
    monkeypatch.delenv("JAX_PLATFORMS")
    a.force = True
    fc.guard(a)   # --force allowed
    a.force = False
    with pytest.raises(SystemExit):
        fc.guard(a)


def test_colours_fixed_and_deterministic():
    assert fc.color_for("F02") == "#1f77b4" and fc.color_for("truth") == "#000000"
    assert fc.color_for("lawson4_d0.2_unroll4") == fc.color_for("lawson4_d0.2") == "#d62728"
    assert fc.color_for("lawson4_d0.1") == "#ff7f0e"
    assert fc.color_for("etdrk4_d0.1") == fc.color_for("etdrk4_d0.1") in fc.PALETTE
    assert fc.linestyle_for("F02") == "-" and fc.linestyle_for("lawson4_d0.2") == "--"


def test_meta_mismatch_refused(tmp_path):
    a = fc.make_parser().parse_args(["--out", str(tmp_path)])
    meta = {"cases": ["c"], "tmax": 30.0, "device": "cpu", "eval": "x", "volatile": ["eval", "device"]}
    fc.check_meta(a, meta)
    fc.check_meta(a, {**meta, "device": "gpu", "eval": "y"})   # volatile keys may differ
    with pytest.raises(SystemExit):
        fc.check_meta(a, {**meta, "tmax": 2.0})
    a.redo = True
    fc.check_meta(a, {**meta, "tmax": 2.0})
    assert json.load(open(tmp_path / "meta.json"))["tmax"] == 2.0


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    ev = _eval_dir()
    if ev is None:
        pytest.skip("eval_test_n20 not found")
    pytest.importorskip("jax")
    try:
        import cardiac.physics  # noqa: F401  (imports ap_simulator via cardiac._compat)
    except Exception:
        pytest.skip("ap_simulator not importable")
    import shutil
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg missing")
    from cardiac.final_figs import load_eval
    _, _, ra = load_eval(ev, headline=True)
    if not os.path.exists(os.path.join(ra["data_root"], "config.json")) or not os.path.exists("cardiac/artifacts/split.json"):
        pytest.skip("data root / split missing")
    root = tmp_path_factory.mktemp("fit_compare")
    out, media = str(root / "out"), str(root / "media")
    argv = ["--eval", ev, "--cases", "1", "--video-cases", "1", "--iters", "3", "--tmax", "1", "--dt", "0.5",
            "--taus-max", "2", "--no-wandb", "--out", out, "--media-dir", media]
    fc.main(argv)
    return argv, out, media


def _names(out):
    meta = json.load(open(os.path.join(out, "meta.json")))
    return meta, meta["cases"][0]


def test_end_to_end_outputs(tiny_run):
    argv, out, media = tiny_run
    meta, nm = _names(out)
    for f in (f"fit_f02_{nm}.npz", f"fit_fast_{nm}.npz", f"roll_{nm}.npz", f"frames_{nm}.npz", "summary.md", "summary.json",
              "fit_compare.png", "fit_compare.pdf", f"videos/fit_compare_{nm}.mp4", "videos/fit_compare_montage.mp4"):
        assert os.path.getsize(os.path.join(out, f)) > 0, f
    for key in ("f02", "fast"):
        z = fc.load_npz(os.path.join(out, f"fit_{key}_{nm}.npz"))
        assert z["n_vjp"] == 3 and "n_failed" in z and z["n_failed"] == 0
        assert len(z["loss_hist"]) == 3 and np.isfinite(z["loss_hist"]).all()
        assert np.isfinite([z["rmse_u0"], z["rmse_v0"], z["final_L_obs"], z["wall_time_s"]]).all()
    assert "diff_stored_u0" in fc.load_npz(os.path.join(out, f"fit_f02_{nm}.npz"))
    r = fc.load_npz(os.path.join(out, f"roll_{nm}.npz"))
    assert np.allclose(r["taus"], [0.0, 0.5, 1.0])
    for k in ("rmse_u_f02", "rmse_v_f02", "rmse_u_fast", "rmse_v_fast"):
        assert r[k].shape == (3,) and np.isfinite(r[k]).all()
    fr = fc.load_frames(os.path.join(out, f"frames_{nm}.npz"))
    assert fr["truth"][0].shape[0] == 3 and fr["truth"][0].shape[1:] == (meta["n"], meta["n"])
    js = json.load(open(os.path.join(out, "summary.json")))
    assert nm in js["cases"] and js["cases"][nm]["speedup"] > 0
    md = open(os.path.join(out, "summary.md")).read()
    assert "speedup" in md and "mean" in md and "excludes compile" in md
    for f in ("fit_compare_figure.png", "fit_compare_summary.md", "fit_compare_montage.mp4", f"fit_compare_{nm}.mp4"):
        assert os.path.exists(os.path.join(media, f)), f


def test_rerun_resumes_and_plot_only_regenerates_frames(tiny_run):
    argv, out, media = tiny_run
    meta, nm = _names(out)
    fits = [os.path.join(out, f"fit_{k}_{nm}.npz") for k in ("f02", "fast")]
    roll = os.path.join(out, f"roll_{nm}.npz")
    m0 = [os.stat(p).st_mtime_ns for p in fits + [roll]]
    fc.main(argv)   # resume: nothing is refitted or re-rolled
    assert [os.stat(p).st_mtime_ns for p in fits + [roll]] == m0
    # drop the frames: --plot-only recomputes the rollouts from the cached IC pairs, never refits
    frames, video = os.path.join(out, f"frames_{nm}.npz"), os.path.join(out, "videos", f"fit_compare_{nm}.mp4")
    os.remove(frames)
    os.remove(video)
    fc.main(["--plot-only", "--eval", argv[1], "--video-cases", "1", "--no-wandb", "--out", out, "--media-dir", media])
    assert os.path.getsize(frames) > 0 and os.path.getsize(video) > 0
    assert [os.stat(p).st_mtime_ns for p in fits + [roll]] == m0
