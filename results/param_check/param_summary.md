# fastsim parameter-gradient / parameter-recovery check

Eval dir: `/home/alex/physics-guidance/cardiac/runs/eval_test_n20`; data n=224, dx=0.0089686; true theta: D=0.001, eps0=0.05, a=0.09, k=8.2, mu1=0.07, mu2=0.3.

**D_max_F02 = 3.5 dx^2 / (8 * 0.02) = 1.7595e-03** (Tsit5 real-axis stability bound 3.5 for the 5-point Laplacian, eigenvalues up to 8D/dx^2, fixed dt 0.02; true D = 0.001, i.e. 1.76x truth). Perturbed starts (factor exp(+-U(log1.2, log1.4)) per param, seeds below): seed 0: D=7.994e-04 (x0.799); seed 1: D=7.198e-04 (x0.720); seed 2: D=1.257e-03 (x1.257). All starts are below D_max_F02 (asserted).

## B1: parameter-gradient accuracy vs the reference solver

Reference: **REF**. One VJP of the obs loss wrt (u0, v0, theta) with traced theta at three points per case: truth (true state, true theta), theta_pert (true state, seed-0 perturbed theta), joint (stored Adam state, seed-0 perturbed theta). Errors are in log-theta scale (theta_k * g_k) so the six entries are comparable. nerr = |g_k - g_ref_k| / ||g_ref|| (median / max over cases x points). Relative errors and sign agreement only where |g_ref_k| >= 0.001 ||g_ref|| (otherwise n/a; at the truth point the reference gradient is noise driven).

### Normalised error nerr, median / max

| setting | status | D | eps0 | a | k | mu1 | mu2 |
|---|---|---|---|---|---|---|---|
| F02 | ok | 2.4e-07 / 3.1e-04 | 7.9e-08 / 3.0e-04 | 3.6e-07 / 1.1e-03 | 1.1e-06 / 1.9e-03 | 5.7e-08 / 2.1e-04 | 2.9e-08 / 3.8e-05 |
| tsit5_d0.02 | ok | 2.3e-07 / 1.2e-03 | 4.7e-08 / 1.5e-04 | 2.6e-07 / 6.0e-04 | 1.1e-06 / 1.7e-03 | 4.6e-08 / 1.5e-04 | 2.8e-08 / 1.1e-04 |
| rk4_d0.025 | ok | 1.7e-06 / 1.2e-03 | 1.5e-07 / 3.6e-04 | 8.5e-07 / 2.0e-03 | 1.1e-06 / 3.7e-03 | 4.4e-07 / 2.0e-04 | 5.7e-07 / 5.2e-05 |
| lawson4_d0.05 | ok | 1.1e-06 / 9.1e-02 | 1.9e-07 / 1.8e-02 | 1.4e-06 / 1.5e-01 | 2.5e-06 / 2.7e-01 | 3.7e-07 / 2.1e-02 | 6.9e-07 / 5.0e-03 |
| lawson4_d0.1 | ok | 5.2e-06 / 3.7e-01 | 7.0e-07 / 6.4e-02 | 4.7e-06 / 5.5e-01 | 6.7e-06 / 1.1e+00 | 2.9e-06 / 5.8e-02 | 4.1e-06 / 1.3e-02 |
| lawson4_d0.2 | ok | 1.5e-05 / 3.5e+00 | 9.6e-06 / 4.5e-01 | 4.4e-05 / 4.2e+00 | 1.0e-04 / 9.3e+00 | 2.0e-05 / 1.8e-01 | 2.1e-05 / 1.8e-01 |
| lawson4_d0.2_unroll4 | ok | 1.5e-05 / 3.5e+00 | 9.6e-06 / 4.5e-01 | 4.4e-05 / 4.2e+00 | 1.0e-04 / 9.3e+00 | 2.0e-05 / 1.8e-01 | 2.1e-05 / 1.8e-01 |
| etdrk4_d0.1 | ok | 5.0e-06 / 6.0e-01 | 5.9e-07 / 6.2e-02 | 4.3e-06 / 6.9e-01 | 1.1e-05 / 1.5e+00 | 2.6e-06 / 3.6e-02 | 3.9e-06 / 2.8e-02 |
| etdrk4_d0.2 | ok | 1.7e-05 / 7.1e+00 | 6.0e-06 / 6.5e-01 | 3.4e-05 / 7.6e+00 | 1.6e-04 / 1.7e+01 | 1.5e-05 / 2.9e-01 | 1.8e-05 / 3.9e-01 |
| strang4_d0.05 | ok | 2.6e-05 / 1.0e+01 | 4.2e-05 / 7.9e-01 | 1.1e-04 / 1.1e+01 | 3.1e-04 / 2.3e+01 | 5.2e-05 / 1.4e-01 | 3.4e-05 / 5.7e-01 |

### Relative error (median, n valid) and sign agreement %

| setting | D | eps0 | a | k | mu1 | mu2 |
|---|---|---|---|---|---|---|
| F02 | 1.3e-06 (n=12) / 100% | 2.1e-06 (n=12) / 100% | 1.2e-06 (n=12) / 100% | 1.1e-06 (n=12) / 100% | 1.3e-05 (n=12) / 100% | 2.7e-06 (n=11) / 100% |
| tsit5_d0.02 | 1.4e-06 (n=12) / 100% | 1.3e-06 (n=12) / 100% | 8.5e-07 (n=12) / 100% | 1.2e-06 (n=12) / 100% | 1.1e-05 (n=12) / 100% | 2.5e-06 (n=11) / 100% |
| rk4_d0.025 | 1.1e-05 (n=12) / 100% | 3.9e-06 (n=12) / 100% | 2.9e-06 (n=12) / 100% | 1.2e-06 (n=12) / 100% | 2.7e-05 (n=12) / 100% | 8.9e-05 (n=11) / 100% |
| lawson4_d0.05 | 5.6e-06 (n=12) / 100% | 5.7e-06 (n=12) / 100% | 4.6e-06 (n=12) / 92% | 2.7e-06 (n=12) / 100% | 4.5e-05 (n=12) / 100% | 9.9e-05 (n=11) / 100% |
| lawson4_d0.1 | 3.1e-05 (n=12) / 100% | 1.9e-05 (n=12) / 100% | 1.6e-05 (n=12) / 92% | 7.1e-06 (n=12) / 92% | 1.9e-04 (n=12) / 100% | 6.0e-04 (n=11) / 100% |
| lawson4_d0.2 | 9.2e-05 (n=12) / 75% | 2.4e-04 (n=12) / 92% | 1.5e-04 (n=12) / 83% | 1.1e-04 (n=12) / 75% | 8.3e-04 (n=12) / 100% | 2.9e-03 (n=11) / 100% |
| lawson4_d0.2_unroll4 | 9.2e-05 (n=12) / 75% | 2.4e-04 (n=12) / 92% | 1.5e-04 (n=12) / 83% | 1.1e-04 (n=12) / 75% | 8.3e-04 (n=12) / 100% | 2.9e-03 (n=11) / 100% |
| etdrk4_d0.1 | 3.1e-05 (n=12) / 100% | 1.5e-05 (n=12) / 100% | 1.4e-05 (n=12) / 92% | 1.2e-05 (n=12) / 83% | 1.9e-04 (n=12) / 100% | 5.7e-04 (n=11) / 100% |
| etdrk4_d0.2 | 1.0e-04 (n=12) / 75% | 1.4e-04 (n=12) / 92% | 1.1e-04 (n=12) / 83% | 1.7e-04 (n=12) / 75% | 8.9e-04 (n=12) / 92% | 2.5e-03 (n=11) / 91% |
| strang4_d0.05 | 1.2e-04 (n=12) / 75% | 1.0e-03 (n=12) / 92% | 3.6e-04 (n=12) / 83% | 3.3e-04 (n=12) / 75% | 3.3e-03 (n=12) / 100% | 4.6e-03 (n=11) / 82% |

### Cosines, state block, forward error and cost

| setting | status | min cos (6-vec) | min cos (state block) | max rel L2 (state block) | max U/V RMSE vs ref | ms theta-VJP | ms joint VJP | joint-VJP speedup vs F02 | compile s | ok points |
|---|---|---|---|---|---|---|---|---|---|---|
| F02 | ok | 0.999999 | 1.000000 | 1.0e-03 | 3.2e-07/1.1e-06 | 562.3 | 564.0 | 1.00x | 5.5 | 12/12 |
| tsit5_d0.02 | ok | 1.000000 | 0.999999 | 1.7e-03 | 3.2e-07/1.1e-06 | 142.8 | 142.4 | 3.96x | 3.7 | 12/12 |
| rk4_d0.025 | ok | 0.999998 | 0.999999 | 1.3e-03 | 1.4e-06/1.5e-06 | 75.2 | 74.7 | 7.55x | 2.4 | 12/12 |
| lawson4_d0.05 | ok | 0.976338 | 0.998903 | 5.2e-02 | 3.8e-06/5.3e-06 | 75.1 | 75.3 | 7.49x | 1.6 | 12/12 |
| lawson4_d0.1 | ok | -0.167296 | 0.990891 | 1.4e-01 | 1.3e-05/2.8e-05 | 38.5 | 38.0 | 14.86x | 1.5 | 12/12 |
| lawson4_d0.2 | ok | -0.888835 | 0.558141 | 9.6e-01 | 1.2e-04/1.3e-04 | 19.2 | 20.1 | 28.11x | 1.7 | 12/12 |
| lawson4_d0.2_unroll4 | ok | -0.888835 | 0.558141 | 9.6e-01 | 1.2e-04/1.3e-04 | 18.0 | 16.8 | 33.54x | 3.3 | 12/12 |
| etdrk4_d0.1 | ok | -0.635970 | 0.990545 | 1.4e-01 | 1.6e-05/1.7e-05 | 45.2 | 45.0 | 12.52x | 2.2 | 12/12 |
| etdrk4_d0.2 | ok | -0.912907 | 0.276116 | 1.6e+00 | 1.9e-04/8.6e-05 | 23.4 | 23.1 | 24.39x | 1.8 | 12/12 |
| strang4_d0.05 | ok | -0.917020 | -0.003374 | 2.4e+00 | 2.8e-04/1.7e-04 | 38.1 | 37.3 | 15.14x | 2.9 | 12/12 |

### 1 - cosine of the 6-vector, max over cases, by point

| setting | truth | theta_pert | joint |
|---|---|---|---|
| F02 | 9.3e-07 | 1.6e-13 | 3.7e-13 |
| tsit5_d0.02 | 2.6e-07 | 1.2e-13 | 3.8e-13 |
| rk4_d0.025 | 2.3e-06 | 1.2e-13 | 6.8e-12 |
| lawson4_d0.05 | 2.4e-02 | 1.1e-12 | 3.7e-12 |
| lawson4_d0.1 | 1.2e+00 | 1.8e-12 | 6.1e-11 |
| lawson4_d0.2 | 1.9e+00 | 5.2e-10 | 1.5e-09 |
| lawson4_d0.2_unroll4 | 1.9e+00 | 5.2e-10 | 1.5e-09 |
| etdrk4_d0.1 | 1.6e+00 | 1.3e-11 | 6.2e-11 |
| etdrk4_d0.2 | 1.9e+00 | 2.9e-09 | 1.5e-09 |
| strang4_d0.05 | 1.9e+00 | 6.9e-09 | 6.1e-09 |

## B2: parameter recovery (true u0, v0 known; BoundedLogRDParams direct mapping, decayed Adam, same VJP count)

Final relative error |theta - theta*| / theta* in %, mean +- std over case x seed fits (an unstable fit contributes its last finite params). instability = non-finite loss/gradient or solver not ok (fit stopped); not_converged = a finished fit with some param above 10% relative error (counted among the finished fits only); at_bound = a param pinned at its (widened) bound at the end (zero gradient there).

| method | fits | D | eps0 | a | k | mu1 | mu2 | mean |rel err| % | instability | not_converged | at_bound | wall s (excl. compile) | compile s | speedup vs F02 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F02 | 6 | 3.5 +- 0.8 | 18.9 +- 7.8 | 9.8 +- 4.2 | 2.7 +- 1.4 | 22.2 +- 13.3 | 15.3 +- 4.4 | 12.1 | 0/6 | 6/6 | none | 117.6 | 2.8 | 1.0x |
| lawson4_d0.1 | 6 | 3.5 +- 0.8 | 18.9 +- 7.8 | 9.8 +- 4.2 | 2.7 +- 1.4 | 22.2 +- 13.3 | 15.3 +- 4.4 | 12.1 | 0/6 | 6/6 | none | 8.4 | 0.9 | 14.0x |
| lawson4_d0.2 | 6 | 3.5 +- 0.8 | 18.9 +- 7.8 | 9.8 +- 4.2 | 2.7 +- 1.4 | 22.2 +- 13.3 | 15.2 +- 4.4 | 12.1 | 0/6 | 6/6 | none | 4.6 | 0.9 | 25.5x |

### Failures and not-converged params per fit

| method | case | seed | mean |rel err| % | outcome |
|---|---|---|---|---|
| F02 | system_0109_f128 | 0 | 11.6 | not converged: eps0, a, mu2 |
| F02 | system_0109_f128 | 1 | 11.4 | not converged: eps0, mu1, mu2 |
| F02 | system_0109_f128 | 2 | 12.9 | not converged: a, mu1 |
| F02 | system_0119_f128 | 0 | 12.1 | not converged: eps0, a, mu2 |
| F02 | system_0119_f128 | 1 | 11.3 | not converged: eps0, mu1, mu2 |
| F02 | system_0119_f128 | 2 | 13.2 | not converged: eps0, a, mu1, mu2 |
| lawson4_d0.1 | system_0109_f128 | 0 | 11.6 | not converged: eps0, a, mu2 |
| lawson4_d0.1 | system_0109_f128 | 1 | 11.4 | not converged: eps0, mu1, mu2 |
| lawson4_d0.1 | system_0109_f128 | 2 | 12.9 | not converged: a, mu1 |
| lawson4_d0.1 | system_0119_f128 | 0 | 12.1 | not converged: eps0, a, mu2 |
| lawson4_d0.1 | system_0119_f128 | 1 | 11.3 | not converged: eps0, mu1, mu2 |
| lawson4_d0.1 | system_0119_f128 | 2 | 13.2 | not converged: eps0, a, mu1, mu2 |
| lawson4_d0.2 | system_0109_f128 | 0 | 11.6 | not converged: eps0, a, mu2 |
| lawson4_d0.2 | system_0109_f128 | 1 | 11.4 | not converged: eps0, mu1, mu2 |
| lawson4_d0.2 | system_0109_f128 | 2 | 12.8 | not converged: a, mu1 |
| lawson4_d0.2 | system_0119_f128 | 0 | 12.1 | not converged: eps0, a, mu2 |
| lawson4_d0.2 | system_0119_f128 | 1 | 11.3 | not converged: eps0, mu1, mu2 |
| lawson4_d0.2 | system_0119_f128 | 2 | 13.2 | not converged: eps0, a, mu1, mu2 |

### Final loss

| method | mean final loss | median final loss | median ratio to F02 (same case/seed) |
|---|---|---|---|
| F02 | 3.84e+05 | 3.53e+05 | 1.000 |
| lawson4_d0.1 | 3.84e+05 | 3.53e+05 | 1.000 |
| lawson4_d0.2 | 3.84e+05 | 3.53e+05 | 1.001 |

