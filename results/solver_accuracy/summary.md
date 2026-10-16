# fastsim accuracy / cost benchmark

Backend: **gpu**; eval dir: `/home/alex/physics-guidance/cardiac/runs/eval_val4`; cases: 4; points: adam_final, truth, warm_seed0, z_init; timing reps: 5.

Point set: truth / z_init / warm_seed0 / adam_final of the eval dir (no tweedie points, no UNet, no A01 forward), so ACCEPTED here is not identical to the acceptance of cardiac.solver_sweep.

ACCEPTED (thresholds as in solver_sweep, at all points and cases): max-over-tau U RMSE <= 0.002, V RMSE <= 0.005, gradient cosine >= 0.99 and rel. L2 error <= 0.1, no failures; errors vs REF (diffrax tsit5, rtol=atol=1e-6, dtmax 0.0025). Speedup = F02 time / setting time, from this run on this backend (median over cases at the warm_seed0 point).

| setting | scheme | dt | steps | RHS/step | status | fails/rows | max U RMSE | max V RMSE | min grad cos | max grad rel | ms/fwd | ms/VJP | VJP/fwd | VJP speedup vs F02 | temp MB | compile s | accepted |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REF | diffrax | - | 4001 | - | ok | 0/16 | 0 | 0 | 1.00000 | 0 | - | - | - | - | 2358 | 3.0 | no |
| F02 | diffrax | - | 500 | - | ok | 0/16 | 3.698e-07 | 1.057e-06 | 1.00000 | 0.0001309 | 110.6 | 486.0 | 4.39 | 1.00x | 2358 | 3.0 | yes |
| euler_d0.01 | euler | 0.01 | 1000 | 1 | ok | 0/16 | 0.003857 | 0.004883 | 0.91158 | 0.4118 | 8.3 | 39.6 | 4.75 | 12.28x | 425 | 0.4 | no |
| euler_d0.02 | euler | 0.02 | 500 | 1 | ok | 11/16 | 0.2088 | 0.2252 | 0.00030 | 2715 | 7.8 | 19.6 | 2.52 | 24.79x | 214 | 0.4 | no |
| heun_d0.01 | heun | 0.01 | 1000 | 2 | ok | 0/16 | 8.596e-06 | 7.927e-06 | 1.00000 | 0.00115 | 8.1 | 61.4 | 7.60 | 7.91x | 425 | 0.8 | yes |
| heun_d0.02 | heun | 0.02 | 500 | 2 | ok | 7/16 | 0.2229 | 0.2399 | 0.00012 | 2876 | 4.1 | 29.8 | 7.34 | 16.28x | 215 | 0.6 | no |
| rk4_d0.02 | rk4 | 0.02 | 500 | 4 | ok | 0/16 | 2.18e-06 | 4.192e-06 | 1.00000 | 0.00086 | 8.5 | 55.5 | 6.54 | 8.75x | 216 | 0.9 | yes |
| rk4_d0.025 | rk4 | 0.025 | 400 | 4 | ok | 0/16 | 9.427e-06 | 1.801e-05 | 0.99999 | 0.003859 | 6.7 | 44.4 | 6.64 | 10.95x | 174 | 0.9 | yes |
| tsit5_d0.02 | tsit5 | 0.02 | 500 | 6 | ok | 0/16 | 3.758e-07 | 1.057e-06 | 1.00000 | 0.0001301 | 13.4 | 83.9 | 6.24 | 5.80x | 217 | 1.6 | yes |
| strang2_d0.02 | strang | 0.02 | 500 | 4 | ok | 0/16 | 0.0001122 | 0.0001356 | 0.99981 | 0.02063 | 10.0 | 33.3 | 3.34 | 14.57x | 214 | 0.6 | yes |
| strang2_d0.05 | strang | 0.05 | 200 | 4 | ok | 0/16 | 0.0006789 | 0.0007321 | 0.99312 | 0.1214 | 3.9 | 13.2 | 3.36 | 36.79x | 88 | 0.5 | no |
| strang2_d0.1 | strang | 0.1 | 100 | 4 | ok | 0/16 | 0.002678 | 0.002317 | 0.87611 | 0.4855 | 1.9 | 6.9 | 3.57 | 70.15x | 46 | 0.5 | no |
| strang2_d0.2 | strang | 0.2 | 50 | 4 | ok | 0/16 | 0.01068 | 0.006844 | 0.14003 | 1.972 | 1.0 | 3.3 | 3.13 | 148.99x | 25 | 0.8 | no |
| strang4_d0.05 | strang | 0.05 | 200 | 8 | ok | 0/16 | 0.0003166 | 0.0006638 | 0.99925 | 0.0412 | 3.9 | 14.7 | 3.77 | 33.03x | 88 | 0.9 | yes |
| strang4_d0.1 | strang | 0.1 | 100 | 8 | ok | 0/16 | 0.001207 | 0.001979 | 0.98656 | 0.1675 | 2.0 | 7.4 | 3.69 | 65.37x | 46 | 0.9 | no |
| strang4_d0.2 | strang | 0.2 | 50 | 8 | ok | 0/16 | 0.00478 | 0.005127 | 0.73185 | 0.7287 | 1.1 | 3.7 | 3.23 | 131.31x | 25 | 0.9 | no |
| imex_euler_d0.02 | imex_euler | 0.02 | 500 | 1 | ok | 0/16 | 0.01438 | 0.00761 | -0.02224 | 2.821 | 16.4 | 38.5 | 2.35 | 12.64x | 213 | 0.4 | no |
| imex_euler_d0.05 | imex_euler | 0.05 | 200 | 1 | ok | 0/16 | 0.03491 | 0.01851 | -0.23767 | 6.759 | 6.5 | 15.4 | 2.35 | 31.64x | 87 | 0.4 | no |
| etd2_d0.02 | etd2 | 0.02 | 500 | 2 | ok | 0/16 | 0.0002175 | 0.0001102 | 0.99926 | 0.0407 | 35.1 | 88.5 | 2.52 | 5.49x | 215 | 0.9 | yes |
| etd2_d0.05 | etd2 | 0.05 | 200 | 2 | ok | 0/16 | 0.001256 | 0.0005515 | 0.97360 | 0.2309 | 14.2 | 35.5 | 2.50 | 13.69x | 89 | 0.6 | no |
| etd2_d0.1 | etd2 | 0.1 | 100 | 2 | ok | 0/16 | 0.004669 | 0.002035 | 0.64809 | 0.8606 | 7.4 | 18.1 | 2.44 | 26.82x | 47 | 0.6 | no |
| etd2_d0.2 | etd2 | 0.2 | 50 | 2 | ok | 0/16 | 0.01642 | 0.00742 | -0.04325 | 3.009 | 3.4 | 9.4 | 2.75 | 51.59x | 26 | 0.6 | no |
| etdrk4_d0.05 | etdrk4 | 0.05 | 200 | 4 | ok | 0/16 | 6.738e-06 | 1.271e-05 | 1.00000 | 0.0008967 | 26.8 | 74.6 | 2.78 | 6.52x | 90 | 0.9 | yes |
| etdrk4_d0.1 | etdrk4 | 0.1 | 100 | 4 | ok | 0/16 | 1.717e-05 | 7.27e-05 | 1.00000 | 0.002399 | 13.9 | 37.5 | 2.70 | 12.97x | 48 | 0.9 | yes |
| etdrk4_d0.2 | etdrk4 | 0.2 | 50 | 4 | ok | 0/16 | 0.0002071 | 0.0003146 | 0.99971 | 0.02539 | 7.0 | 18.9 | 2.68 | 25.76x | 27 | 0.9 | yes |
| lawson4_d0.05 | lawson4 | 0.05 | 200 | 4 | ok | 0/16 | 3.285e-05 | 7.563e-05 | 1.00000 | 0.002725 | 13.8 | 39.6 | 2.87 | 12.27x | 90 | 0.6 | yes |
| lawson4_d0.1 | lawson4 | 0.1 | 100 | 4 | ok | 0/16 | 0.0001191 | 0.0004051 | 0.99999 | 0.005093 | 6.8 | 20.1 | 2.96 | 24.20x | 47 | 0.6 | yes |
| lawson4_d0.2 | lawson4 | 0.2 | 50 | 4 | ok | 0/16 | 0.0005175 | 0.001529 | 0.99996 | 0.01064 | 3.4 | 10.1 | 3.00 | 47.93x | 26 | 0.7 | yes |

## Best accepted setting per family (lowest VJP time)

| family | setting | dt | ms/VJP | VJP/fwd | VJP speedup vs F02 | fwd speedup vs F02 |
|---|---|---|---|---|---|---|
| lawson4 | lawson4_d0.2 | 0.2 | 10.1 | 3.00 | 47.93x | 32.74x |
| strang4 | strang4_d0.05 | 0.05 | 14.7 | 3.77 | 33.03x | 28.34x |
| etdrk4 | etdrk4_d0.2 | 0.2 | 18.9 | 2.68 | 25.76x | 15.71x |
| strang2 | strang2_d0.02 | 0.02 | 33.3 | 3.34 | 14.57x | 11.09x |
| rk4 | rk4_d0.025 | 0.025 | 44.4 | 6.64 | 10.95x | 16.57x |
| heun | heun_d0.01 | 0.01 | 61.4 | 7.60 | 7.91x | 13.68x |
| tsit5 | tsit5_d0.02 | 0.02 | 83.9 | 6.24 | 5.80x | 8.23x |
| etd2 | etd2_d0.02 | 0.02 | 88.5 | 2.52 | 5.49x | 3.16x |
| diffrax | F02 | - | 486.0 | 4.39 | 1.00x | 1.00x |

Best accepted setting overall: **lawson4_d0.2** (47.93x faster per VJP than F02 on gpu).
