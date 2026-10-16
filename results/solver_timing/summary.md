# fastsim timing

Backend: **gpu**; reps: 5; wall budget: None min (None = no cap).

Point set: truth / z_init / warm_seed0 / adam_final of the eval dir (no tweedie points, no UNet, no A01 forward), so ACCEPTED here is not identical to the acceptance of cardiac.solver_sweep.

Lanes tile the cases' warm_seed0 points; VJP = forward + reverse of the summed gaussian obs loss wrt (u0, v0). Speedup = F02 VJP time / config VJP time at the same B (F02 at B>1 = physics.make_forward_batched).

| config | status | steps | compile s | ms/fwd | ms/VJP | VJP/fwd | ms/case (VJP) | peak MB | peak src | speedup vs F02 | lane-0 diff |
|---|---|---|---|---|---|---|---|---|---|---|---|
| F02|B1 | ok | 500 | 3.0 | 112.0 | 488.3 | 4.36 | 488.3 | 2368 | peak_bytes_in_use | 1.00x | - |
| tsit5_d0.02|step|B1 | ok | 500 | 1.5 | 14.5 | 83.5 | 5.75 | 83.5 | 267 | peak_bytes_in_use | 5.85x | - |
| lawson4_d0.2|step|B1 | ok | 50 | 0.8 | 3.3 | 10.6 | 3.26 | 10.6 | 146 | peak_bytes_in_use | 46.02x | - |
| strang4_d0.05|step|B1 | ok | 200 | 1.0 | 4.4 | 15.0 | 3.37 | 15.0 | 139 | peak_bytes_in_use | 32.65x | - |
| etdrk4_d0.2|step|B1 | ok | 50 | 1.1 | 6.8 | 19.6 | 2.88 | 19.6 | 146 | peak_bytes_in_use | 24.94x | - |
| strang2_d0.02|step|B1 | ok | 500 | 0.7 | 9.4 | 33.6 | 3.59 | 33.6 | 267 | peak_bytes_in_use | 14.54x | - |
| rk4_d0.025|step|B1 | ok | 400 | 1.0 | 7.2 | 44.7 | 6.18 | 44.7 | 267 | peak_bytes_in_use | 10.93x | - |
| heun_d0.01|step|B1 | ok | 1000 | 0.7 | 8.1 | 61.6 | 7.57 | 61.6 | 523 | peak_bytes_in_use | 7.93x | - |
| strang2_d0.1|step|B1 | ok | 100 | 0.7 | 1.9 | 6.4 | 3.34 | 6.4 | 130 | peak_bytes_in_use | 75.93x | - |
| strang2_d0.1_eig|step|B1 | ok | 100 | 0.7 | 3.8 | 11.4 | 2.95 | 11.4 | 130 | peak_bytes_in_use | 43.02x | - |
| strang2_d0.1_dct|step|B1 | ok | 100 | 1.1 | 5.5 | 17.5 | 3.19 | 17.5 | 75 | peak_bytes_in_use | 27.84x | - |
| etd2_d0.1|step|B1 | ok | 100 | 0.7 | 7.3 | 17.9 | 2.45 | 17.9 | 130 | peak_bytes_in_use | 27.31x | - |
| etd2_d0.1_dct|step|B1 | ok | 100 | 1.5 | 11.2 | 35.0 | 3.11 | 35.0 | 75 | peak_bytes_in_use | 13.96x | - |
| tsit5_d0.02|none|B1 | ok | 500 | 2.3 | 13.2 | 231.1 | 17.53 | 231.1 | 8226 | peak_bytes_in_use | 2.11x | - |
| tsit5_d0.02|segment|B1 | ok | 500 | 1.9 | 13.7 | 183.0 | 13.33 | 183.0 | 767 | peak_bytes_in_use | 2.67x | - |
| lawson4_d0.2|none|B1 | ok | 50 | 1.3 | 3.7 | 18.8 | 5.01 | 18.8 | 562 | peak_bytes_in_use | 26.01x | - |
| lawson4_d0.2|segment|B1 | ok | 50 | 1.1 | 3.7 | 17.5 | 4.72 | 17.5 | 130 | peak_bytes_in_use | 27.92x | - |
| strang4_d0.05|none|B1 | ok | 200 | 2.3 | 4.6 | 97.0 | 20.95 | 97.0 | 4393 | peak_bytes_in_use | 5.04x | - |
| strang4_d0.05|segment|B1 | ok | 200 | 1.7 | 4.5 | 66.5 | 14.92 | 66.5 | 523 | peak_bytes_in_use | 7.34x | - |
| etdrk4_d0.2|none|B1 | ok | 50 | 1.5 | 7.6 | 26.6 | 3.49 | 26.6 | 563 | peak_bytes_in_use | 18.36x | - |
| etdrk4_d0.2|segment|B1 | ok | 50 | 1.4 | 7.4 | 28.1 | 3.77 | 28.1 | 146 | peak_bytes_in_use | 17.38x | - |
| strang2_d0.02|none|B1 | ok | 500 | 1.1 | 10.0 | 128.1 | 12.80 | 128.1 | 5488 | peak_bytes_in_use | 3.81x | - |
| strang2_d0.02|segment|B1 | ok | 500 | 0.9 | 9.6 | 95.6 | 10.00 | 95.6 | 523 | peak_bytes_in_use | 5.11x | - |
| rk4_d0.025|none|B1 | ok | 400 | 1.4 | 7.4 | 121.5 | 16.52 | 121.5 | 4394 | peak_bytes_in_use | 4.02x | - |
| rk4_d0.025|segment|B1 | ok | 400 | 1.3 | 7.2 | 95.7 | 13.30 | 95.7 | 523 | peak_bytes_in_use | 5.11x | - |
| heun_d0.01|none|B1 | ok | 1000 | 0.9 | 8.2 | 152.9 | 18.57 | 152.9 | 5488 | peak_bytes_in_use | 3.19x | - |
| heun_d0.01|segment|B1 | ok | 1000 | 0.8 | 8.0 | 116.6 | 14.60 | 116.6 | 523 | peak_bytes_in_use | 4.19x | - |
| strang2_d0.1|none|B1 | ok | 100 | 1.2 | 2.0 | 26.2 | 13.04 | 26.2 | 1108 | peak_bytes_in_use | 18.63x | - |
| strang2_d0.1|segment|B1 | ok | 100 | 0.9 | 2.1 | 18.0 | 8.78 | 18.0 | 139 | peak_bytes_in_use | 27.07x | - |
| strang2_d0.1_eig|none|B1 | ok | 100 | 1.1 | 3.5 | 29.1 | 8.31 | 29.1 | 1108 | peak_bytes_in_use | 16.79x | - |
| strang2_d0.1_eig|segment|B1 | ok | 100 | 0.9 | 3.6 | 24.2 | 6.67 | 24.2 | 139 | peak_bytes_in_use | 20.16x | - |
| strang2_d0.1_dct|none|B1 | ok | 100 | 1.5 | 5.0 | 35.0 | 7.02 | 35.0 | 1108 | peak_bytes_in_use | 13.96x | - |
| strang2_d0.1_dct|segment|B1 | ok | 100 | 1.4 | 4.9 | 29.9 | 6.05 | 29.9 | 139 | peak_bytes_in_use | 16.35x | - |
| etd2_d0.1|none|B1 | ok | 100 | 0.9 | 6.5 | 24.6 | 3.76 | 24.6 | 562 | peak_bytes_in_use | 19.85x | - |
| etd2_d0.1|segment|B1 | ok | 100 | 0.9 | 7.3 | 27.9 | 3.81 | 27.9 | 130 | peak_bytes_in_use | 17.47x | - |
| etd2_d0.1_dct|none|B1 | ok | 100 | 1.6 | 11.8 | 38.7 | 3.28 | 38.7 | 563 | peak_bytes_in_use | 12.61x | - |
| etd2_d0.1_dct|segment|B1 | ok | 100 | 1.6 | 11.7 | 44.5 | 3.79 | 44.5 | 75 | peak_bytes_in_use | 10.98x | - |
| F02|B4 | ok | 500 | 3.0 | 221.7 | 1030.8 | 4.65 | 257.7 | 9473 | peak_bytes_in_use | 1.00x | 0.00e+00 |
| lawson4_d0.2|step|B4 | ok | 50 | 1.0 | 8.8 | 30.3 | 3.43 | 7.6 | 172 | peak_bytes_in_use | 34.01x | 1.91e-06 |
| strang4_d0.05|step|B4 | ok | 200 | 1.2 | 7.3 | 32.1 | 4.37 | 8.0 | 397 | peak_bytes_in_use | 32.07x | 0.00e+00 |
| F02|B16 | oom | - | - | - | - | - | - | - | - | - | - |
| lawson4_d0.2|step|B16 | ok | 50 | 1.0 | 29.9 | 96.8 | 3.24 | 6.1 | 686 | peak_bytes_in_use | - | 5.07e-06 |
| strang4_d0.05|step|B16 | ok | 200 | 1.2 | 24.7 | 110.3 | 4.47 | 6.9 | 1586 | peak_bytes_in_use | - | 1.53e-05 |
| F02|B32 | oom | - | - | - | - | - | - | - | - | - | - |
| lawson4_d0.2|step|B32 | ok | 50 | 0.9 | 60.5 | 192.2 | 3.18 | 6.0 | 1185 | peak_bytes_in_use | - | 1.70e-06 |
| strang4_d0.05|step|B32 | ok | 200 | 1.2 | 45.9 | 216.1 | 4.71 | 6.8 | 3179 | peak_bytes_in_use | - | 1.53e-05 |
| lawson4_d0.2|step|unroll4|B1 | ok | 50 | 1.5 | 3.4 | 9.4 | 2.76 | 9.4 | 146 | peak_bytes_in_use | 51.84x | - |
| strang4_d0.05|step|unroll4|B1 | ok | 200 | 2.1 | 3.6 | 13.9 | 3.82 | 13.9 | 139 | peak_bytes_in_use | 35.21x | - |

Not run / failed: F02|B16 (oom); F02|B32 (oom)
