# Transformer results

**2026-11-16 17:29:04 (UTC+8)**

## Training and performance

| Experiment | Size | Seed | Status | Prediction tokens / budget | Learning rate | Dropout | Microbatch × accumulation | Best validation NLL / PPL | Final validation NLL / PPL | Training tokens/s | Timing scope | Training seconds | Wall seconds / million tokens | Peak allocated GPU memory (GiB) | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Transformer-only | L | 13 | Complete | 250,000,000 / 250M | 0.001 | 0.0 | 64 × 1 | 3.197021 / 24.46 | 3.197838 / 24.48 | 30,288.25 | Whole run | 8,254.03 | — | 18.80 | [summary](runs/transformer_only_250M_3seed/l/seed13/summary.json) |
| Transformer-only | L | 37 | Complete | 250,000,000 / 250M | 0.001 | 0.0 | 64 × 1 | 3.207560 / 24.72 | 3.208693 / 24.75 | 29,822.19 | Whole run | 8,383.02 | — | 18.80 | [summary](runs/transformer_only_250M_3seed/l/seed37/summary.json) |
| Transformer-only | L | 73 | Running | 71,917,652 / 250M | 0.001 | 0.0 | 64 × 1 | 3.706375 / 40.71 | — | — | — | — | — | — | [metrics](runs/transformer_only_250M_3seed/l/seed73/metrics.jsonl) |
| Transformer-only | M | 13 | Running | 167,903,984 / 250M | 0.001 | 0.0 | 64 × 1 | 3.535605 / 34.32 | — | — | — | — | — | — | [metrics](runs/transformer_only_250M_3seed/m/seed13/metrics.jsonl) |
| Transformer-only | M | 37 | Queued | 0 / 250M | 0.001 | 0.0 | 64 × 1 | — | — | — | — | — | — | — | — |
| Transformer-only | M | 73 | Queued | 0 / 250M | 0.001 | 0.0 | 64 × 1 | — | — | — | — | — | — | — | — |
| Transformer-only | S | 13 | Queued | 0 / 250M | 0.001 | 0.0 | 64 × 1 | — | — | — | — | — | — | — | — |
| Transformer-only | S | 37 | Queued | 0 / 250M | 0.001 | 0.0 | 64 × 1 | — | — | — | — | — | — | — | — |
| Transformer-only | S | 73 | Queued | 0 / 250M | 0.001 | 0.0 | 64 × 1 | — | — | — | — | — | — | — | — |
| Initial exploratory | S | 101 | Complete | 250M / 250M | 0.0003 | 0.1 | 2 × 32 | 4.110006 / 60.95 | 4.110006 / 60.95 | 56,818.05 | Whole run | 4,400.01 | — | 0.49 | [summary](runs/s/exploratory_250M_seed101/summary.json) |
| Initial exploratory | M | 101 | Complete | 250M / 250M | 0.0003 | 0.1 | 2 × 32 | 3.772511 / 43.49 | 3.773657 / 43.54 | 37,332.10 | Whole run | 6,696.65 | — | 0.73 | [summary](runs/m/exploratory_250M_seed101/summary.json) |
| Initial exploratory | L | 101 | Complete | 250M / 250M | 0.0003 | 0.1 | 2 × 32 | 3.452494 / 31.58 | 3.453102 / 31.60 | 16,153.18 | Whole run | 15,476.82 | — | 1.39 | [summary](runs/l/exploratory_250M_seed101/summary.json) |
| Batch benchmark | S | 101 | Complete | 2M / 2M | 0.001 | 0.0 | 16 × 4 | — | — | 162,081.43 | After warm-up | 10.37 | 10.72 | 2.69 | [report](runs/batch_benchmark_20261116/s_mb16/attempt_1794806926532507791/transformer_pilot.json) |
| Batch benchmark | S | 101 | Complete | 2M / 2M | 0.001 | 0.0 | 32 × 2 | — | — | 164,341.41 | After warm-up | 10.23 | 10.66 | 5.21 | [report](runs/batch_benchmark_20261116/s_mb32/attempt_1794806917294340222/transformer_pilot.json) |
| Batch benchmark | S | 101 | Complete; common batch selected | 2M / 2M | 0.001 | 0.0 | 64 × 1 | — | — | 161,123.71 | After warm-up | 10.43 | 10.83 | 10.18 | [report](runs/batch_benchmark_20261116/s_mb64/attempt_1794806888615514897/transformer_pilot.json) |
| Batch benchmark | M | 101 | Complete | 2M / 2M | 0.001 | 0.0 | 16 × 4 | — | — | 79,456.30 | After warm-up | 21.15 | 18.56 | 3.47 | [report](runs/batch_benchmark_20261116/m_mb16/attempt_1794806882470941934/transformer_pilot.json) |
| Batch benchmark | M | 101 | Complete | 2M / 2M | 0.001 | 0.0 | 32 × 2 | — | — | 84,593.95 | After warm-up | 19.87 | 17.77 | 6.62 | [report](runs/batch_benchmark_20261116/m_mb32/attempt_1794806840022283298/transformer_pilot.json) |
| Batch benchmark | M | 101 | Complete; common batch selected | 2M / 2M | 0.001 | 0.0 | 64 × 1 | — | — | 85,852.31 | After warm-up | 19.58 | 17.69 | 12.82 | [report](runs/batch_benchmark_20261116/m_mb64/attempt_1794806797695560610/transformer_pilot.json) |
| Batch benchmark | L | 101 | Complete | 2M / 2M | 0.001 | 0.0 | 16 × 4 | — | — | 32,692.28 | After warm-up | 51.41 | 42.01 | 5.36 | [report](runs/batch_benchmark_20261116/l_mb16/attempt_1794806797492295354/transformer_pilot.json) |
| Batch benchmark | L | 101 | Complete | 2M / 2M | 0.001 | 0.0 | 32 × 2 | — | — | 34,526.26 | After warm-up | 48.68 | 40.70 | 9.90 | [report](runs/batch_benchmark_20261116/l_mb32/attempt_1794806709112607453/transformer_pilot.json) |
| Batch benchmark | L | 101 | Complete; common batch selected | 2M / 2M | 0.001 | 0.0 | 64 × 1 | — | — | 34,575.05 | After warm-up | 48.61 | 40.51 | 18.76 | [report](runs/batch_benchmark_20261116/l_mb64/attempt_1794806709159186779/transformer_pilot.json) |
| Timing pilot | S | 101 | Complete | 5M / 5M | 0.0003 | 0.1 | 16 × 4 | — | 6.833965 / 928.87 | 157,637.18 | After warm-up | 29.64 | 8.79 | 2.71 | [report](runs/timing_mb16_job2557/transformer_pilot.json) |
| Timing pilot | M | 101 | Complete | 5M / 5M | 0.0003 | 0.1 | 16 × 4 | — | 6.608971 / 741.72 | 67,579.13 | After warm-up | 69.14 | 18.74 | 3.51 | [report](runs/timing_mb16_job2557/transformer_pilot.json) |
| Timing pilot | L | 101 | Complete | 5M / 5M | 0.0003 | 0.1 | 16 × 4 | — | 6.501252 / 665.97 | 24,843.05 | After warm-up | 188.07 | 49.16 | 5.44 | [report](runs/timing_mb16_job2557/transformer_pilot.json) |

## Size-S tuning

| Trial | Seed | Status | Prediction tokens | Updates | Learning rate | Dropout | Microbatch × accumulation | Best validation NLL | Best validation PPL | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 00 | 101 | Complete | 50,000,000 | 1,617 | 0.0001 | 0.0 | 16 × 4 | 5.914180 | 370.25 | [summary](runs/s/tuning_50M_seed101_6lr/trial_00/summary.json) |
| 01 | 101 | Complete | 50,000,000 | 1,617 | 0.0001 | 0.1 | 16 × 4 | 5.952900 | 384.87 | [summary](runs/s/tuning_50M_seed101_6lr/trial_01/summary.json) |
| 02 | 101 | Complete | 50,000,000 | 1,617 | 0.0003 | 0.0 | 16 × 4 | 5.227199 | 186.27 | [summary](runs/s/tuning_50M_seed101_6lr/trial_02/summary.json) |
| 03 | 101 | Complete | 50,000,000 | 1,617 | 0.0003 | 0.1 | 16 × 4 | 5.291914 | 198.72 | [summary](runs/s/tuning_50M_seed101_6lr/trial_03/summary.json) |
| 04 | 101 | **Selected** | 50,000,000 | 1,617 | 0.001 | 0.0 | 16 × 4 | 4.546444 | 94.30 | [summary](runs/s/tuning_50M_seed101_6lr/trial_04/summary.json) |
| 05 | 101 | Complete | 50,000,000 | 1,617 | 0.001 | 0.1 | 16 × 4 | 4.661616 | 105.81 | [summary](runs/s/tuning_50M_seed101_6lr/trial_05/summary.json) |
| 06 | 101 | Complete | 50,000,000 | 1,617 | 0.003 | 0.0 | 16 × 4 | 4.790328 | 120.34 | [summary](runs/s/tuning_50M_seed101_6lr/trial_06/summary.json) |
| 07 | 101 | Complete | 50,000,000 | 1,617 | 0.003 | 0.1 | 16 × 4 | 4.814863 | 123.33 | [summary](runs/s/tuning_50M_seed101_6lr/trial_07/summary.json) |
| 08 | 101 | Complete | 50,000,000 | 1,617 | 0.01 | 0.0 | 16 × 4 | 5.820429 | 337.12 | [summary](runs/s/tuning_50M_seed101_6lr/trial_08/summary.json) |
| 09 | 101 | Complete | 50,000,000 | 1,617 | 0.01 | 0.1 | 16 × 4 | 6.085196 | 439.31 | [summary](runs/s/tuning_50M_seed101_6lr/trial_09/summary.json) |
| 10 | 101 | Complete | 50,000,000 | 1,617 | 0.03 | 0.0 | 16 × 4 | 6.193571 | 489.59 | [summary](runs/s/tuning_50M_seed101_6lr/trial_10/summary.json) |
| 11 | 101 | Complete | 50,000,000 | 1,617 | 0.03 | 0.1 | 16 × 4 | 6.404502 | 604.56 | [summary](runs/s/tuning_50M_seed101_6lr/trial_11/summary.json) |
