# FD001 — Multi-seed evaluation of frozen Chronos embeddings (run 1, 2026-10-31)

Notebook: `notebooks/W09_FD001_MultiSeed_Stats.ipynb` · raw output: `fd001_multiseed_run1_raw_output.txt`

## Protocol
- C-MAPSS FD001, official split: 100 train engines (20,631 cycles), 100 test engines (13,096 cycles).
- 24 raw variables (21 sensors + 3 settings), global MinMax fitted on train, W = 30, RUL cap 125.
- Windows: 17,731 train / 10,196 test; evaluation on the last window of each test engine (N = 100).
- Shortest test trajectory: 31 cycles (no test engine shorter than W). Train trajectories 128–362 cycles.
- Feature selection: variance 24→17 (drops s1, s5, s10, s16, s18, s19, setting_3), correlation 17→16 (drops s14).
  AFICv: 8 variables, 90.5% cumulative XGBoost importance (s11, s4, s9, s12, s7, s15, s14, s17).
- Chronos-2 = `amazon/chronos-2`, 119.5M parameters, frozen. Chronos-1 = `chronos-t5-small`, frozen.
- Seeds 42, 123, 456, 789, 1000. Ridge and XGBoost (default params) are deterministic w.r.t. the seed → std 0.

## Results (RMSE mean ± std over 5 seeds, NASA score)

| Configuration | Dim | RMSE | Score |
|---|---|---|---|
| XGBoost on raw windows (baseline) | 480 | 13.63 | 339.3 |
| **Chronos-2 + MLP, mean, correlation** | 768 | **14.61 ± 0.17** (14.39–14.84) | 357.9 ± 30.4 |
| Chronos-2 + MLP, concat, correlation | 3072 | 15.07 ± 0.23 | 379.5 ± 45.0 |
| Chronos-2 + Ridge, last, correlation | 768 | 15.20 | 349.8 |
| Chronos-2 + MLP, last, correlation | 768 | 15.21 ± 0.23 | 413.7 ± 34.9 |
| Chronos-2 + Ridge, concat, correlation | 3072 | 15.49 | 365.1 |
| Chronos-2 + Ridge, mean, correlation | 768 | 15.68 | 387.9 |
| Chronos-1 + Ridge, correlation | 8192 | 15.68 | 361.0 |
| Chronos-2 + XGBoost, concat, correlation | 3072 | 15.87 | 476.0 |
| Chronos-2 + XGBoost, mean, correlation | 768 | 16.97 | 702.0 |
| Chronos-2 + XGBoost, last, correlation | 768 | 17.11 | 534.7 |
| Chronos-1 + XGBoost, correlation | 8192 | 17.52 | 563.9 |
| Chronos-1 + MLP, correlation | 8192 | 18.19 ± 0.32 | 685.5 ± 12.1 |
| Chronos-2, AFICv (all poolings/heads) | — | 18.49 – 21.19 | 662 – 1997 |
| Task-specific LSTM v1 (**unstable, do not use**) | — | 19.25 ± 11.83 (13.30–40.40) | — |

## Statistics (vs Chronos-2 + MLP mean; bootstrap over 100 engines, paired Wilcoxon)
- Chronos-2 + MLP mean: seed-averaged RMSE 14.49, 95% CI [12.44, 16.44].
- vs XGBoost raw: ΔRMSE −0.86 [−3.90, 2.27], p = 0.26 → not significant.
- vs LSTM v1 (seed-averaged 14.74): Δ +0.25 [−2.24, 2.71], p = 0.11 → not significant.
- vs Chronos-1 + MLP: Δ +3.45 [0.00, 6.85], p = 0.55 (CI touches 0) → borderline.
- vs Chronos-2 Ridge mean: Δ +1.19 [0.21, 2.24], p = 0.040 → significant.
- All AFICv configurations: Δ ≥ +3.9, CIs exclude 0, p ≤ 0.033 → significant degradation.

## Findings
1. The single-run result of the MCCSAI paper (14.43) is within the 5-seed range (14.61 ± 0.17): robust.
2. Feature diversity effect confirmed and statistically significant (AFICv always worse).
3. Chronos-2 + MLP vs task-specific baselines: differences not significant on FD001.
4. Chronos-2 > Chronos-1 with MLP (14.61 vs 18.19) but tie with Ridge (15.68 vs 15.68).
5. The MLP head is consistently the best head for frozen Chronos-2 embeddings.

## Caveats / to fix
- **LSTM v1 diverged on one seed** (unscaled targets, no gradient clipping). Fixed in notebook (`BASE|LSTMv2`):
  targets scaled by RUL_CAP, gradient clipping 1.0, ReduceLROnPlateau. Re-run needed.
- **XGBoost numbers differ from the March 2026 run (W06)**: mean-pooling XGBoost 16.97 here vs 15.62 in the paper,
  while Ridge is identical (15.68) → embeddings identical; the difference comes from the xgboost library version
  (also changes the AFICv list: s7 replaces s20). Library versions are now saved in `fd001_facts.json`.
- Chronos-1 + MLP here (18.19) is much worse than in the IMSA study (14.92): different pipeline details
  (early stopping, embedding aggregation); to investigate before reusing.
