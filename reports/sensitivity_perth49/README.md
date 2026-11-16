# WaveLab Perth 49 aggregation sensitivity pilot

## Purpose
This is a separate, non-destructive sensitivity experiment. It tests whether replacing repeated row-level targets with the mean `Total_Power` for each exact coordinate layout changes coordinate-MLP performance.

## Safety
- The original `data/raw/WEC_Perth_49.csv` was read only.
- The script writes derived files under `reports/sensitivity_perth49/`.
- It does not overwrite original data, benchmark records, model checkpoints, or dashboard files.

## Reproduce
From the WaveLab project root, with the project's Python environment active:

```bash
python -m src.run_perth49_aggregation_sensitivity --seeds 21 42 123 --epochs 250 --patience 25
```

The packaged pilot result was run with seed 42, split seed 42, a maximum of 60 epochs and patience 15. The script defaults to the full three-seed, 250-epoch configuration. To reproduce the pilot exactly, run: `python -m src.run_perth49_aggregation_sensitivity --seeds 42 --epochs 60 --patience 15`. It uses the same coordinate MLP architecture and grouped split logic as `src/train_mlp_grouped.py`.

## What is compared
1. Original arm: train on all original training rows, retaining repeated layouts and their individual `Total_Power` labels.
2. Aggregated arm: after assigning layouts to train/validation/test, train on one row per training layout, with `mean(Total_Power)` as its target.

Both arms are evaluated against the **same original row-level test records**. The aggregated arm is also reported against one mean target per test layout as a secondary, different evaluation.

## Pilot result (one seed)
- Original training rows: 25,372.
- Aggregated training examples: 7,152 unique layouts (71.8% fewer examples).
- Original model MAE on original test rows: 16,705.98.
- Aggregated model MAE on the same original test rows: 17,695.17 (5.9% higher).
- Original model RMSE: 28,699.07.
- Aggregated model RMSE on original test rows: 27,247.09.
- Original model R²: 0.9403.
- Aggregated model R² on original test rows: 0.9462.

The pilot is exploratory, not conclusive: it uses one seed and a 60-epoch cap. The different MAE and RMSE directions show that aggregation does not uniformly improve performance. Do not replace the original benchmark or claim aggregation is scientifically preferable based on this run alone.

## Output files
- `sensitivity_metrics_by_seed.csv`: per-arm metrics.
- `sensitivity_summary.csv`: summary table.
- `test_predictions_comparison.csv`: predictions on identical original test rows.
- `perth49_layout_mean_targets.csv`: derived one-row-per-layout dataset with mean/median, row count, spread and min/max target diagnostics.
- `experiment_metadata.json`: source SHA-256, split sizes, seeds and methodological notes.
