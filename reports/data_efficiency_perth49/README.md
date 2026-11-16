# WaveLab Perth 49 data-efficiency experiment

## What this experiment does
Compares the project's existing Coordinate MLP, Spatial-feature MLP and GNN on Perth 49 at 20%, 40%, 60%, 80% and 100% of the **unique training layouts**.

It reuses the existing model classes and the same two-stage `GroupShuffleSplit` method in the training scripts. The validation and test partitions remain fixed across fractions and seeds. Training subsets are nested within each seed. All records belonging to a selected layout are retained.

The percentage refers to unique training layouts, not raw CSV rows. This matters because Perth 49 contains repeated layouts.

## Safety
- Original CSV files are read only.
- Existing `results/` and model benchmark files are not overwritten.
- Outputs are written under `reports/data_efficiency_perth49/`.

## Run from the project root
First perform a dry run:

```bash
python -m src.run_data_efficiency --dry-run
```

Then run all models, five fractions and three seeds:

```bash
python -m src.run_data_efficiency --seeds 21 42 123 --fractions 0.2 0.4 0.6 0.8 1.0 --epochs 250 --patience 25
```

This is 45 training runs. On a CPU-only Mac it may take a while. Keep the computer plugged in and prevent it from sleeping. The script prints progress after every run and writes partial results after each run so progress is recoverable.

## Outputs
- `data_efficiency_per_seed.csv`: each run's metrics and sample sizes.
- `data_efficiency_mean_sd.csv`: mean and sample standard deviation across seeds.
- `*_history.csv`: validation history for each run.
- `experiment_metadata.json`: split, sampling and model configuration.

## Methodological notes
- The test and validation sets are fixed across all fractions.
- Sampling is by unique layout groups, not individual rows, preventing repeated layouts from being split across the training subset.
- Subsets are nested: 20% is contained in 40%, etc., within each seed.
- The three architectures use the same selected layouts for a given seed and fraction.
- Scalers are fitted on selected training rows only.
- Spatial features are computed from coordinates only, as in the existing spatial MLP.
- The GNN uses the same architecture and coordinate scaling convention as `src/train_gnn.py`.
