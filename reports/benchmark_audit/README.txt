WaveLab Task 3 - Benchmark validation

Canonical benchmark source:
- Use only the 36 seed-specific metric JSON files with seeds 21, 42 and 123.
- These represent 3 model families x 4 dataset configurations x 3 seeds.
- Each seeded JSON has a corresponding seed-specific prediction CSV.

Validation results:
- All 36 seeded metric records have matching prediction exports.
- MAE, RMSE and R2 recomputed from predictions agree with saved metrics within 0.02 power units (rounding tolerance); R2 agrees to numerical precision.
- All 36 prediction row counts match the test_rows value in the metric JSON.
- All 12 model/dataset combinations have complete seed coverage {21, 42, 123}.
- Dashboard embedded RAW values were out of date for 9 of 36 records. dashboard/index.html has been synchronized to the 36 canonical seeded JSON files.
- 35 of 36 seeded metric JSON files are byte-identical to the pre-validation backup. The remaining file differs only in inference timing fields; all predictive metrics and other metadata match.
- There are 12 un-suffixed *_metrics.json artifacts that duplicate a seeded run identity. Some contain different metrics from the seed-suffixed file, so they must not be mixed into the three-seed summary. They are preserved and listed in un_suffixed_metric_artifacts.csv.

Summary table:
- benchmark_mean_sd.csv contains means and sample standard deviations across seeds 21, 42 and 123.

Re-run from the project root:
    python -m src.validate_benchmarks

To refresh the dashboard's embedded benchmark values from the canonical seeded JSON files:
    python -m src.validate_benchmarks --sync-dashboard

The validation script does not modify raw data, model weights, prediction files or metric JSON files. The --sync-dashboard option modifies only dashboard/index.html.
