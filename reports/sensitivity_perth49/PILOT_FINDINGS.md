# Perth 49 aggregation sensitivity pilot — findings

The experiment compared the existing row-level target formulation with one mean target per exact coordinate layout. Both arms used the same layout-grouped train/validation/test assignment (split seed 42) and the same coordinate MLP architecture. Both were evaluated against the same original row-level test set.

| Metric | Original row-level model | Layout-mean model, evaluated on original test rows |
|---|---:|---:|
| Training examples | 25,372 | 7,152 |
| Test rows | 5,350 | 5,350 |
| Test MAE | 16,705.98 | 17,695.17 |
| Test RMSE | 28,699.07 | 27,247.09 |
| Test R² | 0.9403 | 0.9462 |

Aggregation reduced training examples by 71.8%. On the shared original test rows, MAE increased by 5.9%, while RMSE decreased and R² increased slightly. Thus, the result is mixed rather than a consistent improvement. When evaluated against layout-mean test targets instead, the aggregated model's MAE was 24,983.79; this is a different target definition and must not be compared directly with the original-row MAE.

## Interpretation

This pilot does not justify replacing the original target or changing the main benchmark. It demonstrates that aggregating repeated layouts changes the learning problem and reduces the number of training examples substantially. The original row-level data may contain variation associated with simulation conditions not represented among the coordinate-only inputs. Averaging removes some of that variation but does not necessarily improve prediction of individual original records.

## Limitations and next step

Only seed 42 was run, with a 60-epoch maximum and early stopping. This is a pilot, not a final multi-seed result. Before drawing a conclusion, repeat the paired experiment with seeds 21 and 123 and the full training budget used for the main benchmark. Preserve the raw CSV and existing benchmark records.
