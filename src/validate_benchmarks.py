"""Audit seeded benchmark JSONs against prediction exports and dashboard data.

Run from the project root:
    python -m src.validate_benchmarks
    python -m src.validate_benchmarks --sync-dashboard

The default mode is read-only apart from writing audit reports. The optional
sync flag updates only the embedded RAW benchmark array in dashboard/index.html.
"""
from __future__ import annotations
import argparse, glob, json, os, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BACKUP = ROOT / "backups" / "results_before_validation"
REPORTS = ROOT / "reports" / "benchmark_audit"
MODEL_DIRS = {
    "grouped": "Coordinate MLP",
    "spatial": "Spatial-feature MLP",
    "gnn": "GNN",
}
METRIC_MAP = {"MAE": "test_mae", "RMSE": "test_rmse", "NMAE": "test_nmae", "R2": "test_r2"}


def main(sync_dashboard: bool = False) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    records, issues = [], []
    duplicates = defaultdict(list)
    for folder, model in MODEL_DIRS.items():
        for path in sorted((RESULTS / folder).glob("*_seed*_metrics.json")):
            data = json.loads(path.read_text())
            seed = data.get("seed")
            if seed not in (21, 42, 123):
                issues.append({"type": "unexpected_seed", "file": str(path.relative_to(ROOT)), "detail": str(seed)})
                continue
            dataset_id = Path(data["dataset"]).stem
            dataset_label = dataset_id.replace("WEC_", "").replace("_", " ")
            pred_path = path.with_name(path.name.replace("_metrics.json", "_predictions.csv"))
            row = {
                "dataset": dataset_label, "dataset_id": dataset_id, "model": model,
                "seed": seed, "metrics_file": str(path.relative_to(ROOT)),
                "prediction_file": str(pred_path.relative_to(ROOT)),
                "prediction_rows": None,
            }
            if not pred_path.exists():
                issues.append({"type": "missing_prediction_file", "file": str(path.relative_to(ROOT)), "detail": str(pred_path)})
            else:
                pred = pd.read_csv(pred_path)
                row["prediction_rows"] = len(pred)
                if data.get("test_rows") is not None and len(pred) != int(data["test_rows"]):
                    issues.append({"type": "prediction_row_count_mismatch", "file": str(pred_path.relative_to(ROOT)), "detail": f"prediction_rows={len(pred)}; test_rows={data.get('test_rows')}"})
                if not {"actual_total_power", "predicted_total_power"}.issubset(pred.columns):
                    issues.append({"type": "invalid_prediction_columns", "file": str(pred_path.relative_to(ROOT)), "detail": str(pred.columns.tolist())})
                else:
                    y = pred["actual_total_power"].to_numpy(dtype=float)
                    p = pred["predicted_total_power"].to_numpy(dtype=float)
                    err = y - p
                    recomputed = {
                        "MAE": float(np.mean(np.abs(err))),
                        "RMSE": float(np.sqrt(np.mean(err ** 2))),
                        "R2": float(1 - np.sum(err ** 2) / np.sum((y - np.mean(y)) ** 2)),
                    }
                    for metric, actual in recomputed.items():
                        saved = float(data[METRIC_MAP[metric]])
                        diff = abs(saved - actual)
                        row[f"saved_{metric}"] = saved
                        row[f"recomputed_{metric}"] = actual
                        row[f"absolute_difference_{metric}"] = diff
                        if diff > max(2e-2, abs(saved) * 1e-7):
                            issues.append({"type": f"{metric.lower()}_does_not_match_predictions", "file": str(path.relative_to(ROOT)), "detail": f"saved={saved}; recomputed={actual}; abs_diff={diff}"})
                    row["saved_NMAE"] = float(data["test_nmae"])
                    row["saved_inference_ms_per_sample"] = float(data.get("test_inference_ms_per_sample", np.nan))
            row["rows_in_metrics"] = data.get("rows")
            row["unique_layouts"] = data.get("unique_layouts")
            row["split_method"] = data.get("split_method")
            row["split_seed"] = data.get("split_seed")
            row["test_rows"] = data.get("test_rows")
            records.append(row)
            duplicates[(folder, dataset_id, seed)].append((path, data))

    # Verify complete three-seed coverage for every model x dataset.
    coverage = defaultdict(set)
    for r in records:
        coverage[(r["model"], r["dataset"])].add(r["seed"])
    for model in MODEL_DIRS.values():
        for ds in sorted({r["dataset"] for r in records}):
            seeds = coverage[(model, ds)]
            if seeds != {21, 42, 123}:
                issues.append({"type": "incomplete_seed_coverage", "file": f"{model}/{ds}", "detail": f"seeds={sorted(seeds)}"})

    # Compare the exact seeded JSON artifacts to their pre-validation backups.
    backup_rows = []
    for r in records:
        rel = Path(r["metrics_file"])
        current = ROOT / rel
        old = BACKUP / rel.relative_to("results")
        status = "missing_backup" if not old.exists() else ("identical" if current.read_bytes() == old.read_bytes() else "different")
        backup_rows.append({"metrics_file": str(rel), "backup_file": str(old.relative_to(ROOT)), "status": status})

    # Read the dashboard's embedded benchmark data and compare against seeded JSONs.
    dashboard_path = ROOT / "dashboard" / "index.html"
    html = dashboard_path.read_text()
    match = re.search(r"let RAW = (\[.*?\]);\n", html, re.S)
    dashboard_rows = json.loads(match.group(1)) if match else []
    dashboard_index = {(r["dataset"], r["model"], int(r["seed"])): r for r in dashboard_rows}
    dashboard_diff = []
    canonical_raw = []
    for r in records:
        key = (r["dataset"], r["model"], int(r["seed"]))
        old = dashboard_index.get(key)
        current_json = json.loads((ROOT / r["metrics_file"]).read_text())
        new = {
            "dataset": r["dataset"], "model": r["model"], "seed": int(r["seed"]),
            "MAE": float(current_json["test_mae"]), "RMSE": float(current_json["test_rmse"]),
            "NMAE": float(current_json["test_nmae"]), "R2": float(current_json["test_r2"]),
            "inference_ms_per_sample": float(current_json.get("test_inference_ms_per_sample", np.nan)),
            "source_file": r["metrics_file"],
        }
        canonical_raw.append(new)
        if old is None:
            dashboard_diff.append({"dataset": key[0], "model": key[1], "seed": key[2], "status": "missing_from_dashboard"})
        else:
            diffs = {m: abs(float(old[m]) - float(new[m])) for m in ("MAE", "RMSE", "NMAE", "R2")}
            if max(diffs.values()) > 1e-8:
                dashboard_diff.append({"dataset": key[0], "model": key[1], "seed": key[2], "status": "mismatch", **{f"abs_diff_{k}": v for k, v in diffs.items()}})
    for key in dashboard_index:
        if key not in {(r["dataset"], r["model"], int(r["seed"])) for r in records}:
            dashboard_diff.append({"dataset": key[0], "model": key[1], "seed": key[2], "status": "extra_dashboard_record"})

    # Identify un-suffixed metric JSONs that duplicate seeded runs, but do not use them in summaries.
    aliases = []
    for folder, model in MODEL_DIRS.items():
        for path in sorted((RESULTS / folder).glob("*_metrics.json")):
            if re.search(r"_seed\d+_metrics\.json$", path.name):
                continue
            data = json.loads(path.read_text())
            ds = Path(data.get("dataset", "unknown")).stem
            seed = data.get("seed")
            aliases.append({"file": str(path.relative_to(ROOT)), "model": model, "dataset": ds,
                            "seed": seed, "test_mae": data.get("test_mae"),
                            "note": "Un-suffixed duplicate/alternate artifact; excluded from canonical seeded summary."})

    df = pd.DataFrame(records).sort_values(["dataset", "model", "seed"])
    df.to_csv(REPORTS / "per_seed_validation.csv", index=False)
    pd.DataFrame(backup_rows).to_csv(REPORTS / "backup_comparison.csv", index=False)
    pd.DataFrame(dashboard_diff).to_csv(REPORTS / "dashboard_mismatch.csv", index=False)
    pd.DataFrame(aliases).to_csv(REPORTS / "un_suffixed_metric_artifacts.csv", index=False)

    summary_rows = []
    for (dataset, model), grp in df.groupby(["dataset", "model"]):
        summary_rows.append({
            "dataset": dataset, "model": model, "n_seeds": len(grp),
            "seeds": ",".join(map(str, sorted(grp["seed"].astype(int)))),
            "MAE_mean": grp["saved_MAE"].mean(), "MAE_sample_sd": grp["saved_MAE"].std(ddof=1),
            "RMSE_mean": grp["saved_RMSE"].mean(), "RMSE_sample_sd": grp["saved_RMSE"].std(ddof=1),
            "NMAE_mean": grp["saved_NMAE"].mean(), "NMAE_sample_sd": grp["saved_NMAE"].std(ddof=1),
            "R2_mean": grp["saved_R2"].mean(), "R2_sample_sd": grp["saved_R2"].std(ddof=1),
        })
    pd.DataFrame(summary_rows).to_csv(REPORTS / "benchmark_mean_sd.csv", index=False)
    (REPORTS / "validation_issues.json").write_text(json.dumps(issues, indent=2))

    if sync_dashboard:
        if not match:
            raise RuntimeError("Could not locate embedded RAW array in dashboard/index.html")
        replacement = "let RAW = " + json.dumps(canonical_raw, separators=(",", ":")) + ";\n"
        html = html[:match.start()] + replacement + html[match.end():]
        dashboard_path.write_text(html)

    backup_counts = pd.Series([x["status"] for x in backup_rows]).value_counts().to_dict()
    print(f"Canonical seeded records: {len(records)} (expected 36)")
    print(f"Prediction-level MAE/RMSE/R2 issues: {len([i for i in issues if 'does_not_match' in i['type']])}")
    print(f"Dashboard differences: {len(dashboard_diff)}")
    print(f"Backup comparison: {backup_counts}")
    print(f"Un-suffixed metric artifacts: {len(aliases)}")
    print(f"Other validation issues: {len(issues)}")
    print(f"Reports written to: {REPORTS.relative_to(ROOT)}")
    if sync_dashboard:
        print("Dashboard RAW benchmark array synchronized from canonical seed JSONs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync-dashboard", action="store_true", help="replace embedded dashboard RAW data from canonical seed JSONs")
    args = parser.parse_args()
    main(args.sync_dashboard)
