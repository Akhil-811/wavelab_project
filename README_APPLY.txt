WaveLab demonstration-readiness patch

This patch preserves your datasets, model files, result JSON/CSV files, experiment history and submitted report. It changes only dashboard code and adds a validator.

Apply on macOS:
1. Stop server with Ctrl+C.
2. From your wavelab_connected_project folder, back up: cp server.py server.py.before_demo_fixes.bak ; cp dashboard/index.html dashboard/index.html.before_demo_fixes.bak
3. Copy this patch's server.py to the project root, dashboard/index.html to dashboard/index.html, and src/validate_dashboard_evidence.py to src/.
4. Start: source .venv/bin/activate && python server.py
5. Open http://127.0.0.1:8765 → Validation & tests. Select a dataset/model/seed and click Load run evidence.

Fixes: (a) Model Laboratory comparison chart now shows all three models instead of only the seed-stability model; (b) dashboard model-win claims are recalculated from live metrics; (c) /api/health audits all 36 metrics, split metadata, saved histories and prediction consistency; (d) /api/artifacts serves a selected run's history and held-out residual histogram only if its predictions reproduce the saved MAE/RMSE/NMAE/R²; (e) NMAE definition is documented as test MAE divided by mean absolute actual Total_Power on the held-out test set.

Scientific safeguards: no curves or residuals are fabricated. Missing/mismatched artifacts remain flagged. The residual view is held-out evaluation, not live new-layout prediction. Backend health means local endpoints respond, not production deployment readiness.

Restore: stop server and restore the .bak files.
