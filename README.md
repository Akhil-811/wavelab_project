# WaveLab connected research dashboard

This bundle includes your existing training source, the four raw UCI CSV datasets, a standalone dashboard, and a local Python HTTP server that launches the existing training modules.

## Start on macOS

1. Open Terminal.
2. Change into this project folder. Example if it is in Downloads:
   `cd ~/Downloads/wavelab_connected_project`
3. Create and activate a virtual environment:
   `python3 -m venv .venv`
   `source .venv/bin/activate`
4. Install dependencies:
   `python -m pip install --upgrade pip`
   `python -m pip install -r requirements.txt`
5. Start the local dashboard and backend:
   `python server.py`
6. Open this address in Safari:
   `http://127.0.0.1:8765`

Keep the Terminal window open while using the dashboard. Press Ctrl+C in Terminal to stop the server.

## Launching experiments

On the Experiment controls page, select dataset, model, training seed, fixed split seed, and epochs, then click Build run config and Launch experiment. The backend runs the existing modules:
- `src.train_mlp_grouped`
- `src.train_spatial_mlp`
- `src.train_gnn`

Choose `All three models` to run all three sequentially for the selected dataset and seed. One experiment sequence can run at a time. Training logs appear in the execution status panel. Results are saved by the existing training scripts under `results/` and checkpoints under `models/`.

## Important notes

- Do not open `dashboard/index.html` by double-clicking for backend execution. Use `http://127.0.0.1:8765` so the page can call the local API.
- A long training job will occupy the backend until it finishes. Leave the Terminal/server running.
- The UI can only launch training for datasets included in `data/raw/`.
- The interface's `epochs` value is passed to the existing script. Other hyperparameters remain at each script's default.
- The backend is intended for local use and binds to `127.0.0.1`; it is not exposed to the network.
- The Results & Comparison page reads the fixed original 36-record benchmark from `dashboard/metrics.json`. That baseline is deliberately kept separate and unchanged when you run new experiments.
- The Experiment Controls page includes Run History. Each newly launched run is recorded in `dashboard_runs/run_history.json`, with a run ID, timestamps, configuration, exit code, metrics captured from the run's result JSON, log path, and newly written checkpoint paths. History is exposed by `GET /api/runs` and survives server restarts.
- After a run completes, the dashboard refreshes the run history automatically. Older experiments launched before this history feature was installed are not automatically reconstructed; they remain in their original result files.
- The training modules are run as subprocesses using the same Python interpreter that starts `server.py`. Activate the `.venv` first.
