#!/usr/bin/env python3
"""Local WaveLab dashboard server and experiment runner (standard library only)."""
from __future__ import annotations
import csv, json, os, subprocess, sys, threading, time, uuid
from datetime import datetime
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
DASHBOARD = ROOT / "dashboard"
DATASETS = {
    "Perth 49": "data/raw/WEC_Perth_49.csv",
    "Perth 100": "data/raw/WEC_Perth_100.csv",
    "Sydney 49": "data/raw/WEC_Sydney_49.csv",
    "Sydney 100": "data/raw/WEC_Sydney_100.csv",
}
MODELS = {
    "Coordinate MLP": ("src.train_mlp_grouped", "results/grouped", "models/grouped"),
    "Spatial-feature MLP": ("src.train_spatial_mlp", "results/spatial", "models/spatial"),
    "GNN": ("src.train_gnn", "results/gnn", "models/gnn"),
}
lock = threading.Lock()
active = {"process": None, "started_at": None, "config": None, "log_path": None, "command": None, "last_returncode": None, "running_flag": False, "child_process": None, "run_id": None}
HISTORY_PATH = ROOT / "dashboard_runs" / "run_history.json"

def read_history():
    try:
        data = json.loads(HISTORY_PATH.read_text())
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []

def write_history(history):
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = HISTORY_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(history, indent=2))
    tmp.replace(HISTORY_PATH)

def dataset_label_from_path(value):
    name = Path(str(value or "")).stem
    location = "Perth" if "Perth" in name else ("Sydney" if "Sydney" in name else "Unknown")
    size = "49" if "49" in name else ("100" if "100" in name else "")
    return f"{location} {size}".strip()

def snapshot_metric(path, model_name):
    try:
        raw = json.loads(path.read_text())
        return {
            "dataset": dataset_label_from_path(raw.get("dataset", path.name)),
            "model": model_name,
            "seed": int(raw.get("seed", 0)),
            "split_seed": int(raw.get("split_seed", 42)),
            "MAE": raw.get("test_mae", raw.get("MAE")),
            "RMSE": raw.get("test_rmse", raw.get("RMSE")),
            "NMAE": raw.get("test_nmae", raw.get("NMAE")),
            "R2": raw.get("test_r2", raw.get("R2")),
            "inference_ms_per_sample": raw.get("test_inference_ms_per_sample", raw.get("inference_ms_per_sample")),
            "best_epoch": raw.get("best_epoch"),
            "result_file": str(path.relative_to(ROOT)),
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None

def find_run_metric(model_name, dataset, seed, since):
    _, result_dir, _ = MODELS[model_name]
    matches = []
    for path in (ROOT / result_dir).glob("*_metrics.json"):
        try:
            if path.stat().st_mtime < since - 2:
                continue
            raw = json.loads(path.read_text())
            if int(raw.get("seed", -1)) != seed:
                continue
            if dataset_label_from_path(raw.get("dataset", path.name)) != dataset:
                continue
            snap = snapshot_metric(path, model_name)
            if snap and all(snap.get(k) is not None for k in ("MAE", "RMSE", "NMAE", "R2")):
                matches.append((path.stat().st_mtime, snap))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return max(matches, key=lambda item: item[0])[1] if matches else None

def read_log():
    path = active.get("log_path")
    if not path or not Path(path).exists():
        return ""
    try:
        return Path(path).read_text(errors="replace")[-18000:]
    except Exception:
        return ""

def metrics_files():
    out=[]
    for folder in ("results/grouped","results/spatial","results/gnn"):
        p=ROOT/folder
        if not p.exists(): continue
        for f in p.glob("*_metrics.json"):
            try:
                d=json.loads(f.read_text())
                name=Path(str(d.get("dataset",""))).stem
                loc="Perth" if "Perth" in name else ("Sydney" if "Sydney" in name else "")
                n="49" if "49" in name else ("100" if "100" in name else "")
                model = "Coordinate MLP" if "grouped" in folder else ("Spatial-feature MLP" if "spatial" in folder else "GNN")
                d["dataset_label"] = f"{loc} {n}".strip()
                d["model_label"] = model
                d["result_file"] = str(f.relative_to(ROOT))
                out.append(d)
            except Exception:
                pass
    return out

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD), **kwargs)
    def log_message(self, fmt, *args):
        print("[http]", fmt % args)
    def send_json(self, obj, status=200):
        body=json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Content-Length",str(len(body)))
        self.send_header("Cache-Control","no-store")
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        path=urlparse(self.path).path
        if path=="/api/status":
            with lock:
                running=bool(active.get("running_flag", False))
                payload={
                    "connected":True,"running":running,
                    "started_at":active["started_at"],
                    "config":active["config"],
                    "command":active["command"],
                    "run_id":active.get("run_id"),
                    "returncode":None if running else active["last_returncode"],
                    "log":read_log(),
                    "metrics_count":len(metrics_files())
                }
            return self.send_json(payload)
        if path == "/api/metrics":
            try:
                # Return the 36 seeded benchmark results across all datasets
                # and model families. Exclude older unseeded metric files.
                model_folders = {
                    "grouped": "Coordinate MLP",
                    "spatial": "Spatial-feature MLP",
                    "gnn": "GNN",
                }
                allowed_seeds = (21, 42, 123)
                records = []

                for folder, model_label in model_folders.items():
                    result_dir = ROOT / "results" / folder
                    if not result_dir.is_dir():
                        continue

                    for file in result_dir.glob("*_metrics.json"):
                        if not any(
                            file.name.endswith(f"_seed{seed}_metrics.json")
                            for seed in allowed_seeds
                        ):
                            continue

                        try:
                            record = json.loads(file.read_text())
                            dataset_label = dataset_label_from_path(
                                record.get("dataset", file.name)
                            )
                            if dataset_label not in DATASETS:
                                continue

                            # Keep original metric fields and add the labels
                            # consumed by the WaveLab dashboard.
                            record["dataset_label"] = dataset_label
                            record["model_label"] = model_label
                            record["result_file"] = str(file.relative_to(ROOT))
                            records.append(record)
                        except (OSError, ValueError, TypeError, json.JSONDecodeError):
                            continue

                records.sort(
                    key=lambda record: (
                        record["dataset_label"],
                        record["model_label"],
                        int(record.get("seed", 0)),
                    )
                )

                return self.send_json(records)
            except Exception as exc:
                return self.send_json({
                    "error": f"Could not load metrics: {exc}"
                }, 500)

        if path=="/api/runs":
            return self.send_json(read_history())
        return super().do_GET()
    def do_POST(self):
        if urlparse(self.path).path!="/api/experiments":
            return self.send_json({"error":"Unknown endpoint"},404)
        try:
            length=int(self.headers.get("Content-Length","0"))
            cfg=json.loads(self.rfile.read(length))
        except Exception as e:
            return self.send_json({"error":f"Invalid JSON: {e}"},400)
        dataset=cfg.get("dataset")
        model=cfg.get("model")
        if dataset not in DATASETS:
            return self.send_json({"error":"Unknown dataset"},400)
        if model not in MODELS and model!="All three models":
            return self.send_json({"error":"Unknown model"},400)
        try:
            seed=int(cfg.get("training_seed",21)); split_seed=int(cfg.get("split_seed",42))
            epochs=int(cfg.get("epochs",100))
        except Exception:
            return self.send_json({"error":"Seed and epochs must be integers"},400)
        if not (1 <= epochs <= 10000) or seed<0 or split_seed<0:
            return self.send_json({"error":"Use epochs 1–10000 and non-negative seeds"},400)
        data_path=ROOT/DATASETS[dataset]
        if not data_path.exists():
            return self.send_json({"error":f"Dataset file not found: {DATASETS[dataset]}. Put the UCI CSV files in data/raw/."},400)
        with lock:
            if active.get("running_flag", False):
                return self.send_json({"error":"An experiment is already running. Wait for it to finish before starting another."},409)
            chosen=list(MODELS.items()) if model=="All three models" else [(model,MODELS[model])]
            commands=[]
            for model_name,(module,outdir,modeldir) in chosen:
                commands.append([sys.executable,"-m",module,"--data",DATASETS[dataset],"--seed",str(seed),"--split-seed",str(split_seed),"--epochs",str(epochs),"--output-dir",outdir,"--model-dir",modeldir])
            log_dir=ROOT/"dashboard_runs"; log_dir.mkdir(exist_ok=True)
            run_id = uuid.uuid4().hex[:12]
            started_iso = datetime.now().astimezone().isoformat(timespec="seconds")
            started_epoch = time.time()
            log_path=log_dir/f"run_{time.strftime('%Y%m%d_%H%M%S')}_{run_id}.log"
            history = read_history()
            history.append({
                "run_id": run_id, "started_at": started_iso, "finished_at": None,
                "status": "running", "returncode": None, "dataset": dataset,
                "model": model, "training_seed": seed, "split_seed": split_seed,
                "epochs_requested": epochs, "notes": str(cfg.get("notes", "")),
                "metrics": [], "log_file": str(log_path.relative_to(ROOT)),
                "checkpoint_paths": [], "config": cfg
            })
            write_history(history)
            active.update({"process":None,"started_at":started_iso,"config":cfg,"log_path":str(log_path),"command":commands,"last_returncode":None,"running_flag":True,"child_process":None,"run_id":run_id})

            def worker():
                code=0
                collected=[]
                checkpoints=[]
                with open(log_path,"w",buffering=1) as log:
                    for model_name, (module, outdir, modeldir), cmd in zip([x[0] for x in chosen], [x[1] for x in chosen], commands):
                        log.write("\n$ "+" ".join(cmd)+"\n\n")
                        log.flush()
                        proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,text=True,env={**os.environ,"PYTHONUNBUFFERED":"1"})
                        with lock:
                            active["child_process"]=proc
                        code=proc.wait()
                        metric=find_run_metric(model_name, dataset, seed, started_epoch)
                        if metric:
                            collected.append(metric)
                        ckpt_dir=ROOT/modeldir
                        if ckpt_dir.exists():
                            candidates=sorted(ckpt_dir.glob("*"),key=lambda p:p.stat().st_mtime,reverse=True)
                            for candidate in candidates[:30]:
                                if candidate.is_file() and candidate.stat().st_mtime >= started_epoch - 2 and candidate.suffix in (".pt", ".pth"):
                                    rel=str(candidate.relative_to(ROOT))
                                    if rel not in checkpoints: checkpoints.append(rel)
                        if code!=0:
                            log.write(f"\nProcess failed with exit code {code}; stopping this run.\n")
                            break
                    log.write(f"\nRun finished with exit code {code}.\n")
                finished_iso = datetime.now().astimezone().isoformat(timespec="seconds")
                with lock:
                    active["last_returncode"]=code
                    active["child_process"]=None
                    active["running_flag"]=False
                hist=read_history()
                for item in hist:
                    if item.get("run_id")==run_id:
                        item.update({"finished_at":finished_iso,"status":"completed" if code==0 else "failed","returncode":code,"metrics":collected,"checkpoint_paths":checkpoints})
                        break
                write_history(hist)
            t=threading.Thread(target=worker,daemon=True)
            t.start()
        return self.send_json({"accepted":True,"run_id":run_id,"message":"Experiment started on this Mac.","log_file":str(log_path.relative_to(ROOT)),"commands":commands},202)

if __name__=="__main__":
    os.chdir(ROOT)
    host="127.0.0.1"; port=int(os.environ.get("WAVELAB_PORT","8765"))
    print(f"WaveLab running at http://{host}:{port}")
    print("Press Ctrl+C to stop the local server.")
    ThreadingHTTPServer((host,port),Handler).serve_forever()
