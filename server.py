#!/usr/bin/env python3
"""Local WaveLab dashboard server and experiment runner (standard library only)."""
from __future__ import annotations
import csv, json, os, subprocess, sys, threading, time, uuid, math
from datetime import datetime
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

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

def read_log(path_value=None):
    """Read the active or a recorded run log, safely constrained to this project."""
    value = path_value or active.get("log_path")
    if not value:
        return ""
    try:
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        path = path.resolve()
        if path != ROOT and ROOT not in path.parents:
            return ""
        if not path.is_file():
            return ""
        return path.read_text(errors="replace")[-18000:]
    except (OSError, ValueError, RuntimeError):
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


ARTIFACT_FOLDERS={"Coordinate MLP":("grouped","coordinate_mlp_grouped"),"Spatial-feature MLP":("spatial","spatial_mlp"),"GNN":("gnn","gnn")}
SEEDS=(21,42,123)
def artifact_stem(dataset,model,seed):
    loc,count=dataset.split(); folder,suffix=ARTIFACT_FOLDERS[model]
    return folder,f"WEC_{loc}_{count}_{suffix}_seed{int(seed)}"
def read_csv_rows(path):
    with path.open(newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def compare_close(a,b,absolute=0.05,relative=2e-5):
    try:return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b))<=max(absolute,relative*max(abs(float(a)),abs(float(b))))
    except (TypeError,ValueError):return False
def inspect_run(dataset,model,seed,with_rows=False):
    folder,stem=artifact_stem(dataset,model,seed); base=ROOT/"results"/folder
    mp,hp,pp=[base/(stem+suffix) for suffix in ("_metrics.json","_history.csv","_predictions.csv")]
    missing=[str(p.relative_to(ROOT)) for p in (mp,hp,pp) if not p.is_file()]
    if missing:return {"ok":False,"missing":missing}
    metric=json.loads(mp.read_text()); hist=read_csv_rows(hp); pred=read_csv_rows(pp)
    if not hist or not {"epoch","train_mse_scaled","validation_mae"}.issubset(hist[0]):raise ValueError("Invalid training-history CSV schema")
    if not pred or not {"actual_total_power","predicted_total_power"}.issubset(pred[0]):raise ValueError("Invalid prediction CSV schema")
    y=[float(x["actual_total_power"]) for x in pred]; p=[float(x["predicted_total_power"]) for x in pred]
    if not all(math.isfinite(v) for v in y+p):raise ValueError("Prediction file contains non-finite values")
    n=len(y); residual=[a-b for a,b in zip(y,p)]; mae=sum(abs(v) for v in residual)/n; mse=sum(v*v for v in residual)/n; mean_y=sum(y)/n; ss=sum((v-mean_y)**2 for v in y); r2=1-sum(v*v for v in residual)/ss if ss else None; mean_abs=sum(abs(v) for v in y)/n; nmae=mae/mean_abs if mean_abs>1e-12 else None
    checks={"MAE":compare_close(metric.get("test_mae"),mae),"RMSE":compare_close(metric.get("test_rmse"),mse**0.5),"NMAE":compare_close(metric.get("test_nmae"),nmae,1e-8,2e-5),"R2":compare_close(metric.get("test_r2"),r2,2e-5,2e-5)}
    out={"ok":all(checks.values()),"comparisons":checks,"history_count":len(hist),"prediction_count":n,"metric_file":str(mp.relative_to(ROOT)),"history_file":str(hp.relative_to(ROOT)),"predictions_file":str(pp.relative_to(ROOT)),"split_seed":metric.get("split_seed"),"split_method":metric.get("split_method")}
    if with_rows:
        bins=25; lo=min(residual); hi=max(residual); span=hi-lo or 1.0; counts=[0]*bins
        for v in residual:counts[min(bins-1,int((v-lo)/span*bins))]+=1
        out["history"]=[{"epoch":int(float(r["epoch"])),"train_mse_scaled":float(r["train_mse_scaled"]),"validation_mae":float(r["validation_mae"])} for r in hist]
        out["residual_summary"]={"count":n,"mae":mae,"rmse":mse**0.5,"nmae":nmae,"r2":r2,"mean_residual":sum(residual)/n}
        out["residual_histogram"]=[{"count":c,"label":f"{lo+span*i/bins:.0f}" if i in (0,bins-1) else f"{lo+span*i/bins:.0f}–{lo+span*(i+1)/bins:.0f}"} for i,c in enumerate(counts)]
    return out
def dashboard_health():
    rows=metrics_files(); valid=[r for r in rows if r.get("dataset_label") in DATASETS and r.get("model_label") in ARTIFACT_FOLDERS and int(r.get("seed",-1)) in SEEDS and Path(str(r.get("result_file",""))).name.endswith(f"_seed{int(r.get('seed',-1))}_metrics.json")]
    unique={(r["dataset_label"],r["model_label"],int(r["seed"])):r for r in valid}; expected=[(d,m,s) for d in DATASETS for m in ARTIFACT_FOLDERS for s in SEEDS]
    history_count=prediction_count=consistent=0; split_ok=len(unique)==36 and all(int(r.get("split_seed",-1))==42 for r in unique.values()); grouped_ok=len(unique)==36 and all(r.get("split_method")=="layout_grouped" for r in unique.values()); missing=[]
    for d,m,s in expected:
        try:
            a=inspect_run(d,m,s)
            if a.get("history_count"):history_count+=1
            if a.get("prediction_count"):prediction_count+=1
            if a.get("ok"):consistent+=1
            else:missing.append({"dataset":d,"model":m,"seed":s,"detail":a.get("missing",a.get("comparisons"))})
        except Exception as e:missing.append({"dataset":d,"model":m,"seed":s,"detail":str(e)})
    sources=[ROOT/"src"/x for x in ("train_mlp_grouped.py","train_spatial_mlp.py","train_gnn.py")]
    data_missing=[v for v in DATASETS.values() if not (ROOT/v).is_file()]
    loader=ROOT/"src"/"data_utils.py"
    loader_text=loader.read_text(errors="ignore") if loader.exists() else ""
    leakage_verified=all(p.exists() for p in sources) and loader.exists() and "df[coord_cols]" in loader_text and "find_coordinate_columns" in loader_text and all("load_wave_data" in p.read_text(errors="ignore") for p in sources)
    checks={"metrics":{"ok":len(unique)==36,"detail":f"{len(unique)}/36 unique seeded metric records."},"split":{"ok":split_ok,"detail":"All 36 records use split seed 42." if split_ok else "Missing records or non-42 split seed."},"grouped":{"ok":grouped_ok,"detail":"All 36 records declare layout_grouped." if grouped_ok else "Some records lack layout_grouped metadata."},"leakage":{"ok":not data_missing and leakage_verified,"detail":"All four raw datasets exist; shared loader selects coordinate columns as model inputs and the three training scripts use that loader." if not data_missing and leakage_verified else "Could not verify the coordinate-only input policy or a raw dataset is missing."},"histories":{"ok":history_count==36,"detail":f"Matching history files: {history_count}/36."},"predictions":{"ok":prediction_count==36 and consistent==36,"detail":f"Prediction files: {prediction_count}/36; all four metrics reproduced: {consistent}/36."},"backend":{"ok":True,"detail":"Local health endpoint responded successfully."},"nmae":{"ok":consistent==36,"detail":f"NMAE reproduced for {consistent}/36 runs as MAE / mean(abs(actual held-out Total_Power))."}}
    return {"overall_ok":all(v["ok"] for v in checks.values()),"checks":checks,"metrics_count":len(unique),"histories_count":history_count,"predictions_count":prediction_count,"metric_consistency_count":consistent,"missing_or_mismatched":missing[:15]}

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
                active_log=active.get("log_path")
                active_run_id=active.get("run_id")
                active_config=active.get("config")
                active_started=active.get("started_at")
                active_command=active.get("command")
                active_code=active.get("last_returncode")
            history=read_history()
            latest=next((item for item in reversed(history) if isinstance(item,dict)),None)
            # After a server restart, in-memory process state is empty. Recover the
            # most recent completed run and its log from persistent run history.
            if running:
                last_run=next((item for item in reversed(history)
                               if isinstance(item,dict) and item.get("run_id")==active_run_id),latest)
                log_text=read_log(active_log)
            else:
                last_run=latest
                log_text=read_log((last_run or {}).get("log_file"))
            if running:
                status_code=None
            elif last_run and last_run.get("returncode") is not None:
                status_code=last_run.get("returncode")
            else:
                status_code=active_code
            payload={
                "connected":True,"running":running,
                "started_at":active_started if running else (last_run or {}).get("started_at"),
                "config":active_config if running else ((last_run or {}).get("config")),
                "command":active_command if running else None,
                "run_id":active_run_id if running else (last_run or {}).get("run_id"),
                "returncode":status_code,
                "last_run":last_run,
                "log":log_text,
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
        if path=="/api/health":
            try:return self.send_json(dashboard_health())
            except Exception as exc:return self.send_json({"overall_ok":False,"error":str(exc)},500)
        if path=="/api/artifacts":
            q=parse_qs(urlparse(self.path).query); dataset=(q.get("dataset") or [""])[0]; model=(q.get("model") or [""])[0]
            try:seed=int((q.get("seed") or [""])[0])
            except ValueError:return self.send_json({"error":"Seed must be 21, 42 or 123."},400)
            if dataset not in DATASETS or model not in ARTIFACT_FOLDERS or seed not in SEEDS:return self.send_json({"error":"Unsupported dataset, model or seed."},400)
            try:
                result=inspect_run(dataset,model,seed,True)
                if not result.get("ok"):return self.send_json({"verified":False,"message":"Artifacts missing or saved predictions do not reproduce the metric JSON.",**result},409)
                return self.send_json({"verified":True,**result})
            except Exception as exc:return self.send_json({"verified":False,"error":str(exc)},500)
        return super().do_GET()
    def do_POST(self):
        request_path = urlparse(self.path).path
        if request_path == "/api/predict":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_000_000:
                    return self.send_json({"error": "Request body is empty or too large."}, 400)
                payload = json.loads(self.rfile.read(length))
                from src.live_predict import predict_total_power
                result = predict_total_power(
                    dataset=payload.get("dataset"),
                    model_name=payload.get("model"),
                    seed=int(payload.get("seed", 21)),
                    coordinates=payload.get("coordinates"),
                )
                return self.send_json({"ok": True, **result})
            except (ValueError, TypeError, KeyError, FileNotFoundError) as exc:
                return self.send_json({"ok": False, "error": str(exc)}, 400)
            except Exception as exc:
                return self.send_json({"ok": False, "error": f"Prediction failed: {exc}"}, 500)

        if request_path!="/api/experiments":
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
