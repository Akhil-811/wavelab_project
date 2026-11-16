#!/usr/bin/env python3
"""Exercise WaveLab's local API and evidence consistency checks."""
import json, sys, urllib.parse, urllib.request
BASE = "http://127.0.0.1:8765"
def get_json(path):
    with urllib.request.urlopen(BASE + path, timeout=20) as response:
        if response.status != 200: raise RuntimeError(f"{path}: HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))
try:
    status=get_json("/api/status"); metrics=get_json("/api/metrics"); runs=get_json("/api/runs"); health=get_json("/api/health")
    query=urllib.parse.urlencode({"dataset":"Perth 49","model":"Coordinate MLP","seed":21})
    artifacts=get_json("/api/artifacts?"+query)
except Exception as exc:
    print("[FAIL] Local API check:",exc); print("Start WaveLab with: python server.py"); sys.exit(2)
checks={"status endpoint":bool(status.get("connected")),"metrics endpoint":isinstance(metrics,list) and len(metrics)==36,"run-history endpoint":isinstance(runs,list),"selected artifact endpoint":bool(artifacts.get("verified")),"36-run evidence audit":bool(health.get("overall_ok"))}
for name,passed in checks.items(): print(("[PASS] " if passed else "[CHECK] ")+name)
for name,item in health.get("checks",{}).items(): print(("[PASS] " if item.get("ok") else "[CHECK] ")+name+": "+item.get("detail",""))
print("Metrics:",health.get("metrics_count"),"/ 36"); print("Histories:",health.get("histories_count"),"/ 36"); print("Predictions:",health.get("predictions_count"),"/ 36"); print("Prediction-to-metric consistency:",health.get("metric_consistency_count"),"/ 36")
sys.exit(0 if all(checks.values()) else 1)
