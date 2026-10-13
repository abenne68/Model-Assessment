"""
Run this yourself to fix the missing test split on the Roboflow project:
  1. Rebalances tpicbxaiproject/bottle-detection-conveyor to 70/20/10 train/valid/test.
  2. Generates a new version with no Roboflow-side augmentation (auto-orient only).
Prints the new version number at the end -- tell me that number to continue.
"""
import os, sys, time
import requests
from roboflow import Roboflow
from roboflow.config import API_URL

WORKSPACE = "tpicbxaiproject"
PROJECT = "bottle-detection-conveyor"
key = os.environ.get("ROBOFLOW_API_KEY") or sys.exit("Set ROBOFLOW_API_KEY first.")

print("Rebalancing to 70/20/10 ...")
r = requests.post(
    f"{API_URL}/{WORKSPACE}/{PROJECT}/splits/rebalance?api_key={key}",
    json={"splits": {"train": 70, "valid": 20, "test": 10}, "mode": "rebalance"},
)
r.raise_for_status()
task = r.json()
poll_url = task.get("url") or f"{API_URL}/{WORKSPACE}/asynctasks/{task['taskId']}"
if "api_key" not in poll_url:
    poll_url += ("&" if "?" in poll_url else "?") + f"api_key={key}"

print("Polling task", task.get("taskId"), "...")
for _ in range(60):
    time.sleep(5)
    pr = requests.get(poll_url)
    status = pr.json().get("status", "unknown") if pr.ok else "unknown"
    print(" status:", status)
    if status in ("completed", "success", "done"):
        break
    if status in ("failed", "error"):
        sys.exit(f"Rebalance failed: {pr.text}")
else:
    sys.exit("Timed out waiting for rebalance.")

print("Generating new version (auto-orient only, no augmentation) ...")
rf = Roboflow(api_key=key)
project = rf.workspace(WORKSPACE).project(PROJECT)
new_version = project.generate_version(
    settings={"preprocessing": {"auto-orient": True}, "augmentation": {}}
)
print("NEW VERSION NUMBER:", new_version)