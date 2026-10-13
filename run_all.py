"""Starts one counter.py per camera in cameras.json and restarts any that exit."""
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
cfg = json.loads((HERE / "cameras.json").read_text())
procs = {}

try:
    while True:
        for cam in cfg["cameras"]:
            proc = procs.get(cam["id"])
            if proc is None or proc.poll() is not None:
                if proc is not None:
                    print(f"[{cam['id']}] exited with code {proc.returncode}, restarting")
                procs[cam["id"]] = subprocess.Popen(
                    [sys.executable, str(HERE / "counter.py"), "--cam", cam["id"]], cwd=HERE
                )
        time.sleep(5)
except KeyboardInterrupt:
    print("stopping...")
finally:
    for proc in procs.values():
        proc.terminate()
