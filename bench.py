"""Quick speed test of the model on one real frame.

    python bench.py cam3

Needs MediaMTX running with that camera playing.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
import torch
from ultralytics import YOLO

cfg = json.loads(Path(__file__).with_name("cameras.json").read_text())
cam = sys.argv[1] if len(sys.argv) > 1 else "cam3"

print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()} "
      f"| torch threads: {torch.get_num_threads()} | cpu cores: {os.cpu_count()}")

cap = cv2.VideoCapture(f"{cfg['mediamtx_rtsp']}/{cam}")
ok, frame = cap.read()
if not ok:
    raise SystemExit("could not read a frame - is MediaMTX running and is the camera playing?")
print("frame shape:", frame.shape)

model = YOLO(cfg["model"])
model.predict(frame, verbose=False)            # warm-up, not timed

for size in (320, 480, 640, 960):
    start = time.time()
    for _ in range(5):
        model.predict(frame, imgsz=size, classes=[2, 5, 7], conf=0.25, verbose=False)
    print(f"imgsz {size}: {(time.time() - start) / 5 * 1000:.0f} ms per frame")
