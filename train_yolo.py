"""Fine-tune the three Ultralytics nano detectors on the bottle split (CPU, identical settings)."""
import os, sys
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "bottle-split" / "data.yaml"
MODELS = {"yolo26n": "yolo26n.pt", "yolov10n": "yolov10n.pt", "yolo11n": "yolo11n.pt", "yolov5nu": "yolov5nu.pt"}
IMGSZ, EPOCHS = 320, 25

if __name__ == "__main__":
    os.chdir(ROOT / "weights")  # pretrained weights live / download here
    for name in (sys.argv[1:] or MODELS):
        YOLO(MODELS[name]).train(
            data=str(DATA), imgsz=IMGSZ, epochs=EPOCHS, batch=16, device="cpu", workers=4,
            seed=0, deterministic=True, patience=10, cache="ram", plots=False,
            project=str(ROOT / "runs"), name=name, exist_ok=True, verbose=False)
