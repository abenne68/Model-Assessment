"""
Step 1: environment and model availability check.

Reports OS / Python / PyTorch / CUDA status, then downloads each of the five
COCO-pretrained nano weight files with the ONE installed ultralytics version
and runs a single dummy prediction to prove it loads and runs.

Writes the findings to env_check.json in the project root.
Exit code is non-zero if any model fails, so later steps don't run by accident.
"""
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / "weights"  # pretrained .pt files are downloaded here (gitignored)

# The four models under test (same table as train.py). Never substitute or rename.
# yolo27n.pt was dropped: as of ultralytics==8.4.164 (the newest release, checked
# 2026-09-27) no YOLO27 weights exist on PyPI or in the ultralytics/assets GitHub
# releases, including pre-releases. See env_check.json / README.md for details.
MODELS = {
    "v5": "yolov5nu.pt",
    "v8": "yolov8n.pt",
    "v11": "yolo11n.pt",
    "v26": "yolo26n.pt",
}


def main():
    info = {
        "os": platform.platform(),
        "cpu": platform.processor(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "ultralytics": ultralytics.__version__,
        "models": {},
    }
    for k in ("os", "python", "torch", "cuda_available", "gpu", "ultralytics"):
        print(f"{k:15s}: {info[k]}")

    # Ultralytics downloads a named asset into the current directory,
    # so work inside weights/ to keep the project root clean.
    WEIGHTS_DIR.mkdir(exist_ok=True)
    os.chdir(WEIGHTS_DIR)

    dummy = np.zeros((640, 640, 3), dtype=np.uint8)  # a black test image
    all_ok = True
    for key, weights in MODELS.items():
        entry = {"weights": weights}
        try:
            model = YOLO(weights)  # downloads the file if it isn't here yet
            model.predict(dummy, device="cpu", verbose=False)
            head = model.model.model[-1]  # the detection head module
            # NMS-free ("end-to-end") models are identified by having a second,
            # one-to-one prediction head (one2one_cv2/cv3) alongside the normal
            # one-to-many training head. The head.end2end flag is only set True
            # during export, so it's not a reliable check at load time.
            is_nms_free = getattr(head, "one2one_cv2", None) is not None
            entry.update(
                ok=True,
                params=sum(p.numel() for p in model.model.parameters()),
                head=type(head).__name__,
                nms_free=is_nms_free,
                reg_max=getattr(head, "reg_max", None),  # 1 means no real DFL (YOLO26)
            )
            print(f"[OK]   {key:4s} {weights:14s} params={entry['params']:,} "
                  f"head={entry['head']} nms_free={entry['nms_free']} reg_max={entry['reg_max']}")
        except Exception as e:  # report and keep checking the rest
            all_ok = False
            entry.update(ok=False, error=f"{type(e).__name__}: {e}")
            print(f"[FAIL] {key:4s} {weights:14s} {entry['error']}")
        info["models"][key] = entry

    (ROOT / "env_check.json").write_text(json.dumps(info, indent=2))
    print(f"\nAll {len(MODELS)} models load:", all_ok)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
