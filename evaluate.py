"""Accuracy evaluation of all five models on the held-out test videos.

For each model: predict on val -> pick the confidence threshold that maximises F1 (val only) ->
predict on test -> compute every metric in metrics.evaluate at that threshold.
Writes results/accuracy.json plus ROC and precision-recall plots.

Run with the .venv-dfine interpreter (it can import both ultralytics and transformers).
"""
import sklearn.metrics  # noqa: F401  (import before transformers, see README)
import json, sys
from pathlib import Path
import cv2, numpy as np, torch

sys.path.insert(0, str(Path(__file__).parent))
from metrics import load_gt, best_f1_conf, evaluate, pr_curve  # noqa: E402
import classical  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "bottle-split"
RES = ROOT / "results"; RES.mkdir(exist_ok=True)
IMGSZ = 320
YOLO_RUNS = {"YOLO26n": "yolo26n", "YOLOv10n": "yolov10n", "YOLO11n": "yolo11n", "YOLOv5nu": "yolov5nu"}


def names(split):
    return sorted(p.name for p in (DATA / split / "images").glob("*.jpg"))


def yolo_predict(run, split):
    from ultralytics import YOLO
    model = YOLO(str(ROOT / "runs" / run / "weights" / "best.pt"))
    out = {}
    for n in names(split):
        r = model.predict(str(DATA / split / "images" / n), imgsz=IMGSZ, conf=0.001, iou=0.7,
                          max_det=100, device="cpu", verbose=False)[0]
        b = r.boxes
        out[n] = np.concatenate([b.xyxy.numpy(), b.conf.numpy()[:, None]], 1) if len(b) else np.zeros((0, 5))
    return out


def dfine_predict(split):
    from transformers import DFineForObjectDetection
    import train_dfine
    model = DFineForObjectDetection.from_pretrained(ROOT / "runs" / "dfine_n" / "best").eval()
    out, ns = {}, names(split)
    for i in range(0, len(ns), 16):
        chunk = ns[i:i + 16]
        for n, r in zip(chunk, train_dfine.predict(model, [cv2.imread(str(DATA / split / "images" / c)) for c in chunk])):
            out[n] = r
    return out


def classical_predict(split):
    cfg = json.loads((ROOT / "runs" / "classical_cfg.json").read_text())
    return classical.run(split, cfg)[0]


def main():
    torch.set_num_threads(8)
    models = {k: (lambda s, r=r: yolo_predict(r, s)) for k, r in YOLO_RUNS.items()}
    models["D-FINE-N"] = dfine_predict
    models["Classical OpenCV"] = classical_predict
    only = sys.argv[1:]
    gt_val = load_gt(DATA / "val" / "labels", names("val"))
    gt_test = load_gt(DATA / "test" / "labels", names("test"))
    path = RES / "accuracy.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for name, fn in models.items():
        if only and name not in only:
            continue
        print("evaluating", name, flush=True)
        conf = best_f1_conf(fn("val"), gt_val)
        test_preds = fn("test")
        m = evaluate(test_preds, gt_test, conf)
        m["_pr"] = pr_curve(test_preds, gt_test)
        results[name] = m
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items() if not k.startswith("_")}, flush=True)
        path.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
