"""Why do some models double count? Break down false positives and per-frame count errors.

At each model's operating threshold (chosen on val), every false positive is classed as:
  duplicate   - IoU >= 0.3 with a ground-truth bottle that another prediction already matched
  mislocated  - IoU in [0.1, 0.5) with a still-unmatched bottle (box too loose / shifted)
  background  - overlaps no bottle
Count error per frame = predicted boxes - true bottles (positive = over-count).
For the NMS models, also re-runs prediction with a stricter NMS IoU to see if duplicates go away.
Run with the .venv-dfine interpreter.
"""
import sklearn.metrics  # noqa: F401
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from metrics import load_gt, iou_matrix, match  # noqa: E402
import evaluate as E  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
acc = json.loads((ROOT / "results" / "accuracy.json").read_text())


def analyse(preds, gts, conf):
    dup = misloc = bg = 0; over = under = exact = 0; extra = []
    for n, gt in gts.items():
        p = preds[n]; p = p[p[:, 4] >= conf]
        m = match(p, gt, 0.5)
        p = p[np.argsort(-p[:, 4])]
        ious = iou_matrix(p[:, :4], gt)
        tp_gt = set()
        for i, (s, ok, _) in enumerate(m):  # recover which GT each TP took
            if ok:
                cand = [k for k in range(len(gt)) if k not in tp_gt and ious[i, k] >= 0.5]
                tp_gt.add(max(cand, key=lambda k: ious[i, k]))
        for i, (s, ok, _) in enumerate(m):
            if ok:
                continue
            row = ious[i] if len(gt) else np.zeros(0)
            if len(gt) and any(row[k] >= 0.3 for k in tp_gt):
                dup += 1
            elif len(gt) and row.max() >= 0.1:
                misloc += 1
            else:
                bg += 1
        d = len(p) - len(gt); extra.append(d)
        over += d > 0; under += d < 0; exact += d == 0
    nimg = len(gts)
    return {"FP_duplicate": dup, "FP_mislocated": misloc, "FP_background": bg,
            "frames_overcounted_pct": 100 * over / nimg, "frames_undercounted_pct": 100 * under / nimg,
            "frames_exact_pct": 100 * exact / nimg, "net_count_error": int(sum(extra)),
            "abs_count_error_per_frame": float(np.mean(np.abs(extra)))}


def main():
    gts = load_gt(E.DATA / "test" / "labels", E.names("test"))
    fns = {k: (lambda s, r=r: E.yolo_predict(r, s)) for k, r in E.YOLO_RUNS.items()}
    fns["D-FINE-N"] = E.dfine_predict
    fns["Classical OpenCV"] = E.classical_predict
    out = {}
    for name, fn in fns.items():
        out[name] = analyse(fn("test"), gts, acc[name]["conf_threshold"])
        print(name, {k: round(v, 2) for k, v in out[name].items()}, flush=True)
    # NMS sensitivity for the two NMS-based YOLOs (default eval used iou=0.7)
    from ultralytics import YOLO
    for name, run in (("YOLOv5nu", "yolov5nu"), ("YOLO11n", "yolo11n")):
        model = YOLO(str(ROOT / "runs" / run / "weights" / "best.pt"))
        for nms_iou in (0.5, 0.45):
            preds = {}
            for n in E.names("test"):
                b = model.predict(str(E.DATA / "test" / "images" / n), imgsz=E.IMGSZ, conf=0.001, iou=nms_iou,
                                  max_det=100, device="cpu", verbose=False)[0].boxes
                preds[n] = np.concatenate([b.xyxy.numpy(), b.conf.numpy()[:, None]], 1) if len(b) else np.zeros((0, 5))
            key = f"{name} (NMS IoU {nms_iou})"
            out[key] = analyse(preds, gts, acc[name]["conf_threshold"])
            print(key, {k: round(v, 2) for k, v in out[key].items()}, flush=True)
    (ROOT / "results" / "double_count.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
