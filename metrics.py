"""Detection metrics shared by every model in the benchmark.

All boxes are xyxy in original-image pixels. A prediction is (x1, y1, x2, y2, score).
Matching is greedy by descending score, one prediction per ground-truth box (COCO style).
"""
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def load_gt(label_dir: Path, names, w=640, h=640):
    """YOLO txt labels -> {name: (N,4) xyxy pixel array}."""
    gt = {}
    for n in names:
        rows = [l.split() for l in open(label_dir / (Path(n).stem + ".txt")) if l.strip()]
        a = np.array([[float(v) for v in r[1:5]] for r in rows], dtype=np.float32).reshape(-1, 4)
        cx, cy, bw, bh = a.T
        gt[n] = np.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], 1)
    return gt


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def match(pred, gt, thr):
    """Return per-prediction (score, is_tp, iou_with_matched_gt), sorted by score desc."""
    pred = pred[np.argsort(-pred[:, 4])] if len(pred) else pred.reshape(0, 5)
    ious = iou_matrix(pred[:, :4], gt)
    used = np.zeros(len(gt), bool); out = []
    for i in range(len(pred)):
        j, best = -1, thr
        for k in range(len(gt)):
            if not used[k] and ious[i, k] >= best:
                j, best = k, ious[i, k]
        if j >= 0:
            used[j] = True
        out.append((pred[i, 4], j >= 0, ious[i, j] if j >= 0 else 0.0))
    return out


def average_precision(preds, gts, thr):
    """COCO 101-point interpolated AP at one IoU threshold."""
    recs = [m for n in gts for m in match(preds[n], gts[n], thr)]
    npos = sum(len(g) for g in gts.values())
    if not recs:
        return 0.0
    recs.sort(key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in recs]); fp = np.cumsum([not r[1] for r in recs])
    rec = tp / max(npos, 1); prec = tp / (tp + fp)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.mean([prec[rec >= t].max() if (rec >= t).any() else 0 for t in np.linspace(0, 1, 101)]))


def counts_at(preds, gts, conf, thr=0.5):
    tp = fp = fn = 0; ious = []
    for n in gts:
        p = preds[n]; p = p[p[:, 4] >= conf]
        m = match(p, gts[n], thr)
        t = sum(r[1] for r in m); tp += t; fp += len(m) - t; fn += len(gts[n]) - t
        ious += [r[2] for r in m if r[1]]
    return tp, fp, fn, ious


def best_f1_conf(preds, gts):
    """Confidence threshold that maximises F1 @ IoU 0.5 (chosen on the validation set)."""
    best = (-1, 0.25)
    for c in np.unique(np.round(np.concatenate([p[:, 4] for p in preds.values() if len(p)] + [[0.5]]), 3)):
        tp, fp, fn, _ = counts_at(preds, gts, c)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best[0]:
            best = (f1, float(c))
    return best[1]


def evaluate(preds, gts, conf):
    """Full metric set at a fixed operating threshold `conf`."""
    tp, fp, fn, ious = counts_at(preds, gts, conf)
    # image level: "is there at least one bottle in this frame?" (the only view with true negatives)
    y = np.array([len(gts[n]) > 0 for n in gts], int)
    s = np.array([preds[n][:, 4].max() if len(preds[n]) else 0.0 for n in gts])
    yhat = s >= conf
    fpr, tpr, _ = roc_curve(y, s)
    return {
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "detection_accuracy": tp / max(tp + fp + fn, 1),  # no true negatives exist at box level
        "image_accuracy": float((yhat == y).mean()),
        "auc_roc_image": float(roc_auc_score(y, s)),
        "mean_iou_tp": float(np.mean(ious)) if ious else 0.0,
        "AP50": average_precision(preds, gts, 0.5),
        "AP50_95": float(np.mean([average_precision(preds, gts, t) for t in np.arange(0.5, 0.96, 0.05)])),
        "TP": tp, "FP": fp, "FN": fn, "conf_threshold": conf,
        "_roc": (fpr.tolist(), tpr.tolist()),
    }


def pr_curve(preds, gts, thr=0.5):
    recs = sorted((m for n in gts for m in match(preds[n], gts[n], thr)), key=lambda r: -r[0])
    npos = sum(len(g) for g in gts.values())
    tp = np.cumsum([r[1] for r in recs]); fp = np.cumsum([not r[1] for r in recs])
    return (tp / max(npos, 1)).tolist(), (tp / np.maximum(tp + fp, 1)).tolist()
