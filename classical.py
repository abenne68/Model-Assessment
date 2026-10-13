"""Classical OpenCV baseline: per-scene background model + absdiff + threshold + morphology + contours.

The background for each frame is the per-pixel temporal median of the *other* frames from the same
camera scene (leave-one-out), which is what a running background model (MOG2 / running median)
converges to on a fixed camera. Empty-conveyor frames, whose source video isn't recorded, are grouped
into scenes by k-means on thumbnails. Thresholds are grid-searched on the validation split only.

Processing runs at 320x320 to match the neural models' input size; boxes are scaled back to 640.
Kalman tracking is not evaluated: the dataset is sparse still frames, not continuous video.
"""
import itertools, json, sys, time
from pathlib import Path
import cv2, numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).parent))
from metrics import load_gt, average_precision  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "bottle-split"
S = 320  # working resolution
GRID = {"thr": [15, 20, 25, 35, 45], "close_h": [15, 41, 81], "close_w": [3, 9], "min_area": [400, 1200, 3000]}


def scenes_for(split):
    man = json.loads((DATA / "manifest.json").read_text())[split]
    frames = {m["file"]: cv2.resize(cv2.imread(str(DATA / split / "images" / m["file"])), (S, S)) for m in man}
    scene = {m["file"]: m["video"] for m in man}
    neg = [f for f in scene if scene[f].startswith("neg_")]
    if neg:  # cluster empty-conveyor frames into camera scenes
        thumbs = np.stack([cv2.resize(cv2.cvtColor(frames[f], cv2.COLOR_BGR2GRAY), (32, 32)).ravel() for f in neg])
        lab = KMeans(n_clusters=min(3, len(neg)), n_init=10, random_state=0).fit_predict(thumbs.astype(np.float32))
        for f, l in zip(neg, lab):
            scene[f] = f"negcluster_{l}"
    return frames, scene


def backgrounds(frames, scene):
    """Leave-one-out temporal median per scene."""
    by_scene = {}
    for f, s in scene.items():
        by_scene.setdefault(s, []).append(f)
    bg = {}
    for s, fs in by_scene.items():
        stack = np.stack([frames[f] for f in fs])
        for i, f in enumerate(fs):
            others = np.delete(stack, i, axis=0) if len(fs) > 1 else stack
            bg[f] = np.median(others, axis=0).astype(np.uint8)
    return bg


def detect(frame, bg, thr, close_h, close_w, min_area):
    """One frame (already SxS) -> (K,5) xyxy+score boxes in 640-px coordinates."""
    d = cv2.absdiff(frame, bg).max(axis=2)
    d = cv2.GaussianBlur(d, (5, 5), 0)
    mask = (d > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_h, close_w), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if w * h < min_area:
            continue
        region = d[y:y + h, x:x + w][lab[y:y + h, x:x + w] == k]
        score = min(1.0, float(region.mean()) / 100.0) * np.sqrt(area / (w * h))
        out.append([x * 640 / S, y * 640 / S, (x + w) * 640 / S, (y + h) * 640 / S, score])
    return np.array(out, np.float32).reshape(-1, 5)


def detect_columns(frame, bg, thr, col_thr, row_thr, min_w):
    """Column-projection ("lane") segmentation: upright products on a conveyor occupy vertical bands.

    Columns whose upper 80% is mostly foreground are product; the moving belt in the bottom rows
    can't create a band on its own. Each run of such columns becomes one box.
    """
    d = cv2.GaussianBlur(cv2.absdiff(frame, bg).max(axis=2), (5, 5), 0)
    mask = cv2.morphologyEx((d > thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    col = np.convolve(mask[: int(0.8 * S)].mean(0), np.ones(5) / 5, mode="same")
    on = np.r_[False, col > col_thr, False]
    edges = np.flatnonzero(on[1:] != on[:-1]).reshape(-1, 2)
    out = []
    for x0, x1 in edges:
        if x1 - x0 < min_w:
            continue
        rows = np.flatnonzero(mask[:, x0:x1].mean(1) > row_thr)
        if len(rows) == 0:
            continue
        y0, y1 = rows[0], rows[-1] + 1
        out.append([x0 * 640 / S, y0 * 640 / S, x1 * 640 / S, y1 * 640 / S, float(col[x0:x1].mean())])
    return np.array(out, np.float32).reshape(-1, 5)


COL_GRID = {"thr": [6, 9, 12, 15], "col_thr": [0.25, 0.35, 0.45, 0.55, 0.65], "row_thr": [0.1, 0.2, 0.3], "min_w": [10, 20, 30]}
DETECTORS = {"blob": (detect, GRID), "columns": (detect_columns, COL_GRID)}


def run(split, cfg):
    frames, scene = scenes_for(split)
    bg = backgrounds(frames, scene)
    fn = DETECTORS[cfg["mode"]][0]
    return {f: fn(frames[f], bg[f], **cfg["params"]) for f in frames}, frames, bg


def tune():
    frames, scene = scenes_for("val")
    bg = backgrounds(frames, scene)
    gt = load_gt(DATA / "val" / "labels", list(frames))
    best = (-1, None)
    for mode, (fn, grid) in DETECTORS.items():
        for vals in itertools.product(*grid.values()):
            p = dict(zip(grid, vals))
            ap = average_precision({f: fn(frames[f], bg[f], **p) for f in frames}, gt, 0.5)
            if ap > best[0]:
                best = (ap, {"mode": mode, "params": p})
    print("best val AP50", round(best[0], 4), best[1])
    (ROOT / "runs" / "classical_cfg.json").write_text(json.dumps(best[1]))
    return best[1]


def fps(frames, bg, cfg, seconds=5.0):
    """Frames/s of the whole classical pipeline (resize + diff + morphology + segmentation)."""
    fn = DETECTORS[cfg["mode"]][0]
    names = list(frames); raw = {f: cv2.resize(frames[f], (640, 640)) for f in names}
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        f = names[n % len(names)]
        fn(cv2.resize(raw[f], (S, S)), bg[f], **cfg["params"]); n += 1
    return n / (time.perf_counter() - t0)


if __name__ == "__main__":
    print(tune())
