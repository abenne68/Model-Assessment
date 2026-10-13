"""Throughput of all five models on this machine's CPU (no GPU available).

Neural models are exported to ONNX and run with ONNX Runtime (the usual CPU/edge deployment path):
  * max_throughput: best images/s over batch sizes 1..32, inference only
  * e2e_fps_b1: batch-1 frames/s for the full pipeline on an in-memory 640x640 frame
                (resize + normalise + inference + postprocess/NMS), i.e. a live camera loop
The classical pipeline is single-frame only, so both numbers are the same loop.
Run alone on an otherwise idle machine, with the .venv-dfine interpreter.
"""
import sklearn.metrics  # noqa: F401
import json, platform, sys, time
from pathlib import Path
import cv2, numpy as np, onnxruntime as ort, torch

sys.path.insert(0, str(Path(__file__).parent))
import classical  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "bottle-split"
RES = ROOT / "results"
IMGSZ, BATCHES, SECONDS = 320, [1, 2, 4, 8, 16, 32], 4.0
YOLO_RUNS = {"YOLO26n": "yolo26n", "YOLOv10n": "yolov10n", "YOLO11n": "yolo11n", "YOLOv5nu": "yolov5nu"}


def export_yolo(run):
    from ultralytics import YOLO
    onnx = ROOT / "runs" / run / "weights" / "best.onnx"
    if not onnx.exists():
        YOLO(str(onnx.with_suffix(".pt"))).export(format="onnx", imgsz=IMGSZ, dynamic=True, simplify=True)
    return onnx


def export_dfine():
    from transformers import DFineForObjectDetection
    onnx = ROOT / "runs" / "dfine_n" / "best.onnx"
    if not onnx.exists():
        model = DFineForObjectDetection.from_pretrained(ROOT / "runs" / "dfine_n" / "best").eval()

        class Wrap(torch.nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x):
                o = self.m(pixel_values=x); return o.logits, o.pred_boxes

        torch.onnx.export(Wrap(model), torch.rand(1, 3, IMGSZ, IMGSZ), str(onnx), opset_version=17,
                          input_names=["images"], output_names=["logits", "boxes"],
                          dynamic_axes={"images": {0: "b"}, "logits": {0: "b"}, "boxes": {0: "b"}}, dynamo=False)
    return onnx


def session(onnx):
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(onnx), so, providers=["CPUExecutionProvider"])


def timed(fn, seconds=SECONDS, trials=3):
    """Calls/s, median of `trials` timed windows (laptop turbo/thermals make single runs noisy)."""
    for _ in range(3):
        fn()
    rates = []
    for _ in range(trials):
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            fn(); n += 1
        rates.append(n / (time.perf_counter() - t0))
    return float(np.median(rates))


def preprocess(bgr):
    rgb = cv2.cvtColor(cv2.resize(bgr, (IMGSZ, IMGSZ)), cv2.COLOR_BGR2RGB)
    return (rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)


def post_yolo_nms(out, conf, scale):  # YOLO11 / YOLOv5nu: (1, 5, N) cx,cy,w,h,score -> NMS
    p = out[0][0].T; p = p[p[:, 4] >= conf]
    xywh = np.c_[p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2, p[:, 2], p[:, 3]]
    keep = cv2.dnn.NMSBoxes(xywh.tolist(), p[:, 4].tolist(), conf, 0.7)
    return xywh[np.array(keep).ravel().astype(int)] * scale if len(keep) else np.zeros((0, 4))


def post_end2end(out, conf, scale):  # YOLO26 / YOLOv10: (1, 300, 6) x1,y1,x2,y2,score,cls
    p = out[0][0]; return p[p[:, 4] >= conf, :4] * scale


def post_dfine(out, conf, scale):  # (1, Q, 1) logits, (1, Q, 4) normalised cxcywh
    s = 1 / (1 + np.exp(-out[0][0, :, 0])); keep = s >= conf
    return out[1][0][keep] * 640


def main():
    acc = json.loads((RES / "accuracy.json").read_text())
    frame = cv2.imread(str(next((DATA / "test" / "images").glob("*.jpg"))))
    models = {k: (export_yolo(r), post_yolo_nms if k in ("YOLO11n", "YOLOv5nu") else post_end2end) for k, r in YOLO_RUNS.items()}
    models["D-FINE-N"] = (export_dfine(), post_dfine)
    results = {"machine": {"cpu": "AMD Ryzen 7 170 (8 cores / 16 threads)", "gpu": None,
                           "onnxruntime": ort.__version__, "os": platform.platform(), "input": f"{IMGSZ}x{IMGSZ}"}}
    for name, (onnx, post) in models.items():
        sess = session(onnx); inp = sess.get_inputs()[0].name
        per_batch = {}
        for b in BATCHES:
            x = np.random.rand(b, 3, IMGSZ, IMGSZ).astype(np.float32)
            per_batch[b] = timed(lambda: sess.run(None, {inp: x})) * b
        conf = acc[name]["conf_threshold"]
        e2e = timed(lambda: post(sess.run(None, {inp: preprocess(frame)}), conf, 640 / IMGSZ))
        best_b = max(per_batch, key=per_batch.get)
        results[name] = {"max_throughput_ips": per_batch[best_b], "best_batch": best_b,
                         "b1_inference_fps": per_batch[1], "e2e_fps_b1": e2e,
                         "per_batch_ips": per_batch, "onnx_mb": onnx.stat().st_size / 1e6}
        print(name, {k: round(v, 1) if isinstance(v, float) else v for k, v in results[name].items()}, flush=True)
    cfg = json.loads((ROOT / "runs" / "classical_cfg.json").read_text())
    frames, scene = classical.scenes_for("test"); bg = classical.backgrounds(frames, scene)
    f = float(np.median([classical.fps(frames, bg, cfg, SECONDS) for _ in range(3)]))
    results["Classical OpenCV"] = {"max_throughput_ips": f, "best_batch": 1, "b1_inference_fps": f, "e2e_fps_b1": f}
    print("Classical OpenCV", round(f, 1))
    (RES / "throughput.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
