"""Per-camera worker.

Reads  rtsp://<mediamtx>/<cam>        (the raw stream, decoded by a separate ffmpeg process)
Detects + tracks vehicles with YOLO, counts line crossings,
Publishes rtsp://<mediamtx>/<cam>_det (the annotated stream, via ffmpeg)
Posts counts to the Ignition Web Dev endpoint.

Run one camera:   python counter.py --cam cam1
Run all cameras:  python run_all.py
"""
import argparse
import json
import os
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path

# Only used to read the stream's size once at start-up.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
import numpy as np
import requests
from ultralytics import YOLO

from linecounter import LineCounter

CONFIG_PATH = Path(__file__).with_name("cameras.json")


class FfmpegFrames:
    """Decodes a stream in a separate ffmpeg process and keeps the newest frame.

    ffmpeg does the decoding, the frame-rate reduction and the downscaling, so no heavy
    video work happens inside this Python process. If the stream drops, it reconnects.
    """

    def __init__(self, url, fps, max_width, ffmpeg="ffmpeg"):
        self.url, self.fps, self.max_width, self.ffmpeg = url, fps, max_width, ffmpeg
        self.frame = None
        self.stamp = 0.0
        self.made = 0            # frames delivered so far (for the status line)
        self._lock = threading.Lock()
        self._new = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _probe_size(self):
        cap = cv2.VideoCapture(self.url)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return w, h

    def _run(self):
        while True:
            w, h = self._probe_size()
            if not w or not h:
                time.sleep(2)                       # stream not up yet
                continue
            out_w = min(w, self.max_width)
            out_w -= out_w % 2
            out_h = int(h * out_w / w)
            out_h -= out_h % 2
            cmd = [
                self.ffmpeg, "-loglevel", "error", "-fflags", "nobuffer", "-flags", "low_delay",
                "-rtsp_transport", "tcp", "-i", self.url, "-an",
                "-vf", f"fps={self.fps},scale={out_w}:{out_h}",
                "-pix_fmt", "bgr24", "-f", "rawvideo", "-",
            ]
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
            except FileNotFoundError:
                print(f"ffmpeg not found at {self.ffmpeg!r} - set \"ffmpeg_path\" in cameras.json")
                time.sleep(30)
                continue
            size = out_w * out_h * 3
            while True:
                buf = proc.stdout.read(size)
                if len(buf) < size:                 # ffmpeg ended: stream dropped
                    break
                frame = np.frombuffer(buf, np.uint8).reshape(out_h, out_w, 3).copy()
                with self._lock:
                    self.frame, self.stamp = frame, time.time()
                    self.made += 1
                self._new.set()
            proc.kill()
            proc.wait()
            time.sleep(2)

    def wait_frame(self, timeout):
        """Blocks until a frame newer than the last one returned arrives. None on timeout."""
        if not self._new.wait(timeout):
            return None
        self._new.clear()
        with self._lock:
            return self.frame


class Publisher:
    """Pushes BGR frames to MediaMTX through an ffmpeg subprocess."""

    def __init__(self, url, width, height, fps, ffmpeg="ffmpeg"):
        cmd = [
            ffmpeg, "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
            "-framerate", str(fps), "-use_wallclock_as_timestamps", "1", "-i", "-",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-profile:v", "baseline",
            "-pix_fmt", "yuv420p", "-g", str(max(fps * 2, 2)),
            "-f", "rtsp", "-rtsp_transport", "tcp", url,
        ]
        self.size = (width, height)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame):
        if self.proc.poll() is not None:
            return False
        try:
            self.proc.stdin.write(frame.tobytes())
            return True
        except (BrokenPipeError, OSError):
            return False

    def close(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        self.proc.terminate()


def even_size(width, height, max_width):
    """Output size: scaled down to max_width, both sides even (needed by yuv420p)."""
    scale = min(1.0, max_width / width)
    w, h = int(width * scale), int(height * scale)
    return w - w % 2, h - h % 2


def draw_overlay(img, line_px, totals, in_view):
    cv2.line(img, line_px[0], line_px[1], (0, 255, 255), 2)
    text = f"In view: {in_view}   In: {totals['CountIn']}   Out: {totals['CountOut']}  (since start)"
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", required=True, help="camera id from cameras.json, e.g. cam1")
    ap.add_argument("--show", action="store_true", help="also open a local debug window")
    ap.add_argument("--no-publish", action="store_true", help="skip the annotated stream (for speed tests)")
    args = ap.parse_args()

    cfg = json.loads(CONFIG_PATH.read_text())
    cam = next((c for c in cfg["cameras"] if c["id"] == args.cam), None)
    if cam is None:
        raise SystemExit(f"{args.cam!r} is not in cameras.json")

    # COCO class id -> tag name; a camera can override the default list with its own "classes"
    tag_for_class = {int(k): v for k, v in cam.get("classes", cfg["classes"]).items()}
    base = cfg["mediamtx_rtsp"].rstrip("/")
    src_url, det_url = f"{base}/{cam['id']}", f"{base}/{cam['id']}_det"
    fps = int(cam.get("process_fps", cfg["process_fps"]))
    ffmpeg = cfg.get("ffmpeg_path", "ffmpeg")
    tag = f"[{cam['id']}]"

    model = YOLO(cfg["model"])
    counter = LineCounter(*cam["line"])
    reader = FfmpegFrames(src_url, fps, cfg.get("analyse_width", 960), ffmpeg)

    pending = defaultdict(int)     # not yet accepted by Ignition (deltas, so restarts lose nothing)
    totals = defaultdict(int)      # since this process started, for the overlay only
    publisher, next_publisher_try = None, 0.0
    last_post, last_log = 0.0, 0.0
    in_view = 0
    frames = made_mark = 0         # frames analysed / delivered since the last log line
    t_track = t_draw = t_post = 0.0    # seconds spent per stage since the last log line

    print(f"{tag} reading {src_url}, publishing {det_url}, posting to {cfg['ignition_url']}")

    while True:
        frame = reader.wait_frame(timeout=3.0)          # sleeps until a new frame actually arrives
        if frame is None:
            if time.time() - last_log > 5:
                print(f"{tag} waiting for stream...")
                last_log = time.time()
            continue

        h, w = frame.shape[:2]
        t0 = time.time()
        result = model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            classes=list(tag_for_class), conf=cam.get("conf", cfg["conf"]),
            imgsz=cam.get("imgsz", cfg["imgsz"]), verbose=False,
        )[0]
        t_track += time.time() - t0

        boxes = result.boxes
        in_view = len(boxes)
        frames += 1
        if boxes.id is not None:
            for tid, cls, (x1, y1, x2, y2) in zip(
                boxes.id.int().tolist(), boxes.cls.int().tolist(), boxes.xyxy.tolist()
            ):
                direction = counter.update(tid, ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h))
                if direction:
                    key = "CountIn" if direction == "in" else "CountOut"
                    pending[key] += 1
                    totals[key] += 1
                    pending[tag_for_class[cls]] += 1
        counter.end_frame()

        # ---- annotated stream -> MediaMTX ----
        t1 = time.time()
        out_w, out_h = even_size(w, h, cfg["out_width"])
        annotated = result.plot()
        line_px = [(int(x * annotated.shape[1]), int(y * annotated.shape[0])) for x, y in cam["line"]]
        draw_overlay(annotated, line_px, totals, in_view)
        if annotated.shape[1] != out_w or annotated.shape[0] != out_h:
            annotated = cv2.resize(annotated, (out_w, out_h))

        if publisher is not None and (publisher.size != (out_w, out_h) or publisher.proc.poll() is not None):
            publisher.close()
            publisher = None
        if publisher is None and not args.no_publish and time.time() >= next_publisher_try:
            next_publisher_try = time.time() + 5       # back-off if ffmpeg dies straight away
            try:
                publisher = Publisher(det_url, out_w, out_h, fps, ffmpeg)
            except FileNotFoundError:
                print(f"{tag} ffmpeg not found - set \"ffmpeg_path\" in cameras.json "
                      "(counting still works, only the annotated stream is off)")
                next_publisher_try = time.time() + 30
        if publisher is not None:
            publisher.write(annotated)

        if args.show:
            cv2.imshow(cam["id"], annotated)
            cv2.waitKey(1)
        t_draw += time.time() - t1

        # ---- report to Ignition ----
        if time.time() - last_post >= cfg["post_interval_s"]:
            t2 = last_post = time.time()
            payload = {"camera": cam["id"], "count": in_view, "delta": dict(pending)}
            try:
                resp = requests.post(cfg["ignition_url"], json=payload, timeout=3)
                resp.raise_for_status()
                reply = resp.json()
                if not reply.get("ok", True):
                    print(f"{tag} Ignition accepted the post but reported: {reply}")
                pending.clear()
            except requests.RequestException as exc:
                print(f"{tag} POST failed (will retry, counts kept): {exc}")
            t_post += time.time() - t2

        if time.time() - last_log > 5:
            now = time.time()
            n = max(frames, 1)
            print(f"{tag} in view: {in_view}  in: {totals['CountIn']}  out: {totals['CountOut']}"
                  f"  analysing {frames / (now - last_log):.1f} fps (target {fps})"
                  f"  | feed delivering {(reader.made - made_mark) / (now - last_log):.1f} fps"
                  f"  | per frame: model {t_track / n * 1000:.0f} ms,"
                  f" draw+publish {t_draw / n * 1000:.0f} ms, post {t_post / n * 1000:.0f} ms")
            frames, made_mark, last_log = 0, reader.made, now
            t_track = t_draw = t_post = 0.0


if __name__ == "__main__":
    main()
