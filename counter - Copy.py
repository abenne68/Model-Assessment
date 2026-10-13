import time, requests, cv2
from ultralytics import YOLO

RTSP = "rtsp://localhost:8554/cam1"   # from MediaMTX
URL = "http://localhost:8088/system/webdev/PDA_TRACK/count"   # Ignition Web Dev endpoint

# COCO class IDs: 2=car, 3=motorcycle, 5=bus, 7=truck
CLASSES = [2, 3, 5, 7]

model = YOLO("yolov8n.pt")   # downloads automatically the first time


def process_video_stream():
    cap = cv2.VideoCapture(RTSP)
    last_push, last_ok = 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            if time.time() - last_ok > 10:    # stream down -> reconnect
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(RTSP)
                last_ok = time.time()
            continue
        last_ok = time.time()
        if time.time() - last_push < 1:       # ~1 FPS
            continue
        r = model.predict(frame, classes=CLASSES, conf=0.4, verbose=False)[0]
        cv2.imshow("detections", r.plot())
        cv2.waitKey(1)
        count = len(r.boxes)
        print("count:", count)
        try:
            requests.post(URL, json={"count": count}, timeout=3)
        except requests.RequestException as e:
            print("POST failed:", e)
        last_push = time.time()


while True:
    try:
        process_video_stream()
    except Exception as e:
        print("error:", e)
        time.sleep(5)