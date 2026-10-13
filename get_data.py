"""Download bottle-detection-conveyor v16 (YOLOv8 format) from Roboflow. Key read from env only."""
import os
from pathlib import Path
from roboflow import Roboflow
ROOT = Path(__file__).resolve().parents[1]
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
ds = rf.workspace("tpicbxaiproject").project("bottle-detection-conveyor").version(16).download(
    "yolov8", location=str(ROOT / "datasets" / "bottle-v16"))
print(ds.location)
