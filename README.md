# Top-5 model benchmark (+ YOLOv5nu): bottles on a conveyor

**Dataset:** Roboflow Universe `tpicbxaiproject/bottle-detection-conveyor` v16 (CC BY 4.0). It contains clear PET bottles on a moving conveyor, one class (`bottle`).

**Split** (`benchmark/make_split.py`): whole source videos are held out so that no test frame shares footage with training.

| Split | Images | Boxes | Empty frames | Notes |
|---|---|---|---|---|
| train | 766 | 1514 | 114 | |
| val | 224 | 390 | 39 | Same videos as train. Used only for threshold and model selection. |
| test | 334 | 524 | 98 | Videos `20250103_091746`, `normal_MOV` and `a3_MOV`, plus empty-conveyor groups 30 and up. |

**Training** (CPU only, AMD Ryzen 7 170): all models used 320×320 input and 25 epochs, starting from COCO-pretrained weights.

| Model | Training details |
|---|---|
| YOLO26n / YOLOv10n / YOLO11n / YOLOv5nu | Ultralytics 8.4.164 with default augmentation (`benchmark/train_yolo.py`) |
| D-FINE-N | Hugging Face `ustc-community/dfine-nano-coco` with AdamW and a cosine schedule (`benchmark/train_dfine.py`) |
| Classical OpenCV | Leave-one-out temporal-median background per camera scene, then absdiff and column-projection segmentation. Thresholds were grid-searched on val (`benchmark/classical.py`). |

## Metric definitions (`benchmark/metrics.py`)

The operating confidence threshold for each model is the one that maximizes F1 on **val**. It is then applied unchanged to test.

| Metric | Definition |
|---|---|
| Precision / Recall / F1 | Box level, with a match counting at IoU ≥ 0.5 |
| Detection accuracy | TP / (TP + FP + FN). Box-level detection has no true negatives, so this is the detection analogue of accuracy. |
| Image accuracy | Fraction of frames correctly classified as "contains a bottle" or "empty" |
| AUC-ROC | Image level, using the max box score per frame as the score. This is the only view of detection that has true negatives (the empty frames). |
| Mean IoU | Average IoU of the true-positive matches |
| AP50 / AP50-95 | COCO-style 101-point interpolated average precision |
| Max throughput | ONNX Runtime on CPU; best images/s over batch sizes 1–32, inference only |
| Live loop FPS | Batch 1, in-memory frame: resize, normalize, inference and postprocess (NMS for YOLO11n) |

## Caveats

- **Throughput is CPU-only.** A Jetson or desktop GPU will be several times faster for the neural models.
- **Throughput varies between runs.** Two full runs differed by about 30% in absolute speed (laptop turbo and thermals, Balanced power plan, on AC). Each number is the median of 3 timed windows, and the model ranking was identical in both runs.
- **Thresholds set on val ran conservative on test.** For the YOLO models the val-chosen thresholds (about 0.74–0.83) are high, because val shares scenes with train. On the new test videos that trades recall for precision. The PR curves show what a lower threshold would give.
- **Empty-frame negatives are hard on the classical method.** The empty-conveyor frames come from mixed camera setups, which inflates the classical method's false positives and is why its ROC AUC is low. On a single fixed camera its background model would match better.
- **Kalman tracking was not evaluated.** The data is sparse still frames, not continuous video.

Reproduce with:
`make_split.py` → `train_yolo.py` + `train_dfine.py` → `classical.py` → `evaluate.py` → `throughput.py` → `report.py`.
