"""Fine-tune D-FINE-N (COCO-pretrained, Hugging Face port) on the bottle split, CPU only.

Same input size (320) and epoch budget (25) as the YOLO runs. Model selection: best val AP50-95.
Run with the .venv-dfine interpreter (it has `transformers`).
"""
import sklearn.metrics  # noqa: F401  (must import before transformers on this machine, see README)
import json, math, random, sys, time
from pathlib import Path
import cv2, numpy as np, torch
from transformers import DFineForObjectDetection

sys.path.insert(0, str(Path(__file__).parent))
from metrics import load_gt, average_precision  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "bottle-split"
OUT = ROOT / "runs" / "dfine_n"
HF_ID = "ustc-community/dfine-nano-coco"
IMGSZ, EPOCHS, BATCH, LR = 320, 25, 8, 2e-4
torch.manual_seed(0); random.seed(0); np.random.seed(0)


def to_tensor(bgr):
    rgb = cv2.cvtColor(cv2.resize(bgr, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0  # D-FINE: rescale only, no mean/std


def load_split(split):
    items = []
    for p in sorted((DATA / split / "images").glob("*.jpg")):
        rows = [l.split() for l in open(DATA / split / "labels" / (p.stem + ".txt")) if l.strip()]
        boxes = np.array([[float(v) for v in r[1:5]] for r in rows], np.float32).reshape(-1, 4)  # cxcywh norm
        items.append((p, boxes))
    return items


def augment(img, boxes):
    if random.random() < 0.5:  # horizontal flip
        img = img[:, ::-1].copy(); boxes = boxes.copy(); boxes[:, 0] = 1 - boxes[:, 0]
    a, b = random.uniform(0.75, 1.25), random.uniform(-25, 25)  # contrast / brightness jitter
    img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    return img, boxes


@torch.no_grad()
def predict(model, bgr_list, topk=100):
    """Batch of BGR images -> list of (K,5) xyxy+score arrays in original pixels."""
    x = torch.stack([to_tensor(b) for b in bgr_list])
    out = model(pixel_values=x)
    scores = out.logits.sigmoid()[..., 0]
    res = []
    for i, b in enumerate(bgr_list):
        h, w = b.shape[:2]
        s, idx = scores[i].topk(min(topk, scores.shape[1]))
        cx, cy, bw, bh = out.pred_boxes[i, idx].unbind(-1)
        xyxy = torch.stack([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], -1)
        res.append(torch.cat([xyxy, s[:, None]], 1).numpy())
    return res


def main():
    torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
    OUT.mkdir(parents=True, exist_ok=True)
    model = DFineForObjectDetection.from_pretrained(
        HF_ID, num_labels=1, id2label={0: "bottle"}, label2id={"bottle": 0}, ignore_mismatched_sizes=True)
    train, val = load_split("train"), load_split("val")
    imgs = {p: cv2.imread(str(p)) for p, _ in train + val}  # cache in RAM like Ultralytics cache=ram
    val_names = [p.name for p, _ in val]
    val_gt = load_gt(DATA / "val" / "labels", val_names)

    backbone = [p for n, p in model.named_parameters() if "backbone" in n]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n]
    opt = torch.optim.AdamW([{"params": backbone, "lr": LR / 10}, {"params": rest, "lr": LR}], weight_decay=1e-4)
    steps = EPOCHS * math.ceil(len(train) / BATCH); warm = 100
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))

    best, log = -1, []
    for ep in range(1, EPOCHS + 1):
        model.train(); random.shuffle(train); t0 = time.time(); losses = []
        for i in range(0, len(train), BATCH):
            batch = [augment(imgs[p], b) for p, b in train[i:i + BATCH]]
            x = torch.stack([to_tensor(im) for im, _ in batch])
            labels = [{"class_labels": torch.zeros(len(b), dtype=torch.long), "boxes": torch.from_numpy(b)}
                      for _, b in batch]
            loss = model(pixel_values=x, labels=labels).loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step(); sched.step(); losses.append(loss.item())
        model.eval()
        preds = {}
        for i in range(0, len(val), 16):
            chunk = val[i:i + 16]
            for (p, _), r in zip(chunk, predict(model, [imgs[p] for p, _ in chunk])):
                preds[p.name] = r
        ap = float(np.mean([average_precision(preds, val_gt, t) for t in np.arange(0.5, 0.96, 0.05)]))
        ap50 = average_precision(preds, val_gt, 0.5)
        log.append({"epoch": ep, "loss": float(np.mean(losses)), "val_AP50": ap50, "val_AP50_95": ap,
                    "sec": time.time() - t0})
        print(json.dumps(log[-1]), flush=True)
        if ap > best:
            best = ap; model.save_pretrained(OUT / "best")
        (OUT / "log.json").write_text(json.dumps(log, indent=1))
    print("best val AP50-95", best)


if __name__ == "__main__":
    main()
