"""Turn results/accuracy.json + results/throughput.json into a summary CSV and three charts."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
ORDER = ["Classical OpenCV", "YOLO26n", "YOLOv10n", "YOLO11n", "D-FINE-N", "YOLOv5nu"]  # top-5 list + v5nu
# fixed categorical slots 1-6 of the reference palette + line style as a second, non-color cue
STYLE = {"Classical OpenCV": ("#2a78d6", "-"), "YOLO26n": ("#eb6834", "--"), "YOLOv10n": ("#1baf7a", "-."),
         "YOLO11n": ("#eda100", ":"), "D-FINE-N": ("#e87ba4", (0, (5, 1, 1, 1))),
         "YOLOv5nu": ("#008300", (0, (1, 1)))}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "legend.frameon": False})

acc = json.loads((RES / "accuracy.json").read_text())
thr = json.loads((RES / "throughput.json").read_text())

rows = []
for m in ORDER:
    a, t = acc[m], thr[m]
    rows.append({"model": m, "accuracy_detection": a["detection_accuracy"], "accuracy_image": a["image_accuracy"],
                 "precision": a["precision"], "recall": a["recall"], "f1": a["f1"],
                 "auc_roc_image": a["auc_roc_image"], "mean_iou_tp": a["mean_iou_tp"], "AP50": a["AP50"],
                 "AP50_95": a["AP50_95"], "max_throughput_ips": t["max_throughput_ips"],
                 "best_batch": t["best_batch"], "b1_inference_fps": t["b1_inference_fps"],
                 "e2e_fps_b1": t["e2e_fps_b1"], "conf_threshold": a["conf_threshold"],
                 "TP": a["TP"], "FP": a["FP"], "FN": a["FN"]})
df = pd.DataFrame(rows)
df.to_csv(RES / "summary.csv", index=False, float_format="%.4f")
print(df.round(3).to_string(index=False))


def curves(key, xlabel, ylabel, title, fname, diag=False):
    fig, ax = plt.subplots(figsize=(6.4, 5))
    for m in ORDER:
        x, y = acc[m][key]
        c, ls = STYLE[m]
        lab = f"{m}  (AUC {acc[m]['auc_roc_image']:.3f})" if diag else f"{m}  (AP50 {acc[m]['AP50']:.3f})"
        ax.plot(x, y, color=c, ls=ls, lw=2, label=lab)
    if diag:
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":", label="Chance")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.legend(loc="lower right" if diag else "lower left")
    fig.tight_layout(); fig.savefig(RES / fname, dpi=160); plt.close(fig)


curves("_roc", "False positive rate (empty frames flagged)", "True positive rate (bottle frames detected)",
       "ROC: is there a bottle in this frame? (test set)", "roc_curves.png", diag=True)
curves("_pr", "Recall", "Precision", "Precision-recall at IoU 0.5 (test set)", "pr_curves.png")

fig, ax = plt.subplots(figsize=(7, 3.8))
y = range(len(ORDER))[::-1]
e2e = [thr[m]["e2e_fps_b1"] for m in ORDER]; mx = [thr[m]["max_throughput_ips"] for m in ORDER]
ax.barh([i + 0.2 for i in y], mx, height=0.36, color="#2a78d6", label="Max throughput (best batch, inference only)")
ax.barh([i - 0.2 for i in y], e2e, height=0.36, color="#86b6ef", label="Live loop, batch 1 (pre + infer + post)")
for i, a, b in zip(y, mx, e2e):
    ax.text(a + 4, i + 0.2, f"{a:.0f}", va="center", color=INK, fontsize=9)
    ax.text(b + 4, i - 0.2, f"{b:.0f}", va="center", color=MUTED, fontsize=9)
ax.axvline(100, color=MUTED, lw=1, ls="--"); ax.text(102, len(ORDER) - 0.45, "100 FPS", color=MUTED, fontsize=9)
ax.set_yticks(list(y), ORDER); ax.set_xlabel("Images per second (CPU: AMD Ryzen 7 170, ONNX Runtime, 320 px)")
ax.grid(axis="y", visible=False); ax.set_xlim(0, max(mx) * 1.15)
ax.set_title("Throughput on this laptop's CPU (no GPU)", loc="left", color=INK, fontsize=12)
ax.legend(loc="lower right", fontsize=8.5)
fig.tight_layout(); fig.savefig(RES / "throughput.png", dpi=160); plt.close(fig)
