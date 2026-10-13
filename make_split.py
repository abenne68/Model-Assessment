"""Build a leakage-free train/val/test split from bottle-v16 by holding out whole source videos.

Test = every frame of held-out videos (+ held-out empty-conveyor groups), one copy per frame.
Train/val = the remaining videos (one copy per frame; Ultralytics/our loaders augment on the fly).
For test frames that only exist as Roboflow-augmented copies, pick the copy that best matches
the scene's median background (i.e. the un-flipped / least-rotated one).
"""
import re, shutil, collections, json
from pathlib import Path
import cv2, numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "datasets" / "bottle-v16"
DST = ROOT / "datasets" / "bottle-split"
TEST_VIDEOS = {"20250103_091746_mp4", "normal_MOV", "a3_MOV"}
NEG_TEST_MIN = 30  # frame_no_bottle_<N>_* with N >= 30 go to test

def parse(name):
    base = name.split(".rf.")[0]                       # e.g. normal_MOV-0040_jpg
    m = re.match(r"frame_no_bottle_(\d+)", base)
    if m: return f"neg_{int(m.group(1))}", base
    if base.startswith("frame_no_bottle"): return "neg_0", base
    m = re.match(r"(.+?)-\d+_jpg$", base)
    if m: return m.group(1), base
    if re.match(r"frame_\d+_jpg$", base): return "frame", base
    return base, base

def group_of(vid):
    if vid.startswith("neg_"):
        return "test" if int(vid[4:]) >= NEG_TEST_MIN else "trainval"
    return "test" if vid in TEST_VIDEOS else "trainval"

def gray_small(p): return cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), (160, 160)).astype(np.float32)

def main():
    if DST.exists(): shutil.rmtree(DST)
    frames = collections.defaultdict(list)   # (split, base) -> [paths]
    vid_of = {}
    for split in ("train", "valid"):
        for p in sorted((SRC / split / "images").iterdir()):
            vid, base = parse(p.name)
            frames[(split, base)].append(p); vid_of[base] = vid
    train_bases = {b for (s, b) in frames if s == "train"}

    # median background per test video from its clean (valid-split, un-augmented) frames
    bg = {}
    for v in TEST_VIDEOS:
        ims = [gray_small(ps[0]) for (s, b), ps in frames.items() if s == "valid" and vid_of[b] == v]
        bg[v] = np.median(np.stack(ims), axis=0)

    def best_copy(paths, vid):
        if len(paths) == 1 or vid not in bg: return paths[0]
        return max(paths, key=lambda p: np.corrcoef(gray_small(p).ravel(), bg[vid].ravel())[0, 1])

    out = collections.Counter(); seen_test = set(); manifest = collections.defaultdict(list)
    for (split, base), paths in sorted(frames.items()):
        vid = vid_of[base]; g = group_of(vid)
        if g == "test":
            if base in seen_test: continue
            seen_test.add(base); dst_split = "test"
            # prefer the clean valid copy if it exists
            cands = frames.get(("valid", base)) or paths
            src = best_copy(cands, vid)
        else:
            if split == "valid" and base in train_bases: continue  # drop val frames duplicated in train
            dst_split = "train" if split == "train" else "val"
            src = paths[0]
        (DST / dst_split / "images").mkdir(parents=True, exist_ok=True)
        (DST / dst_split / "labels").mkdir(parents=True, exist_ok=True)
        name = f"{base}.jpg"
        shutil.copy2(src, DST / dst_split / "images" / name)
        lab = src.parent.parent / "labels" / (src.stem + ".txt")
        shutil.copy2(lab, DST / dst_split / "labels" / f"{base}.txt")
        out[dst_split] += 1; manifest[dst_split].append({"file": name, "video": vid})

    (DST / "data.yaml").write_text(
        f"path: {DST.as_posix()}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n  0: bottle\n")
    (DST / "manifest.json").write_text(json.dumps(manifest, indent=1))
    for s in ("train", "val", "test"):
        labs = list((DST / s / "labels").glob("*.txt"))
        nbox = sum(sum(1 for l in open(f) if l.strip()) for f in labs)
        nneg = sum(1 for f in labs if not open(f).read().strip())
        vids = collections.Counter(m["video"] for m in manifest[s])
        print(f"{s:5s} images={out[s]:4d} boxes={nbox:4d} empty_images={nneg:3d} videos={len(vids)}")

if __name__ == "__main__":
    main()
