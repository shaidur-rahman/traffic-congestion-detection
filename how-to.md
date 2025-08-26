# dhaka-illegal-parking-yolov11

A minimal, **GitHub-ready** project to train and run an illegal parking detector on Dhaka street videos using **YOLOv11 (small/mini)**. Includes frame extraction, (optional) CVAT workflow, dataset split, training, inference, and validation.

---

## 📁 Project Structure
```
dhaka-illegal-parking-yolov11/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ data/
│  ├─ raw/                  # your raw videos go here
│  ├─ frames/               # auto-generated frames from videos (if you choose frame-based annotation)
│  └─ yolo/                 # final YOLO dataset root (images/ + labels/ subfolders)
│     ├─ images/
│     │  ├─ train/
│     │  └─ val/
│     └─ labels/
│        ├─ train/
│        └─ val/
├─ configs/
│  └─ data.yaml             # auto-generated/updated; YOLO dataset config
└─ src/
   ├─ extract_frames.py     # extract frames from videos (option A: annotate frames)
   ├─ prepare_dataset.py    # split into train/val, sanity checks, build data.yaml
   ├─ train_yolo.py         # programmatic YOLOv11 training
   ├─ predict_video.py      # run inference on video, save visualized output
   ├─ val_metrics.py        # run YOLO val and print mAP/precision/recall
   └─ utils/
      ├─ common.py
      └─ viz.py
```

---

## 🔧 Installation
Create a fresh environment (conda recommended) and install deps:
```bash
pip install -r requirements.txt
```
> If you have a CUDA GPU, install the correct torch build first (see pytorch.org), then `pip install -r requirements.txt`.

---

## 🏷️ Annotation Options
**Option A: Frame-based (quick MVP)**
1) Put your source video(s) into `data/raw/`.
2) Run `src/extract_frames.py` to sample frames.
3) Annotate frames with LabelImg/Roboflow → save YOLO format labels alongside images.
4) Run `src/prepare_dataset.py` to split and validate the YOLO dataset.

**Option B: Video-based with CVAT (recommended)**
1) Upload your video to CVAT, annotate with tracking, export as **YOLO 1.1** or **YOLO** format.
2) Place exported folders into `data/yolo/` in the expected structure (images/ + labels/ under train/ and val/). If you exported a single split, you can use `src/prepare_dataset.py` to make a train/val split.

---

## 🚂 Training
```bash
python src/train_yolo.py \
  --model yolov11s.pt \
  --data configs/data.yaml \
  --epochs 50 \
  --imgsz 640 \
  --batch 8
```

## 🎥 Inference on a Video
```bash
python src/predict_video.py \
  --weights runs/detect/train/weights/best.pt \
  --source data/raw/dhaka_parking_30s.mp4 \
  --save_path runs/predicts/dhaka_parking_30s_out.mp4
```

## 📊 Validation (mAP/Precision/Recall)
```bash
python src/val_metrics.py --weights runs/detect/train/weights/best.pt --data configs/data.yaml
```

---

## ✅ Class Schema (single-class MVP)
We treat illegal parking as a single class. Put this into your annotator’s class list:
```
0 illegal_parking
```

---

## 🔁 Typical MVP Workflow (Tonight/Tomorrow)
1) Put 30s video into `data/raw/`.
2) (Option A) `extract_frames.py` → annotate ~150–300 images → `prepare_dataset.py`.
   (Option B) Annotate with CVAT → export YOLO → ensure into `data/yolo/` → run `prepare_dataset.py` if you need a split.
3) `train_yolo.py` with `yolov11s.pt` (or `yolov11n.pt` if low GPU).
4) `predict_video.py` on the same 30s video.
5) `val_metrics.py` to check mAP/precision/recall.

---

# -------------------------
# File: requirements.txt
# -------------------------
ultralytics>=8.3.0
opencv-python
numpy
PyYAML
tqdm

# If you plan to use Jupyter:
# jupyter

# -------------------------
# File: .gitignore
# -------------------------
__pycache__/
*.pyc
*.pyo
*.pyd
*.DS_Store
.env
runs/
.data_cache/
*.log

# dataset caches
*.cache

# vscode/idea
.vscode/
.idea/

# -------------------------
# File: README.md
# -------------------------
# Dhaka Illegal Parking – YOLOv11
This repo trains a YOLOv11 model to detect **illegally parked vehicles** in Dhaka videos.

## Quickstart
```bash
pip install -r requirements.txt
python src/train_yolo.py --model yolov11s.pt --data configs/data.yaml --epochs 50 --imgsz 640 --batch 8
```

See `src/` for scripts. Place your datasets under `data/yolo/` and videos under `data/raw/`.

## Annotation
- **CVAT export in YOLO** is recommended. Place into `data/yolo/` with `images/` and `labels/`.
- For a fast MVP, run `src/extract_frames.py`, then annotate frames and proceed.

## Inference
```bash
python src/predict_video.py --weights runs/detect/train/weights/best.pt --source data/raw/your_video.mp4
```

---

# -------------------------
# File: configs/data.yaml
# -------------------------
# Auto-generated or edited by prepare_dataset.py
path: data/yolo
train: images/train
val: images/val
names:
  0: illegal_parking

# -------------------------
# File: src/utils/common.py
# -------------------------
from pathlib import Path

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}


def is_video_file(p: Path) -> bool:
    return p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}


# -------------------------
# File: src/utils/viz.py
# -------------------------
import cv2


def draw_label(img, text, x, y):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x, y - h - 6), (x + w + 6, y), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 3, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                lineType=cv2.LINE_AA)


# -------------------------
# File: src/extract_frames.py
# -------------------------
"""
Sample frames from raw videos into data/frames/ for quick frame-based annotation.
"""
from pathlib import Path
import argparse
import cv2
from utils.common import ensure_dir, is_video_file


def extract_frames(raw_dir: Path, out_dir: Path, stride: int = 5):
    ensure_dir(out_dir)
    vids = [p for p in raw_dir.glob("**/*") if is_video_file(p)]
    if not vids:
        print(f"No videos found in {raw_dir}")
        return
    for v in vids:
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            print(f"Failed to open {v}")
            continue
        i = 0
        stem = v.stem
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                out_path = out_dir / f"{stem}_f{i:06d}.jpg"
                cv2.imwrite(str(out_path), frame)
            i += 1
        cap.release()
        print(f"Extracted frames from {v} -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", type=str, default="data/raw")
    ap.add_argument("--out_dir", type=str, default="data/frames")
    ap.add_argument("--stride", type=int, default=5, help="keep 1 of every N frames")
    args = ap.parse_args()
    extract_frames(Path(args.raw_dir), Path(args.out_dir), args.stride)


# -------------------------
# File: src/prepare_dataset.py
# -------------------------
"""
Prepare YOLO dataset: split images/labels into train/val and build configs/data.yaml.
If you already have a CVAT YOLO export, place it under data/yolo/images and data/yolo/labels
and this script can optionally create a split if needed.
"""
from pathlib import Path
import argparse
import random
import shutil
import yaml
from utils.common import ensure_dir, is_image_file


def copy_pairs(imgs, src_img_root: Path, src_lbl_root: Path, dst_img_root: Path, dst_lbl_root: Path):
    for img_rel in imgs:
        img_src = src_img_root / img_rel
        lbl_src = src_lbl_root / img_rel.with_suffix(".txt")
        img_dst = dst_img_root / img_rel.name
        lbl_dst = dst_lbl_root / img_rel.with_suffix(".txt").name
        ensure_dir(img_dst.parent)
        ensure_dir(lbl_dst.parent)
        shutil.copy2(img_src, img_dst)
        if lbl_src.exists():
            shutil.copy2(lbl_src, lbl_dst)


def make_split(root: Path, val_ratio: float = 0.2):
    img_root = root / "images"
    lbl_root = root / "labels"
    # Collect flat list of images
    imgs = [p.relative_to(img_root) for p in img_root.rglob("*") if is_image_file(p)]
    imgs.sort()
    random.shuffle(imgs)
    n_val = int(len(imgs) * val_ratio)
    val_imgs = imgs[:n_val]
    train_imgs = imgs[n_val:]

    # Create split dirs
    dst_img_train = img_root / "train"
    dst_img_val = img_root / "val"
    dst_lbl_train = lbl_root / "train"
    dst_lbl_val = lbl_root / "val"

    # Clear any previous split subdirs
    for p in [dst_img_train, dst_img_val, dst_lbl_train, dst_lbl_val]:
        if p.exists():
            shutil.rmtree(p)

    copy_pairs(train_imgs, img_root, lbl_root, dst_img_train, dst_lbl_train)
    copy_pairs(val_imgs, img_root, lbl_root, dst_img_val, dst_lbl_val)

    return len(train_imgs), len(val_imgs)


def write_data_yaml(config_path: Path, dataset_root: Path, class_map: dict):
    ensure_dir(config_path.parent)
    data = {
        "path": str(dataset_root.as_posix()),
        "train": "images/train",
        "val": "images/val",
        "names": class_map,
    }
    with open(config_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo_root", type=str, default="data/yolo", help="dataset root containing images/ and labels/")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--config_out", type=str, default="configs/data.yaml")
    ap.add_argument("--class_name", type=str, default="illegal_parking")
    args = ap.parse_args()

    root = Path(args.yolo_root)
    ensure_dir(root)

    # If no split yet (i.e., images/train not present), build one from flat export
    img_train = root / "images/train"
    if not img_train.exists():
        ntr, nval = make_split(root, args.val_ratio)
        print(f"Created split: train={ntr}, val={nval}")
    else:
        print("Split already exists. Skipping split.")

    write_data_yaml(Path(args.config_out), root, {0: args.class_name})
    print(f"Wrote YOLO data config to {args.config_out}")


# -------------------------
# File: src/train_yolo.py
# -------------------------
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="yolov11s.pt", help="yolov11n.pt or yolov11s.pt")
    ap.add_argument("--data", type=str, default="configs/data.yaml")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--project", type=str, default="runs/detect")
    ap.add_argument("--name", type=str, default="train")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                project=args.project, name=args.name)


if __name__ == "__main__":
    main()


# -------------------------
# File: src/predict_video.py
# -------------------------
"""
Run inference on a video and save an annotated output video. Assumes single-class (0: illegal_parking).
"""
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO


def infer_video(weights: str, source: str, save_path: str, conf: float = 0.25):
    model = YOLO(weights)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {source}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    out_dir = Path(save_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model.predict(source=frame, conf=conf, verbose=False)
        r0 = results[0]
        for box in r0.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0]) if box.cls is not None else -1
            score = float(box.conf[0]) if box.conf is not None else 0.0
            if cls_id == 0:  # illegal_parking
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"illegal_parking {score:.2f}", (x1, max(y1 - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, lineType=cv2.LINE_AA)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Saved annotated video to {save_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--source", type=str, required=True)
    ap.add_argument("--save_path", type=str, default="runs/predicts/out.mp4")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    infer_video(args.weights, args.source, args.save_path, args.conf)


# -------------------------
# File: src/val_metrics.py
# -------------------------
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--data", type=str, default="configs/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, imgsz=args.imgsz)
    # Ultralytics returns a Results object; printing will show mAP etc.
    print(metrics)


if __name__ == "__main__":
    main()
