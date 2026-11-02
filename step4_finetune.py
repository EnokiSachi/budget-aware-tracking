#!/usr/bin/env python3
"""
Step 4：在 MOT17 上微调 YOLOv8 检测器（领域适配）

第一阶段发现瓶颈在检测（召回率约 30%）。这里用每个序列"前 25%"（q1）的人工标注微调 YOLOv8n / YOLOv8s：
  - 训练集：q1 的前 85% 帧；验证集（选 best.pt 用）：q1 的后 15% 帧 —— 都不碰 25% 之后的帧；
  - 标注：MOT17 中 class 1（行人）和 class 7（静止的人），且 consider 标记为 1；框裁剪到图像内；
  - 单类别 "person"，从 COCO 预训练权重出发，所以类别 0 仍然是 person，下游代码不用改。

之后用 step2_detect.py --models n-ft s-ft 缓存微调模型的检测结果。

用法：
  .\\.venv\\Scripts\\python.exe step4_finetune.py                 # 默认 n 和 s，各 40 轮，imgsz 1280
  .\\.venv\\Scripts\\python.exe step4_finetune.py --models n --epochs 2   # 先试跑
3070 Ti（8GB）预计：n 约 20–30 分钟，s 约 40–60 分钟。
输出：results/finetune/ft_n/weights/best.pt 等；results/step4_report.txt（发回）
"""
import argparse
import json
import shutil
import time

import numpy as np

import common as C

DATA = C.RESULTS / "finetune" / "mot17_q1"


def build_dataset(mot17, val_frac=0.15, min_vis=0.0, seqs=None, data=None):
    data = data or DATA
    seqs = seqs or C.SEQS
    stats = {"train": [0, 0], "val": [0, 0]}
    for sub in ("train", "val"):
        for kind in ("images", "labels"):
            (data / kind / sub).mkdir(parents=True, exist_ok=True)
    for s in seqs:
        info = C.seqinfo(mot17, s)
        W, H = info["width"], info["height"]
        gt = np.loadtxt(C.seq_dir(mot17, s) / "gt" / "gt.txt", delimiter=",", ndmin=2)
        a, b = C.split_range(info["length"], "q1")
        n_val = max(1, int(round((b - a + 1) * val_frac)))
        for f in range(a, b + 1):
            sub = "val" if f > b - n_val else "train"
            rows = gt[(gt[:, 0] == f) & (gt[:, 6] == 1) & np.isin(gt[:, 7], [1, 7]) & (gt[:, 8] >= min_vis)]
            lines = []
            for _, _, x, y, w, h, *_ in rows:
                x1, y1 = max(0.0, x), max(0.0, y)
                x2, y2 = min(W, x + w), min(H, y + h)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                lines.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
            stem = f"{s}_{f:06d}"
            dst = data / "images" / sub / f"{stem}{info['ext']}"
            if not dst.exists():
                shutil.copy2(C.frame_path(mot17, info, f), dst)
            (data / "labels" / sub / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            stats[sub][0] += 1
            stats[sub][1] += len(lines)
    yaml = data / "data.yaml"
    yaml.write_text(f"path: {data.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: person\n",
                    encoding="utf-8")
    return yaml, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot17", default=C.DEFAULT_MOT17)
    ap.add_argument("--models", nargs="+", default=["n", "s"], choices=["n", "s", "m"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = 自动按显存选择")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    from ultralytics import YOLO

    t0 = time.monotonic()
    yaml, stats = build_dataset(args.mot17)
    report = ["# Step 4 报告：检测器微调（MOT17 前 25%）",
              f"dataset: train {stats['train'][0]} images / {stats['train'][1]} boxes; "
              f"val {stats['val'][0]} images / {stats['val'][1]} boxes",
              f"epochs={args.epochs} imgsz={args.imgsz} batch={args.batch}"]
    print("\n".join(report), flush=True)
    for m in args.models:
        ts = time.monotonic()
        model = YOLO(f"yolov8{m}.pt")
        model.train(data=str(yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, workers=args.workers,
                    project=str(C.RESULTS / "finetune"), name=f"ft_{m}", exist_ok=True, seed=0,
                    deterministic=True, single_cls=True, plots=True, verbose=False)
        best = C.RESULTS / "finetune" / f"ft_{m}" / "weights" / "best.pt"
        line = f"yolov8{m}: trained {time.monotonic() - ts:.0f}s"
        for tag, w in (("pretrained", f"yolov8{m}.pt"), ("fine-tuned", str(best))):
            try:  # 80 类的预训练模型在单类数据上验证，部分版本会报错，失败时跳过即可
                r = YOLO(w).val(data=str(yaml), imgsz=args.imgsz, classes=[0], verbose=False, plots=False,
                                batch=8, workers=args.workers, split="val")
                line += f" | {tag}: mAP50={r.box.map50:.3f} mAP50-95={r.box.map:.3f} P={r.box.mp:.3f} R={r.box.mr:.3f}"
            except Exception as e:
                line += f" | {tag}: val skipped ({type(e).__name__})"
        report.append(line)
        print(line, flush=True)
    report.append(f"total time: {time.monotonic() - t0:.0f}s")
    report.append("注：此处验证集是 q1 末尾 15% 帧，与训练帧来自同一时段，只用于确认训练正常；"
                  "真正的比较在 25%–50%（q2）上用跟踪指标进行。")
    (C.RESULTS / "step4_report.txt").write_text("\n".join(report), encoding="utf-8")
    print(f"\n已保存：{C.RESULTS / 'step4_report.txt'}")


if __name__ == "__main__":
    main()
