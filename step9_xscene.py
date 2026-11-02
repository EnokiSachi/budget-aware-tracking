#!/usr/bin/env python3
"""
Step 9：跨场景微调 —— 区分"熟悉场景"与"真正学会检测行人"

第二阶段的微调检测器在 7 个序列的前 25% 上训练、在同一批场景上评测，提升里可能有"记住场景"的成分。
这里只用 4 个场景的前 25% 微调 YOLOv8n（训练设置与 step4 完全相同），
另外 3 个场景在训练中完全没出现过，最后在第 10 步的测试集上比较：
  未微调 n@960  vs  跨场景微调 nx-ft@960（没见过这些场景）  vs  同场景微调 n-ft@960

场景划分（固定、事先确定）：
  训练场景：MOT17-02（静止相机）、MOT17-04（静止）、MOT17-05（移动相机）、MOT17-13（移动）
  未见场景：MOT17-09（静止）、MOT17-10（移动）、MOT17-11（移动）
两侧都包含静止和移动相机。

用法：.\\.venv\\Scripts\\python.exe step9_xscene.py     （训练约 15–25 分钟 + 检测缓存约 3 分钟）
输出：results/finetune/ft_nx/weights/best.pt、cache/yolov8nx-ft_960_none/、results/step9_report.txt
"""
import argparse
import json
import time

import common as C
import step4_finetune as F
from step2_detect import run_config

TRAIN_SCENES = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-13"]
HELDOUT_SCENES = ["MOT17-09", "MOT17-10", "MOT17-11"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot17", default=C.DEFAULT_MOT17)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    from ultralytics import YOLO
    t0 = time.monotonic()
    data = C.RESULTS / "finetune" / "mot17_q1_xscene"
    yaml, stats = F.build_dataset(args.mot17, seqs=TRAIN_SCENES, data=data)
    YOLO("yolov8n.pt").train(data=str(yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                             workers=args.workers, project=str(C.RESULTS / "finetune"), name="ft_nx",
                             exist_ok=True, seed=0, deterministic=True, single_cls=True, plots=True, verbose=False)
    t_train = time.monotonic() - t0
    key = run_config("nx-ft", 960, "none", False, args.mot17, "0", 0.05, 0.45, False)
    lines = ["# Step 9 报告：跨场景微调",
             f"train scenes: {TRAIN_SCENES} (first 25% of each)", f"held-out scenes: {HELDOUT_SCENES}",
             f"dataset: train {stats['train'][0]} images / {stats['train'][1]} boxes; val {stats['val'][0]} / {stats['val'][1]}",
             f"trained {t_train:.0f}s; detection cache: {key}", f"total {time.monotonic() - t0:.0f}s"]
    (C.RESULTS / "xscene.json").write_text(json.dumps({"train": TRAIN_SCENES, "heldout": HELDOUT_SCENES}), encoding="utf-8")
    (C.RESULTS / "step9_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
