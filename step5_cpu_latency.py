#!/usr/bin/env python3
"""
Step 5：CPU 上的检测延迟实测（低功耗设备的代理）

GPU 延迟已在检测缓存里逐帧记录。这里在 CPU 上限制线程数（默认 4 线程，近似嵌入式 / 低功耗处理器），
对每个检测配置在 3 个 1080p 序列上各取 20 帧测单帧耗时（读图后计时：YOLO 前处理 + 推理 + 后处理），
取中位数，写入 results/latency_cpu.json，供 step6 画 CPU 预算下的精度—延迟前沿。

注意：这是"受限线程的笔记本 CPU"，不是真实的边缘设备；报告里如实这样描述。

用法：.\\.venv\\Scripts\\python.exe step5_cpu_latency.py     （约 10–20 分钟）
"""
import argparse
import itertools
import json
import time

import cv2
import numpy as np

import common as C
from step2_detect import weights_for

SEQS = ["MOT17-04", "MOT17-09", "MOT17-11"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot17", default=C.DEFAULT_MOT17)
    ap.add_argument("--models", nargs="+", default=["n", "s", "m", "n-ft", "s-ft"])
    ap.add_argument("--imgsz", nargs="+", type=int, default=[640, 960, 1280])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--frames", type=int, default=20)
    args = ap.parse_args()
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(args.threads)
    imgs = []
    for s in SEQS:
        info = C.seqinfo(args.mot17, s)
        a, b = C.split_range(info["length"], "q2")
        for f in np.linspace(a, b, args.frames).astype(int):
            imgs.append(cv2.imread(str(C.frame_path(args.mot17, info, int(f)))))
    path = C.RESULTS / "latency_cpu.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    out["_meta"] = {"threads": args.threads, "frames": len(imgs), "seqs": SEQS, "torch": torch.__version__}
    for m, sz in itertools.product(args.models, args.imgsz):
        try:
            model = YOLO(weights_for(m))
        except FileNotFoundError as e:
            print(f"skip {m}: {e}")
            continue
        kw = dict(imgsz=sz, conf=0.05, iou=0.45, classes=[0], device="cpu", verbose=False)
        for _ in range(3):
            model.predict(imgs[0], **kw)
        t = []
        for im in imgs:
            a = time.perf_counter()
            model.predict(im, **kw)
            t.append((time.perf_counter() - a) * 1000)
        key = C.cache_key(m, sz, "none")
        out[key] = {"median_ms": float(np.median(t)), "p90_ms": float(np.percentile(t, 90))}
        print(f"{key:<24} CPU({args.threads} threads) median {np.median(t):7.1f} ms  p90 {np.percentile(t, 90):7.1f} ms", flush=True)
        path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n已保存：{path}")


if __name__ == "__main__":
    main()
