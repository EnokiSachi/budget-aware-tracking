#!/usr/bin/env python3
"""
Step 0：环境与数据检查（只读）

检查：Python / PyTorch / CUDA / ultralytics / deep-sort-realtime 版本；GPU 与 CPU 上 YOLOv8n 的单帧耗时；
MOT17 数据是否完整；原毕设视频是否存在。顺带回答"毕设为什么只有 3 FPS"：对比 GPU 与 CPU 推理速度。

用法（PowerShell，在新项目根目录）：
  .\\.venv\\Scripts\\python.exe step0_env.py --mot17 data/MOT17 --video data/test.mp4
输出：results/step0_report.txt
"""
import argparse
import platform
import sys
import time

import numpy as np

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot17", default=C.DEFAULT_MOT17)
    ap.add_argument("--video", default="data/test.mp4")
    args = ap.parse_args()
    out = [f"python {sys.version.split()[0]} | {platform.platform()} | {sys.executable}"]

    def ver(mod):
        try:
            m = __import__(mod)
            return getattr(m, "__version__", "ok")
        except Exception as e:
            return f"MISSING ({type(e).__name__})"
    for mod in ["torch", "torchvision", "ultralytics", "deep_sort_realtime", "cv2", "numpy", "scipy", "pandas", "matplotlib"]:
        out.append(f"{mod}: {ver(mod)}")

    import torch
    cuda = torch.cuda.is_available()
    out.append(f"cuda_available: {cuda}" + (f" | {torch.cuda.get_device_name(0)} | cuda {torch.version.cuda}" if cuda else ""))
    try:
        import psutil
        out.append(f"cpu: {psutil.cpu_count(logical=False)} cores / {psutil.cpu_count()} threads")
    except Exception:
        pass

    # MOT17
    out.append("\n## MOT17")
    ok = True
    for s in C.SEQS:
        d = C.seq_dir(args.mot17, s)
        if not (d / "seqinfo.ini").exists():
            out.append(f"{s}: MISSING ({d})")
            ok = False
            continue
        info = C.seqinfo(args.mot17, s)
        n_img = len(list((d / info["imdir"]).glob("*" + info["ext"])))
        gt = d / "gt" / "gt.txt"
        out.append(f"{s}: {info['width']}x{info['height']} @{info['fps']:.0f}fps, frames={info['length']} "
                   f"images={n_img} gt={'yes' if gt.exists() else 'NO'}")
        ok &= n_img == info["length"] and gt.exists()
    out.append("MOT17 OK" if ok else "MOT17 不完整，请检查路径（应包含 train/MOT17-02-FRCNN/img1 等）")

    # YOLO 速度：GPU vs CPU（回答毕设 3 FPS 的原因）
    out.append("\n## YOLOv8n 单帧推理耗时（1920x1080 输入，imgsz=640）")
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        img = (np.random.default_rng(0).random((1080, 1920, 3)) * 255).astype(np.uint8)
        devices = (["0"] if cuda else []) + ["cpu"]
        for dev in devices:
            n = 50 if dev != "cpu" else 15
            for _ in range(5):
                model.predict(img, imgsz=640, device=dev, verbose=False, classes=[0])
            t, inf = [], []
            for _ in range(n):
                if dev != "cpu":
                    torch.cuda.synchronize()
                a = time.perf_counter()
                r = model.predict(img, imgsz=640, device=dev, verbose=False, classes=[0])
                if dev != "cpu":
                    torch.cuda.synchronize()
                t.append((time.perf_counter() - a) * 1000)
                inf.append(r[0].speed["inference"])
            out.append(f"device={dev:<4} end-to-end {np.median(t):7.1f} ms/frame (≈{1000 / np.median(t):5.1f} FPS)"
                       f" | model inference {np.median(inf):6.1f} ms")
    except Exception as e:
        out.append(f"[YOLO test failed: {type(e).__name__}: {e}]")

    # 原毕设视频
    from pathlib import Path
    v = Path(args.video)
    out.append("\n## 原毕设视频")
    if v.exists():
        import cv2
        cap = cv2.VideoCapture(str(v))
        out.append(f"{v.name}: {int(cap.get(3))}x{int(cap.get(4))} @{cap.get(5):.1f}fps, frames={int(cap.get(7))}")
        cap.release()
    else:
        out.append(f"not found: {v}（不影响后续步骤，只用于速度剖析）")

    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "step0_report.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\n已保存：{C.RESULTS / 'step0_report.txt'}")


if __name__ == "__main__":
    main()
