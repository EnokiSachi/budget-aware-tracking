#!/usr/bin/env python3
"""
Step 11：端到端实测帧率（新流水线 vs 毕设流水线），为"实时"说法提供依据

在毕设视频（1920x1080，前 300 帧）上逐环节计时：
  A. 毕设流水线：见 step3 报告（预处理 + YOLOv8n + DeepSORT + 写视频），这里不重复
  B. 新流水线 GPU：YOLOv8n-ft@960 每帧检测 + 自实现 ByteTrack + 绘制（分别测 写视频 / 不写视频）
  C. 新流水线 CPU（4 线程，低功耗代理）：YOLOv8n-ft@960 每 3 帧检测一次 + ByteTrack 每帧 + 绘制，不写视频
注意：B/C 的"不写视频"版本对应实时显示/推流场景；写视频的开销与检测无关，单独列出。

用法：.\\.venv\\Scripts\\python.exe step11_speed.py --video <毕设 test.mp4>
输出：results/step11_report.txt
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import common as C
from step2_detect import weights_for
from trackers import ByteTracker


def load_frames(video, mot17, n):
    frames = []
    if video and Path(video).exists():
        cap = cv2.VideoCapture(str(video))
        while len(frames) < n:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
        return frames, Path(video).name
    info = C.seqinfo(mot17, "MOT17-04")
    return [cv2.imread(str(C.frame_path(mot17, info, f))) for f in range(1, n + 1)], "MOT17-04"


def run(frames, model, device, k, write, threads=None):
    import torch
    if threads:
        torch.set_num_threads(threads)
    kw = dict(imgsz=960, conf=0.05, iou=0.45, classes=[0], device=device, verbose=False)
    for _ in range(3):
        model.predict(frames[0], **kw)
    h, w = frames[0].shape[:2]
    tmp = C.RESULTS / "_speed.mp4"
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h)) if write else None
    trk = ByteTracker()
    T = {"detect": [], "track": [], "draw": [], "write": []}
    t_all = time.perf_counter()
    for i, fr in enumerate(frames):
        a = time.perf_counter()
        dets = None
        if i % k == 0:
            r = model.predict(fr, **kw)[0]
            b = r.boxes
            dets = np.hstack([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None]]) if len(b) else np.zeros((0, 5))
        T["detect"].append(time.perf_counter() - a)
        a = time.perf_counter(); out = trk.step(i + 1, dets); T["track"].append(time.perf_counter() - a)
        a = time.perf_counter()
        img = fr.copy()
        for tid, x1, y1, x2, y2, _ in out:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, str(tid), (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        T["draw"].append(time.perf_counter() - a)
        a = time.perf_counter()
        if writer:
            writer.write(img)
        T["write"].append(time.perf_counter() - a)
    total = time.perf_counter() - t_all
    if writer:
        writer.release(); tmp.unlink(missing_ok=True)
    return {k_: 1000 * float(np.mean(v)) for k_, v in T.items()}, len(frames) / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot17", default=C.DEFAULT_MOT17)
    ap.add_argument("--video", default="data/test.mp4")
    ap.add_argument("--frames", type=int, default=300)
    args = ap.parse_args()
    import torch
    from ultralytics import YOLO
    frames, src = load_frames(args.video, args.mot17, args.frames)
    model = YOLO(weights_for("n-ft"))
    h, w = frames[0].shape[:2]
    L = ["# Step 11 报告：端到端帧率实测", f"source: {src} {w}x{h}, {len(frames)} frames (frames pre-loaded; decode excluded)", ""]
    cases = [("B1 GPU, detect every frame, write video", "0", 1, True, None),
             ("B2 GPU, detect every frame, no write", "0", 1, False, None),
             ("C  CPU 4 threads, detect every 3rd frame, no write", "cpu", 3, False, 4)]
    if not torch.cuda.is_available():
        cases = cases[2:]
    for name, dev, k, wr, th in cases:
        T, fps = run(frames, model, dev, k, wr, th)
        L.append(f"{name}: {fps:5.1f} FPS  | " + "  ".join(f"{a} {b:.1f}ms" for a, b in T.items()))
    L += ["", "对照：毕设流水线（step3 报告）约 8.5 FPS（GPU 检测，含写视频）。"]
    (C.RESULTS / "step11_report.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
