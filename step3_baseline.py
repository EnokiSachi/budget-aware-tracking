#!/usr/bin/env python3
"""
Step 3：毕设基线的标准化评测 + 速度剖析（只在 tune 前半段上评测，test 后半段保持未看）

A. 用人工标注、官方 TrackEval 计算 HOTA / MOTA / IDF1 / IDSW / Recall / Precision：
   thesis_deepsort        复现毕设：YOLOv8n@640 + 预处理 + 置信度 0.5 + DeepSORT(max_age 30, n_init 3, cos 0.4)
   thesis_deepsort_shown  同上，但输出毕设画面上实际显示的全部"已确认"轨迹（含本帧未匹配到检测的预测框）
   deepsort_nopre         去掉预处理（检验"预处理提升召回"的说法）
   sort_own               自实现 SORT（卡尔曼 + 匈牙利，只用高分检测）
   bytetrack_own          自实现 ByteTrack（加入低分检测二次关联）
B. 毕设流水线逐环节计时（读帧 / 预处理 / 检测 / DeepSORT / 绘制 / 写视频），GPU 与 CPU 检测对比。

用法：.\\.venv\\Scripts\\python.exe step3_baseline.py --mot17 data/MOT17 --video <毕设 test.mp4>
输出：results/step3_report.txt（发回）  以及 results/tune/trackers/*/data/*.txt
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

import common as C
from step2_detect import thesis_preprocess
from trackers import ByteTracker

SPLIT = "tune"
TRK_ROOT = C.RESULTS / SPLIT / "trackers"


def run_own(key, name, seqs, splits, **kw):
    t_ms = []
    for s in seqs:
        dets, _ = C.load_dets(key, s)
        a, b = splits[s][SPLIT]["first"], splits[s][SPLIT]["last"]
        trk, rows = ByteTracker(**kw), []
        for f in range(a, b + 1):
            t0 = time.perf_counter()
            out = trk.step(f, dets.get(f, np.zeros((0, 5), np.float32)))
            t_ms.append((time.perf_counter() - t0) * 1000)
            rows += [(f - a + 1, tid, x1, y1, x2 - x1, y2 - y1, sc) for tid, x1, y1, x2, y2, sc in out]
        C.write_mot(TRK_ROOT / name / "data" / f"{s}.txt", rows)
    return float(np.mean(t_ms))


def run_deepsort(key, names, seqs, splits, mot17, pre):
    """names = (matched_name, shown_name or None)。返回每帧 DeepSORT 平均耗时（ms，含外观特征提取）。"""
    from deep_sort_realtime.deepsort_tracker import DeepSort
    t_ms = []
    for s in seqs:
        info = C.seqinfo(mot17, s)
        dets, _ = C.load_dets(key, s)
        a, b = splits[s][SPLIT]["first"], splits[s][SPLIT]["last"]
        trk = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.4)
        matched, shown = [], []
        for f in range(a, b + 1):
            img = cv2.imread(str(C.frame_path(mot17, info, f)))
            if pre == "thesis":
                img = thesis_preprocess(img)
            d = dets.get(f, np.zeros((0, 5), np.float32))
            d = d[d[:, 4] >= 0.5]
            inp = [([float(x1), float(y1), float(x2 - x1), float(y2 - y1)], float(sc), "person") for x1, y1, x2, y2, sc in d]
            t0 = time.perf_counter()
            tracks = trk.update_tracks(inp, frame=img)
            t_ms.append((time.perf_counter() - t0) * 1000)
            for t in tracks:
                if not t.is_confirmed():
                    continue
                x1, y1, x2, y2 = t.to_ltrb()
                row = (f - a + 1, int(t.track_id), x1, y1, x2 - x1, y2 - y1, 1.0)
                shown.append(row)
                if t.time_since_update == 0:
                    matched.append(row)
        C.write_mot(TRK_ROOT / names[0] / "data" / f"{s}.txt", matched)
        if names[1]:
            C.write_mot(TRK_ROOT / names[1] / "data" / f"{s}.txt", shown)
    return float(np.mean(t_ms))


def profile_pipeline(video, mot17, n_frames=300):
    """逐环节复现毕设主循环并计时。没有毕设视频时用 MOT17-04（1920x1080）。"""
    import torch
    from deep_sort_realtime.deepsort_tracker import DeepSort
    from ultralytics import YOLO
    lines = []
    frames = []
    if video and Path(video).exists():
        cap = cv2.VideoCapture(str(video))
        src, fps = Path(video).name, cap.get(5) or 25
        t_read = []
        while len(frames) < n_frames:
            a = time.perf_counter()
            ok, fr = cap.read()
            t_read.append((time.perf_counter() - a) * 1000)
            if not ok:
                break
            frames.append(fr)
        cap.release()
    else:
        info = C.seqinfo(mot17, "MOT17-04")
        src, fps, t_read = "MOT17-04", info["fps"], []
        for f in range(1, n_frames + 1):
            a = time.perf_counter()
            frames.append(cv2.imread(str(C.frame_path(mot17, info, f))))
            t_read.append((time.perf_counter() - a) * 1000)
    h, w = frames[0].shape[:2]
    lines.append(f"source: {src} {w}x{h}, {len(frames)} frames")
    model = YOLO("yolov8n.pt")
    cuda = torch.cuda.is_available()
    dev = "0" if cuda else "cpu"
    for _ in range(5):
        model.predict(frames[0], imgsz=640, conf=0.5, iou=0.45, classes=[0], device=dev, verbose=False)
    trk = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.4)
    tmp = C.RESULTS / "_profile.mp4"
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    T = {k: [] for k in ("preprocess", "detect", "deepsort", "draw", "write")}
    for fr in frames:
        a = time.perf_counter(); img = thesis_preprocess(fr); T["preprocess"].append(time.perf_counter() - a)
        a = time.perf_counter()
        r = model.predict(img, imgsz=640, conf=0.5, iou=0.45, classes=[0], device=dev, verbose=False)[0]
        if cuda:
            torch.cuda.synchronize()
        T["detect"].append(time.perf_counter() - a)
        b = r.boxes
        dets = [([float(x1), float(y1), float(x2 - x1), float(y2 - y1)], float(c), "person")
                for (x1, y1, x2, y2), c in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy())]
        a = time.perf_counter(); tracks = trk.update_tracks(dets, frame=img); T["deepsort"].append(time.perf_counter() - a)
        a = time.perf_counter()
        for t in tracks:
            if t.is_confirmed():
                x1, y1, x2, y2 = map(int, t.to_ltrb())
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"ID {t.track_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        T["draw"].append(time.perf_counter() - a)
        a = time.perf_counter(); writer.write(img); T["write"].append(time.perf_counter() - a)
    writer.release()
    tmp.unlink(missing_ok=True)
    total = 0.0
    if t_read:
        lines.append(f"  read        {np.mean(t_read):7.1f} ms"); total += np.mean(t_read)
    for k, v in T.items():
        ms = 1000 * np.mean(v[5:])
        total += ms
        lines.append(f"  {k:<11} {ms:7.1f} ms   (device={dev if k == 'detect' else 'cpu/gpu-embed' if k == 'deepsort' else 'cpu'})")
    lines.append(f"  TOTAL       {total:7.1f} ms/frame  ≈ {1000 / total:.1f} FPS (GPU detection, no window display)")
    # CPU 检测对照
    t_cpu = []
    for fr in frames[:20]:
        a = time.perf_counter()
        model.predict(thesis_preprocess(fr), imgsz=640, conf=0.5, iou=0.45, classes=[0], device="cpu", verbose=False)
        t_cpu.append((time.perf_counter() - a) * 1000)
    cpu_total = total - 1000 * np.mean(T["detect"][5:]) + np.mean(t_cpu[3:])
    lines.append(f"  detect on CPU instead: {np.mean(t_cpu[3:]):.1f} ms → TOTAL ≈ {cpu_total:.1f} ms ≈ {1000 / cpu_total:.1f} FPS")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot17", default=C.DEFAULT_MOT17)
    ap.add_argument("--video", default="data/test.mp4")
    ap.add_argument("--skip-deepsort", action="store_true")
    ap.add_argument("--skip-profile", action="store_true")
    args = ap.parse_args()
    splits = json.loads((C.GT_HALF / "splits.json").read_text())
    seqs = C.SEQS
    k_none, k_thesis = C.cache_key("n", 640, "none"), C.cache_key("n", 640, "thesis")
    lat = {}
    t0 = time.monotonic()
    names = []
    if not args.skip_deepsort:
        print("DeepSORT (thesis config, with preprocessing)...", flush=True)
        lat["thesis_deepsort"] = {"track_ms": run_deepsort(k_thesis, ("thesis_deepsort", "thesis_deepsort_shown"), seqs, splits, args.mot17, "thesis")}
        print("DeepSORT (no preprocessing)...", flush=True)
        lat["deepsort_nopre"] = {"track_ms": run_deepsort(k_none, ("deepsort_nopre", None), seqs, splits, args.mot17, "none")}
        names += ["thesis_deepsort", "thesis_deepsort_shown", "deepsort_nopre"]
    print("own SORT / ByteTrack...", flush=True)
    lat["sort_own"] = {"track_ms": run_own(k_none, "sort_own", seqs, splits, use_low=False)}
    lat["bytetrack_own"] = {"track_ms": run_own(k_none, "bytetrack_own", seqs, splits)}
    names += ["sort_own", "bytetrack_own"]
    print("TrackEval...", flush=True)
    summary, per_seq = C.evaluate(SPLIT, TRK_ROOT, names, seqs)

    det_ms = {}
    for key in (k_none, k_thesis):
        try:
            det_ms[key] = float(np.mean(np.concatenate([C.load_dets(key, s)[1]["det_wall_ms"] for s in seqs])))
        except Exception:
            pass
    lines = ["# Step 3 报告：毕设基线标准化评测（MOT17 前半段 tune，7 个序列）",
             f"detection cache: {k_none}, {k_thesis}  | mean detector wall time: "
             + ", ".join(f"{k}={v:.1f}ms" for k, v in det_ms.items()), "",
             "## 跟踪精度（%）与跟踪器自身耗时（ms/帧，不含检测）",
             C.fmt_table(summary, (lat, ["track_ms"]))]
    lines += ["", "## 逐序列 HOTA（%）", "seq        " + "".join(n[:14].rjust(16) for n in names)]
    for s in seqs:
        lines.append(f"{s:<10} " + "".join(f"{100 * per_seq[n][s]['HOTA']:16.2f}" for n in names))
    if not args.skip_profile:
        print("profiling thesis pipeline...", flush=True)
        try:
            lines += ["", "## 毕设流水线逐环节耗时（复现主循环，不含窗口显示）"] + profile_pipeline(args.video, args.mot17)
        except Exception as e:
            lines += ["", f"[profile failed: {type(e).__name__}: {e}]"]
    lines += ["", f"total time: {time.monotonic() - t0:.0f}s"]
    (C.RESULTS / "step3_report.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n已保存：{C.RESULTS / 'step3_report.txt'}")


if __name__ == "__main__":
    main()
