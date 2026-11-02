#!/usr/bin/env python3
"""
Step 2：检测缓存 —— 对 MOT17 全部训练帧跑一次 YOLOv8，保存所有行人检测框和逐帧耗时

后续所有跟踪实验都读取缓存，不必重复跑检测；检测耗时是每帧实测值（GPU 已同步），
用于后面的"精度—延迟"分析和预算控制。

  --models n s m        检测器规模（YOLOv8n/s/m，COCO 预训练，未在 MOT17 上训练）
                        n-ft s-ft 为 step4_finetune.py 在 MOT17 前 25% 上微调后的模型
  --imgsz 640 960 1280  输入分辨率
  --preprocess thesis   复现毕设预处理（3×3 高斯去噪 + HSV 亮度 ×1.10），用于检验毕设中"预处理提升召回"的说法
  --half                FP16 推理
检测阈值统一取 0.05（保留低分框供 ByteTrack 使用），NMS IoU 0.45（与毕设一致）。

用法：
  第一阶段（基线，约 5 分钟）：
    .\\.venv\\Scripts\\python.exe step2_detect.py --models n --imgsz 640 --preprocess none thesis
  第二阶段（完整网格，约 40–60 分钟）：
    .\\.venv\\Scripts\\python.exe step2_detect.py --models n s m n-ft s-ft --imgsz 640 960 1280
"""
import argparse
import itertools
import json
import time

import cv2
import numpy as np

import common as C


def thesis_preprocess(img):
    img = cv2.GaussianBlur(img, (3, 3), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32) * 1.10
    hsv[:, :, 2] = np.clip(v, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def weights_for(model_name):
    if model_name.endswith("-ft"):
        w = C.RESULTS / "finetune" / f"ft_{model_name[:-3]}" / "weights" / "best.pt"
        if not w.exists():
            raise FileNotFoundError(f"找不到微调权重 {w}，请先运行 step4_finetune.py")
        return str(w)
    return f"yolov8{model_name}.pt"


def run_config(model_name, imgsz, pre, half, mot17, device, conf, iou, overwrite):
    import torch
    from ultralytics import YOLO
    key = C.cache_key(model_name, imgsz, pre, half)
    outdir = C.CACHE / key
    outdir.mkdir(parents=True, exist_ok=True)
    model = YOLO(weights_for(model_name))
    cuda = device != "cpu" and torch.cuda.is_available()
    # 只在需要 FP16 时才传 half（新版 ultralytics 只要传了 half 就会逐帧打印弃用警告）
    pkw = dict(imgsz=imgsz, conf=conf, iou=iou, classes=[0], device=device, verbose=False)
    if half:
        pkw["half"] = True
    warm = cv2.imread(str(C.frame_path(mot17, C.seqinfo(mot17, C.SEQS[0]), 1)))
    for _ in range(10):
        model.predict(warm, **pkw)
    summary = {}
    for s in C.SEQS:
        path = outdir / f"{s}.npz"
        if path.exists() and not overwrite:
            summary[s] = "cached"
            continue
        info = C.seqinfo(mot17, s)
        det_f, boxes, scores = [], [], []
        read_ms, pre_ms, wall_ms, inf_ms, post_ms, prep_ms = [], [], [], [], [], []
        for f in range(1, info["length"] + 1):
            a = time.perf_counter()
            img = cv2.imread(str(C.frame_path(mot17, info, f)))
            read_ms.append((time.perf_counter() - a) * 1000)
            a = time.perf_counter()
            if pre == "thesis":
                img = thesis_preprocess(img)
            pre_ms.append((time.perf_counter() - a) * 1000)
            if cuda:
                torch.cuda.synchronize()
            a = time.perf_counter()
            r = model.predict(img, **pkw)[0]
            if cuda:
                torch.cuda.synchronize()
            wall_ms.append((time.perf_counter() - a) * 1000)
            prep_ms.append(r.speed["preprocess"]); inf_ms.append(r.speed["inference"]); post_ms.append(r.speed["postprocess"])
            b = r.boxes
            if len(b):
                xyxy = b.xyxy.cpu().numpy().astype(np.float32)
                sc = b.conf.cpu().numpy().astype(np.float32)
                det_f.append(np.full(len(sc), f, dtype=np.int32)); boxes.append(xyxy); scores.append(sc)
        np.savez_compressed(
            path,
            det_frame=np.concatenate(det_f) if det_f else np.zeros(0, np.int32),
            boxes=np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float32),
            scores=np.concatenate(scores) if scores else np.zeros(0, np.float32),
            read_ms=np.array(read_ms, np.float32), pre_ms=np.array(pre_ms, np.float32),
            det_wall_ms=np.array(wall_ms, np.float32), yolo_pre_ms=np.array(prep_ms, np.float32),
            yolo_inf_ms=np.array(inf_ms, np.float32), yolo_post_ms=np.array(post_ms, np.float32))
        summary[s] = f"{len(wall_ms)} frames, det median {np.median(wall_ms):.1f} ms, " \
                     f"{sum(len(x) for x in scores)} boxes"
        print(f"  {key} {s}: {summary[s]}", flush=True)
    meta = {"model": f"yolov8{model_name}", "weights": weights_for(model_name), "imgsz": imgsz, "preprocess": pre, "half": half, "conf_min": conf,
            "nms_iou": iou, "device": device, "gpu": torch.cuda.get_device_name(0) if cuda else "cpu",
            "torch": torch.__version__}
    try:
        import ultralytics
        meta["ultralytics"] = ultralytics.__version__
    except Exception:
        pass
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot17", default=C.DEFAULT_MOT17)
    ap.add_argument("--models", nargs="+", default=["n"], choices=["n", "s", "m", "l", "x", "n-ft", "s-ft", "m-ft", "nx-ft"])
    ap.add_argument("--imgsz", nargs="+", type=int, default=[640])
    ap.add_argument("--preprocess", nargs="+", default=["none"], choices=["none", "thesis"])
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    t0 = time.monotonic()
    for m, sz, pre in itertools.product(args.models, args.imgsz, args.preprocess):
        print(f"== yolov8{m} imgsz={sz} preprocess={pre} half={args.half}", flush=True)
        run_config(m, sz, pre, args.half, args.mot17, args.device, args.conf, args.iou, args.overwrite)
    print(f"\n完成，用时 {time.monotonic() - t0:.0f}s。缓存目录：{C.CACHE}")


if __name__ == "__main__":
    main()
