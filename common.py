"""
common.py —— 共用：路径、MOT17 读取、前后半划分、MOT 格式读写、TrackEval 评测、检测缓存读取

项目结构（新项目，不修改原毕设代码）：
  <ROOT>/common.py, step*.py, trackers/, third_party/trackeval/
  <ROOT>/cache/       检测缓存（每个配置每个序列一个 npz）
  <ROOT>/results/     评测结果与报告
数据：MOT17 官方训练集（带人工标注），默认 data/MOT17，可用 --mot17 或环境变量 MOT17_ROOT 指定。
"""
import configparser
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CACHE = ROOT / "cache"
GT_HALF = RESULTS / "gt_half"
DEFAULT_MOT17 = os.environ.get("MOT17_ROOT", "data/MOT17")
SEQS = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]
VARIANT = "FRCNN"  # 三个版本（DPM/FRCNN/SDP）图像与标注相同，只取一个
SPLITS = ("tune", "test", "q1", "q2")
# tune = 前 50%（第一阶段基线评测用）；test = 后 50%（封存，最后一次性评测）
# q1 = 前 25%（只用于微调检测器）；q2 = 25%–50%（对比实验、调参、训练控制器）


# ------------------------------------------------------------------ MOT17
def seq_dir(mot17, seq):
    return Path(mot17) / "train" / f"{seq}-{VARIANT}"


def seqinfo(mot17, seq):
    cp = configparser.ConfigParser()
    cp.read(seq_dir(mot17, seq) / "seqinfo.ini")
    s = cp["Sequence"]
    return {"name": seq, "fps": float(s["frameRate"]), "length": int(s["seqLength"]),
            "width": int(s["imWidth"]), "height": int(s["imHeight"]),
            "imdir": s.get("imDir", "img1"), "ext": s.get("imExt", ".jpg")}


def frame_path(mot17, info, f):
    return seq_dir(mot17, info["name"]) / info["imdir"] / f"{f:06d}{info['ext']}"


def split_range(length, split):
    """返回该划分的 (首帧, 末帧)，1 起算、闭区间。"""
    half, quarter = length // 2, length // 4
    return {"tune": (1, half), "test": (half + 1, length),
            "q1": (1, quarter), "q2": (quarter + 1, half)}[split]


# ------------------------------------------------------------------ MOT 格式读写
def write_mot(path, rows):
    """rows: [(frame, track_id, x, y, w, h, score), ...]，frame 为划分内重新编号后的帧号。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for fr, tid, x, y, w, h, s in rows:
            f.write(f"{int(fr)},{int(tid)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{s:.4f},-1,-1,-1\n")


# ------------------------------------------------------------------ 检测缓存
def cache_key(model, imgsz, pre="none", half=False):
    return f"yolov8{model}_{imgsz}_{pre}" + ("_fp16" if half else "")


def load_dets(key, seq):
    """返回 (dets_by_frame, meta)：dets_by_frame[f] = ndarray[N,5] (x1,y1,x2,y2,score)。"""
    z = np.load(CACHE / key / f"{seq}.npz")
    frames, boxes, scores = z["det_frame"], z["boxes"], z["scores"]
    out = {}
    order = np.argsort(frames, kind="stable")
    frames, boxes, scores = frames[order], boxes[order], scores[order]
    if len(frames):
        uniq, starts = np.unique(frames, return_index=True)
        ends = list(starts[1:]) + [len(frames)]
        for f, a, b in zip(uniq, starts, ends):
            out[int(f)] = np.hstack([boxes[a:b], scores[a:b, None]]).astype(np.float32)
    meta = {k: z[k] for k in z.files if k not in ("det_frame", "boxes", "scores")}
    return out, meta


# ------------------------------------------------------------------ TrackEval
def _import_trackeval():
    p = str(ROOT / "third_party")
    if p not in sys.path:
        sys.path.insert(0, p)
    for a, b in (("float", float), ("int", int), ("bool", bool)):  # 兼容 numpy>=1.24
        if not hasattr(np, a):
            setattr(np, a, b)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):  # 屏蔽可选依赖（BURST/pycocotools）的导入提示
        import trackeval  # noqa: E402
    return trackeval


def evaluate(split, trackers_root, tracker_names, seqs=None):
    """用官方 TrackEval 评测。返回 {tracker: {指标: 值}}（全部序列合并）以及逐序列结果。"""
    te = _import_trackeval()
    info = json.loads((GT_HALF / "splits.json").read_text())
    seqs = seqs or SEQS
    seq_info = {s: info[s][split]["n"] for s in seqs}
    ecfg = te.Evaluator.get_default_eval_config()
    ecfg.update(PRINT_RESULTS=False, PRINT_CONFIG=False, TIME_PROGRESS=False, OUTPUT_SUMMARY=False,
                OUTPUT_DETAILED=False, PLOT_CURVES=False, USE_PARALLEL=False, OUTPUT_EMPTY_CLASSES=False,
                DISPLAY_LESS_PROGRESS=True)
    dcfg = te.datasets.MotChallenge2DBox.get_default_dataset_config()
    dcfg.update(GT_FOLDER=str(GT_HALF / split), TRACKERS_FOLDER=str(trackers_root), SKIP_SPLIT_FOL=True,
                SEQ_INFO=seq_info, TRACKERS_TO_EVAL=list(tracker_names), PRINT_CONFIG=False,
                BENCHMARK="MOT17")
    metrics = [te.metrics.HOTA({"PRINT_CONFIG": False}), te.metrics.CLEAR({"PRINT_CONFIG": False}),
               te.metrics.Identity({"PRINT_CONFIG": False})]
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        res, _ = te.Evaluator(ecfg).evaluate([te.datasets.MotChallenge2DBox(dcfg)], metrics)
    res = res["MotChallenge2DBox"]
    summary, per_seq = {}, {}
    for t in tracker_names:
        per_seq[t] = {}
        for s, r in res[t].items():
            r = r["pedestrian"]
            d = {"HOTA": float(np.mean(r["HOTA"]["HOTA"])), "DetA": float(np.mean(r["HOTA"]["DetA"])),
                 "AssA": float(np.mean(r["HOTA"]["AssA"])), "MOTA": float(r["CLEAR"]["MOTA"]),
                 "IDF1": float(r["Identity"]["IDF1"]), "IDSW": int(r["CLEAR"]["IDSW"]),
                 "Recall": float(r["CLEAR"]["CLR_Re"]), "Precision": float(r["CLEAR"]["CLR_Pr"]),
                 "FP": int(r["CLEAR"]["CLR_FP"]), "FN": int(r["CLEAR"]["CLR_FN"]), "Frag": int(r["CLEAR"]["Frag"])}
            if s == "COMBINED_SEQ":
                summary[t] = d
            else:
                per_seq[t][s] = d
    return summary, per_seq


def fmt_table(summary, extra=None):
    cols = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "Recall", "Precision"]
    w = max(len(k) for k in summary) + 2
    lines = ["tracker".ljust(w) + "".join(c.rjust(10) for c in cols) + ("" if not extra else "".join(e.rjust(12) for e in extra[1]))]
    for k, d in summary.items():
        row = k.ljust(w)
        for c in cols:
            v = d[c]
            row += (f"{v:10d}" if isinstance(v, int) else f"{100 * v:10.2f}")
        if extra:
            row += "".join(f"{extra[0].get(k, {}).get(e, float('nan')):12.2f}" for e in extra[1])
        lines.append(row)
    return "\n".join(lines)
