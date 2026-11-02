#!/usr/bin/env python3
"""
Step 7：检测调度 —— 在相同平均检测率（≈ 相同 CPU 算力预算）下，决定"哪些帧值得检测"

检测器固定为 YOLOv8n-ft@960（CPU 前沿主力）。未检测的帧由卡尔曼滤波预测。比较四种调度策略：

  fixed    固定间隔：每 k 帧检测一次（现有系统的常见做法）
  rule     几何漂移规则（基于模型）：
             漂移 = (距上次检测的帧数) × (卡尔曼估计的最大归一化速度，单位"自身高度/帧")
             漂移 > θ，或预测框之间严重重叠（IoU>0.3，人群交叉），或帧差能量超过阈值（相机运动/场景突变）→ 检测
  learned  学习型（数据驱动）：梯度提升回归，从轨迹状态 + 帧差特征预测"现在不检测会有多大漂移"，超过阈值则检测
             · 标签自监督：漂移 = 卡尔曼预测框与该帧检测框的不一致程度，不使用人工标注
             · 留一序列交叉验证：在 6 个序列上训练，在第 7 个上运行（评测序列训练时没见过）
  rule_motion  消融：只用运动漂移这一项（去掉人群重叠和帧差触发）
  hindsight 事后参照：直接用该帧真实漂移贪心决定（需要先跑检测才知道，不可实现；
             它不是严格上界，只说明"如果漂移能被完美预测"大概能到哪里）
所有自适应策略都有安全上限：连续 GAP_MAX 帧未检测则强制检测。

帧差能量：90×160 灰度缩略图与上一帧的平均绝对差，计算成本 < 1 ms/帧（结果缓存在 cache/framediff/）。
比较方式：每种策略扫一组阈值，得到（检测率，HOTA）曲线；在相同检测率处线性插值比较，并统计逐序列胜负。

用法：
  .\\.venv\\Scripts\\python.exe -m pip install scikit-learn
  .\\.venv\\Scripts\\python.exe step7_schedule.py        （约 20–40 分钟；首次需读图计算帧差）
输出：results/step7_report.txt（发回）、results/figures/schedule_q2.png、results/schedule_q2.csv、
      results/schedule_model.pkl（用全部 q2 训练的学习型模型，留给最后测试集使用）
"""
import argparse
import csv
import json
import pickle
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

import common as C
from trackers import ByteTracker, iou_matrix

SPLIT = "q2"
TRK_ROOT = C.RESULTS / SPLIT / "sched"
GAP_MAX = 10
CROWD_IOU = 0.3
FEATS = ["gap", "n_tracks", "speed_mean", "speed_max", "crowd_max", "crowd_frac", "n_lost",
         "fd_now", "fd_acc", "drift_mean", "drift_max", "box_h_mean"]


# ------------------------------------------------------------------ 帧差能量
def framediff(mot17, seq):
    path = C.CACHE / "framediff" / f"{seq}.npy"
    if path.exists():
        return np.load(path)
    info = C.seqinfo(mot17, seq)
    prev, out, cost = None, [], []
    for f in range(1, info["length"] + 1):
        img = cv2.imread(str(C.frame_path(mot17, info, f)), cv2.IMREAD_GRAYSCALE)
        a = time.perf_counter()
        small = cv2.resize(img, (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
        out.append(0.0 if prev is None else float(np.mean(np.abs(small - prev))))
        cost.append((time.perf_counter() - a) * 1000)
        prev = small
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.array(out, np.float32))
    np.save(path.with_name(f"{seq}_cost_ms.npy"), np.array(cost, np.float32))
    return np.array(out, np.float32)


# ------------------------------------------------------------------ 特征与自监督漂移标签
def features(trk, gap, fd_now, fd_acc):
    tr = trk.tracked()
    sp = trk.norm_speeds() if tr else np.zeros(1)
    pb = trk.predicted_boxes()
    if len(pb) > 1:
        iou = iou_matrix(pb, pb)
        np.fill_diagonal(iou, 0)
        crowd_max, crowd_frac = float(iou.max()), float(np.mean(iou.max(1) > CROWD_IOU))
    else:
        crowd_max = crowd_frac = 0.0
    n_lost = sum(1 for t in trk.tracks if t.state == 2)
    h = float(np.mean(pb[:, 3] - pb[:, 1])) if len(pb) else 0.0
    return [gap, len(tr), float(sp.mean()), float(sp.max()), crowd_max, crowd_frac, n_lost,
            fd_now, fd_acc, gap * float(sp.mean()), gap * float(sp.max()), h]


def drift(trk, dets):
    """卡尔曼预测框与该帧高分检测框的不一致程度（自监督，不用人工标注）。"""
    pb = trk.predicted_boxes()
    d = dets[dets[:, 4] >= 0.5, :4] if len(dets) else np.zeros((0, 4))
    n = max(len(pb), len(d))
    if n == 0:
        return 0.0
    if len(pb) == 0 or len(d) == 0:
        return 1.0
    iou = iou_matrix(pb, d)
    r, c = linear_sum_assignment(-iou)
    m = iou[r, c]
    m = m[m >= 0.3]
    unmatched = (len(pb) - len(m)) + (len(d) - len(m))
    return float((unmatched + np.sum(1 - m)) / n)


# ------------------------------------------------------------------ 策略运行
def run_policy(seq, dets, fd, a, b, policy, param, fd_thr=None, model=None, log=False):
    trk, gap, fd_acc = ByteTracker(), 0, 0.0
    rows, logs, n_det, t_ms = [], [], 0, []
    empty = np.zeros((0, 5), np.float32)
    for f in range(a, b + 1):
        d_f = dets.get(f, empty)
        fd_now = float(fd[f - 1])
        fd_acc += fd_now
        t0 = time.perf_counter()
        x = features(trk, gap + 1, fd_now, fd_acc)
        if f == a:
            detect = True
        elif policy == "fixed":
            detect = (f - a) % int(param) == 0
        elif gap + 1 >= GAP_MAX:
            detect = True
        elif policy == "rule":
            detect = x[10] > param or (x[4] > CROWD_IOU and gap + 1 >= 2) or fd_now > fd_thr
        elif policy == "rule_motion":
            detect = x[10] > param
        elif policy == "learned":
            detect = float(model.predict(np.array([x]))[0]) > param
        elif policy == "hindsight":
            detect = drift(trk, d_f) > param
        else:
            raise ValueError(policy)
        decide_ms = (time.perf_counter() - t0) * 1000
        if log:
            logs.append(x + [drift(trk, d_f)])
        t0 = time.perf_counter()
        out = trk.step(f, d_f if detect else None)
        t_ms.append((time.perf_counter() - t0) * 1000 + (decide_ms if policy in ("rule", "learned") else 0))
        if detect:
            n_det += 1
            gap, fd_acc = 0, 0.0
        else:
            gap += 1
        rows += [(f - a + 1, tid, x1, y1, x2 - x1, y2 - y1, sc) for tid, x1, y1, x2, y2, sc in out]
    return rows, n_det, b - a + 1, float(np.mean(t_ms)), logs


def interp(points, target):
    """points: [(rate, hota)] → 在 target 检测率处线性插值（不外推）。"""
    pts = sorted(points)
    for (r0, h0), (r1, h1) in zip(pts, pts[1:]):
        if r0 <= target <= r1:
            return h0 + (h1 - h0) * (target - r0) / max(r1 - r0, 1e-12)
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot17", default=C.DEFAULT_MOT17)
    ap.add_argument("--det", default="yolov8n-ft_960_none")
    args = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingRegressor
    t0 = time.monotonic()
    splits = json.loads((C.GT_HALF / "splits.json").read_text())
    cpu = json.loads((C.RESULTS / "latency_cpu.json").read_text())
    det_cpu = cpu[args.det]["median_ms"]
    data = {}
    for s in C.SEQS:
        dets, _ = C.load_dets(args.det, s)
        fd = framediff(args.mot17, s)
        a, b = splits[s][SPLIT]["first"], splits[s][SPLIT]["last"]
        data[s] = (dets, fd, a, b)
    fd_cost = float(np.mean(np.concatenate([np.load(C.CACHE / "framediff" / f"{s}_cost_ms.npy") for s in C.SEQS])))
    print(f"framediff ready ({fd_cost:.2f} ms/frame)", flush=True)

    # 1) 采集自监督训练数据：在每个序列上用多种固定间隔运行，逐帧记录（特征，真实漂移）
    logs = {s: [] for s in C.SEQS}
    for s, (dets, fd, a, b) in data.items():
        for k in (2, 4, 8):
            logs[s] += run_policy(s, dets, fd, a, b, "fixed", k, log=True)[4]
    logs = {s: np.array(v, dtype=float) for s, v in logs.items()}
    all_fd = np.concatenate([data[s][1][data[s][2] - 1:data[s][3]] for s in C.SEQS])
    fd_thr = float(np.percentile(all_fd, 95))

    # 2) 留一序列交叉验证训练学习型模型；另训练一个全量模型留给测试集
    def fit(X, y):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0)
        return m.fit(X, y)
    models, cv_r2 = {}, {}
    for s in C.SEQS:
        tr = np.vstack([logs[t] for t in C.SEQS if t != s])
        models[s] = fit(tr[:, :-1], tr[:, -1])
        te = logs[s]
        pred = models[s].predict(te[:, :-1])
        ss = np.sum((te[:, -1] - te[:, -1].mean()) ** 2)
        cv_r2[s] = float(1 - np.sum((te[:, -1] - pred) ** 2) / max(ss, 1e-12))
    full = np.vstack(list(logs.values()))
    with open(C.RESULTS / "schedule_model.pkl", "wb") as f:
        pickle.dump({"model": fit(full[:, :-1], full[:, -1]), "features": FEATS, "fd_thr": fd_thr,
                     "det": args.det, "gap_max": GAP_MAX}, f)
    print("learned models trained", flush=True)

    # 3) 扫描阈值
    drift_all = full[:, -1]
    q = list(np.unique(np.percentile(drift_all, [5, 15, 25, 35, 45, 55, 65, 75, 83, 90, 95, 98])))
    theta = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5]
    grid = {"fixed": [1, 2, 3, 4, 5, 6, 8, 10], "rule": theta, "rule_motion": theta,
            "learned": q, "hindsight": q}
    runs = {}
    for pol, params in grid.items():
        for p in params:
            name = f"{pol}_{p:.4g}"
            tot_det = tot_fr = 0
            per_rate, tms = {}, []
            for s, (dets, fd, a, b) in data.items():
                rows, nd, nf, tm, _ = run_policy(s, dets, fd, a, b, pol, p, fd_thr=fd_thr,
                                                 model=models[s] if pol == "learned" else None)
                C.write_mot(TRK_ROOT / name / "data" / f"{s}.txt", rows)
                tot_det += nd; tot_fr += nf; per_rate[s] = nd / nf; tms.append(tm)
            runs[name] = {"policy": pol, "param": p, "rate": tot_det / tot_fr, "per_rate": per_rate,
                          "track_ms": float(np.mean(tms))}
        print(f"  ran {pol}", flush=True)
    print("TrackEval...", flush=True)
    summary, per_seq = C.evaluate(SPLIT, TRK_ROOT, list(runs))
    for n, r in runs.items():
        r.update(summary[n])
        extra = fd_cost if r["policy"] in ("rule", "learned") else 0.0
        r["cpu_ms"] = r["rate"] * det_cpu + r["track_ms"] + extra

    # 4) 相同检测率下的比较
    targets = [0.15, 0.2, 0.25, 0.33, 0.5]
    curves = {pol: [(r["rate"], r["HOTA"]) for r in runs.values() if r["policy"] == pol] for pol in grid}
    L = ["# Step 7 报告：检测调度（MOT17 各序列 25%–50%，YOLOv8n-ft@960，CPU 4 线程）",
         f"detector CPU {det_cpu:.1f} ms/detection | framediff {fd_cost:.2f} ms/frame | fd_thr(p95)={fd_thr:.2f} | GAP_MAX={GAP_MAX}",
         "learned model leave-one-sequence-out R² (self-supervised drift): "
         + ", ".join(f"{s[-2:]}={v:.2f}" for s, v in cv_r2.items()), "",
         "## 全部运行（按策略、检测率排序）",
         "policy   param     rate   HOTA   DetA   AssA   MOTA   IDF1  IDSW  CPU ms/frame"]
    for n, r in sorted(runs.items(), key=lambda x: (x[1]["policy"], x[1]["rate"])):
        L.append(f"{r['policy']:<8} {r['param']:<8.4g} {r['rate']:5.3f}  {100 * r['HOTA']:5.2f}  {100 * r['DetA']:5.2f}  "
                 f"{100 * r['AssA']:5.2f}  {100 * r['MOTA']:5.2f}  {100 * r['IDF1']:5.2f}  {r['IDSW']:4d}  {r['cpu_ms']:7.1f}")
    L += ["", "## 相同检测率下的 HOTA（%，线性插值；括号内为相对固定间隔的差）",
          "rate    CPU ms   " + "".join(f"{p:>17}" for p in grid)]
    for t in targets:
        base = interp(curves["fixed"], t)
        row = f"{t:4.2f}  {t * det_cpu + 0.5:7.1f}   "
        for p in grid:
            v = interp(curves[p], t)
            row += f"{100 * v:9.2f}({100 * (v - base):+5.2f})" if np.isfinite(v) else f"{'n/a':>17}"
        L.append(row)
    L += ["", "## 逐序列：相同检测率下相对固定间隔的 HOTA 差（%）"]
    for p in ("rule", "rule_motion", "learned", "hindsight"):
        for t in (0.2, 0.33):
            diffs = []
            for s in C.SEQS:
                fx = [(r["per_rate"][s], per_seq[n][s]["HOTA"]) for n, r in runs.items() if r["policy"] == "fixed"]
                px = [(r["per_rate"][s], per_seq[n][s]["HOTA"]) for n, r in runs.items() if r["policy"] == p]
                diffs.append(interp(px, t) - interp(fx, t))
            ok = [d for d in diffs if np.isfinite(d)]
            L.append(f"{p:<8} rate {t:.2f}: wins {sum(d > 0 for d in ok)}/{len(ok)}  | "
                     + "  ".join(f"{s[-2:]}:{100 * d:+.2f}" if np.isfinite(d) else f"{s[-2:]}:n/a" for s, d in zip(C.SEQS, diffs)))

    with open(C.RESULTS / "schedule_q2.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "policy", "param", "rate", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "cpu_ms"])
        for n, r in runs.items():
            w.writerow([n, r["policy"], r["param"], r["rate"], r["HOTA"], r["DetA"], r["AssA"], r["MOTA"], r["IDF1"], r["IDSW"], r["cpu_ms"]])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 5))
        style = {"fixed": ("#555555", "o", "Fixed interval"), "rule": ("#1f77b4", "s", "Geometric-drift rule (model-based)"),
                 "learned": ("#d62728", "^", "Learned drift predictor (self-supervised)"),
                 "rule_motion": ("#9ecae1", "s", "Rule, motion term only (ablation)"),
                 "hindsight": ("#2ca02c", "x", "Hindsight drift (not implementable)")}
        for p, (col, mk, lab) in style.items():
            pts = sorted(curves[p])
            ax.plot([100 * r for r, _ in pts], [100 * h for _, h in pts], marker=mk, color=col, label=lab, lw=1.5,
                    ls="--" if p in ("hindsight", "rule_motion") else "-")
        ax.set_xlabel("frames sent to the detector (%)")
        ax.set_ylabel("HOTA (%)")
        sec = ax.secondary_xaxis("top", functions=(lambda x: x / 100 * det_cpu, lambda m: m / det_cpu * 100))
        sec.set_xlabel("≈ mean detector cost per frame on CPU (ms)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title("Detection scheduling at matched compute (MOT17, 25–50%)", fontsize=10)
        fig.tight_layout()
        (C.RESULTS / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(C.RESULTS / "figures" / "schedule_q2.png", dpi=160)
        L.append("\nfigure: results/figures/schedule_q2.png")
    except Exception as e:
        L.append(f"[plot failed: {e}]")
    L.append(f"total time: {time.monotonic() - t0:.0f}s")
    (C.RESULTS / "step7_report.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n已保存：{C.RESULTS / 'step7_report.txt'}")


if __name__ == "__main__":
    main()
