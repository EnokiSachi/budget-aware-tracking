#!/usr/bin/env python3
"""
Step 8：分段检测频率分配 —— 保留均匀采样，只让不同时段使用不同间隔，在总预算内统一分配

Step 7 发现：逐帧"按需触发"连事后参照都赢不了固定间隔（均匀采样对卡尔曼滤波有结构性优势）。
这里改为：把视频切成约 2 秒一段，每段内仍按固定间隔 k∈{1,2,3,5,8} 均匀检测，不同段可用不同 k；
在总检测预算内用拉格朗日松弛做"多选一背包"分配（与 IEEE 论文的预算约束配置选择同一框架）：
    每段选 k = argmax  收益(k) − λ · 该段检测次数(k)，扫描 λ 得到不同预算
三种收益来源：
  hindsight  事后最优：用人工标注算出每段在各 k 下的真实收益（不可实现，用来量化"分段分配"本身的上限）
  rule       基于模型（采样定理类比）：每段检测率与估计的运动"带宽"成正比，k ≈ c / 风险；
             风险 = 上一段帧差能量 + 卡尔曼估计的最大运动速度（各自除以全体均值，无需标注）；扫描 c 得到不同预算
  learned    学习型：梯度提升，从段起点可得的特征预测各 k 的收益（留一序列交叉验证）
基线 fixed：所有段同一个 k。

收益的代理指标（逐段、可相加）：TP − FP − IDSW（即 MOTA 的分子），由各固定间隔运行结果与人工标注逐帧匹配得到。
最终比较仍用 TrackEval 的 HOTA / MOTA / IDF1，在相同检测率处插值。

依赖 step6（固定间隔运行结果）与 step7（帧差缓存）。
用法：.\\.venv\\Scripts\\python.exe step8_allocation.py      （约 20–40 分钟）
输出：results/step8_report.txt（发回）、results/figures/allocation_q2.png
"""
import argparse
import json
import math
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

import common as C
from step7_schedule import framediff
from trackers import ByteTracker, iou_matrix

SPLIT = "q2"
KS = [1, 2, 3, 5, 8]
SEG_SEC = 2.0
WARMUP_K = 2
TRK6 = C.RESULTS / SPLIT / "trackers"
TRK8 = C.RESULTS / SPLIT / "alloc"
SEG_FEATS = ["fd_prev_mean", "fd_prev_max", "n_tracks", "speed_mean", "speed_max", "crowd_max", "crowd_frac", "box_h_mean"]


def load_mot(path):
    a = np.loadtxt(path, delimiter=",", ndmin=2) if path.exists() and path.stat().st_size else np.zeros((0, 10))
    return a


def frame_counts(gt, res, n):
    """逐帧 TP/FP/FN/IDSW（IoU≥0.5 匈牙利匹配；GT 只计 consider=1 的行人）。"""
    gt = gt[(gt[:, 6] == 1) & (gt[:, 7] == 1)]
    out = np.zeros((n + 1, 4))
    last = {}
    for f in range(1, n + 1):
        g = gt[gt[:, 0] == f]
        r = res[res[:, 0] == f] if len(res) else res
        gb = np.c_[g[:, 2], g[:, 3], g[:, 2] + g[:, 4], g[:, 3] + g[:, 5]]
        rb = np.c_[r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5]] if len(r) else np.zeros((0, 4))
        tp = idsw = 0
        if len(g) and len(rb):
            iou = iou_matrix(gb, rb)
            ri, ci = linear_sum_assignment(-iou)
            for i, j in zip(ri, ci):
                if iou[i, j] >= 0.5:
                    tp += 1
                    gid, tid = int(g[i, 1]), int(r[j, 1])
                    if gid in last and last[gid] != tid:
                        idsw += 1
                    last[gid] = tid
        out[f] = [tp, len(rb) - tp, len(g) - tp, idsw]
    return out


def segments(n, fps):
    w = max(1, int(round(SEG_SEC * fps)))
    return [(s, min(s + w - 1, n)) for s in range(1, n + 1, w)]


def n_det(seg, k):
    return math.ceil((seg[1] - seg[0] + 1) / k)


def seg_features(trk, fd, a, s0, w):
    """段起点（相对帧号 s0）可得的因果特征：上一段帧差 + 当前轨迹状态。"""
    lo, hi = max(1, s0 - w), s0 - 1
    prev = fd[a - 1 + lo - 1: a - 1 + hi] if hi >= lo else np.zeros(0)
    if len(prev) == 0:
        prev = np.zeros(1)
    tr = trk.tracked()
    sp = trk.norm_speeds() if tr else np.zeros(1)
    pb = trk.predicted_boxes()
    if len(pb) > 1:
        iou = iou_matrix(pb, pb)
        np.fill_diagonal(iou, 0)
        cm, cf = float(iou.max()), float(np.mean(iou.max(1) > 0.3))
    else:
        cm = cf = 0.0
    h = float(np.mean(pb[:, 3] - pb[:, 1])) if len(pb) else 0.0
    return [float(np.mean(prev)), float(np.max(prev)), len(tr), float(sp.mean()), float(sp.max()), cm, cf, h]


def execute(dets, fd, a, b, segs, choose):
    """choose(i, feat) → 第 i 段的 k。返回 (MOT 行, 检测次数, 每段特征)。"""
    trk, rows, nd, feats = ByteTracker(), [], 0, []
    empty = np.zeros((0, 5), np.float32)
    w = segs[0][1] - segs[0][0] + 1
    for i, (s0, s1) in enumerate(segs):
        x = seg_features(trk, fd, a, s0, w)
        feats.append(x)
        k = choose(i, x)
        for rf in range(s0, s1 + 1):
            f = a + rf - 1
            use = (rf - s0) % k == 0
            out = trk.step(f, dets.get(f, empty) if use else None)
            nd += use
            rows += [(rf, tid, x1, y1, x2 - x1, y2 - y1, sc) for tid, x1, y1, x2, y2, sc in out]
    return rows, nd, feats


def allocate(gains, segs_all, lam):
    """gains[s][i] = {k: 收益}；返回 {s: [每段 k]}。"""
    return {s: [max(KS, key=lambda k: g[k] - lam * n_det(seg, k)) for g, seg in zip(gains[s], segs_all[s])]
            for s in gains}


def interp(points, target):
    pts = sorted(points)
    for (r0, h0), (r1, h1) in zip(pts, pts[1:]):
        if r0 <= target <= r1:
            return h0 + (h1 - h0) * (target - r0) / max(r1 - r0, 1e-12)
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot17", default=C.DEFAULT_MOT17)
    ap.add_argument("--det", default="yolov8n-ft_960_none")
    ap.add_argument("--seg-sec", type=float, default=SEG_SEC)
    args = ap.parse_args()
    globals()["SEG_SEC"] = args.seg_sec
    from sklearn.ensemble import HistGradientBoostingRegressor
    t0 = time.monotonic()
    splits = json.loads((C.GT_HALF / "splits.json").read_text())
    det_cpu = json.loads((C.RESULTS / "latency_cpu.json").read_text())[args.det]["median_ms"]
    data, segs_all, true_gain = {}, {}, {}
    for s in C.SEQS:
        dets, _ = C.load_dets(args.det, s)
        fd = framediff(args.mot17, s)
        a, b = splits[s][SPLIT]["first"], splits[s][SPLIT]["last"]
        n = b - a + 1
        data[s] = (dets, fd, a, b)
        segs = segments(n, splits[s]["fps"])
        segs_all[s] = segs
        gt = load_mot(C.GT_HALF / SPLIT / s / "gt" / "gt.txt")
        per_k = {}
        for k in KS:
            res = load_mot(TRK6 / f"{args.det}_k{k}" / "data" / f"{s}.txt")
            cnt = frame_counts(gt, res, n)
            per_k[k] = [cnt[s0:s1 + 1].sum(0) for s0, s1 in segs]
        true_gain[s] = [{k: float(per_k[k][i][0] - per_k[k][i][1] - per_k[k][i][3]) for k in KS} for i in range(len(segs))]
    print("hindsight segment gains ready", flush=True)

    # 学习型：记录段特征（固定 k=3 运行中采集），目标 = 各 k 相对 k=8 的真实收益
    logX = {s: execute(*data[s], segs_all[s], lambda i, x: 3)[2] for s in C.SEQS}
    Y = {s: np.array([[g[k] - g[8] for k in KS] for g in true_gain[s]]) for s in C.SEQS}
    models = {}
    for s in C.SEQS:
        X = np.vstack([np.array(logX[t]) for t in C.SEQS if t != s])
        Yt = np.vstack([Y[t] for t in C.SEQS if t != s])
        models[s] = [HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=0).fit(X, Yt[:, j])
                     for j in range(len(KS))]
    allX = np.vstack([np.array(v) for v in logX.values()])
    mean_fd = float(np.mean(allX[:, 0])) + 1e-6
    mean_sp = float(np.mean(allX[:, 4])) + 1e-6

    # λ 网格：以"每多一次检测带来的收益"的中位数为尺度
    ratios = [(g[1] - g[8]) / max(n_det(seg, 1) - n_det(seg, 8), 1)
              for s in C.SEQS for g, seg in zip(true_gain[s], segs_all[s])]
    scale = max(float(np.median(np.abs(ratios))), 1e-3)
    lams = list(scale * np.logspace(-1.5, 1.2, 12))

    runs = {}

    def record(name, policy, param, chooser_for):
        tot_d = tot_f = 0
        per_rate = {}
        for s in C.SEQS:
            rows, nd, _ = execute(*data[s], segs_all[s], chooser_for(s))
            C.write_mot(TRK8 / name / "data" / f"{s}.txt", rows)
            nf = data[s][3] - data[s][2] + 1
            tot_d += nd; tot_f += nf; per_rate[s] = nd / nf
        runs[name] = {"policy": policy, "param": param, "rate": tot_d / tot_f, "per_rate": per_rate}

    for k in [1, 2, 3, 4, 5, 6, 8, 10]:
        record(f"fixed_{k}", "fixed", k, lambda s, k=k: (lambda i, x: k))
    for j, lam in enumerate(lams):
        plan = allocate(true_gain, segs_all, lam)
        record(f"hindsight_{j}", "hindsight", lam, lambda s, plan=plan: (lambda i, x: plan[s][i]))
    for j, c in enumerate([0.6, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 14, 20, 30]):
        def rule_for(s, c=c):
            def ch(i, x):
                if i == 0:
                    return WARMUP_K  # 首段跟踪器为空，所有自适应策略统一用固定预热间隔
                risk = 0.5 * (x[0] / mean_fd + x[4] / mean_sp)  # 全体平均约为 1
                return min(KS, key=lambda k: abs(k - c / max(risk, 1e-3)))
            return ch
        record(f"rule_{j}", "rule", c, rule_for)
    for j, lam in enumerate(lams):
        def learned_for(s, lam=lam):
            def ch(i, x):
                if i == 0:
                    return WARMUP_K
                pred = [m.predict(np.array([x]))[0] for m in models[s]]
                best = max(range(len(KS)), key=lambda q: pred[q] - lam * n_det(segs_all[s][i], KS[q]))
                return KS[best]
            return ch
        record(f"learned_{j}", "learned", lam, learned_for)
    print("TrackEval...", flush=True)
    summary, per_seq = C.evaluate(SPLIT, TRK8, list(runs))
    for n_, r in runs.items():
        r.update(summary[n_])

    pols = ["fixed", "rule", "learned", "hindsight"]
    curves = {p: [(r["rate"], r["HOTA"]) for r in runs.values() if r["policy"] == p] for p in pols}
    curves_m = {p: [(r["rate"], r["MOTA"]) for r in runs.values() if r["policy"] == p] for p in pols}
    L = ["# Step 8 报告：分段检测频率分配（MOT17 各序列 25%–50%，YOLOv8n-ft@960）",
         f"segment ≈ {SEG_SEC}s, k ∈ {KS}, detector CPU {det_cpu:.1f} ms/detection, λ scale {scale:.3f}", "",
         "## 全部运行",
         "policy     rate   HOTA   DetA   AssA   MOTA   IDF1  IDSW"]
    for n_, r in sorted(runs.items(), key=lambda x: (x[1]["policy"], x[1]["rate"])):
        L.append(f"{r['policy']:<10} {r['rate']:5.3f}  {100 * r['HOTA']:5.2f}  {100 * r['DetA']:5.2f}  {100 * r['AssA']:5.2f}  "
                 f"{100 * r['MOTA']:5.2f}  {100 * r['IDF1']:5.2f}  {r['IDSW']:4d}")
    for metric, cv in (("HOTA", curves), ("MOTA", curves_m)):
        L += ["", f"## 相同检测率下的 {metric}（%，插值；括号内为相对固定间隔的差）",
              "rate   CPU ms  " + "".join(f"{p:>17}" for p in pols)]
        for t in (0.15, 0.2, 0.25, 0.33, 0.5):
            base = interp(cv["fixed"], t)
            row = f"{t:4.2f} {t * det_cpu + 0.5:7.1f}  "
            for p in pols:
                v = interp(cv[p], t)
                row += f"{100 * v:9.2f}({100 * (v - base):+5.2f})" if np.isfinite(v) else f"{'n/a':>17}"
            L.append(row)
    L += ["", "## 逐序列：相同检测率下相对固定间隔的 HOTA 差（%）"]
    for p in ("rule", "learned", "hindsight"):
        for t in (0.2, 0.33):
            diffs = []
            for s in C.SEQS:
                fx = [(r["per_rate"][s], per_seq[n_][s]["HOTA"]) for n_, r in runs.items() if r["policy"] == "fixed"]
                px = [(r["per_rate"][s], per_seq[n_][s]["HOTA"]) for n_, r in runs.items() if r["policy"] == p]
                diffs.append(interp(px, t) - interp(fx, t))
            ok = [d for d in diffs if np.isfinite(d)]
            L.append(f"{p:<9} rate {t:.2f}: wins {sum(d > 0 for d in ok)}/{len(ok)} | "
                     + "  ".join(f"{s[-2:]}:{100 * d:+.2f}" if np.isfinite(d) else f"{s[-2:]}:n/a" for s, d in zip(C.SEQS, diffs)))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 5))
        sty = {"fixed": ("#555555", "o", "-", "Fixed interval (all segments)"),
               "rule": ("#1f77b4", "s", "-", "Segment allocation, model-based risk"),
               "learned": ("#d62728", "^", "-", "Segment allocation, learned gains (LOSO)"),
               "hindsight": ("#2ca02c", "x", "--", "Segment allocation, hindsight gains (bound)")}
        for p, (col, mk, ls, lab) in sty.items():
            pts = sorted(curves[p])
            ax.plot([100 * r for r, _ in pts], [100 * h for _, h in pts], marker=mk, color=col, ls=ls, label=lab)
        ax.set_xlabel("frames sent to the detector (%)")
        ax.set_ylabel("HOTA (%)")
        sec = ax.secondary_xaxis("top", functions=(lambda x: x / 100 * det_cpu, lambda m: m / det_cpu * 100))
        sec.set_xlabel("≈ mean detector cost per frame on CPU (ms)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title("Budgeted allocation of detection rate across 2-s segments (MOT17, 25–50%)", fontsize=10)
        fig.tight_layout()
        fig.savefig(C.RESULTS / "figures" / "allocation_q2.png", dpi=160)
        L.append("\nfigure: results/figures/allocation_q2.png")
    except Exception as e:
        L.append(f"[plot failed: {e}]")
    L.append(f"total time: {time.monotonic() - t0:.0f}s")
    (C.RESULTS / "step8_report.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n已保存：{C.RESULTS / 'step8_report.txt'}")


if __name__ == "__main__":
    main()
