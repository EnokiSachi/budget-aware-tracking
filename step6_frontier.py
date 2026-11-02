#!/usr/bin/env python3
"""
Step 6：精度—延迟前沿（在 MOT17 每个序列的 25%–50%，即 q2 上评测）

对每个检测配置（检测器 × 分辨率，读取检测缓存）和每个检测间隔 k（每 k 帧检测一次，其余帧卡尔曼预测），
运行自实现 ByteTrack，用 TrackEval 计算 HOTA / MOTA / IDF1，并计算每帧平均延迟：
    每帧延迟 = 检测耗时 / k + 跟踪耗时
检测耗时两种硬件：GPU（缓存中逐帧实测）和 4 线程 CPU（step5 实测，低功耗设备代理）。
找出帕累托前沿（同等或更低延迟下没有更高 HOTA 的点），并回答：
  - 微调后的小检测器，在相同或更低延迟下，能否超过未微调的大检测器？
  - 固定预算下，"每帧都用小检测器" 与 "隔帧用大检测器" 哪个更好？

用法：.\\.venv\\Scripts\\python.exe step6_frontier.py     （约 10–20 分钟，主要是 TrackEval）
输出：results/step6_report.txt（发回）、results/figures/frontier_q2.png、results/q2_frontier.csv
"""
import argparse
import json
import time

import numpy as np

import common as C
from trackers import ByteTracker

SPLIT = "q2"
TRK_ROOT = C.RESULTS / SPLIT / "trackers"
KS = (1, 2, 3, 5, 8)


def available_keys():
    keys = []
    for d in sorted(C.CACHE.iterdir()):
        if d.is_dir() and d.name.endswith("_none") and all((d / f"{s}.npz").exists() for s in C.SEQS):
            keys.append(d.name)
    return keys


def run(key, k, splits):
    name = f"{key}_k{k}"
    t_ms, det_ms, n_det, n_frames = [], [], 0, 0
    for s in C.SEQS:
        dets, meta = C.load_dets(key, s)
        wall = meta["det_wall_ms"]
        a, b = splits[s][SPLIT]["first"], splits[s][SPLIT]["last"]
        trk, rows = ByteTracker(), []
        for f in range(a, b + 1):
            use = (f - a) % k == 0
            t0 = time.perf_counter()
            out = trk.step(f, dets.get(f, np.zeros((0, 5), np.float32)) if use else None)
            t_ms.append((time.perf_counter() - t0) * 1000)
            if use:
                det_ms.append(float(wall[f - 1]))
                n_det += 1
            n_frames += 1
            rows += [(f - a + 1, tid, x1, y1, x2 - x1, y2 - y1, sc) for tid, x1, y1, x2, y2, sc in out]
        C.write_mot(TRK_ROOT / name / "data" / f"{s}.txt", rows)
    return name, {"track_ms": float(np.mean(t_ms)), "det_gpu_ms": float(np.mean(det_ms)),
                  "det_rate": n_det / n_frames}


def pareto(points):
    """points: [(latency, score, name)] → 前沿上的 name 集合。"""
    front, best = set(), -1
    for lat, sc, name in sorted(points, key=lambda p: (p[0], -p[1])):
        if sc > best:
            front.add(name)
            best = sc
    return front


def parse(key):
    model, size = key.split("_")[0].replace("yolov8", ""), int(key.split("_")[1])
    return model, size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", nargs="+", type=int, default=list(KS))
    args = ap.parse_args()
    t0 = time.monotonic()
    splits = json.loads((C.GT_HALF / "splits.json").read_text())
    if SPLIT not in splits[C.SEQS[0]]:
        raise SystemExit("splits.json 里没有 q2，请先重新运行 step1_prepare.py")
    cpu = json.loads((C.RESULTS / "latency_cpu.json").read_text()) if (C.RESULTS / "latency_cpu.json").exists() else {}
    keys = available_keys()
    print(f"configs: {keys}", flush=True)
    info = {}
    for key in keys:
        for k in args.ks:
            name, d = run(key, k, splits)
            model, size = parse(key)
            d.update(model=model, imgsz=size, k=k, key=key)
            d["lat_gpu"] = d["det_gpu_ms"] * d["det_rate"] + d["track_ms"]
            d["lat_cpu"] = (cpu[key]["median_ms"] * d["det_rate"] + d["track_ms"]) if key in cpu else float("nan")
            info[name] = d
        print(f"  tracked {key}", flush=True)
    print("TrackEval...", flush=True)
    summary, _ = C.evaluate(SPLIT, TRK_ROOT, list(info))
    for n in info:
        info[n].update(summary[n])

    front_gpu = pareto([(d["lat_gpu"], d["HOTA"], n) for n, d in info.items()])
    front_cpu = pareto([(d["lat_cpu"], d["HOTA"], n) for n, d in info.items() if np.isfinite(d["lat_cpu"])])

    import csv
    cols = ["key", "model", "imgsz", "k", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "Recall", "Precision",
            "det_gpu_ms", "track_ms", "lat_gpu", "lat_cpu"]
    with open(C.RESULTS / "q2_frontier.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + ["front_gpu", "front_cpu"])
        for n, d in info.items():
            w.writerow([d[c] for c in cols] + [n in front_gpu, n in front_cpu])

    def row(n):
        d = info[n]
        return (f"{d['model']:>5} {d['imgsz']:5d} k={d['k']:<2d} HOTA {100 * d['HOTA']:5.2f}  DetA {100 * d['DetA']:5.2f}  "
                f"AssA {100 * d['AssA']:5.2f}  MOTA {100 * d['MOTA']:6.2f}  IDF1 {100 * d['IDF1']:5.2f}  IDSW {d['IDSW']:4d}  "
                f"Rec {100 * d['Recall']:5.2f}  | GPU {d['lat_gpu']:6.1f} ms  CPU {d['lat_cpu']:7.1f} ms")
    L = ["# Step 6 报告：精度—延迟前沿（MOT17 各序列 25%–50%，自实现 ByteTrack）",
         f"configs: {len(keys)} detector settings × k∈{list(args.ks)} = {len(info)} runs",
         f"CPU latency: {cpu.get('_meta', 'not measured')}", "",
         "## 全部结果（按 GPU 延迟排序）"]
    L += [row(n) for n in sorted(info, key=lambda n: info[n]["lat_gpu"])]
    L += ["", "## GPU 前沿"] + [row(n) for n in sorted(front_gpu, key=lambda n: info[n]["lat_gpu"])]
    if front_cpu:
        L += ["", "## CPU（4 线程）前沿"] + [row(n) for n in sorted(front_cpu, key=lambda n: info[n]["lat_cpu"])]

    # 问题：微调小模型 vs 未微调大模型（相同或更低延迟）
    L += ["", "## 微调 vs 未微调（k=1，每帧检测；对每个微调配置，找延迟不高于它的最好未微调配置）"]
    for hw in ("lat_gpu", "lat_cpu"):
        for n, d in sorted(info.items(), key=lambda x: x[1][hw]):
            if not d["model"].endswith("-ft") or d["k"] != 1 or not np.isfinite(d[hw]):
                continue
            cands = [(m, e) for m, e in info.items() if not e["model"].endswith("-ft") and e["k"] == 1 and e[hw] <= d[hw]]
            if cands:
                m, e = max(cands, key=lambda x: x[1]["HOTA"])
                L.append(f"[{hw[4:].upper()}] {d['model']}@{d['imgsz']} HOTA {100 * d['HOTA']:.2f} ({d[hw]:.1f} ms)  vs  "
                         f"best pretrained ≤ that latency: {e['model']}@{e['imgsz']} HOTA {100 * e['HOTA']:.2f} ({e[hw]:.1f} ms)")
    # 实时预算下的最优配置
    L += ["", "## 各预算下的最优配置（每帧平均延迟 ≤ 预算）"]
    for hw in ("lat_gpu", "lat_cpu"):
        for budget in (10, 20, 33.3, 50, 100):
            ok = [(n, d) for n, d in info.items() if np.isfinite(d[hw]) and d[hw] <= budget]
            if ok:
                n, d = max(ok, key=lambda x: x[1]["HOTA"])
                L.append(f"[{hw[4:].upper()} ≤{budget:5.1f} ms] {d['model']}@{d['imgsz']} k={d['k']}  HOTA {100 * d['HOTA']:.2f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        models = sorted({d["model"] for d in info.values()})
        cmap = plt.get_cmap("tab10")
        for ax, hw, front, title in ((axes[0], "lat_gpu", front_gpu, "GPU (RTX 3070 Ti Laptop)"),
                                     (axes[1], "lat_cpu", front_cpu, "CPU, 4 threads (low-power proxy)")):
            for i, m in enumerate(models):
                pts = [d for d in info.values() if d["model"] == m and np.isfinite(d[hw])]
                ax.scatter([d[hw] for d in pts], [100 * d["HOTA"] for d in pts], s=[d["imgsz"] / 20 for d in pts],
                           color=cmap(i), alpha=0.75, label=f"YOLOv8{m}")
            fp = sorted((info[n] for n in front), key=lambda d: d[hw])
            ax.plot([d[hw] for d in fp], [100 * d["HOTA"] for d in fp], "k--", lw=1, label="Pareto front")
            for d in fp:
                ax.annotate(f"{d['imgsz']},k{d['k']}", (d[hw], 100 * d["HOTA"]), fontsize=7, xytext=(3, 3),
                            textcoords="offset points")
            ax.set_xscale("log")
            from matplotlib.ticker import FuncFormatter
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if str(v)[0] in "25" else ""))
            ax.set_xlabel("mean latency per frame (ms, log scale)")
            ax.set_ylabel("HOTA (%)")
            ax.set_title(title)
            ax.axvline(33.3, color="gray", lw=0.8, ls=":")
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.suptitle("Accuracy–latency trade-off, MOT17 (25–50% of each sequence)")
        fig.tight_layout()
        (C.RESULTS / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(C.RESULTS / "figures" / "frontier_q2.png", dpi=160)
        L.append("\nfigure: results/figures/frontier_q2.png  (marker size ∝ input size; dotted line = 30 FPS)")
    except Exception as e:
        L.append(f"[plot failed: {e}]")
    L.append(f"total time: {time.monotonic() - t0:.0f}s")
    (C.RESULTS / "step6_report.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n已保存：{C.RESULTS / 'step6_report.txt'}")


if __name__ == "__main__":
    main()
