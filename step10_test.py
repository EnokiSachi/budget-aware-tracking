#!/usr/bin/env python3
"""
Step 10：预注册的一次性测试集评测（MOT17 每个序列的后 50%，此前从未评测过）

运行前把全部假设、配置和比较方式写入 results/test/PROTOCOL.json（锁定），然后只跑一次。
已完成时拒绝重跑（--allow-rerun 会把旧结果改名保留，并在协议中记录重跑次数）。所有配置都在前 50% 上选定。

H1 跟踪器：同一检测器（YOLOv8n@640）下，自实现 ByteTrack 优于毕设 DeepSORT（HOTA、IDF1 更高，IDSW 更少）
H2 领域适配：微调检测器在不高于其延迟的条件下优于最好的未微调检测器（GPU 与 CPU 两种延迟）
H3 预算分配：CPU 预算约 40 ms/帧时，"强检测器隔帧"（n-ft@960, k=2）优于"弱检测器每帧"（n-ft@640, k=1），且更省
H4 自适应调度（预期为零结果）：逐帧学习型触发、分段规则分配，在相同检测率下不优于固定间隔（差距在 ±1 HOTA 内）
H5 跨场景泛化：在训练从未出现的 3 个场景上，跨场景微调 nx-ft@960 优于未微调 n@960；
   与同场景微调 n-ft@960 的差距，量化"熟悉场景"带来的那部分提升

用法：.\\.venv\\Scripts\\python.exe step10_test.py     （约 20–40 分钟，DeepSORT 需逐帧读图）
输出：results/test/test_report.txt（发回）、results/figures/test_schedule.png、results/test/PROTOCOL.json
"""
import argparse
import csv
import datetime
import hashlib
import json
import pickle
import time

import cv2
import numpy as np

import common as C
import step7_schedule as S7
import step8_allocation as S8
from step2_detect import thesis_preprocess
from trackers import ByteTracker

SPLIT = "test"
OUT = C.RESULTS / "test"
TRK = OUT / "final"
FIXED_KS = [1, 2, 3, 4, 5, 6, 8, 10]
SEG_C = [0.6, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 14, 20, 30]
GRID = [(m, s) for m in ("n", "s", "m", "n-ft", "s-ft") for s in (640, 960, 1280)]
FT = "yolov8n-ft_960_none"
NFT640 = "grid_yolov8n-ft_640_none"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def byte_run(key, name, splits, k=1, seqs=None):
    det_ms, n_det = [], 0
    for s in seqs or C.SEQS:
        dets, meta = C.load_dets(key, s)
        a, b = splits[s][SPLIT]["first"], splits[s][SPLIT]["last"]
        trk, rows = ByteTracker(), []
        for f in range(a, b + 1):
            use = (f - a) % k == 0
            out = trk.step(f, dets.get(f, np.zeros((0, 5), np.float32)) if use else None)
            if use:
                det_ms.append(float(meta["det_wall_ms"][f - 1])); n_det += 1
            rows += [(f - a + 1, tid, x1, y1, x2 - x1, y2 - y1, sc) for tid, x1, y1, x2, y2, sc in out]
        C.write_mot(TRK / name / "data" / f"{s}.txt", rows)
    return {"det_gpu_ms": float(np.mean(det_ms)), "k": k, "key": key}


def deepsort_run(key, name, splits, mot17):
    from deep_sort_realtime.deepsort_tracker import DeepSort
    for s in C.SEQS:
        info = C.seqinfo(mot17, s)
        dets, _ = C.load_dets(key, s)
        a, b = splits[s][SPLIT]["first"], splits[s][SPLIT]["last"]
        trk, rows = DeepSort(max_age=30, n_init=3, max_cosine_distance=0.4), []
        for f in range(a, b + 1):
            img = thesis_preprocess(cv2.imread(str(C.frame_path(mot17, info, f))))
            d = dets.get(f, np.zeros((0, 5), np.float32))
            d = d[d[:, 4] >= 0.5]
            inp = [([float(x1), float(y1), float(x2 - x1), float(y2 - y1)], float(sc), "person") for x1, y1, x2, y2, sc in d]
            for t in trk.update_tracks(inp, frame=img):
                if t.is_confirmed() and t.time_since_update == 0:
                    x1, y1, x2, y2 = t.to_ltrb()
                    rows.append((f - a + 1, int(t.track_id), x1, y1, x2 - x1, y2 - y1, 1.0))
        C.write_mot(TRK / name / "data" / f"{s}.txt", rows)


def interp(points, target):
    pts = sorted(points)
    for (r0, h0), (r1, h1) in zip(pts, pts[1:]):
        if r0 <= target <= r1:
            return h0 + (h1 - h0) * (target - r0) / max(r1 - r0, 1e-12)
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot17", default=C.DEFAULT_MOT17)
    ap.add_argument("--allow-rerun", action="store_true")
    args = ap.parse_args()
    t0 = time.monotonic()
    done = OUT / "COMPLETE.json"
    reruns = 0
    if done.exists():
        if not args.allow_rerun:
            raise SystemExit(f"测试集结果已存在（{done}）。按预注册规则不重跑；确需重跑请加 --allow-rerun。")
        n = 1
        while (C.RESULTS / f"test_previous_{n}").exists():
            n += 1
        OUT.rename(C.RESULTS / f"test_previous_{n}")
        reruns = n
    splits = json.loads((C.GT_HALF / "splits.json").read_text())
    cpu = json.loads((C.RESULTS / "latency_cpu.json").read_text())
    xs = json.loads((C.RESULTS / "xscene.json").read_text())
    with open(C.RESULTS / "schedule_model.pkl", "rb") as f:
        sched = pickle.load(f)
    learned_thr = sorted({float(r["param"]) for r in csv.DictReader(open(C.RESULTS / "schedule_q2.csv")) if r["policy"] == "learned"})

    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"), "reruns_before_this": reruns,
        "split": "second 50% of each MOT17 training sequence (never evaluated before)",
        "hypotheses": {
            "H1": "ByteTrack(own) > thesis DeepSORT on HOTA and IDF1, fewer IDSW (detector YOLOv8n@640)",
            "H2": "fine-tuned detector > best pretrained detector at <= its latency (GPU and CPU, k=1)",
            "H3": "n-ft@960 k=2 > n-ft@640 k=1 in HOTA at lower CPU cost",
            "H4": "adaptive scheduling (per-frame learned, segment rule) does NOT beat fixed interval at matched rate (|diff| <= 1 HOTA)",
            "H5": "on held-out scenes, nx-ft@960 > n@960; gap to n-ft@960 = scene-familiarity share"},
        "configs": {"grid_k1": [C.cache_key(m, s) for m, s in GRID], "fixed_ks": FIXED_KS,
                    "learned_thresholds_from_q2": learned_thr, "segment_rule_c": SEG_C,
                    "xscene": xs, "detector_for_scheduling": FT},
        "metrics": "TrackEval HOTA/MOTA/IDF1 (official), matched-rate linear interpolation, per-sequence signs",
        "hashes": {"schedule_model.pkl": sha(C.RESULTS / "schedule_model.pkl"),
                   "gt_splits.json": sha(C.GT_HALF / "splits.json")}}
    for w in ("ft_n", "ft_s", "ft_nx"):
        p = C.RESULTS / "finetune" / w / "weights" / "best.pt"
        if p.exists():
            protocol["hashes"][f"{w}/best.pt"] = sha(p)
    (OUT / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8")
    print("protocol locked", flush=True)

    info = {}
    print("H1 DeepSORT...", flush=True)
    deepsort_run(C.cache_key("n", 640, "thesis"), "thesis_deepsort", splits, args.mot17)
    info["thesis_deepsort"] = {}
    info["byte_n640"] = byte_run(C.cache_key("n", 640), "byte_n640", splits)
    print("H2 grid...", flush=True)
    for m, s in GRID:
        key = C.cache_key(m, s)
        info[f"grid_{key}"] = byte_run(key, f"grid_{key}", splits)
    print("H3/H4 schedules...", flush=True)
    for k in FIXED_KS:
        info[f"fixed_{k}"] = byte_run(FT, f"fixed_{k}", splits, k=k)
    data = {}
    for s in C.SEQS:
        dets, _ = C.load_dets(FT, s)
        data[s] = (dets, S7.framediff(args.mot17, s), splits[s][SPLIT]["first"], splits[s][SPLIT]["last"])
    rates = {}
    for j, thr in enumerate(learned_thr):
        name = f"learned_{j}"
        nd = nf = 0
        rates[name] = {}
        for s, (dets, fd, a, b) in data.items():
            rows, d_, f_, _, _ = S7.run_policy(s, dets, fd, a, b, "learned", thr, fd_thr=sched["fd_thr"], model=sched["model"])
            C.write_mot(TRK / name / "data" / f"{s}.txt", rows)
            nd += d_; nf += f_; rates[name][s] = d_ / f_
        info[name] = {"rate": nd / nf}
    # 分段规则：归一化常数只用前 50%（q2）的无标注统计
    q2X = []
    for s in C.SEQS:
        dets, fd, _, _ = data[s]
        a, b = splits[s]["q2"]["first"], splits[s]["q2"]["last"]
        q2X += S8.execute(dets, fd, a, b, S8.segments(b - a + 1, splits[s]["fps"]), lambda i, x: 3)[2]
    q2X = np.array(q2X)
    mean_fd, mean_sp = float(q2X[:, 0].mean()) + 1e-6, float(q2X[:, 4].mean()) + 1e-6
    for j, c in enumerate(SEG_C):
        name = f"segrule_{j}"
        nd = nf = 0
        rates[name] = {}
        for s, (dets, fd, a, b) in data.items():
            segs = S8.segments(b - a + 1, splits[s]["fps"])

            def ch(i, x, c=c):
                if i == 0:
                    return S8.WARMUP_K
                risk = 0.5 * (x[0] / mean_fd + x[4] / mean_sp)
                return min(S8.KS, key=lambda kk: abs(kk - c / max(risk, 1e-3)))
            rows, d_, _ = S8.execute(dets, fd, a, b, segs, ch)
            C.write_mot(TRK / name / "data" / f"{s}.txt", rows)
            nd += d_; nf += b - a + 1; rates[name][s] = d_ / (b - a + 1)
        info[name] = {"rate": nd / nf}
    for k in FIXED_KS:
        info[f"fixed_{k}"]["rate"] = 1 / k
        rates[f"fixed_{k}"] = {s: 1 / k for s in C.SEQS}
    print("H5 cross-scene...", flush=True)
    held = xs["heldout"]
    for key, nm in ((C.cache_key("nx-ft", 960), "xs_nxft960"), (C.cache_key("n", 960), "xs_n960"), (FT, "xs_nft960")):
        info[nm] = byte_run(key, nm, splits, seqs=held)

    print("TrackEval...", flush=True)
    main_names = [n for n in info if not n.startswith("xs_")]
    summ, per = C.evaluate(SPLIT, TRK, main_names)
    xsumm, xper = C.evaluate(SPLIT, TRK, ["xs_nxft960", "xs_n960", "xs_nft960"], seqs=held)
    summ.update(xsumm); per.update(xper)
    for n, d in info.items():
        d.update(summ[n])
        if "key" in d:
            d["lat_gpu"] = d["det_gpu_ms"] / d["k"]
            d["lat_cpu"] = cpu.get(d["key"], {}).get("median_ms", float("nan")) / d["k"]

    def m(n):
        d = info[n]
        return f"HOTA {100 * d['HOTA']:5.2f}  MOTA {100 * d['MOTA']:5.2f}  IDF1 {100 * d['IDF1']:5.2f}  IDSW {d['IDSW']:4d}  Rec {100 * d['Recall']:5.2f}"

    def seqdiff(a, b, metric="HOTA", seqs=None):
        ds = [per[a][s][metric] - per[b][s][metric] for s in (seqs or C.SEQS)]
        return f"wins {sum(x > 0 for x in ds)}/{len(ds)} | " + "  ".join(f"{s[-2:]}:{100 * x:+.2f}" for s, x in zip(seqs or C.SEQS, ds))

    L = ["# Step 10 报告：预注册测试集一次性评测（MOT17 各序列后 50%）",
         f"protocol: results/test/PROTOCOL.json  reruns_before_this={reruns}", "",
         "## H1 跟踪器（YOLOv8n@640）",
         f"thesis DeepSORT : {m('thesis_deepsort')}", f"ByteTrack (own) : {m('byte_n640')}",
         f"HOTA per-seq (ByteTrack − DeepSORT): {seqdiff('byte_n640', 'thesis_deepsort')}",
         f"IDF1 per-seq: {seqdiff('byte_n640', 'thesis_deepsort', 'IDF1')}", "",
         "## H2 检测器网格（k=1）"]
    for mm, ss in GRID:
        n = f"grid_{C.cache_key(mm, ss)}"
        L.append(f"{mm:>5}@{ss:<5} {m(n)}  | GPU {info[n]['lat_gpu']:5.1f} ms  CPU {info[n]['lat_cpu']:6.1f} ms")
    for hw in ("lat_gpu", "lat_cpu"):
        for mm in ("n-ft", "s-ft"):
            for ss in (640, 960, 1280):
                n = f"grid_{C.cache_key(mm, ss)}"
                c_ = [(x, info[x]) for x in info if x.startswith("grid_") and "-ft" not in x and info[x][hw] <= info[n][hw]]
                if c_:
                    bn, bd = max(c_, key=lambda z: z[1]["HOTA"])
                    L.append(f"[{hw[4:].upper()}] {mm}@{ss} HOTA {100 * info[n]['HOTA']:.2f} vs best pretrained ≤ latency "
                             f"{bn[5:]} HOTA {100 * bd['HOTA']:.2f}  ({seqdiff(n, bn)})")
    L += ["", "## H3 预算分配（CPU）",
          f"n-ft@960 k=2 : {m('fixed_2')}  | CPU {info['fixed_2']['lat_cpu']:.1f} ms/frame",
          f"n-ft@640 k=1 : {m(NFT640)}  | CPU {info[NFT640]['lat_cpu']:.1f} ms/frame",
          f"per-seq HOTA: {seqdiff('fixed_2', NFT640)}", "",
          "## H4 自适应调度（相同检测率插值，HOTA %）"]
    pols = {"fixed": [n for n in info if n.startswith("fixed_")], "learned": [n for n in info if n.startswith("learned_")],
            "segrule": [n for n in info if n.startswith("segrule_")]}
    curves = {p: [(info[n]["rate"], info[n]["HOTA"]) for n in ns] for p, ns in pols.items()}
    L.append("rate    fixed      learned(Δ)       segrule(Δ)")
    for t in (0.2, 0.25, 0.33, 0.5):
        b_ = interp(curves["fixed"], t)
        row = f"{t:4.2f}  {100 * b_:6.2f}"
        for p in ("learned", "segrule"):
            v = interp(curves[p], t)
            row += f"   {100 * v:6.2f}({100 * (v - b_):+5.2f})" if np.isfinite(v) else "        n/a      "
        L.append(row)
    for p in ("learned", "segrule"):
        for t in (0.25, 0.33):
            ds = []
            for s in C.SEQS:
                fx = [(rates[n][s], per[n][s]["HOTA"]) for n in pols["fixed"]]
                px = [(rates[n][s], per[n][s]["HOTA"]) for n in pols[p]]
                ds.append(interp(px, t) - interp(fx, t))
            ok = [d for d in ds if np.isfinite(d)]
            L.append(f"{p:<8} rate {t:.2f}: wins {sum(d > 0 for d in ok)}/{len(ok)} | "
                     + "  ".join(f"{s[-2:]}:{100 * d:+.2f}" if np.isfinite(d) else f"{s[-2:]}:n/a" for s, d in zip(C.SEQS, ds)))
    L += ["", f"## H5 跨场景（只在未见场景 {held} 的后 50% 上评测）",
          f"pretrained n@960          : {m('xs_n960')}", f"cross-scene nx-ft@960     : {m('xs_nxft960')}",
          f"same-scene n-ft@960       : {m('xs_nft960')}",
          f"per-seq HOTA (cross − pretrained): {seqdiff('xs_nxft960', 'xs_n960', seqs=held)}",
          f"per-seq HOTA (same − cross)      : {seqdiff('xs_nft960', 'xs_nxft960', seqs=held)}"]
    g_all = info["xs_nft960"]["HOTA"] - info["xs_n960"]["HOTA"]
    g_x = info["xs_nxft960"]["HOTA"] - info["xs_n960"]["HOTA"]
    if g_all > 0:
        L.append(f"share of fine-tuning gain that transfers to unseen scenes: {100 * g_x / g_all:.0f}%")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.6))
        for p, (col, mk, lab) in {"fixed": ("#555555", "o", "Fixed interval"),
                                  "learned": ("#d62728", "^", "Per-frame learned trigger"),
                                  "segrule": ("#1f77b4", "s", "Segment allocation, model-based")}.items():
            pts = sorted(curves[p])
            ax.plot([100 * r for r, _ in pts], [100 * h for _, h in pts], marker=mk, color=col, label=lab)
        ax.set_xlabel("frames sent to the detector (%)"); ax.set_ylabel("HOTA (%)"); ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_title("Pre-registered test (MOT17, last 50% of each sequence)", fontsize=10)
        fig.tight_layout(); fig.savefig(C.RESULTS / "figures" / "test_schedule.png", dpi=160)
    except Exception as e:
        L.append(f"[plot failed: {e}]")
    L.append(f"\ntotal time: {time.monotonic() - t0:.0f}s")
    (OUT / "test_report.txt").write_text("\n".join(L), encoding="utf-8")
    done.write_text(json.dumps({"finished": datetime.datetime.now().isoformat(timespec="seconds")}), encoding="utf-8")
    print("\n".join(L))
    print(f"\n已保存：{OUT / 'test_report.txt'}")


if __name__ == "__main__":
    main()
