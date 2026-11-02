#!/usr/bin/env python3
"""
Step 1：建立评测协议 —— 把 MOT17 每个训练序列按时间切成前后两半

  tune（前半）：用于基线评测、调参、设计和训练自适应控制器；
  test（后半）：最后一次性评测，之前不看结果（与 ByteTrack 等论文的 val-half 协议一致）。

为 TrackEval 写出两套 GT（帧号在各自划分内从 1 重新编号），并记录各划分的帧范围、
标注框数、行人 ID 数和 GT 文件哈希，写入 results/gt_half/splits.json。

用法：.\\.venv\\Scripts\\python.exe step1_prepare.py --mot17 data/MOT17
"""
import argparse
import hashlib
import json

import numpy as np

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mot17", default=C.DEFAULT_MOT17)
    args = ap.parse_args()
    info_all, lines = {}, ["seq        split  frames   gt_boxes(ped,vis)  ids"]
    for s in C.SEQS:
        info = C.seqinfo(args.mot17, s)
        gt_file = C.seq_dir(args.mot17, s) / "gt" / "gt.txt"
        gt = np.loadtxt(gt_file, delimiter=",", ndmin=2)
        rec = {"fps": info["fps"], "length": info["length"], "width": info["width"], "height": info["height"],
               "gt_sha256": hashlib.sha256(gt_file.read_bytes()).hexdigest()}
        for split in C.SPLITS:
            a, b = C.split_range(info["length"], split)
            sub = gt[(gt[:, 0] >= a) & (gt[:, 0] <= b)].copy()
            sub[:, 0] -= a - 1
            out = C.GT_HALF / split / s / "gt" / "gt.txt"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                for r in sub:
                    f.write(f"{int(r[0])},{int(r[1])},{r[2]:g},{r[3]:g},{r[4]:g},{r[5]:g},{int(r[6])},{int(r[7])},{r[8]:g}\n")
            ped = sub[(sub[:, 7] == 1) & (sub[:, 6] == 1)]
            rec[split] = {"first": a, "last": b, "n": b - a + 1, "ped_boxes": int(len(ped)),
                          "ped_ids": int(len(np.unique(ped[:, 1])))}
            lines.append(f"{s}  {split:<5} {b - a + 1:6d}   {len(ped):10d}          {rec[split]['ped_ids']:4d}")
        info_all[s] = rec
    (C.GT_HALF / "splits.json").write_text(json.dumps(info_all, indent=2), encoding="utf-8")
    tot = {sp: sum(info_all[s][sp]["n"] for s in C.SEQS) for sp in C.SPLITS}
    lines.append(f"total frames: tune={tot['tune']} test={tot['test']}")
    print("\n".join(lines))
    print(f"\n已写入：{C.GT_HALF}")


if __name__ == "__main__":
    main()
