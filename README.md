# Where should a real-time tracker spend its compute?

This repo studies tracking-by-detection on MOT17 under a fixed compute budget. The main claims were written down in a protocol file before I looked at the held-out test split, and the test was run once.

![accuracy vs latency](results/figures/frontier_q2.png)

Accuracy vs. latency for 75 configurations on the validation segment. Left plot is a laptop GPU, right plot is a CPU limited to 4 threads, which I use as a rough stand-in for a low-power device.

## Motivation

On edge devices like street cameras or embedded boards, each frame has to be processed within a fixed time budget, and in tracking-by-detection most of that time goes to the detector. There are basically three ways to spend it:

- use a stronger detector (bigger model, higher input resolution, or fine-tuning on the target domain)
- detect less often, e.g. every k-th frame, and let a Kalman filter predict the tracks in between
- decide at run time which frames are worth running the detector on

In practice these are often picked by feel. I wanted to measure them directly, on both a GPU and a CPU budget.

## Results

All numbers below are from the pre-registered test (the last 50% of each of the 7 MOT17 training sequences).

| | Hypothesis | Result | Numbers |
|---|---|---|---|
| H1 | With the same detector, a motion-only tracker (ByteTrack) does at least as well as an appearance-based one (DeepSORT) | supported | HOTA 35.1 to 38.6, IDF1 38.0 to 44.6, ID switches 103 to 88, better on 7 of 7 sequences. Tracker cost about 0.4 ms vs 55-70 ms per frame |
| H2 | A fine-tuned detector beats the best pretrained one at the same or lower latency | partly | +9.7 HOTA overall on CPU, but only 3 to 5 of 7 sequences improve. Most of the gain comes from the densest scene |
| H3 | On a CPU budget, a stronger detector every other frame beats a weaker one every frame | supported | HOTA 47.4 vs 44.5 at 35.1 vs 39.1 ms per frame, 5 of 7 sequences |
| H4 | Adaptive scheduling does not beat a fixed interval at the same compute (expected null) | supported | learned trigger 1.5 to 2.4 HOTA lower, segment allocation between -1.1 and +0.1 |
| H5 | Fine-tuning on some scenes helps on scenes it has not seen | refuted | 7.5 HOTA lower than the pretrained detector on 3 held-out scenes (0 of 3) |

Speed at 1080p: 44 FPS end to end on the laptop GPU including writing the output video (81 FPS without writing), and 40 FPS on 4 CPU threads when detecting every third frame. For reference, a YOLOv8 + DeepSORT pipeline with image preprocessing ran at 8.5 FPS on the same video.

### What I take from this

1. Spending compute on a better detector pays off more than detecting more often. At every budget, the best configurations used a stronger detector and skipped frames, and the Kalman filter handled 2 or 3 skipped frames without much loss.
2. On this data, appearance features are not worth their cost. In the DeepSORT pipeline they took almost half of each frame's time, and the motion-only tracker was still more accurate on every sequence.
3. A fixed detection interval is hard to beat on MOT17. I tried seven adaptive policies and none of them beat it at the same compute:
   - per-frame triggers based on Kalman drift
   - a learned trigger trained on self-supervised labels
   - budgeted allocation over 2-second segments

   Even a per-frame trigger that knew the true drift in hindsight didn't beat it. For segment allocation, the hindsight version leaves at most about 1.4 HOTA of room. My guess is that dense pedestrian scenes are hard almost all the time, so there isn't much to adapt to. The biggest hindsight gain showed up in a moving-camera sequence.
4. Fine-tuning on a small amount of data mostly learns the scenes it was trained on. On validation it looked much better than using a bigger model, but on unseen scenes it made things worse. For a new deployment, a pretrained detector at a higher resolution seems like the safer choice.

## Method

- Detection: YOLOv8 n/s/m at 640, 960 and 1280 input, COCO-pretrained or fine-tuned on MOT17 (fine-tuning only uses the first 25% of each sequence). Detections and per-frame latency are cached once and reused.
- Tracking: my own Kalman filter (constant velocity, state is center, aspect ratio and height) plus Hungarian matching. ByteTrack mode adds a second matching round with low-score detections. On frames without detection the tracks are only predicted.
- Budget: latency per frame = detector cost x detection rate + tracker cost. GPU cost is measured per frame; CPU cost is measured with 4 threads.
- Scheduling policies compared:
  - fixed interval
  - a geometric drift rule (Kalman speed times frames since the last detection, plus crowding and frame-difference triggers)
  - a learned drift predictor (gradient boosting, leave-one-sequence-out)
  - segment allocation (multiple-choice knapsack solved with a Lagrangian, risk from frame differences and Kalman speed)
  - hindsight versions of these, as references

## Evaluation

Every MOT17 training sequence is split by time:

- first 25%: only for fine-tuning the detector
- 25% to 50%: all comparisons, design decisions and threshold choices
- last 50%: held out. Evaluated once, under the hypotheses, configurations and model hashes in [results/test/PROTOCOL.json](results/test/PROTOCOL.json)

Metrics come from the official TrackEval code (HOTA, MOTA, IDF1), included in `third_party/`. Policies are compared at the same detection rate by linear interpolation, and I also report how many of the 7 sequences each one wins. For the cross-scene check, the detector was fine-tuned on 4 scenes (2 static camera, 2 moving camera) and tested on the remaining 3.

## Files

- `results/test/test_report.txt`: pre-registered test (H1 to H5)
- `results/step6_report.txt`: all 75 accuracy/latency configurations
- `results/step7_report.txt`, `results/step8_report.txt`: adaptive scheduling on the validation segment
- `results/step3_report.txt`: DeepSORT baseline and per-stage timing
- `results/step11_report.txt`: end-to-end FPS

## Running it

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt scikit-learn
# get MOT17 from https://motchallenge.net/data/MOT17/ and put it at data/MOT17/train/MOT17-02-FRCNN/...
python step1_prepare.py --mot17 data/MOT17          # time-based splits
python step2_detect.py  --mot17 data/MOT17 --models n s m --imgsz 640 960 1280
python step4_finetune.py --mot17 data/MOT17         # fine-tune n/s on the first 25%
python step2_detect.py  --mot17 data/MOT17 --models n-ft s-ft --imgsz 640 960 1280
python step5_cpu_latency.py --mot17 data/MOT17
python step3_baseline.py --mot17 data/MOT17        # DeepSORT baseline
python step6_frontier.py                            # accuracy/latency frontier
python step7_schedule.py --mot17 data/MOT17         # per-frame scheduling
python step8_allocation.py --mot17 data/MOT17       # segment allocation
python step9_xscene.py  --mot17 data/MOT17          # cross-scene fine-tuning
python step10_test.py   --mot17 data/MOT17          # held-out test, runs once
```

I ran everything on Windows 11 with Python 3.11, PyTorch 2.14 and CUDA 12.6, on an RTX 3070 Ti laptop GPU and an i7-12700H.

## Limitations

- Only one benchmark (MOT17, 7 scenes) and only pedestrians. Results vary a lot between sequences, and the overall scores are dominated by the densest one (MOT17-04).
- The CPU budget comes from a laptop CPU limited to 4 threads, not from a real embedded device, and I didn't measure energy.
- Fine-tuning used about 1,100 frames with default hyperparameters. More data or better regularization might transfer better.
- The scheduling experiments use one tracker family (Kalman + IoU matching). Trackers that rely on appearance might react differently to sparse detection.

## References

- Zhang et al., ByteTrack: Multi-Object Tracking by Associating Every Detection Box, ECCV 2022.
- Wojke et al., Simple Online and Realtime Tracking with a Deep Association Metric, ICIP 2017.
- Luiten et al., HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking, IJCV 2021.
- Milan et al., MOT16: A Benchmark for Multi-Object Tracking, arXiv 2016.
- Jocher et al., Ultralytics YOLOv8, 2023.

## Note

This started from my undergraduate thesis on pedestrian tracking. I wrote the code with help from an AI coding assistant. I ran all the experiments on my own machine and I'm responsible for the results and how they're interpreted.
