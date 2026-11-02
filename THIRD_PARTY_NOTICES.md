# Third-party notices

- **TrackEval** (official MOTChallenge evaluation code, Jonathon Luiten et al.) is vendored unmodified in
  `third_party/trackeval/` under its MIT License (see that folder's `LICENSE`; upstream commit in `COMMIT.txt`).
- **Ultralytics YOLOv8** is used as a dependency (not redistributed) and is licensed under AGPL-3.0;
  users are responsible for complying with its terms.
- **deep-sort-realtime** is used as a dependency for the thesis-baseline DeepSORT tracker; see its own license.
- **MOT17** (MOTChallenge, Milan et al., 2016) is used for evaluation. Images and annotations are **not**
  redistributed; download them from https://motchallenge.net/data/MOT17/ under the dataset's own terms.
  Fine-tuned detector weights are not included because they are derived from MOT17 annotations.
