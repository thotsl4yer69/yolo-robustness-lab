# YOLO Robustness Lab

Reproducible robustness benchmark for YOLO object detectors.

This repository measures detector reliability under controlled, reproducible image transformations using independent ground-truth annotations. It is intended for defensive ML robustness evaluation.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m yolo_robustness --help
```

Run a benchmark:

```bash
python -m yolo_robustness benchmark --images data/images --labels data/labels --model yolov8n.pt --output runs/baseline --class-id 0
```

Generate plots:

```bash
python -m yolo_robustness report runs/baseline
```

Metrics include TP/FP/FN, precision, recall, F1, IoU, confidence statistics, prediction count, and fragmentation. Every run stores configuration, raw detections, metrics, and annotated images.

See `docs/METHODOLOGY.md` and `docs/ARCHITECTURE.md` for the experimental protocol.
