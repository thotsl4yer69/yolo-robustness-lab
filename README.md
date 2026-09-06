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

Run the standard benchmark:

```bash
python -m yolo_robustness benchmark --images data/images --labels data/labels --model yolov8n.pt --output runs/standard --class-id 0 --transform-suite standard
```

Run the severity benchmark:

```bash
python -m yolo_robustness benchmark --images data/images --labels data/labels --model yolov8n.pt --output runs/severity --class-id 0 --transform-suite severity
```

Generate plots:

```bash
python -m yolo_robustness report runs/severity
```

Metrics include TP/FP/FN, precision, recall, F1, IoU, confidence statistics, prediction count, and fragmentation. Every run stores configuration, raw detections, metrics, and annotated images.

## Web dashboard / Vercel

The repository also contains a lightweight static dashboard for Vercel. Vercel is the presentation/reporting layer; the Python/Ultralytics benchmark runs on a local, Docker, CPU, or GPU host.

See `docs/VERCEL_DEPLOYMENT.md` for deployment details.

## Suites

- **Standard:** 17 controlled conditions covering brightness, contrast, blur, motion blur, JPEG compression, noise, color casts, vignette, occlusion and stripes.
- **Severity:** 19 conditions with mild/moderate/severe levels for brightness, blur, noise, JPEG compression, motion blur and center occlusion.
- **Clean:** baseline-only run for quick detector checks.

See `docs/METHODOLOGY.md` and `docs/ARCHITECTURE.md` for the experimental protocol.
