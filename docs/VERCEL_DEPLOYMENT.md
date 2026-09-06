# Vercel deployment

YOLO Robustness Lab is a Python benchmark engine plus a static web dashboard.

## What runs on Vercel

Vercel serves `index.html` as the dashboard. The deployment intentionally does not install or execute the Ultralytics benchmark stack as a serverless function.

This keeps the deployment small and avoids treating a serverless web runtime as a GPU inference worker.

## Deploy

1. Import `thotsl4yer69/yolo-robustness-lab` into Vercel.
2. Use the repository root as the Root Directory.
3. The repository contains `package.json` and `vercel.json`; the build command is `npm run build` and the static output is the repository root.
4. Deploy.

If Vercel has an old failed deployment cached, redeploy the latest `main` commit after the new files are visible in the import screen.

## Run benchmarks

Run benchmarks on a local machine, Docker host, or GPU workstation:

```bash
pip install -e ".[dev]"
python -m yolo_robustness benchmark --images data/images --labels data/labels --model yolov8n.pt --output runs/standard --transform-suite standard
python -m yolo_robustness benchmark --images data/images --labels data/labels --model yolov8n.pt --output runs/severity --transform-suite severity
python -m yolo_robustness report runs/severity
```

The resulting `metrics.csv`, `detections.csv`, `summary.json`, plots and annotated images are the benchmark artifacts. They can be published or inspected separately from the Vercel dashboard.

## Architecture

```text
Browser
  |
  v
Vercel static dashboard
  |
  +---- methodology / transform catalog / run instructions

GPU or CPU host
  |
  v
Python benchmark engine
  |
  +---- transforms -> YOLO detector -> IoU matcher -> metrics
  |
  v
CSV / JSON / plots / annotations
```
