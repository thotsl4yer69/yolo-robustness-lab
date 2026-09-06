# Methodology

An experimental unit is an image/ground-truth pair evaluated under one named transformation.

Ground truth is read from YOLO normalized bounding-box labels and is independent of detector predictions.

Predictions and ground-truth boxes are class matched. Candidate pairs are sorted by IoU descending and matched when IoU is at least the configured threshold.

Metrics: TP, FP, FN, precision, recall, F1, mean matched IoU, confidence statistics, prediction count, and fragmentation. Fragmentation counts additional predictions overlapping a ground-truth object at IoU >= 0.1 beyond its first overlap.

A single image is not sufficient evidence of detector failure. Use held-out images, frozen model weights, frozen thresholds, independent labels, and aggregate results across transformations and models.
