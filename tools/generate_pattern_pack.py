"""Generate deterministic visual QA patterns for the robustness suites.

These images are not benchmark ground truth. They are used to inspect whether
corruptions are being applied as intended before running detector benchmarks.
"""
from pathlib import Path
import argparse
import json
import cv2
import numpy as np
from yolo_robustness.transforms import standard_suite, severity_suite, apply


def base_pattern(width=1280, height=720):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ramp = np.linspace(30, 225, width, dtype=np.uint8)
    image[:] = ramp[None, :, None]
    for x in range(40, width, 160):
        cv2.rectangle(image, (x, 80), (x + 80, 240), (20, 40, 220), 4)
    for y in range(320, height - 40, 80):
        cv2.line(image, (40, y), (width - 40, y), (220, 40, 20), 3)
    cv2.circle(image, (width // 2, height // 2), 110, (30, 220, 30), 5)
    cv2.putText(image, "YOLO ROBUSTNESS LAB", (55, height - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def write_suite(image, specs, output, prefix):
    manifest = []
    for spec in specs:
        transformed = apply(image, spec)
        filename = f'{prefix}{spec.name}.png'
        path = output / filename
        if not cv2.imwrite(str(path), transformed):
            raise RuntimeError(f'Failed to write {path}')
        manifest.append({'file': filename, 'name': spec.name, 'params': spec.params})
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='artifacts/patterns')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    image = base_pattern()

    standard = write_suite(image, standard_suite(), output, 'standard_')
    severity = write_suite(image, severity_suite(), output, 'severity_')
    manifest = {
        'purpose': 'Visual QA only; not benchmark ground truth.',
        'image_size': [1280, 720],
        'standard_count': len(standard),
        'severity_count': len(severity),
        'standard': standard,
        'severity': severity,
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Generated {len(standard)} standard and {len(severity)} severity patterns in {output}')


if __name__ == '__main__':
    main()
