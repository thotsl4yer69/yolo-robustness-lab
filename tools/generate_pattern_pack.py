"""Generate deterministic visual QA patterns for the robustness transform suite.

These images are not benchmark ground truth. They are used to inspect whether
corruptions are being applied as intended before running a detector benchmark.
"""
from pathlib import Path
import argparse
import cv2
import numpy as np
from yolo_robustness.transforms import standard_suite, apply


def base_pattern(width=1280, height=720):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    # Smooth luminance ramp makes brightness/contrast/color changes obvious.
    ramp = np.linspace(30, 225, width, dtype=np.uint8)
    image[:] = ramp[None, :, None]
    # High-frequency geometry exposes blur, JPEG, noise and stripe artifacts.
    for x in range(40, width, 160):
        cv2.rectangle(image, (x, 80), (x + 80, 240), (20, 40, 220), 4)
    for y in range(320, height - 40, 80):
        cv2.line(image, (40, y), (width - 40, y), (220, 40, 20), 3)
    cv2.circle(image, (width // 2, height // 2), 110, (30, 220, 30), 5)
    cv2.putText(image, "YOLO ROBUSTNESS LAB", (55, height - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='artifacts/patterns')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    image = base_pattern()
    cv2.imwrite(str(output / '00_clean.png'), image)
    for spec in standard_suite():
        if spec.name == 'clean':
            continue
        transformed = apply(image, spec)
        cv2.imwrite(str(output / f'{spec.name}.png'), transformed)
    print(f'Generated {len(standard_suite())} deterministic pattern previews in {output}')


if __name__ == '__main__':
    main()
