from dataclasses import dataclass, asdict
import cv2
import numpy as np

@dataclass(frozen=True)
class TransformSpec:
    name: str
    params: dict

def standard_suite():
    """Deterministic corruption suite for repeatable detector robustness tests."""
    return [
        TransformSpec('clean', {}),
        TransformSpec('brightness_down', {'factor': .65}),
        TransformSpec('brightness_up', {'factor': 1.25}),
        TransformSpec('contrast_down', {'factor': .70}),
        TransformSpec('gaussian_blur', {'sigma': 2.0}),
        TransformSpec('gaussian_blur_strong', {'sigma': 5.0}),
        TransformSpec('motion_blur', {'size': 9, 'angle': 0}),
        TransformSpec('jpeg_quality_50', {'quality': 50}),
        TransformSpec('jpeg_quality_20', {'quality': 20}),
        TransformSpec('gaussian_noise', {'std': 10.0, 'seed': 7}),
        TransformSpec('gaussian_noise_strong', {'std': 25.0, 'seed': 7}),
        TransformSpec('color_cast_red', {'strength': .25}),
        TransformSpec('color_cast_blue', {'strength': .25}),
        TransformSpec('vignette', {'strength': .55}),
        TransformSpec('center_occlusion', {'fraction': .18}),
        TransformSpec('checkerboard_occlusion', {'cell': 32, 'opacity': .65}),
        TransformSpec('horizontal_stripes', {'spacing': 24, 'thickness': 3, 'alpha': .35}),
    ]

def _motion_kernel(size, angle):
    size = max(3, int(size) | 1)
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0 / size
    center = (size / 2 - .5, size / 2 - .5)
    matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    return cv2.warpAffine(kernel, matrix, (size, size))

def apply(image, spec):
    if spec.name == 'clean':
        return image.copy()
    if spec.name.startswith('brightness_'):
        return np.clip(image.astype(np.float32) * spec.params['factor'], 0, 255).astype(np.uint8)
    if spec.name.startswith('contrast_'):
        mean = image.mean(axis=(0, 1), keepdims=True)
        f = spec.params['factor']
        return np.clip((image.astype(np.float32) - mean) * f + mean, 0, 255).astype(np.uint8)
    if spec.name.startswith('gaussian_blur'):
        return cv2.GaussianBlur(image, (0, 0), sigmaX=spec.params['sigma'])
    if spec.name == 'motion_blur':
        return cv2.filter2D(image, -1, _motion_kernel(spec.params['size'], spec.params['angle']))
    if spec.name.startswith('jpeg_quality'):
        ok, enc = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, int(spec.params['quality'])])
        if not ok:
            raise RuntimeError('JPEG encoding failed')
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if spec.name.startswith('gaussian_noise'):
        rng = np.random.default_rng(spec.params['seed'])
        noise = rng.normal(0, spec.params['std'], image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if spec.name.startswith('color_cast_'):
        strength = float(spec.params['strength'])
        out = image.astype(np.float32)
        channel = 2 if spec.name.endswith('red') else 0
        out[:, :, channel] = out[:, :, channel] * (1.0 + strength) + 255.0 * strength
        return np.clip(out, 0, 255).astype(np.uint8)
    if spec.name == 'vignette':
        h, w = image.shape[:2]
        y, x = np.ogrid[:h, :w]
        dx = (x - (w - 1) / 2) / max(w / 2, 1)
        dy = (y - (h - 1) / 2) / max(h / 2, 1)
        radius = np.sqrt(dx * dx + dy * dy)
        mask = 1.0 - float(spec.params['strength']) * np.clip(radius, 0, 1) ** 2
        return np.clip(image.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)
    if spec.name == 'center_occlusion':
        out = image.copy()
        h, w = out.shape[:2]
        frac = float(spec.params['fraction'])
        oh, ow = max(1, int(h * frac)), max(1, int(w * frac))
        y1, x1 = (h - oh) // 2, (w - ow) // 2
        out[y1:y1 + oh, x1:x1 + ow] = 0
        return out
    if spec.name == 'checkerboard_occlusion':
        cell = max(2, int(spec.params['cell']))
        opacity = float(spec.params['opacity'])
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for y in range(0, image.shape[0], cell):
            for x in range(0, image.shape[1], cell):
                if ((x // cell) + (y // cell)) % 2 == 0:
                    mask[y:y + cell, x:x + cell] = 1
        dark = image.astype(np.float32) * (1.0 - opacity)
        return np.where(mask[..., None].astype(bool), dark, image).astype(np.uint8)
    if spec.name == 'horizontal_stripes':
        out = image.copy().astype(np.float32)
        spacing = max(2, int(spec.params['spacing']))
        thickness = max(1, int(spec.params['thickness']))
        alpha = float(spec.params['alpha'])
        for y in range(0, out.shape[0], spacing):
            out[y:y + thickness] *= (1.0 - alpha)
        return np.clip(out, 0, 255).astype(np.uint8)
    raise ValueError(f'Unknown transform: {spec.name}')

def specs_to_dict(specs):
    return [asdict(s) for s in specs]
