import numpy as np
from yolo_robustness.transforms import standard_suite, apply

def test_suite_has_clean():
    assert any(x.name == 'clean' for x in standard_suite())

def test_brightness_changes_pixels():
    image = np.full((10, 10, 3), 100, dtype=np.uint8)
    spec = next(x for x in standard_suite() if x.name == 'brightness_down')
    out = apply(image, spec)
    assert out.mean() < image.mean()

def test_all_transforms_preserve_shape_and_dtype():
    image = np.full((64, 96, 3), 120, dtype=np.uint8)
    for spec in standard_suite():
        out = apply(image, spec)
        assert out.shape == image.shape, spec.name
        assert out.dtype == np.uint8, spec.name
        assert int(out.min()) >= 0 and int(out.max()) <= 255

def test_noise_is_deterministic():
    image = np.full((32, 32, 3), 120, dtype=np.uint8)
    spec = next(x for x in standard_suite() if x.name == 'gaussian_noise')
    assert np.array_equal(apply(image, spec), apply(image, spec))


def test_pattern_transforms_are_not_noops():
    image = np.full((64, 96, 3), 120, dtype=np.uint8)
    for name in ('motion_blur', 'vignette', 'center_occlusion', 'checkerboard_occlusion', 'horizontal_stripes'):
        spec = next(x for x in standard_suite() if x.name == name)
        assert not np.array_equal(apply(image, spec), image), name
