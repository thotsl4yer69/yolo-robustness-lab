import numpy as np
from yolo_robustness.transforms import apply, severity_suite, standard_suite, _motion_kernel

def sample():
    rng = np.random.default_rng(123)
    return rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)

def test_motion_kernel_is_normalized():
    for angle in (0, 15, 45, 90):
        kernel = _motion_kernel(9, angle)
        assert np.isclose(kernel.sum(), 1.0, atol=1e-6)
        assert np.all(kernel >= 0)

def test_severity_suite_is_complete():
    specs = severity_suite()
    assert specs[0].name == 'clean'
    assert len(specs) == 19
    for family in ('brightness_down', 'gaussian_blur', 'gaussian_noise', 'jpeg', 'motion_blur', 'center_occlusion'):
        members = [s for s in specs if s.name.startswith(family + '_')]
        assert [s.params['severity'] for s in members] == ['mild', 'moderate', 'severe']

def test_severity_transforms_preserve_shape_dtype_range():
    image = sample()
    for spec in severity_suite():
        out = apply(image, spec)
        assert out.shape == image.shape and out.dtype == np.uint8
        assert 0 <= int(out.min()) <= int(out.max()) <= 255

def test_standard_suite_still_covers_all_legacy_conditions():
    names = {s.name for s in standard_suite()}
    assert {'clean', 'brightness_down', 'brightness_up', 'contrast_down', 'motion_blur', 'jpeg_quality_20', 'gaussian_noise_strong', 'vignette', 'center_occlusion'}.issubset(names)
