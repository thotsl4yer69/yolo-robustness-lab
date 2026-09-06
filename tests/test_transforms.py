import numpy as np
from yolo_robustness.transforms import standard_suite,apply

def test_suite_has_clean(): assert any(x.name=='clean' for x in standard_suite())

def test_brightness_changes_pixels():
    image=np.full((10,10,3),100,dtype=np.uint8); spec=next(x for x in standard_suite() if x.name=='brightness_down'); out=apply(image,spec)
    assert out.mean()<image.mean()
