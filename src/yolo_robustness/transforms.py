from dataclasses import dataclass, asdict
import cv2
import numpy as np

@dataclass(frozen=True)
class TransformSpec:
    name:str
    params:dict

def standard_suite():
    return [
        TransformSpec('clean',{}), TransformSpec('brightness_down',{'factor':.65}),
        TransformSpec('brightness_up',{'factor':1.25}), TransformSpec('contrast_down',{'factor':.70}),
        TransformSpec('gaussian_blur',{'sigma':2.0}), TransformSpec('gaussian_blur_strong',{'sigma':5.0}),
        TransformSpec('jpeg_quality_50',{'quality':50}), TransformSpec('jpeg_quality_20',{'quality':20}),
        TransformSpec('gaussian_noise',{'std':10.0,'seed':7}), TransformSpec('gaussian_noise_strong',{'std':25.0,'seed':7})]

def apply(image,spec):
    if spec.name=='clean': return image.copy()
    if spec.name.startswith('brightness_'): return np.clip(image.astype(np.float32)*spec.params['factor'],0,255).astype(np.uint8)
    if spec.name.startswith('contrast_'):
        mean=image.mean(axis=(0,1),keepdims=True); f=spec.params['factor']
        return np.clip((image.astype(np.float32)-mean)*f+mean,0,255).astype(np.uint8)
    if spec.name.startswith('gaussian_blur'): return cv2.GaussianBlur(image,(0,0),sigmaX=spec.params['sigma'])
    if spec.name.startswith('jpeg_quality'):
        ok,enc=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,int(spec.params['quality'])])
        if not ok: raise RuntimeError('JPEG encoding failed')
        return cv2.imdecode(enc,cv2.IMREAD_COLOR)
    if spec.name.startswith('gaussian_noise'):
        rng=np.random.default_rng(spec.params['seed']); noise=rng.normal(0,spec.params['std'],image.shape)
        return np.clip(image.astype(np.float32)+noise,0,255).astype(np.uint8)
    raise ValueError(f'Unknown transform: {spec.name}')

def specs_to_dict(specs): return [asdict(s) for s in specs]
