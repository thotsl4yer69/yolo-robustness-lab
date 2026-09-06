from pathlib import Path
from .types import Box, Sample

def discover_samples(images_dir: str | Path, labels_dir: str | Path) -> list[Sample]:
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    image_exts = {'.jpg','.jpeg','.png','.bmp','.webp'}
    samples = []
    for p in sorted(images_dir.iterdir()):
        if p.suffix.lower() not in image_exts: continue
        label = labels_dir / f'{p.stem}.txt'
        if label.exists(): samples.append(Sample(p, label))
    if not samples: raise ValueError('No image/label pairs found.')
    return samples

def read_yolo_labels(path: str | Path, width: int, height: int, class_id: int | None = None) -> list[Box]:
    boxes=[]
    for line in Path(path).read_text().splitlines():
        if not line.strip(): continue
        parts=line.split()
        if len(parts)!=5: raise ValueError(f'Invalid YOLO label line: {line!r}')
        cls,xc,yc,w,h=map(float,parts); cls=int(cls)
        if class_id is not None and cls != class_id: continue
        boxes.append(Box((xc-w/2)*width,(yc-h/2)*height,(xc+w/2)*width,(yc+h/2)*height,cls,1.0))
    return boxes
