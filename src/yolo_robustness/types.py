from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    confidence: float = 1.0
    @property
    def xyxy(self):
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=float)

@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Path
