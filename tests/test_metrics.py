from yolo_robustness.metrics import iou,evaluate
from yolo_robustness.types import Box

def test_iou_identical(): assert iou(Box(0,0,10,10,0),Box(0,0,10,10,0))==1.0

def test_evaluate_match():
    m=evaluate([Box(0,0,10,10,0,.9)],[Box(0,0,10,10,0)])
    assert (m.tp,m.fp,m.fn,m.recall)==(1,0,0,1.0)

def test_evaluate_fragmentation():
    m=evaluate([Box(0,0,6,10,0,.9),Box(4,0,10,10,0,.8)],[Box(0,0,10,10,0)],.2)
    assert m.fragmentation>=0
