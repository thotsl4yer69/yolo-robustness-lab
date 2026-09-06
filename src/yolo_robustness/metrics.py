from dataclasses import dataclass
import numpy as np
from .types import Box

@dataclass
class Metrics:
    tp:int; fp:int; fn:int; precision:float; recall:float; f1:float; mean_iou:float
    max_confidence:float; mean_confidence:float; confidence_std:float
    prediction_count:int; fragmentation:int

def iou(a:Box,b:Box)->float:
    x1,y1=max(a.x1,b.x1),max(a.y1,b.y1); x2,y2=min(a.x2,b.x2),min(a.y2,b.y2)
    inter=max(0.0,x2-x1)*max(0.0,y2-y1)
    aa=max(0.0,a.x2-a.x1)*max(0.0,a.y2-a.y1); ab=max(0.0,b.x2-b.x1)*max(0.0,b.y2-b.y1)
    u=aa+ab-inter
    return inter/u if u else 0.0

def evaluate(predictions:list[Box],ground_truth:list[Box],iou_threshold=.5)->Metrics:
    candidates=[]
    for pi,p in enumerate(predictions):
        for gi,g in enumerate(ground_truth):
            if p.class_id==g.class_id: candidates.append((iou(p,g),pi,gi))
    candidates.sort(reverse=True); used_p=set(); used_g=set(); matched=[]
    for score,pi,gi in candidates:
        if score<iou_threshold or pi in used_p or gi in used_g: continue
        used_p.add(pi); used_g.add(gi); matched.append(score)
    tp=len(used_p); fp=len(predictions)-tp; fn=len(ground_truth)-len(used_g)
    precision=tp/(tp+fp) if tp+fp else 0.0; recall=tp/(tp+fn) if tp+fn else 0.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    conf=np.array([p.confidence for p in predictions],dtype=float)
    frag=0
    for g in ground_truth:
        overlaps=sum(iou(p,g)>=.1 for p in predictions); frag += max(0,overlaps-1)
    return Metrics(tp,fp,fn,precision,recall,f1,float(np.mean(matched)) if matched else 0.0,
        float(conf.max()) if conf.size else 0.0,float(conf.mean()) if conf.size else 0.0,
        float(conf.std()) if conf.size else 0.0,len(predictions),frag)
