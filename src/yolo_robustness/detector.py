from .types import Box

class UltralyticsDetector:
    def __init__(self,model_path:str,device=None,imgsz=640):
        from ultralytics import YOLO
        self.model=YOLO(model_path); self.device=device; self.imgsz=imgsz
    def predict(self,image,conf=.25)->list[Box]:
        results=self.model.predict(source=image,conf=conf,imgsz=self.imgsz,device=self.device,verbose=False)
        r=results[0]; boxes=[]
        if r.boxes is None: return boxes
        xyxy=r.boxes.xyxy.cpu().numpy(); cls=r.boxes.cls.cpu().numpy().astype(int); confs=r.boxes.conf.cpu().numpy()
        for b,c,s in zip(xyxy,cls,confs): boxes.append(Box(float(b[0]),float(b[1]),float(b[2]),float(b[3]),int(c),float(s)))
        return boxes
