from pathlib import Path
import csv,json,platform,sys,cv2
from .data import discover_samples,read_yolo_labels
from .metrics import evaluate
from .transforms import standard_suite,severity_suite,apply,specs_to_dict
from .detector import UltralyticsDetector

def run_benchmark(images,labels,model,output,conf=.25,iou_threshold=.5,imgsz=640,class_id=0,device=None,transform_suite='standard'):
    output=Path(output); output.mkdir(parents=True,exist_ok=True); annotated=output/'annotated'; annotated.mkdir(exist_ok=True)
    samples=discover_samples(images,labels); detector=UltralyticsDetector(model,device,imgsz)
    if transform_suite=='standard': specs=standard_suite()
    elif transform_suite=='severity': specs=severity_suite()
    elif transform_suite=='clean': specs=[s for s in standard_suite() if s.name=='clean']
    else: raise ValueError(f'Unknown transform suite: {transform_suite}')
    config={'model':str(model),'confidence_threshold':conf,'matching_iou_threshold':iou_threshold,'imgsz':imgsz,'class_id':class_id,'device':device,'transform_suite_name':transform_suite,'transform_suite':specs_to_dict(specs),'python':sys.version,'platform':platform.platform()}
    (output/'config.json').write_text(json.dumps(config,indent=2)); rows=[]; detection_rows=[]
    for sample in samples:
        image=cv2.imread(str(sample.image_path))
        if image is None: raise RuntimeError(f'Could not read {sample.image_path}')
        h,w=image.shape[:2]; gt=read_yolo_labels(sample.label_path,w,h,class_id)
        for spec in specs:
            transformed=apply(image,spec); preds=[p for p in detector.predict(transformed,conf) if p.class_id==class_id]; m=evaluate(preds,gt,iou_threshold)
            rows.append({'image':sample.image_path.name,'transform':spec.name,'tp':m.tp,'fp':m.fp,'fn':m.fn,'precision':m.precision,'recall':m.recall,'f1':m.f1,'mean_iou':m.mean_iou,'max_confidence':m.max_confidence,'mean_confidence':m.mean_confidence,'confidence_std':m.confidence_std,'prediction_count':m.prediction_count,'fragmentation':m.fragmentation,'params':json.dumps(spec.params,sort_keys=True)})
            for idx,p in enumerate(preds): detection_rows.append({'image':sample.image_path.name,'transform':spec.name,'prediction_index':idx,'class_id':p.class_id,'confidence':p.confidence,'x1':p.x1,'y1':p.y1,'x2':p.x2,'y2':p.y2})
            vis=transformed.copy()
            for g in gt: cv2.rectangle(vis,(int(g.x1),int(g.y1)),(int(g.x2),int(g.y2)),(0,0,255),2)
            for p in preds:
                cv2.rectangle(vis,(int(p.x1),int(p.y1)),(int(p.x2),int(p.y2)),(0,255,0),2); cv2.putText(vis,f'{p.confidence:.2f}',(int(p.x1),max(15,int(p.y1)-4)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
            cv2.imwrite(str(annotated/f'{sample.image_path.stem}__{spec.name}{sample.image_path.suffix}'),vis)
    if not rows: raise RuntimeError('No image/label samples were discovered')
    with (output/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (output/'detections.csv').open('w',newline='') as f:
        fields=list(detection_rows[0]) if detection_rows else ['image','transform','prediction_index','class_id','confidence','x1','y1','x2','y2']; writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(detection_rows)
    summary={}
    for row in rows:
        s=summary.setdefault(row['transform'],{'n':0,'recall':0,'precision':0,'f1':0}); s['n']+=1
        for k in ('recall','precision','f1'): s[k]+=row[k]
    for s in summary.values():
        n=s.pop('n')
        for k in list(s): s[k]/=n
    (output/'summary.json').write_text(json.dumps(summary,indent=2)); return summary
