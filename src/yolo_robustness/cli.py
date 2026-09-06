import argparse,json
from .runner import run_benchmark
from .report import generate_report

def main():
    parser=argparse.ArgumentParser(prog='yolo-robustness'); sub=parser.add_subparsers(dest='command',required=True)
    b=sub.add_parser('benchmark'); b.add_argument('--images',required=True); b.add_argument('--labels',required=True); b.add_argument('--model',required=True); b.add_argument('--output',required=True); b.add_argument('--conf',type=float,default=.25); b.add_argument('--iou-threshold',type=float,default=.5); b.add_argument('--imgsz',type=int,default=640); b.add_argument('--class-id',type=int,default=0); b.add_argument('--device',default=None); b.add_argument('--transform-suite',choices=['standard','severity','clean'],default='standard')
    r=sub.add_parser('report'); r.add_argument('run_dir')
    args=parser.parse_args()
    if args.command=='benchmark': print(json.dumps(run_benchmark(args.images,args.labels,args.model,args.output,args.conf,args.iou_threshold,args.imgsz,args.class_id,args.device,args.transform_suite),indent=2))
    else: print(generate_report(args.run_dir).to_string(index=False))
