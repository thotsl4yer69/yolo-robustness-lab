from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def generate_report(run_dir):
    run_dir=Path(run_dir); df=pd.read_csv(run_dir/'metrics.csv'); plots=run_dir/'plots'; plots.mkdir(exist_ok=True)
    summary=df.groupby('transform',as_index=False).agg(recall=('recall','mean'),precision=('precision','mean'),f1=('f1','mean'),mean_iou=('mean_iou','mean'),max_confidence=('max_confidence','mean'),fragmentation=('fragmentation','mean'))
    summary.to_csv(plots/'transform_summary.csv',index=False)
    for metric in ['recall','precision','f1','max_confidence','fragmentation']:
        fig=plt.figure(figsize=(11,5)); plt.bar(summary['transform'],summary[metric]); plt.xticks(rotation=45,ha='right'); plt.ylabel(metric); plt.tight_layout(); fig.savefig(plots/f'{metric}.png',dpi=160); plt.close(fig)
    return summary
