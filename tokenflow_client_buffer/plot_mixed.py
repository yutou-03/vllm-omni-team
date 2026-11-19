"""Export reproducible comparison figures without mixing output units."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(root):
    root=Path(root);report=root/'reports';data=json.loads((report/'formal_results.json').read_text())
    comparisons=data['comparisons']
    if not comparisons:return
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    metrics=[('mean_first_ready_s','Mean first-ready latency'),('p95_first_ready_s','P95 first-ready latency'),('generated_units_per_s','Generation throughput')]
    for row,m in enumerate(('text','audio')):
        order={'text':0,'audio':0,'mixed_t150':1,'mixed_t100':2,'mixed_t50':3}
        cc=sorted((c for c in comparisons if c['modality']==m),key=lambda c:order[c['kind']])
        for col,(key,title) in enumerate(metrics):
            ax=axes[row,col];ys=[c['metrics'][key]['change_percent'] for c in cc]
            ax.bar(range(len(cc)),ys,color=['#297f80' if (y<0 if col<2 else y>0) else '#cb6952' for y in ys])
            ax.axhline(0,color='black',linewidth=.7)
            ax.set_xticks(range(len(cc)),[{'text':'Pure text','audio':'Pure audio','mixed_t150':'T:A=3:1','mixed_t100':'T:A=1:1','mixed_t50':'T:A=1:3'}[c['kind']] for c in cc],rotation=25,ha='right')
            ax.set_title(m.capitalize()+': '+title);ax.set_ylabel('Buffer vs FCFS change (%)')
            for i,y in enumerate(ys):ax.annotate(f'{y:+.1f}%',(i,y),ha='center',va='bottom' if y>=0 else 'top',fontsize=9)
            ax.margins(y=.25)
    fig.suptitle('Primary matrix: original global FIFO admission; max_num_seqs=32; 200 requests\nNegative latency and positive throughput changes are improvements')
    fig.savefig(report/'comparison.png',dpi=180);fig.savefig(report/'comparison.pdf');plt.close(fig)
    for kind in sorted({c['kind'] for c in comparisons}):
        for m in ('text','audio'):
            paths={a:root/'runs'/f'{kind}_formal_0_{a}'/m/'client/results.json' for a in ('fcfs','buffer')}
            if not all(p.exists() for p in paths.values()):continue
            fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
            for arm,path in paths.items():
                rows=json.loads(path.read_text());rows.sort(key=lambda r:r['workload']['payload']['omni_scheduling']['ingress_order'])
                x=[r['workload']['payload']['omni_scheduling']['ingress_order'] for r in rows];tt=[r['first_ready_s'] for r in rows]
                axes[0].plot(x,tt,'.',markersize=3,label=arm)
                axes[1].step(sorted(tt),np.arange(1,len(tt)+1)/len(tt),where='post',label=arm)
            axes[0].set(xlabel='Arrival order in the whole workload',ylabel='First-ready latency (s)')
            axes[1].set(xlabel='First-ready latency (s)',ylabel='Empirical cumulative fraction')
            for ax in axes:ax.grid(alpha=.2);ax.legend()
            fig.suptitle(f'{kind} / {m}: repetition 0 (all requests, no startup gate)')
            fig.savefig(report/f'{kind}_{m}_ttft.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();plot(a.root)
