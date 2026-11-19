"""Standalone figures for late-run validation; primary comparisons stay separate."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(root,source,name,policies,labels,modalities,title):
    path=root/'reports'/source
    if not path.exists():return
    data=json.loads(path.read_text())['per_case']
    fig,axes=plt.subplots(len(modalities),3,figsize=(13,4*len(modalities)),squeeze=False,layout='constrained')
    metrics=(('mean_first_ready_s','Mean first-ready latency (s)'),('p95_first_ready_s','P95 first-ready latency (s)'),('cumulative_stall_s','Sum of client stalls (s)'))
    for row,m in enumerate(modalities):
        for col,(key,label) in enumerate(metrics):
            ax=axes[row,col]
            for i,policy in enumerate(policies):
                values=[r[key] for r in data if r['policy']==policy and r.get('modality','audio')==m]
                if not values:continue
                mean=sum(values)/len(values)
                ax.bar(i,mean,color=['#777777','#cb6952','#dba250','#297f80'][i],width=.65)
                ax.scatter([i+(j-(len(values)-1)/2)*.12 for j in range(len(values))],values,color='black',s=15,zorder=3)
                ax.annotate(f'{mean:.2f}',(i,mean),ha='center',va='bottom',fontsize=9)
            ax.set_xticks(range(len(policies)),labels,rotation=20,ha='right');ax.set_ylabel(label);ax.set_title(m.capitalize());ax.margins(y=.2)
    fig.suptitle(title+'\nDots are repetitions; later validation on the same model and workloads, not an interleaved primary comparison')
    fig.savefig(root/'reports'/(name+'.png'),dpi=180);fig.savefig(root/'reports'/(name+'.pdf'));plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();r=a.root
    render(r,'resource_fifo_validation.json','resource_fifo_validation',
           ['fcfs','buffer_global_fifo','buffer_resource_fifo'],['FCFS','Buffer / global FIFO','Buffer / resource FIFO'],['text','audio'],
           'Resource-aware admission validation: 100 text + 100 audio, native max_num_seqs=32')
    render(r,'audio_diagnostics.json','audio_diagnostics',
           ['fcfs','buffer','capacity_only','no_reserve'],['FCFS','Full buffer / 85%','Admission only / 85%','Full buffer / 100%'],['audio'],
           'Audio attribution: 200 requests, 1x playback, unchanged physical Gamma')

if __name__=='__main__':main()
