"""Standalone plots for the completed, soft-penalty multibatch comparison."""
import json,argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();d=json.loads((a.root/'multibatch_summary.json').read_text())
order=[x for x in ['native64','nohold16','fixed16','dynamic16','nohold8','dynamic8','guard16'] if x in d['arms']]
labels={'native64':'Native64','nohold16':'NoPause16','fixed16':'Fixed16','dynamic16':'Dynamic16','nohold8':'NoPause8','dynamic8':'Dynamic8','guard16':'Guard16*'}
colors=['#777777' if 'nohold' in x or x=='native64' else '#286090' for x in order]
fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
ax=axs[0,0];vals=[d['arms'][x]['G_mean'] for x in order];ax.bar(range(len(order)),vals,color=colors)
for i,x in enumerate(order):
 ax.scatter([i]*len(d['arms'][x]['runs']),[r['G'] for r in d['arms'][x]['runs']],s=15,color='black',zorder=3)
 ax.text(i,vals[i]+.5,f'{vals[i]:.2f}',ha='center',fontsize=9)
ax.set_title('Streaming utility rate (beta=3.3)');ax.set_ylabel('Equivalent utility units / arrival-window second')
ax=axs[0,1];vals=[100*d['arms'][x]['groups']['text_standard']['first_slo_met']/d['arms'][x]['groups']['text_standard']['requests'] for x in order];ax.bar(range(len(order)),vals,color=colors);ax.set_ylim(0,110);ax.set_title('Urgent text TTFT <= 1.3 s');ax.set_ylabel('Requests meeting target (%)')
ax=axs[1,0];vals=[d['arms'][x]['groups']['audio_loose']['mean_rebuffer_s'] for x in order];ax.bar(range(len(order)),vals,color=colors);ax.set_title('Loose audio cumulative rebuffering');ax.set_ylabel('Mean seconds / request')
for ax in [axs[0,0],axs[0,1],axs[1,0]]:ax.set_xticks(range(len(order)),[labels[x] for x in order],rotation=25,ha='right');ax.grid(axis='y',alpha=.2)
ax=axs[1,1]
for label in ['dynamic16_vs_nohold16','fixed16_vs_nohold16','dynamic8_vs_nohold8','dynamic16_vs_native64','guard16_vs_nohold16']:
 if label in d['comparisons']:
  b=[1.65,3.3,6.6,13.2];y=[d['comparisons'][label]['beta_gain_percent'][str(v)] for v in b];ax.plot(b,y,marker='o',label=label.replace('_vs_',' / '))
ax.axhline(0,color='black',lw=.8);ax.set_title('Sensitivity to first-output penalty weight');ax.set_xlabel('beta');ax.set_ylabel('Relative streaming score change (%)');ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle('Experiment 2 revision: real multi-request batching, 3 paired traces per arm',fontsize=13)
fig.savefig(a.root/'multibatch_v2_metrics.png',dpi=180);fig.savefig(a.root/'multibatch_v2_metrics.pdf');print(a.root/'multibatch_v2_metrics.png')
