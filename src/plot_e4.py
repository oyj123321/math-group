from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path('outputs/moe_e4')
rows=json.loads((out/'results.json').read_text())
def get(pr,condition,metric):
    return next(r for r in rows if r['profile']==pr and r['condition']==condition and r['metric']==metric)
plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8})
fig,axes=plt.subplots(1,2,figsize=(7.2,2.65),layout='constrained')
for ax,pr in zip(axes,['independent','shortcut']):
    for arm,label,color in [('clean','Clean training','C0'),('augmentation','Noise augmentation','C1'),
                             ('consistency','Augmentation + KL','C2')]:
        conditions=['clean','noise_0.25','noise_0.5','noise_1.0']
        rr=[get(pr,c,arm+'_mse') for c in conditions]
        m=np.array([r['mean'] for r in rr]);lo=np.array([r['ci_low'] for r in rr]);hi=np.array([r['ci_high'] for r in rr])
        ax.errorbar([0,.25,.5,1],m,yerr=[m-lo,hi-m],label=label,color=color,marker='o',capsize=2,lw=1)
    ax.set(title=pr.capitalize(),xlabel='Nuisance noise SD',ylabel='Test MSE')
    ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.grid(alpha=.2)
axes[0].legend()
fig.savefig(out/'e4_robustness.png',dpi=300);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(7.2,2.65),layout='constrained')
for ax,pr in zip(axes,['independent','shortcut']):
    for arm,label in [('clean','Clean training'),('augmentation','Noise augmentation'),('consistency','Augmentation + KL')]:
        rr=[get(pr,f'noise_{s}',arm+'_switch_rate') for s in [.25,.5,1.]]
        m=np.array([r['mean'] for r in rr]);lo=np.array([r['ci_low'] for r in rr]);hi=np.array([r['ci_high'] for r in rr])
        ax.errorbar([.25,.5,1],m,yerr=[m-lo,hi-m],marker='o',capsize=2,label=label,lw=1)
    ax.set(title=pr.capitalize(),xlabel='Nuisance noise SD',ylabel='Expert-ID switch rate');ax.grid(alpha=.2)
axes[0].legend();fig.savefig(out/'e4_switching.png',dpi=300);plt.close(fig)
