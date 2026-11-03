import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('outputs/moe_factorial');rows=json.loads((root/'results.json').read_text())
plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8})
fig,axes=plt.subplots(1,2,figsize=(7.2,2.8),layout='constrained')
metrics=['identity_restore','amplitude_restore','joint_restore','interaction']
for ax,profile in zip(axes,['independent','shortcut']):
    for index,(arm,label) in enumerate([('clean','Clean training'),('augmentation','Augmentation'),('consistency','Augmentation + KL')]):
        rr=[next(r for r in rows if r['profile']==profile and r['arm']==arm and r['metric']==k) for k in metrics]
        m=np.array([r['mean'] for r in rr]);lo=np.array([r['ci_low'] for r in rr]);hi=np.array([r['ci_high'] for r in rr])
        ax.errorbar(np.arange(4)+.2*(index-1),m,yerr=[m-lo,hi-m],fmt='o',markersize=3,capsize=2,label=label)
    ax.axhline(0,color='gray',lw=.8);ax.set_xticks(range(4),['ID only','Weight only','Joint','Interaction'],rotation=20)
    ax.set(title=profile.capitalize(),ylabel='Signed MSE contrast');ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.grid(alpha=.2)
axes[0].legend(loc='best');fig.savefig(root/'factorial_contrasts.png',dpi=300);plt.close(fig)
