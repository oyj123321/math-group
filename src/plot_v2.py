import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--analysis',required=True);a=p.parse_args()
    root=Path(a.analysis)
    for profile in ['independent','shortcut']:
        directory=root/profile
        rows=json.loads((directory/'confidence_intervals.json').read_text())
        fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
        for ax,family,xlabel,title in zip(axs,['noise','mask_blind','cue'],
            ['Nuisance noise standard deviation','Fraction of relevant features masked','Fraction of samples with regime cue removed'],
            ['Nuisance noise','Missing information, no marker','Task-cue removal']):
            for metric,label in [('moe_corrupt_mse','MoE'),('dense_active_corrupt_mse','Dense: MAC-matched'),('dense_total_corrupt_mse','Dense: parameter-matched')]:
                data=sorted([r for r in rows if r['family']==family and r['metric']==metric],key=lambda x:x['strength'])
                mean=np.array([r['mean'] for r in data]);lo=np.array([r['ci_low'] for r in data]);hi=np.array([r['ci_high'] for r in data])
                ax.errorbar([r['strength'] for r in data],mean,yerr=np.maximum(0,np.vstack([mean-lo,hi-mean])),marker='o',capsize=3,label=label)
            ax.set_title(title);ax.set_xlabel(xlabel);ax.set_ylabel('Normalized-target test MSE');ax.grid(alpha=.2)
        axs[0].legend(fontsize=8)
        fig.suptitle(f'Protocol v2 synthetic test, Oct 13 2026 | {profile} | 5 seeds, 20,000 items | pointwise 95% intervals',fontsize=10)
        fig.savefig(directory/'robustness_ci.png',dpi=170);plt.close(fig)
        fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
        for ax,family in zip(axs,['noise','mask_blind','cue']):
            for metric,label in [('routing_term','Restore identity and gate'),('id_only_routing_term','Restore identity only')]:
                data=sorted([r for r in rows if r['family']==family and r['metric']==metric],key=lambda x:x['strength'])
                mean=np.array([r['mean'] for r in data]);lo=np.array([r['ci_low'] for r in data]);hi=np.array([r['ci_high'] for r in data])
                ax.errorbar([r['strength'] for r in data],mean,yerr=np.maximum(0,np.vstack([mean-lo,hi-mean])),marker='o',capsize=3,label=label)
            ax.axhline(0,color='black',linewidth=.8);ax.set_title(family)
            ax.set_xlabel({'noise':'Nuisance noise standard deviation','mask_blind':'Fraction of relevant features masked','cue':'Fraction of samples with cue removed'}[family])
            ax.set_ylabel('Restoration benefit: MSE before minus after');ax.grid(alpha=.2)
        axs[0].legend(fontsize=8)
        fig.suptitle(f'Protocol v2, {profile}, 5 seeds | positive favors restoration | privileged diagnostic | 95% intervals',fontsize=10)
        fig.savefig(directory/'routing_ci.png',dpi=170);plt.close(fig)
        entries=json.loads((directory/'specialization.json').read_text())
        fig,axs=plt.subplots(1,5,figsize=(13,3.4),layout='constrained')
        for seed,ax in enumerate(axs):
            matrix=np.zeros((4,4))
            for row in entries:
                if row['seed']==seed:matrix[row['regime'],row['expert']]=row['routing_fraction']
            image=ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues');ax.set_title(f'Seed {seed}');ax.set_xlabel('Expert ID');ax.set_ylabel('Task regime')
            ax.set_xticks(range(4));ax.set_yticks(range(4))
        fig.colorbar(image,ax=axs,label='P(expert | regime)',shrink=.75)
        fig.suptitle(f'Protocol v2 synthetic test | {profile}: task-conditioned routing | expert IDs not aligned across seeds',fontsize=10)
        fig.savefig(directory/'routing_specialization.png',dpi=170);plt.close(fig)
    paired=json.loads((root/'paired_controls'/'paired_controls.json').read_text())
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,profile in zip(axs,['independent','shortcut']):
        for metric,label in [('empirical_bayes_mse','Gaussian Bayes reference'),('moe_blind_mse','MoE, no marker'),('moe_marked_mse','MoE, with marker')]:
            data=sorted([r for r in paired if r['profile']==profile and r['metric']==metric],key=lambda x:x['strength'])
            mean=np.array([r['mean'] for r in data]);lo=np.array([r['ci_low'] for r in data]);hi=np.array([r['ci_high'] for r in data])
            ax.errorbar([r['strength'] for r in data],mean,yerr=np.maximum(0,np.vstack([mean-lo,hi-mean])),marker='o',capsize=3,label=label)
        ax.set_title(profile+' training');ax.set_xlabel('Fraction of relevant features masked');ax.set_ylabel('Normalized-target test MSE');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Protocol v2 | identical masks, 5 seeds, 20,000 items, pointwise 95% intervals\nGaussian oracle knows missing locations and intact task cue',fontsize=10)
    fig.savefig(root/'paired_controls'/'bayes_reference.png',dpi=170);plt.close(fig)
    print('Scientific plots written:',root)

if __name__=='__main__':main()
