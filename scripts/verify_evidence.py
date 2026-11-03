"""Check the distributed summaries against report tables and their algebra.

This validates saved evidence; it does not rerun training or reconstruct SARCOS
predictions. The independently replayed synthetic mask audit is recorded separately.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import re

ROOT=Path(__file__).resolve().parents[1]

def close(a,b,tol=1e-10):
    if not math.isclose(a,b,abs_tol=tol,rel_tol=tol):
        raise AssertionError(f'{a} != {b}')

def verify():
    evidence=ROOT/'results/supplied_report_evidence'
    baseline=json.loads((evidence/'baseline.json').read_text())
    assert len(baseline)==30
    for lang in ['en','zh']:
        text=(ROOT/f'paper/{lang}/main.tex').read_text('utf-8')
        assert r'pbar\_e' not in text and r'$\bar p_e$' in text
        assert '9015--9017' in text and '9024--9026' in text
        assert text.count(r'\begin{table}')==7
        assert text.count(r'\begin{figure}')==4
        assert text.count(r'\[')==15
        baseline_table=text.split(r'\label{tab:results1}')[1].split(r'\end{table}')[0]
        for row in baseline:
            assert f"{row['mean']:.6f}" in baseline_table
        bib=(ROOT/f'paper/{lang}/references.bib').read_text('utf-8')
        assert len(re.findall(r'^@',bib,re.M))==14
        assert 'author = {Z. Qiu and others}' not in bib
    control=json.loads((ROOT/'results/synthetic/phase2/paired_controls/paired_controls.json').read_text())
    paired={(r['profile'],float(r['strength']),r['metric']):r for r in control}
    for profile in ['independent','shortcut']:
        for strength in [.125,.25,.5]:
            for model in ['moe','dense_active','dense_total']:
                get=lambda metric:paired[profile,strength,metric]['mean']
                close(get(model+'_blind_mse')-get('empirical_bayes_mse'),get(model+'_blind_excess_over_bayes'))
                close(get(model+'_marked_mse')-get(model+'_blind_mse'),get(model+'_marker_penalty'))
    rows=list(csv.DictReader((evidence/'sarcos_results.csv').open(encoding='utf-8')))
    assert len(rows)==264
    values={(r['condition'],r['configuration'],r['metric']):float(r['mean']) for r in rows}
    for lang in ['en','zh']:
        text=(ROOT/f'paper/{lang}/main.tex').read_text('utf-8')
        report_table=text.split(r'\label{tab:results7}')[1].split(r'\end{table}')[0]
        configs=['moe_clean','dense_active_clean','dense_total_clean','moe_augmentation','dense_active_augmentation','dense_total_augmentation','moe_consistency']
        for config in configs:
            for condition in ['clean','noise_0.25','mask_6']:
                assert f'{values[condition,config,"mse"]:.6f}' in report_table
            if config.startswith('moe_'):
                assert f'{values["noise_0.25",config,"switch"]:.4f}' in report_table
    for r in rows:
        assert int(r['training_seeds'])==5
        assert int(r['source_records'])==4449
        assert int(r['joints_per_source'])==7
        assert math.isfinite(float(r['mean']))
        assert float(r['ci_low'])<=float(r['ci_high'])
    decomp_checks=0;comparison_checks=0
    for (condition,config,metric),value in values.items():
        if metric=='degradation':
            close(value,values[condition,config,'mse']-values['clean',config,'mse'])
            if (condition,config,'input') in values:
                close(value,values[condition,config,'input']+values[condition,config,'routing'])
                decomp_checks+=1
        if '-minus-' in config:
            a,b=config.split('-minus-')
            close(value,values[condition,a,metric]-values[condition,b,metric])
            comparison_checks+=1
    verification=json.loads((evidence/'sarcos_verification.json').read_text())
    primary=verification['primary']
    close(primary['mean'],values['noise_0.25','moe_consistency-minus-moe_augmentation','mse'])
    close(sum(primary['seed_values'])/len(primary['seed_values']),primary['mean'])
    manifest=json.loads((evidence/'sarcos_data_manifest.json').read_text())
    assert sum(manifest['source_counts'][k] for k in ['train','val'])+manifest['development_sources_removed_for_test_input_overlap']==manifest['official_development_sources']
    for name,expected in manifest['source_hashes'].items():
        source=ROOT/'src'/name
        if source.exists():assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
    replay=json.loads((ROOT/'audit/mask_replay.json').read_text())
    assert replay['status']=='passed'
    return dict(status='passed',baseline_values=30,sarcos_rows=len(rows),sarcos_decomposition_checks=decomp_checks,sarcos_comparison_checks=comparison_checks,paired_marker_and_excess_identities=36,languages=['en','zh'],note='Summary consistency checks only. See audit/REVIEW.md for independently replayed checks and unavailable SARCOS artifacts.')

if __name__=='__main__':
    print(json.dumps(verify(),indent=2))
