"""User-authorized scoped revision: K300 pending, native K400 with195 left ROIs.

The original evaluator's numerical body is unchanged. Only the map availability,
declared realized counts and scoped fit metadata differ. Outputs are separate.
No fitting, label repair or original result mutation is performed.
"""
import datetime, inspect, json, os, subprocess, sys, traceback
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.stats import wilcoxon
from s_diagnose_labelcost_20261011 import B, CODE, frozen, sha, save
import s_step3_evaluate_group as E

OUT=B.parent.with_name('group_accepted_revision_20261012_v1')
KS=[100,200,400]
ACTUAL={100:50,200:100,400:195}


def main():
    import fcntl
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    before=frozen()
    maps={100:B/'maps/labels_L_50.mat',200:B/'maps/labels_L_100.mat',
          400:B/'independent_K400_20261011/maps/labels_L_200.mat'}
    provenance={'authorization':'User accepts K300 pending and K400 native195/200 output and requests a new editable PPT.',
        'scientific_scope':'Separate partial descriptive/comparative revision, not the original full35-map campaign.',
        'requested_K':KS,'realized_left_K':ACTUAL,'K300_pending':True,
        'source_maps':{str(k):{'path':str(p),'sha256':sha(p)} for k,p in maps.items()},
        'author_and_preserved_maps':before,'unchanged_numeric_evaluator_sha256':sha(CODE/'s_step3_evaluate_group.py'),
        'no_label_repair':True,'no_fitting':True,'bandpass_Hz':[.01,.1]}
    for k,p in maps.items():
        y=loadmat(p)['labels'].reshape(-1)
        assert y.shape==(29696,) and np.isfinite(y).all() and np.array_equal(y,y.astype('int32'))
        assert len(np.unique(y))==ACTUAL[k] and y.min()>=0 and y.max()<k//2
        dest=OUT/f'RNCut/maps/labels_L_{k//2}.npy';dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():assert np.array_equal(np.load(dest),y)
        else:np.save(dest,y.astype('int32'))
    save(OUT/'provenance.json',provenance)
    save(OUT/'RNCut/complete.json',dict(state='complete_accepted_three_maps',
        native_voxel_equivalence_passed=True,realized_left_K=ACTUAL,K300_pending=True,
        source_maps=provenance['source_maps'],no_label_repair=True))
    source=inspect.getsource(E.f_evaluate)
    changes={
        'for K in [100,200,300,400]:':'for K in [100,200,400]:',
        'assert len(np.unique(v))==K//2, (path,len(np.unique(v)))':'assert len(np.unique(v))=={100:50,200:100,400:195}[K], (path,len(np.unique(v)))',
        "fit['state']=='complete'":"fit['state']=='complete_accepted_three_maps'",
        'K_nominal_bilateral=[100,200,300,400]':'K_nominal_bilateral=[100,200,400]',
    }
    for old,new in changes.items():
        assert source.count(old)==1,old
        source=source.replace(old,new,1)
    save(OUT/'runtime_input_adaptation.json',dict(changes=changes,numerical_operations_unchanged=True,
         source_script_sha256=sha(CODE/'s_step3_evaluate_group.py')))
    (OUT/'frozen_evaluator_body.py').write_text(source)
    E.OUT=OUT
    exec(compile(source,str(OUT/'frozen_evaluator_body.py'),'exec'),E.__dict__)
    E.f_evaluate('RNCut')
    original=json.loads((E.BASE/'summary.json').read_text())
    nmf=json.loads((B.parent/'NMF/summary.json').read_text())
    rn=json.loads((OUT/'RNCut/summary.json').read_text())
    by_id={p['subject']:p for p in original['participants']}
    maps_all=dict(np.load(E.BASE/'atlas_labels_full_L.npz'))
    for label,more,labels in [('NMF',nmf,B.parent/'NMF/atlas_labels_full_L.npz'),
                             ('RNCut',rn,OUT/'RNCut/atlas_labels_full_L.npz')]:
        assert more['state']=='complete' and set(by_id)=={p['subject'] for p in more['participants']}
        for person in more['participants']:by_id[person['subject']]['metrics'].update(person['metrics'])
        maps_all.update(dict(np.load(labels)))
        for k in ([100,200,300,400] if label=='NMF' else KS):
            original['protocol']['metadata'][f'{label}_{k}']=dict(title=label,family=label,K=k,
                K_left=(k//2 if label=='NMF' else ACTUAL[k]),kind='multiscale',training_people=500)
    tests=[]
    metrics=['homogeneity','DCBC_x100','cosine_silhouette']
    for k in [100,200,300,400]:
        for metric in metrics:
            a=np.array([p['metrics'][f'CoBRA_{k}'][metric] for p in original['participants']])
            for family in ['DRFC','Schaefer','CoBRA_raw','NMF','RNCut']:
                if family=='RNCut' and k==300:continue
                d=a-np.array([p['metrics'][f'{family}_{k}'][metric] for p in original['participants']])
                p=float(wilcoxon(d,alternative='two-sided',zero_method='wilcox',method='approx',correction=False).pvalue) if np.any(d) else 1.
                tests.append(dict(K=k,metric=metric,comparator=family,n=50,p_raw=p,
                     mean_paired_difference=float(d.mean()),median_paired_difference=float(np.median(d)),
                     CoBRA_K_left=k//2,comparator_K_left=ACTUAL[k] if family=='RNCut' else k//2,
                     matched_realized_K=not(family=='RNCut' and k==400)))
    assert len(tests)==57
    order=np.argsort([t['p_raw'] for t in tests],kind='stable')
    adjusted=np.maximum.accumulate([(60-j)*tests[i]['p_raw'] for j,i in enumerate(order)])
    for i,p in zip(order,adjusted):tests[i]['p_holm_upper_bound']=min(1.,float(p))
    pending=[dict(K=300,metric=m,comparator='RNCut',state='pending_no_measurement',p_raw=None,
                  p_holm_upper_bound=None) for m in metrics]
    original['comparisons']=tests
    original['pending_comparisons']=pending
    original['protocol']['accepted_revision']=provenance
    original['protocol']['statistics']='Paired two-sided Wilcoxon. Conservative Holm upper bound over60 planned comparisons with3 unavailable comparisons reserved. Pending measurements/p-values are absent, never imputed. RNCut K400 compares realized195 versus200 left ROIs, not exact count matching.'
    folder=OUT/'figure_data';folder.mkdir(exist_ok=True)
    save(folder/'summary.json',original)
    np.savez_compressed(folder/'atlas_labels_full_L.npz',**maps_all)
    np.save(folder/'evaluation_mask_L.npy',np.load(E.BASE/'evaluation_mask_L.npy'))
    save(folder/'native_equivalence.json',json.loads((B/'native_equivalence.json').read_text()))
    save(folder/'provenance.json',provenance)
    assert len(maps_all)==34 and len(by_id)==50 and all(len(p['metrics'])==34 for p in original['participants'])
    assert frozen()==before
    save(folder/'complete.json',dict(state='complete_user_authorized_partial_revision',
         participants=50,maps=34,displayed_maps=19,observed_tests=57,planned_tests=60,
         pending_tests=3,full_original_campaign_complete=False,K300_pending=True,
         K400_realized_left=195,no_label_repair=True))
    manifest={str(p.relative_to(folder)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in folder.iterdir() if p.is_file()}
    save(folder/'manifest.json',manifest)
    subprocess.run(['tar','-czf',str(OUT/'figure_data.tar.gz'),'-C',str(OUT),'figure_data'],check=True)
    save(OUT/'accepted_revision_complete.json',dict(state='complete',bundle_sha256=sha(OUT/'figure_data.tar.gz')))


if __name__=='__main__':
    try:main()
    except Exception as e:
        OUT.mkdir(exist_ok=True);save(OUT/'accepted_revision_failure.json',dict(error=repr(e),traceback=traceback.format_exc()))
        raise
