"""Finite dependent evaluation for the user's accepted RNCut comparison.

No fitting or label edits. Wait for the existing K300 launcher to validate,
evaluate that one map with the unchanged numerical evaluator, reuse the earlier
34 verified maps/measurements, and calculate all60 planned Holm tests.
The already accepted K400 native195/evaluation192 mismatch remains disclosed.
"""
from pathlib import Path
import datetime
import fcntl
import hashlib
import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

W = Path('/dataspace/yuhang_data/IndividualFunctionalParcellation/GPT_workplace')
CODE = Path('/dataspace/yuhang_data/IndividualFunctionalParcellation/CoBRA/TraditionalMethod/experiments/group_nmf_rncut_20261009')
D = W/'runs/group_nmf_rncut_20261009_v2/RNCut/initialization_guard_20261012_v2'
PREVIOUS = W/'runs/group_accepted_revision_20261012_v1'
OUT = W/'runs/group_accepted_revision_20261012_v2_k300'
PY = '/home/yuhang/miniconda3/envs/pytorch/bin/python'
sys.path[:0] = [str(CODE), str(W/'paper_figure_campaign'), str(W)]


def f_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def f_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''): h.update(block)
    return h.hexdigest()


def f_save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.partial')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def f_status(state, **kwargs):
    f_save(OUT/'postfit_status.json',dict(state=state,pid=os.getpid(),
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kwargs))


def f_verify_previous():
    gate=f_read(PREVIOUS/'figure_data/complete.json')
    assert gate['maps']==34 and gate['participants']==50 and gate['observed_tests']==57
    assert gate['K400_realized_left']==195 and gate['no_label_repair']
    assert f_sha(PREVIOUS/'figure_data.tar.gz')==f_read(PREVIOUS/'accepted_revision_complete.json')['bundle_sha256']
    manifest=f_read(PREVIOUS/'figure_data/manifest.json')
    for name,info in manifest.items():
        p=PREVIOUS/'figure_data'/name
        assert f_sha(p)==info['sha256'] and p.stat().st_size==info['bytes']
    return manifest


def f_evaluator():
    spec=importlib.util.spec_from_file_location('k300_same_numeric_evaluator',CODE/'s_step3_evaluate_group.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original=inspect.getsource(module.f_evaluate)
    changes={'for K in [100,200,300,400]:':'for K in [300]:',
             'K_nominal_bilateral=[100,200,300,400]':'K_nominal_bilateral=[300]'}
    modified=original
    for old,new in changes.items():
        assert modified.count(old)==1,old
        modified=modified.replace(old,new,1)
    compile(modified,'frozen_evaluator_body.py','exec')
    return module,modified,changes


def f_finish(receipt):
    import numpy as np
    from scipy.io import loadmat
    from scipy.stats import wilcoxon
    from s_diagnose_labelcost_20261011 import frozen
    previous_manifest=f_verify_previous()
    before=frozen()
    assert receipt['state']=='validated_candidate_resolution'
    assert receipt['integrity_before']==receipt['integrity_after']
    assert receipt['integrity_after']['author']==before[0] and receipt['integrity_after']['maps']==before[1]
    source_map=D/'maps/labels_L_150.mat'
    assert f_sha(source_map)==receipt['map_sha256']
    init=f_read(D/'initialization_K150.json')
    assert init['K_occupied_selected']==150 and init['no_posthoc_label_repair'] and not init['seed_search']
    y=loadmat(source_map)['labels'].reshape(-1)
    assert y.shape==(29696,) and np.isfinite(y).all() and np.array_equal(y,y.astype('int32'))
    assert len(np.unique(y))==150 and y.min()==0 and y.max()==149
    evaluation_mask=np.load(PREVIOUS/'figure_data/evaluation_mask_L.npy')
    evaluation_K300=int(np.unique(y[evaluation_mask]).size)
    path=OUT/'RNCut/maps/labels_L_150.npy';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():assert np.array_equal(np.load(path),y)
    else:np.save(path,y.astype('int32'))
    f_save(OUT/'RNCut/complete.json',dict(state='complete',native_voxel_equivalence_passed=True,
         training_people=500,K_nominal_bilateral=[300],K_left=[150],initialization_K300=init,
         validated_candidate_receipt_sha256=f_sha(D/'receipt.json'),no_label_repair=True))
    E,body,changes=f_evaluator();E.OUT=OUT
    f_save(OUT/'runtime_input_adaptation.json',dict(changes=changes,numerical_operations_unchanged=True,
         original_evaluator_sha256=f_sha(CODE/'s_step3_evaluate_group.py')))
    (OUT/'frozen_evaluator_body.py').write_text(body,encoding='utf-8')
    exec(compile(body,str(OUT/'frozen_evaluator_body.py'),'exec'),E.__dict__)
    f_status('running_K300_same50_evaluation')
    E.f_evaluate('RNCut')
    result=f_read(PREVIOUS/'figure_data/summary.json')
    old_tests={(t['K'],t['metric'],t['comparator']):t for t in result['comparisons']}
    added=f_read(OUT/'RNCut/summary.json');assert added['state']=='complete'
    people={p['subject']:p for p in result['participants']}
    assert len(people)==50 and set(people)=={p['subject'] for p in added['participants']}
    for p in added['participants']:
        assert set(p['metrics'])=={'RNCut_300'}
        people[p['subject']]['metrics'].update(p['metrics'])
    maps=dict(np.load(PREVIOUS/'figure_data/atlas_labels_full_L.npz'))
    maps.update(dict(np.load(OUT/'RNCut/atlas_labels_full_L.npz')))
    assert len(maps)==35 and all(len(p['metrics'])==35 for p in people.values())
    result['protocol']['metadata']['RNCut_300']=dict(title='RNCut (surface adaptation)',family='RNCut',
        K=300,K_left=150,K_left_evaluation=evaluation_K300,kind='multiscale',training_people=500)
    provenance=dict(authorization='User accepts native K400195/200; complete the pending K300 comparison.',
        source_partial_bundle_sha256=f_sha(PREVIOUS/'figure_data.tar.gz'),
        K300_map_sha256=receipt['map_sha256'],K300_initialization=init,
        K300_native_left=150,K300_evaluation_left=evaluation_K300,
        K400_native_left=195,K400_evaluation_left=192,K300_pending=False,
        no_label_repair=True,bandpass_Hz=[.01,.1],all_original_results_preserved=True,
        strict_original_matched_resolution_campaign_complete=False,
        disclosure='K300 uses the declared training-only initialization fallback; original author optimizer retained. Nominal K400 has195 native left parcels (192 on the common evaluation mask), versus200 requested. This accepted comparison is not exact realized-count matching.')
    result['protocol']['accepted_revision']=provenance
    result['protocol']['statistics']='Paired two-sided Wilcoxon, normal approximation, Holm correction over all60 observed planned comparisons. Native K400 realized counts differ and are disclosed.'
    tests=[]
    for k in [100,200,300,400]:
        for metric in ['homogeneity','DCBC_x100','cosine_silhouette']:
            for family in ['DRFC','Schaefer','CoBRA_raw','NMF','RNCut']:
                diff=np.array([p['metrics'][f'CoBRA_{k}'][metric]-p['metrics'][f'{family}_{k}'][metric] for p in result['participants']])
                p=float(wilcoxon(diff,alternative='two-sided',zero_method='wilcox',method='approx',correction=False).pvalue) if np.any(diff) else 1.
                key=(k,metric,family)
                if key in old_tests:assert abs(p-old_tests[key]['p_raw'])<1e-14
                tests.append(dict(K=k,metric=metric,comparator=family,n=50,p_raw=p,
                    mean_paired_difference=float(diff.mean()),median_paired_difference=float(np.median(diff)),
                    CoBRA_K_left=k//2,comparator_K_left=195 if family=='RNCut' and k==400 else k//2,
                    matched_realized_K=not(family=='RNCut' and k==400)))
    assert len(tests)==60
    order=np.argsort([t['p_raw'] for t in tests],kind='stable')
    adjusted=np.maximum.accumulate([(60-j)*tests[i]['p_raw'] for j,i in enumerate(order)])
    for i,p in zip(order,adjusted):tests[i]['p_holm']=min(1.,float(p))
    result['comparisons']=tests;result['pending_comparisons']=[]
    target=OUT/'figure_data';target.mkdir(exist_ok=True)
    f_save(target/'summary.json',result)
    np.savez_compressed(target/'atlas_labels_full_L.npz',**maps)
    shutil.copy2(PREVIOUS/'figure_data/evaluation_mask_L.npy',target/'evaluation_mask_L.npy')
    shutil.copy2(PREVIOUS/'figure_data/native_equivalence.json',target/'native_equivalence.json')
    f_save(target/'provenance.json',provenance)
    f_save(target/'K300_candidate_receipt.json',receipt)
    f_save(target/'complete.json',dict(state='complete_user_authorized_all_maps_revision',participants=50,
        maps=35,displayed_maps=20,observations_all_atlases=5250,observations_displayed=3000,
        observed_tests=60,planned_tests=60,pending_tests=0,K300_pending=False,
        K400_realized_left=195,full_original_campaign_complete=False,
        accepted_measured_comparison_complete=True,no_label_repair=True,PPT_visual_review_complete=False))
    f_save(target/'manifest.json',{p.name:dict(sha256=f_sha(p),bytes=p.stat().st_size)
        for p in sorted(target.iterdir()) if p.is_file() and p.name!='manifest.json'})
    subprocess.run(['tar','-czf',str(OUT/'figure_data.tar.gz'),'-C',str(OUT),'figure_data'],check=True)
    assert frozen()==before and f_verify_previous()==previous_manifest
    f_status('ready_for_figure_authoring',bundle=str(OUT/'figure_data.tar.gz'),
             bundle_sha256=f_sha(OUT/'figure_data.tar.gz'),measured_tests=60,PPT_visual_review_complete=False)


def f_run():
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'postfit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'postfit_status.json').exists() and f_read(OUT/'postfit_status.json')['state']=='ready_for_figure_authoring':return
    deadline=time.monotonic()+12*3600
    f_status('waiting_for_existing_K300',maximum_wait_hours=12)
    while time.monotonic()<deadline:
        receipt=f_read(D/'receipt.json')
        if receipt['state']=='validated_candidate_resolution':break
        if receipt['state']=='resolution_mismatch_preserved' or (receipt['state']=='optimizer_returned' and receipt.get('returncode')!=0):
            f_status('blocked_by_K300_validation',fit_state=receipt['state']);return
        if not Path(f'/proc/{receipt["pid"]}').exists():
            f_status('blocked_existing_launcher_exited',fit_state=receipt['state']);return
        time.sleep(30)
    else:f_status('expired_waiting_for_existing_fit');return
    while time.monotonic()<deadline:
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','1'],text=True).strip())
        if free>=3072:break
        f_status('waiting_for_GPU1_memory',free_MiB=free);time.sleep(30)
    else:f_status('expired_waiting_for_GPU');return
    f_finish(receipt)


def f_main():
    if '--check' in sys.argv:
        f_verify_previous();_,_,changes=f_evaluator()
        print(json.dumps(dict(passed=True,only_evaluator_changes=changes,no_fitting=True)))
    elif '--launch' in sys.argv:
        OUT.mkdir(exist_ok=True)
        assert not (OUT/'postfit_launch.json').exists(),'Inspect existing queue before a new launch.'
        with (OUT/'postfit.log').open('ab',buffering=0) as stream:
            child=subprocess.Popen([PY,str(Path(__file__).resolve()),'--run'],stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
        f_save(OUT/'postfit_launch.json',dict(pid=child.pid,script_sha256=f_sha(Path(__file__))))
        print(json.dumps(dict(pid=child.pid,output=str(OUT))))
    elif '--run' in sys.argv:f_run()
    else:raise ValueError('Use --check, then --launch. No fitting is started.')


if __name__=='__main__':
    try:f_main()
    except Exception as exc:
        if '--run' in sys.argv:f_status('failed',error=repr(exc),traceback=traceback.format_exc())
        raise
