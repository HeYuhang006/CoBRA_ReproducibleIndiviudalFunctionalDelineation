"""Finite post-fit queue, preserving originals and the full scientific gates.

Waits for the K300 candidate AND independent K400 to validate. Copies genuine
maps into an explicitly versioned comparison, reuses unchanged NMF measurements,
then calls the original evaluator/60-test merger/bundler without numerical edits.
It never fabricates a map, loosens matched-K checks, or marks PPT QA complete.
"""
from pathlib import Path
import datetime,fcntl,importlib.util,json,os,shutil,subprocess,sys,time
import numpy as np
from scipy.io import loadmat
from s_diagnose_labelcost_20261011 import B,CODE,PY,sha,save,frozen

D=B/'initialization_guard_20261012'
ORIGINAL=B.parent
OUT=ORIGINAL.with_name('group_nmf_rncut_20261012_initialization_guard_v3')

def state():
    a=json.loads((D/'receipt.json').read_text())
    b=json.loads((B/'independent_K400_20261011/receipt.json').read_text())
    return a,b

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

def ready(a,b):
    return a['state']=='validated_candidate_resolution' and b['state']=='validated_single_resolution'

def finish(a,b):
    original_hashes,original_maps=frozen()
    assert a['integrity_before']==a['integrity_after']
    assert a['integrity_after']['author']==original_hashes
    assert a['integrity_after']['maps']==original_maps
    assert b['integrity_after']['frozen_sources']==original_hashes
    assert b['integrity_after']['preserved_maps']==original_maps
    init=json.loads((D/'initialization_K150.json').read_text())
    assert init['K_occupied_selected']==150 and init['no_posthoc_label_repair'] and not init['seed_search']
    assert a['map_sha256']==sha(D/'maps/labels_L_150.mat')
    assert b['map_sha256']==sha(B/'independent_K400_20261011/maps/labels_L_200.mat')
    maps={50:B/'maps/labels_L_50.mat',100:B/'maps/labels_L_100.mat',
          150:D/'maps/labels_L_150.mat',200:B/'independent_K400_20261011/maps/labels_L_200.mat'}
    arrays={}
    for k,p in maps.items():
        y=loadmat(p)['labels'].reshape(-1)
        assert y.shape==(29696,) and np.isfinite(y).all()
        assert np.array_equal(y,y.astype('int32')) and y.min()==0 and y.max()==k-1
        assert len(np.unique(y))==k
        arrays[k]=y.astype('int32')
    original_protocol=json.loads((B/'protocol.json').read_text())
    affinity_sha=sha(B/'mean_individual_absolute_FC_L.npy')
    assert affinity_sha=='ca62ce329a5917b39d1cfc008a9be59669e78b443673eb3fd834075d011896b8'
    metric_script=CODE/'s_step3_evaluate_group.py';bundle_script=CODE/'s_step5_bundle.py'
    scripts={str(p):sha(p) for p in [metric_script,bundle_script,Path(__file__).resolve()]}
    provenance=dict(state='verified_inputs',source_run=str(ORIGINAL),output_run=str(OUT),
        original_protocol=original_protocol,author_code_sha256=original_hashes,
        source_maps={str(k):dict(path=str(p),sha256=sha(p)) for k,p in maps.items()},
        affinity_sha256=affinity_sha,initialization_K300=init,candidate_receipt=a,K400_receipt=b,
        metric_and_merger_code_unchanged=True,scripts_sha256=scripts,
        resolution_scope='Nominal bilateral100/200/300/400; actual left50/100/150/200; no bilateral coupling',
        disclosure='RNCut surface adaptation. K300 uses the declared training-only initialization fallback if selected; author P-code objective and optimizer unchanged. Other resolutions retain the preserved original fits.',
        scientific_gates='Exactly35 maps,50 heldout identities,5250 measurements,60 paired Holm-corrected tests. No posthoc label repair or evaluation tuning.',
        PPT_visual_review_complete=False)
    OUT.mkdir(exist_ok=True)
    path=OUT/'repair_provenance.json'
    if path.exists():assert json.loads(path.read_text())==provenance,'Changed provenance needs a new run'
    save(path,provenance)
    target=OUT/'RNCut';(target/'maps').mkdir(parents=True,exist_ok=True)
    for k,y in arrays.items():
        dest=target/f'maps/labels_L_{k}.npy'
        if dest.exists():assert np.array_equal(np.load(dest),y)
        else:np.save(dest,y)
    for name in ['native_equivalence.json']:
        src=B/name;dst=target/name
        if dst.exists():assert sha(dst)==sha(src)
        else:shutil.copy2(src,dst)
    fit=dict(state='complete',training_people=500,native_voxel_equivalence_passed=True,
        K_nominal_bilateral=[100,200,300,400],K_left=[50,100,150,200],
        code_sha256=original_hashes,affinity_sha256=affinity_sha,
        adaptation=provenance['disclosure'],initialization_K300=init,
        provenance_sha256=sha(path),runtime_compatibility_sha256=a['integrity_after']['repair'])
    save(target/'complete.json',fit)
    nmf_source=ORIGINAL/'NMF';nmf_target=OUT/'NMF'
    assert json.loads((nmf_source/'summary.json').read_text())['state']=='complete'
    if not nmf_target.exists():shutil.copytree(nmf_source,nmf_target)
    for src in nmf_source.rglob('*'):
        if src.is_file():assert sha(src)==sha(nmf_target/src.relative_to(nmf_source))
    save(D/'postfit_status.json',dict(state='running',phase='same50_original_evaluator',output=str(OUT)))
    E=module('rncut_repaired_original_evaluator',metric_script);E.OUT=OUT
    E.f_evaluate('RNCut');E.f_merge()
    summary=json.loads((OUT/'figure_data/summary.json').read_text())
    assert len(summary['participants'])==50 and len(summary['comparisons'])==60
    assert sum(len(x['metrics'])*3 for x in summary['participants'])==5250
    # Add an explicit disclosure alongside unchanged numerical output, not just
    # in an external note. No metric, comparison or adjusted p-value changes.
    summary['protocol']['RNCut_initialization_fix']=provenance['disclosure']
    save(OUT/'figure_data/summary.json',summary)
    shutil.copy2(path,OUT/'figure_data/initialization_repair_provenance.json')
    P=module('rncut_repaired_original_bundler',bundle_script);P.ROOT=OUT;P.f_main()
    assert {str(p):sha(p) for p in [metric_script,bundle_script,Path(__file__).resolve()]}==scripts
    assert frozen()==(original_hashes,original_maps)
    save(D/'postfit_status.json',dict(state='complete',phase='verified_measured_bundle',output=str(OUT),
        bundle=str(OUT/'group_figure_data.zip'),bundle_sha256=sha(OUT/'group_figure_data.zip'),
        full_metrics_complete=True,PPT_visual_review_complete=False))

def main():
    lock=(D/'postfit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if '--run' not in sys.argv:
        assert not (D/'postfit_launch.json').exists(),'Finite post-fit queue already exists'
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
        with (D/'postfit.log').open('ab',buffering=0) as stream:
            p=subprocess.Popen([PY,str(Path(__file__).resolve()),'--run'],stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
        save(D/'postfit_launch.json',dict(pid=p.pid));print(json.dumps(dict(pid=p.pid)));return
    save(D/'postfit_status.json',dict(state='waiting_for_validated_maps',pid=os.getpid(),
         started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),maximum_wait_seconds=72*3600))
    deadline=time.monotonic()+72*3600
    while time.monotonic()<deadline:
        a,b=state()
        if ready(a,b):break
        terminal_bad={'resolution_mismatch_preserved','optimizer_returned'}
        if (a['state'] in terminal_bad and a.get('returncode',0)!=0) or a['state']=='resolution_mismatch_preserved' or (b['state']=='optimizer_returned' and b.get('returncode')!=0):
            save(D/'postfit_status.json',dict(state='blocked_by_fit_failure',K300=a['state'],K400=b['state']));return
        for label,r in [('K300',a),('K400',b)]:
            if r['state']=='optimizer_returned' and r.get('returncode')==0 and not Path(f'/proc/{r["pid"]}').exists():
                save(D/'postfit_status.json',dict(state='blocked_by_incomplete_map_validation',resolution=label));return
        time.sleep(30)
    else:
        save(D/'postfit_status.json',dict(state='expired_without_modifying_fits'));return
    while time.monotonic()<deadline:
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
            '--format=csv,noheader,nounits','-i','1'],text=True).strip())
        if free>=3072:break
        save(D/'postfit_status.json',dict(state='waiting_for_free_GPU_memory',GPU=1,free_MiB=free))
        time.sleep(30)
    else:
        save(D/'postfit_status.json',dict(state='expired_waiting_for_GPU_without_stopping_other_jobs'));return
    finish(a,b)

if __name__=='__main__':
    try:main()
    except Exception as e:
        import traceback
        save(D/'postfit_status.json',dict(state='failed',error=repr(e),traceback=traceback.format_exc()))
        raise
