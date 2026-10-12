"""Versioned K300 initialization repair; frozen author P-code is never edited.

Only launches after the captured real failure proves an initially unused class.
No zero-padding, NaN replacement, seed search, or posthoc label repair is used.
"""
from pathlib import Path
import datetime,fcntl,json,os,subprocess,sys
import numpy as np
from scipy.io import loadmat
from s_diagnose_labelcost_20261011 import B,CODE,MATLAB,PY,sha,save,frozen

D=B/'initialization_guard_20261012_v2'
CAPTURE=B/'k300_context_20261012'
LAYER=CODE/'initialization_guard_20261012_v2'

def verify():
    author,maps=frozen()
    protocol=json.loads((B/'protocol.json').read_text())
    for p,identity in protocol['input_identities'].items():
        st=Path(p).stat();assert dict(size=st.st_size,mtime_ns=st.st_mtime_ns)==identity
    assert sha(B/'mean_individual_absolute_FC_L.npy')=='ca62ce329a5917b39d1cfc008a9be59669e78b443673eb3fd834075d011896b8'
    paths=[LAYER/'ncutW.m',LAYER/'alphaExpansion_v1.m',
           CODE/'f_discretisation_diagnostic_20261012.m',
           CODE/'f_validate_initialization_guard_v2_20261012.m']
    return dict(author=author,maps=maps,repair={str(p):sha(p) for p in paths})

def main():
    D.mkdir(exist_ok=True)
    lock=(D/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if '--run' in sys.argv:
        receipt=json.loads((D/'receipt.json').read_text())
        assert verify()==receipt['integrity_before']
        assert sha(B/'initialization_guard_20261012/continuous_K150.mat')==receipt['continuous_cache_sha256']
        receipt.update(state='running',pid=os.getpid());save(D/'receipt.json',receipt)
        with (D/'optimizer.log').open('ab',buffering=0) as stream:
            p=subprocess.Popen([MATLAB,'-singleCompThread','-batch',
                f"addpath('{D}');f_rncut_initialization_guard('fit','{D}');"],
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            save(D/'matlab_launch.json',dict(pid=p.pid));result=p.wait()
        receipt.update(state='optimizer_returned',returncode=result,
            finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),integrity_after=verify())
        assert receipt['integrity_before']==receipt['integrity_after']
        assert sha(B/'initialization_guard_20261012/continuous_K150.mat')==receipt['continuous_cache_sha256']
        save(D/'receipt.json',receipt)
        if result!=0:return
        labels=loadmat(D/'maps/labels_L_150.mat')['labels'].reshape(-1)
        assert labels.shape==(29696,) and np.isfinite(labels).all()
        assert np.array_equal(labels,labels.astype('int64'))
        occupied=len(np.unique(labels))
        receipt.update(occupied_left=occupied,map_sha256=sha(D/'maps/labels_L_150.mat'))
        if occupied!=150:
            receipt.update(state='resolution_mismatch_preserved',no_label_repair=True)
            save(D/'receipt.json',receipt);return
        assert labels.min()==0 and labels.max()==149
        receipt.update(state='validated_candidate_resolution',full_campaign_complete=False,
            official_results_not_promoted=True,initialization=json.loads((D/'initialization_K150.json').read_text()))
        save(D/'receipt.json',receipt)
        save(D/'candidate_map_validated.json',receipt)
        return
    assert not (D/'receipt.json').exists(),'Do not duplicate repair run'
    capture=json.loads((CAPTURE/'receipt.json').read_text())
    assert capture['state']=='capture_finished' and capture['returncode']==1
    a=loadmat(CAPTURE/'capture/failed_call_context.mat',simplify_cells=True)['alpha']
    initial=np.asarray(a['iniLabel']).reshape(-1)
    args=loadmat(CAPTURE/'capture/labelcost_arguments.mat',simplify_cells=True)['inputs']
    first_cost_labels=np.asarray(args[1]).reshape(-1)
    assert len(np.unique(initial))<150 and len(np.unique(first_cost_labels))<150, 'Failure is not an initially unused class: inspect saved context before choosing another repair'
    assert set(np.unique(initial))==set(np.unique(first_cost_labels))
    validation=json.loads((D/'native_validation/validation.json').read_text());assert validation['passed']
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name)==os.getpid():continue
        try:argv=(process/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        assert not any(b"f_rncut_initialization_guard('fit'" in x or
                       b"f_rncut_capture_k300('fit'" in x or
                       b"f_rncut_surface_driver('fit'" in x for x in argv),('Existing K300 fit',process.name)
    cache=B/'initialization_guard_20261012/continuous_K150.mat'
    assert cache.exists(), 'Cached original continuous eigensystem required; never recompute silently'
    diagnostic=json.loads((B/'raw_rounding_convergence_20261012/raw_cap200.json').read_text())
    assert diagnostic['occupied']==150 and diagnostic['seed']==20261009
    assert diagnostic['trace']['reason']=='original_objective_tolerance'
    integrity=verify()
    source=(CODE/'f_rncut_surface_driver.m').read_text()
    replacements={
        'function f_rncut_surface_driver(mode,output)':'function f_rncut_initialization_guard(mode,output)',
        "code=fileparts(mfilename('fullpath'));":f"code='{CODE}';addpath(code);",
        "fileread([output '/native_equivalence.json'])":f"fileread('{B}/native_equivalence.json')",
        "fileread([output '/input.json'])":f"fileread('{B}/input.json')",
        'for k=[50 100 150 200]':'for k=[150]',
        "[output '/optimizer_complete.json']":"[output '/candidate_optimizer_returned.json']"
    }
    for old,new in replacements.items():
        assert source.count(old)==1,old;source=source.replace(old,new,1)
    marker="mkdir(output);warning('off','all');"
    setup=f"""
global RN_INIT_ORIGINAL_ALPHA RN_INIT_CACHE RN_INIT_AUDIT RN_INIT_SEED RN_INIT_ALPHA_CALLS;
RN_INIT_ORIGINAL_ALPHA=@alphaExpansion_v1;
RN_INIT_CACHE='{B}/initialization_guard_20261012/continuous_K150.mat';RN_INIT_AUDIT='{D}/initialization_K150';
RN_INIT_SEED=20261009;RN_INIT_ALPHA_CALLS=0;
"""
    source=source.replace(marker,marker+setup,1)
    marker="f_activate(mask,neighbors);addpath([code '/surface_adapter'],'-begin');"
    assert source.count(marker)==1
    source=source.replace(marker,marker+f"addpath('{LAYER}','-begin');",1)
    (D/'f_rncut_initialization_guard.m').write_text(source)
    receipt=dict(state='verified_before_launch',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        integrity_before=integrity,driver_sha256=sha(D/'f_rncut_initialization_guard.m'),
        real_capture_sha256=sha(CAPTURE/'capture/failed_call_context.mat'),
        first_cost_occupied=len(np.unique(first_cost_labels)),native_validation=validation,
        continuous_cache_sha256=sha(cache),convergence_diagnostic=diagnostic,
        nominal_bilateral_K=300,requested_left_K=150,seed=20261009,
        scientific_change='Conditional initialization fallback only: original rounding; then same solver cap200; then single author-supplied raw first-center rule with fixed seed, extended cap200 to convergence. Never select using evaluation metrics.',
        unchanged=['500-person mean absolute FC','bandpass 0.01-0.1Hz','mask','mesh adjacency','author RNcut P-code','KBound','NcutEnergy','pair costs','component penalty','author alpha expansion'],
        safeguards=['finite optimizer inputs before int32 conversion','save continuous eigenvectors for replay','preserve original maps','no zero-padding','no NaN replacement','no posthoc label repair','no seed search'],
        full_campaign_complete=False,official_results_not_promoted=True)
    save(D/'receipt.json',receipt)
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    with (D/'runner.log').open('ab',buffering=0) as stream:
        p=subprocess.Popen([PY,str(Path(__file__).resolve()),'--run'],stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,
            env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
    save(D/'launch.json',dict(pid=p.pid));print(json.dumps(dict(pid=p.pid,folder=str(D))))

if __name__=='__main__':main()
