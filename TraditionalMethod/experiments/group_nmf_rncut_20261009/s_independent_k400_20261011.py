"""Run the already specified K400 independently; never repair failed K300."""
from pathlib import Path
import datetime, fcntl, json, os, subprocess, sys
from s_diagnose_labelcost_20261011 import B, CODE, MATLAB, PY, sha, save, frozen

D = B / 'independent_K400_20261011'

def verify():
    hashes, maps = frozen()
    protocol = json.loads((B / 'protocol.json').read_text())
    for path, identity in protocol['input_identities'].items():
        st = Path(path).stat()
        assert dict(size=st.st_size, mtime_ns=st.st_mtime_ns) == identity
    affinity_sha = sha(B / 'mean_individual_absolute_FC_L.npy')
    assert affinity_sha == 'ca62ce329a5917b39d1cfc008a9be59669e78b443673eb3fd834075d011896b8'
    return dict(frozen_sources=hashes, preserved_maps=maps,
                verified_input_identities=len(protocol['input_identities']), affinity_sha256=affinity_sha)

def main():
    D.mkdir(exist_ok=True)
    lock = (B / 'workflow.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if '--run' in sys.argv:
        receipt = json.loads((D / 'receipt.json').read_text())
        receipt.update(state='running', pid=os.getpid())
        save(D / 'receipt.json', receipt)
        result = subprocess.run([MATLAB, '-singleCompThread', '-batch',
            f"addpath('{D}');f_rncut_independent_k400('fit','{D}');"], stdin=subprocess.DEVNULL)
        receipt.update(state='optimizer_returned', returncode=result.returncode,
                       finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       integrity_after=verify())
        save(D / 'receipt.json', receipt)
        if result.returncode == 0:
            import numpy as np
            from scipy.io import loadmat
            p = D / 'maps/labels_L_200.mat'
            labels = loadmat(p)['labels'].reshape(-1)
            cfg = json.loads((B / 'input.json').read_text())
            assert labels.shape == (cfg['vertices'],) and np.isfinite(labels).all()
            assert np.array_equal(labels, labels.astype('int64'))
            assert labels.min() == 0 and labels.max() == 199 and len(np.unique(labels)) == 200
            receipt.update(state='validated_single_resolution', map_sha256=sha(p),
                           occupied_left=200, full_campaign_complete=False,
                           official_results_not_promoted=True)
            save(D / 'receipt.json', receipt)
        return
    assert not (D / 'receipt.json').exists(), 'Do not launch the same independent run twice'
    assert not (B / 'maps/labels_L_200.mat').exists(), 'Original K400 result already exists'
    diag = json.loads((B / 'labelcost_diagnostic_20261011/receipt.json').read_text())
    assert diag['state'] == 'diagnostic_finished' and diag['returncode'] == 1
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid(): continue
        try: argv = (process / 'cmdline').read_bytes().split(b'\0')
        except OSError: continue
        assert not any(a.endswith(b's_step1_prepare_and_fit_rncut.py') or
                       b"f_rncut_surface_driver('fit'" in a or
                       b"f_rncut_labelcost_diag('fit'" in a or
                       b"f_rncut_independent_k400('fit'" in a for a in argv), ('Existing fit', process.name)
    integrity = verify()
    source = (CODE / 'f_rncut_surface_driver.m').read_text()
    replacements = {
        'function f_rncut_surface_driver(mode,output)': 'function f_rncut_independent_k400(mode,output)',
        "code=fileparts(mfilename('fullpath'));": f"code='{CODE}';",
        "fileread([output '/native_equivalence.json'])": f"fileread('{B}/native_equivalence.json')",
        "fileread([output '/input.json'])": f"fileread('{B}/input.json')",
        'for k=[50 100 150 200]': 'for k=[200]',
        "[output '/optimizer_complete.json']": "[output '/single_resolution_optimizer_complete.json']"
    }
    for old, new in replacements.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new, 1)
    # Original loop resets rng(cfg.seed) before EVERY resolution. Ordering
    # therefore cannot change K400's random initialization. All numeric calls,
    # mask, spatial adapter, FC matrix and author optimizer remain identical.
    assert 'rng(cfg.seed);started=tic;labelsVolume=RNcut(W,mask,k);' in source
    (D / 'f_rncut_independent_k400.m').write_text(source)
    receipt = dict(state='verified_before_launch', checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        integrity_before=integrity, source_driver_sha256=sha(CODE / 'f_rncut_surface_driver.m'),
        runtime_driver_sha256=sha(D / 'f_rncut_independent_k400.m'),
        nominal_bilateral_K=400, actual_left_K=200, random_seed=20261009,
        algorithm_and_numeric_inputs_unchanged=True, ordering_only=True,
        no_label_repair=True, official_outputs_unchanged=True,
        full_campaign_complete=False, K300_still_failed=True)
    save(D / 'receipt.json', receipt)
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    with (D / 'optimizer.log').open('ab', buffering=0) as stream:
        child = subprocess.Popen([PY, str(Path(__file__).resolve()), '--run'],
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
            env=dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
    save(D / 'launch.json', dict(pid=child.pid, nominal_bilateral_K=400, log=str(D / 'optimizer.log')))
    print(json.dumps(dict(pid=child.pid, receipt=str(D / 'receipt.json'), independent_K400=True)))

if __name__ == '__main__': main()
