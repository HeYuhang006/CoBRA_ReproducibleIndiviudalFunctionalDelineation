"""Read-only scientific diagnostic of the K400 initialization on frozen inputs.

The original RNcut invokes a shadow of public ncutW that saves the continuous
eigensystem and declared rounding diagnostics, then deliberately stops BEFORE
optimization. The original 195-label output is preserved. No result promotion,
seed search, posthoc label repair or evaluation selection is performed.
"""
import datetime, fcntl, json, os, subprocess, sys
from pathlib import Path
from s_diagnose_labelcost_20261011 import B, CODE, MATLAB, PY, frozen, sha, save

D = B / 'k400_initialization_diagnostic_20261012'
LAYER = CODE / 'k400_initialization_diagnostic_20261012'


def verify():
    author, maps = frozen()
    cfg = json.loads((B / 'protocol.json').read_text())
    for p, identity in cfg['input_identities'].items():
        st = Path(p).stat()
        assert dict(size=st.st_size, mtime_ns=st.st_mtime_ns) == identity
    assert sha(B / 'mean_individual_absolute_FC_L.npy') == 'ca62ce329a5917b39d1cfc008a9be59669e78b443673eb3fd834075d011896b8'
    return dict(author=author, maps=maps,
                original_K400=sha(B / 'independent_K400_20261011/maps/labels_L_200.mat'),
                diagnostic_layer=sha(LAYER / 'ncutW.m'),
                rounding_code=sha(CODE / 'f_discretisation_diagnostic_20261012.m'))


def main():
    D.mkdir(exist_ok=True)
    lock = (D / 'workflow.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if '--run' in sys.argv:
        receipt = json.loads((D / 'receipt.json').read_text())
        assert verify() == receipt['integrity_before']
        receipt.update(state='running', pid=os.getpid())
        save(D / 'receipt.json', receipt)
        with (D / 'optimizer_diagnostic.log').open('ab', buffering=0) as out:
            p = subprocess.Popen([MATLAB, '-singleCompThread', '-batch',
                f"addpath('{D}');f_rncut_k400_initialization_diagnostic('fit','{D}');"],
                stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
            save(D / 'matlab_launch.json', dict(pid=p.pid))
            rc = p.wait()
        receipt.update(returncode=rc, integrity_after=verify(),
                       finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        assert receipt['integrity_before'] == receipt['integrity_after']
        audit = D / 'initialization_K200.json'
        expected = 'Initialization diagnostic saved; no optimization was executed'
        if rc == 1 and audit.exists() and expected in (D / 'optimizer_diagnostic.log').read_text():
            receipt.update(state='diagnostic_complete_expected_stop',
                           initialization=json.loads(audit.read_text()),
                           continuous_cache_sha256=sha(D / 'continuous_K200.mat'))
        else:
            receipt.update(state='diagnostic_failed')
        save(D / 'receipt.json', receipt)
        return
    assert not (D / 'receipt.json').exists(), 'Never duplicate this diagnostic'
    assert not (D / 'continuous_K200.mat').exists(), 'Preserve any existing cache'
    source = (CODE / 'f_rncut_surface_driver.m').read_text()
    replacements = {
        'function f_rncut_surface_driver(mode,output)': 'function f_rncut_k400_initialization_diagnostic(mode,output)',
        "code=fileparts(mfilename('fullpath'));": f"code='{CODE}';addpath(code);",
        "fileread([output '/native_equivalence.json'])": f"fileread('{B}/native_equivalence.json')",
        "fileread([output '/input.json'])": f"fileread('{B}/input.json')",
        'for k=[50 100 150 200]': 'for k=[200]'
    }
    for old, new in replacements.items():
        assert source.count(old) == 1, old
        source = source.replace(old, new, 1)
    marker = "mkdir(output);warning('off','all');"
    setup = f"""
global RN_INIT_CACHE RN_INIT_AUDIT RN_INIT_SEED;
RN_INIT_CACHE='{D}/continuous_K200.mat';RN_INIT_AUDIT='{D}/initialization_K200';
RN_INIT_SEED=20261009;
"""
    source = source.replace(marker, marker + setup, 1)
    marker = "f_activate(mask,neighbors);addpath([code '/surface_adapter'],'-begin');"
    assert source.count(marker) == 1
    source = source.replace(marker, marker + f"addpath('{LAYER}','-begin');", 1)
    (D / 'f_rncut_k400_initialization_diagnostic.m').write_text(source)
    save(D / 'receipt.json', dict(state='verified_before_launch',
         integrity_before=verify(), diagnostic_only=True, no_optimizer_executed=True,
         nominal_bilateral_K=400, requested_left_K=200, seed=20261009,
         rounding_protocol='original; if empty same solver cap200; if still empty author raw first center fixed seed cap200',
         no_evaluation_selection=True, no_posthoc_label_repair=True,
         driver_sha256=sha(D / 'f_rncut_k400_initialization_diagnostic.m')))
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    with (D / 'runner.log').open('ab', buffering=0) as out:
        p = subprocess.Popen([PY, str(Path(__file__).resolve()), '--run'],
             stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
             start_new_session=True, close_fds=True,
             env=dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
    save(D / 'launch.json', dict(pid=p.pid))
    print(json.dumps(dict(pid=p.pid, folder=str(D))))


if __name__ == '__main__':
    main()
