"""Observe the failed author call without changing arguments or official output."""
from pathlib import Path
import datetime, fcntl, hashlib, json, os, shutil, subprocess, sys

W = Path('/dataspace/yuhang_data/IndividualFunctionalParcellation/GPT_workplace')
B = W / 'runs/group_nmf_rncut_20261009_v2/RNCut'
CODE = Path('/dataspace/yuhang_data/IndividualFunctionalParcellation/CoBRA/TraditionalMethod/experiments/group_nmf_rncut_20261009')
D = B / 'labelcost_diagnostic_20261011'
AUTHOR = W / 'resources/rncut_author/source'
MATLAB = '/usr/local/MATLAB/R2023a/bin/matlab'
PY = '/home/yuhang/miniconda3/envs/pytorch/bin/python'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**22), b''): h.update(chunk)
    return h.hexdigest()

def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)

def frozen():
    p = json.loads((B / 'protocol.json').read_text())
    actual = {s: sha(s) for s in p['code_sha256']}
    assert actual == p['code_sha256'] and len(actual) == 10
    assert sha(B / 'native_equivalence.json') == p['native_equivalence_sha256']
    maps = {str(k): sha(B / f'maps/labels_L_{k}.mat') for k in [50,100]}
    assert maps['50'] == '751edc5b7b8a8b4b320106dc61dd128d418d58069e30d419a276a137f9778b19'
    assert maps['100'] == '9d223637923795d41aea40206786224194bab6fe07dc4b7cf32e35a18d9d9e54'
    return actual, maps

def main():
    D.mkdir(exist_ok=True)
    lock = (D / 'launch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if '--run' in sys.argv:
        receipt = json.loads((D / 'receipt.json').read_text())
        frozen()
        result = subprocess.run([MATLAB, '-singleCompThread', '-batch',
            f"addpath('{D}');f_rncut_labelcost_diag('fit','{D}/full');"], stdin=subprocess.DEVNULL)
        receipt.update(state='diagnostic_finished', returncode=result.returncode,
                       finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        receipt['frozen_after'], receipt['maps_after'] = frozen()
        save(D / 'receipt.json', receipt)
        return
    assert not (D / 'receipt.json').exists(), 'Do not repeat this diagnostic launch'
    for process in Path('/proc').iterdir():
        if not process.name.isdigit(): continue
        try: argv = (process / 'cmdline').read_bytes().split(b'\0')
        except OSError: continue
        if int(process.name) == os.getpid(): continue
        assert not any(a.endswith(b's_step1_prepare_and_fit_rncut.py') or
                       b"f_rncut_surface_driver('fit'" in a or
                       b"f_rncut_labelcost_diag('fit'" in a for a in argv), ('Existing fit', process.name)
    workflow = (B / 'workflow.lock').open('a')
    fcntl.flock(workflow, fcntl.LOCK_EX | fcntl.LOCK_NB)
    hashes, maps = frozen()
    protocol = json.loads((B / 'protocol.json').read_text())
    for path, identity in protocol['input_identities'].items():
        st = Path(path).stat()
        assert dict(size=st.st_size, mtime_ns=st.st_mtime_ns) == identity
    affinity_sha = sha(B / 'mean_individual_absolute_FC_L.npy')
    assert affinity_sha == 'ca62ce329a5917b39d1cfc008a9be59669e78b443673eb3fd834075d011896b8'
    assert json.loads((B / 'preparation_status.json').read_text())['state'] == 'failed'
    for name in ['preparation_status.json','status.json','matlab_fit.log']:
        shutil.copy2(B / name, D / ('failed_' + name))
    observer = D / 'observer'
    observer.mkdir(exist_ok=True)
    (observer / 'GCO_SetLabelCost.m').write_text(WRAPPER)
    driver = (CODE / 'f_rncut_surface_driver.m').read_text()
    assert driver.count("code=fileparts(mfilename('fullpath'));") == 1
    driver = driver.replace('function f_rncut_surface_driver(mode,output)',
                            'function f_rncut_labelcost_diag(mode,output)', 1)
    driver = driver.replace("code=fileparts(mfilename('fullpath'));", f"code='{CODE}';")
    marker = "mkdir(output);warning('off','all');"
    assert driver.count(marker) == 1
    driver = driver.replace(marker, marker + f"\nglobal RN_DIAG_ORIGINAL_SETLABELCOST RN_DIAG_OUTPUT;\nRN_DIAG_ORIGINAL_SETLABELCOST=@GCO_SetLabelCost;RN_DIAG_OUTPUT=output;\noriginalInfo=functions(RN_DIAG_ORIGINAL_SETLABELCOST);fprintf('ORIGINAL_GCO %s\\n',originalInfo.file);\n")
    adapter = "addpath([code '/surface_adapter'],'-begin');"
    assert driver.count(adapter) == 2
    driver = driver.replace(adapter, adapter + f"addpath('{observer}','-begin');")
    driver = driver.replace("fileread([output '/native_equivalence.json'])", f"fileread('{B}/native_equivalence.json')")
    driver = driver.replace("fileread([output '/input.json'])", f"fileread('{B}/input.json')")
    assert driver.count('for k=[50 100 150 200]') == 1
    driver = driver.replace('for k=[50 100 150 200]', 'for k=[150]')
    driver = driver.replace("[output '/optimizer_complete.json']", "[output '/diagnostic_complete.json']")
    (D / 'f_rncut_labelcost_diag.m').write_text(driver)
    # Exact native-vs-adapter toy fixtures with the observational forwarding hook.
    with (D / 'probe.log').open('wb') as stream:
        result = subprocess.run([MATLAB, '-singleCompThread', '-batch',
            f"addpath('{D}');f_rncut_labelcost_diag('validate','{D}/probe');"],
            stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, timeout=180)
    assert result.returncode == 0, 'Diagnostic hook failed the original native equivalence fixtures'
    assert json.loads((D / 'probe/native_equivalence.json').read_text())['passed']
    (D / 'full').mkdir(exist_ok=True)
    receipt = dict(state='verified_before_diagnostic',
        checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        frozen_before=hashes, maps_before=maps, affinity_sha256=affinity_sha,
        verified_input_identities=len(protocol['input_identities']),
        observer_sha256=sha(observer / 'GCO_SetLabelCost.m'),
        diagnostic_driver_sha256=sha(D / 'f_rncut_labelcost_diag.m'),
        native_equivalence_probe_passed=True, original_arguments_unchanged=True,
        official_outputs_unchanged=True, diagnostic_only_nominal_K=300,
        original_seed=protocol['seed'], failure_log_preserved=True)
    save(D / 'receipt.json', receipt)
    fcntl.flock(workflow, fcntl.LOCK_UN)
    workflow.close()
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    with (D / 'full.log').open('ab', buffering=0) as stream:
        child = subprocess.Popen([PY, str(Path(__file__).resolve()), '--run'],
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True,
            env=dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'))
    # The child owns subsequent receipt updates; write PID separately to avoid races.
    save(D / 'launch.json', dict(pid=child.pid, diagnostic_only=True, log=str(D / 'full.log')))
    print(json.dumps(dict(receipt=str(D / 'receipt.json'), pid=child.pid,
                         native_equivalence_probe_passed=True, diagnostic_only=True)))

WRAPPER = r'''function GCO_SetLabelCost(Handle,LabelCost,varargin)
% Observational forwarding hook. No padding, relabeling, reseeding, or correction.
global RN_DIAG_ORIGINAL_SETLABELCOST RN_DIAG_OUTPUT;
NumLabels=gco_matlab('gco_getnumlabels',Handle);
if length(LabelCost)~=1 && length(LabelCost)~=NumLabels
    variables=evalin('caller','whos');
    records=cell(1,numel(variables));
    for j=1:numel(variables)
        v=variables(j);
        records{j}=struct('name',v.name,'size',v.size,'class',v.class,'bytes',v.bytes);
        if v.bytes<=1024 && any(strcmp(v.class,{'double','single','int32','uint32','logical'}))
            records{j}.small_value=evalin('caller',v.name);
        end
    end
    mismatch=struct('NumLabels',double(NumLabels),'LabelCost_size',size(LabelCost),...
        'LabelCost_length',length(LabelCost),'LabelCost_class',class(LabelCost),...
        'caller_variables',{records},'stack',dbstack('-completenames'),...
        'arguments_forwarded_unchanged',true);
    save([RN_DIAG_OUTPUT '/labelcost_mismatch.mat'],'LabelCost','NumLabels','mismatch','-v7');
    fid=fopen([RN_DIAG_OUTPUT '/labelcost_mismatch.json'],'w');
    fwrite(fid,jsonencode(mismatch));fclose(fid);
    fprintf('OBSERVED_LABELCOST_MISMATCH cost_size=%s NumLabels=%d\n',mat2str(size(LabelCost)),NumLabels);
end
feval(RN_DIAG_ORIGINAL_SETLABELCOST,Handle,LabelCost,varargin{:});
end
'''

if __name__ == '__main__': main()
