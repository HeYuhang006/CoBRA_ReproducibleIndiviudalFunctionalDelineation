"""Capture the original failed optimizer inputs for an auditable K300 repair."""
from pathlib import Path
import datetime, fcntl, json, os, subprocess, sys
from s_diagnose_labelcost_20261011 import B, CODE, MATLAB, PY, sha, save, frozen
D=B/'k300_context_20261012'

def main():
    D.mkdir(exist_ok=True)
    lock=(D/'launch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if '--run' in sys.argv:
        receipt=json.loads((D/'receipt.json').read_text())
        with (D/'context.log').open('ab',buffering=0) as stream:
            p=subprocess.Popen([MATLAB,'-singleCompThread','-batch',
                f"addpath('{D}');f_rncut_capture_k300('fit','{D}/capture');"],
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            save(D/'matlab_launch.json',dict(pid=p.pid))
            result=p.wait()
        receipt.update(state='capture_finished',returncode=result,
                       finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        receipt['frozen_after'],receipt['maps_after']=frozen()
        save(D/'receipt.json',receipt);return
    assert not (D/'receipt.json').exists(),'Do not duplicate context capture'
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name)==os.getpid():continue
        try:argv=(process/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        assert not any(b"f_rncut_capture_k300('fit'" in a or
                       b"f_rncut_labelcost_diag('fit'" in a or
                       b"f_rncut_surface_driver('fit'" in a for a in argv),('Existing K300 run',process.name)
    hashes,maps=frozen()
    cfg=json.loads((B/'protocol.json').read_text())
    for path,identity in cfg['input_identities'].items():
        st=Path(path).stat();assert dict(size=st.st_size,mtime_ns=st.st_mtime_ns)==identity
    assert sha(B/'mean_individual_absolute_FC_L.npy')=='ca62ce329a5917b39d1cfc008a9be59669e78b443673eb3fd834075d011896b8'
    observer=D/'observer';observer.mkdir()
    (observer/'GCO_SetLabelCost.m').write_text(GCO_WRAPPER)
    (observer/'calcLabelCost_v1.m').write_text(COST_WRAPPER)
    driver=(CODE/'f_rncut_surface_driver.m').read_text()
    substitutions={
        'function f_rncut_surface_driver(mode,output)':'function f_rncut_capture_k300(mode,output)',
        "code=fileparts(mfilename('fullpath'));":f"code='{CODE}';",
        "fileread([output '/native_equivalence.json'])":f"fileread('{B}/native_equivalence.json')",
        "fileread([output '/input.json'])":f"fileread('{B}/input.json')",
        'for k=[50 100 150 200]':'for k=[150]',
        "[output '/optimizer_complete.json']":"[output '/capture_completed_without_error.json']"
    }
    for old,new in substitutions.items():
        assert driver.count(old)==1,old
        driver=driver.replace(old,new,1)
    before="mkdir(output);warning('off','all');"
    driver=driver.replace(before,before+"\nglobal RN_CTX_ORIGINAL_GCO RN_CTX_ORIGINAL_COST RN_CTX_OUTPUT;\nRN_CTX_ORIGINAL_GCO=@GCO_SetLabelCost;RN_CTX_ORIGINAL_COST=@calcLabelCost_v1;RN_CTX_OUTPUT=output;\n")
    adapter="addpath([code '/surface_adapter'],'-begin');"
    assert driver.count(adapter)==2
    driver=driver.replace(adapter,adapter+f"addpath('{observer}','-begin');")
    (D/'f_rncut_capture_k300.m').write_text(driver)
    # Small signature/provenance probe; no guesses about encrypted source internals.
    probe=f"""addpath(genpath('{B.parents[2]}/resources/rncut_author/source'));
addpath('{B.parents[2]}/runs/rncut_native_interface_probe_20261008_v1/linux_mex');
names={{'calcLabelCost_v1','getPairsCost','alphaExpansion_v1','KBound','NcutEnergy'}};
rows=cell(size(names));
for j=1:numel(names)
  rows{{j}}=struct('name',names{{j}},'nargin',nargin(names{{j}}),'nargout',nargout(names{{j}}),'path',which(names{{j}}));
end
fid=fopen('{D}/signatures.json','w');fwrite(fid,jsonencode(rows));fclose(fid);
"""
    (D/'signature_probe.m').write_text(probe)
    with (D/'signature_probe.log').open('wb') as stream:
        result=subprocess.run([MATLAB,'-singleCompThread','-batch',f"run('{D}/signature_probe.m');"],
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,timeout=90)
    assert result.returncode==0,'Read signature_probe.log'
    (D/'capture').mkdir()
    receipt=dict(state='verified_before_capture',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        frozen_before=hashes,maps_before=maps,random_seed=cfg['seed'],nominal_bilateral_K=300,
        numeric_arguments_unchanged=True,source_unchanged=True,official_results_unchanged=True,
        captures=['calcLabelCost arguments and output','alphaExpansion inputs','small RNcut caller variables'],
        author_code_not_decompiled=True,driver_sha256=sha(D/'f_rncut_capture_k300.m'))
    save(D/'receipt.json',receipt)
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    with (D/'runner.log').open('ab',buffering=0) as stream:
        child=subprocess.Popen([PY,str(Path(__file__).resolve()),'--run'],stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True,
            env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4'))
    save(D/'launch.json',dict(pid=child.pid));print(json.dumps(dict(pid=child.pid,folder=str(D))))

GCO_WRAPPER=r"""function GCO_SetLabelCost(Handle,LabelCost,varargin)
global RN_CTX_ORIGINAL_GCO RN_CTX_OUTPUT;
NumLabels=gco_matlab('gco_getnumlabels',Handle);
if length(LabelCost)~=1 && length(LabelCost)~=NumLabels
  names={'Dcost','Scost','Lcost','Neighbor','iniLabel','nLabels','nPoints'};
  alpha=struct();
  for j=1:numel(names),alpha.(names{j})=evalin('caller',names{j});end
  parentWhos=evalin('caller','evalin(''caller'',''whos'')');
  parent=struct();total=0;
  for j=1:numel(parentWhos)
    v=parentWhos(j);
    if v.bytes<=64*2^20 && total+v.bytes<=256*2^20 && any(strcmp(v.class,{'double','single','int32','uint32','logical','struct','cell'}))
      parent.(v.name)=evalin('caller',['evalin(''caller'',''' v.name ''')']);total=total+v.bytes;
    end
  end
  used=unique(double(alpha.iniLabel(:)))';
  counts=histcounts(double(alpha.iniLabel(:)),.5:double(NumLabels)+.5);
  report=struct('NumLabels',double(NumLabels),'cost_length',length(LabelCost),...
    'initial_occupied_labels',used,'initial_counts',counts,...
    'missing_initial_labels',find(counts==0),'parent_whos',parentWhos,'unchanged_arguments',true);
  save([RN_CTX_OUTPUT '/failed_call_context.mat'],'alpha','parent','LabelCost','NumLabels','report','-v7');
  fid=fopen([RN_CTX_OUTPUT '/failed_call_context.json'],'w');fwrite(fid,jsonencode(report));fclose(fid);
  fprintf('CAPTURED_K300 cost_length=%d nLabels=%d missing_initial=%s\n',length(LabelCost),NumLabels,mat2str(find(counts==0)'));
end
feval(RN_CTX_ORIGINAL_GCO,Handle,LabelCost,varargin{:});
end
"""
COST_WRAPPER=r"""function varargout=calcLabelCost_v1(varargin)
global RN_CTX_ORIGINAL_COST RN_CTX_OUTPUT;
[varargout{1:nargout}]=feval(RN_CTX_ORIGINAL_COST,varargin{:});
file=[RN_CTX_OUTPUT '/labelcost_arguments.mat'];
if ~isfile(file)
  inputs=cell(size(varargin));sizes=cell(size(varargin));
  for j=1:numel(varargin)
    a=varargin{j};w=whos('a');sizes{j}=struct('size',size(a),'class',class(a),'bytes',w.bytes);
    if w.bytes<=64*2^20,inputs{j}=a;end
  end
  outputs=varargout;
  save(file,'inputs','outputs','sizes','-v7');
  fid=fopen([RN_CTX_OUTPUT '/labelcost_arguments.json'],'w');fwrite(fid,jsonencode(sizes));fclose(fid);
end
end
"""
if __name__=='__main__':main()
