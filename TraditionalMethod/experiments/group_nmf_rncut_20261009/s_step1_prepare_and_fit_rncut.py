"""Mean of individual absolute FC -> unchanged author RNcut, mesh interfaces.

Native voxel equivalence must pass first. This is an explicitly declared surface
adaptation, not a feature-kNN or mesh-only spectral substitute.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import fcntl, hashlib, json, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
from scipy.io import savemat, loadmat
import torch

W=Path('/dataspace/yuhang_data/IndividualFunctionalParcellation/GPT_workplace')
sys.path.insert(0,str(W))
from run import filtered_timeseries
CODE=Path(__file__).resolve().parent
D=W/'runs/full500_MSMAll_samplez'
OUT=W/'runs/group_nmf_rncut_20261009_v2/RNCut'


def f_save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False));temp.replace(path)


def f_digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(2**22),b''):h.update(b)
    return h.hexdigest()


def f_status(**kw):
    f_save(OUT/'preparation_status.json',dict(pid=os.getpid(),updated_unix=time.time(),**kw))


def f_code_hashes():
    paths=[Path(__file__),CODE/'f_rncut_surface_driver.m',*sorted((CODE/'surface_adapter').glob('*.m'))]
    author=W/'resources/rncut_author/source'
    paths += [author/f for f in ['RNcut.p','KBound.p','NcutEnergy.p','optimize_tools/getPairsCost.p',
        'optimize_tools/calcLabelCost_v1.p','optimize_tools/alphaExpansion_v1.p']]
    return {str(p):f_digest(p) for p in paths}


@torch.no_grad()
def f_affinity(protocol,mapping):
    target=OUT/'mean_individual_absolute_FC_L.npy'
    if (OUT/'affinity_complete.json').exists():
        stored=json.loads((OUT/'affinity_complete.json').read_text())
        assert stored['protocol']==protocol and f_digest(target)==stored['sha256']
        return target
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.25)
    assert torch.cuda.mem_get_info()[0]>5*2**30,'Insufficient spare GPU memory; existing jobs are never stopped'
    n=len(mapping['L']['vertices']);acc=torch.zeros(n,n,device='cuda',dtype=torch.float32)
    checkpoint=OUT/'affinity_sum_checkpoint.npy';meta=OUT/'affinity_checkpoint.json';start=0;sources=[]
    if meta.exists():
        old=json.loads(meta.read_text());assert old['protocol']==protocol
        assert f_digest(checkpoint)==old['sha256']
        acc.copy_(torch.from_numpy(np.load(checkpoint)));start=old['completed'];sources=old['sources']
    for index,row in enumerate(protocol['train_rows']):
        if index<start:continue
        f_status(state='running',phase='training_subject_absolute_FC',completed=index,total=500,subject=row['subject'])
        ts,provenance=filtered_timeseries(row,mapping,{'use_verified_practice_cache':True})
        y=np.array(ts[:,:n],copy=True,dtype='float32');del ts
        y-=y.mean(0);y/=y.std(0).clip(1e-8)
        y=torch.as_tensor(np.ascontiguousarray(y),device='cuda')
        for begin in range(0,n,512):
            block=(y[:,begin:begin+512].T@y)/len(y);acc[begin:begin+len(block)].add_(block.abs())
        sources.append(dict(subject=row['subject'],filter_Hz=[.01,.1],verification=provenance))
        del y,block;torch.cuda.empty_cache()
        if (index+1)%25==0 or index+1==500:
            temporary=checkpoint.with_suffix('.tmp.npy');np.save(temporary,acc.cpu().numpy());temporary.replace(checkpoint)
            f_save(meta,dict(protocol=protocol,completed=index+1,sources=sources,sha256=f_digest(checkpoint)))
        f_status(state='running',phase='training_subject_complete',completed=index+1,total=500,subject=row['subject'])
        print('ABSOLUTE_FC',index+1,row['subject'],flush=True)
    acc/=500;acc.fill_diagonal_(0)
    assert torch.isfinite(acc).all() and acc.min()>=0
    np.save(target,acc.cpu().numpy());del acc;torch.cuda.empty_cache()
    f_save(OUT/'affinity_complete.json',dict(protocol=protocol,subjects=500,sources=sources,
        sha256=f_digest(target),formula='mean_s(abs(corr(Y_s))); diagonal zero',
        not_equivalent_to='abs(mean_s(corr(Y_s)))',completed_unix=time.time()))
    return target


def f_main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    test=json.loads((OUT/'native_equivalence.json').read_text());assert test['passed']
    split=json.loads((D/'subjects.json').read_text());rows=split['train'];assert len(rows)==500
    eval_ids=json.loads((W/'runs/published_atlas_group_20261008_v1/protocol.json').read_text())['subjects']
    assert not {r['subject'] for r in rows}&set(eval_ids)
    packed=np.load(D/'mapping.npz');mapping={h:{key:packed[h+'_'+key] for key in ['vertices','columns']} for h in ['L','R']}
    hashes=f_code_hashes()
    input_identities={r['path']:dict(size=Path(r['path']).stat().st_size,mtime_ns=Path(r['path']).stat().st_mtime_ns) for r in rows}
    protocol=dict(train_rows=rows,training_people=500,evaluation_people=50,evaluation_subjects=eval_ids,
        bandpass_Hz=[.01,.1],run='rfMRI_REST1_LR',surface_registration='MSMAll',
        K_nominal_bilateral=[100,200,300,400],K_left=[50,100,150,200],seed=20261009,
        affinity='Mean of individual absolute Pearson FC, diagonal zero, matching author group similarity script',
        inference='Original RNcut P-code, KBound, NcutEnergy, pair costs, component label costs and alpha expansion',
        spatial_adaptation='Mesh one-ring replaces voxel26 neighbors and components; container voxel geometry is never used',
        fair_scope='Left cortex matched support. Algorithmic compatibility tested on native voxel domains. This is a surface-domain extension, not the original volumetric experiment.',
        tuning='Author hard-coded defaults unchanged; no evaluation-label/metric/task tuning or extra connectivity repair',
        code_sha256=hashes,input_identities=input_identities,native_equivalence_sha256=f_digest(OUT/'native_equivalence.json'))
    dest=OUT/'protocol.json'
    if dest.exists():assert json.loads(dest.read_text())==protocol,'Changed experiment requires a new output version'
    f_save(dest,protocol)
    target=f_affinity(protocol,mapping)
    n=len(mapping['L']['vertices'])
    with target.open('rb') as f:
        version=np.lib.format.read_magic(f);shape,fortran,dtype=np.lib.format._read_array_header(f,version);offset=f.tell()
    assert shape==(n,n) and dtype==np.dtype('float32') and not fortran
    edges_path=OUT/'surface_edges_L.mat';savemat(edges_path,dict(edges=np.load(D/'edges_L.npy')))
    f_save(OUT/'input.json',dict(affinity_npy=str(target),header_bytes=offset,vertices=n,edges_mat=str(edges_path),seed=protocol['seed']))
    f_status(state='running',phase='author_RNcut_optimization')
    command=['/usr/local/MATLAB/R2023a/bin/matlab','-singleCompThread','-batch',
        f"addpath('{CODE}');f_rncut_surface_driver('fit','{OUT}');"]
    with (OUT/'matlab_fit.log').open('a') as log:
        completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    assert completed.returncode==0,f'MATLAB RNcut failed, see {OUT}/matlab_fit.log'
    assert json.loads((OUT/'optimizer_complete.json').read_text())['state']=='complete'
    for k in [50,100,150,200]:
        path=OUT/f'maps/labels_L_{k}.mat';a=loadmat(path);labels=a['labels'].ravel().astype('int32')
        assert labels.shape==(n,) and labels.min()>=0
        if len(np.unique(labels))!=k:
            raise RuntimeError(f'Author RNcut returned {len(np.unique(labels))}/{k} occupied labels; preserve output and report resolution mismatch, never repair it with CoBRA.')
        np.save(path.with_suffix('.npy'),labels)
    assert f_code_hashes()==hashes,'Source changed during fitting'
    f_save(OUT/'complete.json',dict(state='complete',training_people=500,native_voxel_equivalence_passed=True,
        adaptation='Original author optimizer; mesh spatial interfaces',K_nominal_bilateral=[100,200,300,400],
        code_sha256=hashes,affinity_sha256=f_digest(target),completed_unix=time.time()))
    f_status(state='complete',phase='maps_complete')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1')
    evaluator=CODE/'s_step3_evaluate_group.py'
    subprocess.run([sys.executable,str(evaluator),'--family','RNCut'],env=env,check=True)
    subprocess.run([sys.executable,str(evaluator),'--family','merge'],env=env,check=True)
    f_status(state='complete',phase='group_comparison_complete')


if __name__=='__main__':
    try:f_main()
    except Exception as e:
        f_status(state='failed',error=repr(e),traceback=traceback.format_exc());raise
