"""Add genuine training-group NMF/RNcut maps to the frozen 50-person benchmark.

No VAE features, tuning on evaluation participants, changed support, or fabricated
missing observations. --family NMF can finish while RNcut is being prepared.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse, fcntl, json, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from scipy.stats import wilcoxon

W=Path('/dataspace/yuhang_data/IndividualFunctionalParcellation/GPT_workplace')
sys.path.insert(0,str(W/'paper_figure_campaign'))
import group_subject_boxplot_experiment as G
import group_raw_ablation_metrics as R
OUT=W/'runs/group_nmf_rncut_20261009_v2'
BASE=W/'runs/published_atlas_group_20261008_v1'
NMF=W/'runs/group_subject_boxplots_20261008_v1/NMF'


def f_save(path, data):
    G.save(path, data)


def f_status(family, **kw):
    f_save(OUT/family/'status.json', dict(pid=os.getpid(),updated_unix=time.time(),**kw))


@torch.no_grad()
def f_evaluate(family):
    folder=OUT/family;folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'evaluation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.08)
    G.tests()
    base=json.loads((BASE/'summary.json').read_text());assert base['state']=='complete'
    original_inputs={r['subject']:r for r in json.loads((G.OUT/'summary.json').read_text())['participants']}
    mask=np.load(BASE/'evaluation_mask_L.npy');ids=np.flatnonzero(mask)
    mp=np.load(G.D/'mapping.npz');n=len(mask)
    maps={};sources={}
    for K in [100,200,300,400]:
        path=(NMF if family=='NMF' else OUT/'RNCut/maps')/f'labels_L_{K//2}.npy'
        v=np.load(path);assert v.shape==(n,) and np.min(v)>=0
        assert len(np.unique(v))==K//2, (path,len(np.unique(v)))
        maps[f'{family}_{K}']=v;sources[str(K)]=dict(path=str(path),sha256=G.digest(path))
    if family=='NMF':
        fit=json.loads((NMF/'complete.json').read_text())
        assert fit['training_subjects']==500 and fit['feature'].startswith('positive part')
        assert G.digest(G.D/'fc_train_L.npy')==fit['source_fc_sha256']
    else:
        fit=json.loads((OUT/'RNCut/complete.json').read_text())
        assert fit['state']=='complete' and fit['native_voxel_equivalence_passed']
    frozen=dict(family=family,source_labels=sources,fit=fit,
        evaluation_support_sha256=G.digest(BASE/'evaluation_mask_L.npy'),
        participants=base['protocol']['subjects'],bandpass_Hz=[.01,.1],
        implementation_sha256=G.digest(Path(__file__)),
        n=50,training_people=500,training_test_identity_overlap=0,
        hemisphere='L',K_nominal_bilateral=[100,200,300,400],
        scope='Same left support, filtering, FC profiles and geodesic pairs as the 27-map benchmark; no bilateral coupling is claimed.')
    dest=folder/'protocol.json'
    if dest.exists():assert json.loads(dest.read_text())==frozen,'Use a new run for a changed protocol'
    f_save(dest,frozen);np.savez_compressed(folder/'atlas_labels_full_L.npz',**maps)
    labels={k:v[mask] for k,v in maps.items()}
    baseline=np.load(BASE/'atlas_labels_full_L.npz')['DRFC_100'][mask]
    labels['DRFC_100_check']=baseline
    f_status(family,state='running',phase='common_geodesic_pairs',completed=0,total=50)
    distance=np.load(G.D/'distances_L.npy',mmap_mode='r')[np.ix_(ids,ids)]
    ii,jj,bins,coef=R.coefficients(distance,labels);del distance
    order=np.argsort(ii,kind='stable');ii,jj,bins,coef=ii[order],jj[order],bins[order],coef[:,order];del order
    rows=[]
    for index,old in enumerate(base['participants']):
        dest=folder/'participants'/f'{old["subject"]}.json'
        if dest.exists():rows.append(json.loads(dest.read_text()));continue
        f_status(family,state='running',phase='participant_metrics',completed=index,total=50,subject=old['subject'])
        y,source=R.filtered_left(original_inputs[old['subject']],n,mp['L_columns']);y=y[:,mask];y-=y.mean(0);y/=y.std(0).clip(1e-8)
        y=torch.as_tensor(np.ascontiguousarray(y),device='cuda');gram=y@y.T
        norm=((y*(gram@y)).sum(0).clamp_min(1e-12).sqrt()/len(y))
        rec=dict(subject=old['subject'],source=source,metrics={})
        for name,v in labels.items():rec['metrics'][name]=G.low_rank_metrics(y,v,gram,norm)
        values=np.empty(len(ii),dtype='float32')
        for begin in range(0,len(ids),256):
            end=min(begin+256,len(ids));a,b=np.searchsorted(ii,[begin,end]);block=y[:,begin:end].T@y/len(y)
            values[a:b]=block[torch.as_tensor(ii[a:b]-begin,device='cuda',dtype=torch.long),torch.as_tensor(jj[a:b],device='cuda',dtype=torch.long)].cpu().numpy()
        for name,value in zip(labels,coef@values.astype('float64')):rec['metrics'][name]['DCBC_x100']=float(value*100)
        errors={k:abs(v-old['metrics']['DRFC_100'][k]) for k,v in rec['metrics']['DRFC_100_check'].items()}
        assert max(errors.values())<2e-6,(old['subject'],errors)
        reference=R.binwise_reference(values,bins,baseline[ii]==baseline[jj])
        assert abs(reference-rec['metrics']['DRFC_100_check']['DCBC_x100']/100)<1e-9
        rec['same_input_DRFC_reference_errors']=errors;del rec['metrics']['DRFC_100_check']
        assert all(np.isfinite(list(v.values())).all() for v in rec['metrics'].values())
        f_save(dest,rec);rows.append(rec)
        del y,gram,norm,values,block;torch.cuda.empty_cache()
        f_status(family,state='running',phase='participant_complete',completed=index+1,total=50)
        print(family,index+1,old['subject'],flush=True)
    f_save(folder/'summary.json',dict(state='complete',protocol=frozen,participants=rows))
    f_status(family,state='complete',completed=50,total=50)


def f_merge():
    result=json.loads((BASE/'summary.json').read_text())
    maps=dict(np.load(BASE/'atlas_labels_full_L.npz'))
    by_id={r['subject']:r for r in result['participants']}
    for family in ['NMF','RNCut']:
        more=json.loads((OUT/family/'summary.json').read_text());assert more['state']=='complete'
        assert set(by_id)=={r['subject'] for r in more['participants']}
        for person in more['participants']:by_id[person['subject']]['metrics'].update(person['metrics'])
        maps.update(dict(np.load(OUT/family/'atlas_labels_full_L.npz')))
        for K in [100,200,300,400]:
            result['protocol']['metadata'][f'{family}_{K}']=dict(title=family if family=='NMF' else 'RNCut (surface adaptation)',
                family=family,K=K,K_left=K//2,kind='multiscale',training_people=500)
    result['protocol']['added_experiments']={f:json.loads((OUT/f/'protocol.json').read_text()) for f in ['NMF','RNCut']}
    result['protocol']['statistics']='Paired two-sided Wilcoxon normal approximation; Holm across all 60 CoBRA-versus-five-comparator tests at matched K.'
    comparisons=[]
    for K in [100,200,300,400]:
        for metric in ['homogeneity','DCBC_x100','cosine_silhouette']:
            a=np.array([r['metrics'][f'CoBRA_{K}'][metric] for r in result['participants']])
            for family in ['DRFC','Schaefer','CoBRA_raw','NMF','RNCut']:
                d=a-np.array([r['metrics'][f'{family}_{K}'][metric] for r in result['participants']])
                p=float(wilcoxon(d,alternative='two-sided',zero_method='wilcox',method='approx',correction=False).pvalue) if np.any(d) else 1.
                comparisons.append(dict(K=K,metric=metric,comparator=family,n=50,p_raw=p,
                    mean_paired_difference=float(d.mean()),median_paired_difference=float(np.median(d))))
    order=np.argsort([c['p_raw'] for c in comparisons]);adjusted=np.maximum.accumulate([(len(order)-j)*comparisons[i]['p_raw'] for j,i in enumerate(order)])
    for i,p in zip(order,adjusted):comparisons[i]['p_holm']=min(1.,float(p))
    result['comparisons']=comparisons
    result['protocol']['RNCut']='Validated graph-domain adaptation of the unchanged author RNcut optimizer, with surface mesh neighborhoods replacing voxel neighborhoods; not the original volumetric experiment.'
    f_save(OUT/'figure_data/summary.json',result)
    np.savez_compressed(OUT/'figure_data/atlas_labels_full_L.npz',**maps)
    np.save(OUT/'figure_data/evaluation_mask_L.npy',np.load(BASE/'evaluation_mask_L.npy'))
    f_save(OUT/'figure_data/complete.json',dict(state='complete',participants=50,maps=len(maps),paired_tests=60))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--family',choices=['NMF','RNCut','merge'],required=True);args=ap.parse_args()
    try:
        f_merge() if args.family=='merge' else f_evaluate(args.family)
    except Exception as e:
        f_status(args.family,state='failed',error=repr(e),traceback=traceback.format_exc());raise
