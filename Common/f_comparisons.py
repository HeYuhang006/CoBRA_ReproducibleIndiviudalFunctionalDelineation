"""Fresh-output comparison tables; same subjects/support; no invented missing results."""
from pathlib import Path
import csv,json
import numpy as np
from f_runtime import CODE,f_read,f_save,f_import

def f_write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def f_load_maps(w,K,method,subject,visit,index):
    from scipy.io import loadmat
    N=w/'runs/coarse_network18_20261008_v1';A=w/'runs/connectome_validation_20261007_v1/retest'
    if K==18:
        masks=[np.load(N/f'geometry/evaluation_mask_{h}.npy') for h in ['L','R']]
        if method in ['PCA','SpatialWard']:
            name='PCA64_KMeans' if method=='PCA' else 'PCA64_SpatialWard_areal_control'
            return np.load(N/f'classical18/{visit}/{subject}/{name}.npy')
        if method=='Wang':parts=[np.load(N/f'Wang_author_HFR18/{visit}/{subject}/{h}.npy') for h in ['L','R']]
        elif method=='CoBRA':parts=[np.load(N/f'CoBRA18/{visit}/{subject}/{h}/CoBRA_fs4.npy') for h in ['L','R']]
        else:
            cells=loadmat(w/f'runs/gpip_author_20261009_v2/jobs/K18_{visit}/maps.mat')['labels'].ravel()
            full=cells[index].ravel().astype('int32')-1;parts=[full[:2562],full[2562:]]
        return np.concatenate([p[m] for p,m in zip(parts,masks)])
    if method in ['PCA','SpatialWard']:
        name='PCA64_KMeans' if method=='PCA' else 'PCA64_SpatialWard';v={'Day1':'baseline','Day2':'day2','Retest':'retest'}[visit]
        return np.load(w/f'runs/individual_classical_matched36_3timepoints_20261008_v1/maps/{subject}/L/{v}/{name}_100.npy')
    if method=='Wang':return np.load(w/f'runs/published_baselines_matched36_k100_20261008_v1/Wang_style_adapted_diagnostic/maps/{visit}/{subject}.npy')
    if method=='CoBRA':
        cal={'Day1':'baseline_REST1','Day2':'baseline_REST2','Retest':'retest_REST1'}[visit]
        return np.load(A/f'individual/{subject}/L/50/{cal}/CoBRA.npy')
    full=loadmat(w/f'runs/gpip_author_20261009_v2/jobs/K100_{visit}/maps.mat')['labels'].ravel()[index].ravel().astype('int32')-1
    return full[np.load(w/'runs/pilot32_MSMAll_samplez/mapping.npz')['L_vertices']]

def f_evaluate_individual(w,require_all=True):
    from scipy.sparse import block_diag,load_npz,csr_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.stats import wilcoxon
    SS=f_import('gpip_author_20261009/code/subject_specificity.py')
    ids=[r['subject'] for r in f_read(w/'runs/connectome_validation_20261007_v1/retest/subjects.json')]
    records=[];missing=[];tests=[];N=w/'runs/coarse_network18_20261008_v1';out=w/'reports/individual'
    for K in [18,100]:
        k=18 if K==18 else 50
        try:
            if K==18:
                masks=[np.load(N/f'geometry/evaluation_mask_{h}.npy') for h in ['L','R']]
                adj=block_diag([load_npz(N/f'geometry/adjacency_{h}.npz')[m][:,m] for h,m in zip(['L','R'],masks)],format='csr')
                prior=np.concatenate([np.load(N/f'geometry/author_labels_{h}.npy')[m] for h,m in zip(['L','R'],masks)])
            else:
                e=np.load(w/'runs/pilot32_MSMAll_samplez/edges_L.npy');n=29696
                adj=csr_matrix((np.ones(2*len(e)),(np.r_[e[:,0],e[:,1]],np.r_[e[:,1],e[:,0]])),shape=(n,n));prior=np.load(w/'runs/corrected_personal_multires_20261006_v1/priors/L/50/labels.npy')
        except FileNotFoundError as exc:missing.append(str(exc));continue
        prior_cc=np.array([connected_components(adj[prior==j][:,prior==j],directed=False,return_labels=False) for j in range(k)])
        manifest=dict(subjects=ids,methods={},protocol='Independent Day1/Day2/Retest fits; common support; subject resampling',bandpass_hz=[.01,.1])
        for method in ['PCA','Wang','SpatialWard','GPIP','CoBRA']:
            maps={};paths={v:[] for v in ['Day1','Day2','Retest']}
            try:
                for index,s in enumerate(ids):
                    for v in paths:
                        y=f_load_maps(w,K,method,s,v,index).astype('int32');assert len(y)==adj.shape[0] and y.min()>=0 and y.max()<k
                        p=out/f'maps/K{K}/{method}/{v}/{s}.npy';p.parent.mkdir(parents=True,exist_ok=True);np.save(p,y);paths[v].append(str(p));maps[s,v]=y
            except FileNotFoundError as exc:missing.append(f'K{K}/{method}: {exc}');continue
            manifest['methods'][method]=dict(K=k,maps=paths,allow_missing=True)
            for s in ids:
                for first,last in [('Day1','Day2'),('Day1','Retest')]:
                    for metric,value in SS.f_agreement(maps[s,first],maps[s,last],k).items():
                        records.append(dict(subject=s,K=K,method=method,metric=metric,contrast=first+'_'+last,value=value))
                cc=[]
                for v in paths:
                    y=maps[s,v];component=np.array([connected_components(adj[y==j][:,y==j],directed=False,return_labels=False) for j in range(k)])
                    cc.append(float(np.maximum(component-(prior_cc if K==18 else 1),0).sum()))
                records.append(dict(subject=s,K=K,method=method,metric='Fragments',contrast='Mean_three_visits',value=float(np.mean(cc))))
        if len(manifest['methods'])==5:
            path=out/f'specificity_K{K}/manifest.json';f_save(path,manifest);SS.f_run(path,path.parent)
    for K in [18,100]:
        for metric in ['ARI','Parcel_Dice','Fragments']:
            for contrast in (['Mean_three_visits'] if metric=='Fragments' else ['Day1_Day2','Day1_Retest']):
                target={r['subject']:r['value'] for r in records if r['K']==K and r['metric']==metric and r['contrast']==contrast and r['method']=='CoBRA'}
                for method in ['PCA','Wang','SpatialWard','GPIP']:
                    comp={r['subject']:r['value'] for r in records if r['K']==K and r['metric']==metric and r['contrast']==contrast and r['method']==method}
                    common=sorted(target.keys()&comp.keys())
                    if len(common)!=len(ids):continue
                    d=np.array([target[s]-comp[s] for s in common]);p=float(wilcoxon(d).pvalue) if np.any(d) else 1.
                    tests.append(dict(K=K,metric=metric,contrast=contrast,comparator=method,n=len(common),mean_paired_difference=float(d.mean()),p_raw=p))
    order=np.argsort([r['p_raw'] for r in tests]);maximum=0
    for j,i in enumerate(order):maximum=max(maximum,min(1,(len(tests)-j)*tests[i]['p_raw']));tests[i]['p_holm']=maximum
    f_write_csv(out/'subject_metrics.csv',records);f_write_csv(out/'paired_tests.csv',tests)
    f_save(out/'completion.json',dict(complete=not missing,missing=missing,n=36,
        topology18='Excess components over distributed author18 prior; Ward9+9 is areal control, not semantic networks.',
        Wang18='Later HFR18 release with author-arithmetic equivalence check; not bitwise original2015.',
        Wang100='Adapted diagnostic, not original Wang2015.',GPIP='Unmodified private author core, external runtime adapter, fixed20 iterations; inspect outer-loop change before convergence claims.'))
    if missing and require_all:raise RuntimeError('Comparisons incomplete; see '+str(out/'completion.json'))

def f_evaluate_group(c,w):
    import torch
    from scipy.stats import wilcoxon
    R=f_import('final/population_training/run.py');D=f_import('final/population_training/drfc.py')
    mapping=np.load(w/'runs/pilot32_MSMAll_samplez/mapping.npz');vertices=mapping['L_vertices']
    models={};globalK={}
    for K in c['K_bilateral']:
        k=K//2
        models[f'CoBRA_{K}']=np.load(w/f'runs/corrected_personal_multires_20261006_v1/priors/L/{k}/labels.npy')
        models[f'CoBRA_raw_{K}']=np.load(w/f'runs/vae450_spatialweight_20261006_v1/labels_LPE1_L_{k}.npy')
        models[f'DRFC_{K}']=np.load(w/f'runs/full500_MSMAll_samplez/labels_L_{k}.npy')
        image=__import__('nibabel').load(CODE/f'resources/atlases/Schaefer2018_{K}Parcels_7Networks_order.dlabel.nii')
        models[f'Schaefer_{K}']=np.asarray(image.dataobj).ravel()[mapping['L_columns']]-1
        for prefix in ['CoBRA','CoBRA_raw','DRFC','Schaefer']:globalK[f'{prefix}_{K}']=K
    # Additional atlas resolutions are independent points, never interpolated.
    nib=__import__('nibabel')
    for p in (CODE/'resources/atlases/dcbc_collection_20261008/parcellations').glob('*.L.label.gii'):
        y=np.asarray(nib.load(p).agg_data()).ravel()[vertices];name=p.name.replace('.32k.L.label.gii','');models[name]=np.where(y>0,y-1,-1)
        right=p.with_name(p.name.replace('.L.label.gii','.R.label.gii'));both=np.r_[y,np.asarray(nib.load(right).agg_data()).ravel()] if right.exists() else y
        globalK[name]=int(len(np.unique(both[both>0])))
    mask=np.logical_and.reduce([v>=0 for v in models.values()]);assert mask.any()
    labels={name:np.unique(y[mask],return_inverse=True)[1]+1 for name,y in models.items()}
    rows=f_read(w/'manifests/splits.json')['test'][:50];train={r['subject'] for r in f_read(w/'runs/modulated_vae_full_crossday_v1/subjects.json')['train']}
    assert not train&{r['subject'] for r in rows}
    distances=np.load(w/'runs/full500_MSMAll_samplez/distances_L.npy',mmap_mode='r')[np.ix_(mask,mask)]
    out=w/'reports/group';out.mkdir(parents=True,exist_ok=True);records=[]
    for row in rows:
        p=out/'subjects'/f"{row['subject']}.json"
        if p.exists():records.extend(f_read(p));continue
        y,provenance=R.f_filtered_timeseries(row,R.f_get_mapping(row['path']),dict(use_verified_practice_cache=False))
        y=y[:,:len(vertices)][:,mask];fc=y.T@y/len(y);rec=[]
        for name,v in labels.items():
            values=dict(homogeneity=float(D.f_functional_homogeneity(v,fc)),
                        FC_profile_silhouette=float(D.f_exact_cosine_silhouette(fc,v,device='cuda')),
                        DCBC_x100=float(D.f_dcbc_from_standardized_fc(fc,distances,v)['dcbc']*100))
            for metric,value in values.items():rec.append(dict(subject=row['subject'],method=name,K_bilateral=globalK[name],metric=metric,value=value))
        f_save(p,rec);records.extend(rec);del fc,y;torch.cuda.empty_cache()
    f_write_csv(out/'subject_metrics.csv',records)
    tests=[]
    for K in c['K_bilateral']:
        for metric in ['homogeneity','FC_profile_silhouette','DCBC_x100']:
            target={r['subject']:r['value'] for r in records if r['method']==f'CoBRA_{K}' and r['metric']==metric}
            for family in ['DRFC','Schaefer','CoBRA_raw']:
                comp={r['subject']:r['value'] for r in records if r['method']==f'{family}_{K}' and r['metric']==metric}
                delta=np.array([target[s]-comp[s] for s in sorted(target)])
                tests.append(dict(K=K,metric=metric,comparator=family,n=len(delta),mean_paired_difference=float(delta.mean()),p_raw=float(wilcoxon(delta).pvalue) if np.any(delta) else 1.))
    maximum=0
    for j,i in enumerate(np.argsort([r['p_raw'] for r in tests])):maximum=max(maximum,min(1,(len(tests)-j)*tests[i]['p_raw']));tests[i]['p_holm']=maximum
    f_write_csv(out/'paired_tests.csv',tests)
    f_save(out/'protocol.json',dict(subjects=[r['subject'] for r in rows],hemisphere='L',common_grayordinates=int(mask.sum()),
                                   atlases=list(labels),DRFC_training=500,CoBRA_training=450,
                                   silhouette='exact cosine of full FC profiles; separate from adjacent BOLD silhouette'))
