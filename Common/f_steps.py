"""Manually invoked scientific stages. No background launch or historical model reuse."""
from pathlib import Path
from types import SimpleNamespace
import copy, importlib.metadata, json, os, shutil, subprocess, sys
from f_runtime import CODE,f_read,f_save,f_sha,f_import,f_link

GEOMETRY='runs/pilot32_MSMAll_samplez'
POP='runs/fullfc450_crossday_v1'
CACHE='runs/modulated_vae_full_crossday_v1'
PERSONAL='runs/corrected_personal_multires_20261006_v1'
APP='runs/connectome_validation_20261007_v1'
GPIP='runs/gpip_author_20261009_v2'

def f_plan(method,stage,a,c,w):
    names=['Environment and numerical checks','Raw data, frozen identities, geometry and QC','Population/shared-network training',
           'Group atlas and neural prior','Visit-specific individual adaptation and connected inference',
           'Independent quality, reproducibility and inter/intra validation','Task / frozen-ROI connectome applications','Figures and editable PPT']
    return dict(method=method,step=stage,operation=names[stage],mode=a.mode,cohort=a.cohort,K=a.K,visit=a.visit,
                code_root=str(CODE),output_root=str(w),raw_inputs=[c['raw_YA'],c['raw_Retest']],
                no_historical_results=True,bandpass_hz=c['bandpass_hz'],
                prerequisites={0:[],1:['step0'],2:['Main step1'],3:['step2'],4:['Main step3','step1 matched inputs'],
                               5:['step4 complete maps'],6:['step4 complete maps','step5 QC'],7:['step5/6 finished statistics']}[stage],
                note='The README lists cross-method dependencies and optional modes. Dry-run does not allocate GPU or create files.')

def f_selfcheck():
    report={}
    for package in ['numpy','scipy','nibabel','torch','scikit-learn','numba','matplotlib','threadpoolctl']:
        report[package]=importlib.metadata.version(package)
    X=f_import('validation_recovery_20261008/code/validation_exact_runtime.py')
    report['full_FC_loss_projection_gradients']=X.f_validate_small()
    for original in ['final/test_pipeline.py','final/test_application.py']:
        m=f_import(original)
        for name in dir(m):
            if name.startswith('f_test_'):
                getattr(m,name)();report[name]='passed'
    import torch
    report['CUDA_available']=torch.cuda.is_available()
    print(json.dumps(report,indent=2))
    return report

def f_remap_raw(row,c):
    def visit(x):
        if isinstance(x,list):return [visit(v) for v in x]
        if isinstance(x,dict):return {k:visit(v) for k,v in x.items()}
        if isinstance(x,str):
            x=x.replace('/dataspace/yuhang_data/HCP-YA',c['raw_YA'])
            x=x.replace('/dataspace/yuhang_data/HCP-Retest',c['raw_Retest'])
        return x
    return visit(row)

def f_seed_metadata(c,w):
    pop=f_remap_raw(f_read(CODE/'configs/cohorts/population.json'),c)
    splits=f_remap_raw(f_read(CODE/'configs/cohorts/splits.json'),c)
    dev=f_remap_raw(f_read(CODE/'configs/cohorts/development.json'),c)
    train={r['subject'] for r in pop['train']}
    assert len(train)==450 and not train&set(pop['reserved_retest'])
    for rows in [pop['train'],pop['validation']]:
        for row in rows:
            for p in row['paths']:
                if not Path(p).is_file():raise FileNotFoundError(p)
    f_save(w/'manifests/splits.json',splits)
    f_save(w/CACHE/'subjects.json',pop)
    f_save(w/PERSONAL/'subjects.json',dev)
    for name in ['cache','population','population_distance','DRFC']:
        cfg=f_read(CODE/f'configs/{name}.json');cfg['gpu']=str(c['gpu']);cfg['use_verified_practice_cache']=False
        f_save(w/f'configs/{name}.json',cfg)
    # Structural surfaces and task maps are data inputs. Models, FC, labels,
    # FD arrays and area arrays are NEVER linked from historical runs.
    resources=Path(c['ancillary_input_root'])
    for p in (resources/'subject_surfaces').glob('*.surf.gii'):
        f_link(p,w/'resources/subject_surfaces'/p.name)
    if Path(c['task_input_root']).exists():
        for p in Path(c['task_input_root']).rglob('*'):
            if p.is_file() and p.suffix in ['.nii','.txt','.fsf']:
                f_link(p,w/'resources/task_individual'/p.relative_to(c['task_input_root']))
    return pop,splits

def f_prepare_geometry(c,w,pop,splits):
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans
    R=f_import('final/population_training/run.py')
    folder=w/GEOMETRY;folder.mkdir(parents=True,exist_ok=True)
    rows=splits['train'][:32];mp=R.f_get_mapping(rows[0]['path'])
    np.savez(folder/'mapping.npz',**{h+'_'+k:v for h,d in mp.items() for k,v in d.items()})
    for h in ['L','R']:
        for row in rows:
            name=f"{row['subject']}.{h}.midthickness_MSMAll.32k_fs_LR.surf.gii"
            if not (w/'resources/subject_surfaces'/name).exists():raise FileNotFoundError(name)
        R.f_distance_matrix(rows,h,mp,folder,dict(seed=c['seed'],surface_registration='MSMAll'),True)
        # Exact archived constructor: LPE MiniBatchKMeans(128), n_init3,
        # batch4096, seed20260930. No pilot-derived patch array is loaded.
        patches=MiniBatchKMeans(128,random_state=c['seed'],batch_size=4096,n_init=3).fit_predict(np.load(folder/f'lpe_{h}.npy')).astype('int64')
        assert len(np.unique(patches))==128
        p=w/CACHE/f'anchor_patches_{h}.npy';p.parent.mkdir(parents=True,exist_ok=True);np.save(p,patches)
    f_save(folder/'geometry.json',dict(mode='subject_mean',subjects=[r['subject'] for r in rows],registration='MSMAll',
                                       geometry_only=True,no_old_arrays=True))

def f_prepare_validation(c,w):
    """Freeze the same identities; recompute motion QC and vertex areas from raw ancillary inputs."""
    import numpy as np,nibabel as nib
    D=f_import('final/application_data.py');M=f_import('final/data.py')
    old=f_remap_raw(f_read(CODE/'configs/cohorts/validation.json'),c);mp=np.load(w/GEOMETRY/'mapping.npz')
    result={k:copy.deepcopy(v) for k,v in old.items() if k not in ['YA400','retest','excluded','provenance_sha256']}
    result.update(YA400=[],retest=[],excluded=[])
    for cohort in ['YA400','retest']:
        for oldrow in old[cohort]:
            row=dict(subject=oldrow['subject'],visits=oldrow['visits'],ancillary={},area={},QC={})
            for visit,entry in row['visits'].items():
                row['ancillary'][visit]={};row['QC'][visit]={}
                for run,path in entry['runs'].items():
                    if not Path(path).is_file():raise FileNotFoundError(path)
                    raw=Path(path).parent/'Movement_Regressors.txt'
                    if not raw.exists():raw=Path(oldrow['ancillary'][visit][run]['path'])
                    if not raw.is_file():raise FileNotFoundError(raw)
                    dest=w/'resources/connectome_ancillary'/('HCP_1200' if visit=='baseline' else 'HCP_Retest')/row['subject']/run/'Movement_Regressors.txt'
                    f_link(raw,dest);fd=D.f_fd(np.loadtxt(raw));np.save(dest.with_name('FD.npy'),fd)
                    row['ancillary'][visit][run]=dict(path=str(dest),sha256=f_sha(raw))
                    row['QC'][visit][run]={str(stop):D.f_qc_segment(fd,0,stop) for stop in [240,480,960,1200]}
                    row['QC'][visit][run]['tail480']=D.f_qc_segment(fd,720,1200)
                    if not row['QC'][visit][run]['480']['passed']:raise RuntimeError('Frozen-cohort QC changed: '+row['subject']+'/'+visit+'/'+run)
            for h in ['L','R']:
                surf=Path(oldrow['area'][h]['surface'])
                if not surf.exists():raise FileNotFoundError(surf)
                img=nib.load(surf);xyz=img.agg_data('pointset');tri=img.agg_data('triangle')
                area=np.zeros(len(xyz),dtype='float64');mass=np.linalg.norm(np.cross(xyz[tri[:,1]]-xyz[tri[:,0]],xyz[tri[:,2]]-xyz[tri[:,0]]),axis=1)/6
                for j in range(3):np.add.at(area,tri[:,j],mass)
                area=area[mp[h+'_vertices']];assert (area>0).all()
                p=w/'resources/vertex_areas'/f"{row['subject']}_{h}.npy";p.parent.mkdir(parents=True,exist_ok=True);np.save(p,area)
                row['area'][h]=dict(path=str(p),surface=str(surf),surface_sha256=f_sha(surf),frozen_visit='baseline')
            row.update(passed=True,exclusions=[]);f_save(w/APP/'QC'/f"{row['subject']}.json",row);result[cohort].append(row)
        f_save(w/APP/cohort/'subjects.json',result[cohort])
    f_save(w/APP/'cohorts.json',result)
    f_save(w/APP/'prepare_status.json',dict(state='complete',YA_selected=400,retest_selected=36,recomputed_QC=True))

def f_population(c,w,distance=False):
    import torch
    Cache=f_import('final/population_training/train_modulated_vae.py')
    x=Cache.Experiment(w/'configs/cache.json');del x
    if distance:
        if not all((w/POP/f'group_reference_{h}.done.json').exists() for h in c['hemispheres']):raise RuntimeError('Finish raw population step2 before distance ablation.')
        f_import('final/population_training/prepare_fullfc_distance.py').f_prepare(w/'configs/population_distance.json')
        module=f_import('final/population_training/train_distance_fullfc.py')
        cls=module.DistanceExperiment;path=w/'configs/population_distance.json'
    else:cls=f_import('final/population_training/train_fullfc_vae.py').FullExperiment;path=w/'configs/population.json'
    exp=cls(path);exp.prepare_reference()
    for h in c['hemispheres']:exp.fit_group(h)
    for h in c['hemispheres']:exp.fit_individual(h)
    exp.export();del exp;torch.cuda.empty_cache()

def f_group(c,w,prior_only=False):
    import numpy as np,torch
    from sklearn.cluster import KMeans
    if not all((w/POP/h/'completed.json').exists() for h in c['hemispheres']):raise RuntimeError('Complete population step2 on both hemispheres before step3.')
    if not all((w/f'resources/fullfc_distance/fullfc450_distance_crossday_v1/normalization_{h}.done.json').exists() for h in c['hemispheres']):
        f_import('final/population_training/prepare_fullfc_distance.py').f_prepare(w/'configs/population_distance.json')
    if not prior_only:
        E=f_import('final/population_training/export_group.py')
        for h in c['hemispheres']:
            mu,var,meta=E.f_embedding(w,w/POP,h,'individual_checkpoint.pt')
            p=w/'runs/vae450_multires_20261006_v1/population_epoch50';p.mkdir(parents=True,exist_ok=True)
            np.save(p/f'latent_mean_{h}.npy',mu);np.save(p/f'latent_variance_{h}.npy',var);f_save(p/f'encoding_{h}.json',meta)
            features=np.c_[mu,np.load(w/GEOMETRY/f'lpe_{h}.npy')]
            for k in [v//2 for v in c['K_bilateral']]:
                labels=KMeans(k,n_init=10,max_iter=300,random_state=c['seed']).fit_predict(features)
                p=w/f'runs/vae450_spatialweight_20261006_v1/labels_LPE1_{h}_{k}.npy';p.parent.mkdir(parents=True,exist_ok=True);np.save(p,labels.astype('int16'))
        f_import('final/priors.py').f_prepare(w,w/PERSONAL)
    P=f_import('final/pipeline.py')
    for h in c['hemispheres']:
        exp=P.f_geometry(w,h,w/PERSONAL,'raw');model,path=P.f_group_model(w,w/PERSONAL,h,exp,'raw');del exp,model;torch.cuda.empty_cache()

def f_models(w,out):
    X=f_import('validation_recovery_20261008/code/validation_exact_runtime.py')
    A=f_import('final/application_runner.py')
    A.f_fit=X.f_fit;A.f_calibration_decode=X.f_calibration_decode
    return A,X.f_load_models(w,w/PERSONAL,out)

def f_cohorts(a):return ['YA400','retest'] if a.cohort=='all' else [a.cohort]

def f_adapt(a,c,w):
    import numpy as np,torch
    mp=np.load(w/GEOMETRY/'mapping.npz')
    for cohort in f_cohorts(a):
        out=w/APP/cohort;rows=f_read(out/'subjects.json');A,models=f_models(w,out)
        for row in rows:
            calibrations=['baseline_REST1','baseline_REST2']+(['retest_REST1'] if cohort=='retest' else [])
            for k in [v//2 for v in c['K_bilateral']]:
                for cal in calibrations:
                    for h in c['hemispheres']:A.f_adapt(w,w/PERSONAL,out,row,h,k,cal,models,mp,c['personal_steps'])
        del models;torch.cuda.empty_cache()

def f_individual_validation(a,c,w,application=False):
    import numpy as np,torch
    mp=np.load(w/GEOMETRY/'mapping.npz')
    for cohort in f_cohorts(a):
        out=w/APP/cohort;rows=f_read(out/'subjects.json');A,models=f_models(w,out)
        pairs={h:A.f_pair_geometry(w,h,w/APP) for h in c['hemispheres']}
        for row in rows:
            # Secondary map metrics also require primary connectomes on Retest.
            for k in [v//2 for v in c['K_bilateral']]:A.f_measure(w,w/PERSONAL,out,row,k,mp,cohort)
            A.f_secondary(w,w/PERSONAL,out,row,mp,models,cohort)
            if not application:A.f_heldout_quality(w,w/PERSONAL,out,row,mp,models,pairs,cohort)
        if application:
            A.f_wrong_person(w,w/PERSONAL,out,rows,mp,cohort);A.f_statistics(out,rows)
        del models;torch.cuda.empty_cache()

def f_controlled(a,c,w):
    """Duration-matched 960-frame single-day vs cross-day, independent filtering."""
    C=f_import('paper_figure_campaign/controlled_adaptation.py')
    X=f_import('validation_recovery_20261008/code/validation_exact_runtime.py')
    C.f_load_models=X.f_load_models;C.f_fit=X.f_fit;C.f_infer=X.f_infer
    # Main comparison uses parcels100–400. Optional Yeo17 requires its own prior step.
    C.RESOLUTIONS=[(f'parcel{K}',dict(L=K//2,R=K//2)) for K in c['K_bilateral']]
    f_save(w/'runs/paper_figure_campaign_20261007_v1/low_resolution_status.json',dict(state='complete',scope='parcels100_400_only'))
    for cohort in f_cohorts(a):C.f_run(SimpleNamespace(cohort=cohort,gpu=int(c['gpu'])))

def f_baseline_classical(a,c,w,measure=False):
    import numpy as np
    B=f_import('paper_figure_campaign/individual_baselines_3timepoints.py');rows=f_read(w/APP/'retest/subjects.json');mp=np.load(w/GEOMETRY/'mapping.npz')
    for row in rows:
        if measure:B.f_map_agreement(row,B.ROOT);B.f_measure(row,mp,B.ROOT);continue
        for h in c['hemispheres']:
            edges=np.load(w/GEOMETRY/f'edges_{h}.npy')
            for visit,actual,day in [('baseline','baseline',1),('day2','baseline',2),('retest','retest',1)]:
                ys=[B.f_signals(w,w/APP/'retest',row,actual,f'rfMRI_REST{day}_{pe}',h,mp) for pe in ['LR','RL']]
                B.f_fit_maps(np.concatenate(ys),edges,row['subject'],h,visit,B.ROOT)

def f_network18_classical(w):
    import numpy as np
    from scipy.sparse import block_diag,load_npz
    from sklearn.utils.extmath import randomized_svd
    from sklearn.cluster import KMeans,ward_tree
    N=f_import('coarse_network18_20261008/code/coarse_network18.py');B=f_import('paper_figure_campaign/individual_baselines_3timepoints.py')
    masks=[np.load(N.OUT/f'geometry/evaluation_mask_{h}.npy') for h in ['L','R']]
    adj=block_diag([load_npz(N.OUT/f'geometry/adjacency_{h}.npz')[m][:,m] for h,m in zip(['L','R'],masks)],format='csr')
    for row in f_read(w/APP/'retest/subjects.json'):
        for visit,_,_ in N.VISITS:
            dest=N.OUT/f'classical18/{visit}/{row["subject"]}';dest.mkdir(parents=True,exist_ok=True)
            y=np.vstack([np.load(N.OUT/f'inputs/{visit}/{row["subject"]}/{h}.npy')[m] for h,m in zip(['L','R'],masks)])
            U,S,_=randomized_svd(y,n_components=64,n_iter=3,n_oversamples=16,random_state=20261008);features=U*S
            labels=KMeans(n_clusters=18,n_init=10,max_iter=300,random_state=20261008).fit_predict(features)
            np.save(dest/'PCA64_KMeans.npy',labels.astype('int32'));ward=[];offset=0
            for h,m in zip(['L','R'],masks):
                n=int(m.sum());children,*_=ward_tree(features[offset:offset+n],connectivity=adj[offset:offset+n,offset:offset+n],return_distance=False)
                ward.append(B.f_cut_tree(children,n,9)+(0 if h=='L' else 9));offset+=n
            np.save(dest/'PCA64_SpatialWard_areal_control.npy',np.concatenate(ward).astype('int32'))
            f_save(dest/'complete.json',dict(K=18,scope='distributed KMeans18; Ward9+9 is areal control'))

def f_gpip(a,c,w,stage):
    G=f_import('gpip_author_20261009/code/workflow.py');out=w/GPIP;out.mkdir(parents=True,exist_ok=True)
    ids=[r['subject'] for r in f_read(w/APP/'retest/subjects.json')]
    if stage==2:
        code=w/'gpip_author_20261009/code';expr=f"addpath('{code}');f_validate_author_runtime('{w}/gpip_author_20261009/reference/gpip_share','{w}/gpip_author_20261009/compat','{out}')"
        subprocess.run(['/usr/bin/matlab','-singleCompThread','-batch',expr],check=True);G.f_source_check();return
    code=G.f_source_check()
    if stage in [1,4]:
        visits=G.VISITS if a.visit=='all' else [a.visit]
        for visit in visits:
            (G.f_prepare if stage==1 else G.f_run_job)(a.K,visit,ids,code)
    elif stage==5:
        # Independent benchmark evaluator supplied by common validation module.
        from f_comparisons import f_evaluate_individual
        f_evaluate_individual(w,require_all=True)

def f_drfc(c,w,stage):
    R=f_import('final/population_training/run.py');cfg=f_read(w/'configs/DRFC.json');folder=w/'runs/full500_MSMAll_samplez';folder.mkdir(parents=True,exist_ok=True)
    if stage==1:R.f_prepare(cfg,folder)
    elif stage==2:
        for h in c['hemispheres']:R.f_train(cfg,folder,h)
    elif stage==3:
        for h in c['hemispheres']:R.f_cluster(cfg,folder,h)
    elif stage==5:
        for h in c['hemispheres']:R.f_evaluate(cfg,folder,h)

def f_execute(method,stage,a,c,w):
    if stage==0:f_selfcheck();return
    if method=='Main':
        if stage==1:
            pop,splits=f_seed_metadata(c,w);f_prepare_geometry(c,w,pop,splits);f_prepare_validation(c,w)
        elif stage==2:f_population(c,w,a.mode=='distance')
        elif stage==3:f_group(c,w,a.mode=='personal_prior')
        elif stage==4:
            if a.mode=='controlled':f_controlled(a,c,w)
            elif a.mode=='network18':
                N=f_import('coarse_network18_20261008/code/cobra_network18.py');N.f_main(False)
            else:f_adapt(a,c,w)
        elif stage==5:
            if a.mode=='group':
                from f_comparisons import f_evaluate_group
                f_evaluate_group(c,w)
            else:f_individual_validation(a,c,w)
        elif stage==6:
            if a.mode=='task':
                f_import('task_validation/download.py').f_main();f_import('task_validation/evaluate.py').f_main()
            else:f_individual_validation(a,c,w,True)
        elif stage==7:
            from f_figures import f_figures
            f_figures(c,w)
    elif method=='DRFC':
        if stage in [1,2,3,5]:f_drfc(c,w,stage)
        elif stage in [4,6]:raise ValueError('DRFC is a group method: no individual training stage. See method README.')
        else:
            from f_figures import f_figures
            f_figures(c,w)
    elif method=='Wang_style':
        N=f_import('coarse_network18_20261008/code/coarse_network18.py')
        if stage==1:
            if a.K==18:N.f_protocol();N.f_prepare_inputs()
            else:f_import('paper_figure_campaign/published_baselines_matched36.py').f_prepare()
        elif stage==2:print('Wang has no population network to train; step2 intentionally not applicable.')
        elif stage==3:
            N.f_author_equivalence_check();f_save(N.OUT/'validation/smoke_complete.json',dict(passed=True,author_label_equivalence=True))
        elif stage==4:
            if a.K==18:N.f_run_wang()
            else:f_import('paper_figure_campaign/published_baselines_matched36.py').f_wang()
        elif stage==5:
            from f_comparisons import f_evaluate_individual
            f_evaluate_individual(w,False)
        elif stage==6:raise ValueError('Wang100 is an adapted diagnostic; no validated bilateral frozen-ROI application is claimed.')
        else:
            from f_figures import f_figures
            f_figures(c,w)
    elif method=='TraditionalMethod':
        if a.mode=='group_nmf':
            if stage!=2:raise ValueError('Use TraditionalMethod step2 --mode group_nmf after DRFC step1.')
            import torch
            from f_group_nmf import f_fit_nmf
            if not torch.cuda.is_available():raise RuntimeError('The full-FC NMF production path requires CUDA; CPU mmap input must not be modified.')
            f_fit_nmf('cuda');return
        if stage==1:print('Shared data from Main step1; network18 native inputs from Wang_style step1 --K18.')
        elif stage in [2,3]:print('No population neural training/group prior required by PCA/Ward; intentionally not applicable.')
        elif stage==4:f_network18_classical(w) if a.K==18 else f_baseline_classical(a,c,w)
        elif stage in [5,6]:
            if a.K==100:f_baseline_classical(a,c,w,True)
            from f_comparisons import f_evaluate_individual
            f_evaluate_individual(w,False)
        else:
            from f_figures import f_figures
            f_figures(c,w)
    elif method=='GPIP':
        if stage in [1,2,4,5]:f_gpip(a,c,w,stage)
        elif stage==3:print('GPIP uses the shared group initial atlas prepared at Main step3; no neural group training.')
        elif stage==6:raise ValueError('Author GPIP task/connectome benchmark was not in the frozen validated protocol. Do not invent published-method application results.')
        else:
            from f_figures import f_figures
            f_figures(c,w)
