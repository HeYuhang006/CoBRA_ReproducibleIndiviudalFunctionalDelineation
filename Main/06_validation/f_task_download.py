from f_runtime import f_work_root
"""Download a prespecified independent HCP task-validation subset on 4090.

Only analyzed MSMAll level-2 z-statistic maps, metadata, and small task
motion files are downloaded. No task information enters atlas adaptation.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse, hashlib, json, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import nibabel as nib
W = f_work_root()
OUT = W / 'runs/task_validation_20261007_v1'
DATA = W / 'resources/task_individual'
TASKS = {'EMOTION': ['FACES-SHAPES'], 'GAMBLING': ['PUNISH-REWARD'], 'LANGUAGE': ['MATH-STORY'], 'RELATIONAL': ['MATCH-REL'], 'SOCIAL': ['RANDOM-TOM'], 'WM': ['0BK-2BK', 'FACE-AVG', 'PLACE-AVG', 'TOOL-AVG', 'BODY-AVG'], 'MOTOR': ['T-AVG', 'LH-AVG', 'RH-AVG', 'LF-AVG', 'RF-AVG']}

def f_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def f_save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    temp.replace(path)

def f_download_one(client, row, task, mapping):
    from botocore.exceptions import ClientError
    s = row['subject']
    dest = DATA / s / task / 's4'
    record_path = OUT / 'downloads' / s / (task + '.json')
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record.get('state') == 'complete' and all((Path(f['path']).exists() and f_sha(Path(f['path'])) == f['sha256'] for f in record['files'])):
            return record
        if record.get('state') == 'unavailable':
            return record
    record = dict(subject=s, task=task, state='downloading', files=[], updated_unix=time.time())
    base = f'HCP_1200/{s}/MNINonLinear/Results/tfMRI_{task}/tfMRI_{task}_hp200_s4_level2_MSMAll.feat/'
    name = f'{s}_tfMRI_{task}_level2_hp200_s4_MSMAll.dscalar.nii'

    def fetch(key, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            record['files'].append(dict(key=key,path=str(path),bytes=path.stat().st_size,sha256=f_sha(path)))
            return path
        response = client.get_object(Bucket='hcp-openaccess', Key=key)
        payload = response['Body'].read()
        assert len(payload) == response['ContentLength'], 'Incomplete S3 response'
        temp = path.with_name(path.name + '.part')
        temp.write_bytes(payload)
        temp.replace(path)
        record['files'].append(dict(key=key, path=str(path), bytes=len(payload), sha256=f_sha(path)))
        return path
    try:
        map_path = fetch(base + name, dest / name)
        txt = fetch(base + 'Contrasts.txt', dest / 'Contrasts.txt')
        fetch(base + 'design.fsf', dest / 'design.fsf')
        img = nib.load(map_path)
        axis = img.header.get_axis(1)
        assert img.shape[1] == 91282
        values = np.asarray(img.dataobj)
        assert np.isfinite(values).all(), 'Nonfinite activation map'
        names = [v.strip() for v in txt.read_text().splitlines() if v.strip()]
        assert len(names) == img.shape[0]
        expected = [f'{s}_tfMRI_{task}_level2_{v}_hp200_s4_MSMAll' for v in names]
        assert list(img.header.get_axis(0).name) == expected, 'Contrast header/order mismatch'
        for h, structure in [('L', 'CIFTI_STRUCTURE_CORTEX_LEFT'), ('R', 'CIFTI_STRUCTURE_CORTEX_RIGHT')]:
            cols = np.flatnonzero(axis.surface_mask & (axis.name == structure))
            assert np.array_equal(cols, mapping[h + '_columns'])
            assert np.array_equal(axis.vertex[cols], mapping[h + '_vertices'])
        missing = [v for v in TASKS[task] if v not in names]
        assert not missing, f'Missing prespecified contrasts: {missing}'
        record.update(map_path=str(map_path), shape=list(img.shape), all_contrasts=names, selected_contrasts={v: names.index(v) for v in TASKS[task]}, cortex_mapping_verified=True)
        qc = {}
        for pe in ['LR', 'RL']:
            key = f'HCP_1200/{s}/MNINonLinear/Results/tfMRI_{task}_{pe}/Movement_Regressors.txt'
            p = fetch(key, dest / f'Movement_Regressors_{pe}.txt')
            motion = np.loadtxt(p)
            assert motion.ndim == 2 and motion.shape[1] == 12 and np.isfinite(motion).all()
            delta = np.diff(motion[:, :6], axis=0, prepend=motion[0:1, :6])
            fd = np.abs(delta[:, :3]).sum(1) + 50 * np.abs(np.deg2rad(delta[:, 3:6])).sum(1)
            qc[pe] = dict(frames=len(fd), mean_FD_mm=float(fd.mean()), fraction_FD_above_05=float((fd > 0.5).mean()), passed=bool(fd.mean() <= 0.3 and (fd > 0.5).mean() <= 0.2))
        record.update(state='complete', task_motion_QC=qc, task_QC_passed=all((v['passed'] for v in qc.values())))
    except ClientError as e:
        code = e.response.get('Error', {}).get('Code', '')
        record.update(state='unavailable' if code in ['NoSuchKey', '404', 'NotFound'] else 'error', error=str(e)[:500])
    except Exception as e:
        record.update(state='error', error=f'{type(e).__name__}: {e}')
    record['updated_unix'] = time.time()
    f_save(record_path, record)
    return record

def f_main():
    import logging
    logging.getLogger('nibabel.global').setLevel(logging.ERROR)
    sys.path.insert(0, str(W / 'deps'))
    import boto3
    from botocore.config import Config
    OUT.mkdir(parents=True, exist_ok=True)
    cohort_path = W / 'runs/connectome_validation_20261007_v1/cohorts.json'
    cohorts = json.loads(cohort_path.read_text())
    rows = cohorts['YA400'][:40]
    protocol = dict(version='task_validation_20261007_v1', subjects=[r['subject'] for r in rows], subset='first40 in already locked YA400 random/QC order; no selection on activation or atlas quality', cohort_sha256=f_sha(cohort_path), tasks=TASKS, smoothing_mm=4, registration='MSMAll', data='HCP individual level2 analyzed z-statistic maps combining task LR/RL', fit='rest-only baseline REST1 LR/RL; task never used for VAE, labels, or hyperparameters', primary_K_bilateral=200, secondary_K_bilateral=[100, 300, 400], methods=['connected_group', 'CoBRA', 'no_hard_connectivity'], primary='area-weighted within-parcel task variance / global area-weighted task variance; lower better', faithful_GPIP='unweighted mean of parcel population variances in individual task z-score maps', stats='participant-paired two-sided sign-flip permutation5000, bootstrap5000, Holm7domains per comparator at K200; secondary K exploratory', missing='report per-domain availability/QC and paired N; no replacement subjects, no atlas-dependent exclusions', task_QC=dict(mean_FD_max_mm=0.3, max_fraction_FD_above_05=0.2, radius_mm=50, rotation_units='degrees'), smoothing_sensitivity='not downloaded in this scoped batch; 2mm is a proposed subsequent analysis', family_metadata_available=False, publication='Chong et al. NeuroImage 2017 doi:10.1016/j.neuroimage.2017.04.054')
    locked = OUT / 'locked_protocol.json'
    if locked.exists():
        assert json.loads(locked.read_text()) == protocol, 'Protocol drift; use a new run'
    else:
        f_save(locked, protocol)
    client = boto3.client('s3', config=Config(connect_timeout=10, read_timeout=60, max_pool_connections=8, retries={'max_attempts': 3}))
    mapping = np.load(W / 'runs/pilot32_MSMAll_samplez/mapping.npz')
    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = [pool.submit(f_download_one, client, row, task, mapping) for row in rows for task in TASKS]
        for f in as_completed(jobs):
            r = f.result()
            results.append(r)
            state = dict(pid=os.getpid(), state='downloading', finished=len(results), total=len(jobs), complete=sum((v['state'] == 'complete' for v in results)), unavailable=sum((v['state'] == 'unavailable' for v in results)), errors=sum((v['state'] == 'error' for v in results)), task_QC_passed=sum((v.get('task_QC_passed', False) for v in results)), downloaded_bytes=sum((f['bytes'] for v in results for f in v['files'])), updated_unix=time.time())
            f_save(OUT / 'download_status.json', state)
            print(r['subject'], r['task'], r['state'], r.get('error', ''), flush=True)
    f_save(OUT / 'download_manifest.json', dict(protocol=protocol, records=sorted(results, key=lambda x: (x['subject'], x['task']))))
    state['state'] = 'complete_with_errors' if state['errors'] else 'complete'
    f_save(OUT / 'download_status.json', state)
