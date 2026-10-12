from f_runtime import f_work_root
"""Locked cohort/QC and calibration-only input for connectome validation."""
import os, sys, io, json, time, hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import nibabel as nib
from f_data import f_read, f_save, f_sha, f_identity, f_segment, f_standardize, RUNS
MOTION_SOURCE = 'https://wiki.humanconnectome.org/docs/How%20To%20Handle%20Downloaded%20HCP%20Data.html'
SEED = 20261007

def f_fd(parameters):
    p = np.asarray(parameters, dtype='float64')
    assert p.shape == (1200, 12) and np.isfinite(p).all()
    difference = np.diff(p[:, :6], axis=0, prepend=p[0:1, :6])
    return np.abs(difference[:, :3]).sum(1) + 50 * np.abs(np.deg2rad(difference[:, 3:6])).sum(1)

def f_qc_segment(displacement, start, stop):
    x = displacement[start:stop]
    keep = x <= 0.5
    return dict(mean_FD_mm=float(x.mean()), fraction_FD_above_05=float((~keep).mean()), retained_frames=int(keep.sum()), passed=bool(x.mean() <= 0.3 and keep.mean() >= 0.8), frames=[start, stop])

def f_paths(root, subject):
    result = {}
    for r in RUNS:
        p = list((root / subject / 'MNINonLinear/Results' / r).glob('*MSMAll*clean*.dtseries.nii'))
        if len(p) != 1:
            return None
        result[r] = str(p[0])
    return result

def f_prepare(w, out):
    """Obtain only small motion/surface ancillary files; never bulk raw data."""
    w = Path(w)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'cohorts.json').exists():
        return f_read(out / 'cohorts.json')
    sys.path.insert(0, str(w / 'deps'))
    import boto3
    from botocore.config import Config
    client = boto3.client('s3', config=Config(connect_timeout=10, read_timeout=30, retries={'max_attempts': 3}))
    import logging
    logging.getLogger('nibabel.global').setLevel(logging.ERROR)
    population = f_read(w / 'runs/fullfc450_crossday_v1/subjects.json')
    old = f_read(w / 'runs/modulated_vae_full_crossday_v1/subjects.json')
    splits = f_read(w / 'manifests/splits.json')
    reserved = set(old['reserved_retest'])
    train = {r['subject'] for r in population['train']}
    previous = {r['subject'] for key in ['train', 'validation', 'test'] for r in splits[key]}
    previous |= {p.name for p in (w / 'runs').glob('*/individual/*') if p.is_dir() and p.name.isdigit()}
    reference = f_read(w / 'runs/corrected_personal_multires_20261006_v1/subjects.json')
    assert [r['subject'] for r in reference['group_training']] == [r['subject'] for r in population['train']]
    assert [r['subject'] for r in old['train']] == [r['subject'] for r in population['train']]
    assert len(train) == 450 and (not train & reserved)
    roots = {'baseline': Path('/dataspace/yuhang_data/HCP-YA'), 'retest': Path('/dataspace/yuhang_data/HCP-Retest')}
    candidates = []
    retest = []
    for p in sorted(roots['baseline'].iterdir()):
        if p.is_dir() and p.name.isdigit() and (p.name not in train | previous | reserved):
            r = f_paths(roots['baseline'], p.name)
            if r:
                candidates.append(dict(subject=p.name, visits={'baseline': dict(runs=r)}))
    for s in sorted(reserved):
        a = f_paths(roots['baseline'], s)
        b = f_paths(roots['retest'], s)
        if a and b:
            retest.append(dict(subject=s, visits={'baseline': dict(runs=a), 'retest': dict(runs=b)}))
    rng = np.random.default_rng(SEED)
    candidates = [candidates[i] for i in rng.permutation(len(candidates))]
    ancillary = w / 'resources/connectome_ancillary'

    def fetch(key, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            response = client.get_object(Bucket='hcp-openaccess', Key=key)
            payload = response['Body'].read()
            assert len(payload) == response['ContentLength']
            temp = dest.with_suffix(dest.suffix + '.part')
            temp.write_bytes(payload)
            temp.replace(dest)
        return dict(key=key, **f_identity(dest), sha256=f_sha(dest))

    def audit(row):
        saved = out / 'QC' / f"{row['subject']}.json"
        if saved.exists():
            cached = f_read(saved)
            if cached['visits'] == row['visits']:
                return cached
        row = json.loads(json.dumps(row))
        s = row['subject']
        row['QC'] = {}
        row['ancillary'] = {}
        problems = []
        try:
            for visit, v in row['visits'].items():
                prefix = 'HCP_1200' if visit == 'baseline' else 'HCP_Retest'
                row['QC'][visit] = {}
                row['ancillary'][visit] = {}
                for r, source in v['runs'].items():
                    img = nib.load(source, mmap=False)
                    assert img.shape == (1200, 91282) and abs(img.header.get_axis(0).step - 0.72) < 1e-05
                    axis = img.header.get_axis(1)
                    mapping = np.load(w / 'runs/pilot32_MSMAll_samplez/mapping.npz')
                    for h, name in [('L', 'CIFTI_STRUCTURE_CORTEX_LEFT'), ('R', 'CIFTI_STRUCTURE_CORTEX_RIGHT')]:
                        cols = np.flatnonzero(axis.surface_mask & (axis.name == name))
                        assert np.array_equal(cols, mapping[h + '_columns']) and np.array_equal(axis.vertex[cols], mapping[h + '_vertices'])
                    key = f'{prefix}/{s}/MNINonLinear/Results/{r}/Movement_Regressors.txt'
                    dest = ancillary / prefix / s / r / 'Movement_Regressors.txt'
                    record = fetch(key, dest)
                    displacement = f_fd(np.loadtxt(dest))
                    np.save(dest.with_name('FD.npy'), displacement)
                    row['ancillary'][visit][r] = record
                    row['QC'][visit][r] = {str(stop): f_qc_segment(displacement, 0, stop) for stop in [240, 480, 960, 1200]}
                    row['QC'][visit][r]['tail480'] = f_qc_segment(displacement, 720, 1200)
                    if not row['QC'][visit][r]['480']['passed']:
                        problems.append(visit + '/' + r + '/motion_QC')
            for h in ['L', 'R']:
                name = f'{s}.{h}.midthickness_MSMAll.32k_fs_LR.surf.gii'
                existing = w / 'resources/subject_surfaces' / name
                if existing.exists():
                    dest = existing
                else:
                    dest = ancillary / 'surfaces' / name
                    fetch(f'HCP_1200/{s}/MNINonLinear/fsaverage_LR32k/{name}', dest)
                surf = nib.load(dest)
                xyz = surf.agg_data('pointset')
                tri = surf.agg_data('triangle')
                assert xyz.shape == (32492, 3) and tri.ndim == 2 and (tri.shape[1] == 3)
                area = np.zeros(len(xyz), dtype='float64')
                a = 0.5 * np.linalg.norm(np.cross(xyz[tri[:, 1]] - xyz[tri[:, 0]], xyz[tri[:, 2]] - xyz[tri[:, 0]]), axis=1)
                for j in range(3):
                    np.add.at(area, tri[:, j], a / 3)
                vertices = np.load(w / 'runs/pilot32_MSMAll_samplez/mapping.npz')[h + '_vertices']
                area = area[vertices]
                assert np.isfinite(area).all() and (area > 0).all()
                ap = ancillary / 'area' / f'{s}_{h}.npy'
                ap.parent.mkdir(parents=True, exist_ok=True)
                np.save(ap, area)
                row.setdefault('area', {})[h] = dict(path=str(ap), surface=str(dest), surface_sha256=f_sha(dest), frozen_visit='baseline')
        except Exception as e:
            problems.append(type(e).__name__ + ': ' + str(e)[:250])
        row['passed'] = not problems
        row['exclusions'] = problems
        f_save(out / 'QC' / f'{s}.json', row)
        return row
    rows = retest + candidates
    results = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(audit, row): row['subject'] for row in rows}
        for f in as_completed(futures):
            r = f.result()
            results[r['subject']] = r
            f_save(out / 'prepare_status.json', dict(state='preparing', finished=len(results), total=len(rows), passed=sum((r['passed'] for r in results.values())), updated_unix=time.time()))
    accepted = [results[r['subject']] for r in candidates if results[r['subject']]['passed']][:400]
    paired = [results[r['subject']] for r in retest if results[r['subject']]['passed']]
    assert not {r['subject'] for r in accepted} & (train | previous | reserved)
    assert not {r['subject'] for r in paired} & train
    selected = set((r['subject'] for r in accepted + paired))
    result = dict(YA400=accepted, retest=paired, requested_YA=400, available_candidates=len(candidates), YA_passed_all=sum((results[r['subject']]['passed'] for r in candidates)), retest_file_complete=len(retest), excluded=[results[r['subject']] for r in rows if not results[r['subject']]['passed']], eligible_unselected=[r['subject'] for r in candidates if r['subject'] not in selected and results[r['subject']]['passed']], training_subjects=sorted(train), prior_used_subjects=sorted(previous), reserved_retest=sorted(reserved), split_seed=SEED, motion_units_source=MOTION_SOURCE, motion_units='translation mm; rotation degrees; radius50mm', QC=dict(mean_FD_max_mm=0.3, censor_FD_above_mm=0.5, max_censored_fraction=0.2, minimum_frames=384), family_metadata_available=False, retest_prior_development_warning='Some Retest identities were inspected in early pilots; not a wholly untouched model-selection cohort.', provenance_sha256={str(p): f_sha(p) for p in [w / 'runs/fullfc450_crossday_v1/subjects.json', w / 'runs/modulated_vae_full_crossday_v1/subjects.json', w / 'manifests/splits.json']})
    f_save(out / 'cohorts.json', result)
    f_save(out / 'prepare_status.json', dict(state='complete', YA_selected=len(accepted), retest_selected=len(paired)))
    return result

def f_signals(w, out, row, visit, run, h, mapping, start=0, stop=480):
    entry = row['visits'][visit]
    cache = out / 'filtered_segments' / visit
    y = f_segment(dict(subject=row['subject'], runs=entry['runs']), run, h, start, stop, mapping, cache)
    motion = row['ancillary'][visit][run]['path']
    displacement = np.load(Path(motion).with_name('FD.npy'))[start:stop]
    keep = displacement <= 0.5
    return f_standardize(y[keep])
