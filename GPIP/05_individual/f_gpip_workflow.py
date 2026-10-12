from f_runtime import f_work_root
"""Finite, checkpointed author-GPIP benchmark. Original distribution is read-only.

Runs three visits independently; group coupling is across people within a visit.
No custom connectivity guard, affinity clipping, label repair or outcome selection.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import fcntl, hashlib, json, shutil, subprocess, sys, time, traceback, zipfile
import numpy as np
from scipy import sparse
from scipy.io import loadmat, savemat
from scipy.sparse.csgraph import connected_components
from scipy.stats import wilcoxon
import nibabel as nib
W = f_work_root()
G = W / 'gpip_author_20261009'
OUT = W / 'runs/gpip_author_20261009_v2'
N = W / 'runs/coarse_network18_20261008_v1'
M = W / 'runs/published_baselines_matched36_k100_20261008_v1'
OLD = W / 'runs/final_figure_layout_20261009_v1'
VISITS = ['Day1', 'Day2', 'Retest']
METHODS = ['PCA', 'Wang', 'Ward', 'GPIP', 'CoBRA']
PAIRS = ['Day1_Day2', 'Day1_Retest']
sys.path.insert(0, str(G / 'code'))
import f_gpip_subject_specificity as SS

def f_read(p):
    return json.loads(Path(p).read_text())

def f_save(p, d):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(d, indent=2, allow_nan=False))
    t.replace(p)

def f_sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2 ** 20), b''):
            h.update(b)
    return h.hexdigest()

def f_status(phase, **kw):
    f_save(OUT / 'queue_status.json', dict(state='running', phase=phase, pid=os.getpid(), updated_unix=time.time(), **kw))
    print(json.dumps(dict(phase=phase, **kw)), flush=True)

def f_source_check():
    manifest = f_read(G / 'source_manifest.json')
    ref = G / 'reference/gpip_share'
    for r in manifest['files']:
        p = ref / r['path']
        assert p.stat().st_size == r['bytes'] and f_sha(p) == r['sha256'], p
    f_save(OUT / 'source_integrity.json', dict(passed=True, files=len(manifest['files']), original_files_unmodified=True, source='User-provided author distribution', manifest_sha256=f_sha(G / 'source_manifest.json')))
    code = {str(p.relative_to(G)): f_sha(p) for p in [G / 'compat/eigs_15.m', G / 'code/f_run_author_gpip.m', G / 'code/f_validate_author_runtime.m', G / 'code/f_gpip_subject_specificity.py', Path(__file__)]}
    frozen = OUT / 'driver_integrity.json'
    if frozen.exists():
        assert f_read(frozen)['sha256'] == code, 'Frozen GPIP benchmark driver changed; use a new output version.'
    else:
        f_save(frozen, dict(sha256=code, created_unix=time.time()))
    assert f_read(OUT / 'runtime_validation.json')['passed']
    return code

def f_mesh(path):
    a = nib.load(str(path))
    return (np.asarray(a.darrays[0].data, dtype='float64'), np.asarray(a.darrays[1].data, dtype='int64'))

def f_geometry(K):
    target = OUT / f'geometry/K{K}.mat'
    meta = OUT / f'geometry/K{K}.json'
    if target.exists():
        d = f_read(meta)
        assert f_sha(target) == d['sha256']
        return (target, d)
    target.parent.mkdir(parents=True, exist_ok=True)
    if K == 18:
        res = W / 'references/wang2015_author_hfr/resampling'
        vs = []
        fs = []
        labs = []
        sources = []
        for j, h in enumerate(['L', 'R']):
            p = res / f'fsaverage4_std_sphere.{h}.3k_fsavg_{h}.surf.gii'
            v, f = f_mesh(p)
            assert v.shape == (2562, 3)
            vs.append(v)
            fs.append(f + j * 2562)
            p2 = N / f'geometry/author_labels_{h}.npy'
            y = np.load(p2)
            assert y.shape == (2562,)
            labs.append(y + 1)
            sources += [p, p2]
        v = np.concatenate(vs)
        f = np.concatenate(fs)
        labels = np.concatenate(labs)
        clusters = 18
    else:
        row = f_read(W / 'runs/connectome_validation_20261007_v1/retest/subjects.json')[0]
        p = Path(row['area']['L']['surface'])
        assert f_sha(p) == row['area']['L']['surface_sha256']
        v, f = f_mesh(p)
        assert v.shape == (32492, 3)
        mp = W / 'runs/pilot32_MSMAll_samplez/mapping.npz'
        mapping = np.load(mp)['L_vertices']
        lp = W / 'runs/corrected_personal_multires_20261006_v1/priors/L/50/labels.npy'
        y = np.load(lp)
        assert y.shape == (29696,) and np.array_equal(np.unique(y), np.arange(50))
        labels = np.zeros(32492, dtype='int32')
        labels[mapping] = y + 1
        clusters = 50
        sources = [p, mp, lp]
    assert labels.min() == 0 and np.array_equal(np.unique(labels), np.arange(clusters + 1))
    temp = target.with_suffix('.tmp.mat')
    savemat(temp, dict(surface=dict(vertices=v, faces=f + 1), labels=labels[:, None]), do_compression=False)
    temp.replace(target)
    d = dict(K=K, clusters=clusters, vertices=len(labels), background=int(np.sum(labels == 0)), sha256=f_sha(target), sources={str(p): f_sha(p) for p in sources}, initialization='Author HFR18' if K == 18 else 'Corrected connected group prior, left50 of nominal bilateral100', native_paper_resolution=False, preprocess_atlas_split_components=False)
    f_save(meta, d)
    return (target, d)

def f_prepare(K, visit, ids, code):
    f_status('prepare_inputs', K=K, visit=visit)
    geom, gd = f_geometry(K)
    manifest = f_read((N if K == 18 else M) / 'inputs_manifest.json')
    records = {(r['subject'], r['visit']): r for r in manifest['records']}
    mats = []
    frames = []
    provenance = []
    mapping = np.load(W / 'runs/pilot32_MSMAll_samplez/mapping.npz')['L_vertices'] if K == 100 else None
    for i, s in enumerate(ids):
        r = records[s, visit]
        assert r['bandpass_hz'] == [0.01, 0.1]
        target = OUT / f'inputs/K{K}/{visit}/{s}.mat'
        meta = target.with_suffix('.json')
        paths = [Path(r['paths'][h]) for h in ['L', 'R']] if K == 18 else [Path(r['path'])]
        inputhash = {str(p): f_sha(p) for p in paths}
        if K == 100:
            assert inputhash[str(paths[0])] == r['sha256']
        if target.exists():
            info = f_read(meta)
            assert info['source_sha256'] == inputhash and info['mat_sha256'] == f_sha(target)
        else:
            if K == 18:
                arrays = [np.load(p) for p in paths]
                assert arrays[0].shape == arrays[1].shape
                ts = np.concatenate(arrays, axis=0)
                assert ts.shape[0] == 5124
            else:
                y = np.load(paths[0])
                assert list(y.shape) == r['shape']
                ts = np.zeros((32492, y.shape[1]), dtype='float32')
                ts[mapping] = y
            assert np.isfinite(ts).all()
            target.parent.mkdir(parents=True, exist_ok=True)
            t = target.with_suffix('.tmp.mat')
            savemat(t, dict(ts=np.asfortranarray(ts, dtype='float32')), do_compression=False)
            t.replace(target)
            info = dict(subject=s, visit=visit, K=K, shape=list(ts.shape), source_sha256=inputhash, mat_sha256=f_sha(target), bandpass_hz=[0.01, 0.1], filter_applied_again=False, retained_T=int(ts.shape[1]), preprocessing='Frozen matched inputs; author mean-zero/sample-standard-deviation normalization in MATLAB', smoothing_FWHM_mm=6 if K == 18 else 0, other_visit_used=False)
            f_save(meta, info)
            del ts
        mats.append(str(target))
        frames.append(info['retained_T'])
        provenance.append(info)
        f_status('prepare_inputs', K=K, visit=visit, subject_index=i + 1, subjects=len(ids))
    job = OUT / f'jobs/K{K}_{visit}'
    job.mkdir(parents=True, exist_ok=True)
    cfg = dict(reference=str(G / 'reference/gpip_share'), adapter=str(G / 'compat'), geometry=str(geom), geometry_sha256=gd['sha256'], subjects=ids, clusters=gd['clusters'], mat_files=mats, frames=frames, iterations=20, lambda_=0.01, beta=1e-08, seed=20261009 + K, output=str(job), nominal_K=K, visit=visit, driver_sha256=code, source_manifest_sha256=f_sha(G / 'source_manifest.json'))
    cfg['lambda'] = cfg.pop('lambda_')
    cfg['config_sha256'] = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    cp = job / 'config.json'
    if cp.exists():
        assert f_read(cp) == cfg
    else:
        f_save(cp, cfg)
    f_save(job / 'input_provenance.json', provenance)
    return (job, cp, cfg)

def f_run_job(K, visit, ids, code):
    job, cp, cfg = f_prepare(K, visit, ids, code)
    done = job / 'complete.json'
    if done.exists():
        assert f_read(done)['config_sha256'] == cfg['config_sha256']
        return
    f_status('optimization', K=K, visit=visit, output=str(job))
    expr = f"addpath('{G}/code');f_run_author_gpip('{cp}')"
    with (job / 'matlab.log').open('a') as f:
        p = subprocess.Popen(['/usr/bin/matlab', '-singleCompThread', '-batch', expr], stdout=f, stderr=subprocess.STDOUT)
        f_save(job / 'process.json', dict(pid=p.pid, started_unix=time.time()))
        rc = p.wait()
    assert rc == 0 and done.exists(), f'MATLAB failed rc={rc}; inspect {job}/matlab.log'
    assert f_read(done)['config_sha256'] == cfg['config_sha256']

def f_topology(y, K, graphs):
    cs = []
    occupied = 0
    hemis = [y[:2562], y[2562:]] if K == 18 else [y]
    for z, a in zip(hemis, graphs):
        counts = []
        for k in range(18 if K == 18 else 50):
            ix = np.flatnonzero(z == k)
            c = connected_components(a[ix][:, ix], directed=False, return_labels=False) if len(ix) else 0
            counts.append(int(c))
        cs.extend(counts)
        occupied += sum((c > 0 for c in counts))
    assert occupied > 0
    extra = sum(cs) - occupied
    return dict(value=float(sum(cs) / occupied) if K == 18 else extra / 50, extra_components=extra, occupied_units=occupied, components=cs, missing_units=len(cs) - occupied, units='Connected components per occupied hemisphere-network unit' if K == 18 else 'Extra connected components per left parcel')

def f_evaluate(ids):
    f_status('evaluation')
    allmaps = {}
    records = []
    tops = []
    rep = ids[0]
    arrays = {}
    native_hash = {}
    mapping = np.load(W / 'runs/pilot32_MSMAll_samplez/mapping.npz')['L_vertices']
    masks = [np.load(N / f'geometry/evaluation_mask_{h}.npy') for h in ['L', 'R']]
    nearest = np.load(N / 'geometry/nearest4_L.npy')
    for K in [18, 100]:
        clusters = 18 if K == 18 else 50
        allmaps[K] = {}
        map_paths = {v: [] for v in VISITS}
        if K == 18:
            graphs = [sparse.load_npz(N / f'geometry/adjacency_{h}.npz').tocsr() for h in ['L', 'R']]
        else:
            e = np.load(W / 'runs/pilot32_MSMAll_samplez/edges_L.npy')
            graphs = [sparse.csr_matrix((np.ones(len(e) * 2), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(29696, 29696))]
        for visit in VISITS:
            p = OUT / f'jobs/K{K}_{visit}/maps.mat'
            native_hash[str(p)] = f_sha(p)
            a = loadmat(p)
            assert len(a['history'].reshape(-1)) == 20
            cells = a['labels'].reshape(-1)
            assert len(cells) == len(ids)
            allmaps[K][visit] = []
            for s, c in zip(ids, cells):
                full = c.reshape(-1).astype('int32') - 1
                y = np.concatenate([full[:2562][masks[0]], full[2562:][masks[1]]]) if K == 18 else full[mapping]
                assert y.min() >= 0 and y.max() < clusters
                assert y.shape == ((4649,) if K == 18 else (29696,))
                native = full.copy() if K == 18 else y
                if K == 18:
                    native[~np.concatenate(masks)] = -1
                t = f_topology(native, K, graphs)
                tops.append(dict(subject=s, method='GPIP', K=K, visit=visit, **t))
                path = OUT / f'evaluation/maps/K{K}/{visit}/{s}.npy'
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, y)
                map_paths[visit].append(str(path))
                allmaps[K][visit].append(y)
                if s == rep:
                    if K == 18:
                        z = np.full(2562, -1, dtype='int32')
                        z[masks[0]] = full[:2562][masks[0]]
                        z = z[nearest]
                    else:
                        z = y
                    arrays[f'dual{K}_GPIP_{visit}'] = z
        for first, second in [('Day1', 'Day2'), ('Day1', 'Retest')]:
            for s, a, b in zip(ids, allmaps[K][first], allmaps[K][second]):
                for metric, value in SS.f_agreement(a, b, clusters).items():
                    records.append(dict(subject=s, method='GPIP', K=K, metric=metric, contrast=first + '_' + second, value=value))
        for s in ids:
            values = [r['value'] for r in tops if r['subject'] == s and r['K'] == K]
            assert len(values) == 3
            records.append(dict(subject=s, method='GPIP', K=K, metric='Topology', contrast='Mean_three_visits', value=float(np.mean(values))))
        cobra = (f_read(N / 'evaluation/manifest.json') if K == 18 else f_read(W / 'runs/subject_specificity_K100_20261008_v1/summary.json')['manifest'])['methods']['CoBRA']
        spec = dict(subjects=ids, methods={'GPIP_author': dict(K=clusters, maps=map_paths, allow_missing=True), 'CoBRA': cobra}, protocol='Matched native support, independently fitted visits; author implementation with explicit runtime adapter', bandpass_hz=[0.01, 0.1])
        manifest = OUT / f'evaluation/specificity_K{K}/manifest.json'
        f_save(manifest, spec)
        SS.f_run(manifest, manifest.parent)
    f_save(OUT / 'evaluation/author_records.json', dict(records=records, topology=tops, native_map_sha256=native_hash))
    np.savez_compressed(OUT / 'evaluation/author_representative_maps.npz', **arrays)
    return (records, tops, arrays, native_hash)

def f_integrate(ids, records, tops, arrays, native_hash):
    f_status('figure_data_integration')
    d = f_read(OLD / 'individual_data.json')
    assert d['protocol']['subjects'] == ids
    d['records'] = [r for r in d['records'] if r['method'] != 'GPIP'] + records
    d['topology'] = [r for r in d['topology'] if r['method'] != 'GPIP'] + tops
    tests = []
    for K in [18, 100]:
        for metric, contrasts in [('ARI', PAIRS), ('Parcel_Dice', PAIRS), ('Topology', ['Mean_three_visits'] if K == 100 else [])]:
            for contrast in contrasts:

                def values(method):
                    return np.array([next((r['value'] for r in d['records'] if r['subject'] == s and r['method'] == method and (r['K'] == K) and (r['metric'] == metric) and (r['contrast'] == contrast))) for s in ids])
                for comparator in METHODS[:-1]:
                    delta = values('CoBRA') - values(comparator)
                    p = float(wilcoxon(delta, zero_method='wilcox', method='approx', correction=False).pvalue) if np.any(delta) else 1.0
                    tests.append(dict(K=K, metric=metric, contrast=contrast, comparator=comparator, n=36, median_paired_difference=float(np.median(delta)), p_raw=p))
    assert len(tests) == 36
    order = np.argsort([t['p_raw'] for t in tests])
    adjusted = np.maximum.accumulate([(len(tests) - j) * tests[i]['p_raw'] for j, i in enumerate(order)])
    for i, p in zip(order, adjusted):
        tests[i]['p_holm'] = min(1.0, float(p))
    d['comparisons'] = tests
    p = d['protocol']
    p['gpip_source'] = 'user_provided_author_code_with_disclosed_runtime_adapter'
    for K in [18, 100]:
        p['implementations'][f'GPIP{K}'] = 'Author GPIP code: original group precision, signed Ncut and relabeling; modern MATLAB eigs adapter, actual censored T bookkeeping. Matched initialization/resolution/preprocessing, not a bitwise replication of original2017 study.'
    p['GPIP_protocol'] = dict(source_verified=True, iterations=20, lambda_group=0.01, beta=1e-08, rho=0.5, precision_iterations=100, visit_coupling='36 people within each visit; no cross-visit coupling', K18='Bilateral fsaverage4 HFR18 semantic network initialization, without splitting components. Extension of the original GPIP resolution.', K100='Left fsLR32k50 parcels of nominal bilateral100, corrected connected group prior. Single-hemisphere matched extension.', native_original_initializations='Paper: Yeo17 split/merged to58 and Glasser360 merged to338; not claimed here.', no_custom_connectivity_guard=True, no_empty_cluster_repair=True, negative_affinities_preserved=True, source_manifest_sha256=f_sha(G / 'source_manifest.json'), driver_sha256=f_read(OUT / 'driver_integrity.json')['sha256'])
    p['source_sha256'].update(native_hash)
    p['collector_sha256'] = f_sha(__file__)
    a = {k: v.copy() for k, v in np.load(OLD / 'representative_maps.npz').items()}
    a.update(arrays)
    assert len(a) == 32 and all((v.shape == (29696,) for v in a.values()))
    assert len(d['records']) == 1800 and len(d['topology']) == 1080
    original = f_read(OLD / 'individual_data.json')
    assert [r for r in d['records'] if r['method'] != 'GPIP'] == [r for r in original['records'] if r['method'] != 'GPIP']
    target = OUT / 'figure_data'
    target.mkdir(exist_ok=True)
    f_save(target / 'individual_data.json', d)
    np.savez_compressed(target / 'representative_maps.npz', **a)
    with zipfile.ZipFile(target / 'figure_data.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for f in [target / 'individual_data.json', target / 'representative_maps.npz']:
            z.write(f, f.name)
    return dict(path=str(target), zip_sha256=f_sha(target / 'figure_data.zip'), individual_data_sha256=f_sha(target / 'individual_data.json'), representative_maps_sha256=f_sha(target / 'representative_maps.npz'))

def f_main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'workflow.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('ALREADY_RUNNING')
        return
    if (OUT / 'final_integrity.json').exists() and f_read(OUT / 'final_integrity.json').get('passed'):
        print('ALREADY_COMPLETE')
        return
    try:
        f_status('preflight')
        code = f_source_check()
        assert shutil.disk_usage(OUT).free > 30 * 2 ** 30
        ids = f_read(OLD / 'individual_data.json')['protocol']['subjects']
        assert len(ids) == len(set(ids)) == 36
        f_save(OUT / 'protocol.json', dict(subjects=ids, visits=VISITS, K=[18, 100], bandpass_hz=[0.01, 0.1], independent_visits=True, group_within_visit=True, iterations=20, lambda_group=0.01, beta=1e-08, task_connectome_validation_in_this_queue=False, scope='ARI, macro Hungarian Dice, topology, intra/inter specificity, figure source integration', original_source_unmodified=True, driver_sha256=code, created_unix=time.time()))
        for K in [18, 100]:
            for visit in VISITS:
                f_run_job(K, visit, ids, code)
        records, tops, arrays, native_hash = f_evaluate(ids)
        integrated = f_integrate(ids, records, tops, arrays, native_hash)
        f_source_check()
        checks = {f'K{K}_{v}': f_read(OUT / f'jobs/K{K}_{v}/complete.json') for K in [18, 100] for v in VISITS}
        assert all((c['subjects'] == 36 and c['iterations'] == 20 and c['source_core_unmodified'] for c in checks.values()))
        f_save(OUT / 'final_integrity.json', dict(passed=True, source_verified=True, all_required_comparisons_complete=True, figure_data_integrated=True, required_comparisons_scope='18/100 reproducibility, native topology and intra/inter specificity; excludes new task/connectome GPIP evaluation', checks=checks, figure_bundle=integrated, completed_unix=time.time()))
        f_save(OUT / 'queue_status.json', dict(state='complete', updated_unix=time.time(), figure_bundle=integrated))
    except BaseException as e:
        f_save(OUT / 'queue_status.json', dict(state='failed', error=str(e), traceback=traceback.format_exc(), updated_unix=time.time(), pid=os.getpid()))
        raise
