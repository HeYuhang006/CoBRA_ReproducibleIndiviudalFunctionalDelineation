from f_runtime import f_work_root
"""Matched classical individual controls; never claims original Wang/GPIP equivalence."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
os.environ.setdefault('OMP_NUM_THREADS', '8')
from pathlib import Path
import sys, json, time, argparse, fcntl, heapq, traceback
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.utils.extmath import randomized_svd
from sklearn.cluster import KMeans, ward_tree
from threadpoolctl import threadpool_limits
W = f_work_root()
sys.path.insert(0, str(W / 'final'))
from f_data import f_read, f_save, f_sha
from f_application_data import f_signals
from f_metrics import f_agreement, f_topology
from f_application_metrics import f_regional, f_connectome
from f_application_statistics import f_summary
APP = W / 'runs/connectome_validation_20261007_v1/retest'
ROOT = W / 'runs/individual_classical_matched36_3timepoints_20261008_v1'
KS = [50, 100, 150, 200]
METHODS = ['PCA64_KMeans', 'PCA64_SpatialWard']

def f_cut_tree(children, n, k):
    heap = [-(2 * n - 2)]
    while len(heap) < k:
        node = -heapq.heappop(heap)
        assert node >= n
        for child in children[node - n]:
            heapq.heappush(heap, -int(child))
    labels = np.full(n, -1, dtype='int32')
    for label, negative in enumerate(sorted(heap)):
        stack = [-negative]
        while stack:
            node = stack.pop()
            if node < n:
                labels[node] = label
            else:
                stack.extend(map(int, children[node - n]))
    assert (labels >= 0).all() and len(np.unique(labels)) == k
    return labels

def f_fit_maps(y, edges, subject, h, visit, out):
    dest = out / 'maps' / subject / h / visit
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'complete.json').exists():
        return
    n = y.shape[1]
    e = np.r_[edges, edges[:, ::-1]]
    adj = csr_matrix((np.ones(len(e), dtype='float32'), (e[:, 0], e[:, 1])), shape=(n, n))
    assert connected_components(adj, directed=False, return_labels=False) == 1, 'Do not silently add graph bridges'
    f_save(out / 'status.json', dict(state='running', phase='PCA', subject=subject, hemisphere=h, visit=visit, pid=os.getpid(), updated_unix=time.time()))
    U, S, _ = randomized_svd(y.T, n_components=64, n_iter=3, n_oversamples=16, random_state=20261008)
    features = np.ascontiguousarray(U * S, dtype='float32')
    del U
    f_save(out / 'status.json', dict(state='running', phase='spatial_Ward', subject=subject, hemisphere=h, visit=visit, pid=os.getpid(), updated_unix=time.time()))
    children, n_components, n_leaves, parents = ward_tree(features, connectivity=adj, n_clusters=None, return_distance=False)
    assert n_components == 1 and n_leaves == n
    for k in KS:
        v = f_cut_tree(children, n, k)
        t = f_topology(v, adj, k)
        assert t['components'] == [1] * k
        np.save(dest / f'PCA64_SpatialWard_{2 * k}.npy', v)
        f_save(dest / f'PCA64_SpatialWard_{2 * k}.topology.json', t)
        f_save(out / 'status.json', dict(state='running', phase='KMeans', subject=subject, hemisphere=h, visit=visit, K=2 * k, pid=os.getpid(), updated_unix=time.time()))
        estimator = KMeans(n_clusters=k, n_init=5, max_iter=100, random_state=20261008, algorithm='lloyd')
        v = estimator.fit_predict(features).astype('int32')
        assert len(np.unique(v)) == k
        np.save(dest / f'PCA64_KMeans_{2 * k}.npy', v)
        f_save(dest / f'PCA64_KMeans_{2 * k}.topology.json', f_topology(v, adj, k))
    f_save(dest / 'complete.json', dict(subject=subject, hemisphere=h, visit=visit, input_shape=list(y.shape), K_bilateral=[100, 200, 300, 400], measurement_data_used=False))

def f_map_agreement(row, out):
    s = row['subject']
    dest = out / 'map_reproducibility' / f'{s}.json'
    if dest.exists():
        return
    result = {}
    for h in ['L', 'R']:
        result[h] = {}
        for k in KS:
            result[h][str(2 * k)] = {}
            for name in METHODS:
                base = out / f'maps/{s}/{h}/baseline/{name}_{2 * k}'
                later = out / f'maps/{s}/{h}/retest/{name}_{2 * k}'
                a = np.load(base.with_suffix('.npy'))
                b = np.load(later.with_suffix('.npy'))
                result[h][str(2 * k)][name] = dict(**f_agreement(a, b, k), first_topology=f_read(base.with_name(base.name + '.topology.json')), second_topology=f_read(later.with_name(later.name + '.topology.json')))
                day2 = out / f'maps/{s}/{h}/day2/{name}_{2 * k}'
                d = np.load(day2.with_suffix('.npy'))
                result[h][str(2 * k)][name]['Day1_Day2'] = f_agreement(a, d, k)
                result[h][str(2 * k)][name]['Day2_Retest'] = f_agreement(d, b, k)
                result[h][str(2 * k)][name]['Day2_topology'] = f_read(day2.with_name(day2.name + '.topology.json'))
    f_save(dest, result)

def f_measure(row, mapping, out):
    s = row['subject']
    dest = out / 'measurements' / s
    dest.mkdir(parents=True, exist_ok=True)
    if all(((dest / f'{2 * k}.npz').exists() for k in KS)):
        return
    areas = {h: np.load(row['area'][h]['path']) for h in ['L', 'R']}
    values = {(m, k): [] for m in METHODS for k in KS}
    for visit in ['baseline', 'retest']:
        runs = {(m, k): [] for m in METHODS for k in KS}
        for pe in ['LR', 'RL']:
            ys = {h: f_signals(W, APP, row, visit, 'rfMRI_REST2_' + pe, h, mapping) for h in ['L', 'R']}
            for m in METHODS:
                for k in KS:
                    roi = {h: f_regional(ys[h], np.load(out / f'maps/{s}/{h}/baseline/{m}_{2 * k}.npy'), areas[h], k) for h in ['L', 'R']}
                    runs[m, k].append(f_connectome(roi['L'], roi['R']))
            del ys
        for key in values:
            values[key].append(np.mean(runs[key], axis=0))
    for k in KS:
        np.savez_compressed(dest / f'{2 * k}.npz', **{m: np.stack(values[m, k]) for m in METHODS})

def f_main(a):
    out = ROOT / ('smoke' if a.smoke else 'full')
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'workflow.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = f_read(APP / 'subjects.json')
    assert len(rows) == 36
    protocol = dict(subjects=[r['subject'] for r in rows], K_bilateral=[100, 200, 300, 400], hemispheres=['L', 'R'], methods=METHODS, calibration='Day1: baseline REST1; Day2: baseline REST2; Retest: retest REST1. LR+RL first480 per run, same filtering/censoring as CoBRA', timepoints=['Day1', 'Day2', 'Retest'], bandpass_Hz=[0.01, 0.1], TR_seconds=0.72, measurement='Independent REST2 first480 LR/RL Fisher-z mean, fixed baseline labels; same CoBRA measurements', PCA=dict(components=64, randomized_power_iterations=3, oversamples=16, seed=20261008), KMeans=dict(n_init=5, max_iter=100), spatial_Ward='Full hierarchical Ward tree on PCA scores with actual cortical mesh adjacency; no hidden bridging; native K cuts', official_published_method_claim=False, Wang2015='Not implemented here; historical Wang-style adaptation is not an original-author baseline', GPIP='Separate 40-person diagnostic run; do not silently merge mismatched cohorts or visits', CPU_threads=8, code_sha256=f_sha(Path(__file__)))
    if (out / 'protocol.json').exists():
        assert f_read(out / 'protocol.json') == protocol
    else:
        f_save(out / 'protocol.json', protocol)
    if not a.smoke:
        assert f_read(ROOT / 'smoke/complete.json')['passed']
    mapping = np.load(W / 'runs/pilot32_MSMAll_samplez/mapping.npz')
    selected = rows[:1] if a.smoke else rows
    with threadpool_limits(limits=8):
        for i, row in enumerate(selected):
            for h in ['L', 'R']:
                edges = np.load(W / f'runs/full500_MSMAll_samplez/edges_{h}.npy')
                for visit in ['baseline', 'day2', 'retest']:
                    actual_visit = 'baseline' if visit == 'day2' else visit
                    rest = 2 if visit == 'day2' else 1
                    ys = [f_signals(W, APP, row, actual_visit, f'rfMRI_REST{rest}_' + pe, h, mapping) for pe in ['LR', 'RL']]
                    f_fit_maps(np.concatenate(ys, axis=0), edges, row['subject'], h, visit, out)
                    del ys
            f_map_agreement(row, out)
            f_measure(row, mapping, out)
            f_save(out / 'status.json', dict(state='running', phase='subject_complete', completed=i + 1, total=len(selected), subject=row['subject'], pid=os.getpid(), updated_unix=time.time()))
        if not a.smoke:
            result = {}
            for k in KS:
                result[str(2 * k)] = {}
                files = [np.load(out / f"measurements/{r['subject']}/{2 * k}.npz") for r in rows]
                for m in METHODS:
                    x = np.stack([f[m] for f in files])
                    result[str(2 * k)][m] = f_summary(x[:, 0], x[:, 1])
            f_save(out / 'connectome_summary.json', result)
    f_save(out / 'complete.json', dict(passed=True, participants=len(selected), both_hemispheres=True, all_four_resolutions=True, independent_visit_fits=True, completed_unix=time.time()))
    f_save(out / 'status.json', dict(state='complete', subjects=len(selected), pid=os.getpid(), updated_unix=time.time()))
