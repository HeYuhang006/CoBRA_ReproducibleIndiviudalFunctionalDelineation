from f_runtime import f_work_root
"""Matched inputs for Wang2015 / GPIP comparison, with explicit validity gates.

Preparing inputs does not authorize calling an adapted implementation the
published method. Diagnostic fits are segregated from publication-ready figures.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
from pathlib import Path
import sys, json, time, argparse, hashlib, fcntl
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
W = f_work_root()
sys.path.insert(0, str(W / 'final'))
sys.path.insert(1, str(W / 'gpip_comparison'))
from f_data import f_read, f_save, f_sha
from f_application_data import f_signals
OUT = W / 'runs/published_baselines_matched36_k100_20261008_v1'
APP = W / 'runs/connectome_validation_20261007_v1/retest'
PRIOR = W / 'runs/corrected_personal_multires_20261006_v1'
VISITS = [('Day1', 'baseline', 1), ('Day2', 'baseline', 2), ('Retest', 'retest', 1)]

def f_prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'prepare.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = f_read(APP / 'subjects.json')
    assert len(rows) == 36
    mapping = np.load(W / 'runs/pilot32_MSMAll_samplez/mapping.npz')
    initial = np.load(PRIOR / 'priors/L/50/labels.npy')
    n = len(initial)
    edges = np.load(W / 'runs/pilot32_MSMAll_samplez/edges_L.npy')
    adj = coo_matrix((np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n)).tocsr()
    assert np.array_equal(np.unique(initial), np.arange(50))
    assert all((connected_components(adj[initial == k][:, initial == k], directed=False)[0] == 1 for k in range(50)))
    cohorts = f_read(APP.parent / 'cohorts.json')
    assert not set((r['subject'] for r in rows)) & set(cohorts['training_subjects'])
    protocol = dict(subjects=[r['subject'] for r in rows], n_subjects=36, K_bilateral=100, K_left=50, hemisphere='L', bandpass_hz=[0.01, 0.1], calibration='Per visit, LR+RL first480/run; canonical filtering, FD censoring, restandardization; no temporal continuity across runs assumed.', visits=[v[0] for v in VISITS], initialization='Corrected connected CoBRA group prior, common to both requested diagnostic adaptations.', prior_sha256=f_sha(PRIOR / 'priors/L/50/labels.npy'), canonical_code_sha256={p.name: f_sha(p) for p in (W / 'final').glob('*.py')}, cohort_sha256=f_sha(APP / 'subjects.json'), prior_training_overlap=0, validity='Input equivalence only. Official algorithm equivalence is a separate gate. No publication ranking from unverified adaptations.', Wang2015='Existing Salehi-comparison adaptation omits original variability/tSNR weighting; cannot label as exact Wang2015.', GPIP2017='Independent paper implementation; signed affinities, covariance scaling and anchored discretization not author-verified.', publication_ranking_ready=False)
    p = OUT / 'protocol.json'
    if p.exists():
        assert f_read(p) == protocol
    else:
        f_save(p, protocol)
    np.savez_compressed(OUT / 'geometry.npz', initial=initial, edges=edges)
    records = []
    for visit, actual, day in VISITS:
        for index, row in enumerate(rows):
            dest = OUT / f"inputs/{visit}/{row['subject']}.npy"
            marker = dest.with_suffix('.json')
            dest.parent.mkdir(parents=True, exist_ok=True)
            if marker.exists():
                r = f_read(marker)
                assert f_sha(dest) == r['sha256']
                records.append(r)
                continue
            f_save(OUT / 'prepare_status.json', dict(state='preparing', visit=visit, index=index + 1, total_per_visit=36, completed=len(records), total=108, pid=os.getpid(), updated_unix=time.time()))
            ys = [f_signals(W, APP, row, actual, f'rfMRI_REST{day}_{pe}', 'L', mapping) for pe in ['LR', 'RL']]
            d = np.ascontiguousarray(np.concatenate(ys).T, dtype='float32')
            assert d.shape[0] == n and np.isfinite(d).all()
            assert np.max(np.abs(d.mean(1))) < 0.0001 and np.max(np.abs(d.std(1) - 1)) < 0.0001
            with dest.with_suffix('.tmp').open('wb') as f:
                np.save(f, d)
            dest.with_suffix('.tmp').replace(dest)
            r = dict(subject=row['subject'], visit=visit, path=str(dest), sha256=f_sha(dest), shape=list(d.shape), retained_frames=[len(y) for y in ys], bandpass_hz=[0.01, 0.1], source_runs=[row['visits'][actual]['runs'][f'rfMRI_REST{day}_{pe}'] for pe in ['LR', 'RL']], no_other_visit_in_fit=True)
            f_save(marker, r)
            records.append(r)
            print('INPUT_COMPLETE', visit, index + 1, row['subject'], flush=True)
    f_save(OUT / 'inputs_manifest.json', dict(records=records, geometry_sha256=f_sha(OUT / 'geometry.npz')))
    f_save(OUT / 'prepare_status.json', dict(state='complete', completed=108, subjects=36, visits=3, pid=os.getpid(), updated_unix=time.time()))

def f_wang():
    """Existing public-comparison adaptation, explicitly not original Wang2015."""
    import torch
    torch.set_num_threads(4)
    assert f_read(OUT / 'prepare_status.json')['state'] == 'complete'
    out = OUT / 'Wang_style_adapted_diagnostic'
    out.mkdir(exist_ok=True)
    lock = (out / 'workflow.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    initial = np.load(OUT / 'geometry.npz')['initial']
    records = f_read(OUT / 'inputs_manifest.json')['records']
    reference = W / 'references/wang2015/Wang_parcellation.m'
    assert reference.exists()
    protocol = dict(method='Wang-style public-comparison adaptation (Salehi comparison code)', original_paper='https://doi.org/10.1038/nn.4164', comparison_source='https://github.com/YaleMRRC/Network-Parcellation-Rest/blob/master/Wang_parcellation.m', comparison_source_sha256=f_sha(reference), iterations=10, confidence_ratio_threshold=1.1, core_previous_reference_weights=[2, 1], empty_core_policy='retain previous reference', missing_original_components=['interindividual variability weighting', 'temporal SNR weighting', 'original atlas-reference/iteration weighting'], common_input_protocol_sha256=f_sha(OUT / 'protocol.json'), initialization='same connected CoBRA prior as GPIP diagnostic;50 left parcels', official_author_code_used=False, validated_as_original_equivalent=False, publication_ranking_ready=False, no_hidden_repairs='Do not force missing parcels back into maps; record actual occupied parcel count.', implementation_sha256=f_sha(Path(__file__)))
    p = out / 'protocol.json'
    if p.exists():
        assert f_read(p) == protocol
    else:
        f_save(p, protocol)

    @torch.no_grad()
    def assign(data, labels, iterations=10):
        y = torch.from_numpy(np.array(data.T, copy=True))
        initial_t = torch.as_tensor(labels, dtype=torch.long)
        ref = torch.stack([y[:, initial_t == k].mean(1) for k in range(50)], 1)
        previous = initial_t.clone()
        history = []
        for i in range(iterations):
            normalized = (ref - ref.mean(0)) / ref.std(0, unbiased=False).clamp_min(1e-06)
            corr = y.T @ normalized / len(y)
            best, idx = torch.topk(corr, 2, dim=1)
            v = idx[:, 0]
            confidence = best[:, 0] / best[:, 1]
            confidence = torch.where(best[:, 1] == 0, torch.where(best[:, 0] > 0, torch.inf, 0.0), confidence)
            confident = (confidence > 1.1) & (best[:, 0] > 0)
            core = ref.clone()
            missing = []
            for k in range(50):
                members = (v == k) & confident
                if members.any():
                    core[:, k] = y[:, members].mean(1)
                else:
                    missing.append(k)
            ref = (2 * core + ref) / 3
            history.append(dict(iteration=i + 1, unchanged=float((v == previous).float().mean()), occupied=int(v.unique().numel()), confident_vertices=int(confident.sum()), empty_core_retained=missing))
            previous = v
        assert torch.isfinite(ref).all()
        return (v.numpy().astype('int32'), history)
    rng = np.random.default_rng(20261008)
    a = rng.normal(size=(100, 60)).astype('float32')
    a = (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)
    init = np.repeat(np.arange(50), 2)
    ref = np.stack([a[init == k].mean(0) for k in range(50)], 1)
    ref = (ref - ref.mean(0)) / ref.std(0)
    expected = (a @ ref / 60).argmax(1)
    observed, _ = assign(a, init, 1)
    assert np.array_equal(expected, observed)
    f_save(out / 'implementation_check.json', dict(first_step_matches_independent_dense_Pearson=True, original_method_equivalence=False))
    results = []
    for j, r in enumerate(records):
        p = out / f"maps/{r['visit']}/{r['subject']}.npy"
        marker = p.with_suffix('.json')
        p.parent.mkdir(parents=True, exist_ok=True)
        if marker.exists():
            results.append(f_read(marker))
            continue
        f_save(out / 'status.json', dict(state='running_adapted_diagnostic', index=j + 1, total=108, subject=r['subject'], visit=r['visit'], pid=os.getpid(), updated_unix=time.time(), publication_ranking_ready=False))
        assert f_sha(Path(r['path'])) == r['sha256']
        v, h = assign(np.load(r['path']), initial)
        np.save(p, v)
        result = dict(subject=r['subject'], visit=r['visit'], occupied_parcels=int(len(np.unique(v))), nominal_left_K=50, iterations=h, map_sha256=f_sha(p), publication_ranking_ready=False)
        f_save(marker, result)
        results.append(result)
        print('WANG_STYLE_DIAGNOSTIC', j + 1, r['visit'], r['subject'], result['occupied_parcels'], flush=True)
    f_save(out / 'complete.json', dict(state='complete_adapted_diagnostic', records=results, publication_ranking_ready=False, completed_unix=time.time()))
    f_save(out / 'status.json', dict(state='complete_adapted_diagnostic', subjects=36, visits=3, publication_ranking_ready=False, updated_unix=time.time()))
