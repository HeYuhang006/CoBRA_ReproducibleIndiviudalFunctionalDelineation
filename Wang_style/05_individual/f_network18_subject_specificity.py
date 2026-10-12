from f_runtime import f_work_root
"""Cross-visit subject specificity; participant resampling, never pair pseudoreplication."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
from scipy.optimize import linear_sum_assignment

def f_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)

def f_agreement(a, b, k):
    a = np.asarray(a, dtype='int64')
    b = np.asarray(b, dtype='int64')
    assert a.shape == b.shape and a.ndim == 1
    assert a.min() >= 0 and b.min() >= 0 and (a.max() < k) and (b.max() < k)
    c = np.bincount(a * k + b, minlength=k * k).reshape(k, k).astype('float64')
    na, nb = (c.sum(1), c.sum(0))
    pairs = lambda x: np.sum(x * (x - 1) / 2)
    observed = pairs(c)
    left = pairs(na)
    right = pairs(nb)
    total = len(a) * (len(a) - 1) / 2
    expected = left * right / total
    maximum = (left + right) / 2
    ari = 1.0 if maximum == expected else (observed - expected) / (maximum - expected)
    den = na[:, None] + nb[None, :]
    dice = np.divide(2 * c, den, out=np.zeros_like(c), where=den > 0)
    i, j = linear_sum_assignment(-dice)
    return dict(ARI=float(ari), Parcel_Dice=float(dice[i, j].mean()))

def f_summaries(matrix, weights, permutations):
    n = len(matrix)
    mask = ~np.eye(n, dtype=bool)
    diagonal = np.diag(matrix)
    intra = float(diagonal.mean())
    inter = float(matrix[mask].mean())
    intra_boot = weights @ diagonal / n
    inter_boot = np.einsum('bi,ij,bj->b', weights, matrix * mask, weights)
    denominator = n * n - (weights * weights).sum(1)
    assert (denominator > 0).all()
    inter_boot /= denominator
    gap_boot = intra_boot - inter_boot
    null = matrix[np.arange(n)[None, :], permutations].mean(1)
    p = (1 + np.sum(null >= intra - 1e-12)) / (1 + len(null))
    top = matrix.max(1, keepdims=True)
    ties = np.isclose(matrix, top, atol=1e-12, rtol=0)
    identification = float(np.mean(ties.diagonal() / ties.sum(1)))
    ci = lambda x: np.quantile(x, [0.025, 0.975]).tolist()
    return (dict(intra_similarity_mean=intra, inter_similarity_mean=inter, inter_dissimilarity_mean=1 - inter, specificity_gap=intra - inter, intra_ci95=ci(intra_boot), inter_ci95=ci(inter_boot), specificity_gap_ci95=ci(gap_boot), identity_permutation_p=float(p), identification_fractional_ties=identification, chance_identification=1 / n), gap_boot)

def f_run(manifest, out, bootstrap=5000, nperm=10000):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(manifest).read_text())
    subjects = cfg['subjects']
    n = len(subjects)
    import fcntl
    lock = (out / 'workflow.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('ALREADY_RUNNING')
        return
    if (out / 'summary.json').exists():
        assert json.loads((out / 'summary.json').read_text())['manifest'] == cfg
        print('ALREADY_COMPLETE')
        return
    assert len(set(subjects)) == n and n >= 3
    rng = np.random.default_rng(20261008)
    weights = rng.multinomial(n, np.full(n, 1 / n), size=bootstrap)
    permutations = np.array([rng.permutation(n) for _ in range(nperm)])
    pairs = [('Day1', 'Day2'), ('Day1', 'Retest')]
    results = {}
    boots = {}
    tests = []
    for name, spec in cfg['methods'].items():
        k = spec['K']
        maps = {}
        for visit in ['Day1', 'Day2', 'Retest']:
            paths = spec['maps'][visit]
            assert len(paths) == n
            maps[visit] = [np.load(p) for p in paths]
            if not spec.get('allow_missing', False):
                assert all((len(np.unique(y)) == k for y in maps[visit])), (name, visit, 'occupied K mismatch')
        results[name] = {}
        boots[name] = {}
        for first, second in pairs:
            key = first + '_' + second
            path = out / f'matrices/{name}_{key}.npz'
            if path.exists():
                saved = np.load(path)
                matrices = {m: saved[m] for m in ['ARI', 'Parcel_Dice']}
            else:
                matrices = {m: np.zeros((n, n)) for m in ['ARI', 'Parcel_Dice']}
                for i, a in enumerate(maps[first]):
                    for j, b in enumerate(maps[second]):
                        for m, value in f_agreement(a, b, k).items():
                            matrices[m][i, j] = value
                path.parent.mkdir(exist_ok=True)
                np.savez_compressed(path, **matrices)
            results[name][key] = {}
            boots[name][key] = {}
            for metric, matrix in matrices.items():
                record, samples = f_summaries(matrix, weights, permutations)
                results[name][key][metric] = record
                boots[name][key][metric] = samples
                if not spec.get('fixed_reference', False):
                    tests.append((name, key, metric, record['identity_permutation_p']))
            f_save(out / 'status.json', dict(state='running', method=name, pair=key, updated_unix=time.time()))
            print('SPECIFICITY', name, key, results[name][key]['ARI']['specificity_gap'], flush=True)
    order = np.argsort([r[3] for r in tests])
    running = 0.0
    for rank, idx in enumerate(order):
        name, key, metric, p = tests[idx]
        running = max(running, (len(tests) - rank) * p)
        results[name][key][metric]['identity_permutation_p_holm'] = min(1.0, running)
    comparisons = {}
    if 'CoBRA' in results:
        for name in results:
            if name == 'CoBRA' or cfg['methods'][name].get('fixed_reference', False):
                continue
            comparisons[name] = {}
            for key in boots[name]:
                comparisons[name][key] = {}
                for metric in boots[name][key]:
                    delta = boots['CoBRA'][key][metric] - boots[name][key][metric]
                    comparisons[name][key][metric] = dict(mean_gap_difference=results['CoBRA'][key][metric]['specificity_gap'] - results[name][key][metric]['specificity_gap'], participant_bootstrap_ci95=np.quantile(delta, [0.025, 0.975]).tolist())
    f_save(out / 'summary.json', dict(state='complete', n_subjects=n, methods=results, CoBRA_minus_comparator_specificity_gap=comparisons, manifest=cfg, statistics=dict(bootstrap=bootstrap, identity_permutations=nperm, holm_family=len(tests), resampling='Participant multinomial weights shared by all methods. Off-diagonal original identities only, including duplicated bootstrap participants; exclude all same-identity pairs.', null='Random reassignment of second-visit identities, preserving complete similarity matrix. One-sided test of identity correspondence; conditional on this cohort, family metadata unavailable.', identification='Row-max cross-visit identification with fractional tie credit.', inter='Ordered distinct-subject cross-visit pairs, never treated as independent participants.', Dice='Hungarian matched macro parcel Dice; differs from Wang original fixed-network vertex agreement.', interpretation='Positive intra-minus-inter gap is evidence of repeatable individual specificity, not proof of biological or clinical validity.'), completed_unix=time.time()))
    f_save(out / 'status.json', dict(state='complete', updated_unix=time.time()))

def f_selftest():
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(1)
    for k in [3, 18, 50]:
        a = rng.integers(k, size=1000)
        b = rng.integers(k, size=1000)
        assert abs(f_agreement(a, b, k)['ARI'] - adjusted_rand_score(a, b)) < 1e-12
        perm = rng.permutation(k)
        assert f_agreement(a, perm[a], k) == dict(ARI=1.0, Parcel_Dice=1.0)
    weights = np.ones((10, 6), dtype=int)
    perms = np.array([rng.permutation(6) for _ in range(20)])
    x, _ = f_summaries(np.ones((6, 6)), weights, perms)
    assert x['specificity_gap'] == 0 and abs(x['identification_fractional_ties'] - 1 / 6) < 1e-12
    x, _ = f_summaries(np.eye(6), weights, perms)
    assert x['specificity_gap'] == 1 and x['identification_fractional_ties'] == 1
    print('SELFTEST_PASS')
