from f_runtime import f_work_root
"""Absolute-agreement edge ICC and paired participant bootstrap."""
import numpy as np

def f_icc_from_sums(sx, sy, sxx, syy, sxy, n):
    grand = (sx + sy) / (2 * n)
    ss_subject = 0.5 * (sxx + syy + 2 * sxy) - 2 * n * grand ** 2
    ss_visit = (sx - sy) ** 2 / (2 * n)
    ss_error = sxx + syy - 2 * n * grand ** 2 - ss_subject - ss_visit
    ms_subject = ss_subject / (n - 1)
    ms_error = ss_error / (n - 1)
    denominator = ms_subject + ms_error + 2 * (ss_visit - ms_error) / n
    return np.divide(ms_subject - ms_error, denominator, out=np.full_like(denominator, np.nan), where=np.abs(denominator) > 1e-12)

def f_edge_icc(x, y):
    x = np.asarray(x, dtype='float64')
    y = np.asarray(y, dtype='float64')
    assert x.shape == y.shape and x.ndim == 2 and (len(x) > 2)
    return f_icc_from_sums(x.sum(0), y.sum(0), (x * x).sum(0), (y * y).sum(0), (x * y).sum(0), len(x))

def f_row_correlation(x, y):
    a = x - x.mean(1, keepdims=True)
    b = y - y.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))

def f_summary(x, y):
    x = np.asarray(x, dtype='float64')
    y = np.asarray(y, dtype='float64')
    i = f_edge_icc(x, y)
    ax = x - x.mean(1, keepdims=True)
    by = y - y.mean(1, keepdims=True)
    sim = ax / np.linalg.norm(ax, axis=1, keepdims=True) @ (by / np.linalg.norm(by, axis=1, keepdims=True)).T
    n = len(x)
    within = sim.diagonal()
    between = sim[~np.eye(n, dtype=bool)].mean()
    concordance = 2 * (ax * by).mean(1) / ((ax * ax).mean(1) + (by * by).mean(1) + (x.mean(1) - y.mean(1)) ** 2)
    return dict(n_subjects=n, n_edges=x.shape[1], median_edge_ICC_A1=float(np.nanmedian(i)), undefined_edges=int(np.isnan(i).sum()), fraction_ICC_above_04=float(np.mean(i[np.isfinite(i)] > 0.4)), negative_ICC_fraction=float(np.mean(i[np.isfinite(i)] < 0)), mean_within_person_similarity=float(within.mean()), mean_MSE=float(((x - y) ** 2).mean()), mean_Lin_concordance=float(concordance.mean()), differential_identifiability=float(within.mean() - between), identification_forward=float((sim.argmax(1) == np.arange(n)).mean()), identification_reverse=float((sim.argmax(0) == np.arange(n)).mean()))

def f_paired_bootstrap(methods, reference, draws=5000, seed=20261007):
    """Resample subjects, recomputing edge ICC; never treat edges as independent."""
    n = len(methods[reference][0])
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(n, np.full(n, 1 / n), size=draws).astype('float64')
    medians = {}
    for name, (x, y) in methods.items():
        x = np.asarray(x, dtype='float64')
        y = np.asarray(y, dtype='float64')
        inputs = np.c_[x, y, x * x, y * y, x * y]
        p = x.shape[1]
        values = []
        for a in range(0, draws, 16):
            sums = weights[a:a + 16] @ inputs
            i = f_icc_from_sums(*[sums[:, j * p:(j + 1) * p] for j in range(5)], n)
            values.extend(np.nanmedian(i, axis=1))
        medians[name] = np.array(values)
    result = {}
    for name in methods:
        if name == reference:
            continue
        d = medians[name] - medians[reference]
        point = np.nanmedian(f_edge_icc(*methods[name])) - np.nanmedian(f_edge_icc(*methods[reference]))
        result[name] = dict(median_ICC_difference=float(point), ci95=np.quantile(d, [0.025, 0.975]).tolist(), draws=draws, resampling='participants paired across methods; family metadata unavailable')
    return result

def f_paired_permutation(a, b, draws=5000, seed=20261007):
    """Swap each participant's two entire method connectome pairs together."""
    x, y = (np.asarray(z, dtype='float64') for z in a)
    u, v = (np.asarray(z, dtype='float64') for z in b)
    n = len(x)
    rng = np.random.default_rng(seed)
    observed = float(np.nanmedian(f_edge_icc(x, y)) - np.nanmedian(f_edge_icc(u, v)))
    hits = 0
    A = np.c_[x, y, x * x, y * y, x * y]
    B = np.c_[u, v, u * u, v * v, u * v]
    delta = A - B
    sa = A.sum(0)
    sb = B.sum(0)
    p = x.shape[1]
    for start in range(0, draws, 16):
        masks = rng.integers(0, 2, (min(16, draws - start), n)).astype('float64')
        changes = masks @ delta
        first = sb + changes
        second = sa - changes
        ia = f_icc_from_sums(*[first[:, j * p:(j + 1) * p] for j in range(5)], n)
        ib = f_icc_from_sums(*[second[:, j * p:(j + 1) * p] for j in range(5)], n)
        d = np.nanmedian(ia, axis=1) - np.nanmedian(ib, axis=1)
        hits += int((np.abs(d) >= abs(observed)).sum())
    return dict(difference=observed, two_sided_p=float((hits + 1) / (draws + 1)), draws=draws)

def f_holm(values):
    keys = sorted(values, key=values.get)
    m = len(keys)
    adjusted = {}
    current = 0.0
    for rank, key in enumerate(keys):
        current = max(current, min(1.0, (m - rank) * values[key]))
        adjusted[key] = current
    return adjusted
