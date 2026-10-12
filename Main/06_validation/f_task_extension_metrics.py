from f_runtime import f_work_root
"""Independent task metrics; no model fitting or atlas modification."""
import numpy as np
from scipy import sparse

def f_assignment(labels, k):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.array_equal(np.unique(labels), np.arange(k)):
        raise ValueError('Every fixed-K parcel must be present and zero-indexed')
    return sparse.csr_matrix((np.ones(len(labels)), (labels, np.arange(len(labels)))), shape=(k, len(labels)))

def f_task_r2(profiles, area, operator):
    """One area-weighted piecewise-constant R2 per contrast, on full cortex."""
    profiles = np.asarray(profiles, dtype=np.float64)
    area = np.asarray(area, dtype=np.float64)
    if profiles.ndim != 2 or profiles.shape[0] != len(area) or operator.shape[1] != len(area):
        raise ValueError('Task, area, and atlas dimensions differ')
    if not np.isfinite(profiles).all() or not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError('Invalid task responses or surface areas')
    centered = profiles - np.average(profiles, axis=0, weights=area)
    weighted = centered * area[:, None]
    total = np.sum(weighted * centered, axis=0)
    if np.any(total <= 1e-12):
        raise ValueError('Degenerate task contrast')
    mass = operator @ area
    sums = operator @ weighted
    explained = np.sum(sums * sums / mass[:, None], axis=0)
    result = explained / total
    if np.any(result < -1e-09) or np.any(result > 1 + 1e-09):
        raise ValueError('Task R2 outside its analytical range')
    return np.clip(result, 0, 1)

def f_dcbc_reduce(counts, covariance_sum, sd_product_sum):
    """Official DCBC ratio-of-means, with precision weighting of distance bins.

    Axis 0: between/within. Counts include zero-profile vertices; they are not
    removed on a method-dependent basis. Undefined bins receive zero weight.
    """
    counts = np.asarray(counts, dtype=np.float64)
    covariance_sum = np.asarray(covariance_sum, dtype=np.float64)
    sd_product_sum = np.asarray(sd_product_sum, dtype=np.float64)
    valid = (counts > 0).all(axis=0) & (sd_product_sum > 0).all(axis=0)
    if not valid.any():
        raise ValueError('No distance bins have both valid within/between profiles')
    weight = np.zeros(counts.shape[1])
    weight[valid] = 1 / (1 / counts[0, valid] + 1 / counts[1, valid])
    weight /= weight.sum()
    correlation = np.zeros_like(counts)
    np.divide(covariance_sum, sd_product_sum, out=correlation, where=sd_product_sum > 0)
    value = float(weight @ (correlation[1] - correlation[0]))
    return dict(DCBC=value, counts=counts.astype(np.int64).tolist(), correlation=correlation.tolist(), weight=weight.tolist(), valid_bins=np.flatnonzero(valid).tolist())

def f_task_dcbc(profiles, labels_by_method, pairs, bins=35, chunk=200000):
    """Exact all-unordered-pair DCBC, without allocating a dense FC matrix."""
    x = np.asarray(profiles, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] < 3 or (not np.isfinite(x).all()):
        raise ValueError('Task DCBC requires finite multi-contrast vertex profiles')
    x = x - x.mean(axis=1, keepdims=True)
    sd = np.sqrt(np.sum(x * x, axis=1) / (x.shape[1] - 1))
    if not (sd > 0).any():
        raise ValueError('All task profiles have zero variance')
    accum = {name: [np.zeros((2, bins), dtype=np.int64), np.zeros((2, bins)), np.zeros((2, bins))] for name in labels_by_method}
    for start in range(0, len(pairs['row']), chunk):
        stop = start + chunk
        rr = pairs['row'][start:stop]
        cc = pairs['col'][start:stop]
        bb = pairs['bin'][start:stop].astype(np.int64)
        covariance = np.einsum('ij,ij->i', x[rr], x[cc]) / (x.shape[1] - 1)
        products = sd[rr] * sd[cc]
        for name, labels in labels_by_method.items():
            category = bb + bins * (labels[rr] == labels[cc])
            accum[name][0] += np.bincount(category, minlength=2 * bins).reshape(2, bins)
            accum[name][1] += np.bincount(category, weights=covariance, minlength=2 * bins).reshape(2, bins)
            accum[name][2] += np.bincount(category, weights=products, minlength=2 * bins).reshape(2, bins)
    return {name: f_dcbc_reduce(*values) for name, values in accum.items()}

def f_paired_difference(a, b, draws=5000, seed=20261007):
    delta = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if delta.ndim != 1 or len(delta) < 3 or (not np.isfinite(delta).all()):
        raise ValueError('Need at least three finite participant-paired observations')
    rng = np.random.default_rng(seed)
    bootstrap = delta[rng.integers(len(delta), size=(draws, len(delta)))].mean(axis=1)
    null = (rng.choice([-1.0, 1.0], size=(draws, len(delta))) * delta).mean(axis=1)
    return dict(n=len(delta), mean_difference=float(delta.mean()), ci95=np.quantile(bootstrap, [0.025, 0.975]).tolist(), two_sided_p=float((1 + np.sum(np.abs(null) >= abs(delta.mean()))) / (draws + 1)), draws=draws)

def f_specificity(matrix, own_columns, draws=5000, seed=20261007):
    """Own minus all-other R2; one global donor permutation per null draw.

    Global permutations preserve donor reuse instead of treating 39 donors or
    parcels as independent subjects. Target-bootstrap CI conditions on this
    fixed donor pool; it is not an unconditional population bootstrap.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    own_columns = np.asarray(own_columns, dtype=np.int64)
    if matrix.ndim != 2 or len(own_columns) != len(matrix) or len(matrix) < 3:
        raise ValueError('Invalid subject-by-donor matrix')
    if not np.isfinite(matrix).all() or np.any(own_columns < 0) or np.any(own_columns >= matrix.shape[1]):
        raise ValueError('Missing donor values or invalid own-donor correspondence')
    if len(np.unique(own_columns)) != len(own_columns):
        raise ValueError('Targets must have distinct own donors')
    own = matrix[np.arange(len(matrix)), own_columns]
    others = (matrix.sum(axis=1) - own) / (matrix.shape[1] - 1)
    margins = own - others
    rng = np.random.default_rng(seed)
    bootstrap = margins[rng.integers(len(margins), size=(draws, len(margins)))].mean(axis=1)
    permutations = np.stack([rng.permutation(matrix.shape[1]) for _ in range(draws)])
    shuffled = matrix[np.arange(len(matrix))[None, :], permutations[:, own_columns]]
    null = ((matrix.shape[1] * shuffled - matrix.sum(axis=1)[None, :]) / (matrix.shape[1] - 1)).mean(axis=1)
    return dict(n=len(matrix), donor_count=matrix.shape[1], own_mean_R2=float(own.mean()), others_mean_R2=float(others.mean()), mean_difference=float(margins.mean()), ci95=np.quantile(bootstrap, [0.025, 0.975]).tolist(), two_sided_p=float((1 + np.sum(np.abs(null) >= abs(margins.mean()))) / (draws + 1)), margins=margins.tolist(), draws=draws, inference='global donor-label permutation; target-bootstrap CI conditional on fixed donor pool')

def f_holm(pvalues):
    corrected = {}
    previous = 0.0
    for i, name in enumerate(sorted(pvalues, key=pvalues.get)):
        previous = max(previous, min(1.0, (len(pvalues) - i) * pvalues[name]))
        corrected[name] = previous
    return corrected
