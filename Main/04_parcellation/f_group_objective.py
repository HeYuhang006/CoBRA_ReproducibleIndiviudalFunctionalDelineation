from f_runtime import f_work_root
"""Exact raw training-group FC objective; no validation data enter prior fitting."""
import numpy as np
import torch
from numba import njit

def f_fit_term(norm, diagonal, count):
    return float(np.sum((norm - diagonal) / (count - 1)))

def f_direct_energy(labels, initial, fc, cost, edges, perimeter):
    k = cost.shape[1]
    counts = np.bincount(labels, minlength=k)
    reference = np.bincount(initial, minlength=k)
    norms = np.zeros(k)
    diag = np.zeros(k)
    for a in range(k):
        ids = np.flatnonzero(labels == a)
        assert len(ids) >= 2
        norms[a] = np.sum(fc[np.ix_(ids, ids)], dtype=np.float64)
        diag[a] = np.diag(fc)[ids].sum(dtype=np.float64)
    fit = f_fit_term(norms, diag, counts)
    energy = -fit + 0.01 * cost[np.arange(len(labels)), labels].sum() + 0.005 * np.sum(labels != initial) + perimeter * np.sum(labels[edges[:, 0]] != labels[edges[:, 1]]) + 0.01 * np.sum((counts - reference) ** 2 / reference)
    return (float(energy), fit / len(labels))

@njit(cache=True)
def f_optimize(initial, fc, affinities, cost, indptr, indices, seeds, orders, minimum, maximum, perimeter, maximum_drift):
    labels = initial.copy()
    n, k = affinities.shape
    affinity = affinities.copy()
    counts = np.zeros(k, np.int64)
    norms = np.zeros(k, np.float64)
    diagonal = np.zeros(k, np.float64)
    pins = np.zeros(n, np.bool_)
    for seed in seeds:
        pins[seed] = True
    for v in range(n):
        a = labels[v]
        counts[a] += 1
        norms[a] += affinity[v, a]
        diagonal[a] += fc[v, v]
    reference = counts.copy()
    moves = np.zeros(len(orders), np.int64)
    drift = 0
    for sweep in range(len(orders)):
        for v in orders[sweep]:
            if pins[v]:
                continue
            old = labels[v]
            if counts[old] <= minimum[old]:
                continue
            nb = indices[indptr[v]:indptr[v + 1]]
            candidates = np.unique(labels[nb])
            if len(candidates) == 1:
                continue
            same = nb[labels[nb] == old]
            if len(same) == 0:
                continue
            seen = np.zeros(len(same), np.bool_)
            seen[0] = True
            for iteration in range(len(same)):
                for i in range(len(same)):
                    if seen[i]:
                        u = same[i]
                        for q in range(indptr[u], indptr[u + 1]):
                            for j in range(len(same)):
                                if indices[q] == same[j]:
                                    seen[j] = True
            if not np.all(seen):
                continue
            removed = norms[old] - 2 * affinity[v, old] + fc[v, v]
            oldfit = (norms[old] - diagonal[old]) / (counts[old] - 1)
            removedfit = (removed - diagonal[old] + fc[v, v]) / (counts[old] - 2)
            oldcuts = np.sum(labels[nb] != old)
            best, bestgain, bestnorm = (old, 1e-07, 0.0)
            for target in candidates:
                if target == old or counts[target] >= maximum[target]:
                    continue
                dd = int(target != initial[v]) - int(old != initial[v])
                if drift + dd > maximum_drift:
                    continue
                added = norms[target] + 2 * affinity[v, target] + fc[v, v]
                newfit = (added - diagonal[target] - fc[v, v]) / counts[target]
                previousfit = (norms[target] - diagonal[target]) / (counts[target] - 1)
                gain = removedfit - oldfit + newfit - previousfit
                gain -= 0.01 * (cost[v, target] - cost[v, old])
                gain -= 0.005 * dd
                gain -= perimeter * (np.sum(labels[nb] != target) - oldcuts)
                gain -= 0.01 * ((-2 * (counts[old] - reference[old]) + 1) / reference[old] + (2 * (counts[target] - reference[target]) + 1) / reference[target])
                if gain > bestgain:
                    best, bestgain, bestnorm = (target, gain, added)
            if best != old:
                for u in range(n):
                    affinity[u, old] -= fc[v, u]
                    affinity[u, best] += fc[v, u]
                drift += int(best != initial[v]) - int(old != initial[v])
                labels[v] = best
                counts[old] -= 1
                counts[best] += 1
                norms[old], norms[best] = (removed, bestnorm)
                diagonal[old] -= fc[v, v]
                diagonal[best] += fc[v, v]
                moves[sweep] += 1
        if moves[sweep] == 0:
            break
    return (labels, moves, affinity)

def f_affinities_gpu(fc, labels, k):
    onehot = torch.nn.functional.one_hot(torch.from_numpy(labels.astype(np.int64)).cuda(), k).double()
    result = np.empty((len(labels), k), np.float64)
    with torch.no_grad():
        for a in range(0, len(labels), 512):
            b = min(a + 512, len(labels))
            result[a:b] = (torch.from_numpy(fc[a:b]).cuda().double() @ onehot).cpu().numpy()
    return result
