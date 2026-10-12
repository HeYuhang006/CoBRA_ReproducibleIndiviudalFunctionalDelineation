from f_runtime import f_work_root
"""Mesh connectivity and exact sequential topology-preserving boundary moves."""
import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import heapq
SEED = 20260930

def f_graph_data(n, edges):
    rc = np.r_[edges, edges[:, ::-1]]
    return csr_matrix((np.ones(len(rc), dtype='float32'), (rc[:, 0], rc[:, 1])), shape=(n, n))

def f_decode(initial, y, cost, adj, seeds, depth, edges, sweeps=20, hard_connectivity=True):
    k = cost.shape[1]
    assert len(seeds) == k and np.array_equal(np.unique(initial), np.arange(k))
    f = np.ascontiguousarray(y.T, dtype='float32').copy()
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    counts = np.bincount(initial, minlength=k)
    assert counts.min() > 2
    minimum = np.minimum(counts, np.maximum(20, (0.25 * counts).astype('int64')))
    maximum = np.maximum(counts, np.full(k, int(np.ceil(4 * len(initial) / k))))
    orders = np.array([np.random.default_rng(SEED + i).permutation(len(initial)) for i in range(sweeps)], dtype='int32')
    allowed = depth <= 12
    ew = np.ones(len(edges), dtype='float32')
    before = f_energy(initial, initial, f, cost, edges, ew, 0.02)
    labels, history = f_boundary_optimize(initial, f, cost, adj.indptr, adj.indices, adj.data, seeds, orders, allowed, minimum, maximum, 0.02, hard_connectivity)
    after = f_energy(labels, initial, f, cost, edges, ew, 0.02)
    assert after <= before + 0.002 and np.array_equal(labels[seeds], np.arange(k)) and np.array_equal(labels[~allowed], initial[~allowed])
    cc = [int(connected_components(adj[labels == a][:, labels == a], directed=False)[0]) for a in range(k)]
    if hard_connectivity:
        assert cc == [1] * k
    sizes = np.bincount(labels, minlength=k)
    assert (sizes >= minimum).all() and (sizes <= maximum).all()
    return (labels.astype('int16'), dict(energy_before=before, energy_after=after, moves_by_sweep=history.tolist(), converged=bool(history[-1] == 0), changed_fraction=float(np.mean(labels != initial)), components=cc, counts=sizes.tolist(), maximum_boundary_depth=12, size_maximum_rule='max(original count,4*N/K)', hard_connectivity=bool(hard_connectivity)))

@njit(cache=True)
def f_functional_delta(normold, normnew, removed, added, nold, nnew):
    return (removed - (nold - 1)) / (nold - 2) - (normold - nold) / (nold - 1) + (added - (nnew + 1)) / nnew - (normnew - nnew) / (nnew - 1)

def f_boundary_depth(labels, adj):
    src = np.repeat(np.arange(len(labels)), np.diff(adj.indptr))
    border = np.zeros(len(labels), bool)
    border[src[labels[src] != labels[adj.indices]]] = True
    depth = np.full(len(labels), len(labels), np.int32)
    depth[border] = 0
    frontier = np.flatnonzero(border)
    d = 0
    while len(frontier):
        nxt = []
        for v in frontier:
            for u in adj.indices[adj.indptr[v]:adj.indptr[v + 1]]:
                if depth[u] == len(labels):
                    depth[u] = d + 1
                    nxt.append(u)
        frontier = np.asarray(nxt, np.int32)
        d += 1
    return depth

@njit(cache=True)
def f_boundary_optimize(initial, features, cost, indptr, indices, weights, seeds, orders, allowed, minimum, maximum, neural_weight, hard_connectivity=True):
    labels = initial.copy()
    n, d = features.shape
    k = len(seeds)
    counts = np.zeros(k, np.int64)
    sums = np.zeros((k, d), np.float64)
    pins = np.zeros(n, np.bool_)
    for v in seeds:
        pins[v] = True
    for v in range(n):
        counts[labels[v]] += 1
        for t in range(d):
            sums[labels[v], t] += features[v, t]
    reference = counts.copy()
    norms = np.zeros(k, np.float64)
    for a in range(k):
        norms[a] = np.dot(sums[a], sums[a])
    history = np.zeros(len(orders), np.int64)
    for sweep in range(len(orders)):
        for v in orders[sweep]:
            if pins[v] or not allowed[v]:
                continue
            old = labels[v]
            if counts[old] <= minimum[old]:
                continue
            start, end = (indptr[v], indptr[v + 1])
            nb = indices[start:end]
            candidate = np.unique(labels[nb])
            if len(candidate) == 1 and (hard_connectivity or candidate[0] == old):
                continue
            if hard_connectivity:
                same = nb[labels[nb] == old]
                if len(same) == 0:
                    continue
                seen = np.zeros(len(same), np.bool_)
                seen[0] = True
                for iteration in range(len(same)):
                    for i in range(len(same)):
                        if not seen[i]:
                            continue
                        u = same[i]
                        for q in range(indptr[u], indptr[u + 1]):
                            for j in range(len(same)):
                                if same[j] == indices[q]:
                                    seen[j] = True
                if not np.all(seen):
                    continue
            dotold, xnorm = (0.0, 0.0)
            for t in range(d):
                dotold += sums[old, t] * features[v, t]
                xnorm += features[v, t] * features[v, t]
            removed = norms[old] - 2 * dotold + xnorm
            oldcut = 0.0
            for q in range(start, end):
                if labels[indices[q]] != old:
                    oldcut += weights[q]
            best, bestgain, bestnorm = (old, 1e-09, 0.0)
            for target in candidate:
                if target == old or counts[target] >= maximum[target]:
                    continue
                dotnew = 0.0
                for t in range(d):
                    dotnew += sums[target, t] * features[v, t]
                newnorm = norms[target] + 2 * dotnew + xnorm
                gain = f_functional_delta(norms[old], norms[target], removed, newnorm, counts[old], counts[target])
                gain -= neural_weight * (cost[v, target] - cost[v, old])
                gain -= 0.025 * ((target != initial[v]) - (old != initial[v]))
                newcut = 0.0
                for q in range(start, end):
                    if labels[indices[q]] != target:
                        newcut += weights[q]
                gain -= 0.025 * (newcut - oldcut)
                gain -= 0.01 * ((-2 * (counts[old] - reference[old]) + 1) / reference[old] + (2 * (counts[target] - reference[target]) + 1) / reference[target])
                if gain > bestgain:
                    best, bestgain, bestnorm = (target, gain, newnorm)
            if best != old:
                for t in range(d):
                    sums[old, t] -= features[v, t]
                    sums[best, t] += features[v, t]
                counts[old] -= 1
                counts[best] += 1
                norms[old], norms[best] = (removed, bestnorm)
                labels[v] = best
                history[sweep] += 1
        if history[sweep] == 0:
            break
    return (labels, history)

def f_energy(labels, initial, f, cost, edges, ew, neural_weight):
    counts = np.bincount(labels, minlength=cost.shape[1])
    reference = np.bincount(initial, minlength=cost.shape[1])
    assert np.all(counts >= 2)
    sums = np.zeros((cost.shape[1], f.shape[1]), np.float64)
    np.add.at(sums, labels, f)
    diag = np.bincount(labels, weights=(f.astype(np.float64) ** 2).sum(1), minlength=cost.shape[1])
    fit = np.sum(((sums * sums).sum(1) - diag) / (counts - 1))
    return float(-fit + neural_weight * cost[np.arange(len(labels)), labels].sum() + 0.025 * np.sum(labels != initial) + 0.025 * np.sum(ew * (labels[edges[:, 0]] != labels[edges[:, 1]])) + 0.01 * np.sum((counts - reference) ** 2 / reference))

def f_connected_assignment(cost, adj, seeds, step, edge_weights):
    """Every newly settled vertex attaches to a settled vertex of the same label."""
    n, k = cost.shape
    assert connected_components(adj, directed=False)[0] == 1 and len(np.unique(seeds)) == k
    excess = cost - cost.min(1, keepdims=True)
    labels = np.full(n, -1, dtype='int16')
    dist = np.full(n, np.inf)
    done = np.zeros(n, bool)
    parent = np.full(n, -1, dtype='int32')
    heap = []
    for label, v in enumerate(seeds):
        dist[v] = 0
        labels[v] = label
        heapq.heappush(heap, (0.0, int(v), label))
    while heap:
        d, v, label = heapq.heappop(heap)
        if done[v] or d != dist[v] or label != labels[v]:
            continue
        done[v] = True
        for ix in range(adj.indptr[v], adj.indptr[v + 1]):
            u = adj.indices[ix]
            if done[u]:
                continue
            cand = d + float(excess[u, label]) + step * (2 - float(edge_weights[ix]))
            if cand < dist[u]:
                dist[u] = cand
                labels[u] = label
                parent[u] = v
                heapq.heappush(heap, (cand, int(u), label))
    assert done.all() and np.array_equal(labels[seeds], np.arange(k))
    nonseed = parent >= 0
    assert np.array_equal(labels[nonseed], labels[parent[nonseed]])
    return labels

def f_polish(labels, cost, adj, seeds, lam=0.015, sweeps=3):
    """Sequential boundary moves with a sufficient local connectivity condition."""
    labels = labels.copy()
    pins = np.zeros(len(labels), bool)
    pins[seeds] = True
    neighbors = [adj.indices[adj.indptr[v]:adj.indptr[v + 1]] for v in range(len(labels))]

    def energy():
        v = np.repeat(np.arange(len(labels)), np.diff(adj.indptr))
        return float(cost[np.arange(len(labels)), labels].sum() + lam * 0.5 * np.sum(adj.data * (labels[v] != labels[adj.indices])))
    before = f_energy()
    moves = 0
    for sweep in range(sweeps):
        changed = 0
        for v, neigh in enumerate(neighbors):
            if pins[v]:
                continue
            old = int(labels[v])
            other = np.unique(labels[neigh])
            other = other[other != old]
            if not len(other):
                continue
            same = neigh[labels[neigh] == old]
            if not len(same):
                continue
            allowed = set((int(u) for u in same))
            seen = {int(same[0])}
            stack = list(seen)
            while stack:
                u = stack.pop()
                for w in neighbors[u]:
                    if int(w) in allowed and int(w) not in seen:
                        seen.add(int(w))
                        stack.append(int(w))
            if len(seen) != len(same):
                continue
            weights = adj.data[adj.indptr[v]:adj.indptr[v + 1]]
            candidates = np.r_[old, other]
            energies = np.array([cost[v, k] + lam * weights[labels[neigh] != k].sum() for k in candidates])
            best = int(energies.argmin())
            if best and energies[best] < energies[0] - 1e-09:
                labels[v] = candidates[best]
                changed += 1
        moves += changed
        if not changed:
            break
    after = f_energy()
    assert after <= before + 1e-05
    assert np.array_equal(labels[seeds], np.arange(len(seeds)))
    assert all((connected_components(adj[labels == k][:, labels == k], directed=False)[0] == 1 for k in range(len(seeds))))
    return (labels, dict(boundary_moves=moves, energy_before=before, energy_after=after))
