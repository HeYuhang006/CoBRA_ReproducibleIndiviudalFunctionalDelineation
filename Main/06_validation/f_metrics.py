from f_runtime import f_work_root
"""K-generic heldout H, adjacent silhouette, DCBC and topology diagnostics."""
import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score

def f_topology(v, adj, k):
    counts = np.bincount(v, minlength=k)
    assert len(counts) == k and np.all(counts > 0)
    return dict(counts=counts.tolist(), components=[int(connected_components(adj[v == a][:, v == a], directed=False)[0]) for a in range(k)], minimum=int(counts.min()), maximum=int(counts.max()))

@torch.no_grad()
def f_scores(y, labels, edges, dist, k, chunk=256):
    p = F.one_hot(torch.as_tensor(labels, device=y.device, dtype=torch.long), k).float()
    cnt = p.sum(0)
    corr = y.T @ (y @ p) / len(y)
    index = torch.as_tensor(labels, device=y.device, dtype=torch.long)
    own = corr.gather(1, index[:, None]).squeeze(1)
    diag = y.square().mean(0)
    n = cnt[index]
    within = (own - diag) / (n - 1).clamp_min(1)
    valid = n > 1
    H = float(within[valid].mean())
    adjacency = np.zeros((k, k), bool)
    lab = labels[edges]
    adjacency[lab[:, 0], lab[:, 1]] = True
    adjacency |= adjacency.T
    np.fill_diagonal(adjacency, False)
    a = torch.as_tensor(adjacency, dtype=torch.float32, device=y.device)
    nn = (cnt @ a.T)[index]
    between = 1 - (corr @ a.T).gather(1, index[:, None]).squeeze(1) / nn.clamp_min(1)
    wi = 1 - within
    ok = valid & (nn > 0)
    S = float(((between - wi) / torch.maximum(wi, between).clamp_min(1e-07))[ok].mean())
    sums = np.zeros(70)
    counts = np.zeros(70, dtype='int64')
    N = len(labels)
    for start in range(0, N, chunk):
        stop = min(start + chunk, N)
        d = np.asarray(dist[start:stop])
        rr, cc = np.nonzero((d > 0) & (d <= 35) & (np.arange(start, stop)[:, None] < np.arange(N)[None]))
        bins = (np.ceil(d[rr, cc]) - 1).astype('int64')
        f = (y[:, start:stop].T @ y / len(y)).cpu().numpy()
        category = bins + 35 * (labels[start + rr] == labels[cc])
        counts += np.bincount(category, minlength=70)
        sums += np.bincount(category, weights=f[rr, cc], minlength=70)
    nb, nw = (counts[:35], counts[35:])
    valid = (nb > 0) & (nw > 0)
    w = nw[valid].astype('float64') * nb[valid] / (nw[valid] + nb[valid])
    w /= w.sum()
    DCBC = float(w @ (sums[35:][valid] / nw[valid] - sums[:35][valid] / nb[valid]))
    return dict(homogeneity=H, adjacent_silhouette=S, DCBC=DCBC, DCBC_max_mm=35)

def f_agreement(a, b, k):
    c = np.bincount(a.astype('int64') * k + b, minlength=k * k).reshape(k, k)
    i, j = linear_sum_assignment(-c)
    dice = 2 * c[i, j] / (np.bincount(a, minlength=k)[i] + np.bincount(b, minlength=k)[j])
    return dict(ARI=float(adjusted_rand_score(a, b)), Dice=float(dice.mean()))
