from f_runtime import f_work_root
"""Area-weighted FC and shared full-cortex heldout functional scoring."""
from pathlib import Path
import os
import numpy as np
import torch
from scipy.sparse import csr_matrix

def f_regional(y, labels, area, k):
    weights = np.zeros((len(labels), k), dtype='float32')
    weights[np.arange(len(labels)), labels] = area
    weights /= weights.sum(0, keepdims=True)
    return np.asarray(y @ weights, dtype='float32')

def f_connectome(left, right):
    x = np.c_[left, right].astype('float64')
    x -= x.mean(0)
    sd = np.linalg.norm(x, axis=0)
    assert (sd > 1e-08).all()
    x /= sd
    r = np.clip(x.T @ x, -1 + 1e-07, 1 - 1e-07)
    return np.arctanh(r[np.triu_indices(x.shape[1], 1)]).astype('float32')

def f_pair_geometry(w, h, out):
    """Cache upper-triangle geodesic pairs <=35mm once, without truncating FC."""
    p = out / f'geodesic_pairs_{h}.npz'
    if p.exists():
        return dict(np.load(p))
    d = np.load(w / f'resources/fullfc_distance/fullfc450_distance_crossday_v1/distances_{h}.npy', mmap_mode='r')
    n = len(d)
    rr = []
    cc = []
    bb = []
    for start in range(0, n, 256):
        block = np.asarray(d[start:start + 256])
        r, c = np.nonzero((block > 0) & (block <= 35) & (np.arange(start, min(start + 256, n))[:, None] < np.arange(n)[None]))
        rr.append((r + start).astype('int32'))
        cc.append(c.astype('int32'))
        bb.append((np.ceil(block[r, c]) - 1).astype('int8'))
    result = dict(row=np.concatenate(rr), col=np.concatenate(cc), bin=np.concatenate(bb))
    temporary = p.with_name(p.stem + f'.{os.getpid()}.npz')
    np.savez(temporary, **result)
    temporary.replace(p)
    return result

@torch.no_grad()
def f_quality(y, maps, edges, pairs):
    """Each independently standardized/censored run scored identically."""
    T, N = y.shape
    Y = torch.from_numpy(y).cuda()
    result = {}
    sums = {}
    counts = {}
    for name, labels in maps.items():
        k = int(labels.max() + 1)
        index = torch.as_tensor(labels, device=Y.device, dtype=torch.long)
        counts_k = np.bincount(labels, minlength=k)
        membership = torch.nn.functional.one_hot(index, k).float()
        corr = Y.T @ (Y @ membership) / T
        own = corr.gather(1, index[:, None]).squeeze(1)
        diag = Y.square().mean(0)
        number = torch.as_tensor(counts_k[labels], device=Y.device)
        within = (own - diag) / (number - 1).clamp_min(1)
        adj = np.zeros((k, k), dtype=bool)
        lab = labels[edges]
        adj[lab[:, 0], lab[:, 1]] = True
        adj |= adj.T
        np.fill_diagonal(adj, False)
        adjacency = torch.as_tensor(adj, dtype=torch.float32, device=Y.device)
        cnt = membership.sum(0)
        nn = (cnt @ adjacency.T)[index]
        between = 1 - (corr @ adjacency.T).gather(1, index[:, None]).squeeze(1) / nn.clamp_min(1)
        wi = 1 - within
        ok = (number > 1) & (nn > 0)
        result[name] = dict(homogeneity=float(within[number > 1].mean()), adjacent_silhouette=float(((between - wi) / torch.maximum(wi, between).clamp_min(1e-07))[ok].mean()))
        sums[name] = np.zeros(70)
        counts[name] = np.zeros(70, dtype='int64')
    row, col, bin_ = (pairs['row'], pairs['col'], pairs['bin'])
    begin = 0
    for start in range(0, N, 256):
        stop = min(N, start + 256)
        end = np.searchsorted(row, stop)
        rr = row[begin:end] - start
        cc = col[begin:end]
        bins = bin_[begin:end].astype('int64')
        block = (Y[:, start:stop].T @ Y / T).cpu().numpy()
        values = block[rr, cc]
        for name, labels in maps.items():
            category = bins + 35 * (labels[row[begin:end]] == labels[cc])
            counts[name] += np.bincount(category, minlength=70)
            sums[name] += np.bincount(category, weights=values, minlength=70)
        begin = end
    for name in maps:
        nb, nw = (counts[name][:35], counts[name][35:])
        valid = (nb > 0) & (nw > 0)
        if not valid.any():
            raise ValueError('No valid DCBC distance bins')
        weight = nw[valid].astype('float64') * nb[valid] / (nw[valid] + nb[valid])
        weight /= weight.sum()
        result[name]['DCBC'] = float(weight @ (sums[name][35:][valid] / nw[valid] - sums[name][:35][valid] / nb[valid]))
    del Y
    torch.cuda.empty_cache()
    return result
