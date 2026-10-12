from f_runtime import f_work_root
"""Paper equations and surface/connectivity utilities; no experimental side effects."""
import os
from pathlib import Path
SOURCE_DIR = f_work_root() / 'final/population_training'
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import eigsh

class FCVAE(nn.Module):

    def __init__(self, n_features, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(n_features, 1024), nn.ReLU(), nn.LayerNorm(1024), nn.Linear(1024, 512), nn.ReLU(), nn.LayerNorm(512))
        self.mu = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 512), nn.ReLU(), nn.LayerNorm(512), nn.Linear(512, 1024), nn.ReLU(), nn.LayerNorm(1024), nn.Linear(1024, n_features))
        self.observation_logvar = nn.Parameter(torch.zeros(n_features))

    def encode(self, x):
        h = self.encoder(x)
        return (self.mu(h), self.logvar(h).clamp(-12, 12))

    @staticmethod
    def sample(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

def f_gaussian_nll(x, predicted, logvar):
    return 0.5 * ((x - predicted).square() * torch.exp(-logvar) + logvar).mean()

def f_normal_kl(mu, logvar):
    return 0.5 * (mu.square() + logvar.exp() - logvar - 1).mean()

def f_neighborhood_nce(anchor, positives, negatives, temperature=0.1):
    a = F.normalize(anchor, dim=-1)
    p = F.normalize(positives, dim=-1)
    n = F.normalize(negatives, dim=-1)
    lp = (a[:, None, :] * p).sum(-1) / temperature
    ln = (a[:, None, :] * n).sum(-1) / temperature
    return (torch.logsumexp(torch.cat([lp, ln], -1), -1) - torch.logsumexp(lp, -1)).mean()

def f_cortical_graph(coords, faces, vertices):
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    mapping = np.full(len(coords), -1, dtype=np.int64)
    mapping[vertices] = np.arange(len(vertices))
    keep = (mapping[edges[:, 0]] >= 0) & (mapping[edges[:, 1]] >= 0)
    edges = edges[keep]
    lengths = np.linalg.norm(coords[edges[:, 0]] - coords[edges[:, 1]], axis=-1)
    compact = mapping[edges]
    rows = np.concatenate([compact[:, 0], compact[:, 1]])
    cols = np.concatenate([compact[:, 1], compact[:, 0]])
    graph = csr_matrix((np.tile(lengths, 2), (rows, cols)), shape=(len(vertices), len(vertices)))
    if connected_components(graph, directed=False, return_labels=False) != 1:
        raise ValueError('Cortical mesh is disconnected; inspect mask/mapping before fitting LPE.')
    return (graph, compact)

def f_laplacian_features(graph, k=5, seed=20260930):
    adjacency = graph.copy()
    adjacency.data[:] = 1
    degree = np.asarray(adjacency.sum(1)).ravel()
    norm = diags(1 / np.sqrt(degree))
    lap = eye(graph.shape[0], format='csr') - norm @ adjacency @ norm
    rng = np.random.default_rng(seed)
    values, vectors = eigsh(lap, k=k + 1, which='SM', tol=1e-06, v0=rng.normal(size=graph.shape[0]))
    order = np.argsort(values)
    values, vectors = (values[order], vectors[:, order])
    if abs(values[0]) > 1e-05:
        raise ValueError('Laplacian zero mode not found')
    features = vectors[:, 1:k + 1]
    features = (features - features.mean(0)) / features.std(0)
    return (features.astype(np.float32), values)

def f_residualize(fc, distances, gmax=45.0, bin_width=1.0, chunk=512):
    n = len(fc)
    glob_mean = float(np.mean(fc, dtype=np.float64))
    glob_std = float(np.std(fc, dtype=np.float64))
    if not np.isfinite(glob_std) or glob_std <= 0:
        raise ValueError('Degenerate FC matrix')
    nbins = int(np.ceil(gmax / bin_width))
    counts = np.zeros(nbins, np.int64)
    sums = np.zeros(nbins, np.float64)
    squares = np.zeros(nbins, np.float64)
    for begin in range(0, n, chunk):
        d = np.asarray(distances[begin:begin + chunk])
        f = np.asarray(fc[begin:begin + chunk])
        mask = np.isfinite(d) & (d <= gmax)
        b = np.minimum((d[mask] / bin_width).astype(np.int64), nbins - 1)
        x = f[mask].astype(np.float64)
        counts += np.bincount(b, minlength=nbins)
        sums += np.bincount(b, weights=x, minlength=nbins)
        squares += np.bincount(b, weights=x * x, minlength=nbins)
    means = sums / np.maximum(counts, 1)
    scales = np.sqrt(np.maximum(squares / np.maximum(counts, 1) - means ** 2, 0))
    scales = np.maximum(scales, 1e-06)
    return {'global_mean': glob_mean, 'global_std': glob_std, 'counts': counts, 'means': means, 'scales': scales, 'gmax': gmax, 'bin_width': bin_width}

def f_apply_residualization(fc, distances, statistics, output, chunk=512):
    means, scales = (statistics['means'], statistics['scales'])
    for begin in range(0, len(fc), chunk):
        d = np.asarray(distances[begin:begin + chunk])
        f = np.asarray(fc[begin:begin + chunk])
        z = (f - statistics['global_mean']) / statistics['global_std']
        mask = np.isfinite(d) & (d <= statistics['gmax'])
        bins = np.minimum((d[mask] / statistics['bin_width']).astype(np.int64), len(means) - 1)
        z[mask] = (f[mask] - means[bins]) / scales[bins]
        output[begin:begin + len(f)] = z
    output.flush()

def f_candidate_pools(features, distances, gmax=45.0, top=0.15, bottom=0.25):
    pos, neg = ([], [])
    n = len(features)
    for i in range(n):
        neighbors = np.flatnonzero((distances[i] > 0) & (distances[i] <= gmax))
        if len(neighbors) < 40:
            raise ValueError(f'Vertex {i} has only {len(neighbors)} local candidates')
        vals = np.asarray(features[i, neighbors])
        order = np.argsort(vals, kind='stable')
        pcount = max(6, int(np.ceil(top * len(neighbors))))
        ncount = max(32, int(np.ceil(bottom * len(neighbors))))
        if pcount + ncount > len(neighbors):
            raise ValueError('Candidate sets overlap; neighborhood too small')
        pos.append(neighbors[order[-pcount:]].astype(np.int32))
        neg.append(neighbors[order[:ncount]].astype(np.int32))
    return (pos, neg)

def f_packed_pools(pools):
    sizes = np.array([len(x) for x in pools], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    return (np.concatenate(pools), offsets)

def f_sample_pools(flat, offsets, anchors, count, rng):
    return np.stack([rng.choice(flat[offsets[i]:offsets[i + 1]], count, replace=False) for i in anchors])

def f_functional_homogeneity(labels, fc):
    """Size-weighted mean of unordered off-diagonal within-parcel Pearson FC."""
    total, weight = (0.0, 0)
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label)
        if len(ids) < 2:
            continue
        block = fc[np.ix_(ids, ids)]
        value = (block.sum(dtype=np.float64) - np.trace(block)) / (len(ids) * (len(ids) - 1))
        total += value * len(ids)
        weight += len(ids)
    return float(total / weight)

def f_exact_cosine_silhouette(features, labels, device='cpu'):
    """Exact cosine silhouette via cluster mean directions, avoiding N x N distances."""
    x = torch.as_tensor(np.asarray(features), dtype=torch.float32, device=device).clone()
    x = F.normalize(x, dim=1)
    unique, compact = np.unique(labels, return_inverse=True)
    idx = torch.as_tensor(compact, device=device)
    count = torch.bincount(idx, minlength=len(unique))
    sums = torch.zeros((len(unique), x.shape[1]), device=device)
    sums.index_add_(0, idx, x)
    centers = sums / count[:, None]
    values = []
    for start in range(0, len(x), 512):
        z = x[start:start + 512]
        group = idx[start:start + len(z)]
        distances = (1 - z @ centers.T).clamp_min(0)
        row = torch.arange(len(z), device=device)
        a = distances[row, group] * count[group] / (count[group] - 1).clamp_min(1)
        distances[row, group] = float('inf')
        b = distances.min(1).values
        silhouette = (b - a) / torch.maximum(a, b).clamp_min(1e-12)
        silhouette[count[group] == 1] = 0
        values.append(silhouette)
    result = torch.cat(values).mean().item()
    del x, sums, centers
    return float(result)

def f_dcbc_from_standardized_fc(fc, distances, labels, max_distance=35, chunk=128):
    """Official DCBC weighting for temporally standardized data, all vertex pairs.

    Standardization makes the official covariance/mean-sd-product ratio equal to
    mean Pearson FC. Group FC yields the mean subject DCBC for fixed distances/labels.
    Uses (lower, upper] bins, matching DiedrichsenLab/DCBC main/dcbc.py.
    """
    within_sums, between_sums = (np.zeros(max_distance), np.zeros(max_distance))
    within_n, between_n = (np.zeros(max_distance, np.int64), np.zeros(max_distance, np.int64))
    for start in range(0, len(fc), chunk):
        d = np.asarray(distances[start:start + chunk])
        values = np.asarray(fc[start:start + chunk])
        valid = (d > 0) & (d <= max_distance)
        same = labels[start:start + len(d), None] == labels[None, :]
        for mask, sums, counts in [(valid & same, within_sums, within_n), (valid & ~same, between_sums, between_n)]:
            bins = np.ceil(d[mask]).astype(int) - 1
            sums += np.bincount(bins, weights=values[mask], minlength=max_distance)
            counts += np.bincount(bins, minlength=max_distance)
    usable = (within_n > 0) & (between_n > 0)
    weight = within_n[usable] * between_n[usable] / (within_n[usable] + between_n[usable])
    differences = within_sums[usable] / within_n[usable] - between_sums[usable] / between_n[usable]
    return {'dcbc': float(np.average(differences, weights=weight)), 'max_distance_mm': max_distance, 'bin_width_mm': 1, 'within_counts': within_n, 'between_counts': between_n, 'input_representation': 'z-scored BOLD time-series correlations', 'equivalence': 'official ratio-of-mean-covariance to mean-sd-product after per-vertex standardization'}

def f_parcel_components(labels, edges):
    counts = []
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label)
        mapping = np.full(len(labels), -1, np.int32)
        mapping[ids] = np.arange(len(ids))
        e = edges[(labels[edges[:, 0]] == label) & (labels[edges[:, 1]] == label)]
        e = mapping[e]
        adjacency = csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(ids), len(ids)))
        counts.append(connected_components(adjacency, directed=False, return_labels=False))
    return {'mean_components': float(np.mean(counts)), 'max_components': int(np.max(counts)), 'counts': [int(c) for c in counts]}
