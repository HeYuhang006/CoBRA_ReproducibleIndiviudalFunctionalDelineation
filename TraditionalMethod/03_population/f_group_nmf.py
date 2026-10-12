"""Group NMF: exact production numerical body, renamed functions only.
All outputs use the active fresh CoBRA work root. No VAE embedding enters NMF.
"""
from pathlib import Path
import hashlib,json,os,time
import numpy as np
import torch
from sklearn.cluster import KMeans
from f_runtime import f_work_root
W=f_work_root()
D=W/'runs/full500_MSMAll_samplez'
OUT=W/'runs/group_subject_boxplots_20261008_v1'
SEED=20261008

def f_save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)

def f_status(phase, **kw):
    f_save(OUT / 'status.json', dict(state='running', phase=phase, pid=os.getpid(), updated_unix=time.time(), **kw))

def f_digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2 ** 22), b''):
            sha.update(block)
    return sha.hexdigest()

@torch.no_grad()
def f_fit_nmf(device):
    folder = OUT / 'NMF'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        return
    f_status('NMF_training_FC_load')
    x = torch.as_tensor(np.load(D / 'fc_train_L.npy', mmap_mode='r'), device=device)
    negative = sum((int((x[i:i + 256] < 0).sum()) for i in range(0, len(x), 256))) / x.numel()
    x.clamp_min_(0)
    x.fill_diagonal_(0)
    n = x.shape[0]
    rank = 32
    torch.manual_seed(SEED)
    f_status('NMF_NNDSVDa_initialization')
    q = torch.linalg.qr(x @ torch.randn(n, rank + 10, device=device), mode='reduced').Q
    for _ in range(2):
        q = torch.linalg.qr(x @ q, mode='reduced').Q
    ub, s, vh = torch.linalg.svd(q.T @ x, full_matrices=False)
    u = q @ ub
    f = torch.zeros(n, rank, device=device)
    g = torch.zeros(rank, n, device=device)
    f[:, 0] = torch.sqrt(s[0]) * u[:, 0].abs()
    g[0] = torch.sqrt(s[0]) * vh[0].abs()
    for j in range(1, rank):
        up, un = (u[:, j].clamp_min(0), (-u[:, j]).clamp_min(0))
        vp, vn = (vh[j].clamp_min(0), (-vh[j]).clamp_min(0))
        np_, nn = (torch.linalg.vector_norm(up) * torch.linalg.vector_norm(vp), torch.linalg.vector_norm(un) * torch.linalg.vector_norm(vn))
        a, b, weight = (up, vp, np_) if np_ >= nn else (un, vn, nn)
        f[:, j] = torch.sqrt(s[j] * weight) * a / torch.linalg.vector_norm(a).clamp_min(1e-12)
        g[j] = torch.sqrt(s[j] * weight) * b / torch.linalg.vector_norm(b).clamp_min(1e-12)
    mean = x.mean()
    f[f < 1e-07] = mean
    g[g < 1e-07] = mean
    del q, u, ub, s, vh
    x2 = sum((x[i:i + 256].square().sum(dtype=torch.float64) for i in range(0, len(x), 256)))
    history = []
    initial = None
    previous = None
    converged = False
    start = time.time()
    for it in range(1, 1001):
        xg = x @ g.T
        f *= xg / (f @ (g @ g.T)).clamp_min(1e-10)
        fx = f.T @ x
        g *= fx / (f.T @ f @ g).clamp_min(1e-10)
        if it == 1 or it % 10 == 0:
            err = torch.sqrt((x2 - 2 * (g * fx).sum(dtype=torch.float64) + (f.T @ f * (g @ g.T)).sum(dtype=torch.float64)).clamp_min(0)).item()
            if initial is None:
                initial = err
            history.append(dict(iteration=it, error=err, seconds=time.time() - start))
            f_save(folder / 'fit_progress.json', dict(history=history, rank=rank, negative_FC_fraction=negative))
            f_status('NMF_fit', iteration=it, error=err)
            print('NMF', it, err, time.time() - start, flush=True)
            if previous is not None and 0 <= (previous - err) / initial < 0.0001:
                converged = True
                break
            previous = err
    features = f.cpu().numpy()
    np.save(folder / 'embedding.npy', features)
    del x, f, g, xg, fx
    torch.cuda.empty_cache()
    for k in [50, 100, 150, 200]:
        f_status('NMF_Kmeans', K=2 * k)
        km = KMeans(n_clusters=k, n_init=10, max_iter=300, tol=0.0001, random_state=SEED)
        labels = km.fit_predict(features).astype('int32')
        assert len(np.unique(labels)) == k
        np.save(folder / f'labels_L_{k}.npy', labels)
    f_save(folder / 'complete.json', dict(rank=rank, initialization='NNDSVDa, randomized range with two power iterations', solver='multiplicative updates for Frobenius NMF', max_iter=1000, tolerance=0.0001, iterations=it, converged=converged, feature='positive part of signed training-group Pearson FC, diagonal zero; no VAE embedding', training_subjects=500, clustering='NMF32 loadings + Kmeans; n_init10, seed20261008', history=history, negative_FC_fraction=negative, source_fc_sha256=f_digest(D / 'fc_train_L.npy')))
