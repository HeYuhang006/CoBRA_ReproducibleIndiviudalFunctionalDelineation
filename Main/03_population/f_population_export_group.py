from f_runtime import f_work_root
"""Group FC -> posterior means -> LPE1 K-means, in a fresh output folder.

This CPU export is independent of personal fitting. It never overwrites a
completed atlas; connected functional refinement is implemented in f_priors.py.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from f_data import f_save, f_sha
from f_population_full_fc_vae import FullFCVAE
from f_model_ops import f_encode_projected

@torch.no_grad()
def f_embedding(work, source, h, checkpoint_name):
    cfg = json.loads((source / 'configuration.json').read_text())
    path = source / h / checkpoint_name
    state = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint_name == 'individual_checkpoint.pt':
        assert state['epoch'] == cfg['individual_epochs'] and state['next_subject'] == len(json.loads((source / 'subjects.json').read_text())['train'])
    matrix = np.load(source / f'group_{h}.npy', mmap_mode='r')
    n = len(matrix)
    model = FullFCVAE(n, cfg['width'], cfg['latent_dim'], cfg['attention_layers']).eval().requires_grad_(False)
    model.load_state_dict(state['model'])
    scale = torch.from_numpy(np.load(source / f'scale_{h}.npy'))
    weight = model.input[0].weight[:, :n] / scale
    projected = torch.empty(n, cfg['width'])
    for a in range(0, n, 512):
        projected[a:a + 512] = torch.from_numpy(np.array(matrix[a:a + 512], copy=True)) @ weight.T + model.input[0].bias
    geo = work / 'runs/pilot32_MSMAll_samplez'
    pos = torch.from_numpy(np.load(geo / f'lpe_{h}.npy').astype('float32'))
    patches = torch.from_numpy(np.load(source / f'attention_patches_{h}.npy')).long()
    edges = np.load(geo / f'edges_{h}.npy')
    rc = np.r_[edges, edges[:, ::-1]]
    degree = np.bincount(rc[:, 0], minlength=n)
    graph = torch.sparse_coo_tensor(torch.from_numpy(rc.T), torch.from_numpy((1 / degree[rc[:, 0]]).astype('float32')), (n, n)).coalesce()
    mu, lv = f_encode_projected(model, model.input[2](model.input[1](projected))[None], graph, patches, pos)
    assert np.isfinite(mu.numpy()).all()
    return (mu[0].numpy(), lv[0].exp().numpy(), dict(checkpoint=str(path), sha256=f_sha(path), input='training-group FC and zero personal-minus-group residual; not mean individual embeddings'))

def f_main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', required=True)
    p.add_argument('--output', required=True, help='Fresh output directory; existing directories are refused')
    p.add_argument('--population-run', default='fullfc450_crossday_v1')
    p.add_argument('--reencode', action='store_true', help='Encode completed population checkpoint; otherwise reproduce clustering from audited saved group means')
    a = p.parse_args()
    work = Path(a.workspace).resolve()
    out = Path(a.output).resolve()
    assert not out.exists(), 'Use a fresh output folder'
    out.mkdir(parents=True)
    torch.set_num_threads(2)
    threadpool_limits(2)
    audit = {}
    for h in ['L', 'R']:
        if a.reencode:
            mu, var, provenance = f_embedding(work, work / 'runs' / a.population_run, h, 'individual_checkpoint.pt')
            np.save(out / f'latent_variance_{h}.npy', var)
        else:
            path = work / f'runs/vae450_multires_20261006_v1/population_epoch50/latent_mean_{h}.npy'
            mu = np.load(path)
            provenance = dict(saved_group_mean=str(path), sha256=f_sha(path))
        pos = np.load(work / f'runs/pilot32_MSMAll_samplez/lpe_{h}.npy')
        features = np.c_[mu, pos].astype('float32')
        assert features.shape == (len(mu), 37)
        np.save(out / f'latent_mean_{h}.npy', mu)
        audit[h] = provenance
        for k in [50, 100, 150, 200]:
            fit = KMeans(k, n_init=10, max_iter=300, random_state=20260930).fit(features)
            np.save(out / f'labels_LPE1_{h}_{k}.npy', fit.labels_.astype('int16'))
            np.save(out / f'centroids_LPE1_{h}_{k}.npy', fit.cluster_centers_)
            print('KMEANS', h, k, flush=True)
    f_save(out / 'provenance.json', dict(source=audit, LPE_weight=1, n_init=10, max_iter=300, seed=20260930, independent_resolutions=True, hierarchy=False, connected=False, next_step='Connected growth and training-only functional boundary refinement in final/f_priors.py'))
