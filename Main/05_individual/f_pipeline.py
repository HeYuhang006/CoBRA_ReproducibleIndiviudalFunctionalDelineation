from f_runtime import f_work_root
"""Corrected bilateral personal VAE: priors, LR control, joint LR/RL and export."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse, copy, json, subprocess, sys, time, fcntl
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from f_full_fc_vae import FullFCVAE
from f_distance_fc_vae import DistanceNormalization
from f_data import f_read, f_save, f_sha, f_cohort, f_training, f_evaluation_days
from f_model_ops import f_transferred_model, f_encode_projected, f_materialize, f_reconstruction_backward, f_regional_centers, f_distance, f_regularization
from f_surface_prior_vae import f_diagonal_kl_to_prior
from f_connectivity import f_graph_data, f_boundary_depth, f_decode
from f_metrics import f_scores, f_agreement, f_topology
import f_priors as priors, f_plotting as plotting
ARMS = ['lr_only', 'joint_equal', 'joint_all']
KS = [50, 100, 150, 200]

def f_geometry(w, h, out, mode):
    geo = w / 'runs/pilot32_MSMAll_samplez'
    pos = np.load(geo / f'lpe_{h}.npy')
    edges = np.load(geo / f'edges_{h}.npy')
    n = len(pos)
    rc = np.r_[edges, edges[:, ::-1]]
    degree = np.bincount(rc[:, 0], minlength=n)
    graph = torch.sparse_coo_tensor(torch.from_numpy(rc.T), torch.from_numpy((1 / degree[rc[:, 0]]).astype('float32')), (n, n)).coalesce().cuda()
    base = w / 'runs' / ('fullfc450_crossday_v1' if mode == 'raw' else 'fullfc450_distance_crossday_v1')
    exp = SimpleNamespace(n=n, edges=edges, adj=f_graph_data(n, edges), graph=graph, pos=torch.from_numpy(pos).cuda(), patches=torch.from_numpy(np.load(w / f'runs/fullfc450_crossday_v1/attention_patches_{h}.npy')).long().cuda(), scale=torch.from_numpy(np.load(base / f'scale_{h}.npy')).cuda(), group_source=base / f'group_{h}.npy', transform=None)
    if mode == 'distance':
        d = w / 'resources/fullfc_distance/fullfc450_distance_crossday_v1'
        st = np.load(d / f'statistics_{h}.npz')
        bins = torch.from_numpy(np.load(d / f'bins_{h}.npy')).cuda()
        exp.transform = DistanceNormalization(bins, torch.tensor(np.r_[st['means'], st['global_mean']], dtype=torch.float32, device='cuda'), torch.tensor(np.r_[st['scales'], st['global_std']], dtype=torch.float32, device='cuda'))
    return exp

def f_group_matrix(exp):
    matrix = np.load(exp.group_source)
    fc = exp.scale.new_empty(1, exp.n, exp.n)
    for a in range(0, exp.n, 256):
        fc[0, a:a + 256] = torch.from_numpy(np.array(matrix[a:a + 256], copy=True)).cuda() / exp.scale
    return fc

def f_group_model(w, out, h, exp, mode):
    dest = out / 'group' / h
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / 'checkpoint.pt'
    torch.manual_seed(20260930)
    model = f_transferred_model(FullFCVAE(exp.n, 128, 32, 2).cuda(), exp.n).requires_grad_(True)
    existing = w / f"runs/{('dense_fc128_20261004_v1' if mode == 'raw' else 'dense_fc128_distance_20261006_v1')}/group/checkpoint.pt"
    can_reuse = False
    fc = f_group_matrix(exp)
    if path.exists():
        state = torch.load(path, map_location='cpu', weights_only=False)
        start = state['step']
        model.load_state_dict(state['model'])
    elif can_reuse:
        oldpatch = np.load(w / 'runs/modulated_vae_full_crossday_v1/anchor_patches_L.npy')
        assert np.array_equal(oldpatch, exp.patches.cpu().numpy())
        state = torch.load(existing, map_location='cpu', weights_only=False)
        assert state['step'] == 1000 and state['width'] == 128
        model.load_state_dict(state['model'])
        start = 1000
        torch.save(state, path)
        f_save(dest / 'reuse.json', dict(source=str(existing), sha256=f_sha(existing), identical_geometry=True))
    else:
        state = None
        start = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    if start < 1000 and state is not None:
        optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])
    for step in range(start, 1000):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        mu, lv = f_encode_projected(model, model.input(fc), exp.graph, exp.patches, exp.pos)
        hidden = model.decoder_hidden(mu, lv, sample=True)
        rec, g = f_reconstruction_backward(model, hidden, fc, cross=False)
        kl = 0.001 * f_diagonal_kl_to_prior(mu, lv, torch.zeros_like(mu))
        torch.autograd.backward([hidden, kl], [g, None])
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        assert np.isfinite(rec) and torch.isfinite(norm)
        optimizer.step()
        f_save(out / 'status.json', dict(state='running', stage='group_pretraining', hemisphere=h, step=step + 1, total=1000, pid=os.getpid(), loss=rec + float(kl.detach())))
        if (step + 1) % 50 == 0 or step == 999:
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step + 1, width=128, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), path)
    model.eval().requires_grad_(False)
    with torch.no_grad():
        exp.group_mu = f_encode_projected(model, model.input(fc), exp.graph, exp.patches, exp.pos)[0].detach()
    del fc, optimizer
    torch.cuda.empty_cache()
    f_save(dest / 'complete.json', dict(steps=1000, training_people=450, width=128, normalization=mode))
    return (model, path)

def f_set_prior(exp, out, h, k):
    p = out / 'priors' / h / str(k)
    exp.initial = np.load(p / 'labels.npy')
    exp.seeds = np.load(p / 'seeds.npy')
    exp.depth = f_boundary_depth(exp.initial, exp.adj)
    core = exp.depth >= 3
    core[exp.seeds] = True
    exp.core = torch.from_numpy(core).cuda()
    exp.labels = torch.from_numpy(exp.initial.astype('int64')).cuda()
    exp.k = k
    exp.group_centers = f_regional_centers(exp.group_mu[0], exp.labels, exp.core, k)

def f_fit(exp, source, source_path, arrays, dest, out, subject, h, k, arm, steps=30):
    dest.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260930 + int(subject))
    model = copy.deepcopy(source).eval().requires_grad_(True)
    ys = [torch.from_numpy(y).cuda() for y in arrays]
    fc = f_materialize(ys, exp.scale, exp.transform)
    del ys
    with torch.no_grad():
        baseline = f_encode_projected(model, model.input(fc), exp.graph, exp.patches, exp.pos)[0]
        centers = 0.5 * exp.group_centers + 0.5 * f_regional_centers(baseline.mean(0), exp.labels, exp.core, k)
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    history = []
    start = 0
    file = dest / 'checkpoint.pt'
    group_hash = f_sha(source_path)
    if file.exists():
        state = torch.load(file, map_location='cpu', weights_only=False)
        assert state['group_sha256'] == group_hash and state['steps'] == steps
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start = state['step']
        history = state['history']
    for step in range(start, steps):
        optimizer.zero_grad(set_to_none=True)
        mu, lv = f_encode_projected(model, model.input(fc), exp.graph, exp.patches, exp.pos)
        hidden = model.decoder_hidden(mu, lv, sample=False)
        rec, g = f_reconstruction_backward(model, hidden, fc, cross=True)
        penalty, details = f_regularization(exp, model, mu, lv, baseline, centers, initial)
        torch.autograd.backward([hidden, penalty], [g, None])
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert np.isfinite(rec) and torch.isfinite(norm)
        if step == 0:
            audit = {name: float(p.grad.abs().max()) if p.grad is not None else None for name, p in model.named_parameters()}
            assert all((v is not None and np.isfinite(v) and (v > 0) for v in audit.values()))
            f_save(dest / 'gradient_audit.json', audit)
        optimizer.step()
        details.update(step=step + 1, reconstruction=rec, total=rec + float(penalty.detach()))
        history.append(details)
        f_save(out / 'status.json', dict(state='running', stage='personal_finetune', subject=subject, hemisphere=h, K=k, arm=arm, step=step + 1, pid=os.getpid()))
        if (step + 1) % 5 == 0 or step == steps - 1:
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step + 1, steps=steps, group_sha256=group_hash, history=history), file)
    with torch.no_grad():
        mu = f_encode_projected(model, model.input(fc), exp.graph, exp.patches, exp.pos)[0]
        centers = 0.5 * exp.group_centers + 0.5 * f_regional_centers(mu.mean(0), exp.labels, exp.core, k)
    changes = {name: float((p - initial[name]).abs().max()) for name, p in model.named_parameters()}
    assert all((v > 0 for v in changes.values()))
    np.save(dest / 'centers.npy', centers.cpu().numpy())
    f_save(dest / 'fit.json', dict(subject=subject, hemisphere=h, K=k, arm=arm, steps=steps, input_shape=list(fc.shape), all_FC_entries=True, all_active_parameters_train=True, parameter_changes=changes, history=history, peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2 ** 30))
    del fc, optimizer, initial, baseline
    torch.cuda.empty_cache()
    return (model, centers)

@torch.no_grad()
def f_infer(exp, model, centers, y):
    fc = f_materialize([torch.from_numpy(y).cuda()], exp.scale, exp.transform)
    mu = f_encode_projected(model, model.input(fc), exp.graph, exp.patches, exp.pos)[0]
    cost = f_distance(mu[0], centers).cpu().numpy()
    del fc, mu
    torch.cuda.empty_cache()
    return f_decode(exp.initial, y, cost, exp.adj, exp.seeds, exp.depth, exp.edges)
