from f_runtime import f_work_root
"""CoBRA18 network extension; preserve pre-existing distributed components.

This is a separately identified coarse-network extension, not the K100 decoder.
Shared author18 priors define network identities in both hemispheres. Full-FC
pretrained backbone is reused; visit-specific adaptation restarts independently.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
from pathlib import Path
import argparse, copy, fcntl, gc, sys, time, traceback
import numpy as np
import torch
from scipy.sparse.csgraph import connected_components
from f_network18_coarse_network18 import W, OUT, CODE, APP, SOURCE, VISITS, f_read, f_save, f_sha, f_standardize
sys.path.insert(0, str(W / 'paper_figure_campaign'))
from f_exact_fc_adaptation import f_project, f_target_statistics, f_exact_reconstruction
from f_full_fc_vae import FullFCVAE
from f_model_ops import f_transferred_model, f_encode_projected, f_regional_centers, f_regularization, f_distance
from f_pipeline import f_geometry
from f_connectivity import f_boundary_depth, f_boundary_optimize, f_energy

def f_load_group(h):
    exp = f_geometry(W, h, OUT, 'raw')
    model = f_transferred_model(FullFCVAE(exp.n, 128, 32, 2).cuda(), exp.n)
    path = SOURCE / f'group/{h}/checkpoint.pt'
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert state['step'] == 1000
    model.load_state_dict(state['model'])
    del state
    model.eval().requires_grad_(False)
    raw = np.load(exp.group_source, mmap_mode='r')
    linear = model.input[0]
    with torch.no_grad():
        acc = linear.bias.expand(exp.n, -1).clone()
        for a in range(0, exp.n, 256):
            b = min(a + 256, exp.n)
            v = torch.as_tensor(np.array(raw[:, a:b], copy=True), device='cuda') / exp.scale[a:b]
            acc.addmm_(v, linear.weight[:, a:b].T)
        inputs = model.input[2](model.input[1](acc))[None]
        exp.group_mu = f_encode_projected(model, inputs, exp.graph, exp.patches, exp.pos)[0].detach()
    del raw, acc, inputs, v
    exp.initial = np.load(OUT / f'geometry/prior32_{h}.npy').astype('int32')
    exp.k = 18
    exp.depth = f_boundary_depth(exp.initial, exp.adj)
    seeds = []
    components = []
    for k in range(18):
        ids = np.flatnonzero(exp.initial == k)
        count, cc = connected_components(exp.adj[ids][:, ids], directed=False)
        components.append(count)
        for c in range(count):
            members = ids[cc == c]
            seeds.append(members[np.argmax(exp.depth[members])])
    exp.seeds = np.array(seeds, dtype='int32')
    exp.original_components = components
    core = exp.depth >= 3
    core[exp.seeds] = True
    exp.core = torch.from_numpy(core).cuda()
    exp.labels = torch.from_numpy(exp.initial.astype('int64')).cuda()
    exp.group_centers = f_regional_centers(exp.group_mu[0], exp.labels, exp.core, 18)
    return (exp, model, path)

def f_decode_network(exp, y, cost, hard):
    f = np.ascontiguousarray(y.T, dtype='float32')
    f /= np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-08)
    counts = np.bincount(exp.initial, minlength=18)
    minimum = np.minimum(counts, np.maximum(20, (0.25 * counts).astype('int64')))
    maximum = np.maximum(counts, np.full(18, int(np.ceil(4 * len(f) / 18))))
    orders = np.array([np.random.default_rng(20260930 + i).permutation(len(f)) for i in range(20)], dtype='int32')
    before = f_energy(exp.initial, exp.initial, f, cost, exp.edges, np.ones(len(exp.edges)), 0.02)
    labels, history = f_boundary_optimize(exp.initial, f, cost, exp.adj.indptr, exp.adj.indices, exp.adj.data, exp.seeds, orders, exp.depth <= 12, minimum, maximum, 0.02, hard)
    after = f_energy(labels, exp.initial, f, cost, exp.edges, np.ones(len(exp.edges)), 0.02)
    cc = [int(connected_components(exp.adj[labels == k][:, labels == k], directed=False, return_labels=False)) for k in range(18)]
    assert after <= before + 0.002 and np.array_equal(labels[exp.seeds], exp.initial[exp.seeds])
    if hard:
        assert np.all(np.array(cc) <= exp.original_components)
    return (labels, dict(energy_before=before, energy_after=after, initial_components=exp.original_components, final_components=cc, excess_over_prior=int(np.maximum(np.array(cc) - exp.original_components, 0).sum()), moves=history.tolist(), changed_fraction=float(np.mean(labels != exp.initial)), hard_component_guard=hard))

def f_fit_visit(exp, group, group_path, row, h, visit, steps=30):
    dest = OUT / f"CoBRA18/{visit}/{row['subject']}/{h}"
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'complete.json').exists():
        return
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    rec = f_read(OUT / f"inputs/{visit}/{row['subject']}/complete.json")
    raw = np.load(OUT / f"inputs/{visit}/{row['subject']}/{h}_sm6_32.npy").T
    split = rec['retained_frames_per_run'][0]
    arrays = [f_standardize(y)[0] for y in [raw[:split], raw[split:]]]
    assert all((np.isfinite(y).all() for y in arrays))
    torch.manual_seed(20260930 + int(row['subject']))
    model = copy.deepcopy(group).eval().requires_grad_(True)
    ys = [torch.from_numpy(y).cuda() for y in arrays]
    stats = f_target_statistics(ys, exp.scale)
    with torch.no_grad():
        baseline = f_encode_projected(model, f_project(model, ys, exp.scale), exp.graph, exp.patches, exp.pos)[0]
        centers = 0.5 * exp.group_centers + 0.5 * f_regional_centers(baseline.mean(0), exp.labels, exp.core, 18)
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    history = []
    start = 0
    checkpoint = dest / 'checkpoint.pt'
    group_hash = f_sha(group_path)
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert state['group_sha256'] == group_hash and state['steps'] == steps
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start = state['step']
        history = state['history']
        del state
    for step in range(start, steps):
        optimizer.zero_grad(set_to_none=True)
        mu, lv = f_encode_projected(model, f_project(model, ys, exp.scale), exp.graph, exp.patches, exp.pos)
        hidden = model.decoder_hidden(mu, lv, sample=False)
        rec_loss = f_exact_reconstruction(hidden, model.decoder[2], ys, stats)
        penalty, details = f_regularization(exp, model, mu, lv, baseline, centers, initial)
        (rec_loss + penalty).backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert torch.isfinite(rec_loss) and torch.isfinite(norm)
        optimizer.step()
        details.update(step=step + 1, reconstruction=float(rec_loss.detach()), total=float((rec_loss + penalty).detach()))
        history.append(details)
        f_save(OUT / 'CoBRA18/status.json', dict(state='running', subject=row['subject'], visit=visit, hemisphere=h, step=step + 1, pid=os.getpid(), updated_unix=time.time()))
        if (step + 1) % 5 == 0 or step == steps - 1:
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step + 1, steps=steps, history=history, group_sha256=group_hash), checkpoint)
        del mu, lv, hidden, rec_loss, penalty, norm
    with torch.no_grad():
        mu = f_encode_projected(model, f_project(model, ys, exp.scale), exp.graph, exp.patches, exp.pos)[0]
        centers = 0.5 * exp.group_centers + 0.5 * f_regional_centers(mu.mean(0), exp.labels, exp.core, 18)
        cost = f_distance(mu.mean(0), centers).cpu().numpy()
    audits = {}
    for name, hard in [('CoBRA', True), ('no_component_guard', False)]:
        labels, audit = f_decode_network(exp, np.concatenate(arrays), cost, hard)
        audits[name] = audit
        np.save(dest / f'{name}_32.npy', labels)
        nearest = np.load(OUT / f'geometry/nearest32_{h}.npy')
        native = labels[nearest].astype('int32')
        native[np.load(OUT / f'geometry/author_labels_{h}.npy') < 0] = -1
        np.save(dest / f'{name}_fs4.npy', native)
    f_save(dest / 'complete.json', dict(subject=row['subject'], visit=visit, hemisphere=h, steps=steps, network_IDs=18, extension='Distributed author18 network prior with pinned initial components and topology-preserving boundary moves; distinct from K100 one-component-per-parcel decoder.', full_FC=True, all_ordered_cross_PE_pairs=True, neural_prior_people=450, group_sha256=group_hash, decoder=audits, input='Same native source,6mm smoothing,canonical filtering and frames as native Wang inputs; full fsLR cortex before fs4 resampling.', peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2 ** 30, runtime_memory_fraction=0.18, code_sha256=f_sha(Path(__file__))))
    del model, optimizer, initial, baseline, ys, stats, mu, centers
    torch.cuda.empty_cache()

def f_main(smoke=False):
    lock = (OUT / 'CoBRA18_gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert f_read(OUT / 'validation/smoke_complete.json')['passed']
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.18)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == os.environ['COBRA_GPU']
    assert torch.cuda.mem_get_info()[0] > 5.5 * 2 ** 30, 'Insufficient spare GPU1 memory; leave other tasks untouched.'
    rows = f_read(APP / 'subjects.json')
    rows = rows[:1] if smoke else rows
    for h in ['L', 'R']:
        exp, group, path = f_load_group(h)
        for row in rows:
            for visit, _, _ in VISITS[:1] if smoke else VISITS:
                while not (OUT / f"inputs/{visit}/{row['subject']}/complete.json").exists() or not (OUT / f"inputs/{visit}/{row['subject']}/{h}_sm6_32.npy").exists():
                    assert not (OUT / 'cpu_failure.json').exists(), 'Input preparation failed; inspect CPU failure audit'
                    f_save(OUT / 'CoBRA18/status.json', dict(state='waiting_input', subject=row['subject'], visit=visit, hemisphere=h, updated_unix=time.time()))
                    time.sleep(15)
                f_fit_visit(exp, group, path, row, h, visit, steps=30)
        del exp, group
        torch.cuda.empty_cache()
    f_save(OUT / ('CoBRA18/smoke_complete.json' if smoke else 'CoBRA18/complete.json'), dict(passed=True, subjects=len(rows), hemispheres=['L', 'R'], networks=18, steps=30, code_sha256=f_sha(Path(__file__))))
