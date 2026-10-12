from f_runtime import f_work_root
"""Same complete-FC adaptation objective with bounded intermediate memory.

Frozen canonical source files are not edited. This module is an execution
alternative, with explicit dense-reference loss and gradient checks.
"""
import copy, gc, os, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
W = f_work_root()
sys.path[:0] = [str(W / 'final'), str(W / 'paper_figure_campaign')]
from f_data import f_read, f_save, f_sha
from f_pipeline import f_geometry
from f_full_fc_vae import FullFCVAE
from f_model_ops import f_transferred_model, f_encode_projected, f_regional_centers, f_regularization, f_distance
from f_exact_fc_adaptation import f_project, f_target_statistics, f_exact_reconstruction
from f_connectivity import f_decode

def f_validate_small():
    reports = []
    for B in [2, 4]:
        torch.manual_seed(19)
        n = 29
        dtype = torch.float64
        ys = [torch.randn(31 + j, n, dtype=dtype) for j in range(B)]
        scale = torch.rand(n, dtype=dtype) + 0.5
        h = torch.randn(B, n, 7, dtype=dtype, requires_grad=True)
        linear = torch.nn.Linear(7, n, dtype=dtype)
        fc = torch.stack([y.T @ y / len(y) / scale for y in ys])
        pred = linear(h)
        exact = f_exact_reconstruction(h, linear, ys, f_target_statistics(ys, scale))
        dense = sum(((pred[i] - fc[j]).square().mean() for i in range(B) for j in range(B) if i != j)) / (B * (B - 1))
        a = torch.autograd.grad(exact, [h, linear.weight, linear.bias], retain_graph=True)
        b = torch.autograd.grad(dense, [h, linear.weight, linear.bias])
        error = max((float((x - y).abs().max()) for x, y in zip(a, b)))
        assert abs(float(exact - dense)) < 1e-12 and error < 1e-12
        m = torch.nn.Sequential(torch.nn.Linear(n, 11, dtype=dtype), torch.nn.LayerNorm(11, dtype=dtype), torch.nn.GELU())
        p = f_project(m if hasattr(m, 'input') else type('M', (), {'input': m})(), ys, scale)
        q = m(fc)
        a = torch.autograd.grad(p.square().mean(), list(m.parameters()), retain_graph=True)
        b = torch.autograd.grad(q.square().mean(), list(m.parameters()))
        pe = float((p - q).abs().max())
        ge = max((float((x - y).abs().max()) for x, y in zip(a, b)))
        assert pe < 1e-11 and ge < 1e-11
        reports.append(dict(runs=B, loss_absolute_error=abs(float(exact - dense)), decoder_gradient_max_error=error, projection_max_error=pe, projection_gradient_max_error=ge))
    return dict(passed=True, all_ordered_distinct_run_pairs=True, reports=reports)

def f_load_models(w, source, out):
    result = {}
    for h in ['L', 'R']:
        exp = f_geometry(w, h, out, 'raw')
        model = f_transferred_model(FullFCVAE(exp.n, 128, 32, 2).cuda(), exp.n)
        path = source / f'group/{h}/checkpoint.pt'
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert state['step'] == 1000 and state['width'] == 128
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
        result[h] = (exp, model, path)
        torch.cuda.empty_cache()
    return result

def f_validate_real(model, ys, scale, hidden, path):
    """Every cortical FC entry, all ordered pairs, direct blockwise reference."""
    B, N, _ = hidden.shape
    linear = model.decoder[2]
    leaf = hidden.detach().requires_grad_(True)
    analytic = f_exact_reconstruction(leaf, linear, ys, f_target_statistics(ys, scale))
    ag = torch.autograd.grad(analytic, [leaf, linear.weight, linear.bias])
    leaf2 = hidden.detach().requires_grad_(True)
    linear.zero_grad(set_to_none=True)
    total = 0.0
    for start in range(0, N, 256):
        stop = min(start + 256, N)
        pred = F.linear(leaf2, linear.weight[start:stop], linear.bias[start:stop])
        with torch.no_grad():
            target = torch.stack([y.T @ y[:, start:stop] / len(y) / scale[start:stop] for y in ys])
        for i in range(B):
            t = target[[j for j in range(B) if j != i]]
            loss = (pred[i:i + 1] - t).square().sum() / (B * (B - 1) * N * N)
            loss.backward(retain_graph=i < B - 1)
            total += float(loss.detach())
        del target, pred, t, loss
    bg = [leaf2.grad, linear.weight.grad, linear.bias.grad]
    errors = [dict(max_absolute=float((a - b).abs().max()), relative_l2=float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-12))) for a, b in zip(ag, bg)]
    relative = abs(float(analytic) - total) / max(total, 1e-12)
    assert relative < 0.0001 and all((e['relative_l2'] < 0.0001 for e in errors)), (relative, errors)
    linear.zero_grad(set_to_none=True)
    f_save(path, dict(passed=True, runs=B, cortical_vertices=N, all_N_squared_entries=True, all_ordered_distinct_run_pairs=True, relative_loss_error=relative, gradient_errors=errors, not_bitwise_equivalence=True))

def f_fit(exp, source, source_path, arrays, dest, out, subject, h, k, arm, steps=30):
    assert exp.transform is None, 'This factorization applies to the locked raw-FC arm only'
    dest.mkdir(parents=True, exist_ok=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(20260930 + int(subject))
    model = copy.deepcopy(source).eval().requires_grad_(True)
    ys = [torch.from_numpy(y).cuda() for y in arrays]
    stats = f_target_statistics(ys, exp.scale)
    with torch.no_grad():
        baseline = f_encode_projected(model, f_project(model, ys, exp.scale), exp.graph, exp.patches, exp.pos)[0]
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
        del state
    for step in range(start, steps):
        optimizer.zero_grad(set_to_none=True)
        mu, lv = f_encode_projected(model, f_project(model, ys, exp.scale), exp.graph, exp.patches, exp.pos)
        hidden = model.decoder_hidden(mu, lv, sample=False)
        rec = f_exact_reconstruction(hidden, model.decoder[2], ys, stats)
        penalty, details = f_regularization(exp, model, mu, lv, baseline, centers, initial)
        (rec + penalty).backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert torch.isfinite(rec) and torch.isfinite(norm)
        if step == 0:
            audit = {name: float(p.grad.abs().max()) if p.grad is not None else None for name, p in model.named_parameters()}
            assert all((v is not None and np.isfinite(v) and (v > 0) for v in audit.values()))
            f_save(dest / 'gradient_audit.json', audit)
        optimizer.step()
        details.update(step=step + 1, reconstruction=float(rec.detach()), total=float((rec + penalty).detach()))
        history.append(details)
        f_save(out / 'status.json', dict(state='running', stage='personal_finetune', subject=subject, hemisphere=h, K=k, arm=arm, step=step + 1, pid=os.getpid(), updated_unix=time.time(), bounded_exact_FC=True))
        if (step + 1) % 5 == 0 or step == steps - 1:
            temporary = file.with_name(file.name + f'.{os.getpid()}.tmp')
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step + 1, steps=steps, group_sha256=group_hash, history=history), temporary)
            temporary.replace(file)
        del mu, lv, hidden, rec, penalty, norm
    with torch.no_grad():
        mu = f_encode_projected(model, f_project(model, ys, exp.scale), exp.graph, exp.patches, exp.pos)[0]
        centers = 0.5 * exp.group_centers + 0.5 * f_regional_centers(mu.mean(0), exp.labels, exp.core, k)
    changes = {name: float((p - initial[name]).abs().max()) for name, p in model.named_parameters()}
    assert all((v > 0 for v in changes.values()))
    np.save(dest / 'centers.npy', centers.cpu().numpy())
    f_save(dest / 'fit.json', dict(subject=subject, hemisphere=h, K=k, arm=arm, steps=steps, input_shape=[len(arrays), exp.n, exp.n], input_shape_is_mathematical_not_allocated=True, all_FC_entries=True, all_active_parameters_train=True, parameter_changes=changes, history=history, peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2 ** 30, runtime_implementation='validated exact FC factorization; unchanged full reconstruction objective and regularization'))
    del optimizer, initial, baseline, ys, stats, mu
    gc.collect()
    torch.cuda.empty_cache()
    return (model, centers)

@torch.no_grad()
def f_infer(exp, model, centers, arrays, hard=True):
    ys = [torch.from_numpy(y).cuda() for y in arrays]
    mu = f_encode_projected(model, f_project(model, ys, exp.scale), exp.graph, exp.patches, exp.pos)[0]
    cost = f_distance(mu.mean(0), centers).cpu().numpy()
    del mu, ys
    torch.cuda.empty_cache()
    return f_decode(exp.initial, np.concatenate(arrays), cost, exp.adj, exp.seeds, exp.depth, exp.edges, hard_connectivity=hard)

def f_calibration_decode(exp, model, centers, arrays):
    with torch.no_grad():
        ys = [torch.from_numpy(y).cuda() for y in arrays]
        mu = f_encode_projected(model, f_project(model, ys, exp.scale), exp.graph, exp.patches, exp.pos)[0]
        cost = f_distance(mu.mean(0), centers).cpu().numpy()
        del mu, ys
    result = {}
    audit = {}
    y = np.concatenate(arrays)
    for name, hard in [('CoBRA', True), ('no_hard_connectivity', False)]:
        result[name], audit[name] = f_decode(exp.initial, y, cost, exp.adj, exp.seeds, exp.depth, exp.edges, hard_connectivity=hard)
    torch.cuda.empty_cache()
    return (result, audit)
