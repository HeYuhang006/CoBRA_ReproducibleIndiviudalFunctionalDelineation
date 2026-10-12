from f_runtime import f_work_root
"""Memory-bounded, algebraically exact full-FC personal adaptation.

All N*N entries and all ordered distinct-run reconstruction pairs contribute.
No dimensional reduction of FC references, sampled targets, or loss changes.
"""
from pathlib import Path
import sys, copy, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
W = f_work_root()
sys.path.insert(0, str(W / 'final'))
from f_data import f_save, f_sha
from f_pipeline import f_geometry, f_set_prior
from f_full_fc_vae import FullFCVAE
from f_model_ops import f_transferred_model, f_encode_projected, f_regional_centers, f_regularization, f_distance
from f_connectivity import f_decode

def f_project(model, ys, scale):
    linear = model.input[0]
    weighted = linear.weight / scale
    q = torch.stack([y.T @ (y @ weighted.T) / len(y) + linear.bias for y in ys])
    return model.input[2](model.input[1](q))

def f_target_statistics(ys, scale):
    result = []
    with torch.no_grad():
        for y in ys:
            s = y / scale
            norm = (y @ y.T * (s @ s.T)).sum(dtype=torch.float64) / len(y) ** 2
            columns = s.T @ y.sum(1) / len(y)
            result.append((s, norm, columns))
    return result

def f_exact_reconstruction(hidden, linear, ys, stats):
    B, N, _ = hidden.shape
    weight, bias = (linear.weight, linear.bias)
    wtw = weight.T @ weight
    wtb = weight.T @ bias
    bb = bias.square().sum(dtype=torch.float64)
    loss = hidden.new_zeros((), dtype=torch.float64)
    for i, h in enumerate(hidden):
        pred2 = (h.T @ h * wtw).sum(dtype=torch.float64) + 2 * (h.sum(0) * wtb).sum(dtype=torch.float64) + N * bb
        for j, y in enumerate(ys):
            if i == j:
                continue
            s, norm, columns = stats[j]
            cross = (y @ h * (s @ weight)).sum(dtype=torch.float64) / len(y) + (bias * columns).sum(dtype=torch.float64)
            loss = loss + (pred2 + norm - 2 * cross) / (B * (B - 1) * N * N)
    return loss

def f_tests():
    torch.manual_seed(19)
    dtype = torch.float64
    n = 29
    ys = [torch.randn(t, n, dtype=dtype) for t in [31, 37]]
    scale = torch.rand(n, dtype=dtype) + 0.5
    hidden = torch.randn(2, n, 7, dtype=dtype, requires_grad=True)
    linear = nn.Linear(7, n, dtype=dtype)
    stats = f_target_statistics(ys, scale)
    analytic = f_exact_reconstruction(hidden, linear, ys, stats)
    predictions = linear(hidden)
    fc = torch.stack([y.T @ y / len(y) / scale for y in ys])
    dense = (predictions - fc.flip(0)).square().mean()
    ga = torch.autograd.grad(analytic, [hidden, linear.weight, linear.bias], retain_graph=True)
    gd = torch.autograd.grad(dense, [hidden, linear.weight, linear.bias])
    assert abs(float(analytic - dense)) < 1e-12
    assert all((float((a - b).abs().max()) < 1e-12 for a, b in zip(ga, gd)))
    projection = nn.Linear(n, 11, dtype=dtype)
    p1 = torch.stack([y.T @ (y @ (projection.weight / scale).T) / len(y) + projection.bias for y in ys])
    p2 = projection(fc)
    a = torch.autograd.grad(p1.square().mean(), [projection.weight, projection.bias], retain_graph=True)
    b = torch.autograd.grad(p2.square().mean(), [projection.weight, projection.bias])
    assert float((p1 - p2).abs().max()) < 1e-12 and all((float((x - y).abs().max()) < 1e-12 for x, y in zip(a, b)))
    print('PASS complete-FC reconstruction loss and hidden/decoder gradients; full-FC input projection and gradients vs explicit dense matrices', flush=True)

@torch.no_grad()
def f_load_left(source, out):
    exp = f_geometry(W, 'L', out, 'raw')
    model = f_transferred_model(FullFCVAE(exp.n, 128, 32, 2).cuda(), exp.n)
    path = source / 'group/L/checkpoint.pt'
    state = torch.load(path, map_location='cpu', weights_only=False)
    assert state['step'] == 1000 and state['width'] == 128
    model.load_state_dict(state['model'])
    del state
    model.eval().requires_grad_(False)
    raw = np.load(exp.group_source, mmap_mode='r')
    linear = model.input[0]
    acc = linear.bias.expand(exp.n, -1).clone()
    for a in range(0, exp.n, 256):
        b = min(a + 256, exp.n)
        v = torch.as_tensor(np.array(raw[:, a:b], copy=True), device='cuda') / exp.scale[a:b]
        acc.addmm_(v, linear.weight[:, a:b].T)
    inputs = model.input[2](model.input[1](acc))[None]
    exp.group_mu = f_encode_projected(model, inputs, exp.graph, exp.patches, exp.pos)[0].detach()
    del inputs, acc, v
    torch.cuda.empty_cache()
    return (exp, model, path)

def f_validate_real(hidden, linear, ys, scale, stats, path):
    """Full-cortex real-input gradient comparison with blockwise N*N reference."""
    h = hidden.detach().requires_grad_(True)
    analytic = f_exact_reconstruction(h, linear, ys, stats)
    ag = torch.autograd.grad(analytic, [h, linear.weight, linear.bias])
    h2 = hidden.detach().requires_grad_(True)
    linear.zero_grad(set_to_none=True)
    total = 0.0
    B, N, _ = h.shape
    for start in range(0, N, 256):
        stop = min(start + 256, N)
        pred = F.linear(h2, linear.weight[start:stop], linear.bias[start:stop])
        for i in range(B):
            j = 1 - i
            with torch.no_grad():
                target = ys[j].T @ ys[j][:, start:stop] / len(ys[j]) / scale[start:stop]
            loss = (pred[i] - target).square().sum() / (B * N * N)
            loss.backward(retain_graph=i < B - 1)
            total += float(loss.detach())
    bg = [h2.grad, linear.weight.grad, linear.bias.grad]
    errors = [dict(max_absolute=float((a - b).abs().max()), relative_l2=float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1e-12))) for a, b in zip(ag, bg)]
    relative_loss = abs(float(analytic) - total) / max(total, 1e-12)
    assert relative_loss < 0.0001 and all((e['relative_l2'] < 0.0001 for e in errors)), (relative_loss, errors)
    f_save(path, dict(passed=True, full_N_by_N_reference=True, input_shape=[list(y.shape) for y in ys], analytic_loss=float(analytic), blockwise_loss=total, relative_loss_error=relative_loss, gradient_errors=errors, objective='cross-run reconstruction only; all ordered distinct-run pairs; every FC element'))
    linear.zero_grad(set_to_none=True)

def f_adapt(exp, group, path, source, out, subject, calibration, arrays, steps=30):
    dest = out / f'individual/{subject}/L/50/{calibration}'
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'complete.json').exists():
        return
    f_set_prior(exp, source, 'L', 50)
    torch.manual_seed(20260930 + int(subject))
    model = copy.deepcopy(group).eval().requires_grad_(True)
    ys = [torch.as_tensor(y, device='cuda') for y in arrays]
    stats = f_target_statistics(ys, exp.scale)
    with torch.no_grad():
        baseline = f_encode_projected(model, f_project(model, ys, exp.scale), exp.graph, exp.patches, exp.pos)[0]
        centers = 0.5 * exp.group_centers + 0.5 * f_regional_centers(baseline.mean(0), exp.labels, exp.core, 50)
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    history = []
    start = 0
    checkpoint = dest / 'checkpoint.pt'
    group_hash = f_sha(path)
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert state['group_sha256'] == group_hash and state['steps'] == steps
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        history = state['history']
        start = state['step']
    for step in range(start, steps):
        optimizer.zero_grad(set_to_none=True)
        mu, lv = f_encode_projected(model, f_project(model, ys, exp.scale), exp.graph, exp.patches, exp.pos)
        hidden = model.decoder_hidden(mu, lv, sample=False)
        if not (out / 'full_cortex_equivalence.json').exists():
            f_validate_real(hidden, model.decoder[2], ys, exp.scale, stats, out / 'full_cortex_equivalence.json')
        rec = f_exact_reconstruction(hidden, model.decoder[2], ys, stats)
        penalty, details = f_regularization(exp, model, mu, lv, baseline, centers, initial)
        (rec + penalty).backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert torch.isfinite(rec) and torch.isfinite(norm)
        if step == 0:
            audit = {n: float(p.grad.abs().max()) if p.grad is not None else None for n, p in model.named_parameters()}
            assert all((v is not None and np.isfinite(v) and (v > 0) for v in audit.values()))
            f_save(dest / 'gradient_audit.json', audit)
        optimizer.step()
        details.update(step=step + 1, reconstruction=float(rec.detach()), total=float((rec + penalty).detach()))
        history.append(details)
        f_save(out / 'status.json', dict(state='running', subject=subject, calibration=calibration, step=step + 1, pid=__import__('os').getpid(), updated_unix=time.time()))
        if (step + 1) % 5 == 0 or step == steps - 1:
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step + 1, steps=steps, history=history, group_sha256=group_hash), checkpoint)
    with torch.no_grad():
        mu = f_encode_projected(model, f_project(model, ys, exp.scale), exp.graph, exp.patches, exp.pos)[0]
        centers = 0.5 * exp.group_centers + 0.5 * f_regional_centers(mu.mean(0), exp.labels, exp.core, 50)
        cost = f_distance(mu.mean(0), centers).cpu().numpy()
    audit = {}
    y = np.concatenate(arrays)
    for name, hard in [('CoBRA', True), ('no_hard_connectivity', False)]:
        labels, audit[name] = f_decode(exp.initial, y, cost, exp.adj, exp.seeds, exp.depth, exp.edges, hard_connectivity=hard)
        np.save(dest / f'{name}.npy', labels)
    f_save(dest / 'decode.json', audit)
    f_save(dest / 'complete.json', dict(subject=subject, calibration=calibration, steps=30, K_bilateral=100, hemisphere='L', full_FC=True, all_ordered_distinct_run_pairs=True, implementation='exact algebraic FC factorization; no approximate reference patches or reconstruction sampling', group_sha256=group_hash, history=history, peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2 ** 30))
    del model, optimizer, initial, baseline, ys, stats, mu, centers
    torch.cuda.empty_cache()
