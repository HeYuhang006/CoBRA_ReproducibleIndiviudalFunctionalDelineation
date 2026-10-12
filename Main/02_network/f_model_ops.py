from f_runtime import f_work_root
"""Complete FC projection, graph/context encoding and blockwise exact loss."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from f_full_fc_vae import FullFCVAE
from f_surface_prior_vae import f_diagonal_kl_to_prior, f_neighbor_mean

def f_transferred_model(source, n):
    """Drop only the unused random residual-input half, preserve learned path."""
    model = copy.deepcopy(source).eval()
    old = model.input[0]
    projection = nn.Linear(n, old.out_features, bias=True).to(old.weight)
    with torch.no_grad():
        projection.weight.copy_(old.weight[:, :n])
        projection.bias.copy_(old.bias)
    model.input[0] = projection
    return model.requires_grad_(False)

def f_encode_projected(model, inputs, adjacency, patches, positions):
    features = inputs
    for layer in model.local_layers:
        features = layer(features, adjacency)
    count = int(patches.max()) + 1
    sizes = torch.bincount(patches, minlength=count).to(features)
    pooled = features.new_zeros(len(features), count, features.shape[-1])
    pooled.index_add_(1, patches, features)
    pos = positions.new_zeros(count, 5)
    pos.index_add_(0, patches, positions)
    pooled = pooled / sizes[None, :, None] + model.position(pos / sizes[:, None])[None]
    features = features + model.context_gate.sigmoid() * model.context_output(model.context(pooled)[:, patches])
    return (model.mu(features), model.logvar(features).clamp(-8, 5))

def f_distance(mu, centers):
    return (mu.square().mean(-1, keepdim=True) + centers.square().mean(-1) - 2 * (mu @ centers.T) / mu.shape[-1]).clamp_min(0) / 0.05

def f_regional_centers(mu, labels, core, k):
    centers = mu.new_zeros(k, mu.shape[-1])
    centers.index_add_(0, labels[core], mu[core])
    counts = torch.bincount(labels[core], minlength=k)
    assert (counts > 0).all()
    return centers / counts[:, None]

def f_materialize(signals, scale, transform=None, chunk=256):
    n = len(scale)
    out = scale.new_empty(len(signals), n, n)
    with torch.no_grad():
        for j, y in enumerate(signals):
            for a in range(0, n, chunk):
                b = min(a + chunk, n)
                v = y.T @ y[:, a:b] / len(y)
                if transform is not None:
                    v = transform.columns(v, a, b)
                v = v / scale[a:b]
                assert torch.isfinite(v).all()
                out[j, :, a:b].copy_(v)
    assert out.is_contiguous()
    return out

def f_reconstruction_backward(model, hidden, fc, cross=True, chunk=256):
    """Exact mean loss over all ordered distinct-session pairs and FC entries."""
    leaf = hidden.detach().requires_grad_(True)
    linear = model.decoder[2]
    B, N, _ = fc.shape
    total = 0.0
    assert not cross or B > 1
    for start in range(0, N, chunk):
        stop = min(start + chunk, N)
        prediction = F.linear(leaf, linear.weight[start:stop], linear.bias[start:stop])
        for i in range(B):
            target = fc[[j for j in range(B) if j != i], :, start:stop] if cross else fc[i:i + 1, :, start:stop]
            loss = (prediction[i:i + 1] - target).square().sum() / (B * len(target) * N * N)
            loss.backward(retain_graph=i < B - 1)
            total += float(loss.detach())
    assert leaf.grad is not None and torch.isfinite(leaf.grad).all()
    return (total, leaf.grad.detach())

def f_regularization(exp, model, mu, lv, baseline, centers, initial):
    B = len(mu)
    align = 2 * B / (B - 1) * (mu - mu.mean(0, keepdim=True)).square().mean() if B > 1 else mu.sum() * 0
    drift = (mu - baseline).square().mean()
    kl = f_diagonal_kl_to_prior(mu, lv, exp.group_mu, 1.0)
    logits = -f_distance(mu, centers)
    ce = sum((F.cross_entropy(logits[b, exp.core], exp.labels[exp.core]) for b in range(B))) / B
    probs = logits.softmax(-1)
    smooth = (probs - f_neighbor_mean(exp.graph, probs)).square().sum(-1).mean()
    param = torch.stack([(p - initial[name]).square().mean() for name, p in model.named_parameters()]).mean()
    penalty = 0.05 * align + 0.01 * drift + 0.005 * kl + 0.02 * ce + 0.02 * smooth + 0.01 * param
    return (penalty, {name: float(value.detach()) for name, value in dict(alignment=align, feature_drift=drift, group_kl=kl, core_ce=ce, smoothness=smooth, parameter_drift=param).items()})
