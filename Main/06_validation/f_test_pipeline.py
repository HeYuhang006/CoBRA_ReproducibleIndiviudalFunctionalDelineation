from f_runtime import f_work_root
"""Numerical checks of4-run gradients, K-generic centers and topology safety."""
import copy
import numpy as np
import torch
from torch import nn
from f_model_ops import f_reconstruction_backward, f_regional_centers
from f_connectivity import f_graph_data, f_boundary_depth, f_decode
from f_metrics import f_topology

def f_test_cross_gradients():
    torch.manual_seed(17)
    for B in [2, 4]:
        n = 13
        model = nn.Module()
        model.decoder = nn.Sequential(nn.Linear(3, 5), nn.GELU(), nn.Linear(5, n)).double()
        clone = copy.deepcopy(model)
        h = torch.randn(B, n, 5, dtype=torch.float64, requires_grad=True)
        h2 = h.detach().clone().requires_grad_(True)
        fc = torch.randn(B, n, n, dtype=torch.float64)
        prediction = model.decoder[2](h)
        loss = sum(((prediction[i] - fc[j]).square().mean() for i in range(B) for j in range(B) if j != i)) / (B * (B - 1))
        loss.backward()
        got, g = f_reconstruction_backward(clone, h2, fc, cross=True, chunk=4)
        h2.backward(g)
        assert abs(got - float(loss.detach())) < 1e-10
        torch.testing.assert_close(h.grad, h2.grad)
        for a, b in zip(model.decoder[2].parameters(), clone.decoder[2].parameters()):
            torch.testing.assert_close(a.grad, b.grad)
        mu = torch.randn(B, n, 3)
        computed = 2 * B / (B - 1) * (mu - mu.mean(0, keepdim=True)).square().mean()
        explicit = sum(((mu[i] - mu[j]).square().mean() for i in range(B) for j in range(i + 1, B))) / (B * (B - 1) / 2)
        torch.testing.assert_close(computed, explicit)
    print('PASS2/4-run all-pair FC loss, hidden/decoder gradients and alignment', flush=True)

def f_test_k_generic():
    for k in [50, 100, 150, 200]:
        labels = torch.arange(k).repeat_interleave(2)
        mu = torch.arange(2 * k * 3).reshape(2 * k, 3).float()
        got = f_regional_centers(mu, labels, torch.ones(2 * k, dtype=torch.bool), k)
        torch.testing.assert_close(got, mu.reshape(k, 2, 3).mean(1))
    n = 108
    edges = []
    for v in range(n):
        if v % 12 < 11:
            edges.append((v, v + 1))
        if v < 96:
            edges.append((v, v + 12))
        if v < 96 and v % 12 < 11:
            edges.append((v, v + 13))
    edges = np.array(edges)
    adj = f_graph_data(n, edges)
    initial = np.array([v % 12 // 4 for v in range(n)], dtype='int16')
    seeds = np.array([36, 40, 44])
    rng = np.random.default_rng(4)
    signals = rng.normal(size=(80, n)).astype('float32')
    signals -= signals.mean(0)
    signals /= signals.std(0)
    labels, a = f_decode(initial, signals, np.zeros((n, 3)), adj, seeds, f_boundary_depth(initial, adj), edges, sweeps=3)
    assert f_topology(labels, adj, 3)['components'] == [1, 1, 1] and np.array_equal(labels[seeds], np.arange(3))
    assert a['energy_after'] <= a['energy_before'] + 0.002
    print('PASS K50/100/150/200 centers and K3 connected sequential decoding', flush=True)
