from f_runtime import f_work_root
"""Mechanistic topology, FC weighting and ICC formula tests."""
import numpy as np
from f_connectivity import f_boundary_optimize, f_graph_data, f_energy
from f_application_data import f_fd, f_qc_segment
from f_application_statistics import f_edge_icc
from f_application_metrics import f_regional, f_connectome

def f_test_topology_switch():
    edges = np.array([(i, i + 1) for i in range(11)] + [(4, 8)], dtype='int32')
    adj = f_graph_data(12, edges)
    labels = np.array([0] * 8 + [1] * 4, dtype='int16')
    rng = np.random.default_rng(18)
    f = rng.normal(size=(12, 40)).astype('float32')
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    cost = np.zeros((12, 2))
    cost[4, 0] = 1000000.0
    seeds = np.array([0, 8])
    orders = np.array([[4]], dtype='int32')
    args = (labels, f, cost, adj.indptr, adj.indices, adj.data, seeds, orders, np.ones(12, dtype=bool), np.array([2, 2]), np.array([12, 12]), 0.02)
    hard, h = f_boundary_optimize(*args, True)
    soft, s = f_boundary_optimize(*args, False)
    assert hard[4] == 0 and soft[4] == 1 and (h[0] == 0) and (s[0] == 1)
    from scipy.sparse.csgraph import connected_components
    assert connected_components(adj[soft == 0][:, soft == 0], directed=False)[0] == 2
    assert np.array_equal(hard[seeds], np.arange(2)) and np.array_equal(soft[seeds], np.arange(2))
    assert f_energy(soft, labels, f, cost, edges, np.ones(len(edges)), 0.02) < f_energy(labels, labels, f, cost, edges, np.ones(len(edges)), 0.02)

def f_direct_icc(a):
    n, k = a.shape
    grand = a.mean()
    ss_subject = k * ((a.mean(1) - grand) ** 2).sum()
    ss_visit = n * ((a.mean(0) - grand) ** 2).sum()
    err = a - a.mean(1)[:, None] - a.mean(0)[None] + grand
    ms_s = ss_subject / (n - 1)
    ms_v = ss_visit / (k - 1)
    ms_e = (err ** 2).sum() / ((n - 1) * (k - 1))
    return (ms_s - ms_e) / (ms_s + (k - 1) * ms_e + k * (ms_v - ms_e) / n)

def f_test_icc():
    rng = np.random.default_rng(29)
    x = rng.normal(size=(23, 9))
    y = 0.7 * x + rng.normal(size=x.shape)
    np.testing.assert_allclose(f_edge_icc(x, y), [f_direct_icc(np.c_[x[:, i], y[:, i]]) for i in range(9)], atol=1e-12)
    np.testing.assert_allclose(f_edge_icc(x, x), 1, atol=1e-12)
    assert np.all(f_edge_icc(x, x + 5) < f_edge_icc(x, x))

def f_test_FC_and_FD():
    p = np.zeros((1200, 12))
    p[1:, 0] = 1
    p[1:, 3] = 1
    d = f_fd(p)
    np.testing.assert_allclose(d[1], 1 + 50 * np.pi / 180)
    assert d[0] == 0
    assert f_qc_segment(d, 0, 480)['passed']
    rng = np.random.default_rng(51)
    y = rng.normal(size=(100, 4)).astype('float32')
    labels = np.array([0, 0, 1, 1])
    area = np.array([1, 3, 2, 2])
    z = f_regional(y, labels, area, 2)
    np.testing.assert_allclose(z[:, 0], (y[:, 0] + 3 * y[:, 1]) / 4, atol=1e-07)
    got = f_connectome(z, z)
    assert len(got) == 6 and np.isfinite(got).all()
