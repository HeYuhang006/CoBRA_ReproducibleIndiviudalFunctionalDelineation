from f_runtime import f_work_root
"""Connected LPE1 group priors at K50/100/150/200, fitted on training data only."""
import time
from pathlib import Path
import numpy as np
from f_data import f_save, f_sha
from f_connectivity import f_graph_data, f_connected_assignment, f_polish
from f_group_objective import f_affinities_gpu, f_optimize, f_direct_energy
from f_metrics import f_topology

def f_prepare(workspace, out, hemispheres=('L', 'R'), counts=(50, 100, 150, 200)):
    for h in hemispheres:
        mu_path = workspace / f'runs/vae450_multires_20261006_v1/population_epoch50/latent_mean_{h}.npy'
        mu = np.load(mu_path)
        geo = workspace / 'runs/pilot32_MSMAll_samplez'
        pos = np.load(geo / f'lpe_{h}.npy')
        edges = np.load(geo / f'edges_{h}.npy')
        adj = f_graph_data(len(mu), edges)
        features = np.c_[mu, pos].astype('float32')
        if all(((out / 'priors' / h / str(k) / 'complete.json').exists() for k in counts)):
            for k in counts:
                raw_path = workspace / f'runs/vae450_spatialweight_20261006_v1/labels_LPE1_{h}_{k}.npy'
                expected = {str(p): f_sha(p) for p in [raw_path, mu_path, geo / f'lpe_{h}.npy', geo / f'edges_{h}.npy']}
                assert __import__('f_data').f_read(out / 'priors' / h / str(k) / 'complete.json')['source_sha256'] == expected
            continue
        reference = np.load(workspace / f'runs/fullfc450_crossday_v1/group_{h}.npy', mmap_mode='r')
        fc = np.empty_like(reference)
        for a in range(0, len(mu), 256):
            fc[a:a + 256] = 0.5 * (reference[a:a + 256] + reference[:, a:a + 256].T)
        for k in counts:
            dest = out / 'priors' / h / str(k)
            dest.mkdir(parents=True, exist_ok=True)
            raw_path = workspace / f'runs/vae450_spatialweight_20261006_v1/labels_LPE1_{h}_{k}.npy'
            sources = {str(p): f_sha(p) for p in [raw_path, mu_path, geo / f'lpe_{h}.npy', geo / f'edges_{h}.npy']}
            if (dest / 'complete.json').exists():
                assert __import__('f_data').f_read(dest / 'complete.json')['source_sha256'] == sources
                continue
            f_save(out / 'status.json', dict(state='running', stage='corrected_LPE1_group_prior', hemisphere=h, K=k))
            tick = time.time()
            raw = np.load(raw_path)
            centers = np.stack([features[raw == a].mean(0) for a in range(k)])
            cost = ((features ** 2).sum(1)[:, None] + (centers ** 2).sum(1)[None] - 2 * features @ centers.T).clip(0) / features.shape[1]
            seeds = np.array([ids[np.argmin(cost[ids, a])] for a in range(k) for ids in [np.flatnonzero(raw == a)]], dtype='int32')
            initial = f_connected_assignment(cost, adj, seeds, 0.005, np.ones(adj.nnz, dtype='float32'))
            initial, diag = f_polish(initial, cost, adj, seeds, lam=0.015, sweeps=3)
            counts0 = np.bincount(initial, minlength=k)
            assert counts0.min() > 2, ('tiny connected parcel', h, k, counts0.min())
            affinity = f_affinities_gpu(fc, initial, k)
            minimum = np.minimum(counts0, np.maximum(20, (0.25 * counts0).astype('int64')))
            maximum = np.maximum(counts0, np.full(k, int(np.ceil(4 * len(mu) / k))))
            orders = np.array([np.random.default_rng(20260930 + i).permutation(len(mu)) for i in range(30)], dtype='int32')
            before, Hbefore = f_direct_energy(initial, initial, fc, cost / 0.05, edges, 0.1)
            labels, moves, updated = f_optimize(initial, fc, affinity, cost / 0.05, adj.indptr, adj.indices, seeds, orders, minimum, maximum, 0.1, int(0.3 * len(mu)))
            after, Hafter = f_direct_energy(labels, initial, fc, cost / 0.05, edges, 0.1)
            assert after <= before + 1e-05 and np.array_equal(labels[seeds], np.arange(k))
            qc = f_topology(labels, adj, k)
            assert qc['components'] == [1] * k
            probes = np.unique(np.linspace(0, len(mu) - 1, 32).astype('int64'))
            error = float(np.abs(updated[probes] - fc[probes].astype('float64') @ np.eye(k)[labels]).max())
            assert error < 1e-07
            np.save(dest / 'labels.npy', labels.astype('int16'))
            np.save(dest / 'seeds.npy', seeds)
            f_save(dest / 'complete.json', dict(hemisphere=h, K=k, source_sha256=sources, training_people=450, LPE_weight=1, training_only=True, connected=True, topology=qc, growth_polish=diag, group_energy_before=before, group_energy_after=after, H_training_before=Hbefore, H_training_after=Hafter, moves=moves.tolist(), affinity_error=error, seconds=time.time() - tick, maximum_group_drift=0.3, group_perimeter_weight=0.1))
            print('PRIOR', h, k, 'H', Hbefore, Hafter, 'minimum', qc['minimum'], flush=True)
        del fc
