from f_runtime import f_work_root
import os
from pathlib import Path
SOURCE_DIR = f_work_root() / 'final/population_training'
import argparse
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import multiprocessing as mp
import shutil
import time
import nibabel as nib
import numpy as np
from scipy import signal
from scipy.sparse.csgraph import dijkstra
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
from f_population_drfc import FCVAE, f_gaussian_nll, f_normal_kl, f_neighborhood_nce, f_cortical_graph, f_laplacian_features, f_residualize, f_apply_residualization, f_candidate_pools, f_packed_pools, f_sample_pools, f_functional_homogeneity, f_parcel_components
from f_population_drfc import f_exact_cosine_silhouette, f_dcbc_from_standardized_fc
from f_population_task_metrics import f_task_profiles, f_task_inhomogeneity
ROOT = Path(os.environ['PARCELLATION_WORKSPACE']).resolve()
YA = Path('/dataspace/yuhang_data/HCP-YA')
RETEST = Path('/dataspace/yuhang_data/HCP-Retest')
OLD = Path('/dataspace/yuhang_data/IndividualFunctionalParcellation/practice/outputs/pretrain_cache')
TEMPLATE = Path('/home/yuhang/.cache/templateflow/tpl-fsLR')
_DISTANCE_GRAPH = None
_DISTANCE_OUTPUT = None
_DISTANCE_DENOMINATOR = None

def f_initialize_distance_worker(graph, path, denominator):
    global _DISTANCE_GRAPH, _DISTANCE_OUTPUT, _DISTANCE_DENOMINATOR
    _DISTANCE_GRAPH = graph
    _DISTANCE_OUTPUT = np.load(path, mmap_mode='r+')
    _DISTANCE_DENOMINATOR = denominator

def f_accumulate_distance_rows(begin):
    n = _DISTANCE_GRAPH.shape[0]
    ds = dijkstra(_DISTANCE_GRAPH, directed=False, indices=np.arange(begin, min(begin + 128, n)), limit=50.0)
    np.minimum(ds, 50.0, out=ds)
    _DISTANCE_OUTPUT[begin:begin + len(ds)] += ds.astype(np.float32) / _DISTANCE_DENOMINATOR
    return begin

def f_write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)))
    temporary.replace(path)

def f_inventory():
    rows = []
    reserved = sorted((p.name for p in RETEST.iterdir() if p.is_dir() and p.name.isdigit()))
    for folder in sorted(YA.iterdir()):
        if not folder.name.isdigit():
            continue
        runs = {}
        for run in ['rfMRI_REST1_LR', 'rfMRI_REST1_RL', 'rfMRI_REST2_LR', 'rfMRI_REST2_RL']:
            files = list((folder / 'MNINonLinear/Results' / run).glob('*Atlas_MSMAll_hp2000_clean.dtseries.nii'))
            if files:
                runs[run] = str(files[0])
        if 'rfMRI_REST1_LR' in runs:
            rows.append({'subject': folder.name, 'path': runs['rfMRI_REST1_LR'], 'runs': runs})
    rng = np.random.default_rng(20260930)
    available = [x for x in rows if x['subject'] not in reserved]
    rng.shuffle(available)
    first, last = (available[:500], available[500:])
    test_reserved = [x for x in rows if x['subject'] in reserved]
    ntest = min(50, len(last) + len(test_reserved))
    test = test_reserved + last[-(ntest - len(test_reserved)):]
    validation = last[:len(last) - (ntest - len(test_reserved))]
    split = {'seed': 20260930, 'train': first, 'validation': validation, 'test': test, 'retest_subjects': reserved, 'available_subjects': len(rows), 'family_metadata_available': False, 'paper_split': [500, 100, 400]}
    f_write_json(ROOT / 'manifests/splits.json', split)
    print('Available/split:', len(rows), len(first), len(validation), len(test), flush=True)
    return split

def f_config_and_dir(path):
    c = json.loads(Path(path).read_text())
    c.setdefault('seed', 20260930)
    signature = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()
    folder = ROOT / 'runs' / c['name']
    folder.mkdir(parents=True, exist_ok=True)
    meta = folder / 'configuration.json'
    if meta.exists() and json.loads(meta.read_text())['signature'] != signature:
        raise RuntimeError('Configuration changed: use a new run name, never mix artifacts.')
    f_write_json(meta, {'configuration': c, 'signature': signature})
    return (c, folder, signature)

def f_get_mapping(path):
    image = nib.load(path)
    axis = image.header.get_axis(1)
    result = {}
    for hemi, name in [('L', 'CIFTI_STRUCTURE_CORTEX_LEFT'), ('R', 'CIFTI_STRUCTURE_CORTEX_RIGHT')]:
        ids = np.flatnonzero(axis.surface_mask & (axis.name == name))
        result[hemi] = {'columns': ids, 'vertices': axis.vertex[ids].astype(np.int32)}
    return result

def f_filtered_timeseries(row, mapping, config):
    image = nib.load(row['path'])
    series = image.header.get_axis(0)
    tr = float(series.step)
    actual = f_get_mapping(row['path'])
    for h in ['L', 'R']:
        if not np.array_equal(mapping[h]['vertices'], actual[h]['vertices']):
            raise RuntimeError('CIFTI cortical vertex mapping mismatch: ' + row['subject'])
    columns = np.concatenate([actual['L']['columns'], actual['R']['columns']])
    cache = OLD / (row['subject'] + '.npy')
    sos = signal.butter(3, [0.01, 0.1], btype='bandpass', fs=1 / tr, output='sos')
    pad = int(3 / tr / 0.01) | 1
    pad = min(pad, image.shape[0] - 1)

    def clean(x):
        x = signal.detrend(x.astype(np.float64), axis=0)
        x = signal.sosfiltfilt(sos, x, axis=0, padlen=pad)
        return ((x - x.mean(0)) / np.maximum(x.std(0), 1e-08)).astype(np.float32)
    if False and cache.exists():
        stored = np.load(cache, mmap_mode='r')
        if stored.shape != (image.shape[0], len(columns)):
            raise RuntimeError('Cached data dimensions do not match source')
        select = np.linspace(0, len(columns) - 1, 16, dtype=int)
        original = np.stack([np.asarray(image.dataobj[:, int(columns[i])]) for i in select], 1)
        expected = clean(original)
        error = float(np.max(np.abs(expected - stored[:, select])))
        if error > 0.005:
            raise RuntimeError(f"Practice cache fails source/filter check: {row['subject']} error={error}")
        return (np.asarray(stored, dtype=np.float32), {'source': row['path'], 'cache': str(cache), 'cache_check_max_error': error, 'tr': tr})
    data = np.asanyarray(image.dataobj, dtype=np.float32)[:, columns]
    out = np.empty_like(data)
    for begin in range(0, len(columns), 4096):
        out[:, begin:begin + 4096] = clean(data[:, begin:begin + 4096])
    return (out, {'source': row['path'], 'cache': None, 'tr': tr})

def f_mean_fc(rows, hemi, mapping, output, config):
    n = len(mapping[hemi]['vertices'])
    offset = 0 if hemi == 'L' else len(mapping['L']['vertices'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = True
    acc = torch.zeros((n, n), device=device, dtype=torch.float32)
    records = []
    start = time.time()
    for i, row in enumerate(rows):
        ts, provenance = f_filtered_timeseries(row, mapping, config)
        ts = ts[:, offset:offset + n]
        ts = ts - ts.mean(0)
        scale = np.sqrt((ts * ts).mean(0))
        if (scale <= 1e-07).any() or not np.isfinite(ts).all():
            raise RuntimeError('Non-finite or constant cortical signals; review QC instead of fitting silently.')
        ts /= scale
        x = torch.from_numpy(np.ascontiguousarray(ts)).to(device)
        acc.addmm_(x.T, x, beta=1, alpha=1 / len(ts))
        records.append({'subject': row['subject'], **provenance})
        del ts, x
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            print(f'FC {hemi} {output.name}: {i + 1}/{len(rows)} elapsed={time.time() - start:.1f}s', flush=True)
    acc /= len(rows)
    temporary = output.with_name(output.stem + '.partial.npy')
    np.save(temporary, acc.cpu().numpy())
    temporary.replace(output)
    del acc
    torch.cuda.empty_cache()
    f_write_json(output.with_suffix('.provenance.json'), records)

def f_fetch_surfaces(subjects, config):
    destination = ROOT / 'resources/subject_surfaces'
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    try:
        import boto3
        from botocore.config import Config
        client = boto3.client('s3', config=Config(connect_timeout=10, read_timeout=30, retries={'max_attempts': 2}))
    except Exception as e:
        print('Surface client unavailable:', type(e).__name__, flush=True)
        client = None

    def fetch(subject):
        status = {'subject': subject, 'files': {}, 'errors': {}}
        for hemi in ['L', 'R']:
            registration = '_MSMAll' if config.get('surface_registration', 'MSMSulc') == 'MSMAll' else ''
            name = f'{subject}.{hemi}.midthickness{registration}.32k_fs_LR.surf.gii'
            target = destination / name
            if not target.exists() and client is not None:
                key = f'HCP_1200/{subject}/MNINonLinear/fsaverage_LR32k/{name}'
                try:
                    client.download_file('hcp-openaccess', key, str(target.with_suffix('.part')))
                    target.with_suffix('.part').replace(target)
                except Exception as e:
                    status['errors'][hemi] = type(e).__name__
            if target.exists():
                g = nib.load(target)
                if g.agg_data('pointset').shape != (32492, 3):
                    raise RuntimeError('Downloaded surface has wrong dimensions')
                status['files'][hemi] = str(target)
        return status
    if subjects:
        results.append(fetch(subjects[0]))
    if results and len(results[0]['files']) == 2:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results.extend(pool.map(fetch, subjects[1:]))
    f_write_json(ROOT / 'manifests/surface_fetch.json', results)
    all_available = len(results) == len(subjects) and all((len(r['files']) == 2 for r in results))
    print('Subject surfaces complete:', all_available, 'subjects_checked:', len(results), flush=True)
    return all_available

def f_distance_matrix(rows, hemi, mapping, folder, config, subject_surfaces):
    vertices = mapping[hemi]['vertices']
    template = TEMPLATE / f'tpl-fsLR_den-32k_hemi-{hemi}_midthickness.surf.gii'
    surface = nib.load(template)
    coords, faces = (surface.agg_data('pointset'), surface.agg_data('triangle'))
    template_graph, edges = f_cortical_graph(coords, faces, vertices)
    np.save(folder / f'edges_{hemi}.npy', edges)
    lpe, values = f_laplacian_features(template_graph, seed=config['seed'])
    np.save(folder / f'lpe_{hemi}.npy', lpe)
    f_write_json(folder / f'lpe_{hemi}.json', {'eigenvalues': values, 'source': str(template), 'max_zscore_mean': abs(lpe.mean(0)).max()})
    output = folder / f'distances_{hemi}.npy'
    if output.exists() and output.with_suffix('.done.json').exists():
        return np.load(output, mmap_mode='r')
    n = len(vertices)
    partial = output.with_name(f'distances_{hemi}.partial.npy')
    mmap = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float32, shape=(n, n))
    mmap[:] = 0
    inputs = rows if subject_surfaces else [{'subject': 'template'}]
    checkpoint_metadata = folder / f'distances_{hemi}.checkpoint.json'
    completed_subjects = 0
    last_checkpoint = None
    if checkpoint_metadata.exists():
        progress = json.loads(checkpoint_metadata.read_text())
        if progress['subjects'] != [x['subject'] for x in inputs]:
            raise RuntimeError('Geodesic checkpoint belongs to a different subject list')
        last_checkpoint = folder / progress['file']
        del mmap
        shutil.copyfile(last_checkpoint, partial)
        mmap = np.load(partial, mmap_mode='r+')
        completed_subjects = progress['completed_subjects']
        print('Resuming geodesics', hemi, completed_subjects, flush=True)
    start = time.time()
    for i, row in enumerate(inputs):
        if i < completed_subjects:
            continue
        if subject_surfaces:
            registration = '_MSMAll' if config.get('surface_registration', 'MSMSulc') == 'MSMAll' else ''
            s = nib.load(ROOT / 'resources/subject_surfaces' / f"{row['subject']}.{hemi}.midthickness{registration}.32k_fs_LR.surf.gii")
            graph, _ = f_cortical_graph(s.agg_data('pointset'), s.agg_data('triangle'), vertices)
        else:
            graph = template_graph
        with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context('fork'), initializer=f_initialize_distance_worker, initargs=(graph, str(partial), len(inputs))) as executor:
            list(executor.map(f_accumulate_distance_rows, range(0, n, 128)))
        mmap.flush()
        print(f'Geodesic {hemi}: {i + 1}/{len(inputs)} elapsed={time.time() - start:.1f}s', flush=True)
        if (i + 1) % 10 == 0 and i + 1 < len(inputs):
            saved = folder / f'distances_{hemi}.checkpoint{i + 1:04d}.npy'
            shutil.copyfile(partial, saved)
            f_write_json(checkpoint_metadata, {'completed_subjects': i + 1, 'file': saved.name, 'subjects': [x['subject'] for x in inputs]})
            if last_checkpoint and last_checkpoint.parent.resolve() == folder.resolve():
                last_checkpoint.unlink(missing_ok=True)
            last_checkpoint = saved
    del mmap
    output.with_name(f'distances_{hemi}.partial.npy').replace(output)
    f_write_json(output.with_suffix('.done.json'), {'mode': 'subject_mean' if subject_surfaces else 'template', 'n_subjects': len(inputs), 'cutoff_mm': 50})
    return np.load(output, mmap_mode='r')

def f_prepare(c, folder):
    split = json.loads((ROOT / 'manifests/splits.json').read_text())
    f_train = split['train'][:c['train_subjects']]
    validation = split['validation'][:c['validation_subjects']]
    test = split['test'][:c['test_subjects']]
    if not all([f_train, validation, test]):
        raise RuntimeError('Each split needs data')
    f_write_json(folder / 'subjects.json', {'train': f_train, 'validation': validation, 'test': test})
    mapping = f_get_mapping(f_train[0]['path'])
    np.savez(folder / 'mapping.npz', **{f'{h}_{k}': v for h, m in mapping.items() for k, v in m.items()})
    subjects = f_fetch_surfaces([x['subject'] for x in f_train], c) if c.get('try_subject_surfaces', True) else False
    if not subjects and (not c.get('allow_template_geometry', False)):
        raise RuntimeError('Subject surfaces unavailable; template fallback is forbidden by configuration.')
    f_write_json(folder / 'geometry.json', {'mode': 'subject_mean' if subjects else 'template', 'surface_registration': c.get('surface_registration', 'MSMSulc'), 'functional_registration': 'MSMAll', 'paper_deviation': None if subjects else 'Template distances replace mean subject distances.'})
    for hemi in c.get('hemispheres', ['L', 'R']):
        distances = f_distance_matrix(f_train, hemi, mapping, folder, c, subjects)
        for name, rows in [('train', f_train), ('validation', validation), ('test', test)]:
            out = folder / f'fc_{name}_{hemi}.npy'
            if c.get('raw_fc_reference') and (not out.exists()):
                reference = Path(c['raw_fc_reference'])
                source_fc = reference / out.name
                source_meta = source_fc.with_suffix('.provenance.json')
                if not source_meta.exists():
                    raise RuntimeError('Requested raw FC reference is incomplete')
                records = json.loads(source_meta.read_text())
                if [(x['subject'], x['source']) for x in records] != [(x['subject'], x['path']) for x in rows]:
                    raise RuntimeError('Raw FC reference subject/run mismatch')
                old_mapping = np.load(reference / 'mapping.npz')
                if not np.array_equal(old_mapping[hemi + '_vertices'], mapping[hemi]['vertices']):
                    raise RuntimeError('Raw FC reference vertex mapping mismatch')
                if np.load(source_fc, mmap_mode='r').shape != (len(mapping[hemi]['vertices']),) * 2:
                    raise RuntimeError('Raw FC reference shape mismatch')
                os.link(source_fc, out)
                shutil.copyfile(source_meta, out.with_suffix('.provenance.json'))
                f_write_json(out.with_suffix('.reuse.json'), {'source': str(source_fc), 'subjects_paths_and_vertices_checked': True, 'geometry_independent': True})
            if not (out.exists() and out.with_suffix('.provenance.json').exists()):
                f_mean_fc(rows, hemi, mapping, out, c)
        train_fc = np.load(folder / f'fc_train_{hemi}.npy', mmap_mode='r')
        stats_file = folder / f'residual_statistics_{hemi}.npz'
        stats = f_residualize(train_fc, distances)
        np.savez(stats_file, **stats)
        for name in ['train', 'validation']:
            out = folder / f'features_{name}_{hemi}.npy'
            if not (out.exists() and out.with_suffix('.done.json').exists()):
                features = np.lib.format.open_memmap(out.with_name(out.stem + '.partial.npy'), mode='w+', dtype=np.float32, shape=train_fc.shape)
                fc = np.load(folder / f'fc_{name}_{hemi}.npy', mmap_mode='r')
                f_apply_residualization(fc, distances, stats, features)
                del features
                out.with_name(out.stem + '.partial.npy').replace(out)
                f_write_json(out.with_suffix('.done.json'), {'statistics': str(stats_file), 'shape': train_fc.shape})
        pool_file = folder / f'pools_{hemi}.npz'
        if not pool_file.exists():
            pos, neg = f_candidate_pools(np.load(folder / f'features_train_{hemi}.npy', mmap_mode='r'), distances)
            pf, po = f_packed_pools(pos)
            nf, no = f_packed_pools(neg)
            np.savez(pool_file, positive_flat=pf, positive_offsets=po, negative_flat=nf, negative_offsets=no)
        print('Prepared hemisphere', hemi, flush=True)
    f_write_json(folder / 'prepare.done.json', {'completed': time.time()})

def f_training_lock(function):

    @wraps(function)
    def locked(c, folder, hemi, *args, **kwargs):
        import fcntl
        with (folder / f'training_{hemi}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return function(c, folder, hemi, *args, **kwargs)
    return locked

@f_training_lock
def f_train(c, folder, hemi, stop_after=None):
    torch.set_num_threads(4)
    torch.manual_seed(c['seed'])
    np.random.seed(c['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this full-resolution training')
    device = torch.device('cuda')
    x = torch.from_numpy(np.load(folder / f'features_train_{hemi}.npy')).to(device)
    xv = np.load(folder / f'features_validation_{hemi}.npy', mmap_mode='r')
    model = FCVAE(x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    pools = np.load(folder / f'pools_{hemi}.npz')
    rng = np.random.default_rng(c['seed'])
    checkpoint = folder / f'checkpoint_{hemi}.pt'
    log_path = folder / f'training_{hemi}.jsonl'
    first = 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state = state['numpy_rng']
        torch.set_rng_state(state['torch_rng'].cpu())
        torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        first = state['epoch']
        del state
    start = time.time()
    target_epoch = min(c['epochs'], stop_after) if stop_after is not None else c['epochs']
    for epoch in range(first + 1, target_epoch + 1):
        model.train()
        beta = 2.0 * min(epoch / 50.0, 1.0)
        gamma = 0.1 if epoch > 50 and c.get('contrastive', True) else 0.0
        ids = rng.permutation(len(x))
        sums = np.zeros(4, np.float64)
        steps = 0
        epoch_start = time.time()
        for begin in range(0, len(ids), 32):
            anchors = ids[begin:begin + 32]
            aidx = torch.as_tensor(anchors, device=device)
            optimizer.zero_grad(set_to_none=True)
            if gamma:
                positive = f_sample_pools(pools['positive_flat'], pools['positive_offsets'], anchors, 6, rng)
                negative = f_sample_pools(pools['negative_flat'], pools['negative_offsets'], anchors, 32, rng)
                all_ids, inverse = np.unique(np.concatenate([anchors, positive.ravel(), negative.ravel()]), return_inverse=True)
                mu, logvar = model.encode(x[torch.as_tensor(all_ids, device=device)])
                z = model.sample(mu, logvar)
                b = len(anchors)
                ai = torch.as_tensor(inverse[:b], device=device)
                pi = torch.as_tensor(inverse[b:b + b * 6].reshape(b, 6), device=device)
                ni = torch.as_tensor(inverse[b + b * 6:].reshape(b, 32), device=device)
                contrast_z = mu if c.get('contrast_latent', 'sample') == 'mean' else z
                nce = f_neighborhood_nce(contrast_z[ai], contrast_z[pi], contrast_z[ni])
                anchor_mu, anchor_logvar, anchor_z = (mu[ai], logvar[ai], z[ai])
            else:
                anchor_mu, anchor_logvar = model.encode(x[aidx])
                anchor_z = model.sample(anchor_mu, anchor_logvar)
                nce = torch.tensor(0.0, device=device)
            prediction = model.decoder(anchor_z)
            reconstruction = f_gaussian_nll(x[aidx], prediction, model.observation_logvar)
            kl = f_normal_kl(anchor_mu, anchor_logvar)
            loss = reconstruction + beta * kl + gamma * nce
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite training loss epoch={epoch} step={steps}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            sums += [loss.item(), reconstruction.item(), kl.item(), nce.item()]
            steps += 1
        model.eval()
        with torch.no_grad():
            diagnostic_ids = np.linspace(0, len(x) - 1, 1024, dtype=int)
            m, lv = model.encode(x[torch.as_tensor(diagnostic_ids, device=device)])
            variances = m.var(0).cpu().numpy()
            diagnostic = {'active_latent_dimensions': int((variances > 0.01).sum()), 'mean_posterior_mu_variance': float(variances.mean()), 'mean_posterior_variance': float(lv.exp().mean().item()), 'observation_logvar_min': float(model.observation_logvar.min().item()), 'observation_logvar_max': float(model.observation_logvar.max().item())}
            validation_nll, validation_kl = (0.0, 0.0)
            for part in np.array_split(diagnostic_ids, 16):
                v = torch.from_numpy(np.array(xv[part], copy=True)).to(device)
                vm, vl = model.encode(v)
                validation_nll += f_gaussian_nll(v, model.decoder(vm), model.observation_logvar).item() / 16
                validation_kl += f_normal_kl(vm, vl).item() / 16
        values = sums / steps
        record = {'epoch': epoch, 'loss': values[0], 'reconstruction': values[1], 'kl': values[2], 'nce': values[3], 'beta': beta, 'gamma': gamma, 'validation_nll_at_mean': validation_nll, 'validation_kl': validation_kl, 'seconds': time.time() - epoch_start, 'elapsed_seconds': time.time() - start, **diagnostic}
        with log_path.open('a') as log:
            log.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        if epoch % c.get('checkpoint_every', 5) == 0 or epoch == target_epoch:
            state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'configuration': c, 'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(), 'versions': {'torch': torch.__version__, 'numpy': np.__version__, 'python': platform.python_version()}}
            torch.save(state, checkpoint.with_suffix('.partial.pt'))
            checkpoint.with_suffix('.partial.pt').replace(checkpoint)
            if epoch in [50, 100, 200]:
                torch.save(state, folder / f'checkpoint_{hemi}_epoch{epoch}.pt')
    if target_epoch == c['epochs']:
        f_write_json(folder / f'train_{hemi}.done.json', {'epochs': c['epochs'], 'completed': time.time()})

def f_cluster(c, folder, hemi):
    torch.set_num_threads(4)
    x = np.load(folder / f'features_train_{hemi}.npy', mmap_mode='r')
    state = torch.load(folder / f'checkpoint_{hemi}.pt', map_location='cpu', weights_only=False)
    model = FCVAE(x.shape[1]).cuda()
    model.load_state_dict(state['model'])
    model.eval()
    means, variances = ([], [])
    with torch.no_grad():
        for begin in range(0, len(x), 128):
            m, v = model.encode(torch.from_numpy(np.array(x[begin:begin + 128], copy=True)).cuda())
            means.append(m.cpu().numpy())
            variances.append(v.exp().cpu().numpy())
    means, variances = (np.concatenate(means), np.concatenate(variances))
    np.save(folder / f'latent_mean_{hemi}.npy', means)
    np.save(folder / f'latent_variance_{hemi}.npy', variances)
    lpe = np.load(folder / f'lpe_{hemi}.npy')
    features = np.concatenate([means, lpe], 1)
    outputs = {}
    for k in c.get('clusters_per_hemisphere', [50, 100, 150, 200]):
        km = KMeans(n_clusters=k, init='k-means++', n_init=1, max_iter=300, tol=0.0001, random_state=c['seed'])
        labels = km.fit_predict(features)
        np.save(folder / f'labels_{hemi}_{k}.npy', labels)
        np.save(folder / f'centroids_{hemi}_{k}.npy', km.cluster_centers_)
        counts = np.bincount(labels, minlength=k)
        outputs[str(k)] = {'iterations': km.n_iter_, 'inertia': km.inertia_, 'min_vertices': int(counts.min()), 'max_vertices': int(counts.max())}
        print('Clustering', hemi, k, outputs[str(k)], flush=True)
    f_write_json(folder / f'cluster_{hemi}.json', {'epoch': state['epoch'], 'latent_dim': 32, 'latent_coordinate_std': means.std(0), 'lpe_coordinate_std': lpe.std(0), 'results': outputs})
    mapping = np.load(folder / 'mapping.npz')
    for k in c.get('clusters_per_hemisphere', [50, 100, 150, 200]):
        labels = np.load(folder / f'labels_{hemi}_{k}.npy')
        values = np.zeros(32492, np.int32)
        values[mapping[f'{hemi}_vertices']] = labels + 1
        gi = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(values, intent='NIFTI_INTENT_LABEL')])
        rng = np.random.default_rng(c['seed'])
        for label in range(k + 1):
            entry = nib.gifti.GiftiLabel(key=label)
            entry.label = 'MedialWall' if label == 0 else f'DRFC_{hemi}_{label}'
            entry.red, entry.green, entry.blue = (0.0, 0.0, 0.0) if label == 0 else tuple(rng.uniform(0.15, 0.9, 3))
            entry.alpha = 1.0
            gi.labeltable.labels.append(entry)
        nib.save(gi, folder / f'DRFC_{hemi}_{k}.label.gii')

def f_distance_controlled_fc_gap(fc, distances, labels):
    rng = np.random.default_rng(20260930)
    within_sums, between_sums = (np.zeros(20), np.zeros(20))
    within_n, between_n = (np.zeros(20, int), np.zeros(20, int))
    anchors = rng.choice(len(labels), min(3000, len(labels)), replace=False)
    for i in anchors:
        neighbors = np.flatnonzero((distances[i] > 0) & (distances[i] <= 20))
        bins = np.minimum(distances[i, neighbors].astype(int), 19)
        same = labels[i] == labels[neighbors]
        for mask, sums, counts in [(same, within_sums, within_n), (~same, between_sums, between_n)]:
            sums += np.bincount(bins[mask], weights=fc[i, neighbors[mask]], minlength=20)
            counts += np.bincount(bins[mask], minlength=20)
    usable = (within_n > 0) & (between_n > 0)
    weights = within_n[usable] * between_n[usable] / (within_n[usable] + between_n[usable])
    differences = within_sums[usable] / within_n[usable] - between_sums[usable] / between_n[usable]
    return float(np.average(differences, weights=weights))

def f_evaluate(c, folder, hemi):
    torch.set_num_threads(4)
    fc = np.load(folder / f'fc_test_{hemi}.npy', mmap_mode='r')
    distances = np.load(folder / f'distances_{hemi}.npy', mmap_mode='r')
    edges = np.load(folder / f'edges_{hemi}.npy')
    rng = np.random.default_rng(c['seed'])
    selected = np.sort(rng.choice(len(fc), min(2000, len(fc)), replace=False))
    profiles = np.array(fc[selected], copy=True)
    vertices = np.load(folder / 'mapping.npz')[hemi + '_vertices']
    tasks = f_task_profiles(hemi, vertices)
    results = {}
    for k in c.get('clusters_per_hemisphere', [50, 100, 150, 200]):
        labels = np.load(folder / f'labels_{hemi}_{k}.npy')
        scores = {'functional_homogeneity_raw': f_functional_homogeneity(labels, fc), 'sampled_cosine_silhouette_raw': float(silhouette_score(profiles, labels[selected], metric='cosine')), 'cosine_silhouette_raw_exact': f_exact_cosine_silhouette(fc, labels, device='cuda'), 'dcbc_standardized_timecourses': f_dcbc_from_standardized_fc(fc, distances, labels), 'sampled_distance_controlled_direct_fc_gap': f_distance_controlled_fc_gap(fc, distances, labels), **f_parcel_components(labels, edges)}
        if tasks is not None:
            scores.update(f_task_inhomogeneity(labels, tasks))
        results[str(k)] = scores
        print('Evaluation', hemi, k, json.dumps(scores, default=lambda x: x.tolist()), flush=True)
    geometry = json.loads((folder / 'geometry.json').read_text())
    f_write_json(folder / f'evaluation_{hemi}.json', {'results': results, 'geometry': geometry, 'silhouette_n_vertices': len(selected), 'held_out_subjects': c['test_subjects'], 'official_dcbc': 'all-pairs implementation with official weighting; paper maximum distance unspecified', 'task_inhomogeneity': '86 HCP S1200 group contrasts, raw variance scores' if tasks is not None else 'unavailable', 'spin_null_zscores': 'not yet computed', 'table1_reproduced': False})
