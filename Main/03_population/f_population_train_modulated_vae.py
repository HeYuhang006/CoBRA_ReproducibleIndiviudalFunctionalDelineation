from f_runtime import f_work_root
"""Scalable, resumable group-prior graph/transformer VAE training on GPU 0.

The original full500 MLP workflow is independent. This trainer streams subjects,
uses a REST1 training-group prior, and fine-tunes on independent REST1/REST2 FC.
"""
import os
from pathlib import Path
SOURCE_DIR = f_work_root() / 'final/population_training'
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import nibabel as nib
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
import torch
from torch.nn import functional as F
from f_population_audit_modulated_cohort import f_audit, RUNS
from f_population_drfc import f_parcel_components
from f_population_run import ROOT, f_filtered_timeseries, f_get_mapping
from f_population_surface_prior_vae import SurfacePriorVAE, f_diagonal_kl_to_prior, f_prototype_probabilities, f_soft_homogeneity
GEOMETRY = ROOT / 'runs/pilot32_MSMAll_samplez'

def f_json_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)

def f_array_save(path, value):
    temporary = Path(str(path) + '.partial')
    with temporary.open('wb') as output:
        np.save(output, value)
    temporary.replace(path)

def f_checkpoint_save(path, value):
    temporary = path.with_name(path.name + '.partial')
    torch.save(value, temporary)
    temporary.replace(path)

def f_streaming_reference(arrays):
    """Group FC and residual feature scales, without stacking subjects."""
    count, total, second = (0, None, None)
    for array in arrays:
        x = np.asarray(array, dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite reference input')
        if total is None:
            total, second = (np.zeros_like(x), np.zeros(x.shape[-1]))
        total += x
        second += np.square(x).mean(0)
        count += 1
    if count < 2:
        raise ValueError('At least two training subjects are required')
    group = total / count
    variance = second / count - np.square(group).mean(0)
    return (group.astype(np.float32), np.sqrt(np.maximum(variance, 0.03 ** 2)).astype(np.float32))

def f_leave_one_out(group, own_rest1, count):
    if count < 2:
        raise ValueError('At least two training subjects are required')
    return (count * group - own_rest1) / (count - 1)

class Experiment:

    def __init__(self, config_path):
        self.c = json.loads(Path(config_path).read_text())
        self.folder = ROOT / 'runs' / self.c['name']
        self.folder.mkdir(parents=True, exist_ok=True)
        self.status_path = ROOT / 'logs' / (self.c['name'] + '.status.json')
        self.lock = (self.folder / 'workflow.lock').open('a')
        fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != str(self.c['gpu']):
            raise RuntimeError('Device isolation mismatch; use --launch or explicitly select configured GPU')
        if self.c['bandpass_hz'] != [0.01, 0.1] or self.c['distance_residualization']:
            raise RuntimeError('This input implementation requires raw FC after 0.01-0.1 Hz filtering')
        if self.c['individual_sessions'] != RUNS or self.c['group_reference_sessions'] != [RUNS[0]]:
            raise RuntimeError('Session definitions do not match the implemented reference protocol')
        saved = self.folder / 'configuration.json'
        if saved.exists() and json.loads(saved.read_text()) != self.c:
            raise RuntimeError('Changed configuration: use a new run name')
        f_json_save(saved, self.c)
        hashes = {p: hashlib.sha256((SOURCE_DIR / p).read_bytes()).hexdigest() for p in ['f_population_train_modulated_vae.py', 'f_population_surface_prior_vae.py', 'f_population_audit_modulated_cohort.py', 'f_population_run.py', 'f_population_drfc.py']}
        saved = self.folder / 'code_hashes.json'
        if saved.exists() and json.loads(saved.read_text()) != hashes:
            raise RuntimeError('Code changed since launch: preserve this experiment and use a new run name')
        f_json_save(saved, hashes)
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = True
        if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 8 * 1024 ** 3:
            raise RuntimeError('Need an available configured GPU with at least 8 GiB free')
        if shutil.disk_usage(ROOT).free < 400 * 1024 ** 3:
            raise RuntimeError('Need at least 400 GiB free for filtered caches and outputs')
        self.status('cohort_audit')
        saved = self.folder / 'subjects.json'
        if saved.exists():
            self.subjects = json.loads(saved.read_text())
        else:
            self.subjects = f_audit({part: self.c[part + '_limit'] for part in ['train', 'validation']})
            if min((len(self.subjects[p]) for p in ['train', 'validation'])) < 2:
                raise RuntimeError('Insufficient usable train/validation subjects')
            f_json_save(saved, self.subjects)
        print('COHORT', json.dumps({p: len(self.subjects[p]) for p in ['train', 'validation']}), flush=True)
        self.mapping = f_get_mapping(self.subjects['train'][0]['paths'][0])
        reference = np.load(GEOMETRY / 'mapping.npz')
        for h in self.c['hemispheres']:
            if not np.array_equal(self.mapping[h]['vertices'], reference[h + '_vertices']):
                raise RuntimeError('CIFTI/mesh mapping mismatch')
        self.graphs, self.pooling = ({}, {})
        cache_hash = hashlib.sha256(hashes['f_population_run.py'].encode())
        cache_hash.update(json.dumps({'filter': self.c['bandpass_hz'], 'runs': RUNS, 'cache_version': 1}, sort_keys=True).encode())
        for h in self.c['hemispheres']:
            positions = np.load(GEOMETRY / f'lpe_{h}.npy').astype(np.float32)
            edges = np.load(GEOMETRY / f'edges_{h}.npy').astype(np.int64)
            n = len(positions)
            existing = self.folder / f'anchor_patches_{h}.npy'
            if existing.exists():
                patches = np.load(existing)
            elif h == 'L' and self.c['anchors'] == 128:
                patches = np.load(ROOT / 'runs/surface_prior_architecture_pilot/anchor_patches.npy')
                f_array_save(existing, patches)
            else:
                patches = MiniBatchKMeans(self.c['anchors'], random_state=self.c['seed'], batch_size=4096, n_init=3).fit_predict(positions).astype(np.int64)
                f_array_save(existing, patches)
            counts = np.bincount(patches, minlength=self.c['anchors'])
            if len(counts) != self.c['anchors'] or (counts == 0).any():
                raise RuntimeError('Invalid anchor assignment')
            self.pooling[h] = coo_matrix((1 / counts[patches], (np.arange(n), patches)), shape=(n, self.c['anchors'])).tocsr()
            rows, cols = (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])
            degree = np.bincount(rows, minlength=n)
            if (degree == 0).any():
                raise RuntimeError('Isolated cortical vertex')
            adjacency = torch.sparse_coo_tensor(np.stack([rows, cols]), (1 / degree[rows]).astype(np.float32), (n, n)).coalesce().cuda()
            self.graphs[h] = (adjacency, torch.from_numpy(patches).cuda(), torch.from_numpy(positions).cuda(), edges)
            cache_hash.update(h.encode())
            cache_hash.update(patches.astype(np.int64).tobytes())
            cache_hash.update(self.mapping[h]['vertices'].tobytes())
        self.cache = ROOT / 'resources/individual_fc128_crossday_cache' / cache_hash.hexdigest()[:20]
        self.cache.mkdir(parents=True, exist_ok=True)
        f_json_save(self.folder / 'cache_definition.json', {'directory': str(self.cache), 'fingerprint': cache_hash.hexdigest(), 'timecourses_dtype': 'float16', 'feature_dtype': 'float32', 'filter_hz': [0.01, 0.1]})

    def status(self, stage, **fields):
        f_json_save(self.status_path, {'state': 'running', 'pid': os.getpid(), 'gpu': self.c['gpu'], 'stage': stage, 'updated_at_unix': time.time(), **fields})

    def scan_path(self, row, scan):
        return self.cache / row['subject'] / RUNS[scan]

    def ensure_scan(self, row, scan):
        folder = self.scan_path(row, scan)
        source = Path(row['paths'][scan])
        stat = source.stat()
        identity = {'source': str(source), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        marker = folder / 'complete.json'
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved['source_identity'] != identity:
                raise RuntimeError('Source changed after cache creation')
            if all(((folder / f'{h}_{kind}.npy').exists() for h in self.c['hemispheres'] for kind in ['features', 'timeseries'])):
                return folder
        folder.mkdir(parents=True, exist_ok=True)
        ts, provenance = f_filtered_timeseries({'subject': row['subject'], 'path': str(source)}, self.mapping, {'use_verified_practice_cache': False})
        image = nib.load(source)
        if ts.shape[0] != 1200 or abs(image.header.get_axis(0).step - 0.72) > 1e-05:
            raise RuntimeError('Scan duration/TR changed after cohort audit')
        for h in self.c['hemispheres']:
            offset = 0 if h == 'L' else len(self.mapping['L']['vertices'])
            n = len(self.mapping[h]['vertices'])
            y = ts[:, offset:offset + n].copy()
            y -= y.mean(0)
            sd = y.std(0)
            if not np.isfinite(y).all() or (sd < 1e-07).any():
                raise RuntimeError('Constant/nonfinite cortical signal')
            y /= sd
            anchor = np.asarray(y @ self.pooling[h], dtype=np.float32)
            anchor -= anchor.mean(0)
            anchor_sd = anchor.std(0)
            if (anchor_sd < 1e-07).any():
                raise RuntimeError('Constant anchor signal')
            anchor /= anchor_sd
            fc = (y.T @ anchor / len(y)).astype(np.float32)
            if not np.isfinite(fc).all() or abs(fc).max() > 1.001:
                raise RuntimeError('Invalid FC feature')
            f_array_save(folder / f'{h}_features.npy', fc)
            f_array_save(folder / f'{h}_timeseries.npy', y.astype(np.float16))
        f_json_save(marker, {'source_identity': identity, 'provenance': provenance, 'subject': row['subject'], 'scan': RUNS[scan], 'filter_hz': [0.01, 0.1], 'hemispheres': self.c['hemispheres'], 'shape_timepoints': ts.shape[0]})
        return folder

    def load_features(self, row, h, pair=True):
        scans = [0, 1] if pair else [0]
        data = [np.load(self.ensure_scan(row, i) / f'{h}_features.npy') for i in scans]
        return torch.from_numpy(np.stack(data)).cuda()

    def load_timecourses(self, row, h):
        result = []
        for scan in [0, 1]:
            y = torch.from_numpy(np.load(self.ensure_scan(row, scan) / f'{h}_timeseries.npy').astype(np.float32)).cuda()
            result.append((y - y.mean(0)) / y.std(0, unbiased=False))
        return result

    def prepare_reference(self):
        count = len(self.subjects['train'])
        for index, row in enumerate(self.subjects['train']):
            self.status('prepare_group_reference', subject=row['subject'], prepared=index, total=count)
            self.ensure_scan(row, 0)
            if (index + 1) % 10 == 0 or index + 1 == count:
                print('Prepared group REST1', index + 1, '/', count, flush=True)
        for h in self.c['hemispheres']:
            group_path, scale_path = (self.folder / f'group_{h}.npy', self.folder / f'scale_{h}.npy')
            if not (group_path.exists() and scale_path.exists()):
                group, scale = f_streaming_reference((np.load(self.scan_path(r, 0) / f'{h}_features.npy', mmap_mode='r') for r in self.subjects['train']))
                f_array_save(group_path, group)
                f_array_save(scale_path, scale)

    def model(self):
        torch.manual_seed(self.c['seed'])
        return SurfacePriorVAE(self.c['anchors'], self.c['width'], self.c['latent_dim'], self.c['attention_layers'], 'graph_transformer').cuda()

    @staticmethod
    def rng_state():
        return {'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state()}

    @staticmethod
    def restore_rng(state):
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'])

    def fit_group(self, h):
        folder = self.folder / h
        folder.mkdir(exist_ok=True)
        checkpoint = folder / 'group_checkpoint.pt'
        model = self.model()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.c['learning_rate_group'])
        group = torch.from_numpy(np.load(self.folder / f'group_{h}.npy')).cuda()
        scale = torch.from_numpy(np.load(self.folder / f'scale_{h}.npy')).cuda()
        adjacency, patches, positions, _ = self.graphs[h]
        start = 1
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            self.restore_rng(saved)
            start = saved['step'] + 1
        model.train()
        for step in range(start, self.c['group_steps'] + 1):
            out = model(group, group, scale, adjacency, patches, positions)
            rec = ((out['prediction'][0] - group) / scale).square().mean()
            loss = rec + 0.001 * f_diagonal_kl_to_prior(out['mu'], out['logvar'], torch.zeros_like(out['mu']))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise RuntimeError('Nonfinite group optimization')
            optimizer.step()
            if step % 50 == 0 or step == self.c['group_steps']:
                f_checkpoint_save(checkpoint, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step, **self.rng_state()})
                self.status('group_pretraining', hemisphere=h, step=step, total_steps=self.c['group_steps'], loss=float(loss))
                print('GROUP', h, step, float(loss), flush=True)
        f_json_save(folder / 'group_completed.json', {'steps': self.c['group_steps'], 'subjects': len(self.subjects['train'])})
        prototype_path = folder / 'prototypes.pt'
        if not prototype_path.exists():
            model.eval()
            with torch.no_grad():
                mu = model.encode(group, group, scale, adjacency, patches, positions)[0][0]
                labels = KMeans(self.c['parcels_per_hemisphere'], n_init=5, random_state=self.c['seed']).fit_predict(torch.cat([mu, 0.25 * positions], -1).cpu().numpy())
                labels = torch.as_tensor(labels, device='cuda')
                prototypes = torch.stack([mu[labels == k].mean(0) for k in range(self.c['parcels_per_hemisphere'])])
                proto_pos = torch.stack([positions[labels == k].mean(0) for k in range(self.c['parcels_per_hemisphere'])])
                group_labels = f_prototype_probabilities(mu, prototypes, positions, proto_pos).argmax(-1)
            f_checkpoint_save(prototype_path, {'latent': prototypes.cpu(), 'positions': proto_pos.cpu(), 'group_labels': group_labels.cpu()})

    @torch.no_grad()
    def evaluate(self, model, h, group, scale, prototypes, proto_pos, group_labels, epoch):
        rows = self.subjects['validation']
        adjacency, patches, positions, edges = self.graphs[h]
        model.eval()
        labels, results = ([], [])
        self.status('validation_inference', hemisphere=h, epoch=epoch, subjects=len(rows))
        for index, row in enumerate(rows):
            self.status('validation_inference', hemisphere=h, epoch=epoch, subject=row['subject'], completed=index)
            out = model(self.load_features(row, h), group, scale, adjacency, patches, positions, sample=False)
            labels.append(f_prototype_probabilities(out['mu'], prototypes, positions, proto_pos).argmax(-1))
        group_p = F.one_hot(group_labels, len(prototypes)).float()
        donor_count = min(self.c['donors_per_validation_subject'], len(rows) - 1)
        for index, row in enumerate(rows):
            y = self.load_timecourses(row, h)
            donors = np.random.default_rng(self.c['seed'] + index).choice([j for j in range(len(rows)) if j != index], donor_count, replace=False)
            own, baseline, others = ([], [], [])
            for a, b in [(0, 1), (1, 0)]:
                own.append(float(f_soft_homogeneity(y[b], F.one_hot(labels[index][a], len(prototypes)).float())))
                baseline.append(float(f_soft_homogeneity(y[b], group_p)))
                others.append(float(torch.stack([f_soft_homogeneity(y[b], F.one_hot(labels[j][a], len(prototypes)).float()) for j in donors]).mean()))
            maps = labels[index].cpu().numpy()
            results.append({'subject': row['subject'], 'cross_session_homogeneity': float(np.mean(own)), 'group_homogeneity': float(np.mean(baseline)), 'own_minus_group': float(np.mean(own) - np.mean(baseline)), 'own_minus_donor': float(np.mean(own) - np.mean(others)), 'donors': [rows[j]['subject'] for j in donors], 'same_subject_ARI': float(adjusted_rand_score(maps[0], maps[1])), 'parcels_present': [len(np.unique(m)) for m in maps], 'mean_components': [f_parcel_components(m, edges)['mean_components'] for m in maps]})
        report = {'epoch': epoch, 'hemisphere': h, 'subjects': results}
        for field in ['cross_session_homogeneity', 'own_minus_group', 'own_minus_donor', 'same_subject_ARI']:
            report['mean_' + field] = float(np.mean([r[field] for r in results]))
        report['min_parcels_present'] = min((min(r['parcels_present']) for r in results))
        report['mean_components_per_parcel'] = float(np.mean([r['mean_components'] for r in results]))
        f_array_save(self.folder / h / f'validation_maps_epoch{epoch}.npy', np.asarray([x.cpu().numpy() for x in labels]))
        f_json_save(self.folder / h / f'validation_epoch{epoch}.json', report)
        print('VALIDATION', h, json.dumps({k: v for k, v in report.items() if k != 'subjects'}), flush=True)
        return report

    def fit_individual(self, h):
        folder, rows = (self.folder / h, self.subjects['train'])
        if (folder / 'completed.json').exists():
            return
        adjacency, patches, positions, edges = self.graphs[h]
        model = self.model()
        group_state = torch.load(folder / 'group_checkpoint.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(group_state['model'])
        prior = copy.deepcopy(model).eval().requires_grad_(False)
        group = torch.from_numpy(np.load(self.folder / f'group_{h}.npy')).cuda()
        scale = torch.from_numpy(np.load(self.folder / f'scale_{h}.npy')).cuda()
        proto = torch.load(folder / 'prototypes.pt', map_location='cuda', weights_only=False)
        prototypes, proto_pos = (proto['latent'], proto['positions'])
        optimizer = torch.optim.Adam(model.parameters(), lr=self.c['learning_rate_individual'])
        checkpoint = folder / 'individual_checkpoint.pt'
        start_epoch, next_subject, total, total_count = (1, 0, np.zeros(6), 0)
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            self.restore_rng(saved)
            start_epoch, next_subject = (saved['epoch'], saved['next_subject'])
            total, total_count = (np.array(saved['running_total']), saved['running_count'])
        sampled = torch.as_tensor(edges[::8], device='cuda')
        for epoch in range(start_epoch, self.c['individual_epochs'] + 1):
            model.train()
            order = np.random.default_rng(self.c['seed'] + epoch).permutation(len(rows))
            for offset in range(next_subject, len(rows)):
                row = rows[order[offset]]
                self.status('individual_finetuning', hemisphere=h, epoch=epoch, total_epochs=self.c['individual_epochs'], completed_subjects=offset, total_subjects=len(rows), subject=row['subject'])
                x = self.load_features(row, h)
                reference = f_leave_one_out(group, x[0], len(rows))
                with torch.no_grad():
                    prior_mu = prior.encode(reference, reference, scale, adjacency, patches, positions)[0]
                out = model(x, reference, scale, adjacency, patches, positions)
                self_rec = ((out['prediction'] - x) / scale).square().mean()
                cross_rec = ((out['prediction'] - x.flip(0)) / scale).square().mean()
                kl = f_diagonal_kl_to_prior(out['mu'], out['logvar'], prior_mu)
                align = (out['mu'][0] - out['mu'][1]).square().mean()
                probs = f_prototype_probabilities(out['mu'], prototypes, positions, proto_pos)
                y = self.load_timecourses(row, h)
                hom = -0.5 * (f_soft_homogeneity(y[1], probs[0]) + f_soft_homogeneity(y[0], probs[1]))
                occupancy = F.relu(0.1 / len(prototypes) - probs.mean(1)).square().mean() * len(prototypes) ** 2
                smooth = (probs[:, sampled[:, 0]] - probs[:, sampled[:, 1]]).square().mean()
                ramp = min(1.0, epoch / 8.0)
                loss = 0.5 * (self_rec + cross_rec) + self.c['prior_kl_weight'] * kl + ramp * (self.c['alignment_weight'] * align + self.c['homogeneity_weight'] * hom + 0.01 * occupancy + 0.01 * smooth)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if not torch.isfinite(loss) or not torch.isfinite(norm):
                    raise RuntimeError('Nonfinite individual optimization')
                optimizer.step()
                total += np.array([float(v) for v in [loss, self_rec, cross_rec, kl, align, hom]])
                total_count += 1
                if (offset + 1) % self.c['checkpoint_subjects'] == 0 or offset + 1 == len(rows):
                    f_checkpoint_save(checkpoint, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'next_subject': offset + 1, 'running_total': total.tolist(), 'running_count': total_count, **self.rng_state()})
                    print('INDIVIDUAL', h, epoch, offset + 1, '/', len(rows), 'loss', float(loss), flush=True)
            report = dict(zip(['loss', 'self_mse', 'cross_mse', 'prior_kl', 'alignment', 'negative_homogeneity'], (total / total_count).tolist()))
            report.update(epoch=epoch, hemisphere=h, subjects=total_count)
            f_json_save(folder / f'training_epoch{epoch}.json', report)
            if epoch % self.c['validation_every'] == 0 or epoch == self.c['individual_epochs']:
                if not (folder / f'validation_epoch{epoch}.json').exists():
                    self.evaluate(model, h, group, scale, prototypes, proto_pos, proto['group_labels'], epoch)
                f_checkpoint_save(folder / f'epoch{epoch}.pt', {'model': model.state_dict(), 'epoch': epoch})
            next_subject, total, total_count = (0, np.zeros(6), 0)
        f_json_save(folder / 'completed.json', {'epochs': self.c['individual_epochs'], 'training_subjects': len(rows), 'validation_subjects': len(self.subjects['validation']), 'test_set_used': False})

    def export(self):
        epoch, k = (self.c['individual_epochs'], self.c['parcels_per_hemisphere'])
        rows = self.subjects['validation']
        arrays = {h: np.load(self.folder / h / f'validation_maps_epoch{epoch}.npy') for h in self.c['hemispheres']}
        colors = {0: ('Unassigned', (0.0, 0.0, 0.0, 0.0))}
        rng = np.random.default_rng(self.c['seed'])
        for hi, h in enumerate(self.c['hemispheres']):
            for j in range(k):
                colors[hi * k + j + 1] = (f'VAE_{h}_{j + 1}', tuple(rng.uniform(0.15, 0.9, 3)) + (1.0,))
        output = self.folder / 'validation_parcellations'
        output.mkdir(exist_ok=True)
        files = []
        for index, row in enumerate(rows):
            for scan in [0, 1]:
                mapping = f_get_mapping(row['paths'][scan])
                columns = np.concatenate([mapping[h]['columns'] for h in self.c['hemispheres']])
                axis = nib.load(row['paths'][scan]).header.get_axis(1)[columns]
                values = np.concatenate([arrays[h][index, scan] + 1 + hi * k for hi, h in enumerate(self.c['hemispheres'])])
                label_axis = nib.cifti2.LabelAxis([row['subject'] + '_' + RUNS[scan]], [colors])
                image = nib.Cifti2Image(values[None].astype(np.int32), header=nib.cifti2.Cifti2Header.from_axes((label_axis, axis)))
                image.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_DENSE_LABELS')
                path = output / (row['subject'] + '_' + RUNS[scan] + '.dlabel.nii')
                nib.save(image, path)
                check = nib.load(path)
                if check.shape != (1, len(columns)) or int(check.nifti_header['intent_code']) != 3007:
                    raise RuntimeError('Invalid parcellation export')
                if not np.array_equal(check.get_fdata()[0], values):
                    raise RuntimeError('Parcellation export changed labels')
                files.append(str(path))
        f_json_save(self.folder / 'exports.json', {'files': files, 'cortex_only': True, 'nominal_parcels': k * len(self.c['hemispheres']), 'empty_or_fragmented_parcels_not_hidden': True})

    def run(self):
        self.prepare_reference()
        for h in self.c['hemispheres']:
            self.status('group_pretraining', hemisphere=h, step=0, total_steps=self.c['group_steps'])
            self.fit_group(h)
        for h in self.c['hemispheres']:
            self.fit_individual(h)
        self.status('exporting_validation_parcellations')
        self.export()
        f_json_save(self.status_path, {'state': 'completed', 'pid': os.getpid(), 'gpu': self.c['gpu'], 'updated_at_unix': time.time(), 'test_set_used': False, 'interpretation': 'Training and validation complete; scientific acceptance requires reviewing metrics'})
