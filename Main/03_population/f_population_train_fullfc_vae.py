from f_runtime import f_work_root
"""Independent full-FC ablation: same cohort, graph/attention, losses and parcels.

Uses exact reassociated FC projection and all-entry chunked reconstruction.
Both existing experiments and their immutable caches remain untouched.
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
from sklearn.cluster import KMeans
import torch
from torch.nn import functional as F
from f_population_full_fc_vae import FullFCVAE, f_reconstruction_backward
from f_population_surface_prior_vae import f_diagonal_kl_to_prior, f_prototype_probabilities, f_soft_homogeneity
from f_population_train_modulated_vae import Experiment, f_json_save, f_array_save, f_checkpoint_save, GEOMETRY
from f_population_run import ROOT, f_get_mapping, f_filtered_timeseries
from f_population_audit_modulated_cohort import RUNS

class FullExperiment(Experiment):

    def __init__(self, config_path):
        self.c = json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
        self.folder = ROOT / 'runs' / self.c['name']
        self.folder.mkdir(parents=True, exist_ok=True)
        self.status_path = ROOT / 'logs' / (self.c['name'] + '.status.json')
        self.lock = (self.folder / 'workflow.lock').open('a')
        fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != str(self.c['gpu']):
            raise RuntimeError('GPU isolation mismatch')
        if self.c['bandpass_hz'] != [0.01, 0.1]:
            raise RuntimeError('Expected 0.01-0.1 Hz filtering')
        if self.c['distance_residualization'] and type(self) is FullExperiment:
            raise RuntimeError('Use the distance-corrected trainer for this configuration')
        if self.c['individual_sessions'] != RUNS or self.c['group_reference_sessions'] != [RUNS[0]]:
            raise RuntimeError('Session mismatch')
        torch.set_num_threads(2)
        torch.manual_seed(self.c['seed'])
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.set_per_process_memory_fraction(self.c['gpu_memory_fraction'])
        if torch.cuda.mem_get_info()[0] < self.c['minimum_start_free_GiB'] * 1024 ** 3:
            raise RuntimeError('Insufficient free GPU memory to coexist with existing runs')
        if shutil.disk_usage(ROOT).free < 100 * 1024 ** 3:
            raise RuntimeError('Insufficient disk reserve')
        for name, value in [('configuration.json', self.c), ('code_hashes.json', {name: hashlib.sha256((SOURCE_DIR / name).read_bytes()).hexdigest() for name in ['f_population_full_fc_vae.py', 'f_population_train_fullfc_vae.py', 'f_population_surface_prior_vae.py', 'f_population_train_modulated_vae.py', 'f_population_run.py', 'f_population_drfc.py']})]:
            saved = self.folder / name
            if saved.exists() and json.loads(saved.read_text()) != value:
                raise RuntimeError('Changed experiment definition: ' + name)
            f_json_save(saved, value)
        base = ROOT / 'runs' / self.c['comparison_run']
        frozen = json.loads((base / 'subjects.json').read_text())
        self.subjects = {part: frozen[part][:self.c.get(part + '_limit')] for part in ['train', 'validation']}
        train_ids = {r['subject'] for r in self.subjects['train']}
        validation_ids = {r['subject'] for r in self.subjects['validation']}
        if train_ids & validation_ids or (train_ids | validation_ids) & set(frozen['reserved_retest']):
            raise RuntimeError('Subject split/retest exclusion violation')
        f_json_save(self.folder / 'cohort_provenance.json', {'source': str(base / 'subjects.json'), 'sha256': hashlib.sha256((base / 'subjects.json').read_bytes()).hexdigest(), 'reserved_retest': frozen['reserved_retest'], 'original_test_subjects_untouched': frozen['original_test_subjects_untouched']})
        saved = self.folder / 'subjects.json'
        if saved.exists() and json.loads(saved.read_text()) != self.subjects:
            raise RuntimeError('Frozen cohort changed')
        f_json_save(saved, self.subjects)
        self.base_cache = Path(json.loads((base / 'cache_definition.json').read_text())['directory'])
        self.mapping = f_get_mapping(self.subjects['train'][0]['paths'][0])
        ref = np.load(GEOMETRY / 'mapping.npz')
        self.graphs = {}
        for h in self.c['hemispheres']:
            if not np.array_equal(self.mapping[h]['vertices'], ref[h + '_vertices']):
                raise RuntimeError('Mapping mismatch')
            positions = np.load(GEOMETRY / f'lpe_{h}.npy').astype(np.float32)
            edges = np.load(GEOMETRY / f'edges_{h}.npy').astype(np.int64)
            patches = np.load(base / f'anchor_patches_{h}.npy')
            if int(patches.max()) + 1 != self.c['attention_patches']:
                raise RuntimeError('Attention patch mismatch')
            f_array_save(self.folder / f'attention_patches_{h}.npy', patches)
            rows, cols = (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])
            degrees = np.bincount(rows, minlength=len(positions))
            adjacency = torch.sparse_coo_tensor(np.stack([rows, cols]), (1 / degrees[rows]).astype(np.float32), (len(positions),) * 2).coalesce().cuda()
            self.graphs[h] = (adjacency, torch.from_numpy(patches).cuda(), torch.from_numpy(positions).cuda(), edges)
        fingerprint = hashlib.sha256((base / 'cache_definition.json').read_bytes()).hexdigest()[:20]
        self.own_cache = ROOT / 'resources/fullfc_additional_timecourses' / fingerprint
        f_json_save(self.folder / 'cache_definition.json', {'shared_read_only_cache': str(self.base_cache), 'additional_cache': str(self.own_cache), 'input': 'All same-hemisphere cortical vertices are FC references', 'filter_hz': [0.01, 0.1], 'timeseries_storage_dtype': 'float16, restandardized in float32 after loading', 'full_individual_fc_files_saved': False, 'attention_patch_count': self.c['attention_patches']})
        print('COHORT', {k: len(v) for k, v in self.subjects.items()}, flush=True)

    def ensure_scan(self, row, scan):
        source = Path(row['paths'][scan])
        stat = source.stat()
        identity = {'source': str(source), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        for parent in [self.base_cache, self.own_cache]:
            folder = parent / row['subject'] / RUNS[scan]
            marker = folder / 'complete.json'
            if marker.exists():
                saved = json.loads(marker.read_text())
                if saved['source_identity'] != identity:
                    raise RuntimeError('Source identity changed')
                if saved['filter_hz'] != [0.01, 0.1]:
                    raise RuntimeError('Wrong filter cache')
                if all(((folder / f'{h}_timeseries.npy').exists() for h in self.c['hemispheres'])):
                    return folder
        folder = self.own_cache / row['subject'] / RUNS[scan]
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / 'creation.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            ts, provenance = f_filtered_timeseries({'subject': row['subject'], 'path': str(source)}, self.mapping, {'use_verified_practice_cache': False})
            if ts.shape[0] != 1200 or abs(nib.load(source).header.get_axis(0).step - 0.72) > 1e-05:
                raise RuntimeError('Unexpected timepoints or TR')
            for h in self.c['hemispheres']:
                offset = 0 if h == 'L' else len(self.mapping['L']['vertices'])
                y = ts[:, offset:offset + len(self.mapping[h]['vertices'])].copy()
                y -= y.mean(0)
                sd = y.std(0)
                if not np.isfinite(y).all() or (sd < 1e-07).any():
                    raise RuntimeError('Invalid temporal variance')
                y /= sd
                f_array_save(folder / f'{h}_timeseries.npy', y.astype(np.float16))
            f_json_save(folder / 'complete.json', {'source_identity': identity, 'filter_hz': [0.01, 0.1], 'provenance': provenance, 'subject': row['subject'], 'scan': RUNS[scan]})
        return folder

    def load_signal(self, row, h, scan):
        y = torch.from_numpy(np.load(self.ensure_scan(row, scan) / f'{h}_timeseries.npy').astype(np.float32)).cuda()
        if y.shape != (1200, len(self.mapping[h]['vertices'])) or not torch.isfinite(y).all():
            raise RuntimeError('Invalid timecourse cache shape/values')
        sd = y.std(0, unbiased=False)
        if (sd < 1e-07).any():
            raise RuntimeError('Degenerate signal')
        return (y - y.mean(0)) / sd

    def load_timecourses(self, row, h):
        return [self.load_signal(row, h, scan) for scan in [0, 1]]

    def load_features(self, row, h, pair=True):
        return self.load_timecourses(row, h)

    def prepare_reference(self):
        count = len(self.subjects['train'])
        for h in self.c['hemispheres']:
            done = self.folder / f'group_reference_{h}.done.json'
            if done.exists():
                continue
            n = len(self.mapping[h]['vertices'])
            transform = self.transform_for(h)
            resume = self.folder / f'group_reference_{h}.progress.pt'
            if resume.exists():
                state = torch.load(resume, map_location='cpu', weights_only=False)
                total, second, start = (state['sum'].cuda(), state['second'].cuda(), state['next_subject'])
                del state
            else:
                total, second, start = (torch.zeros(n, n, device='cuda'), torch.zeros(n, dtype=torch.float64, device='cuda'), 0)
            for i in range(start, count):
                row = self.subjects['train'][i]
                self.status('prepare_full_group_fc', hemisphere=h, prepared=i, total=count, subject=row['subject'])
                y = self.load_signal(row, h, 0)
                for a in range(0, n, self.c['fc_block_size']):
                    b = min(a + self.c['fc_block_size'], n)
                    block = y.T @ y[:, a:b] / len(y)
                    if transform is not None:
                        block = transform.columns(block, a, b)
                    total[:, a:b] += block
                    second[a:b] += block.square().mean(0).double()
                del y, block
                if (i + 1) % self.c['reference_checkpoint_subjects'] == 0 or i + 1 == count:
                    f_checkpoint_save(resume, {'sum': total.cpu(), 'second': second.cpu(), 'next_subject': i + 1})
                if (i + 1) % 10 == 0 or i + 1 == count:
                    print('FULL GROUP FC', h, i + 1, '/', count, flush=True)
            total /= count
            scale = torch.empty(n, device='cuda')
            for a in range(0, n, self.c['fc_block_size']):
                b = min(a + self.c['fc_block_size'], n)
                variance = second[a:b] / count - total[:, a:b].double().square().mean(0)
                scale[a:b] = variance.clamp_min(0.03 ** 2).sqrt().float()
            f_array_save(self.folder / f'group_{h}.npy', total.cpu().numpy())
            f_array_save(self.folder / f'scale_{h}.npy', scale.cpu().numpy())
            f_json_save(done, {'subjects': count, 'shape': [n, n], 'reference_sessions': [RUNS[0]], 'bandpass_hz': [0.01, 0.1], 'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 1024 ** 3})
            del total, second, scale
            torch.cuda.empty_cache()

    def transform_for(self, h):
        return None

    def model(self, h):
        torch.manual_seed(self.c['seed'])
        return FullFCVAE(len(self.mapping[h]['vertices']), self.c['width'], self.c['latent_dim'], self.c['attention_layers']).cuda()

    def tensors(self, h):
        return (torch.from_numpy(np.load(self.folder / f'group_{h}.npy')).cuda(), torch.from_numpy(np.load(self.folder / f'scale_{h}.npy')).cuda())

    def fit_group(self, h):
        folder = self.folder / h
        folder.mkdir(exist_ok=True)
        if (folder / 'group_completed.json').exists() and (folder / 'prototypes.pt').exists():
            return
        model = self.model(h)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.c['learning_rate_group'])
        group, scale = self.tensors(h)
        adjacency, patches, positions, _ = self.graphs[h]
        checkpoint = folder / 'group_checkpoint.pt'
        first = 1
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            self.restore_rng(saved)
            first = saved['step'] + 1
            del saved
        for step in range(first, self.c['group_steps'] + 1):
            tick = time.time()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            mu, lv = model.encode_full(None, group, scale, adjacency, patches, positions, group_mode=True)
            hidden = model.decoder_hidden(mu, lv)
            rec, _, grad = f_reconstruction_backward(model, hidden, None, group, scale, self.c['fc_block_size'], group_mode=True)
            regularization = 0.001 * f_diagonal_kl_to_prior(mu, lv, torch.zeros_like(mu))
            torch.autograd.backward([hidden, regularization], [grad, None])
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            loss = rec + float(regularization.detach())
            if not np.isfinite(loss) or not torch.isfinite(norm):
                raise RuntimeError('Nonfinite group training')
            optimizer.step()
            if step % 50 == 0 or step == self.c['group_steps']:
                f_checkpoint_save(checkpoint, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step, **self.rng_state()})
                self.status('group_pretraining', hemisphere=h, step=step, total_steps=self.c['group_steps'], loss=loss, step_seconds=time.time() - tick)
                print('GROUP', h, step, loss, flush=True)
        model.eval()
        with torch.no_grad():
            mu, _ = model.encode_full(None, group, scale, adjacency, patches, positions, group_mode=True)
            mu = mu[0]
            labels = KMeans(self.c['parcels_per_hemisphere'], n_init=5, random_state=self.c['seed']).fit_predict(torch.cat([mu, 0.25 * positions], -1).cpu().numpy())
            labels = torch.as_tensor(labels, device='cuda')
            latent = torch.stack([mu[labels == k].mean(0) for k in range(self.c['parcels_per_hemisphere'])])
            pos = torch.stack([positions[labels == k].mean(0) for k in range(self.c['parcels_per_hemisphere'])])
            assignments = f_prototype_probabilities(mu, latent, positions, pos).argmax(-1)
            f_checkpoint_save(folder / 'prototypes.pt', {'latent': latent.cpu(), 'positions': pos.cpu(), 'group_labels': assignments.cpu()})
        f_json_save(folder / 'group_completed.json', {'steps': self.c['group_steps'], 'subjects': len(self.subjects['train']), 'parameters': sum((p.numel() for p in model.parameters())), 'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 1024 ** 3})
        del model, optimizer, group, scale
        torch.cuda.empty_cache()

    def evaluate(self, model, h, group, scale, prototypes, proto_pos, group_labels, epoch):

        class EvaluationView(torch.nn.Module):

            def __init__(self, base):
                super().__init__()
                self.base = base

            def forward(self, signals, group, scale, adjacency, patches, positions, sample=False):
                mu, lv = self.base.encode_full(signals, group, scale, adjacency, patches, positions)
                return {'mu': mu, 'logvar': lv}
        return super().evaluate(EvaluationView(model), h, group, scale, prototypes, proto_pos, group_labels, epoch)

    def fit_individual(self, h):
        folder, rows = (self.folder / h, self.subjects['train'])
        if (folder / 'completed.json').exists():
            return
        adjacency, patches, positions, edges = self.graphs[h]
        model = self.model(h)
        saved = torch.load(folder / 'group_checkpoint.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'])
        del saved
        prior = copy.deepcopy(model).eval().requires_grad_(False)
        group, scale = self.tensors(h)
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
            del saved
        sampled = torch.as_tensor(edges[::8], device='cuda')
        for epoch in range(start_epoch, self.c['individual_epochs'] + 1):
            model.train()
            order = np.random.default_rng(self.c['seed'] + epoch).permutation(len(rows))
            for offset in range(next_subject, len(rows)):
                tick = time.time()
                row = rows[order[offset]]
                self.status('individual_finetuning', hemisphere=h, epoch=epoch, total_epochs=self.c['individual_epochs'], completed_subjects=offset, total_subjects=len(rows), subject=row['subject'])
                y = self.load_timecourses(row, h)
                with torch.no_grad():
                    prior_mu = prior.encode_full(y, group, scale, adjacency, patches, positions, group_mode=True, leave_out_count=len(rows))[0]
                optimizer.zero_grad(set_to_none=True)
                mu, lv = model.encode_full(y, group, scale, adjacency, patches, positions, leave_out_count=len(rows))
                hidden = model.decoder_hidden(mu, lv)
                rec, cross, grad = f_reconstruction_backward(model, hidden, y, group, scale, self.c['fc_block_size'])
                kl = f_diagonal_kl_to_prior(mu, lv, prior_mu)
                align = (mu[0] - mu[1]).square().mean()
                probs = f_prototype_probabilities(mu, prototypes, positions, proto_pos)
                hom = -0.5 * (f_soft_homogeneity(y[1], probs[0]) + f_soft_homogeneity(y[0], probs[1]))
                occupancy = F.relu(0.1 / len(prototypes) - probs.mean(1)).square().mean() * len(prototypes) ** 2
                smooth = (probs[:, sampled[:, 0]] - probs[:, sampled[:, 1]]).square().mean()
                regularization = self.c['prior_kl_weight'] * kl + min(1.0, epoch / 8.0) * (self.c['alignment_weight'] * align + self.c['homogeneity_weight'] * hom + 0.01 * occupancy + 0.01 * smooth)
                torch.autograd.backward([hidden, regularization], [grad, None])
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                loss = 0.5 * (rec + cross) + float(regularization.detach())
                if not np.isfinite(loss) or not torch.isfinite(norm):
                    raise RuntimeError('Nonfinite individual training')
                optimizer.step()
                total += [loss, rec, cross, float(kl.detach()), float(align.detach()), float(hom.detach())]
                total_count += 1
                if (offset + 1) % self.c['checkpoint_subjects'] == 0 or offset + 1 == len(rows):
                    f_checkpoint_save(checkpoint, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'next_subject': offset + 1, 'running_total': total.tolist(), 'running_count': total_count, **self.rng_state()})
                    print('INDIVIDUAL', h, epoch, offset + 1, '/', len(rows), 'loss', loss, 'subject_seconds', time.time() - tick, flush=True)
            report = dict(zip(['loss', 'self_mse', 'cross_mse', 'prior_kl', 'alignment', 'negative_homogeneity'], (total / total_count).tolist()))
            report.update(epoch=epoch, hemisphere=h, subjects=total_count, peak_allocated_GiB=torch.cuda.max_memory_allocated() / 1024 ** 3, peak_reserved_GiB=torch.cuda.max_memory_reserved() / 1024 ** 3)
            f_json_save(folder / f'training_epoch{epoch}.json', report)
            if epoch % self.c['validation_every'] == 0 or epoch == self.c['individual_epochs']:
                if not (folder / f'validation_epoch{epoch}.json').exists():
                    self.evaluate(model, h, group, scale, prototypes, proto_pos, proto['group_labels'], epoch)
                f_checkpoint_save(folder / f'epoch{epoch}.pt', {'model': model.state_dict(), 'epoch': epoch})
            next_subject, total, total_count = (0, np.zeros(6), 0)
        f_json_save(folder / 'completed.json', {'epochs': self.c['individual_epochs'], 'training_subjects': len(rows), 'validation_subjects': len(self.subjects['validation']), 'test_set_used': False})
        del model, prior, optimizer, group, scale
        torch.cuda.empty_cache()

    def run(self):
        self.prepare_reference()
        for h in self.c['hemispheres']:
            self.fit_group(h)
        for h in self.c['hemispheres']:
            self.fit_individual(h)
        self.export()
        f_json_save(self.status_path, {'state': 'completed', 'pid': os.getpid(), 'gpu': self.c['gpu'], 'updated_at_unix': time.time(), 'interpretation': 'Execution complete; scientific improvement requires matched validation.'})
