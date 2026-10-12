from f_runtime import f_work_root
"""Training-only geometry and Zhu distance-bin statistics for the full-FC arm."""
import os
from pathlib import Path
SOURCE_DIR = f_work_root() / 'final/population_training'
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from f_population_run import ROOT, f_get_mapping, f_distance_matrix
from f_population_drfc import f_residualize
from f_population_train_modulated_vae import f_json_save

def f_prepare(config_path, geometry_only=False):
    c = json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
    frozen = json.loads((ROOT / 'runs' / c['comparison_run'] / 'subjects.json').read_text())
    rows = frozen['train'][:c.get('train_limit')]
    subject_ids = [r['subject'] for r in rows]
    if set(subject_ids) & set(frozen['reserved_retest']):
        raise RuntimeError('Retest subject in training reference')
    folder = ROOT / 'resources/fullfc_distance' / c['name']
    folder.mkdir(parents=True, exist_ok=True)
    definition = {'subjects': subject_ids, 'mode': c['geometry_mode'], 'registration': 'MSMAll', 'cutoff_mm': 50, 'distance_bin_mm': 1, 'local_max_mm': 45, 'raw_group_run': c['normalization_source_run'], 'source_reference_sessions': ['rfMRI_REST1_LR']}
    saved = folder / 'definition.json'
    if saved.exists() and json.loads(saved.read_text()) != definition:
        raise RuntimeError('Changed distance/normalization cohort definition')
    f_json_save(saved, definition)
    mapping = f_get_mapping(rows[0]['paths'][0])
    np.savez(folder / 'mapping.npz', **{f'{h}_{k}': v for h, m in mapping.items() for k, v in m.items()})
    template = c['geometry_mode'] == 'template_execution_check'
    if template and len(rows) > 2:
        raise RuntimeError('Template shortcut is permitted only in the execution smoke test')
    if not template and c['geometry_mode'] != 'training_subject_mean':
        raise RuntimeError('Unknown geometry mode')
    status = ROOT / 'logs' / (c['name'] + '.status.json')
    for h in c['hemispheres']:
        f_json_save(status, {'state': 'running', 'stage': 'prepare_training_geometry', 'pid': os.getpid(), 'updated_at_unix': time.time(), 'hemisphere': h, 'total_subjects': len(rows), 'mode': c['geometry_mode']})
        f_distance_matrix(rows, h, mapping, folder, {'seed': c['seed'], 'surface_registration': 'MSMAll'}, not template)
    if geometry_only:
        return
    source = ROOT / 'runs' / c['normalization_source_run']
    while not all(((source / f'group_reference_{h}.done.json').exists() for h in c['hemispheres'])):
        f_json_save(status, {'state': 'running', 'stage': 'waiting_for_raw_training_group', 'pid': os.getpid(), 'updated_at_unix': time.time(), 'source_run': c['normalization_source_run']})
        time.sleep(30)
    source_ids = [r['subject'] for r in json.loads((source / 'subjects.json').read_text())['train']]
    if subject_ids != source_ids:
        raise RuntimeError('Raw FC and geometry cohorts differ')
    for h in c['hemispheres']:
        marker = folder / f'normalization_{h}.done.json'
        if marker.exists():
            continue
        f_json_save(status, {'state': 'running', 'stage': 'fit_distance_normalization', 'pid': os.getpid(), 'updated_at_unix': time.time(), 'hemisphere': h})
        fc = np.load(source / f'group_{h}.npy', mmap_mode='r')
        distances = np.load(folder / f'distances_{h}.npy', mmap_mode='r')
        stats = f_residualize(fc, distances)
        np.savez(folder / f'statistics_{h}.npz', **stats)
        out = folder / f'bins_{h}.npy'
        tmp = folder / f'bins_{h}.partial.npy'
        bins = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.uint8, shape=fc.shape)
        counts, sums, second = (np.zeros(45, np.int64), np.zeros(45), np.zeros(45))
        for start in range(0, len(fc), 256):
            d = np.asarray(distances[start:start + 256])
            local = d <= 45
            ids = np.full(d.shape, 45, np.uint8)
            ids[local] = np.minimum(d[local].astype(np.int32), 44).astype(np.uint8)
            bins[start:start + len(d)] = ids
            bi = ids[local].astype(np.int32)
            z = (np.asarray(fc[start:start + len(d)])[local].astype(np.float64) - stats['means'][bi]) / stats['scales'][bi]
            counts += np.bincount(bi, minlength=45)
            sums += np.bincount(bi, weights=z, minlength=45)
            second += np.bincount(bi, weights=z * z, minlength=45)
        bins.flush()
        del bins
        tmp.replace(out)
        means = sums / np.maximum(counts, 1)
        variance = second / np.maximum(counts, 1) - means ** 2
        nondegenerate = (counts > 0) & (stats['scales'] > 1e-06)
        if not np.array_equal(counts, stats['counts']) or abs(means).max() > 0.001:
            raise RuntimeError('Distance-bin mean/count QC failed')
        if np.max(abs(variance[nondegenerate] - 1)) > 0.001:
            raise RuntimeError('Distance-bin variance QC failed')
        f_json_save(marker, {'training_subjects': len(rows), 'raw_group_source': str(source / f'group_{h}.npy'), 'statistics_source': 'Paper-style training-group FC, not pooled individual FC variance', 'individual_application': 'Fixed training-derived bin mean/SD; not refitted on validation or individual scans', 'max_abs_local_mean': float(abs(means).max()), 'max_local_unit_variance_error_nondegenerate': float(np.max(abs(variance[nondegenerate] - 1))), 'definition_sha256': hashlib.sha256(saved.read_bytes()).hexdigest()})
        print('NORMALIZATION READY', h, flush=True)
