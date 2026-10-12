from f_runtime import f_work_root
"""Independent task-map validation of frozen rest-derived parcellations."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import json, csv, time, hashlib
from pathlib import Path
import numpy as np
import nibabel as nib
W = f_work_root()
OUT = W / 'runs/task_validation_20261007_v1'
APP = W / 'runs/connectome_validation_20261007_v1/YA400'
SOURCE = W / 'runs/corrected_personal_multires_20261006_v1'
METHODS = ['connected_group', 'CoBRA', 'no_hard_connectivity']

def f_variance_metrics(z, labels, area, k):
    """Population variance; full cortex and fixed K, no parcel removal."""
    z = np.asarray(z, dtype='float64')
    area = np.asarray(area, dtype='float64')
    assert z.ndim == 1 and z.shape == labels.shape == area.shape
    assert np.isfinite(z).all() and np.isfinite(area).all() and (area > 0).all()
    assert np.array_equal(np.unique(labels), np.arange(k))
    counts = np.bincount(labels, minlength=k)
    sums = np.bincount(labels, weights=z, minlength=k)
    sums2 = np.bincount(labels, weights=z * z, minlength=k)
    unweighted = np.maximum(0, sums2 / counts - (sums / counts) ** 2)
    mass = np.bincount(labels, weights=area, minlength=k)
    weighted = np.bincount(labels, weights=area * z, minlength=k) / mass
    within = float(np.sum(area * (z - weighted[labels]) ** 2) / area.sum())
    total = float(np.sum(area * (z - np.average(z, weights=area)) ** 2) / area.sum())
    assert total > 1e-12, 'Degenerate activation map'
    ratio = within / total
    assert -1e-09 <= ratio <= 1 + 1e-09
    return dict(GPIP_task_variance=float(unweighted.mean()), area_weighted_task_variance=within, normalized_task_variance=ratio, piecewise_constant_R2=1 - ratio, global_task_variance=total)

def f_paired(a, b):
    """Participant-paired mean difference, negative favors first method."""
    delta = np.asarray(a) - np.asarray(b)
    n = len(delta)
    rng = np.random.default_rng(20261007)
    boot = delta[rng.integers(0, n, (5000, n))].mean(1)
    perm = (rng.choice([-1, 1], (5000, n)) * delta).mean(1)
    return dict(n=n, mean_difference=float(delta.mean()), ci95=np.quantile(boot, [0.025, 0.975]).tolist(), two_sided_p=float((1 + np.sum(np.abs(perm) >= abs(delta.mean()))) / 5001), draws=5000)

def f_holm(values):
    result = {}
    current = 0
    for i, key in enumerate(sorted(values, key=values.get)):
        current = max(current, min(1, (len(values) - i) * values[key]))
        result[key] = current
    return result

def f_main():
    import logging
    logging.getLogger('nibabel.global').setLevel(logging.ERROR)
    protocol = json.loads((OUT / 'locked_protocol.json').read_text())
    cohorts = json.loads((W / 'runs/connectome_validation_20261007_v1/cohorts.json').read_text())
    subjects = {r['subject']: r for r in cohorts['YA400']}
    mapping = np.load(W / 'runs/pilot32_MSMAll_samplez/mapping.npz')
    records = []
    exclusions = []
    pending = []
    for s in protocol['subjects']:
        areas = {h: np.load(subjects[s]['area'][h]['path']) for h in ['L', 'R']}
        for task, contrasts in protocol['tasks'].items():
            p = OUT / 'downloads' / s / (task + '.json')
            if not p.exists():
                pending.append([s, task, 'download'])
                continue
            d = json.loads(p.read_text())
            if d['state'] != 'complete' or not d.get('task_QC_passed'):
                exclusions.append(dict(subject=s, task=task, state=d['state'], reason=d.get('error', 'task motion QC')))
                continue
            image = nib.load(d['map_path'])
            arr = np.asarray(image.dataobj)
            for K in [200, 100, 300, 400]:
                k = K // 2
                maps = {m: {} for m in METHODS}
                if not all(((APP / 'individual' / s / h / str(k) / 'baseline_REST1/complete.json').exists() for h in ['L', 'R'])):
                    pending.append([s, task, f'labels K{K}'])
                    continue
                for h in ['L', 'R']:
                    maps['connected_group'][h] = np.load(SOURCE / f'priors/{h}/{k}/labels.npy')
                    for m in METHODS[1:]:
                        maps[m][h] = np.load(APP / 'individual' / s / h / str(k) / 'baseline_REST1' / f'{m}.npy')
                area = np.r_[areas['L'], areas['R']]
                for contrast in contrasts:
                    z = np.concatenate([arr[d['selected_contrasts'][contrast], mapping[h + '_columns']] for h in ['L', 'R']])
                    for m in METHODS:
                        labels = np.r_[maps[m]['L'], maps[m]['R'] + k]
                        result = f_variance_metrics(z, labels, area, K)
                        records.append(dict(subject=s, task=task, contrast=contrast, K_bilateral=K, method=m, **result))
    domains = {}
    for r in records:
        key = (r['subject'], r['task'], r['K_bilateral'], r['method'])
        domains.setdefault(key, []).append(r['normalized_task_variance'])
    domains = {key: float(np.mean(v)) for key, v in domains.items()}
    summary = {}
    for K in [200, 100, 300, 400]:
        comparisons = {}
        for comparator in ['connected_group', 'no_hard_connectivity']:
            results = {}
            for task in protocol['tasks']:
                ss = [s for s in protocol['subjects'] if (s, task, K, 'CoBRA') in domains and (s, task, K, comparator) in domains]
                if len(ss) < 3:
                    continue
                a = [domains[s, task, K, 'CoBRA'] for s in ss]
                b = [domains[s, task, K, comparator] for s in ss]
                results[task] = dict(**f_paired(a, b), CoBRA_mean=float(np.mean(a)), comparator_mean=float(np.mean(b)), subjects=ss)
            adjusted = f_holm({t: results[t]['two_sided_p'] if t in results else 1.0 for t in protocol['tasks']})
            for t, r in results.items():
                r['Holm_p_within_comparator_domains'] = adjusted[t]
            comparisons[comparator] = results
        summary[str(K)] = comparisons
    OUT.mkdir(exist_ok=True)
    dest = OUT / 'task_metrics.csv'
    if records:
        with dest.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    primary_pending = [r for r in pending if r[2] in ['download', 'labels K200']]
    result = dict(state='complete' if not pending else 'partial_waiting_for_labels_or_download', primary_K200_state='complete_for_available_QC_passing_tasks' if not primary_pending else 'pending', updated_unix=time.time(), protocol_sha256=hashlib.sha256((OUT / 'locked_protocol.json').read_bytes()).hexdigest(), independent_task=True, record_count=len(records), summary=summary, exclusions=exclusions, pending=pending, interpretation='Task activation is independent functional evidence, not ground-truth parcel labels; secondary K exploratory; family metadata unavailable.')
    temp = OUT / 'task_summary.json.tmp'
    temp.write_text(json.dumps(result, indent=2, allow_nan=False))
    temp.replace(OUT / 'task_summary.json')
    print(json.dumps(dict(state=result['state'], records=len(records), pending=len(pending), excluded=len(exclusions))))
