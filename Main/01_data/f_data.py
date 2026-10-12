from f_runtime import f_work_root
"""CIFTI source checks and independently filtered, disjoint temporal segments."""
import hashlib, json, os
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.signal import butter, detrend, sosfiltfilt
RUNS = ['rfMRI_REST1_LR', 'rfMRI_REST1_RL', 'rfMRI_REST2_LR', 'rfMRI_REST2_RL']

def f_read(p):
    return json.loads(Path(p).read_text())

def f_save(p, data):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(p)

def f_sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2 ** 20), b''):
            h.update(b)
    return h.hexdigest()

def f_identity(p):
    p = Path(p)
    s = p.stat()
    return dict(path=str(p), size=s.st_size, mtime_ns=s.st_mtime_ns)

def f_standardize(y):
    y = np.asarray(y, dtype='float32').copy()
    y -= y.mean(0)
    sd = y.std(0)
    assert np.isfinite(y).all() and np.min(sd) > 1e-07
    y /= sd
    return np.ascontiguousarray(y)

def f_cohort(workspace):
    c = f_read(workspace / 'runs/fullfc450_crossday_v1/subjects.json')
    audited = f_read(workspace / 'runs/modulated_vae_full_crossday_v1/subjects.json')
    assert [r['subject'] for r in c['train']] == [r['subject'] for r in audited['train']]
    c['reserved_retest'] = audited['reserved_retest']
    manifest = f_read(workspace / 'session_ablation_inputs.json')
    wanted = c['validation'][30:42]
    entries = {r['subject']: r for r in manifest['subjects']}
    result = []
    for row in wanted:
        e = entries[row['subject']]
        assert e['complete']
        paths = {run: files[0] for run, files in e['runs'].items() if len(files) == 1}
        assert all((name in paths for name in RUNS))
        result.append(dict(subject=row['subject'], runs={run: paths[run] for run in RUNS}))
    assert len(result) == 12 and (not {r['subject'] for r in result} & ({r['subject'] for r in c['train']} | set(c['reserved_retest'])))
    return (c, result)

def f_segment(row, run, h, start, stop, mapping, cache):
    """Read raw source BEFORE segmentation; filter each segment separately.

 No full-run zero-phase filter is reused across the train/evaluation boundary.
 """
    source = Path(row['runs'][run])
    img = nib.load(source, mmap=False)
    axis = img.header.get_axis(1)
    series = img.header.get_axis(0)
    name = 'CIFTI_STRUCTURE_CORTEX_LEFT' if h == 'L' else 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    columns = np.flatnonzero(axis.surface_mask & (axis.name == name))
    assert np.array_equal(columns, mapping[h + '_columns']) and np.array_equal(axis.vertex[columns], mapping[h + '_vertices'])
    assert img.shape == (1200, 91282) and abs(series.step - 0.72) < 1e-05 and (0 <= start < stop <= 1200)
    meta = dict(source=f_identity(source), run=run, hemisphere=h, frames=[start, stop], TR=float(series.step), bandpass_hz=[0.01, 0.1], order=3, independent_segment_filter=True, mapping_hash=hashlib.sha256(mapping[h + '_vertices'].tobytes()).hexdigest())
    folder = cache / row['subject'] / run
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f'{h}_{start}_{stop}.npy'
    marker = p.with_suffix('.json')
    if p.exists() and marker.exists():
        assert f_read(marker) == meta
        return f_standardize(np.load(p))
    assert np.array_equal(columns, np.arange(columns[0], columns[-1] + 1))
    whole = np.array(img.dataobj[:, int(columns[0]):int(columns[-1]) + 1], dtype='float32', copy=True)
    raw = whole[start:stop].copy()
    del whole
    sos = butter(3, [0.01, 0.1], btype='bandpass', fs=1 / series.step, output='sos')
    pad = min(int(3 / series.step / 0.01) | 1, len(raw) - 1)
    out = np.empty_like(raw)
    for a in range(0, len(columns), 2048):
        x = detrend(raw[:, a:a + 2048].astype('float64'), axis=0)
        x = sosfiltfilt(sos, x, axis=0, padlen=pad)
        out[:, a:a + 2048] = f_standardize(x)
    np.save(p, out)
    f_save(marker, meta)
    return out

def f_training(row, h, arm, mapping, cache):
    if arm == 'lr_only':
        runs = ['rfMRI_REST1_LR', 'rfMRI_REST2_LR']
        stop = 480
    elif arm == 'joint_equal':
        runs = RUNS
        stop = 240
    elif arm == 'joint_all':
        runs = RUNS
        stop = 480
    else:
        raise ValueError(arm)
    return ([f_segment(row, r, h, 0, stop, mapping, cache) for r in runs], dict(runs=runs, frames_per_run=stop, total_frames=len(runs) * stop, evaluation_frames=[720, 1200], gap_frames=720 - stop))

def f_evaluation_days(row, h, mapping, cache):
    return [np.concatenate([f_segment(row, f'rfMRI_REST{d}_{pe}', h, 720, 1200, mapping, cache) for pe in ['LR', 'RL']], axis=0) for d in [1, 2]]
