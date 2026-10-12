from f_runtime import f_work_root
"""Native Wang/HFR 18-network experiment, with independently reset visits.

The author HFR release is the source of truth, not the previous Salehi adaptation.
Network components are permitted. CoBRA preserves initial component topology;
18 spatial Ward parcels are explicitly an areal control rather than 18 networks.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
from pathlib import Path
import argparse, copy, fcntl, hashlib, json, sys, time, traceback, subprocess
import numpy as np
from scipy.io import loadmat, savemat
from scipy.sparse import csr_matrix, block_diag, save_npz, load_npz
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
import nibabel as nib
W = f_work_root()
CODE = W / 'coarse_network18_20261008/code'
OUT = W / 'runs/coarse_network18_20261008_v1'
AUTHOR = W / 'references/wang2015_author_hfr/HFR_ai'
RESAMPLE = AUTHOR.parent / 'resampling'
APP = W / 'runs/connectome_validation_20261007_v1/retest'
SOURCE = W / 'runs/corrected_personal_multires_20261006_v1'
WB = W / 'coarse_network18_20261008/tools/workbench/bin_linux64/wb_command'
sys.path.insert(0, str(W / 'final'))
sys.path.insert(1, str(W / 'paper_figure_campaign'))
from f_data import f_read, f_save, f_sha
from f_application_data import f_signals
from f_network18_subject_specificity import f_agreement
VISITS = [('Day1', 'baseline', 1), ('Day2', 'baseline', 2), ('Retest', 'retest', 1)]

def f_status(**kwargs):
    f_save(OUT / 'status.json', dict(pid=os.getpid(), updated_unix=time.time(), **kwargs))

def f_standardize(y):
    y = y.astype('float64')
    y -= y.mean(0)
    sd = y.std(0)
    good = sd > 1e-08
    y[:, good] /= sd[good]
    y[:, ~good] = 0
    return (y.astype('float32'), good)

def f_gifti(path, data):
    nib.save(nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(np.asarray(data, dtype='float32'))]), str(path))

def f_geometry():
    folder = OUT / 'geometry'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        return
    mapping = np.load(W / 'runs/pilot32_MSMAll_samplez/mapping.npz')
    native = loadmat(AUTHOR / 'Utilities/fs4_Firstadjacent_vertex.mat')
    variability = loadmat(AUTHOR / 'Utilities/Variability_FS4.mat')
    var = np.r_[variability['lh'].ravel(), variability['rh'].ravel()]
    var = 0.4 + 0.6 * (var - var.min()) / (var.max() - var.min())
    np.save(folder / 'variability.npy', var)
    for h, lh in [('L', 'lh'), ('R', 'rh')]:
        masks = []
        values = []
        for k in range(1, 20):
            value = nib.load(str(AUTHOR / f'Templates/Parcellation_template/{lh}_network_{k}_asym_fs4.mgh')).get_fdata().ravel()
            a = value > 0
            masks.append(a)
            if k > 1:
                values.append(value)
        values = np.column_stack(values)
        atlas = values.argmax(1).astype('int32')
        atlas[masks[0]] = -1
        assert not np.any(np.stack(masks)[1:, masks[0]])
        np.save(folder / f'template_values_{h}.npy', values)
        np.save(folder / f'template_masks_{h}.npy', np.column_stack(masks[1:]))
        assert np.array_equal(np.unique(atlas[atlas >= 0]), np.arange(18))
        np.save(folder / f'author_labels_{h}.npy', atlas)
        cells = native[f'fs4_Firstadjacent_vertex_{lh}'].ravel()
        rr = []
        cc = []
        weights = []
        for v, members in enumerate(cells):
            ids = members.ravel().astype('int64') - 1
            rr.extend([v] * len(ids))
            cc.extend(ids)
            weights.extend([1 / len(ids)] * len(ids))
        smoothing = csr_matrix((weights, (rr, cc)), shape=(2562, 2562))
        save_npz(folder / f'score_smoothing_{h}.npz', smoothing)
        sphere4 = nib.load(str(RESAMPLE / f'fsaverage4_std_sphere.{h}.3k_fsavg_{h}.surf.gii')).darrays[0].data
        sphere32 = nib.load(str(RESAMPLE / f'fs_LR-deformed_to-fsaverage.{h}.sphere.32k_fs_LR.surf.gii')).darrays[0].data
        sphere4 = sphere4 / np.linalg.norm(sphere4, axis=1, keepdims=True)
        sphere32 = sphere32 / np.linalg.norm(sphere32, axis=1, keepdims=True)
        valid = np.flatnonzero(atlas >= 0)
        nearest4 = valid[cKDTree(sphere4[valid]).query(sphere32[mapping[h + '_vertices']])[1]]
        np.save(folder / f'prior32_{h}.npy', atlas[nearest4])
        np.save(folder / f'nearest4_{h}.npy', nearest4)
        nearest32 = cKDTree(sphere32[mapping[h + '_vertices']]).query(sphere4)[1]
        np.save(folder / f'nearest32_{h}.npy', nearest32)
        roi = np.zeros(32492, dtype='float32')
        roi[mapping[h + '_vertices']] = 1
        f_gifti(folder / f'roi32_{h}.func.gii', roi)
        faces = nib.load(str(RESAMPLE / f'fsaverage4_std_sphere.{h}.3k_fsavg_{h}.surf.gii')).darrays[1].data
        edges = np.unique(np.sort(np.r_[faces[:, :2], faces[:, 1:3], faces[:, [0, 2]]], axis=1), axis=0)
        adj = csr_matrix((np.ones(2 * len(edges)), (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(2562, 2562))
        save_npz(folder / f'adjacency_{h}.npz', adj)
    f_save(folder / 'complete.json', dict(author_commit=f_read(AUTHOR.parent / 'commit.json')['commit'], author_algorithm_sha256=f_sha(AUTHOR / 'Subfunctions/Func_IterativeParcellation.m'), atlas='Original HFR18 positive masks overlap. All overlapping masks retained for author seed/bias arithmetic. Categorical reference/CoBRA prior uses argmax template value; label1 is excluded medial mask.', sphere_registration='Official HCP fsLR-deformed-to-fsaverage spheres; native fsaverage4 ordering.', resampling_manifest=f_read(RESAMPLE / 'manifest.json')))

def f_prepare_inputs(selected=None):
    f_geometry()
    rows = f_read(APP / 'subjects.json') if selected is None else selected
    mapping = np.load(W / 'runs/pilot32_MSMAll_samplez/mapping.npz')
    records = []
    for row in rows:
        for visit, actual, day in VISITS:
            dest = OUT / 'inputs' / visit / row['subject']
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / 'complete.json').exists() and all(((dest / f'{h}_sm6_32.npy').exists() for h in ['L', 'R'])):
                records.append(f_read(dest / 'complete.json'))
                if False:
                    f_wang_visit(row, visit)
                continue
            f_status(state='running', phase='native_fs4_input_preparation', subject=row['subject'], visit=visit)
            shapes = {}
            covered = {}
            retained = None
            for h in ['L', 'R']:
                arrays = [f_signals(W, APP, row, actual, f'rfMRI_REST{day}_{pe}', h, mapping) for pe in ['LR', 'RL']]
                if retained is None:
                    retained = [len(y) for y in arrays]
                else:
                    assert retained == [len(y) for y in arrays]
                y = np.concatenate(arrays)
                full = np.zeros((32492, len(y)), dtype='float32')
                full[mapping[h + '_vertices']] = y.T
                metric = dest / f'{h}_32.func.gii'
                smoothed = dest / f'{h}_sm6.func.gii'
                resampled = dest / f'{h}_fs4.func.gii'
                valid = dest / f'{h}_valid.func.gii'
                f_gifti(metric, full)
                surface = Path(row['area'][h]['surface'])
                assert surface.exists(), surface
                assert f_sha(surface) == row['area'][h]['surface_sha256']
                subprocess.run([str(WB), '-metric-smoothing', str(surface), str(metric), str(6 / np.sqrt(8 * np.log(2))), str(smoothed), '-roi', str(OUT / f'geometry/roi32_{h}.func.gii')], check=True, stdout=subprocess.DEVNULL)
                data32 = nib.load(str(smoothed)).agg_data()
                data32 = np.column_stack(data32) if isinstance(data32, tuple) else np.asarray(data32)
                assert data32.shape == full.shape
                np.save(dest / f'{h}_sm6_32.npy', data32[mapping[h + '_vertices']].astype('float32'))
                subprocess.run([str(WB), '-metric-resample', str(smoothed), str(RESAMPLE / f'fs_LR-deformed_to-fsaverage.{h}.sphere.32k_fs_LR.surf.gii'), str(RESAMPLE / f'fsaverage4_std_sphere.{h}.3k_fsavg_{h}.surf.gii'), 'ADAP_BARY_AREA', str(resampled), '-area-metrics', str(RESAMPLE / f'fs_LR.{h}.midthickness_va_avg.32k_fs_LR.shape.gii'), str(RESAMPLE / f'fsaverage4.{h}.midthickness_va_avg.3k_fsavg_{h}.shape.gii'), '-current-roi', str(OUT / f'geometry/roi32_{h}.func.gii'), '-valid-roi-out', str(valid)], check=True, stdout=subprocess.DEVNULL)
                data = nib.load(str(resampled)).agg_data()
                a = np.column_stack(data) if isinstance(data, tuple) else np.asarray(data)
                assert a.shape == (2562, len(y)), a.shape
                a, good = f_standardize(a.T)
                np.save(dest / f'{h}.npy', a.T)
                np.save(dest / f'{h}_valid.npy', good)
                shapes[h] = list(a.T.shape)
                covered[h] = int(np.sum(good & (np.load(OUT / f'geometry/author_labels_{h}.npy') >= 0)))
                for p in [metric, smoothed, resampled, valid]:
                    p.unlink()
            rec = dict(subject=row['subject'], visit=visit, shapes=shapes, retained_frames_per_run=retained, covered_author_cortex=covered, bandpass_hz=[0.01, 0.1], smoothing_FWHM_mm=6, paths={h: str(dest / f'{h}.npy') for h in ['L', 'R']})
            f_save(dest / 'complete.json', rec)
            records.append(rec)
            print('INPUT18', visit, row['subject'], flush=True)
            if False:
                f_wang_visit(row, visit)
    if selected is not None:
        return records
    masks = {}
    for h in ['L', 'R']:
        mask = np.load(OUT / f'geometry/author_labels_{h}.npy') >= 0
        for r in records:
            mask &= np.load(Path(r['paths'][h]).with_name(h + '_valid.npy'))
        np.save(OUT / f'geometry/evaluation_mask_{h}.npy', mask)
        masks[h] = int(mask.sum())
    f_save(OUT / 'inputs_manifest.json', dict(records=records, common_evaluation_vertices=masks, n_subjects=36, visits=3))

def f_normalized_rows(a):
    a = a.astype('float64')
    a = a - a.mean(1, keepdims=True)
    norm = np.linalg.norm(a, axis=1, keepdims=True)
    return np.divide(a, norm, out=np.zeros_like(a), where=norm > 0)

def f_wang_fit(datas, iterations=40, early_stop=True):
    labels0 = [np.load(OUT / f'geometry/author_labels_{h}.npy') for h in ['L', 'R']]
    smooth = [load_npz(OUT / f'geometry/score_smoothing_{h}.npz') for h in ['L', 'R']]
    variability = np.load(OUT / 'geometry/variability.npy')
    inv = [1 / variability[:2562], 1 / variability[2562:]]
    template_masks = [np.load(OUT / f'geometry/template_masks_{h}.npy') for h in ['L', 'R']]
    group = [np.stack([inv[j][v[:, k]] @ datas[j][v[:, k]] for k in range(18)]) for j, v in enumerate(template_masks)]
    previous = [v.copy() for v in labels0]
    confidences = None
    history = []
    outputs = []
    stable = 0
    for cnt in range(1, iterations + 1):
        seeds = [g.copy() for g in group]
        if cnt > 1:
            for j in range(2):
                for k in range(18):
                    conf = confidences[j][k]
                    idx = conf >= 1.1
                    if not idx.any():
                        idx = conf == conf.max()
                    seeds[j][k] = datas[j][idx].sum(0) + group[j][k] / (cnt - 1)
        assigned = []
        confidences = []
        changes = []
        for j in range(2):
            correlation = f_normalized_rows(datas[j]) @ f_normalized_rows(seeds[j]).T
            with np.errstate(divide='ignore', invalid='ignore'):
                z = np.arctanh(np.clip(correlation, -1, 1))
            z[np.isnan(z)] = 0
            z = smooth[j] @ z
            for k in range(18):
                ids = template_masks[j][:, k]
                z[ids, k] += (inv[j][ids] - 1) / (3 * cnt)
            order = np.argsort(-z, axis=1, kind='stable')
            membership = order[:, 0]
            with np.errstate(divide='ignore', invalid='ignore'):
                confidence = z[np.arange(2562), order[:, 0]] / z[np.arange(2562), order[:, 1]]
            confidence[np.isnan(confidence)] = 0
            membership[labels0[j] < 0] = -1
            c = np.stack([np.where(membership == k, confidence, 0) for k in range(18)])
            mask = labels0[j] >= 0
            changes.append(float(np.mean(membership[mask] != previous[j][mask])))
            assigned.append(membership.astype('int32'))
            confidences.append(c)
        history.append(dict(iteration=cnt, changed_fraction_LR=changes))
        outputs.append([v.copy() for v in assigned])
        previous = assigned
        stable = stable + 1 if max(changes) <= 0.02 else 0
        if early_stop and stable >= 2:
            break
    return (previous, history, outputs)

def f_author_equivalence_check():
    """Run actual author MATLAB source for three iterations on real native input."""
    marker = OUT / 'validation/author_equivalence.json'
    if marker.exists():
        assert f_read(marker)['passed']
        return
    row = f_read(APP / 'subjects.json')[0]
    s = row['subject']
    folder = OUT / 'validation'
    folder.mkdir(exist_ok=True)
    dest = folder / 'OrganizedData'
    dest.mkdir(exist_ok=True)
    datas = [np.load(OUT / f'inputs/Day1/{s}/{h}.npy').astype('float64') for h in ['L', 'R']]
    datas = [a[:, :32] for a in datas]
    savemat(dest / 'equivalence_timeframes_fs4.mat', dict(lhData=datas[0], rhData=datas[1]))
    source = (AUTHOR / 'Subfunctions/Func_IterativeParcellation.m').read_text()
    source = source.replace('Func_IterativeParcellation(', 'Func_IterativeParcellation_headless(', 1)
    source = '\n'.join((line for line in source.splitlines() if 'waitbar(' not in line and 'close(hwait)' not in line))
    (folder / 'Func_IterativeParcellation_headless.m').write_text(source)
    (folder / 'load_mgh.m').write_text("function [vol,M,mr_parms,volsz]=load_mgh(p)\nf=fopen(p,'rb','b');h=fread(f,7,'int32');fseek(f,284,'bof');assert(h(6)==3);vol=reshape(fread(f,prod(h(2:5)),'float32=>double'),h(2:5)');fclose(f);M=eye(4);mr_parms=[];volsz=h(2:5)';\nend\n")
    (folder / 'save_mgh.m').write_text("function save_mgh(v,p,M)\nf=fopen(p,'wb','b');fwrite(f,[1 numel(v) 1 1 1 3 0],'int32');fwrite(f,0,'int16');fwrite(f,zeros(254,1),'uint8');fwrite(f,v(:),'float32');fwrite(f,zeros(4,1),'float32');fclose(f);\nend\n")
    command = f"addpath('{folder}');maxNumCompThreads(2);Func_IterativeParcellation_headless('{AUTHOR}','{folder}',{{'equivalence'}},3,1.1,0);"
    p = subprocess.run(['/usr/bin/matlab', '-batch', command], capture_output=True, text=True, timeout=300)
    (folder / 'matlab.log').write_text(p.stdout + '\n' + p.stderr)
    assert p.returncode == 0, p.stderr[-1200:]
    _, _, ported = f_wang_fit(datas, iterations=3, early_stop=False)
    records = []
    for iteration in range(1, 4):
        for j, lh in enumerate(['lh', 'rh']):
            observed = np.full(2562, -1, dtype='int32')
            for k in range(18):
                a = nib.load(str(folder / f'IndiPar/equivalence/Iter_{iteration}/Network_{k + 2}_{lh}.mgh')).get_fdata().ravel()
                observed[a > 0] = k
            fraction = float(np.mean(observed == ported[iteration - 1][j]))
            assert fraction == 1, (iteration, lh, fraction)
            records.append(dict(iteration=iteration, hemisphere=lh, identical_label_fraction=fraction))
    f_save(marker, dict(passed=True, source_sha256=f_sha(AUTHOR / 'Subfunctions/Func_IterativeParcellation.m'), display_only_patch='Removed waitbar/close(hwait) only; supplied simple MGH float32 I/O wrappers; mathematical source unchanged.', real_data_prefix_frames=32, records=records, limitation='Label equivalence for this deterministic smoke input is not full paper replication; HFR is a later author release with constant SNR.'))

def f_wang_visit(row, visit):
    assert f_read(OUT / 'validation/author_equivalence.json')['passed']
    dest = OUT / f"Wang_author_HFR18/{visit}/{row['subject']}"
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'complete.json').exists():
        return f_read(dest / 'complete.json')
    f_status(state='running', phase='Wang_author_HFR18', subject=row['subject'], visit=visit)
    datas = [np.load(OUT / f"inputs/{visit}/{row['subject']}/{h}.npy").astype('float64') for h in ['L', 'R']]
    labels, history, _ = f_wang_fit(datas)
    for h, v in zip(['L', 'R'], labels):
        np.save(dest / f'{h}.npy', v)
    rec = dict(subject=row['subject'], visit=visit, iterations=len(history), history=history, stopping='<=2% label changes in each hemisphere for two consecutive iterations, maximum40', core='Author HFR18 arithmetic, constant SNR as released; weighted original atlas, variability prior, Fisher-z score smoothing and declining anchor.', occupied=[len(np.unique(v[v >= 0])) for v in labels])
    f_save(dest / 'complete.json', rec)
    print('WANG18', visit, row['subject'], len(history), flush=True)
    return rec

def f_run_wang():
    f_author_equivalence_check()
    records = [f_wang_visit(row, visit) for row in f_read(APP / 'subjects.json') for visit, _, _ in VISITS]
    f_save(OUT / 'Wang_author_HFR18/complete.json', dict(subjects=36, visits=3, records=records, author_check=f_read(OUT / 'validation/author_equivalence.json')))

def f_protocol():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = f_read(APP / 'subjects.json')
    cfg = dict(subjects=[r['subject'] for r in rows], networks=18, existing_K100_preserved=True, visits=[v[0] for v in VISITS], bandpass_hz=[0.01, 0.1], frames_per_run=480, phase_encodings=['LR', 'RL'], fit='Independent visit-specific calibrations reset to group prior; never use Retest for Day1/Day2 fitting.', author_reference='https://github.com/MeilingAva/Homologous-Functional-Regions', author_commit=f_read(AUTHOR.parent / 'commit.json')['commit'], Wang='Native2562 vertices/hemi, original18 author atlas and variability, MATLAB source label equivalence gate, HFR released constant SNR; confidence1.1, <=2% change for2 rounds/max40.', preprocessing_difference='Matched HCP ICA-FIX/MSMAll and requested0.01-0.1Hz,6mm surface smoothing then official HCP fsLR-to-fsaverage4 area resampling; not identical2015 preprocessing0.01-0.08Hz/GSR/fsaverage6.', network_topology='Distributed networks may have multiple components. Report descriptive CC plus changes from fixed prior, not zero-component superiority across network and areal tasks.', individual_methods=['Wang_author_HFR18', 'GPIP_independent18', 'PCA64_KMeans18', 'PCA64_SpatialWard18_areal_control', 'CoBRA18_network_extension', 'CoBRA18_no_component_guard'], fixed_atlas='Author18 reference, intra/inter=1 by construction; no individualization claim.', metrics=['ARI', 'Hungarian macro Parcel Dice', 'fixed-network vertex agreement where IDs have semantic correspondence', 'intra-minus-inter specificity', 'identification with tie handling', 'components relative to network prior'], limitations='Wang/HFR later release with matched preprocessing, GPIP independent diagnostic, and CoBRA18 decoder extension must be identified explicitly. Do not pool Ward spatial parcels into canonical-network superiority claims.', code_sha256=f_sha(Path(__file__)))
    p = OUT / 'protocol.json'
    if p.exists() and f_read(p) != cfg:
        assert not (OUT / 'cpu_launch.json').exists() and (not (OUT / 'inputs_manifest.json').exists()), 'Use a new run for changed code/protocol'
        p.rename(p.with_name('protocol_preflight_' + f_sha(p)[:8] + '.json'))
    f_save(p, cfg)
