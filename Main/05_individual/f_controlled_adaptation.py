from f_runtime import f_work_root
"""Matched single-day and cross-day fits plus17 network/parcel extensions.

Uses locked canonical functions without altering existing400/Retest jobs.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import sys, argparse, time, fcntl, traceback
from pathlib import Path
import numpy as np, torch
W = f_work_root()
sys.path.insert(0, str(W / 'final'))
from f_data import f_read, f_save, f_sha
from f_application_data import f_signals, f_qc_segment
from f_application_runner import f_load_models, f_calibration_decode
from f_pipeline import f_set_prior, f_fit
from f_model_ops import f_materialize, f_encode_projected, f_distance
from f_connectivity import f_decode
from f_metrics import f_agreement, f_topology
SOURCE = W / 'runs/corrected_personal_multires_20261006_v1'
APP = W / 'runs/connectome_validation_20261007_v1'
ROOT = W / 'runs/paper_figure_campaign_20261007_v1'
RESOLUTIONS = [('parcel17', {'L': 9, 'R': 8}), ('network17', {'L': 17, 'R': 17}), ('parcel200', {'L': 100, 'R': 100}), ('parcel100', {'L': 50, 'R': 50}), ('parcel300', {'L': 150, 'R': 150}), ('parcel400', {'L': 200, 'R': 200})]
ARMS = {'single_day1': [(1, 480)], 'single_day2': [(2, 480)], 'cross_equal': [(1, 240), (2, 240)]}

def f_arrays(row, h, mapping, arm):
    result = []
    audit = []
    for day, stop in ARMS[arm]:
        for pe in ['LR', 'RL']:
            f_run = f'rfMRI_REST{day}_{pe}'
            fd = np.load(Path(row['ancillary']['baseline'][f_run]['path']).with_name('FD.npy'))
            qc = f_qc_segment(fd, 0, stop)
            if not qc['passed']:
                raise ValueError('Calibration_QC/' + f_run + '/' + str(stop))
            y = f_signals(W, APP, row, 'baseline', f_run, h, mapping, 0, stop)
            result.append(y)
            audit.append(dict(run=f_run, frames=[0, stop], retained=len(y), QC=qc))
    return (result, audit)

@torch.no_grad()
def f_infer(exp, model, centers, ys, hard=True):
    fc = f_materialize([torch.from_numpy(y).cuda() for y in ys], exp.scale, exp.transform)
    mu = f_encode_projected(model, model.input(fc), exp.graph, exp.patches, exp.pos)[0]
    cost = f_distance(mu.mean(0), centers).cpu().numpy()
    del fc, mu
    torch.cuda.empty_cache()
    return f_decode(exp.initial, np.concatenate(ys), cost, exp.adj, exp.seeds, exp.depth, exp.edges, hard_connectivity=hard)

def f_run(a):
    out = ROOT / a.cohort
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'controlled.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert f_read(ROOT / 'low_resolution_status.json')['state'] == 'complete'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(a.gpu)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.78)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.mem_get_info()[0] > 18 * 2 ** 30
    rows = f_read(APP / 'cohorts.json')[a.cohort]
    mapping = np.load(W / 'runs/pilot32_MSMAll_samplez/mapping.npz')
    cfg = dict(cohort=a.cohort, subjects=[r['subject'] for r in rows], resolutions=RESOLUTIONS, arms=ARMS, personal_steps=30, bandpass_Hz=[0.01, 0.1], actual_retained_frame_counts=True, heldout=[720, 1200], training='baseline only', longitudinal='inference without Retest weight updates', decoder_trainable=True, source_hashes={p.name: f_sha(p) for p in Path(__file__).parent.glob('*.py')}, canonical_hashes={p.name: f_sha(p) for p in (W / 'final').glob('*.py')})
    if (out / 'controlled_protocol.json').exists():
        assert f_read(out / 'controlled_protocol.json') == cfg
    else:
        f_save(out / 'controlled_protocol.json', cfg)
    models = f_load_models(W, SOURCE, out)
    phases = [('execution_check', rows[:1], [RESOLUTIONS[0]], ['single_day1', 'cross_equal'], 1), ('individual', rows, RESOLUTIONS, list(ARMS), 30)]
    for phase, selected, resolutions, arms, steps in phases:
        for definition, counts in resolutions:
            prior = ROOT / definition if definition in ['parcel17', 'network17'] else SOURCE
            for index, row in enumerate(selected):
                s = row['subject']
                for h in ['L', 'R']:
                    exp, group, path = models[h]
                    k = counts[h]
                    f_set_prior(exp, prior, h, k)
                    for arm in arms:
                        dest = out / phase / s / definition / h / arm
                        if (dest / 'complete.json').exists() or (dest / 'excluded.json').exists():
                            continue
                        try:
                            ys, audit = f_arrays(row, h, mapping, arm)
                            heldout = {}
                            for visit in row['visits']:
                                for day in [1, 2]:
                                    pair = []
                                    for pe in ['LR', 'RL']:
                                        f_run = f'rfMRI_REST{day}_{pe}'
                                        fd = np.load(Path(row['ancillary'][visit][f_run]['path']).with_name('FD.npy'))
                                        if not f_qc_segment(fd, 720, 1200)['passed']:
                                            raise ValueError('Heldout_QC/' + visit + '/' + f_run)
                                        if steps == 30:
                                            pair.append(f_signals(W, APP, row, visit, f_run, h, mapping, 720, 1200))
                                    if steps == 30:
                                        heldout[f'{visit}_day{day}'] = pair
                        except ValueError as e:
                            f_save(dest / 'excluded.json', dict(reason=str(e), before_optimization=True))
                            continue
                        f_save(out / 'controlled_status.json', dict(state='running', phase=phase, subject=s, index=index + 1, total=len(selected), definition=definition, hemisphere=h, arm=arm, steps=steps, pid=os.getpid()))
                        model, centers = f_fit(exp, group, path, ys, dest, out, s, h, k, definition + '_' + arm, steps)
                        hard = definition != 'network17'
                        primary, diag = f_infer(exp, model, centers, ys, hard)
                        np.save(dest / 'calibration_labels.npy', primary)
                        f_save(dest / 'training_segments.json', audit)
                        maps = {}
                        topos = {}
                        decodes = {'calibration': diag}
                        for key, pair in heldout.items():
                            v, d = f_infer(exp, model, centers, pair, hard)
                            np.save(dest / (key + '.npy'), v)
                            maps[key] = v
                            decodes[key] = d
                            topos[key] = f_topology(v, exp.adj, k)
                        comparisons = {}
                        if maps:
                            comparisons['baseline_day1_day2'] = f_agreement(maps['baseline_day1'], maps['baseline_day2'], k)
                            if 'retest_day1' in maps:
                                for day in [1, 2]:
                                    comparisons[f'baseline_retest_day{day}'] = f_agreement(maps[f'baseline_day{day}'], maps[f'retest_day{day}'], k)
                        f_save(dest / 'decode.json', decodes)
                        f_save(dest / 'map_metrics.json', dict(comparisons=comparisons, topology=topos, network_components_are_not_compact_parcel_fragments=not hard))
                        f_save(dest / 'complete.json', dict(subject=s, definition=definition, hemisphere=h, K_local=k, K_global=17 if '17' in definition else 2 * k, arm=arm, steps=steps, nominal_frames=960, retained_frames=sum((x['retained'] for x in audit)), Retest_fit=False, hard_connectivity=hard, heldout_independently_filtered=True))
                        del model, centers, ys, heldout
                        torch.cuda.empty_cache()
        if phase == 'execution_check':
            completed = list((out / phase).glob('*/*/*/*/complete.json'))
            assert len(completed) == 4, 'Both hemispheres and both arms must pass real-input checks'
            f_save(out / 'controlled_execution_check.json', dict(passed=True, both_hemispheres=True, matched_960_nominal_frames=True, full_FC=True, one_step_only=True))
    f_save(out / 'controlled_status.json', dict(state='complete', subjects=len(rows), updated_unix=time.time(), pid=os.getpid()))
