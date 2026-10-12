from f_runtime import f_work_root
"""Durable independent connectome validation, with matched topology ablation."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse, copy, json, time, fcntl, sys, traceback
from pathlib import Path
import numpy as np
import torch
from f_data import f_read, f_save, f_sha
from f_application_data import f_signals, RUNS, f_prepare
from f_application_metrics import f_regional, f_connectome, f_quality, f_pair_geometry
from f_application_statistics import f_summary, f_paired_bootstrap, f_paired_permutation, f_holm
from f_pipeline import f_geometry, f_set_prior, f_fit, f_group_matrix
from f_full_fc_vae import FullFCVAE
from f_model_ops import f_transferred_model, f_encode_projected, f_materialize, f_distance
from f_connectivity import f_decode
from f_metrics import f_topology, f_agreement
import f_plotting as plotting
KS = [100, 50, 150, 200]
METHODS = ['CoBRA', 'no_hard_connectivity', 'connected_group']

def f_status(out, **kwargs):
    f_save(out / 'status.json', dict(pid=os.getpid(), updated_unix=time.time(), **kwargs))

def f_lock_config(w, out):
    source = w / 'runs/corrected_personal_multires_20261006_v1'
    files = [p for p in Path(__file__).parent.glob('*.py')]
    fixed = [source / f'group/{h}/checkpoint.pt' for h in ['L', 'R']]
    fixed += [source / f'priors/{h}/{k}/{name}.npy' for h in ['L', 'R'] for k in KS for name in ['labels', 'seeds']]
    fixed += [w / 'runs/fullfc450_crossday_v1' / f'{name}_{h}.npy' for h in ['L', 'R'] for name in ['scale', 'attention_patches']]
    cfg = dict(version='connectome_validation_20261007_v1', K_bilateral=[200, 100, 300, 400], steps=30, calibration='REST1_LR+REST1_RL frames[0,480); filter independently then censor', YA400_measurement='baseline REST2_LR vs REST2_RL first480; same day and different PE, not longitudinal', retest_measurement='baseline and retest REST2_LR/RL run-level Fisher-z mean; labels frozen from baseline REST1', secondary_maps='independent day2(REST2) or retest REST1 calibration; reset group model and do not update primary labels', secondary_quality='primary maps scored on independent REST2 measurement runs; secondary map reproducibility reported separately', normalization='raw full within-hemisphere FC; fixed training-reference column scaling', bandpass_Hz=[0.01, 0.1], neural_prior_people=450, group_source=str(source), loss='existing all-ordered-cross-run reconstruction plus posterior-mean consistency and fixed regularizers', topology_ablation='same weights, centers, signals, initial labels, seeds, sweep order, energy, depth/size constraints; disable only hard connectivity gate', primary_statistic='median edgewise absolute-agreement ICC(A,1), K200 bilateral; paired subject bootstrap5000', secondary_statistic='FC similarity/MSE/concordance/identification; same calibration-derived frozen maps', quality='H, adjacent silhouette, DCBC<=35mm; complete cortex', motion_QC=dict(mean_FD_max_mm=0.3, max_fraction_FD_above_05=0.2), area_weights='baseline subject midthickness triangle area frozen across visits', family_metadata=False, Retest_caveat='Some identities inspected in historical pilots; no identity in current450 training but not wholly untouched model selection', model_source_sha256={str(p): f_sha(p) for p in fixed}, code_sha256={p.name: f_sha(p) for p in files})
    path = out / 'locked_protocol.json'
    if path.exists():
        assert f_read(path) == cfg, 'Configuration/code drift: create a new run rather than silently resume'
    else:
        f_save(path, cfg)
    return source

def f_load_models(w, source, out):
    result = {}
    for h in ['L', 'R']:
        exp = f_geometry(w, h, out, 'raw')
        model = f_transferred_model(FullFCVAE(exp.n, 128, 32, 2).cuda(), exp.n)
        path = source / f'group/{h}/checkpoint.pt'
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        assert checkpoint['step'] == 1000 and checkpoint['width'] == 128
        model.load_state_dict(checkpoint['model'])
        model.eval().requires_grad_(False)
        del checkpoint
        with torch.no_grad():
            fc = f_group_matrix(exp)
            exp.group_mu = f_encode_projected(model, model.input(fc), exp.graph, exp.patches, exp.pos)[0].detach()
            del fc
        result[h] = (exp, model, path)
        torch.cuda.empty_cache()
    return result

@torch.no_grad()
def f_calibration_decode(exp, model, centers, arrays):
    fc = f_materialize([torch.from_numpy(y).cuda() for y in arrays], exp.scale, exp.transform)
    mu = f_encode_projected(model, model.input(fc), exp.graph, exp.patches, exp.pos)[0]
    cost = f_distance(mu.mean(0), centers).cpu().numpy()
    del mu, fc
    torch.cuda.empty_cache()
    y = np.concatenate(arrays, axis=0)
    result = {}
    audit = {}
    for name, hard in [('CoBRA', True), ('no_hard_connectivity', False)]:
        result[name], audit[name] = f_decode(exp.initial, y, cost, exp.adj, exp.seeds, exp.depth, exp.edges, hard_connectivity=hard)
    return (result, audit)

def f_label_path(out, subject, h, k, calibration):
    return out / 'individual' / subject / h / str(k) / calibration

def f_adapt(w, source, out, row, h, k, calibration, models, mapping, steps=30):
    dest = f_label_path(out, row['subject'], h, k, calibration)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'complete.json').exists():
        return {name: np.load(dest / f'{name}.npy') for name in METHODS[:2]}
    exp, group, path = models[h]
    f_set_prior(exp, source, h, k)
    visit = 'retest' if calibration == 'retest_REST1' else 'baseline'
    day = 2 if calibration == 'baseline_REST2' else 1
    arrays = [f_signals(w, out, row, visit, f'rfMRI_REST{day}_{pe}', h, mapping) for pe in ['LR', 'RL']]
    model, centers = f_fit(exp, group, path, arrays, dest, out, row['subject'], h, k, calibration, steps)
    maps, audit = f_calibration_decode(exp, model, centers, arrays)
    for name, v in maps.items():
        np.save(dest / f'{name}.npy', v)
    f_save(dest / 'decode.json', audit)
    f_save(dest / 'complete.json', dict(subject=row['subject'], hemisphere=h, K=k, calibration=calibration, steps=steps, calibration_only=True, shared_model_and_centers=True, calibration_source={r: row['visits'][visit]['runs'][r] for r in [f'rfMRI_REST{day}_{pe}' for pe in ['LR', 'RL']]}, inputs_total_retained_frames=sum((len(y) for y in arrays)), all_parcels_connected=audit['CoBRA']['components'] == [1] * k))
    del model, centers, arrays
    torch.cuda.empty_cache()
    return maps

def f_maps_for(source, out, s, k, calibration='baseline_REST1'):
    result = {name: {} for name in METHODS}
    for h in ['L', 'R']:
        for name in METHODS[:2]:
            result[name][h] = np.load(f_label_path(out, s, h, k, calibration) / f'{name}.npy')
        result['connected_group'][h] = np.load(source / f'priors/{h}/{k}/labels.npy')
    return result

def f_measure(w, source, out, row, k, mapping, cohort):
    dest = out / 'measurements' / row['subject'] / f'{2 * k}.npz'
    if dest.exists():
        return
    maps = f_maps_for(source, out, row['subject'], k)
    areas = {h: np.load(row['area'][h]['path']) for h in ['L', 'R']}
    observations = {name: [] for name in METHODS}
    visit_values = {}
    for visit in row['visits']:
        by_run = {name: [] for name in METHODS}
        for pe in ['LR', 'RL']:
            f_run = 'rfMRI_REST2_' + pe
            ys = {h: f_signals(w, out, row, visit, f_run, h, mapping) for h in ['L', 'R']}
            assert len(ys['L']) == len(ys['R'])
            for name in METHODS:
                roi = {h: f_regional(ys[h], maps[name][h], areas[h], k) for h in ['L', 'R']}
                by_run[name].append(f_connectome(roi['L'], roi['R']))
            del ys
        visit_values[visit] = by_run
    for name in METHODS:
        if cohort == 'YA400':
            observations[name] = np.stack(visit_values['baseline'][name])
        else:
            observations[name] = np.stack([np.mean(visit_values[v][name], axis=0) for v in ['baseline', 'retest']])
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dest, **observations)
    f_save(dest.with_suffix('.json'), dict(subject=row['subject'], bilateral_K=2 * k, observation_order=['REST2_LR', 'REST2_RL'] if cohort == 'YA400' else ['baseline', 'retest'], labels_frozen_from='baseline REST1 LR/RL first480 only', area_weights_frozen=True, measurement_fitted=False))

def f_heldout_quality(w, source, out, row, mapping, models, pairs, cohort):
    dest = out / 'quality' / f"{row['subject']}.json"
    if dest.exists():
        return
    result = {}
    metric_list = []
    for h in ['L', 'R']:
        maps = {f'{2 * k}/{name}': m[h] for k in KS for name, m in f_maps_for(source, out, row['subject'], k).items()}
        result[h] = {}
        for visit in row['visits']:
            result[h][visit] = {}
            for pe in ['LR', 'RL']:
                f_status(out, state='running', stage='heldout_quality', subject=row['subject'], hemisphere=h, visit=visit, PE=pe)
                y = f_signals(w, out, row, visit, 'rfMRI_REST2_' + pe, h, mapping)
                result[h][visit][pe] = f_quality(y, maps, models[h][0].edges, pairs[h])
                del y
    f_save(dest, result)

def f_secondary(w, source, out, row, mapping, models, cohort):
    other = 'baseline_REST2' if cohort == 'YA400' else 'retest_REST1'
    agreement_dest = out / 'map_reproducibility' / f"{row['subject']}.json"
    if agreement_dest.exists():
        return
    result = {}
    for h in ['L', 'R']:
        result[h] = {}
        for k in KS:
            later = f_adapt(w, source, out, row, h, k, other, models, mapping)
            first = f_maps_for(source, out, row['subject'], k)
            result[h][str(2 * k)] = {name: dict(**f_agreement(first[name][h], later[name], k), anchored_Dice=float(np.mean([2 * np.sum((first[name][h] == a) & (later[name] == a)) / (np.sum(first[name][h] == a) + np.sum(later[name] == a)) for a in range(k)])), first_topology=f_topology(first[name][h], models[h][0].adj, k), second_topology=f_topology(later[name], models[h][0].adj, k)) for name in METHODS[:2]}
    result['interpretation'] = 'Independent day1/day2 calibration maps' if cohort == 'YA400' else 'Independent baseline/retest calibration maps reset to same group model; no joint longitudinal fitting'
    if cohort == 'retest':
        areas = {h: np.load(row['area'][h]['path']) for h in ['L', 'R']}
        for k in KS:
            primary = np.load(out / 'measurements' / row['subject'] / f'{2 * k}.npz')
            maps = f_maps_for(source, out, row['subject'], k, other)
            data = {}
            for name in METHODS[:2]:
                later = []
                for pe in ['LR', 'RL']:
                    ys = {h: f_signals(w, out, row, 'retest', 'rfMRI_REST2_' + pe, h, mapping) for h in ['L', 'R']}
                    roi = {h: f_regional(ys[h], maps[name][h], areas[h], k) for h in ['L', 'R']}
                    later.append(f_connectome(roi['L'], roi['R']))
                data[name] = np.stack([primary[name][0], np.mean(later, axis=0)])
            p = out / 'recalibrated_measurements' / row['subject'] / f'{2 * k}.npz'
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(p, **data)
    f_save(agreement_dest, result)

def f_wrong_person(w, source, out, rows, mapping, cohort):
    for i, row in enumerate(rows):
        donor = rows[(i + 1) % len(rows)]['subject']
        s = row['subject']
        areas = {h: np.load(row['area'][h]['path']) for h in ['L', 'R']}
        if all(((out / 'wrong_person' / s / f'{2 * k}.npz').exists() for k in KS)):
            continue
        allmaps = {k: f_maps_for(source, out, donor, k)['CoBRA'] for k in KS}
        result = {k: {} for k in KS}
        for visit in row['visits']:
            for k in KS:
                result[k][visit] = []
            for pe in ['LR', 'RL']:
                f_status(out, state='running', stage='wrong_person_control', subject=s, donor=donor, visit=visit, PE=pe)
                ys = {h: f_signals(w, out, row, visit, 'rfMRI_REST2_' + pe, h, mapping) for h in ['L', 'R']}
                for k in KS:
                    roi = {h: f_regional(ys[h], allmaps[k][h], areas[h], k) for h in ['L', 'R']}
                    result[k][visit].append(f_connectome(roi['L'], roi['R']))
        for k in KS:
            observations = np.stack(result[k]['baseline']) if cohort == 'YA400' else np.stack([np.mean(result[k][v], axis=0) for v in ['baseline', 'retest']])
            p = out / 'wrong_person' / s / f'{2 * k}.npz'
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(p, wrong_person=observations)
            f_save(p.with_suffix('.json'), dict(subject=s, donor=donor, derangement='next locked selected subject; cyclic', labels_frozen=True))

def f_statistics(out, rows, draws=5000):
    (out / 'statistics').mkdir(parents=True, exist_ok=True)
    for k in KS:
        p = out / 'statistics' / f'{2 * k}.json'
        if p.exists():
            continue
        f_status(out, state='running', stage='statistics', K=2 * k)
        files = [np.load(out / 'measurements' / r['subject'] / f'{2 * k}.npz') for r in rows]
        methods = {name: np.stack([f[name] for f in files]) for name in METHODS}
        if all(((out / 'wrong_person' / r['subject'] / f'{2 * k}.npz').exists() for r in rows)):
            methods['wrong_person'] = np.stack([np.load(out / 'wrong_person' / r['subject'] / f'{2 * k}.npz')['wrong_person'] for r in rows])
        pairs = {name: (x[:, 0], x[:, 1]) for name, x in methods.items()}
        details = {name: f_summary(*pair) for name, pair in pairs.items()}
        for name, pair in pairs.items():
            np.save(out / 'statistics' / f'ICC_{2 * k}_{name}.npy', __import__('f_application_statistics').f_edge_icc(*pair))
        result = dict(bilateral_K=2 * k, subjects=[r['subject'] for r in rows], methods=details, comparison='K200 confirmatory; other K descriptive/secondary; individual vertices and edges are not independent participants')
        if k == 100:
            result['paired_bootstrap5000'] = f_paired_bootstrap(pairs, 'connected_group', draws=draws)
            result['CoBRA_minus_no_hard_bootstrap'] = f_paired_bootstrap({n: pairs[n] for n in METHODS[:2]}, 'no_hard_connectivity', draws=draws)
            result['paired_permutation5000'] = {name: f_paired_permutation(pairs['CoBRA'], pairs[name], draws=draws) for name in pairs if name != 'CoBRA'}
            result['Holm_secondary_p'] = f_holm({name: q['two_sided_p'] for name, q in result['paired_permutation5000'].items() if name != 'connected_group'})
        if all(((out / 'recalibrated_measurements' / r['subject'] / f'{2 * k}.npz').exists() for r in rows)):
            result['independently_recalibrated'] = {}
            for name in METHODS[:2]:
                z = np.stack([np.load(out / 'recalibrated_measurements' / r['subject'] / f'{2 * k}.npz')[name] for r in rows])
                result['independently_recalibrated'][name] = f_summary(z[:, 0], z[:, 1])
        f_save(p, result)
    f_status(out, state='complete', subjects=len(rows), hemisphere_fits='all selected participants and all4 resolutions', completed_unix=time.time())

def f_run(args):
    w = Path(args.workspace)
    root = w / 'runs' / args.run_name
    root.mkdir(parents=True, exist_ok=True)
    source = f_lock_config(w, root)
    ready = None
    cohort_ready = root / f'cohort_{args.cohort}.json'
    while not (root / 'cohorts.json').exists() and (not cohort_ready.exists()):
        if args.smoke:
            candidates = [f_read(p) for p in sorted((root / 'QC').glob('*.json'))]
            candidates = [r for r in candidates if r['passed'] and (len(r['visits']) == 2 if args.cohort == 'retest' else len(r['visits']) == 1)]
            if candidates:
                ready = candidates[:1]
                break
        f_status(root / f'queue_{args.cohort}', state='waiting_for_locked_QC', gpu=args.gpu)
        time.sleep(30)
    rows = ready if ready is not None else f_read(cohort_ready) if cohort_ready.exists() else f_read(root / 'cohorts.json')[args.cohort]
    assert len(rows) > 2 or args.smoke, 'No usable paired cohort'
    out = root / args.cohort
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'workflow.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    f_save(out / ('smoke_subjects.json' if args.smoke else 'subjects.json'), rows)
    mapping = np.load(w / 'runs/pilot32_MSMAll_samplez/mapping.npz')
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.gpu)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.78)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.mem_get_info()[0] > 18 * 2 ** 30, 'Need18GiB free; never interrupt other users'
    models = f_load_models(w, source, out)
    if args.smoke:
        row = rows[0]
        for h in ['L', 'R']:
            f_adapt(w, source, out / 'smoke', row, h, 100, 'baseline_REST1', models, mapping, steps=1)
        f_measure(w, source, out / 'smoke', row, 100, mapping, args.cohort)
        f_save(out / 'execution_check.json', dict(passed=True, both_hemispheres=True, full_FC=True, two_independent_calibration_runs=True, measurement_fitted=False, hard_connectivity_ablation=True, actual_measurement_FC=True, step_budget=1, subject=row['subject']))
        f_status(out, state='smoke_complete')
        return
    assert f_read(out / 'execution_check.json')['passed']
    for k in KS:
        for i, row in enumerate(rows):
            for h in ['L', 'R']:
                f_status(out, state='running', stage='primary_adaptation', subject=row['subject'], subject_index=i + 1, total_subjects=len(rows), hemisphere=h, K=2 * k, gpu=args.gpu)
                f_adapt(w, source, out, row, h, k, 'baseline_REST1', models, mapping)
            f_measure(w, source, out, row, k, mapping, args.cohort)
    pairs = {h: f_pair_geometry(w, h, root) for h in ['L', 'R']}
    for i, row in enumerate(rows):
        f_heldout_quality(w, source, out, row, mapping, models, pairs, args.cohort)
        f_secondary(w, source, out, row, mapping, models, args.cohort)
        if i < 3:
            for k in KS:
                maps = f_maps_for(source, out, row['subject'], k)
                dest = out / 'figures' / row['subject']
                dest.mkdir(parents=True, exist_ok=True)
                for name, labels in maps.items():
                    plotting.f_export(labels, mapping, row['visits']['baseline']['runs']['rfMRI_REST1_LR'], k, dest / f'{name}_{2 * k}.dlabel.nii', f'{name}_{2 * k}')
    for h, (exp, model, path) in models.items():
        del exp, model
    models.clear()
    torch.cuda.empty_cache()
    f_wrong_person(w, source, out, rows, mapping, args.cohort)
    f_statistics(out, rows)
