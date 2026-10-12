"""Inspect, check and launch the isolated direct-FC512 / patch512 trial."""
import argparse
import copy
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--phase', choices=['test', 'inspect', 'smoke', 'launch', 'run'], required=True)
parser.add_argument('--config', default=str(HERE/'config.json'))
parser.add_argument('--resume', action='store_true')
ARGS = parser.parse_args()
CFG = json.loads(Path(ARGS.config).read_text(encoding='utf-8-sig'))
W = Path(CFG['input_workspace']).resolve()
OUT = Path(CFG['output_root']).resolve()
PACKAGE = Path(CFG['package_root']).resolve()
BASE = W/'runs'/CFG['baseline_run']
assert OUT != W and W not in OUT.parents and OUT != PACKAGE and PACKAGE not in OUT.parents
os.environ.update(CUDA_VISIBLE_DEVICES=CFG['gpu'], OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', PARCELLATION_WORKSPACE=str(W))
ROUTES = json.loads((PACKAGE/'configs/module_routes.json').read_text())
sys.path[:0] = [str(HERE), str(PACKAGE/'Common')] + sorted({str((PACKAGE/p).parent) for p in ROUTES.values()})

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from f_data import f_read, f_save, f_sha, f_cohort, f_training, f_evaluation_days
from f_pipeline import f_geometry, f_set_prior
from f_model_ops import f_regularization, f_regional_centers, f_distance
from f_surface_prior_vae import f_diagonal_kl_to_prior
from f_connectivity import f_decode
from f_metrics import f_scores, f_agreement, f_topology
from f_capacity_vae import f_new_model, f_parameter_count, f_encode_dense, f_encode_signals, f_all_entry_backward, f_test_exact_operations


def f_sources():
    names = ['f_data', 'f_pipeline', 'f_model_ops', 'f_surface_prior_vae', 'f_connectivity', 'f_metrics', 'f_full_fc_vae', 'f_capacity_vae']
    paths = [Path(__file__).resolve()] + [Path(sys.modules[name].__file__).resolve() for name in names]
    return {str(p): f_sha(p) for p in paths}


def f_status(stage, **values):
    f_save(OUT/'status.json', dict(state='running', stage=stage, pid=os.getpid(), updated_unix=time.time(), **values))


def f_memory():
    free, total = torch.cuda.mem_get_info()
    return dict(allocated_GiB=torch.cuda.memory_allocated()/2**30, reserved_GiB=torch.cuda.memory_reserved()/2**30,
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30, peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,
                device_free_GiB=free/2**30, device_total_GiB=total/2**30)


def f_wait_gpu():
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(CFG['gpu_budget_GiB']*2**30/total)
    required = CFG['gpu_budget_GiB']+CFG['gpu_headroom_GiB']
    while True:
        free, _ = torch.cuda.mem_get_info()
        available = (free+torch.cuda.memory_reserved())/2**30
        if available >= required:
            return
        f_status('waiting_for_gpu_headroom', required_GiB=required, available_GiB=available, physical_gpu=CFG['gpu'])
        print('WAIT GPU', available, 'required', required, flush=True)
        time.sleep(30)


def f_inspect():
    assert CFG['width'] == 512 and CFG['patch_count'] == 512 and CFG['latent_dim'] == 32
    assert CFG['attention_layers'] == 2 and CFG['attention_heads'] == 4 and CFG['bandpass_hz'] == [.01, .1]
    assert CFG['personal_arm'] == 'joint_equal' and CFG['training_frames'] == [0, 240]
    cohort, rows = f_cohort(W)
    assert [r['subject'] for r in rows] == CFG['development_subjects']
    assert len(cohort['train']) == CFG['group_training_subjects'] == 450
    definition = f_read(W/'runs/fullfc450_crossday_v1/cache_definition.json')
    assert definition['filter_hz'] == [.01, .1]
    baseline = f_read(BASE/'configuration.json')
    assert baseline['subjects'] == CFG['development_subjects'] and baseline['width'] == 128
    assert baseline['bandpass_hz'] == [.01, .1] and baseline['personal_steps'] == CFG['personal_steps']
    assert baseline['arms']['joint_equal']['frames_per_run'] == 240
    h = CFG['hemisphere']; k = CFG['K_per_hemisphere']
    n = len(np.load(W/f'runs/pilot32_MSMAll_samplez/lpe_{h}.npy'))
    model = f_new_model(n, CFG['width'], CFG['latent_dim'], CFG['attention_layers'])
    actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = f_parameter_count(n, CFG['width'], CFG['latent_dim'])
    assert actual == expected
    del model
    inputs = [W/f'runs/fullfc450_crossday_v1/group_{h}.npy', W/f'runs/fullfc450_crossday_v1/scale_{h}.npy',
              W/f'runs/pilot32_MSMAll_samplez/lpe_{h}.npy', W/f'runs/pilot32_MSMAll_samplez/edges_{h}.npy',
              BASE/f'priors/{h}/{k}/labels.npy', BASE/f'priors/{h}/{k}/seeds.npy',
              W/f'resources/fullfc_distance/fullfc450_distance_crossday_v1/distances_{h}.npy']
    inputs += [BASE/f'individual/{r["subject"]}/{h}/{k}/joint_equal/maps.npy' for r in rows]
    for r in rows:
        for p in r['runs'].values():
            assert Path(p).exists(), p
    provenance = {}
    for p in inputs:
        assert p.is_file(), p
        st = p.stat()
        provenance[str(p)] = dict(size=st.st_size, mtime_ns=st.st_mtime_ns,
                                 sha256=f_sha(p) if st.st_size < 100*2**20 else None)
    return dict(configuration=CFG, vertices=n, trainable_parameters=actual,
                baseline_parameters=f_parameter_count(n, 128, 32), source_sha256=f_sources(),
                config_sha256=f_sha(ARGS.config), reused_input_provenance=provenance,
                large_input_hash_policy='Size and nanosecond mtime recorded; SHA256 only below100MiB.',
                full_FC_projection=True, all_FC_entries_reconstructed=True, reserved_retest_not_used=True,
                changed_factors=['feature_width_128_to_512', 'attention_patch_count_128_to_512'])


def f_setup():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    cohort, rows = f_cohort(W)
    h = CFG['hemisphere']
    exp = f_geometry(W, h, OUT, 'raw')
    patch_file = OUT/f'attention_patches_{h}.npy'
    if patch_file.exists():
        patches = np.load(patch_file)
    else:
        pos = exp.pos.cpu().numpy()
        patches = MiniBatchKMeans(CFG['patch_count'], random_state=CFG['seed'], batch_size=4096, n_init=3).fit_predict(pos).astype('int64')
        np.save(patch_file, patches)
    assert patches.shape == (exp.n,) and np.array_equal(np.unique(patches), np.arange(CFG['patch_count']))
    exp.patches = torch.from_numpy(patches).long().cuda()
    sizes = np.bincount(patches)
    f_save(OUT/'patch_definition.json', dict(count=len(sizes), sizes=sizes.tolist(), minimum=int(sizes.min()), maximum=int(sizes.max()),
        algorithm='MiniBatchKMeans on existing five-dimensional LPE; same constructor as128-token reference',
        n_init=3, batch_size=4096, seed=CFG['seed'], input_sha256=f_sha(W/f'runs/pilot32_MSMAll_samplez/lpe_{h}.npy'),
        geometry_only=True, learned_assignment=False, patch_sha256=f_sha(patch_file)))
    return exp, rows


def f_link_input_segments(rows):
    # Existing filtered arrays are read-only inputs; canonical metadata are
    # revalidated by f_segment before loading. Missing arrays are regenerated.
    records = []
    for row in rows:
        for run in row['runs']:
            for start, stop in [(0, 240), (720, 1200)]:
                name = f'{CFG["hemisphere"]}_{start}_{stop}'
                source = BASE/'filtered_segments'/row['subject']/run
                target = OUT/'filtered_segments'/row['subject']/run
                if all((source/(name+ext)).is_file() for ext in ['.npy', '.json']):
                    target.mkdir(parents=True, exist_ok=True)
                    for ext in ['.npy', '.json']:
                        dst = target/(name+ext); src = source/(name+ext)
                        if not dst.exists():
                            dst.symlink_to(src)
                        if dst.is_symlink():
                            assert dst.resolve() == src.resolve()
                    records.append(dict(source=str(source/(name+'.npy')), metadata_sha256=f_sha(source/(name+'.json'))))
    f_save(OUT/'read_only_filtered_inputs.json', records)


def f_group_fc(exp):
    matrix = np.load(exp.group_source, mmap_mode='r')
    assert matrix.shape == (exp.n, exp.n)
    fc = torch.empty(1, exp.n, exp.n, dtype=torch.float32, device='cuda')
    for start in range(0, exp.n, 256):
        stop = min(start+256, exp.n)
        fc[0, start:stop] = torch.from_numpy(np.array(matrix[start:stop], copy=True)).cuda()/exp.scale
    return fc


def f_gradient_audit(model, path):
    audit = {name: float(p.grad.abs().max()) if p.grad is not None else None for name, p in model.named_parameters()}
    assert all(v is not None and np.isfinite(v) and v > 0 for v in audit.values()), audit
    f_save(path, audit)


def f_train_group(exp, steps, smoke=False):
    dest = OUT/('execution_check/group' if smoke else 'group')
    dest.mkdir(parents=True, exist_ok=True)
    path = dest/'checkpoint.pt'
    torch.manual_seed(CFG['seed'])
    model = f_new_model(exp.n, CFG['width'], CFG['latent_dim'], CFG['attention_layers']).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['group_lr'])
    start = 0; history = []
    if path.exists():
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert state['steps'] == steps and state['architecture'] == [512, 512, 32]
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        start = state['step']; history = state['history']
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state(state['cuda_rng'])
        del state
    fc = f_group_fc(exp)
    for step in range(start, steps):
        tick = time.time(); model.train(); optimizer.zero_grad(set_to_none=True)
        mu, lv = f_encode_dense(model, fc, exp, CFG['activation_checkpointing'])
        hidden = model.decoder_hidden(mu, lv, sample=True)
        rec, gradient = f_all_entry_backward(model, hidden, fc=fc, cross=False, chunk=CFG['reconstruction_chunk'])
        kl = CFG['group_kl']*f_diagonal_kl_to_prior(mu, lv, torch.zeros_like(mu))
        torch.autograd.backward([hidden, kl], [gradient, None])
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        assert np.isfinite(rec) and torch.isfinite(kl) and torch.isfinite(norm)
        if step == 0:
            f_gradient_audit(model, dest/'gradient_audit.json')
        optimizer.step()
        row = dict(step=step+1, reconstruction=rec, total=rec+float(kl.detach()), seconds=time.time()-tick)
        history.append(row)
        f_status('group_pretraining', step=step+1, total_steps=steps, loss=row['total'], step_seconds=row['seconds'], width=512, patch_count=512, memory=f_memory())
        if step == 0 or (step+1) % 10 == 0:
            print('GROUP', json.dumps(row), flush=True)
        if (step+1) % 50 == 0 or step+1 == steps:
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step+1, steps=steps,
                architecture=[512, 512, 32], history=history, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), path)
    model.eval().requires_grad_(False)
    with torch.no_grad():
        exp.group_mu = f_encode_dense(model, fc, exp, False)[0].detach()
    f_save(dest/'complete.json', dict(steps=steps, trainable_parameters=sum(p.numel() for p in model.parameters()), memory=f_memory(), full_FC=True))
    del fc, optimizer
    torch.cuda.empty_cache()
    f_set_prior(exp, BASE, CFG['hemisphere'], CFG['K_per_hemisphere'])
    return model, path


def f_train_person(exp, source, source_path, row, steps, smoke=False):
    subject = row['subject']
    dest = OUT/('execution_check/personal' if smoke else f'individual/{subject}')
    dest.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(CFG['seed']+int(subject))
    model = copy.deepcopy(source).eval().requires_grad_(True)
    arrays, protocol = f_training(row, CFG['hemisphere'], CFG['personal_arm'], np.load(W/'runs/pilot32_MSMAll_samplez/mapping.npz'), OUT/'filtered_segments')
    signals = [torch.from_numpy(y).cuda() for y in arrays]
    del arrays
    with torch.no_grad():
        baseline = f_encode_signals(model, signals, exp.scale, exp, False)[0]
        centers = .5*exp.group_centers+.5*f_regional_centers(baseline.mean(0), exp.labels, exp.core, exp.k)
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['personal_lr'])
    path = dest/'checkpoint.pt'; history = []; start = 0; source_hash = f_sha(source_path)
    if path.exists():
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert state['steps'] == steps and state['group_sha256'] == source_hash and state['subject'] == subject
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        history = state['history']; start = state['step']; del state
    for step in range(start, steps):
        tick = time.time(); optimizer.zero_grad(set_to_none=True)
        mu, lv = f_encode_signals(model, signals, exp.scale, exp, CFG['activation_checkpointing'])
        hidden = model.decoder_hidden(mu, lv, sample=False)
        rec, gradient = f_all_entry_backward(model, hidden, scale=exp.scale, signals=signals, cross=True, chunk=CFG['reconstruction_chunk'])
        penalty, details = f_regularization(exp, model, mu, lv, baseline, centers, initial)
        torch.autograd.backward([hidden, penalty], [gradient, None])
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        assert np.isfinite(rec) and torch.isfinite(penalty) and torch.isfinite(norm)
        if step == 0:
            f_gradient_audit(model, dest/'gradient_audit.json')
        optimizer.step()
        details.update(step=step+1, reconstruction=rec, total=rec+float(penalty.detach()), seconds=time.time()-tick)
        history.append(details)
        f_status('personal_finetuning', subject=subject, step=step+1, total_steps=steps, loss=details['total'], memory=f_memory())
        if step == 0 or (step+1) % 5 == 0 or step+1 == steps:
            print('PERSONAL', subject, json.dumps(details), flush=True)
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step+1, steps=steps, subject=subject,
                group_sha256=source_hash, history=history), path)
    with torch.no_grad():
        mu = f_encode_signals(model, signals, exp.scale, exp, False)[0]
        centers = .5*exp.group_centers+.5*f_regional_centers(mu.mean(0), exp.labels, exp.core, exp.k)
    changes = {name: float((p-initial[name]).abs().max()) for name, p in model.named_parameters()}
    assert all(np.isfinite(v) and v > 0 for v in changes.values())
    np.save(dest/'centers.npy', centers.cpu().numpy())
    f_save(dest/'fit.json', dict(subject=subject, steps=steps, width=512, patch_count=512, history=history, parameter_changes=changes,
        memory=f_memory(), group_sha256=source_hash, full_FC=True, all_ordered_distinct_pairs=True, protocol=protocol))
    del signals, initial, optimizer, baseline
    torch.cuda.empty_cache()
    return model, centers


@torch.no_grad()
def f_infer_person(exp, model, centers, y):
    signal = torch.from_numpy(y).cuda()
    mu = f_encode_signals(model, [signal], exp.scale, exp, False)[0][0]
    cost = f_distance(mu, centers).cpu().numpy()
    del signal, mu
    torch.cuda.empty_cache()
    return f_decode(exp.initial, y, cost, exp.adj, exp.seeds, exp.depth, exp.edges)


def f_evaluate(exp, rows):
    h = CFG['hemisphere']; k = CFG['K_per_hemisphere']
    mapping = np.load(W/'runs/pilot32_MSMAll_samplez/mapping.npz')
    dist = np.load(W/f'resources/fullfc_distance/fullfc450_distance_crossday_v1/distances_{h}.npy', mmap_mode='r')
    assert dist.shape == (exp.n, exp.n)
    methods = ['width512_patch512', 'width128_patch128', 'fixed_connected_group']
    records = {m: [] for m in methods}; maps_all = {m: [] for m in methods}
    for idx, row in enumerate(rows):
        subject = row['subject']; saved = OUT/f'metrics/{subject}.json'
        maps = {methods[0]: np.load(OUT/f'individual/{subject}/maps.npy'),
                methods[1]: np.load(BASE/f'individual/{subject}/{h}/{k}/joint_equal/maps.npy'),
                methods[2]: np.stack([exp.initial, exp.initial])}
        if saved.exists():
            values = f_read(saved)
        else:
            ys = f_evaluation_days(row, h, mapping, OUT/'filtered_segments')
            values = {}
            for method in methods:
                pair = maps[method].astype('int64')
                stats = f_agreement(pair[0], pair[1], k)
                scores = []
                for day in [0, 1]:
                    f_status('heldout_metrics', subject=subject, index=idx+1, total_subjects=len(rows), method=method, day=day+1)
                    signal = torch.from_numpy(ys[1-day]).cuda()
                    scores.append(f_scores(signal, pair[day], exp.edges, dist, k))
                    del signal
                    torch.cuda.empty_cache()
                stats.update({key: float(np.mean([s[key] for s in scores])) for key in ['homogeneity', 'adjacent_silhouette', 'DCBC']})
                topology = [f_topology(pair[day], exp.adj, k) for day in [0, 1]]
                stats.update(extra_components=float(np.mean([sum(t['components'])-k for t in topology])),
                    disconnected_parcel_fraction=float(np.mean([np.mean(np.asarray(t['components']) > 1) for t in topology])),
                    change_from_group=float(np.mean(pair != exp.initial)))
                values[method] = dict(subject=subject, statistics=stats, opposite_day_scores=scores, topology=topology)
            f_save(saved, values)
        for method in methods:
            records[method].append(values[method]); maps_all[method].append(maps[method])
        print('METRICS', subject, json.dumps({m: values[m]['statistics'] for m in methods}), flush=True)
    summary = dict(configuration=CFG, caveat=CFG['notes'], methods={})
    for method in methods:
        stats = {key: dict(mean=float(np.mean([r['statistics'][key] for r in records[method]])), sd=float(np.std([r['statistics'][key] for r in records[method]], ddof=1))) for key in records[method][0]['statistics']}
        mm = np.asarray(maps_all[method]); n = len(rows)
        cross = np.array([[f_agreement(mm[i, 0], mm[j, 1], k)['ARI'] for j in range(n)] for i in range(n)])
        mask = ~np.eye(n, dtype=bool)
        specificity = dict(intra_subject_ARI=float(np.diag(cross).mean()), inter_subject_ARI=float(cross[mask].mean()),
            intra_minus_inter_ARI=float(np.diag(cross).mean()-cross[mask].mean()), cross_subject_day1_day2_ARI=cross.tolist())
        summary['methods'][method] = dict(statistics=stats, subject_specificity=specificity, subjects=records[method])
        np.save(OUT/f'maps_{method}.npy', mm)
    rng = np.random.default_rng(CFG['seed']); boot = rng.integers(0, len(rows), (4000, len(rows)))
    paired = {}
    for key in ['ARI', 'Dice', 'homogeneity', 'adjacent_silhouette', 'DCBC', 'change_from_group']:
        delta = np.array([a['statistics'][key]-b['statistics'][key] for a, b in zip(records[methods[0]], records[methods[1]])])
        paired[key] = dict(mean_difference=float(delta.mean()), participant_bootstrap_ci95=np.quantile(delta[boot].mean(1), [.025, .975]).tolist(), values=delta.tolist())
    paired['caveat'] = 'Exploratory12-person development comparison. Participant bootstrap assumes independence; HCP family relationships not modeled. Width and patch count change together.'
    f_save(OUT/'paired_contrasts.json', paired)
    f_save(OUT/'summary.json', summary)
    with (OUT/'comparison_subjects.csv').open('w', newline='') as stream:
        keys = ['subject', 'method']+list(records[methods[0]][0]['statistics'])
        writer = csv.DictWriter(stream, fieldnames=keys); writer.writeheader()
        for method in methods:
            for row in records[method]:
                writer.writerow(dict(subject=row['subject'], method=method, **row['statistics']))


def f_smoke():
    inspect = f_inspect(); OUT.mkdir(parents=True, exist_ok=True)
    f_save(OUT/'architecture_and_inputs.json', inspect)
    f_wait_gpu(); exp, rows = f_setup(); f_link_input_segments(rows[:1])
    start = time.time(); model, path = f_train_group(exp, 1, True)
    group_seconds = time.time()-start
    start = time.time(); tuned, centers = f_train_person(exp, model, path, rows[0], 1, True)
    result = dict(passed=True, parameters=sum(p.numel() for p in tuned.parameters()), width=512, patch_count=512, latent_dim=32,
                  full_cortex_vertices=exp.n, group_stage_wall_seconds=group_seconds, personal_stage_wall_seconds=time.time()-start,
                  all_active_parameters_have_finite_nonzero_gradients=True, memory=f_memory(), source_sha256=f_sources(), config_sha256=f_sha(ARGS.config))
    f_save(OUT/'execution_check.json', result)
    f_save(OUT/'status.json', dict(state='smoke_complete', **result))
    print(json.dumps(result), flush=True)


def f_run():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'workflow.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        check = f_read(OUT/'execution_check.json')
        assert check['passed'] and check['source_sha256'] == f_sources() and check['config_sha256'] == f_sha(ARGS.config)
        cfg_path = OUT/'configuration.json'
        if cfg_path.exists():
            assert f_read(cfg_path) == CFG
        else:
            f_save(cfg_path, CFG)
        original_inputs = f_read(OUT/'architecture_and_inputs.json')['reused_input_provenance']
        f_wait_gpu(); exp, rows = f_setup(); f_link_input_segments(rows)
        model, path = f_train_group(exp, CFG['group_steps'])
        mapping = np.load(W/'runs/pilot32_MSMAll_samplez/mapping.npz')
        for idx, row in enumerate(rows):
            dest = OUT/f'individual/{row["subject"]}'
            if (dest/'complete.json').exists():
                continue
            f_wait_gpu()
            tuned, centers = f_train_person(exp, model, path, row, CFG['personal_steps'])
            ys = f_evaluation_days(row, CFG['hemisphere'], mapping, OUT/'filtered_segments')
            maps = []; audits = []
            for day, signal in enumerate(ys):
                f_status('connected_inference', subject=row['subject'], index=idx+1, total_subjects=len(rows), day=day+1)
                labels, audit = f_infer_person(exp, tuned, centers, signal)
                maps.append(labels); audits.append(audit)
            np.save(dest/'maps.npy', np.stack(maps)); f_save(dest/'decode.json', audits)
            f_save(dest/'complete.json', dict(subject=row['subject'], days=2, all_parcels_connected=True))
            del tuned, centers, ys
            torch.cuda.empty_cache()
            print('SUBJECT COMPLETE', row['subject'], flush=True)
        del model
        torch.cuda.empty_cache()
        f_evaluate(exp, rows)
        for name, identity in original_inputs.items():
            p = Path(name); st = p.stat()
            assert st.st_size == identity['size'] and st.st_mtime_ns == identity['mtime_ns'], name
            if identity['sha256']:
                assert f_sha(p) == identity['sha256'], name
        assert check['source_sha256'] == f_sources()
        f_save(OUT/'audits.json', dict(read_only_original_inputs_unchanged=True, sources_unchanged=True, full_FC=True,
            bandpass_hz=[.01, .1], all_ordered_distinct_run_reconstruction=True, reserved_retest_unused=True,
            separate_filtered_temporal_holdouts=True, all_parcels_connected=True))
        f_save(OUT/'status.json', dict(state='complete', subjects=len(rows), hemisphere=CFG['hemisphere'], K_per_hemisphere=exp.k, updated_unix=time.time()))
        print('TRIAL COMPLETE', str(OUT), flush=True)


def f_launch():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'launch.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'launch.json').exists():
            previous = f_read(OUT/'launch.json')
            cmd = Path(f'/proc/{previous["pid"]}/cmdline')
            if cmd.exists() and str(Path(__file__).resolve()).encode() in cmd.read_bytes():
                raise RuntimeError('This trial is already running: '+str(previous['pid']))
            if not ARGS.resume:
                raise RuntimeError('A previous launch exists; inspect it and explicitly --resume.')
        if (OUT/'status.json').exists() and f_read(OUT/'status.json').get('state') == 'complete':
            raise RuntimeError('Already complete; do not duplicate.')
        check = f_read(OUT/'execution_check.json')
        assert check['passed'] and check['source_sha256'] == f_sources() and check['config_sha256'] == f_sha(ARGS.config)
        tests = f_read(HERE/'tests.json')
        assert tests['passed'] and tests['source_sha256'] == f_sources()
        with (OUT/'workflow.log').open('ab') as log:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), '--phase', 'run', '--config', str(Path(ARGS.config).resolve())],
                cwd=HERE, env=dict(os.environ), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        launch = dict(pid=child.pid, physical_gpu=CFG['gpu'], launched_unix=time.time(), output=str(OUT), log=str(OUT/'workflow.log'))
        f_save(OUT/'launch.json', launch)
        print(json.dumps(launch), flush=True)


if __name__ == '__main__':
    try:
        if ARGS.phase == 'test':
            test = f_test_exact_operations(); test['source_sha256'] = f_sources()
            f_save(HERE/'tests.json', test); print(json.dumps(test), flush=True)
        elif ARGS.phase == 'inspect':
            inspect = f_inspect(); OUT.mkdir(parents=True, exist_ok=True)
            f_save(OUT/'architecture_and_inputs.json', inspect)
            print(json.dumps({k: v for k, v in inspect.items() if k not in ['source_sha256', 'reused_input_provenance']}), flush=True)
        else:
            {'smoke': f_smoke, 'run': f_run, 'launch': f_launch}[ARGS.phase]()
    except Exception as exc:
        if ARGS.phase in ['smoke', 'run'] and not isinstance(exc, BlockingIOError):
            f_save(OUT/'status.json', dict(state='failed', phase=ARGS.phase, error=repr(exc), pid=os.getpid(), updated_unix=time.time()))
        traceback.print_exc()
        raise
