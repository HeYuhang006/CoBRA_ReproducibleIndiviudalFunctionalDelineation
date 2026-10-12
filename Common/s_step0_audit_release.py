"""Check source structure and dry-run entrypoints without MRI or third-party installs."""
from pathlib import Path
import ast
import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['Main', 'DRFC', 'Wang_style', 'GPIP', 'TraditionalMethod']


def f_audit(check_manifest=True):
    files = [p for p in sorted(ROOT.rglob('*')) if p.is_file()
             and not any(s in p.parts for s in ['.git', '__pycache__', '.deps', '.venv'])]
    python_files = [p for p in files if p.suffix == '.py']
    naming_exceptions = []
    for p in python_files:
        parsed = ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
        nonprefixed = [n.name for n in parsed.body
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and not n.name.startswith('f_')]
        if nonprefixed:
            # Frozen production-recovery snapshots retain their original names/hashes.
            assert 'experiments' in p.parts, (str(p), nonprefixed)
            naming_exceptions.append(dict(path=p.relative_to(ROOT).as_posix(), functions=nonprefixed))

    routes = json.loads((ROOT / 'configs/module_routes.json').read_text(encoding='utf-8-sig'))
    assert all((ROOT / path).is_file() for path in routes.values())
    cfg = json.loads((ROOT / 'configs/project.json').read_text(encoding='utf-8-sig'))
    assert cfg['bandpass_hz'] == [.01, .1] and not cfg['allow_old_results']
    assert cfg['group_prior_steps'] == 1000 and cfg['personal_steps'] == 30
    pop = json.loads((ROOT / 'configs/cohorts/population.json').read_text(encoding='utf-8-sig'))
    train = {row['subject'] for row in pop['train']}
    assert len(train) == 450 and not train.intersection(pop['reserved_retest'])
    val = json.loads((ROOT / 'configs/cohorts/validation.json').read_text(encoding='utf-8-sig'))
    assert len(val['YA400']) == 400 and len(val['retest']) == 36
    assert not train.intersection(row['subject'] for row in val['YA400'])
    assert not train.intersection(row['subject'] for row in val['retest'])

    forbidden = {'.npy', '.npz', '.nii', '.gii', '.mat', '.pt', '.pth', '.ckpt', '.h5', '.p', '.zip'}
    # Standard high-confidence token patterns; never print matched values.
    secret = re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')
    for p in files:
        assert p.suffix not in forbidden and '.mex' not in p.name, p
        assert not p.name.endswith(('.nii.gz', '.tar.gz')), p
        assert not ('author' in p.relative_to(ROOT).parts), p
        assert not secret.search(p.read_text(encoding='utf-8-sig')), 'Potential credential in ' + str(p)

    entrypoints = []
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
    output_root = Path(cfg['work_root'])
    before = output_root.exists()
    for method in METHODS:
        paths = sorted((ROOT / method).glob('s_step*.py'))
        assert len(paths) == 8, method
        assert {int(p.name[6]) for p in paths} == set(range(8)), method
        for path in paths:
            run = subprocess.run([sys.executable, str(path), '--dry-run'],
                                 cwd=ROOT, env=env, capture_output=True, text=True,
                                 encoding='utf-8', timeout=30)
            assert run.returncode == 0, (str(path), run.stderr)
            plan = json.loads(run.stdout)
            assert plan['no_historical_results'] is True
            assert plan['bandpass_hz'] == [.01, .1]
            entrypoints.append(path.relative_to(ROOT).as_posix())
    nmf = subprocess.run([sys.executable, str(ROOT / 'TraditionalMethod/s_step2_train_population.py'),
                          '--mode', 'group_nmf', '--dry-run'], cwd=ROOT, env=env,
                         capture_output=True, text=True, encoding='utf-8', timeout=30)
    assert nmf.returncode == 0, nmf.stderr
    assert json.loads(nmf.stdout)['mode'] == 'group_nmf'
    assert output_root.exists() == before, 'Dry-run changed output-root existence.'

    manifest = ROOT / 'RELEASE_MANIFEST.json'
    manifest_checked = False
    if check_manifest and manifest.exists():
        for item in json.loads(manifest.read_text(encoding='utf-8'))['files']:
            p = ROOT / item['path']
            assert p.stat().st_size == item['bytes']
            assert hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256'], item['path']
        manifest_checked = True
    return dict(passed=True, python_AST_files=len(python_files), resolved_routes=len(routes),
                dry_run_entrypoints=len(entrypoints), optional_NMF_dry_run=True,
                cohort_counts={'VAE_train': 450, 'YA_validation_preQC': 400, 'Retest_preQC': 36},
                train_validation_disjoint=True, bandpass_hz=[.01, .1],
                no_excluded_binary_inputs=True, high_confidence_secret_scan_passed=True,
                production_snapshot_naming_exceptions=naming_exceptions,
                release_manifest_checked=manifest_checked,
                limitation='Source/CLI audit only; no MRI processing or full experimental rerun performed.')


if __name__ == '__main__':
    print(json.dumps(f_audit(), indent=2, ensure_ascii=False))
