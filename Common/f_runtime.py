"""Shared configuration, import routing, safe output-root and manual stage CLI."""
from pathlib import Path
import argparse, hashlib, json, os, sys, time

CODE=Path(__file__).resolve().parents[1]

def f_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def f_save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False,default=str));tmp.replace(path)

def f_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()

def f_work_root():
    if 'PARCELLATION_WORKSPACE' not in os.environ:
        raise RuntimeError('Use a s_stepX_XXXX.py entry point; do not run f_ modules directly.')
    return Path(os.environ['PARCELLATION_WORKSPACE']).resolve()

def f_link(source,dest):
    source=Path(source).resolve();dest=Path(dest)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() or dest.is_symlink():
        if dest.resolve()!=source:raise RuntimeError('Unexpected existing input/code link: '+str(dest))
        return
    dest.symlink_to(source,target_is_directory=source.is_dir())

def f_activate(config,create=False):
    c=f_read(config);w=Path(c['work_root']).resolve()
    forbidden=Path('/dataspace/yuhang_data/IndividualFunctionalParcellation/GPT_workplace')
    if w==forbidden or forbidden in w.parents or w==CODE or w in CODE.parents or CODE in w.parents:
        raise RuntimeError('work_root must be a separate fresh rerun directory.')
    if c['bandpass_hz']!=[.01,.1] or c.get('allow_old_results'):
        raise RuntimeError('Require 0.01–0.1 Hz filtering and no historical result reuse.')
    if c['group_prior_steps']!=1000 or c['personal_steps']!=30 or c['seed']!=20260930 or c['hemispheres']!=['L','R']:
        raise RuntimeError('This package freezes the audited bilateral1000/30-step protocol and seed. A changed protocol requires explicit code/config revision and a new work_root.')
    os.environ.update(PARCELLATION_WORKSPACE=str(w),COBRA_GPU=str(c['gpu']),CUDA_VISIBLE_DEVICES=str(c['gpu']),
                      OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',COBRA_PROJECT=str(Path(config).resolve()))
    routes=f_read(CODE/'configs/module_routes.json')
    folders=sorted({str((CODE/v).parent) for v in routes.values()})
    sys.path[:0]=[str(CODE/'Common')]+folders
    if (CODE/'.deps').exists():sys.path.append(str(CODE/'.deps'))
    if create:
        if w.exists() and not (w/'project_definition.json').exists() and any(w.iterdir()):raise RuntimeError('work_root must be empty for the first run.')
        w.mkdir(parents=True,exist_ok=True)
        marker=w/'project_definition.json'
        signature=f_sha(config)
        code_hash=hashlib.sha256()
        for p in sorted(CODE.rglob('*')):
            if p.suffix in ['.py','.m'] and not any(v in p.parts for v in ['author','references','.deps']):
                code_hash.update(p.relative_to(CODE).as_posix().encode());code_hash.update(p.read_bytes())
        code_sha256=code_hash.hexdigest()
        if marker.exists() and (f_read(marker)['config_sha256']!=signature or f_read(marker)['code_sha256']!=code_sha256):raise RuntimeError('Project/code changed: choose a NEW work_root.')
        f_save(marker,dict(config_sha256=signature,code_sha256=code_sha256,project=c,code_root=str(CODE)))
        # Compatibility source views point only to this package, never old code/results.
        for original,target in routes.items():
            f_link(CODE/target,w/Path(original).parent/Path(target).name)
        f_link(CODE/'references',w/'references')
        f_link(CODE/'resources/atlases',w/'resources/atlases')
        f_link(CODE/'GPIP/author',w/'gpip_author_20261009/reference')
        f_link(CODE/'GPIP/compat',w/'gpip_author_20261009/compat')
        f_link(CODE/'GPIP/source_manifest.json',w/'gpip_author_20261009/source_manifest.json')
        for p in (CODE/'GPIP/05_individual').glob('*.m'):f_link(p,w/'gpip_author_20261009/code'/p.name)
        for name in ['logs','configs','manifests','runs']:(w/name).mkdir(exist_ok=True)
    return c,w

def f_import(original):
    import importlib
    routes=f_read(CODE/'configs/module_routes.json')
    m=importlib.import_module(Path(routes[original]).stem)
    if 'COBRA_PROJECT' in os.environ:
        c=f_read(os.environ['COBRA_PROJECT'])
        for attr,key in [('YA','raw_YA'),('RETEST','raw_Retest'),('TEMPLATE','template_root'),('WB','workbench')]:
            if hasattr(m,attr):setattr(m,attr,Path(c[key]))
    return m

def f_cli(method,stage):
    p=argparse.ArgumentParser(description=f'{method}: manual pipeline step {stage}. See README_运行顺序.md')
    p.add_argument('--config',default=str(CODE/'configs/project.json'))
    p.add_argument('--dry-run',action='store_true',help='Print actual inputs/outputs and dependencies; no file or GPU work')
    p.add_argument('--resume',action='store_true',help='Explicitly resume only this new work_root; never historical outputs')
    p.add_argument('--cohort',choices=['YA400','retest','all'],default='all')
    p.add_argument('--mode',choices=['default','population','personal_prior','distance','controlled','network18','task','connectome','group','group_nmf'],default='default')
    p.add_argument('--K',type=int,choices=[18,100],default=100)
    p.add_argument('--visit',choices=['Day1','Day2','Retest','all'],default='all')
    a=p.parse_args();c,w=f_activate(a.config,False)
    from f_steps import f_plan,f_execute
    plan=f_plan(method,stage,a,c,w)
    print(json.dumps(plan,indent=2,ensure_ascii=False),flush=True)
    if a.dry_run:return
    if stage==0:
        f_execute(method,stage,a,c,w);return
    c,w=f_activate(a.config,True)
    marker=w/'stage_status'/method/f'step{stage}_{a.mode}_{a.cohort}_K{a.K}_{a.visit}.json'
    if marker.exists() and not a.resume:
        raise RuntimeError('This stage was already started. Inspect its status, then explicitly use --resume or a new work_root.')
    f_save(marker,dict(state='running',plan=plan,pid=os.getpid(),started=time.time()))
    try:
        f_execute(method,stage,a,c,w)
        f_save(marker,dict(state='complete',plan=plan,finished=time.time()))
    except Exception as exc:
        f_save(marker,dict(state='failed',plan=plan,error=repr(exc),finished=time.time()));raise
