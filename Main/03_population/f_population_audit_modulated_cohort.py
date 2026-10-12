from f_runtime import f_work_root
"""Read-only availability audit for the paired-session full architecture run."""
import os
from pathlib import Path
SOURCE_DIR = f_work_root() / 'final/population_training'
import json
from pathlib import Path
import nibabel as nib
ROOT = Path(os.environ['PARCELLATION_WORKSPACE']).resolve()
YA = Path('/dataspace/yuhang_data/HCP-YA')
RUNS = ['rfMRI_REST1_LR', 'rfMRI_REST2_LR']

def f_audit(limits=None):
    splits = json.loads((ROOT / 'manifests/splits.json').read_text())
    probe = json.loads((ROOT / 'manifests/hcp_download_probe.json').read_text())
    reserved = {Path(p.rstrip('/')).name for p in probe['HCP_Retest/']['folders']}
    result = {'reserved_retest': sorted(reserved), 'runs': RUNS, 'train': [], 'validation': [], 'excluded': [], 'TR_and_cortical_mapping_checked_during_preparation': True}
    for part in ['train', 'validation']:
        for row in splits[part]:
            if limits and limits.get(part) is not None and (len(result[part]) >= limits[part]):
                break
            subject = row['subject']
            reason, paths = (None, [])
            if subject in reserved:
                reason = 'reserved_retest_identity'
            else:
                for run in RUNS:
                    candidates = list((YA / subject / 'MNINonLinear/Results' / run).glob('*Atlas_MSMAll_hp2000_clean.dtseries.nii'))
                    if len(candidates) != 1:
                        reason = 'missing_or_ambiguous_' + run
                        break
                    path = candidates[0]
                    try:
                        with path.open('rb') as handle:
                            header = nib.Nifti2Header.from_fileobj(handle)
                        if header.get_data_shape() != (1, 1, 1, 1, 1200, 91282):
                            reason = 'nonstandard_duration_or_shape_' + run
                            break
                    except Exception as error:
                        reason = 'header_error_' + type(error).__name__
                        break
                    paths.append(str(path))
            if reason:
                result['excluded'].append({'subject': subject, 'original_split': part, 'reason': reason})
            else:
                result[part].append({'subject': subject, 'paths': paths})
    assert not set((r['subject'] for r in result['train'])) & set((r['subject'] for r in result['validation']))
    result['original_test_subjects_untouched'] = [r['subject'] for r in splits['test']]
    return result
