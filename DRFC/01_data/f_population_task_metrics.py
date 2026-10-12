from f_runtime import f_work_root
"""Evaluation-only HCP group task maps, aligned by cortical vertex IDs."""
import os
from pathlib import Path
SOURCE_DIR = f_work_root() / 'final/population_training'
import json
from pathlib import Path
import nibabel as nib
import numpy as np
ROOT = Path(os.environ['PARCELLATION_WORKSPACE']).resolve()

def f_task_profiles(hemi, vertices):
    manifest = ROOT / 'manifests/task_maps.json'
    if not manifest.exists():
        return None
    metadata = json.loads(manifest.read_text())
    image = nib.load(metadata['path'])
    if image.shape[0] != 86:
        raise ValueError('Expected 86 Cohen d task contrasts')
    axis = image.header.get_axis(1)
    name = 'CIFTI_STRUCTURE_CORTEX_LEFT' if hemi == 'L' else 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    indices = np.flatnonzero(axis.surface_mask & (axis.name == name))
    full = np.full((32492, 86), np.nan, np.float32)
    full[axis.vertex[indices]] = np.asarray(image.dataobj)[:, indices].T
    values = full[vertices]
    if not np.isfinite(values).all():
        raise ValueError('Task maps do not cover the requested native cortical vertex IDs')
    return values

def f_task_inhomogeneity(labels, values):
    if len(labels) != len(values) or values.ndim != 2:
        raise ValueError('Task profiles and labels are not aligned')
    variances, counts = ([], [])
    for label in np.unique(labels):
        rows = values[labels == label]
        variances.append(rows.astype(np.float64).var(0, ddof=0).mean())
        counts.append(len(rows))
    return {'task_inhomogeneity_raw': float(np.mean(variances)), 'task_inhomogeneity_size_weighted_raw': float(np.average(variances, weights=counts)), 'task_contrasts': values.shape[1], 'task_aggregation': 'population within-parcel variance, mean over contrasts and parcels; size-weighted alternative also retained', 'task_weighting_specified_by_paper': False}
