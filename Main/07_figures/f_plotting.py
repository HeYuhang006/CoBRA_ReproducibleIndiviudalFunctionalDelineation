from f_runtime import f_work_root
"""Actual fsLR triangle rendering and reversible display-only ID matching."""
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
TEMPLATE = Path('/home/yuhang/.cache/templateflow/tpl-fsLR')

def f_palette(k):
    rgb = matplotlib.colors.hsv_to_rgb(np.c_[np.arange(k) * 0.61803398875 % 1, np.full(k, 0.63), np.full(k, 0.92)])
    return np.r_[[[0.81, 0.83, 0.85]], rgb, rgb]

def f_export(labels, mapping, source, k, path, stem):
    axis = nib.load(source).header.get_axis(1)
    columns = np.r_[mapping['L_columns'], mapping['R_columns']]
    brain = axis[columns]
    pal = f_palette(k)
    table = {0: ('Unassigned', (0.0, 0.0, 0.0, 0.0))}
    for hi, h in enumerate(['L', 'R']):
        for a in range(k):
            table[a + 1 + hi * k] = (f'{h}_{a + 1}', tuple(pal[a + 1 + hi * k]) + (1.0,))
    values = np.r_[labels['L'] + 1, labels['R'] + 1 + k].astype('int32')
    header = nib.cifti2.Cifti2Header.from_axes((nib.cifti2.LabelAxis([stem], [table]), brain))
    img = nib.Cifti2Image(values[None], header=header)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_DENSE_LABELS')
    nib.save(img, path)
    re = nib.load(path)
    assert re.shape == (1, 59412) and int(re.nifti_header['intent_code']) == 3007
    assert np.array_equal(np.asarray(re.dataobj)[0], values) and len(np.unique(values)) == 2 * k
    for h in ['L', 'R']:
        full = np.zeros(32492, np.int32)
        full[mapping[h + '_vertices']] = labels[h] + 1
        g = nib.gifti.GiftiImage(darrays=[nib.gifti.GiftiDataArray(full, intent='NIFTI_INTENT_LABEL')])
        entries = []
        for a in range(k + 1):
            e = nib.gifti.GiftiLabel(key=a, red=float(pal[a, 0]), green=float(pal[a, 1]), blue=float(pal[a, 2]), alpha=1.0)
            e.label = 'MedialWall' if a == 0 else f'{h}_{a}'
            entries.append(e)
        g.labeltable.labels = entries
        nib.save(g, path.parent / f'{stem}_{h}.label.gii')

def f_align(reference, original, k):
    assert reference.shape == original.shape
    assert np.array_equal(np.unique(reference), np.arange(k)) and np.array_equal(np.unique(original), np.arange(k))
    overlap = np.bincount(original.astype('int64') * k + reference, minlength=k * k).reshape(k, k)
    row, col = linear_sum_assignment(-overlap)
    perm = np.empty(k, dtype='int16')
    perm[row] = col
    aligned = perm[original]
    assert np.array_equal(np.sort(perm), np.arange(k)) and adjusted_rand_score(original, aligned) == 1
    inv = np.argsort(perm)
    assert np.array_equal(inv[aligned], original)
    dice = 2 * overlap[row, col] / (np.bincount(original, minlength=k)[row] + np.bincount(reference, minlength=k)[col])
    return (aligned, dict(original_zero_based_to_reference_zero_based=perm.tolist(), overlap_fraction=float(np.mean(aligned == reference)), matched_parcel_dice=dice.tolist(), mean_matched_dice=float(dice.mean()), partition_unchanged=True))

def f_meshes(mapping):
    result = {}
    for h in ['L', 'R']:
        surf = nib.load(TEMPLATE / f'tpl-fsLR_den-32k_hemi-{h}_inflated.surf.gii')
        xyz, tri = (surf.agg_data('pointset'), surf.agg_data('triangle'))
        normals = np.cross(xyz[tri[:, 1]] - xyz[tri[:, 0]], xyz[tri[:, 2]] - xyz[tri[:, 0]])
        normals /= np.linalg.norm(normals, axis=1)[:, None].clip(1e-08)
        result[h] = (xyz, tri, normals, mapping[h + '_vertices'])
    return result

def f_draw(ax, values, h, view, k, mesh):
    xyz, tri, normals, vertices = mesh[h]
    full = np.zeros(32492, np.int32)
    full[vertices] = values + 1 + (k if h == 'R' else 0)
    v = full[tri]
    face = np.where(v[:, 0] == v[:, 1], v[:, 0], np.where(v[:, 1] == v[:, 2], v[:, 1], v[:, 0]))
    face[(v == 0).any(1)] = 0
    az = (180 if h == 'L' else 0) + (180 if view == 'medial' else 0)
    light = np.array([-1 if az % 360 == 180 else 1, -0.25, 0.45])
    light /= np.linalg.norm(light)
    colors = f_palette(k)[face] * (0.74 + 0.26 * np.abs(normals @ light))[:, None]
    ax.add_collection3d(Poly3DCollection(xyz[tri], facecolors=colors, edgecolors='none', linewidths=0, rasterized=True))
    low, high = (xyz.min(0), xyz.max(0))
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
    ax.set_box_aspect(high - low, zoom=1.12)
    ax.set_proj_type('ortho')
    ax.view_init(elev=0, azim=az)
    ax.set_axis_off()
