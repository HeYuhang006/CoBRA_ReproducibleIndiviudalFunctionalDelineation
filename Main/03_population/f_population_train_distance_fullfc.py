from f_runtime import f_work_root
"""Distance-corrected full-FC arm, including its CPU preparation dependency."""
import os
from pathlib import Path
SOURCE_DIR = f_work_root() / 'final/population_training'
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from f_population_train_fullfc_vae import FullExperiment
from f_population_train_modulated_vae import f_json_save
from f_population_run import ROOT
from f_population_distance_fc_vae import DistanceFCVAE, DistanceNormalization

class DistanceExperiment(FullExperiment):

    def __init__(self, config_path):
        super().__init__(config_path)
        if not self.c['distance_residualization']:
            raise RuntimeError('Corrected arm requires distance_residualization=true')
        self.normalization_folder = ROOT / 'resources/fullfc_distance' / self.c['name']
        expected = [r['subject'] for r in self.subjects['train']]
        definition = json.loads((self.normalization_folder / 'definition.json').read_text())
        if definition['subjects'] != expected:
            raise RuntimeError('Distance cohort mismatch')
        hashes = {name: hashlib.sha256((SOURCE_DIR / name).read_bytes()).hexdigest() for name in ['f_population_distance_fc_vae.py', 'f_population_prepare_fullfc_distance.py', 'f_population_train_distance_fullfc.py']}
        saved = self.folder / 'distance_code_hashes.json'
        if saved.exists() and json.loads(saved.read_text()) != hashes:
            raise RuntimeError('Distance code changed')
        f_json_save(saved, hashes)
        self.active_transform = None
        self.active_hemisphere = None

    def transform_for(self, h):
        if self.active_hemisphere != h:
            self.active_transform = None
            torch.cuda.empty_cache()
            stats = np.load(self.normalization_folder / f'statistics_{h}.npz')
            bins = torch.from_numpy(np.load(self.normalization_folder / f'bins_{h}.npy')).cuda()
            means = torch.tensor(np.r_[stats['means'], stats['global_mean']], dtype=torch.float32, device='cuda')
            scales = torch.tensor(np.r_[stats['scales'], stats['global_std']], dtype=torch.float32, device='cuda')
            self.active_transform = DistanceNormalization(bins, means, scales, self.c['normalization_projection_block'])
            self.active_hemisphere = h
        return self.active_transform

    def model(self, h):
        torch.manual_seed(self.c['seed'])
        model = DistanceFCVAE(len(self.mapping[h]['vertices']), self.c['width'], self.c['latent_dim'], self.c['attention_layers']).cuda()
        model.distance_transform = self.transform_for(h)
        return model
