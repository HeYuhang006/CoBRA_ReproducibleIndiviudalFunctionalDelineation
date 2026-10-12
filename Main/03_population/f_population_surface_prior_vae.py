from f_runtime import f_work_root
"""Surface-convolution and patch-attention VAE with group-conditioned inputs.

Convolution uses true mesh adjacency. FC feature columns are not treated as a
regular spatial image. MLP, graph, and graph-transformer controls share a class.
"""
import os
from pathlib import Path
SOURCE_DIR = f_work_root() / 'final/population_training'
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

def f_neighbor_mean(adjacency, features):
    batch, vertices, channels = features.shape
    flat = features.permute(1, 0, 2).reshape(vertices, batch * channels)
    return torch.sparse.mm(adjacency, flat).reshape(vertices, batch, channels).permute(1, 0, 2)

class SurfaceConvolution(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.self_filter = nn.Linear(width, width)
        self.neighbor_filter = nn.Linear(width, width, bias=False)
        self.norm = nn.LayerNorm(width)

    def forward(self, features, adjacency, use_neighbors=True):
        update = self.self_filter(features)
        if use_neighbors:
            update = update + self.neighbor_filter(f_neighbor_mean(adjacency, features))
        return self.norm(features + F.gelu(update))

class SurfacePriorVAE(nn.Module):

    def __init__(self, n_features, width=64, latent_dim=32, attention_layers=2, architecture='graph_transformer'):
        super().__init__()
        if architecture not in ['mlp', 'graph', 'graph_transformer']:
            raise ValueError('Unsupported architecture')
        self.architecture = architecture
        self.input = nn.Sequential(nn.Linear(2 * n_features, width), nn.LayerNorm(width), nn.GELU())
        self.local_layers = nn.ModuleList([SurfaceConvolution(width) for _ in range(2)])
        self.position = nn.Linear(5, width, bias=False)
        layer = nn.TransformerEncoderLayer(width, nhead=4, dim_feedforward=2 * width, dropout=0.1, activation='gelu', batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, num_layers=attention_layers, enable_nested_tensor=False)
        self.context_output = nn.Linear(width, width)
        self.context_gate = nn.Parameter(torch.tensor(-2.0))
        self.mu = nn.Linear(width, latent_dim)
        self.logvar = nn.Linear(width, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, width), nn.GELU(), nn.Linear(width, n_features))

    def encode(self, individual, group, scale, adjacency, patches, positions):
        if individual.ndim == 2:
            individual = individual[None]
        if group.ndim == 2:
            group = group[None].expand(len(individual), -1, -1)
        features = self.input(torch.cat([group / scale, (individual - group) / scale], dim=-1))
        for layer in self.local_layers:
            features = layer(features, adjacency, use_neighbors=self.architecture != 'mlp')
        if self.architecture == 'graph_transformer':
            patch_count = int(patches.max()) + 1
            counts = torch.bincount(patches, minlength=patch_count).to(features).clamp_min(1)
            pooled = features.new_zeros(len(features), patch_count, features.shape[-1])
            pooled.index_add_(1, patches, features)
            pooled = pooled / counts[None, :, None]
            pos = positions.new_zeros(patch_count, positions.shape[-1])
            pos.index_add_(0, patches, positions)
            pooled = pooled + self.position(pos / counts[:, None])[None]
            contextual = self.context(pooled)
            features = features + self.context_gate.sigmoid() * self.context_output(contextual[:, patches])
        return (self.mu(features), self.logvar(features).clamp(-8, 5))

    def forward(self, individual, group, scale, adjacency, patches, positions, sample=True):
        mu, logvar = self.encode(individual, group, scale, adjacency, patches, positions)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        return {'mu': mu, 'logvar': logvar, 'prediction': self.decoder(z) * scale}

def f_diagonal_kl_to_prior(mu, logvar, prior_mu, prior_variance=1.0):
    """Fixed-width group-centered prior; width is a hyperparameter, not calibrated uncertainty."""
    if prior_variance <= 0:
        raise ValueError('Prior variance must be positive')
    return 0.5 * ((logvar.exp() + (mu - prior_mu).square()) / prior_variance - 1 + np.log(prior_variance) - logvar).mean()

def f_leave_subject_out_reference(subject_means, index):
    if len(subject_means) < 2:
        raise ValueError('At least two training subjects are required')
    return (subject_means.sum(0) - subject_means[index]) / (len(subject_means) - 1)

def f_soft_homogeneity(standardized_timecourses, probabilities):
    """Exact size-weighted off-diagonal Pearson homogeneity without a dense FC.

    Input T by N time courses must be centered with unit population variance.
    The expression also supports differentiable soft parcel assignments.
    """
    mass = probabilities.sum(0)
    squared_mass = probabilities.square().sum(0)
    parcel_signals = standardized_timecourses @ probabilities
    diagonal = standardized_timecourses.square().mean(0)
    numerator = parcel_signals.square().mean(0) - (probabilities.square() * diagonal[:, None]).sum(0)
    denominator = mass.square() - squared_mass
    valid = denominator > 1e-06
    per_parcel = torch.where(valid, numerator / denominator.clamp_min(1e-06), torch.zeros_like(mass))
    weights = mass * valid
    return (weights * per_parcel).sum() / weights.sum().clamp_min(1.0)

def f_prototype_probabilities(mu, prototypes, positions, prototype_positions, temperature=0.2, spatial_weight=0.01):
    feature_distance = (mu.square().mean(-1, keepdim=True) + prototypes.square().mean(-1) - 2 * (mu @ prototypes.T) / mu.shape[-1]).clamp_min(0)
    position_distance = (positions.square().mean(-1, keepdim=True) + prototype_positions.square().mean(-1) - 2 * (positions @ prototype_positions.T) / positions.shape[-1]).clamp_min(0)
    logits = -(feature_distance + spatial_weight * position_distance) / temperature
    return logits.softmax(-1)
