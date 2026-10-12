from f_runtime import f_work_root
"""Exact full-vertex FC operations without retaining paired N-by-N activations."""
import torch
from torch.nn import functional as F
from f_surface_prior_vae import SurfacePriorVAE

def f_fc_project(y, weight):
    """(Y.T @ Y / T) @ weight.T, reassociated exactly; Y is standardized T x N."""
    return y.T @ (y @ weight.T) / len(y)

class FullFCVAE(SurfacePriorVAE):

    def project_fc(self, y, weight):
        return f_fc_project(y, weight)

    def encode_full(self, signals, group, scale, adjacency, patches, positions, group_mode=False, leave_out_count=None):
        n = len(group)
        linear = self.input[0]
        wg, wi = (linear.weight[:, :n] / scale, linear.weight[:, n:] / scale)
        if group_mode:
            projected = group @ wg.T
            if leave_out_count is not None:
                projected = (leave_out_count * projected - self.project_fc(signals[0], wg)) / (leave_out_count - 1)
            projected = projected[None] + linear.bias
        else:
            common = group @ (wg - wi).T
            if leave_out_count is not None:
                common = (leave_out_count * common - self.project_fc(signals[0], wg - wi)) / (leave_out_count - 1)
            projected = common[None] + torch.stack([self.project_fc(y, wi) for y in signals]) + linear.bias
        features = self.input[2](self.input[1](projected))
        for layer in self.local_layers:
            features = layer(features, adjacency)
        count = int(patches.max()) + 1
        sizes = torch.bincount(patches, minlength=count).to(features).clamp_min(1)
        pooled = features.new_zeros(len(features), count, features.shape[-1])
        pooled.index_add_(1, patches, features)
        pooled = pooled / sizes[None, :, None]
        pos = positions.new_zeros(count, positions.shape[-1])
        pos.index_add_(0, patches, positions)
        pooled = pooled + self.position(pos / sizes[:, None])[None]
        context = self.context(pooled)
        features = features + self.context_gate.sigmoid() * self.context_output(context[:, patches])
        return (self.mu(features), self.logvar(features).clamp(-8, 5))

    def decoder_hidden(self, mu, logvar, sample=True):
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if sample else mu
        return self.decoder[1](self.decoder[0](z))

def f_reconstruction_backward(model, hidden, signals, group, scale, block_size=512, group_mode=False):
    """Accumulate exact decoder/hidden gradients; caller backprops hidden once.

    The detached hidden tensor prevents rerunning the whole graph backbone for
    each output block. All N-by-N entries contribute, including diagonal entries
    as in the unmasked original FC reconstruction objective.
    """
    leaf = hidden.detach().requires_grad_(True)
    n = len(group)
    denominator = hidden.shape[0] * n * n
    self_total, cross_total = (0.0, 0.0)
    linear = model.decoder[2]
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        prediction = F.linear(leaf, linear.weight[start:stop], linear.bias[start:stop])
        with torch.no_grad():
            if group_mode:
                target = group[None, :, start:stop] / scale[start:stop]
            else:
                target = torch.stack([y.T @ y[:, start:stop] / len(y) for y in signals])
                transform = getattr(model, 'distance_transform', None)
                if transform is not None:
                    target = transform.columns(target, start, stop)
                target = target / scale[start:stop]
        self_loss = (prediction - target).square().sum() / denominator
        if group_mode:
            loss = self_loss
            cross_loss = self_loss.detach() * 0
        else:
            cross_loss = (prediction - target.flip(0)).square().sum() / denominator
            loss = 0.5 * (self_loss + cross_loss)
        loss.backward()
        self_total += float(self_loss.detach())
        cross_total += float(cross_loss.detach())
    if leaf.grad is None or not torch.isfinite(leaf.grad).all():
        raise RuntimeError('Invalid chunked reconstruction gradient')
    return (self_total, cross_total, leaf.grad.detach())
