from f_runtime import f_work_root
"""Fixed training-group distance-bin z scores and exact blockwise projections."""
import torch
from f_full_fc_vae import FullFCVAE

class NormalizedProjection(torch.autograd.Function):

    @staticmethod
    def forward(ctx, y, weight, bins, means, scales, block_size):
        if y.requires_grad:
            raise ValueError('This operator expects fixed preprocessed timecourses')
        ctx.save_for_backward(y, bins, means, scales)
        ctx.block_size = block_size
        ctx.weight_shape = weight.shape
        n = y.shape[1]
        out = y.new_empty(n, weight.shape[0])
        for a in range(0, n, block_size):
            b = min(a + block_size, n)
            index = bins[a:b].long()
            values = (y[:, a:b].T @ y / len(y) - means[index]) / scales[index]
            out[a:b] = values @ weight.T
        return out

    @staticmethod
    def backward(ctx, grad_output):
        y, bins, means, scales = ctx.saved_tensors
        grad_weight = y.new_zeros(ctx.weight_shape)
        for a in range(0, y.shape[1], ctx.block_size):
            b = min(a + ctx.block_size, y.shape[1])
            index = bins[a:b].long()
            values = (y[:, a:b].T @ y / len(y) - means[index]) / scales[index]
            grad_weight.addmm_(grad_output[a:b].T, values)
        return (None, grad_weight, None, None, None, None)

class DistanceNormalization:

    def __init__(self, bins, means, scales, block_size=256):
        self.bins, self.means, self.scales = (bins, means, scales)
        self.block_size = block_size

    def __deepcopy__(self, memo):
        return self

    def columns(self, values, start, stop):
        index = self.bins[:, start:stop].long()
        return (values - self.means[index]) / self.scales[index]

    def project(self, y, weight):
        return NormalizedProjection.apply(y, weight, self.bins, self.means, self.scales, self.block_size)

class DistanceFCVAE(FullFCVAE):

    def project_fc(self, y, weight):
        return self.distance_transform.project(y, weight)
