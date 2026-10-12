"""Direct full-FC projection and exact, memory-bounded all-entry losses.

Import paths are configured by s_step2_train_trial.py. The mathematical input is
the complete column-scaled Pearson FC. Matrix reassociation avoids allocating
four dense FC tensors; it does not subsample vertices, edges, or output columns.
"""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from f_full_fc_vae import FullFCVAE
from f_model_ops import f_transferred_model, f_encode_projected


def f_new_model(n, width=512, latent_dim=32, attention_layers=2):
    source = FullFCVAE(n, width, latent_dim, attention_layers)
    return f_transferred_model(source, n).requires_grad_(True)


def f_parameter_count(n, width, latent_dim=32):
    # Two graph blocks, two Transformer layers, FFN=2*width, direct FC input.
    return 2*n*width + 21*width*width + (38+3*latent_dim)*width + n+2*latent_dim+1


def f_project_signals(model, signals, scale):
    linear = model.input[0]
    weight = linear.weight / scale
    return torch.stack([y.T @ (y @ weight.T) / len(y) for y in signals]) + linear.bias


def f_encode_projected_raw(model, projected, exp, checkpoint_backbone=True):
    def f_backbone(x):
        features = model.input[2](model.input[1](x))
        return f_encode_projected(model, features, exp.graph, exp.patches, exp.pos)
    if checkpoint_backbone and torch.is_grad_enabled():
        return checkpoint(f_backbone, projected, use_reentrant=False, preserve_rng_state=True)
    return f_backbone(projected)


def f_encode_signals(model, signals, scale, exp, checkpoint_backbone=True):
    return f_encode_projected_raw(model, f_project_signals(model, signals, scale), exp, checkpoint_backbone)


def f_encode_dense(model, fc, exp, checkpoint_backbone=True):
    return f_encode_projected_raw(model, model.input[0](fc), exp, checkpoint_backbone)


def f_all_entry_backward(model, hidden, scale=None, signals=None, fc=None, cross=False, chunk=256):
    """Same ordered-pair MSE as the canonical dense implementation, including diagonal."""
    assert (fc is None) != (signals is None)
    B, N, _ = hidden.shape
    assert not cross or B > 1
    leaf = hidden.detach().requires_grad_(True)
    linear = model.decoder[2]
    total = 0.0
    for start in range(0, N, chunk):
        stop = min(start+chunk, N)
        prediction = F.linear(leaf, linear.weight[start:stop], linear.bias[start:stop])
        with torch.no_grad():
            target = fc[:, :, start:stop] if fc is not None else torch.stack([
                (y.T @ y[:, start:stop] / len(y)) / scale[start:stop] for y in signals
            ])
        for i in range(B):
            truth = target[[j for j in range(B) if j != i]] if cross else target[i:i+1]
            loss = (prediction[i:i+1]-truth).square().sum() / (B*len(truth)*N*N)
            loss.backward(retain_graph=i < B-1)
            total += float(loss.detach())
        del prediction, target, truth, loss
    assert leaf.grad is not None and torch.isfinite(leaf.grad).all()
    return total, leaf.grad.detach()


def f_test_exact_operations():
    """Verify value and every parameter gradient against explicit full-FC objectives."""
    from types import SimpleNamespace
    torch.set_num_threads(2)
    torch.manual_seed(20260930)
    n = 23
    exp = SimpleNamespace(
        graph=((torch.eye(n).roll(1, 0)+torch.eye(n).roll(-1, 0))/2).double().to_sparse(),
        patches=torch.arange(n) % 4,
        pos=torch.randn(n, 5, dtype=torch.float64),
    )
    checks = []
    for B in [1, 2, 4]:
        ys = [torch.randn(31, n, dtype=torch.float64) for _ in range(B)]
        ys = [(y-y.mean(0))/y.std(0, unbiased=False) for y in ys]
        scale = torch.rand(n, dtype=torch.float64)+0.5
        fc = torch.stack([y.T@y/len(y)/scale for y in ys])
        a = f_new_model(n, 16, 4, 2).double().eval()
        b = copy.deepcopy(a)
        direct = a.input[0](fc)
        implicit = f_project_signals(b, ys, scale)
        torch.testing.assert_close(direct, implicit, atol=1e-11, rtol=1e-9)
        ma, la = f_encode_projected_raw(a, direct, exp, False)
        mb, lb = f_encode_projected_raw(b, implicit, exp, True)
        torch.testing.assert_close(ma, mb, atol=1e-11, rtol=1e-9)
        ha = a.decoder_hidden(ma, la, sample=False)
        hb = b.decoder_hidden(mb, lb, sample=False)
        prediction = a.decoder[2](ha)
        cross = B > 1
        loss = sum((prediction[i:i+1]-(fc[[j for j in range(B) if j != i]] if cross else fc[i:i+1])).square().mean() for i in range(B))/B
        # logvar participates through KL in production, so include it here too.
        rega = 0.001*(la.exp()+ma.square()-1-la).mean()
        regb = 0.001*(lb.exp()+mb.square()-1-lb).mean()
        (loss+rega).backward()
        value, gradient = f_all_entry_backward(b, hb, scale=scale, signals=ys, cross=cross, chunk=7)
        torch.autograd.backward([hb, regb], [gradient, None])
        assert abs(value-float(loss.detach())) < 1e-11
        error = 0.0
        for (name, p), (other, q) in zip(a.named_parameters(), b.named_parameters()):
            assert name == other and p.grad is not None and q.grad is not None
            torch.testing.assert_close(p.grad, q.grad, atol=1e-10, rtol=1e-7, msg=name)
            error = max(error, float((p.grad-q.grad).abs().max()))
        checks.append(dict(runs=B, cross=cross, projection_max_error=float((direct-implicit).abs().max()), gradient_max_error=error))
    # Exercise the actual requested token count separately on a small graph.
    n = 1024
    m = f_new_model(n, 32, 8, 2).eval()
    e = SimpleNamespace(graph=torch.eye(n).to_sparse(), patches=torch.arange(n) % 512, pos=torch.randn(n, 5))
    projected = torch.randn(1, n, 32)
    mu, lv = f_encode_projected_raw(m, projected, e, True)
    (mu.square().mean()+lv.square().mean()).backward()
    assert mu.shape == (1, 1024, 8) and torch.isfinite(mu).all()
    return dict(passed=True, all_entry_loss_and_all_parameter_gradients=checks, token512_forward_backward=True)
