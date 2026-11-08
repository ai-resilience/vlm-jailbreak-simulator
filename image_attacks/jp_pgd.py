"""RGB [0,1] optimizers: upstream sequential ImgJP and reference DeltaJP.

ImgJP follows fd98910's pgd_uap_v1.py prompt order. DeltaJP follows the
reference simulator's seeded cross-class universal-delta mini-batches.
"""
import random

import torch


def _gradient(loss, variable, stats):
    if loss.ndim or not bool(torch.isfinite(loss)):
        raise RuntimeError('JP target loss must be a finite scalar')
    loss.backward()
    gradient = variable.grad
    if gradient is None or not bool(torch.isfinite(gradient).all()):
        raise RuntimeError('JP pixel gradient must exist and be finite')
    stats['losses'].append(float(loss.detach()))
    stats['pixel_gradients'].append({'finite': True, 'nonzero': bool(gradient.count_nonzero()),
                                     'mean_abs': float(gradient.detach().abs().mean())})
    return gradient


def optimize_imgjp(base, targets, loss_fn, *, epsilon, alpha, steps, seed=0,
                   random_start=False, callback=None):
    """One sign-descent update per training prompt, in order, per epoch."""
    base = base.detach().clamp(0, 1)
    delta = torch.zeros_like(base)
    if random_start:
        generator = torch.Generator(device=base.device).manual_seed(seed)
        delta.uniform_(-epsilon, epsilon, generator=generator)
        delta = (base + delta).clamp(0, 1) - base
    delta.requires_grad_(True)
    stats = {'losses': [], 'pixel_gradients': []}
    for epoch in range(steps):
        for index, target in enumerate(targets):
            loss = loss_fn(base + delta, index, target)
            gradient = _gradient(loss, delta, stats)
            with torch.no_grad():
                adv_image = base + delta - alpha * gradient.sign()
                delta.copy_((adv_image - base).clamp(-epsilon, epsilon))
                delta.copy_((base + delta).clamp(0, 1) - base)
                delta.grad.zero_()
            if callback is not None:
                callback(epoch, index, loss)
    final = (base + delta).detach().clamp(0, 1)
    stats.update(n_updates=len(stats['losses']), image_shape=list(final.shape),
                 pixel_min=float(final.min()), pixel_max=float(final.max()),
                 linf_from_base=float((final - base).abs().max()))
    return final, stats


def optimize_deltajp(units, targets, loss_fn, *, epsilon, alpha, steps,
                     batch_size, seed=0, random_start=False, callback=None):
    """Fit one bounded delta across the reference's seeded image mini-batches."""
    delta = torch.zeros_like(units[0])
    if random_start:
        generator = torch.Generator(device=delta.device).manual_seed(seed)
        delta.uniform_(-epsilon, epsilon, generator=generator)
    delta.requires_grad_(True)
    rng = random.Random(seed)
    stats = {'losses': [], 'pixel_gradients': []}
    for step in range(steps):
        indices = rng.sample(range(len(units)), min(batch_size, len(units)))
        image = (torch.cat([units[i] for i in indices], dim=0) + delta).clamp(0, 1)
        loss = loss_fn(image, indices, [targets[i] for i in indices])
        gradient = _gradient(loss, delta, stats)
        with torch.no_grad():
            delta.sub_(alpha * gradient.sign()).clamp_(-epsilon, epsilon)
            delta.grad.zero_()
        if callback is not None:
            callback(step, indices, loss)
    final = delta.detach()
    stats.update(n_updates=len(stats['losses']), image_shape=list(final.shape),
                 delta_min=float(final.min()), delta_max=float(final.max()),
                 delta_linf=float(final.abs().max()))
    return final, stats
