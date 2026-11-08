"""VisualAdv pixel PGD, shared by the simulator and paper reproduction worker.

The released attackers use ``range(num_iter + 1)``; ``steps`` here is the actual
number of updates, so callers explicitly choose 5000 (paper) or 5001 (release).
"""

from __future__ import annotations

import random

import torch


def optimize_pixels(base, targets, loss_fn, *, constrained, epsilon, alpha,
                    steps, batch_size=8, seed=None, callback=None):
    """Minimize target CE; unconstrained attacks start at U[0,1], not at base.

    ``callback(step, image, loss)`` runs after each update. No model weights,
    target strings, or evaluation responses are saved by this function.
    """
    if not targets or len(targets) < batch_size or batch_size < 1:
        raise ValueError("VisualAdv requires at least batch_size nonempty targets")
    if steps < 1 or alpha <= 0 or not 0 <= epsilon <= 1:
        raise ValueError("invalid PGD steps, alpha, or epsilon")
    base = base.detach().clamp(0, 1)
    rng = random if seed is None else random.Random(seed)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=base.device).manual_seed(seed)
    noise = torch.rand(base.shape, dtype=base.dtype, device=base.device,
                       generator=generator)
    if constrained:
        delta = noise * (2 * epsilon) - epsilon
        delta = (base + delta).clamp(0, 1) - base
    else:
        delta = noise  # the *whole image*, matching attack_unconstrained upstream
    delta.requires_grad_(True)
    for step in range(steps):
        image = base + delta if constrained else delta
        loss = loss_fn(image, rng.sample(targets, batch_size))
        if loss.ndim or not torch.isfinite(loss):
            raise RuntimeError("VisualAdv target loss must be a finite scalar")
        # backward also supports the legacy wrappers' reentrant checkpointing.
        loss.backward()
        gradient = delta.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError("nonfinite VisualAdv pixel gradient")
        with torch.no_grad():
            delta.sub_(alpha * gradient.sign())
            if constrained:
                delta.clamp_(-epsilon, epsilon)
                delta.copy_((base + delta).clamp(0, 1) - base)
            else:
                delta.clamp_(0, 1)
            delta.grad.zero_()
        if callback is not None:
            final = base + delta if constrained else delta
            callback(step, final.detach(), float(loss.detach()))
    return (base + delta if constrained else delta).detach().clamp(0, 1)
