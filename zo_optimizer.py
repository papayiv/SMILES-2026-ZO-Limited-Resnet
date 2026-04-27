"""
zo_optimizer.py — Zero-order optimizer (SPSA + Adam).

Key idea: replace the per-parameter 2-point central-difference estimator
(2 forward passes *per parameter*) with SPSA — Simultaneous Perturbation
Stochastic Approximation — which uses exactly 2 forward passes *total*,
regardless of the number of parameters, by perturbing them all at once with
a shared random direction.

Update rule: Adam (first + second moment estimates) for adaptive step sizes.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class ZeroOrderOptimizer:
    """Gradient-free optimizer using SPSA with Adam updates.

    SPSA gradient estimate (Rademacher perturbation):
        u_i ~ Rademacher(±1),  shared across all active parameters
        g_i ≈ (f(θ + ε·u) - f(θ - ε·u)) / (2ε)  ·  u_i

    This is an unbiased estimator of ∂f/∂θ_i (first-order approx.) and
    requires only 2 loss evaluations per step, independent of model size.

    Args:
        model:             The nn.Module to optimize.
        lr:                Adam learning rate.
        eps:               SPSA perturbation magnitude.
        perturbation_mode: "rademacher" (default, optimal for SPSA),
                           "gaussian" (N(0,I)), or "uniform" (U(-1,1)).
        beta1:             Adam first-moment decay.
        beta2:             Adam second-moment decay.
        eps_adam:          Adam numerical stability constant.
        n_samples:         Number of SPSA samples averaged per step
                           (uses 2·n_samples forward passes in _estimate_grad).
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-2,
        eps: float = 1e-3,
        perturbation_mode: str = "rademacher",
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps_adam: float = 1e-8,
        n_samples: int = 1,
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps

        if perturbation_mode not in ("gaussian", "uniform", "rademacher"):
            raise ValueError(
                f"perturbation_mode must be 'gaussian', 'uniform', or 'rademacher', "
                f"got '{perturbation_mode}'"
            )
        self.perturbation_mode = perturbation_mode
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps_adam = eps_adam
        self.n_samples = n_samples

        # Active parameters — only the classification head.
        # SPSA allows us to tune all head parameters efficiently with 2 passes.
        self.layer_names: list[str] = ["fc.weight", "fc.bias"]

        # Adam state (lazily initialised on first update, matches parameter device)
        self._m: dict[str, torch.Tensor] = {}
        self._v: dict[str, torch.Tensor] = {}
        self._t: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _active_params(self) -> dict[str, nn.Parameter]:
        """Return name→parameter mapping for all currently active layers."""
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(
                f"The following layer names were not found in the model: "
                f"{missing}. Use [n for n, _ in model.named_parameters()] "
                f"to inspect valid names."
            )
        return {n: named[n] for n in self.layer_names}

    def _sample_direction(self, param: torch.Tensor) -> torch.Tensor:
        """Sample a perturbation direction matching param's shape and device."""
        if self.perturbation_mode == "rademacher":
            # Rademacher ±1: each u_i^{-1} = u_i, giving unbiased per-component estimates
            u = (
                torch.randint(0, 2, param.shape, dtype=param.dtype, device=param.device)
                * 2.0
                - 1.0
            )
        elif self.perturbation_mode == "gaussian":
            u = torch.randn_like(param)
        else:  # uniform
            u = torch.rand_like(param) * 2.0 - 1.0
        return u

    def _estimate_grad(
        self,
        loss_fn: Callable[[], float],
        params: dict[str, nn.Parameter],
    ) -> dict[str, torch.Tensor]:
        """SPSA gradient estimator — 2 forward passes per sample.

        Perturbs ALL active parameters simultaneously with a single shared
        random direction u. The central-difference along u gives a scalar δ
        that approximates the directional derivative ∇f · u.  Multiplying
        back by u_i recovers an unbiased estimate of ∂f/∂θ_i (for Rademacher).

        Uses 2·n_samples total loss evaluations (vs. 2·|params| for the
        per-parameter skeleton).
        """
        grads: dict[str, torch.Tensor] = {
            name: torch.zeros_like(param.data) for name, param in params.items()
        }

        with torch.no_grad():
            for _ in range(self.n_samples):
                directions = {
                    name: self._sample_direction(param)
                    for name, param in params.items()
                }

                # θ + ε·u
                for name, param in params.items():
                    param.data.add_(self.eps * directions[name])
                f_plus = loss_fn()

                # θ - ε·u
                for name, param in params.items():
                    param.data.sub_(2.0 * self.eps * directions[name])
                f_minus = loss_fn()

                # Restore θ
                for name, param in params.items():
                    param.data.add_(self.eps * directions[name])

                # SPSA estimate: δ · u_i  (for Rademacher: 1/u_i = u_i)
                delta = (f_plus - f_minus) / (2.0 * self.eps)
                for name, u in directions.items():
                    grads[name].add_(delta * u)

        if self.n_samples > 1:
            for name in grads:
                grads[name].div_(float(self.n_samples))

        return grads

    def _update_params(
        self,
        params: dict[str, nn.Parameter],
        grads: dict[str, torch.Tensor],
    ) -> None:
        """Adam update: adaptive per-step learning rate via moment estimates."""
        self._t += 1
        bc1 = 1.0 - self.beta1 ** self._t
        bc2 = 1.0 - self.beta2 ** self._t

        with torch.no_grad():
            for name, param in params.items():
                g = grads[name]

                if name not in self._m:
                    self._m[name] = torch.zeros_like(param.data)
                    self._v[name] = torch.zeros_like(param.data)

                self._m[name].mul_(self.beta1).add_((1.0 - self.beta1) * g)
                self._v[name].mul_(self.beta2).add_((1.0 - self.beta2) * g * g)

                m_hat = self._m[name] / bc1
                v_hat = self._v[name] / bc2

                param.data.sub_(self.lr * m_hat / (v_hat.sqrt() + self.eps_adam))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, loss_fn: Callable[[], float]) -> float:
        """One zero-order optimisation step (SPSA + Adam).

        Args:
            loss_fn: Callable returning a scalar loss on the current batch.
                     Called 1 (for loss_before) + 2·n_samples times per step.

        Returns:
            Loss at the start of the step (before any parameter update).
        """
        params = self._active_params()

        with torch.no_grad():
            loss_before = loss_fn()

        grads = self._estimate_grad(loss_fn, params)
        self._update_params(params, grads)

        return float(loss_before)
