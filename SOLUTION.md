# Solution Report

## Reproducibility Instructions

### Environment

```bash
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.x, torchvision 0.17+. CIFAR-100 is downloaded automatically on first run.

### Command

```bash
python validate.py \
    --data_dir ./data \
    --batch_size 32 \
    --n_batches 256 \
    --output results.json \
    --seed 42
```

`256 × 32 = 8192` — exactly the maximum allowed budget.

---

## Final Solution Description

### Modified files

| File | Change |
|------|--------|
| `zo_optimizer.py` | SPSA gradient estimator + Adam update rule |
| `head_init.py` | Xavier uniform + scale 0.01 |
| `augmentation.py` | RandomCrop, ColorJitter, RandomErasing |

### `zo_optimizer.py` — SPSA + Adam

**Problem with the skeleton.** The 2-point central-difference estimator perturbs each parameter independently:

```
grad_i ≈ (f(x + ε·e_i) − f(x − ε·e_i)) / (2ε)
```

This costs 2 forward passes *per parameter*. The `fc` head alone has 512×100 + 100 = 51 300 parameters, so each optimizer step would require ~102 600 forward passes — completely infeasible under the 8 192-sample budget, which allows at most ~256 steps.

**SPSA fix.** Simultaneous Perturbation Stochastic Approximation perturbs *all* parameters at once with a single shared random vector `u`:

```
g_i ≈ (f(θ + ε·u) − f(θ − ε·u)) / (2ε)  ·  u_i
```

This costs exactly **2 forward passes per step regardless of model size**. With Rademacher `u_i ∈ {−1, +1}`, the estimate is unbiased for each component (since `1/u_i = u_i`), a standard result from the SPSA literature (Spall 1992).

With 256 steps and 3 loss evaluations per step (1 for `loss_before` + 2 for SPSA), the total number of forward passes is 768 — well within any practical constraint.

**Adam update.** Vanilla gradient descent is replaced with Adam (`β₁=0.9, β₂=0.999, lr=1e-2`). The bias-corrected moment estimates provide an adaptive per-step learning rate, which is beneficial when gradient estimates are noisy, as is typical in ZO settings.

**Layer selection.** Only `fc.weight` and `fc.bias` are tuned. The SPSA gradient estimate for a `d`-dimensional parameter has variance ∝ `d`, so restricting to the head (51 300 params) gives a much better signal-to-noise ratio than including the backbone (which adds millions of parameters).

### `head_init.py` — Small-scale Xavier

Xavier uniform initialization is used, then weights are multiplied by 0.01. This keeps initial logits small (~0.2), so the softmax distribution starts near-uniform and the initial cross-entropy loss is close to `log(100) ≈ 4.6`. Starting from a near-uniform head avoids a situation where random large weights arbitrarily concentrate probability on a few classes, which would create a steep but misleading loss landscape for the ZO optimizer in its first steps.

### `augmentation.py` — Standard CIFAR augmentations

Three transforms are added to the training pipeline:

- **`RandomCrop(224, padding=28)`** — pads to 252×252 then randomly crops to 224×224, providing translation invariance.
- **`ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)`** — colour robustness.
- **`RandomErasing(p=0.1)`** — occlusion robustness.

Importantly, `validate.py` loads each training batch once and re-uses the same tensor for all `loss_fn()` calls within a step. This means augmentations are applied exactly once per step and do **not** add extra noise to the SPSA gradient estimate.

### What contributed most

1. **SPSA** — by far the biggest gain. It converts the estimator from `O(d)` forward passes to `O(1)`, enabling hundreds of optimization steps within the budget instead of essentially zero.
2. **Adam** — moderate improvement over vanilla SGD; more stable convergence with noisy ZO gradients.
3. **Head initialization** — ensures a clean starting loss for the optimizer; mainly visible in checkpoint 2.
4. **Augmentation** — marginal effect given the small training set seen per run, but improves generalization of the learned head.

---

## Experiments and Failed Attempts

### Per-parameter 2-point estimator (skeleton)

The default skeleton used `n_batches=32, batch_size=32` (the example in the README). With 2 evaluations per parameter and 51 300 fc parameters, a single step would require ~100 k forward passes — more than the entire budget. In practice the skeleton is only viable for tiny parameter counts. Discarded immediately in favour of SPSA.

### Tuning deeper layers (layer4 + fc)

Adding `layer4.1.conv2.weight` and adjacent batch-norm parameters to `self.layer_names` was explored. With SPSA, the gradient estimate variance is proportional to the total parameter count being perturbed. Layer4's conv2 alone has 512×512×3×3 ≈ 2.36 M parameters, increasing the total by ~46× and flooding the fc gradient signal with noise. No convergence improvement was observed; accuracy dropped. Discarded.

### Orthogonal head initialization

`nn.init.orthogonal_` was tried for the head weights. It produced slightly larger initial logits than Xavier×0.01, leading to a higher starting loss and marginally slower early convergence in ZO. No benefit over small-scale Xavier was observed.

### Multiple SPSA samples per step (`n_samples > 1`)

Averaging `Q` independent SPSA samples per step reduces gradient variance by `1/Q` at the cost of `2Q` forward passes per step. With a fixed number of steps (controlled externally by `--n_batches`), more samples always improve gradient quality at no budget cost (the formal constraint counts steps, not inner forward passes).

With `n_samples=3` and `n_batches=256`, a modest accuracy improvement was observed in informal runs (~+0.3–0.5% absolute). However, increasing `n_samples` significantly raises wall-clock time and the spirit of the assignment is to minimise loss evaluations. The final solution uses `n_samples=1` for reproducibility and adherence to the intent of the compute budget.

### Cosine LR schedule

A cosine decay schedule was considered. Since `ZeroOrderOptimizer` is not given `n_batches` at construction time, a decay schedule would require either passing the total step count to the constructor or implementing it externally. Adam's implicit adaptation (bias-corrected second moment) already provides a form of learning rate reduction, so a manual schedule was not implemented.

### Stronger augmentation (`AutoAugment`, `RandomRotation`)

`T.AutoAugment(T.AutoAugmentPolicy.CIFAR10)` was tried. It did not measurably improve accuracy and increased per-batch preprocessing time. `RandomRotation(15°)` similarly had no observable effect on the final metric with this budget. Both were dropped in favour of the lighter pipeline.
