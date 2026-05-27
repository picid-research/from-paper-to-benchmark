---
name: implement-loss
description: Implement a custom PICID/PICID loss class and Hydra config from a paper blueprint.
---

# /implement-loss

Create a custom loss implementation for a paper when existing losses in `picid/loss/default.py` or `picid/loss/cross_entropy.py` do not cover the objective.

## Input contract

The orchestrator must provide:

- **loss_name**: snake_case identifier, e.g. `detransformer_joint`
- **class_name**: PascalCase class name, e.g. `DeTransformerJointLoss`
- **formula**: exact loss equation from `04-implementation-blueprint.json` when structured there, or from rendered `04-implementation-blueprint.md` audit prose when the equation is too long for the compact contract
- **components**: named terms, weights, and reductions
- **required_model_out_keys**: keys the model wrapper must return, e.g. `predictions`, `targets`, `reconstructions`
- **required_batch_keys**: keys needed from the batch, e.g. `features`, `rul`
- **config_parameters**: Hydra parameters with defaults and paper values
- **reduction**: `mean`, `sum`, or `none`
- **paper_reference**: citation, paper title, or vault reference for the docstring

## Procedure

### Step 1 — Read the framework contracts

See `.opencode/reference/README.md` decision tree ("Implementing a new loss"). Also read `picid/loss/base.py`, `picid/loss/default.py`, and `configs/loss/default.yaml`. Cross-skill pitfalls (no metrics in the loss, no manual `.to(device)`, no in-place mutation) are in `.opencode/reference/style.md §7.1`.

Loss contract:
- Subclass `picid.loss.base.AbstractLoss`.
- Implement `forward(self, model_out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]`.
- Return a copy of `model_out` with at least a scalar tensor under key `"loss"`.
- Do not perform optimizer regularization in the loss if `configs/optimization/*` already sets `weight_decay`; avoid double-counting L2 terms.

### Step 2 — Verify the loss is actually novel

Before creating files, compare the requested objective against existing losses:
- `MSELoss`
- `MAELoss`
- `HuberLoss`
- `QuantileLoss`
- `MAPELoss`
- `SMAPELoss`
- `WeightedMSELoss`
- `CombinedLoss`
- `CrossEntropyLoss`

If the objective can be expressed by an existing loss config, stop and report the existing config to use. Do not create a new loss.

### Step 3 — Implement the loss file

Create `picid/loss/{loss_name}.py`.

Use this structure:

```python
from typing import Dict

import torch
import torch.nn as nn

from picid.loss.base import AbstractLoss


class {ClassName}(AbstractLoss):
    """{Short objective description}.

    {Paper reference}.

    Parameters
    ----------
    alpha :
        Weight for the auxiliary loss term.
    reduction :
        Reduction used by component losses.
    """

    def __init__(self, alpha: float = 1.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.mse = nn.MSELoss(reduction=reduction)

    def forward(
        self, model_out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        predictions = model_out["predictions"]
        targets = model_out["targets"]

        pred_loss = self.mse(predictions, targets)
        aux_loss = ...

        result = model_out.copy()
        result["loss"] = pred_loss + self.alpha * aux_loss
        result["pred_loss"] = pred_loss
        result["aux_loss"] = aux_loss
        return result
```

Rules:
- Validate required keys explicitly with clear `KeyError` messages when the loss depends on non-standard keys.
- Match tensor shapes deliberately. If an auxiliary target comes from `batch`, adapt only dimensions that are explained in the blueprint.
- Component loss values may be added to `result` for logging, but the main training loss must be `result["loss"]`.
- Use paper-provided weights exactly when available; otherwise use blueprint defaults.
- Keep the file scoped to the loss. Do not add model code, metrics, or experiment wiring here.

### Step 4 — Create config YAML

Create `configs/loss/{loss_name}.yaml`:

```yaml
_target_: picid.loss.{loss_name}.{ClassName}
reduction: mean
```

Add all config parameters from the blueprint:

```yaml
alpha: 0.5
```

### Step 5 — Validate

Run:

```bash
cd {repo_root}
uv run python -c "from picid.loss.{loss_name} import {ClassName}; print('loss import OK')"
uv run python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/loss/{loss_name}.yaml')
assert cfg._target_ == 'picid.loss.{loss_name}.{ClassName}'
print('loss config OK')
"
```

If the blueprint gives enough tensor shape information, also run a tiny synthetic forward check:

```bash
uv run python -c "
import torch
from picid.loss.{loss_name} import {ClassName}
loss = {ClassName}()
model_out = {
    'predictions': torch.randn(2, 1, 1),
    'targets': torch.randn(2, 1, 1),
}
batch = {}
out = loss(model_out, batch)
assert 'loss' in out and out['loss'].ndim == 0
print('loss forward OK', float(out['loss']))
"
```

Adapt `model_out` and `batch` keys to the blueprint if the objective needs auxiliary tensors.

### Step 6 — Report

Return:

```markdown
## Loss Implementation Complete

**Files created:**
- `picid/loss/{loss_name}.py`
- `configs/loss/{loss_name}.yaml`

**Loss terms:**
- prediction: {description}
- auxiliary: {description}

**Required model_out keys:** {keys}
**Required batch keys:** {keys}

**Validation:**
- import: pass/fail
- config parse: pass/fail
- synthetic forward: pass/fail/skipped

**Remaining wiring needed:**
- experiment config must override `/loss: {loss_name}`
- model wrapper must return required auxiliary keys
```

## Common pitfalls (loss-specific; see also `style.md §7.1`)

- Do not subclass plain `nn.Module`; use `AbstractLoss`.
- Do not read targets from `batch["targets"]` unless the blueprint says that key exists. Most wrappers already expose `model_out["targets"]`.
- Do not return only a tensor; the framework expects the full model output dict.
