---
name: implement-model
description: Implement a new model backbone + wrapper + config for the PICID/PICID framework, given an architecture specification from a paper blueprint.
---

# /implement-model

Create a new model module in the PICID framework: the `nn.Module` backbone, the wrapper class, and the Hydra config YAML(s).

## Input contract

The orchestrator must provide (in the prompt or as structured context):

- **model_name**: snake_case identifier (e.g., `temporal_fusion_transformer`)
- **architecture_description**: What the model does, layer structure, forward pass logic
- **wrapper_type**: One of `FeedForwardTraining`, `FeedForward` (constant loss / no training), or `FitPredict` (sklearn-style)
- **task_types**: Which tasks it supports (subset of: `regression`, `rul`, `ahrul`, `soc`, `classification`, `health_states`, `concepts`, `fault_classification`, `anomaly_detection`, `forecasting`, `state_forecasting`)
- **input_format**: What batch keys the model reads (typically `features`, possibly `time_features`, `target` for autoregressive)
- **model_io_contract**: Batch input keys, target keys, prediction shape, target shape, and whether the model is supervised (`features -> target`) or reconstruction (`features -> features`)
- **hyperparameters**: Dict of model hyperparams with defaults
- **paper_reference**: Citation or URL for the docstring
- **loss_function**: Which existing loss to use, or `"custom"` if a new one is needed (handle separately via implement-loss or inline if trivial)

## Procedure

### Step 1 — Read the framework contracts

See `.opencode/reference/README.md` decision tree ("Implementing a new model"). `style.md §7` covers what NOT to implement; `§7.1` covers cross-skill pitfalls (no device strings, no loss/metric in the model, no inverse-transform in the model).

Internalize:
- Backbone is a plain `nn.Module` in `picid/model/methods/{model_name}.py`
- Wrapper is in `picid/model/wrappers/{model_name}_wrapper.py`
- Wrapper `forward(batch)` receives a dict and must return `{"predictions": ..., "targets": ...}`
- Predictions shape: `(B, 1, num_targets)` for regression; `(B, 1, num_classes)` for classification
- Config in `configs/model/{model_name}.yaml` with `_target_` pointing to the wrapper

### Step 2 — Read existing reference implementations

Read the simplest existing model that matches the wrapper_type:

- **FeedForwardTraining**: Read `picid/model/methods/mlp.py` and `picid/model/wrappers/mlp_wrapper.py`
- **FeedForward (constant loss)**: Read `picid/model/methods/naive_model.py` and `picid/model/wrappers/naive_model_wrapper.py`
- **FitPredict**: Read `picid/model/wrappers/fit_predict_xgboost_wrapper.py`

Also read `picid/baselines/definitions.py` for task type constants.

### Step 3 — Implement the backbone

Create `picid/model/methods/{model_name}.py`:

```python
import torch
import torch.nn as nn
from typing import Dict, Any


class {ModelClassName}(nn.Module):
    """
    {One-line description}.

    {Paper reference}.

    Parameters
    ----------
    config : Dict[str, Any]
        Must contain {list required keys}.
    task_type : str
        'regression' or 'classification'.
    num_targets : int
        Output dimension.
    """

    def __init__(self, config: Dict[str, Any], task_type: str, num_targets: int):
        super().__init__()
        # Extract config, build layers
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape depends on model — document it
        # Return shape: (B, num_targets) or (B, pred_len, num_targets)
        ...
```

Rules:
- Keep the backbone model-only — no loss computation, no metric calculation, no batch unpacking
- Accept a `config` dict (or explicit kwargs) — Hydra will pass constructor args
- Document input/output tensor shapes in the docstring
- Use `nn.Module` conventions: all layers as attributes, `forward()` is the only public method
- Initialize weights if the paper specifies an init scheme

### Step 4 — Implement the wrapper

Create `picid/model/wrappers/{model_name}_wrapper.py`:

```python
import torch
from typing import Dict, Any, Optional
from einops import rearrange

from picid.baselines.definitions import REGRESSION_TASKS, CLASSIFICATION_TASKS
from picid.model.wrappers.base import AbstractFeedForwardTrainingWrapper  # or other base
from picid.model.methods.{model_name} import {ModelClassName}


class {ModelClassName}Wrapper(AbstractFeedForwardTrainingWrapper):
    def __init__(self, task_type: str, seq_len: int, *,
                 input_channels: int, num_targets: Optional[int] = None,
                 num_classes: Optional[int] = None,
                 # ... model-specific hyperparams with defaults ...
                 **kwargs):
        # Determine output dimension from task_type
        # Build config dict
        # Instantiate backbone
        backbone = {ModelClassName}(config, task_type=..., num_targets=...)
        super().__init__(backbone=backbone, **kwargs)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 1. Extract inputs from batch dict
        # 2. Call self.backbone(...)
        # 3. Reshape outputs to (B, 1, num_targets) or (B, pred_len, num_targets)
        # 4. Return {"predictions": ..., "targets": ...}
        ...
```

Rules:
- The wrapper's `__init__` signature becomes the Hydra config schema — every param maps to a YAML key
- Use `${task_definition.task_type}`, `${task_definition.seq_len}` etc. in config for standard params
- Use `${infer_data_dim:features,-1}` for `input_channels` and `${infer_data_dim:${task_definition.task_type},-1}` for `num_targets` — these are resolved at runtime after data loading
- Handle both regression and classification task types if the architecture supports both
- The `forward()` method must handle the `batch["features"]` permutation if needed (framework convention is `(B, T, C)` from the dataloader)
- For reconstruction/autoencoder models, return targets from the same feature tensor being reconstructed; do not assume an RUL/context target unless `model_io_contract` says so.

### Step 5 — Create config YAML

Create `configs/model/{model_name}.yaml`:

```yaml
_target_: picid.model.wrappers.{model_name}_wrapper.{ModelClassName}Wrapper
task_type: ${task_definition.task_type}

seq_len: ${task_definition.seq_len}
label_len: ${task_definition.label_len}
pred_len: ${task_definition.pred_len}

input_channels: null    # resolved by model_configs via infer_data_dim
num_targets: null       # resolved by model_configs via infer_data_dim

# Model-specific hyperparameters with defaults
{hyperparam_name}: {default_value}
```

### Step 6 — Validate

Run these checks:

```bash
# 1. Import check
cd {project_root}
uv run python -c "from picid.model.methods.{model_name} import {ModelClassName}; print('backbone OK')"
uv run python -c "from picid.model.wrappers.{model_name}_wrapper import {ModelClassName}Wrapper; print('wrapper OK')"

# 2. Config parse check
uv run python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/model/{model_name}.yaml')
assert '_target_' in cfg
print('config OK:', cfg._target_)
"
```

If import fails, fix the error. Do not proceed to the next skill until validation passes.

### Step 7 — Report

Output a structured summary:

```
## Model Implementation Complete

**Files created:**
- `picid/model/methods/{model_name}.py` — {ModelClassName} backbone
- `picid/model/wrappers/{model_name}_wrapper.py` — {ModelClassName}Wrapper
- `configs/model/{model_name}.yaml` — Hydra config

**Wrapper type:** {FeedForwardTraining | FeedForward | FitPredict}
**Supported tasks:** {list}
**Hyperparameters:** {list with defaults}

**Remaining wiring needed:**
- model_configs YAML (handled by implement-experiment)
- experiment YAML (handled by implement-experiment)
```

## Common pitfalls (model-specific; see also `style.md §7.1`)

- Do NOT compute loss in the wrapper — that's done by `TrainingLightningModule`
- Do NOT hardcode batch keys — read from `self.task_type` for the target key
- Predictions and targets must be **numpy-convertible tensors** with shape `(B, Seq, Features)` where Seq is typically 1 for regression
- For FitPredict wrappers: `fit(X, y)` receives 2D tensors on CPU; `predict(X)` must return 2D tensor
