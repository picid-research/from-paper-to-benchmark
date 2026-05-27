---
name: implement-transform
description: Implement a new data transform for the PICID/PICID framework, given a preprocessing specification from a paper blueprint.
---

# /implement-transform

Create a new transform class and wire it into a transform pipeline config.

## Input contract

The orchestrator must provide:

- **transform_name**: snake_case identifier (e.g., `wavelet_decomposition`)
- **description**: What the transform does mathematically/logically
- **is_stateful**: Whether it needs fitting on train data (e.g., scaler=yes, FFT=no)
- **supports_ragged**: Whether it handles variable-length sequences (Awkward Arrays)
- **supports_inverse**: Whether an inverse operation exists (needed if evaluator must undo scaling)
- **input_keys**: Which data keys it reads (e.g., `["features"]`, `["features", "descriptors"]`)
- **output_key**: What key it writes to (same as input, or a new key like `"spectral_features"`)
- **parameters**: Constructor hyperparameters with defaults
- **target_pipeline**: Which existing transform config to add it to (or `"new"` to create a fresh pipeline)
- **paper_reference**: Citation for the docstring

## Procedure

### Step 1 — Read the framework contracts

See `.opencode/reference/README.md` decision tree ("Implementing a new transform") for required reading, plus `patterns.md §5.1` (fit-on-train) and `§5.2` (Awkward Arrays). Cross-skill pitfalls (no mutation, no device strings, no manual split) are in `.opencode/reference/style.md §7.1`.

Internalize the decision matrix:

**Mixin selection** (determines fit + multi-source behavior):

| Stateful? | Transform strategy | Mixin to use |
|---|---|---|
| No | Per segment independently | `NoFitPerSegmentMixin` |
| Yes | Fit on concat train, transform per segment | `ConcatFitAndPerSegmentTransformMixin` |
| No | Concat all segments, transform once | `NoFitConcatAlongAxisMixin(axis=N)` |

**Data-type marker** (determines array dispatch):

| Supports | Marker class |
|---|---|
| Dense NumPy only | `DenseTransform` |
| Awkward (ragged) only | `RaggedTransform` |
| Both | `RaggedOrDenseTransform` |

**Optional mixin**:
- Add `InverseTransformMixin` if `supports_inverse` is True

### Step 2 — Read reference implementations

Based on the mixin combination, read the closest existing transform:

- **Stateless, per-segment**: Read `picid/transforms/base_transforms/identity.py`
- **Stateful, per-segment with inverse**: Read `picid/transforms/base_transforms/scaler.py`
- **Stateless, concat along axis**: Read `picid/transforms/base_transforms/concatenate.py`
- **Domain-specific**: Read one from `picid/transforms/bearings/` or `picid/transforms/battery/` for domain patterns

### Step 3 — Determine file location

Place the transform in the appropriate directory:

- **Generic / reusable across datasets**: `picid/transforms/base_transforms/{transform_name}.py`
- **Domain-specific (one dataset family)**: `picid/transforms/{domain}/{transform_name}.py`
  - `bearings/` for bearing-related
  - `battery/` for battery-related
  - `n_cmapss/` for turbofan-related
  - `building/` for HVAC-related
  - `railway/` for railway-related
  - `signal_processing/` for general signal processing
  - Create a new domain directory if none fits

### Step 4 — Implement the transform

```python
import logging
from typing import Any

import numpy as np

from picid.data.data_objects.data import NamedTransformInput
from picid.transforms.base.base_transform import DenseTransform  # or RaggedTransform, RaggedOrDenseTransform
from picid.transforms.base.multisource import (
    NoFitPerSegmentMixin,  # or ConcatFitAndPerSegmentTransformMixin, NoFitConcatAlongAxisMixin
    # InverseTransformMixin,  # if supports_inverse
)

logger = logging.getLogger(__name__)


class {TransformClassName}({MixinClass}, {InverseTransformMixin if needed}, {DataTypeMarker}):
    """
    {Description}.

    {Paper reference}.

    Parameters
    ----------
    {param} : {type}
        {description}. Default: {default}.
    """

    def __init__(self, {params with defaults}):
        super().__init__()  # or super().__init__(axis=N) for NoFitConcatAlongAxisMixin
        self.{param} = {param}
        # For stateful transforms: initialize unfitted state
        # self._fitted_state = None

    def fit_data(self, data: NamedTransformInput, metadata: dict):
        """
        Fit on training data. Called only on the train split.

        data is a dict with one key (the apply_to key) mapping to an array.
        For stateless transforms, this is a no-op.
        """
        # For stateful transforms:
        # keys = list(data.keys())
        # array = data[keys[0]]
        # self._fitted_state = compute_something(array)
        pass

    def transform_data(self, data: NamedTransformInput, metadata: dict) -> Any:
        """
        Transform a single data segment.

        Parameters
        ----------
        data : NamedTransformInput
            Dict with one key mapping to the array to transform.
        metadata : dict
            Contains 'mode' ('train'/'val'/'test'), 'apply_to_keys', 'assign_to_map'.

        Returns
        -------
        np.ndarray or ak.Array
            The transformed array. Shape must be documented.
        """
        keys = list(data.keys())
        array = data[keys[0]]

        # ... transform logic ...
        result = array  # replace with actual transformation

        return result

    # Only if supports_inverse:
    # def inverse_transform(self, data: NamedTransformInput, metadata=None) -> Any:
    #     keys = list(data.keys())
    #     array = data[keys[0]]
    #     return inverse_of_transform(array)
```

**Critical rules for `transform_data`**:
- Do not add `from __future__ import annotations` to generated transform modules. The framework inspects runtime annotations.
- Annotate dispatched transform methods with concrete `metadata: dict` (or `Any` only when necessary), not `typing.Dict[str, Any]`.
- `data` is a `NamedTransformInput` (dict). Typically has ONE key (the `apply_to` key). Extract with `list(data.keys())[0]`.
- Return the **transformed array directly** (not wrapped in a dict). The pipeline handles key assignment via `assign_to_map`.
- If the transform reads MULTIPLE keys (e.g., concatenation), `data` will have multiple keys.
- The `@check_transform_output_consistency` decorator on the base class validates output shape consistency.
- For ragged data, use `awkward` operations instead of NumPy.

### Step 5 — Create or update the transform config YAML

**If adding to an existing pipeline** (`target_pipeline` is a path):

Read the existing YAML and append a new step:

```yaml
# Append to configs/transforms/{dataset}/{pipeline_name}.yaml

{transform_step_name}:
  transform:
    _target_: picid.transforms.{subpackage}.{module}.{TransformClassName}
    {param}: {value}
  metadata:
    apply_to: {input_key}           # string or list of strings
    # assign_to: {output_key}       # only if output key differs from input
    # assign_to_map:                # only if multiple inputs map to specific outputs
    #   input_key1: output_key1
```

**If creating a new pipeline** (`target_pipeline` is `"new"`):

Create `configs/transforms/{dataset}/{pipeline_name}.yaml` with the full step sequence.

**Step ordering matters** — transforms execute top-to-bottom. Place:
- Scalers/normalizers early
- Feature engineering in the middle
- Concatenation/reshaping last

### Step 6 — Validate

```bash
# 1. Import check
cd {project_root}
uv run python -c "from picid.transforms.{subpackage}.{module} import {TransformClassName}; print('OK')"

# 2. Instantiation check
uv run python -c "
from picid.transforms.{subpackage}.{module} import {TransformClassName}
t = {TransformClassName}({constructor args})
print('requires_fit:', t.requires_fit)
print('type:', type(t).__mro__)
"

# 3. Config parse check (if YAML was created/modified)
uv run python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/transforms/{path}.yaml')
print('Steps:', list(cfg.keys()))
"
```

### Step 7 — Report

```
## Transform Implementation Complete

**Files created:**
- `picid/transforms/{subpackage}/{module}.py` — {TransformClassName}

**Config modified:**
- `configs/transforms/{path}.yaml` — added step `{step_name}`

**Mixin chain:** {MixinClass} + {DataTypeMarker} [+ InverseTransformMixin]
**Stateful:** {yes/no}
**Inverse:** {yes/no}

**Pipeline position:** after {previous_step} (step N of M)
```

## Common pitfalls (transform-specific; see also `style.md §7.1`)

- Do NOT return a dict from `transform_data` — return the raw array
- For `ConcatFitAndPerSegmentTransformMixin`: `fit_data` receives the CONCATENATED train data (all units merged), but `transform_data` receives ONE segment at a time
- Single-key transforms: always extract with `list(data.keys())[0]`, never hardcode `"features"`
- `metadata["mode"]` tells you the current split — use it if behavior differs by split (rare)
- If your transform changes the array shape, downstream transforms and the dataset class must be compatible — document the shape change
