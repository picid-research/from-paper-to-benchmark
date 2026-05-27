# §6 + §7 — Style, Conventions, and What NOT to Implement

> Read this before writing any code: the framework already covers most
> infrastructure. Only implement what is genuinely novel to the paper.

## §6.1 Import conventions

```python
# Standard library
import logging
from typing import Any, Dict, List, Optional

# Third party
import numpy as np
import torch
import torch.nn as nn
import awkward as ak

# Framework
from picid.transforms.base.base_transform import DenseTransform
from picid.model.wrappers.base import AbstractFeedForwardTrainingWrapper
```

## §6.2 Naming conventions

- **Files**: `snake_case.py` — e.g., `mlp_wrapper.py`, `sliding_window_batch_dataset.py`
- **Classes**: `PascalCase` — e.g., `MLPWrapper`, `SlidingWindowBatchDataset`
- **Config files**: `snake_case.yaml` matching the Python module — e.g., `mlp.yaml`
- **Transform step names**: `snake_case` in transform YAML — e.g., `scaler_features`
- **Task types**: lowercase strings — e.g., `"rul"`, `"classification"`, `"forecasting"`

## §6.3 Type expectations

- Data arrays: `np.ndarray`, `ak.Array`, or `torch.Tensor`
- Batch dicts: `Dict[str, torch.Tensor]`
- Model output: `Dict[str, torch.Tensor]` with keys `"predictions"`, `"targets"`, optionally `"loss"`
- Predictions/targets shape: `(Batch, Seq, Features)` — typically `(B, 1, num_targets)` for regression
- Config objects: `omegaconf.DictConfig`

## §6.4 Docstring style

NumPy-style docstrings (configured via `tool.numpydoc_validation` in `pyproject.toml`):
```python
def method(self, x: torch.Tensor) -> torch.Tensor:
    """Short description.

    Parameters
    ----------
    x :
        Input tensor of shape (B, T, C).

    Returns
    -------
    torch.Tensor
        Output tensor of shape (B, T, 1).
    """
```

## §6.5 Testing

- Tests mirror the source tree: `test/` mirrors `picid/`
- Pipeline regression tests in `test/pipeline/` run full experiments against snapshots
- Use `pytest` with markers: `@pytest.mark.slow`, `@pytest.mark.requires_snapshots`
- Temp dirs go under `test/.pytest_tmp` (configured in `pyproject.toml`)

## §6.6 Linting and formatting

- **Ruff** for linting (configured in `pyproject.toml [tool.ruff]`)
- Source root: `picid/`
- Excluded: `local_packages/`, `scripts/`, `_archive/`, notebooks
- **mypy** with `ignore_missing_imports = true`, `check_untyped_defs = false`
- **pre-commit** hooks configured in `.pre-commit-config.yaml`

## §6.7 Python version

- Requires Python >= 3.12 (configured in `pyproject.toml`)
- Package manager: `uv` (see `uv.lock`, local packages in `local_packages/`)

---

## §7 What NOT to Implement

**The framework already handles all of the following.  Do NOT reimplement:**

- **Training loops**: Lightning's `Trainer.fit()` / `Trainer.test()` handles epochs, batches, gradient steps, mixed precision, distributed training
- **Logging**: WandB logger is pre-wired; metrics are logged via `self.log()` in Lightning hooks
- **Checkpointing**: `ModelCheckpointWithConfig` callback saves best/last checkpoints automatically
- **Callbacks**: Early stopping, learning rate monitoring, timer, resource tracker — all pre-configured
- **Data splitting**: Use `TimeSplitter`, `BySourceSplitter`, or `TimeStampSplitter` — never split manually
- **Metric computation infrastructure**: Use `DefaultEvaluator` with `metric_names` — never compute metrics in the model
- **Config parsing**: Hydra composes configs — never parse YAML manually
- **Inverse scaling in evaluation**: The evaluator handles this via `ScalingWrapper` — never inverse-transform in the model
- **Batch assembly**: Dataset classes handle sliding windows, padding, collation — never reshape in the model
- **Data caching**: The `PreProcessor` caching system handles disk persistence — never cache manually
- **Optimizer/scheduler creation**: Configured via Hydra, partially instantiated, completed in `configure_optimizers()`
- **Seed management**: `L.seed_everything()` is called in `run()` — never seed manually
- **Thread limiting**: Set in config via `num_threads` — never set `OMP_NUM_THREADS` etc. manually

**What you SHOULD implement** when adding a paper:
1. The **model backbone** (`nn.Module` in `picid/model/methods/`)
2. The **model wrapper** (in `picid/model/wrappers/`)
3. Any **novel transforms** the paper requires (in `picid/transforms/`)
4. A **custom loss** if the paper uses one not already available
5. A **custom metric** if the paper defines one not already available
6. **Config files**: model YAML, model_config YAML, experiment YAML, and optionally transform YAML

## §7.1 Cross-skill pitfalls (applies to every `/implement-*`)

These come up in every implementation skill. Follow them without restating them in individual SKILL.md bodies.

- **Respect framework boundaries**: model backbones live in `picid/model/methods/`, wrappers in `picid/model/wrappers/`, losses in `picid/loss/` (subclass `AbstractLoss`), transforms in `picid/transforms/...` (subclass `DenseTransform`/`SparseTransform`), datasources in `picid/data/datasources/` (subclass the right base in `datasources.md`). Configs mirror the module path under `configs/`.
- **Never mutate input tensors/arrays in place.** Transforms, losses, and models must return new tensors. In-place mutation breaks caching, inverse scaling, and multi-worker dataloading.
- **Never hardcode device strings** (`.cuda()`, `.to("cuda")`, `.to("cpu")`). Lightning handles device placement. Honor `accelerator_tier` from the blueprint when composing configs.
- **Never parse YAML or env vars manually.** Hydra composes configs; read values from `DictConfig` or constructor kwargs.
- **Never compute metrics inside the model.** The evaluator owns metric computation; the model returns a predictions/targets dict.
- **Never inverse-transform inside the model.** `ScalingWrapper` does that in the evaluator.
- **Never split data manually.** Use the existing splitters (`TimeSplitter`, `BySourceSplitter`, `TimeStampSplitter`).
- **Never call `seed_everything` or set thread counts manually.** `run()` handles both.
- **`fit` runs on train only.** Any statistics a transform or loss computes must be fit on the training split, then applied to val/test via stored state. See `patterns.md`.
