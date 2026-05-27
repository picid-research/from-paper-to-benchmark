---
name: implement-datasource
description: Implement a new data source loader + config for the PICID/PICID framework, given a dataset specification from a paper blueprint.
---

# /implement-datasource

Create a new data source loader that integrates a dataset into the PICID pipeline.

## Input contract

The orchestrator must provide:

- **datasource_name**: snake_case identifier (e.g., `femto_bearings`)
- **data_format**: How the raw data is stored (CSV, HDF5, Parquet, PHMD, etc.)
- **data_description**: What the dataset contains (sensors, targets, units)
- **loader_type**: One of:
  - `single_source` — one unit/machine, split by time
  - `multi_source_predefined` — multiple units, you assign units to train/val/test in loader
  - `multi_source_by_splitter` — multiple units, split assignment handled by `BySourceSplitter` in config
- **data_keys**: What keys the loader should produce (e.g., `features`, `rul`, `timestamps`, `unit`, `health_states`)
- **data_shapes**: Expected shapes per key (document ragged vs dense)
- **split_strategy**: How to split — `time` (ratio), `by_source` (unit assignment), `database` (timestamp)
- **task_mode**: Primary task mode (e.g., `rul`, `classification`, `forecasting`, `anomaly_detection`)
- **paper_reference**: Citation or URL

## Procedure

### Step 1 — Read the framework contracts

See `.opencode/reference/README.md` decision tree ("Implementing a new datasource") for required reading. Framework boundaries, in-place-mutation rules, device rules, seeding, and thread limits are in `.opencode/reference/style.md §7.1` — apply them without restating.

Datasource-specific key constraints:
- Each split value must be a **list** of arrays (one per unit), even for single-unit datasets
- Features shape: `(Time, Freq/Samples, Channels)` — can be ragged along Time via Awkward Array
- Targets shape: `(Time, 1, 1)` for regression or `(Time, 1, 1)` with class indices for classification
- Return a `SplitDatasetContainer` (for pre-split data) or `DatasetContainer` (for single-source within multi-source)

### Step 2 — Read reference implementations

Based on loader_type:

- **single_source**: Read `picid/data/datasources/base/single_source_loader.py` (the base) and one concrete example like `picid/data/datasources/ETTh.py` or `picid/data/datasources/railway.py`
- **multi_source_predefined**: Read `picid/data/datasources/toy_example.py` — handles splits internally
- **multi_source_by_splitter**: Read `picid/data/datasources/base/multi_source_loader.py` and a concrete single-source like `picid/data/datasources/concepts_n_cmapss.py` plus the multi-source config `configs/datasource/concepts_n_cmapss_ds02.yaml`

Also check if the dataset is available via PHMD (the `phmd` package in `local_packages/`). If so, read `picid/data/datasources/base/phmd_loader.py` for the PHMD integration pattern — it handles download and caching automatically.

### Step 3 — Implement the loader

#### Option A: SingleSourceLoader (single unit, split by time)

Create `picid/data/datasources/{datasource_name}.py`:

```python
import logging
from typing import Any, Dict, Union, override

import numpy as np
import awkward as ak
from pandas.core.generic import NDFrame

from picid.data.datasources.base.single_source_loader import SingleSourceLoader

logger = logging.getLogger(__name__)


class {LoaderClassName}(SingleSourceLoader):
    """
    Loader for {dataset name}.

    {Paper reference}.

    Parameters
    ----------
    data_dir : str
        Path to the raw data directory.
    data_name : str
        Dataset identifier.
    task_mode : str
        Task mode (e.g., 'rul', 'classification').
    load_arguments : dict, optional
        Dataset-specific loading parameters.
    """

    def __init__(self, data_dir: str, load_arguments: dict = None, **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.load_arguments = load_arguments or {}

    @override
    def _load_data(self) -> Dict[str, Union[np.ndarray, ak.Array, NDFrame]]:
        """
        Load raw data and return as a flat dict of arrays.

        Returns
        -------
        dict
            Keys like 'features', 'timestamps', '{task_mode}', etc.
            Each value is a numpy array or Awkward Array.
            Do NOT split here — SingleSourceLoader.split_data() handles splitting
            using the configured data_splitter.
        """
        # 1. Read raw files from self.data_dir
        # 2. Process into arrays
        # 3. Return dict

        features = ...   # shape: (T, C) or (T, F, C)
        targets = ...    # shape: (T, 1) or (T,)

        return {
            "features": features,
            "timestamps": timestamps,
            "{task_mode}": targets,
        }
```

The parent class handles:
- `load_data()` calls `_load_data()` and stores in `self.data_dict`
- `split_data()` uses the configured `data_splitter` to create train/val/test
- `get_data()` wraps in `SplitDatasetContainer` and ensures list-of-arrays format

#### Option B: Direct AbstractDataSourceLoader (multi-source, predefined splits)

Create `picid/data/datasources/{datasource_name}.py`:

```python
import logging
from typing import Any, Dict, List, override

import numpy as np
import awkward as ak

from picid.data.data_objects.data import SplitDatasetContainer
from picid.data.datasources.base.interfaces import AbstractDataSourceLoader

logger = logging.getLogger(__name__)


class {LoaderClassName}(AbstractDataSourceLoader):
    """
    Loader for {dataset name} with predefined train/val/test splits.

    {Paper reference}.
    """

    def __init__(self, data_dir: str, multisource_data_splitter=None, **kwargs):
        super().__init__(multisource_data_splitter=multisource_data_splitter, **kwargs)
        self.data_dir = data_dir
        self._is_loaded = False
        self._is_splitted = False
        self.data_dict = None
        self.meta_data = {}

    @override
    def load_data(self):
        self.data_dict = self._load_data()
        self._is_loaded = True
        self._is_splitted = True  # splits are predefined

    @override
    def get_data(self) -> SplitDatasetContainer:
        assert self._is_loaded
        return SplitDatasetContainer(**self.data_dict)

    @override
    def get_data_name(self) -> str:
        return self.data_name

    @override
    def split_data(self):
        pass  # splits are predefined in _load_data

    def _load_data(self) -> dict:
        """
        Returns
        -------
        dict
            Structure: {
                "features": {"train": [arr1, arr2, ...], "val": [...], "test": [...]},
                "rul":      {"train": [arr1, arr2, ...], "val": [...], "test": [...]},
                ...
            }
            Each split value is a LIST of arrays (one per unit).
        """
        ...
```

#### Option C: SingleSourceLoader used within MultiSourceLoader (by-source splitting)

Create the single-source loader as in Option A, but with `is_part_of_multisource=True` support.
The multi-source composition is done entirely in config (see Step 4).

### Step 4 — Create config YAML

#### For single-source with time splitting:

Create `configs/datasource/{datasource_name}.yaml`:

```yaml
_target_: picid.data.datasources.{module}.{LoaderClassName}

data_name: {datasource_name}
task_mode: {task_mode}

data_dir: ${paths.data_dir}

data_splitter:
  _target_: picid.data.preprocessing.TimeSplitter
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  create_splits_for:
    - features
    - timestamps
    - {task_mode}
```

#### For multi-source with by-source splitting:

Create `configs/datasource/{datasource_name}.yaml`:

```yaml
_target_: picid.data.datasources.base.multi_source_loader.MultiSourceLoader
_recursive_: False

data_name: MultiSource_{datasource_name}
task_mode: {task_mode}

defaults:
  - /datasource@train1: {single_source_config}
  - /datasource@test1: {single_source_config}
  - /datasource@val1: {single_source_config}

source_list:
  train1: ${..train1}
  test1: ${..test1}
  val1: ${..val1}

multisource_data_splitter:
  _target_: picid.data.preprocessing.BySourceSplitter
  sources_train: [train1]
  sources_test: [test1]
  sources_val: [val1]

train1:
  load_arguments:
    units: [1, 2, 3]        # which units for training
test1:
  load_arguments:
    units: [4, 5]            # which units for testing
val1:
  load_arguments:
    units: [6]               # which units for validation
```

#### For predefined splits (Option B):

```yaml
_target_: picid.data.datasources.{module}.{LoaderClassName}

data_name: {datasource_name}
task_mode: {task_mode}
data_dir: ${paths.data_dir}
```

### Step 5 — Validate

```bash
# 1. Import check
uv run python -c "from picid.data.datasources.{module} import {LoaderClassName}; print('OK')"

# 2. Config parse check
uv run python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/datasource/{datasource_name}.yaml')
assert '_target_' in cfg
print('config OK:', cfg._target_)
print('task_mode:', cfg.task_mode)
"

# 3. Instantiation smoke test (only if data is available locally)
# uv run python -c "
# import hydra
# from omegaconf import OmegaConf
# cfg = OmegaConf.load('configs/datasource/{datasource_name}.yaml')
# # Replace interpolations for test
# cfg.data_dir = '/path/to/data'
# loader = hydra.utils.instantiate(cfg)
# loader.load_data()
# print('Loaded successfully')
# "
```

### Step 6 — Report

```
## Datasource Implementation Complete

**Files created:**
- `picid/data/datasources/{module}.py` — {LoaderClassName}
- `configs/datasource/{datasource_name}.yaml` — Hydra config

**Loader type:** {single_source | multi_source_predefined | multi_source_by_splitter}
**Task mode:** {task_mode}
**Data keys produced:** {list of keys}
**Split strategy:** {time | by_source | predefined}

**Data requirements:**
- Raw data expected at: `${paths.data_dir}/{subdirectory}`
- Format: {CSV/HDF5/etc.}

**Remaining wiring needed:**
- Transform pipeline (handled by implement-transform)
- Task definition config
- Experiment config (handled by implement-experiment)
```

## Common pitfalls

- Each split value MUST be a **list** of arrays, even for single-unit: `[array]` not `array`
- Features must be at least 2D: `(T, C)`. Use `.reshape(-1, 1)` for single-channel
- Targets for regression must be `(T, 1)` — squeeze extra dims if needed
- For Awkward Arrays: convert with `ak.from_regular(ak.from_numpy(arr), axis=1)` — the `axis=1` makes the second dimension variable
- Do NOT normalize/scale data in the loader — that's what transforms are for
- Do NOT shuffle data in the loader — the dataloader handles shuffling
- Store `metadata` (unit_ids, unit_names) for multi-unit datasets — the evaluator may need it
- The `data_dir` should use `${paths.data_dir}` interpolation, not hardcoded paths
