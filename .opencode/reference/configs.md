# §3 — Config Conventions

> Read this when writing any Hydra YAML (model, datasource, transform,
> experiment, etc.) or registering a new component.

## 3.1 Config tree structure

```
configs/
├── run.yaml                     # Root config — composes all groups
├── datasource/                  # One .yaml per dataset
├── transforms/                  # One .yaml per transform pipeline (per dataset)
├── dataset/                     # Dataset class configs (context, sliding_window, fit_predict)
├── datamodule/                  # DataModule config (batch_size, num_workers)
├── model/                       # One .yaml per model architecture
├── model_configs/               # Experiment-level model overrides (dataset+model combos)
├── task_definition/             # Task params: seq_len, pred_len, task_type, etc.
├── optimization/                # Optimizer + scheduler configs
├── loss/                        # Loss function configs
├── evaluator/                   # Evaluator configs (per split: train/val/test)
├── callbacks/                   # Lightning callbacks
├── trainer/                     # Trainer config (max_epochs, accelerator, etc.)
├── logger/                      # WandB / other logger configs
├── paths/                       # Directory path resolution
├── cache/                       # Caching flags
├── hydra/                       # Hydra launcher/sweeper config
├── data_splitter/               # Splitter configs
├── experiment/                  # Full experiment compositions
│   ├── <dataset>/
│   │   ├── <task>/
│   │   │   ├── base.yaml        # Dataset+task defaults
│   │   │   ├── mlp.yaml         # Model-specific override
│   │   │   └── ...
```

## 3.2 The `_target_` pattern

Every component is instantiated via `hydra.utils.instantiate(cfg.component)`.
The YAML must contain `_target_: fully.qualified.ClassName` and the constructor
arguments as keys:

```yaml
# configs/model/mlp.yaml
_target_: picid.model.wrappers.mlp_wrapper.MLPWrapper
task_type: ${task_definition.task_type}
seq_len: ${task_definition.seq_len}
input_channels: null        # resolved at runtime via infer_data_dim
num_targets: null
hidden_dim: 64
```

```yaml
# configs/loss/default.yaml
_target_: picid.loss.default.MSELoss
```

## 3.3 Transform config format

Transforms are defined as an **ordered dict** where each key is a step name:

```yaml
# configs/transforms/concepts_n_cmapss_ds02/depater2023_default.yaml
scaler_features:
  transform:
    _target_: picid.transforms.n_cmapss.n_cmapss_scalers.N_CMAPSSFeaturesScaler
    scaling: standard
  metadata:
    apply_to: features              # which data key to read

scaler_rul:
  transform:
    _target_: picid.transforms.base_transforms.scaler.ConstantScaler
    factor: 0.01
  metadata:
    apply_to: rul

concatenate_features:
  transform:
    _target_: picid.transforms.base_transforms.concatenate.ConcatenateTransform
  metadata:
    apply_to: ["features", "descriptors"]
    assign_to: features             # output key name
```

**Metadata fields**:
- `apply_to`: key(s) the transform reads from
- `assign_to`: key the transform writes to (if different from apply_to)
- `assign_to_map`: dict mapping input keys to output keys

## 3.4 Experiment config composition

A full experiment composes all groups.  Example:

```yaml
# configs/experiment/concepts_n_cmapss_ds02/prognostics/mlp.yaml
#@package _global_

defaults:
  - concepts_n_cmapss_ds02/prognostics/base     # sets datasource, evaluator, task_definition
  - /model_configs/prognostics/mlp               # sets model, dataset, datamodule, optimization
  - override /transforms: concepts_n_cmapss_ds02/depater2023_default
```

Where `base.yaml` provides:
```yaml
#@package _global_
defaults:
  - override /datasource: concepts_n_cmapss_ds02
  - override /evaluator: rul
  - override /task_definition: prognostics/concepts_n_cmapss_rul_prediction

evaluator:
  train:
    inverse_transform_name: scaler_rul
    apply_inverse_scaling: True
  val:
    inverse_transform_name: scaler_rul
    apply_inverse_scaling: True
  test:
    inverse_transform_name: scaler_rul
    apply_inverse_scaling: True
```

And `model_configs/prognostics/mlp.yaml` provides:
```yaml
#@package _global_
defaults:
  - override /dataset: prognostics/rul_multiunit_dataset
  - override /datamodule: base_datamodule
  - override /model: mlp
  - override /optimization: reduce_on_plateau

task_definition:
  requires_training: true

model:
  input_channels: ${infer_data_dim:features,-1}
  num_targets: ${infer_data_dim:${task_definition.task_type},-1}
```

## 3.5 How to register a new component

1. **Write the Python class** in the appropriate `picid/` subdirectory
2. **Create a YAML config** in the matching `configs/` subdirectory with `_target_` pointing to your class
3. **Reference it** from an experiment config using `defaults:` or `override /group: your_config`

No plugin registry, no decorator, no __init__.py entry — Hydra discovers configs
by file path, and `_target_` handles instantiation.

## 3.6 Custom OmegaConf resolvers

Defined in `picid/run.py`, available in configs:
- `${flat:a/b}` → `a+b` (for directory names)
- `${uuid:short}` → 8-char uuid
- `${sum:a,b}` → `a+b`
- `${prod:a,b}` → `a*b`
- `${diff:a,b}` → `a-b`
- `${quot:a,b}` → `a//b` (exact integer division)
- `${mod:a,b}` → `a%b`
- `${int_div:a,b}` → `a//b`
- `${infer_data_dim:key,dim}` → resolved after data loading; reads shape from data dict
- `${infer_dataloader_length:split}` → number of batches in a split's dataloader
