---
name: implement-experiment
description: Wire together datasource + transforms + model + evaluator into a runnable PICID/PICID experiment config.
---

# /implement-experiment

Compose Hydra configs that wire an existing (or newly created) datasource, transform pipeline, model, and evaluator into a runnable experiment.

## Input contract

The orchestrator must provide:

- **experiment_name**: Path-style identifier (e.g., `femto_bearings/prognostics/my_model`)
- **datasource_config**: Name of the datasource YAML (e.g., `femto_bearings`)
- **transforms_config**: Name of the transform pipeline YAML (e.g., `femto_bearings/default`)
- **model_config**: Name of the model YAML (e.g., `my_model`)
- **task_type**: One of the task types from `picid/baselines/definitions.py`
- **task_params**: Dict with `seq_len`, `pred_len`, `label_len`, `stride`, `padding_left_flag`, etc.
- **dataset_class**: Which dataset to use — one of: `sliding_window_batch`, `context_batch`, `fit_predict_task`, or a prognostics/diagnostics variant
- **evaluator_config**: Which evaluator YAML to use (e.g., `default`, `rul`, `classification`, `forecasting`)
- **requires_training**: Whether the model needs gradient training (True for neural nets, False for pretrained/constant-loss)
- **inverse_scaling**: Dict with `enabled`, `transform_name` (name of the scaler step in the transform pipeline)
- **optimization**: Which optimization config to use (e.g., `default`, `reduce_on_plateau`). Must name a config file that actually exists under `configs/optimization/` — the static verifier will BLOCK on a missing referent. If the paper's optimizer does not have a matching existing config, invoke the upstream implementation skill to create one before calling this skill.
- **data_requirements**: List of data keys the model needs from the dataset (e.g., `["features", "rul", "timestamps"]`)
- **dataset_contract**: Compact contract from the blueprint: dataset family, datasource/transform/dataset/evaluator/task configs, processed keys.
- **model_io_contract**: Compact contract from the blueprint: wrapper type, batch input keys, target keys, prediction/target shapes, supervised vs reconstruction.
- **split_contract**: Compact contract from the blueprint: train/val/test units or conditions, row filters, and enforcement point.
- **paper_hyperparameters**: REQUIRED dict mirroring blueprint §8.1. Every field below must be present. Use the literal string `"NOT_SPECIFIED"` when the paper did not state a value:
  - `optimizer` (name: `adam`, `adamw`, `sgd`, ...)
  - `learning_rate` (number or `"NOT_SPECIFIED"`)
  - `lr_schedule` (name or `"NOT_SPECIFIED"`)
  - `weight_decay` (number or `"NOT_SPECIFIED"`)
  - `grad_clip` (number or `"NOT_SPECIFIED"`)
  - `warmup` (number of steps/epochs, or `"NOT_SPECIFIED"`)
  - `max_epochs` (int or `"NOT_SPECIFIED"`)
  - `batch_size` (int or `"NOT_SPECIFIED"`)
  - `training_protocol_notes` (string or `"NOT_SPECIFIED"`)

  The skill MUST refuse invocation if any of these fields are absent from the input dict (presence is required; value may be `"NOT_SPECIFIED"`).
- **dataset_recovery_context**: Optional. If this experiment is being regenerated after a dataset failure, include `failed_dataset`, `reason`, and same-dataset config/path/cache repair. Cross-dataset fallback is disabled.

When `dataset_recovery_context` is provided, reuse the already implemented model, loss, and novel transforms unless verification proves one is broken. Regenerate only the same dataset's Hydra wiring: datasource, transform pipeline selection, dataset class, task definition, evaluator, and experiment config path. Dataset recovery rules themselves are in `.opencode/reference/policies.md`.

## Procedure

### Step 1 — Read the framework contracts

See `.opencode/reference/README.md` decision tree ("Composing a new experiment"). Also read `configs.md §3.4–§3.5`, `inventory.md §4.7–§4.8`, and `patterns.md §5.5–§5.6`.

Understand the three-layer config structure:
1. **Task definition** (`configs/task_definition/`) — task params (seq_len, etc.)
2. **Model configs** (`configs/model_configs/`) — binds model to dataset/datamodule/optimization
3. **Experiment** (`configs/experiment/`) — composes datasource + task + model_configs + transforms

### Step 2 — Read existing reference configs

Read a complete experiment chain for the same task type:

**For RUL/regression:**
```
configs/experiment/concepts_n_cmapss_ds02/prognostics/base.yaml
configs/experiment/concepts_n_cmapss_ds02/prognostics/mlp.yaml
configs/model_configs/prognostics/mlp.yaml
configs/task_definition/prognostics/concepts_n_cmapss_rul_prediction.yaml
configs/evaluator/rul.yaml
configs/dataset/prognostics/rul_multiunit_dataset.yaml
```

**For classification:**
```
configs/evaluator/classification.yaml
configs/dataset/diagnostics/
configs/loss/CrossEntropy.yaml
```

**For forecasting:**
```
configs/evaluator/forecasting.yaml
configs/dataset/sliding_window_batch.yaml
configs/dataset/state_forecasting/
```

### Step 3 — Create or reuse task definition

Check if an existing task definition works. If not, create one.

Create `configs/task_definition/{task_category}/{task_def_name}.yaml`:

```yaml
seq_len: {value}
label_len: {value}
pred_len: {value}
task_type: {task_type}
padding_left_flag: {True/False}
stride: {value}
stride_train: {value}
subset_ratio: 1.0
subset_seed: ${seed}

model:
  data_requirements:
    input_tensors:
      - features
      - {task_type}        # e.g., rul, health_states
      - timestamps
      # ... other keys the model needs

target_metric: val/loss
target_metric_mode: min
```

**Task categories** map to directories: `prognostics/`, `diagnostics/`, `forecasting/`, `state_forecasting/`, `anomaly_detection/`.

### Step 4 — Create model_configs YAML

This bridges the model with the dataset class and optimization strategy.

Create `configs/model_configs/{task_category}/{model_name}.yaml`:

```yaml
#@package _global_

defaults:
  - override /dataset: {dataset_config_path}
  - override /datamodule: base_datamodule
  - override /model: {model_name}
  - override /optimization: {optimization}

task_definition:
  requires_training: {true/false}

model:
  input_channels: ${infer_data_dim:features,-1}
  num_targets: ${infer_data_dim:${task_definition.task_type},-1}
  # Add model-specific overrides here if needed
```

**Dataset config selection guide: choose by model I/O first, task second.**

| Task | Wrapper type | Dataset config |
|---|---|---|
| RUL supervised `features -> rul` | FeedForwardTraining | `prognostics/rul_multiunit_dataset` |
| RUL | FitPredict | `fit_predict_task` |
| Reconstruction/autoencoder `features -> features` | FeedForwardTraining | `sliding_window_batch` or an explicit reconstruction-compatible config; do not use `prognostics/rul_multiunit_dataset` unless target/context lengths are proven equal |
| Classification | FeedForwardTraining | `diagnostics/fault_classification_multiunit_dataset` |
| Forecasting | FeedForwardTraining | `sliding_window_batch` or `state_forecasting/default` |
| Anomaly detection | FeedForwardTraining | `anomaly_detection/anomaly_detection_multiunit_dataset` |

Before writing `model_configs`, verify `dataset_contract.processed_keys` covers every `model_io_contract.batch_input_key` and target key. If the paper states row/unit/condition splits, the experiment config must enforce the `split_contract`; a model-side flag alone is not enforcement.

**Optimization selection guide:**

| Scenario | Config |
|---|---|
| Standard training | `default` (Adam) |
| With LR scheduling | `reduce_on_plateau` |
| No training (constant loss, FitPredict) | `default` (ignored at runtime) |

For validate-paper benchmark validation, LR scheduler/check cadence is a PICID framework invariant. Prefer the scheduler/optimization config used by the closest existing PICID experiment for the selected dataset/task. Record paper scheduler details as provenance, but do not replace the framework scheduler solely because the paper states one unless the user explicitly requested exact paper-protocol training.

### Step 5 — Create experiment base config

If the datasource doesn't already have a base config for this task category, create one.

Create `configs/experiment/{datasource_name}/{task_category}/base.yaml`:

```yaml
#@package _global_

defaults:
  - override /datasource: {datasource_config}
  - override /evaluator: {evaluator_config}
  - override /task_definition: {task_category}/{task_def_name}

# Configure inverse scaling per split (if applicable)
evaluator:
  train:
    inverse_transform_name: {scaler_step_name}    # must match a step in the transforms YAML
    apply_inverse_scaling: {True/False}
  val:
    inverse_transform_name: {scaler_step_name}
    apply_inverse_scaling: {True/False}
  test:
    inverse_transform_name: {scaler_step_name}
    apply_inverse_scaling: {True/False}
```

If `inverse_scaling.enabled` is False, explicitly override every evaluator split to `apply_inverse_scaling: false` and `inverse_transform_name: null`. Do not assume a datasource/task base has false defaults; several existing bases enable inverse scaling.

**Evaluator selection guide:**

| Task | Evaluator config | Typical metrics |
|---|---|---|
| RUL | `rul` | mae, mse, rmse (+ nasa_score, phm_score) |
| Regression | `default` | mae, mse, rmse |
| Classification | `classification` | accuracy, f1, precision, recall |
| Forecasting | `forecasting` | mae, mse, rmse, mape |
| Per-unit aggregation | `per_unit` | per-unit + aggregate |

For classification tasks, also override the loss:
```yaml
defaults:
  - override /loss: CrossEntropy
```

### Step 6 — Create the experiment config and inline paper hyperparameters

Create `configs/experiment/{datasource_name}/{task_category}/{model_name}.yaml`:

```yaml
#@package _global_

defaults:
  - {datasource_name}/{task_category}/base
  - /model_configs/{task_category}/{model_name}
  - override /transforms: {transforms_config}

# --- Paper hyperparameters (from blueprint §8.1) ---
# Every field from paper_hyperparameters is inlined below. A concrete value is
# used when the paper specified one; otherwise the framework default is chosen
# and a `# SUBSTITUTION:` comment identifies the field and the chosen default.
# Exceptions: datamodule batch sizes and LR scheduler/check cadence are governed
# by PICID's framework validation contract. Paper `batch_size` and `lr_schedule`
# are recorded in provenance comments but must not override train/val/test batch
# sizes or the framework scheduler cadence in the primary validation profile.

optimization:
  lr: {learning_rate}              # or # SUBSTITUTION: learning_rate=NOT_SPECIFIED → default 1e-3 ({framework default source})
  optimizer:
    weight_decay: {weight_decay}   # or # SUBSTITUTION: weight_decay=NOT_SPECIFIED → default 0.0 ({framework default source})

trainer:
  max_epochs: {max_epochs}         # or # SUBSTITUTION: max_epochs=NOT_SPECIFIED → default {framework default} ({source})
  gradient_clip_val: {grad_clip}   # or # SUBSTITUTION: grad_clip=NOT_SPECIFIED → default null ({framework default source})

datamodule:
  train_batch_size: 512   # fixed validation batch; paper batch_size={paper_batch_size_or_NOT_SPECIFIED}
  val_batch_size: 1024    # fixed validation batch
  test_batch_size: 1024   # fixed validation batch
```

Rules for Step 6:

1. For every field in `paper_hyperparameters` whose value is a concrete number/name, inline the literal value as a Hydra override. Do not rely on the defaults coming from `/optimization: {name}` or `/datamodule: {name}` — the override ensures the value survives composition regardless of the default group.
2. Exception to rule 1: do not inline the paper's single `batch_size` into `datamodule.train_batch_size`, `datamodule.val_batch_size`, or `datamodule.test_batch_size` for validation runs. Set validation datamodule batch sizes explicitly to train `512`, val `1024`, and test `1024` for every validation config. Do not inspect the base datamodule or run-script override chain to choose these values; record the paper batch size only as provenance.
2a. Scheduler exception: do not replace PICID's LR scheduler/check cadence for the primary validation profile. If the paper states `lr_schedule`, record it in comments/provenance and keep the closest existing PICID scheduler/optimization config. If the paper omits `lr_schedule`, record `NOT_SPECIFIED` and keep the PICID scheduler. The base learning-rate value itself is still model-specific: use the paper value when stated, otherwise impute from framework defaults.
3. For every non-batch, non-scheduler field whose value is `"NOT_SPECIFIED"`, read the PICID framework default from the referenced config under `configs/optimization/`, `configs/trainer/`, or `configs/datamodule/`, then:
   - inline the framework default as the override value, AND
   - emit a YAML comment on the same line starting with `# SUBSTITUTION:` that names the field, records that it was `NOT_SPECIFIED` in the paper, names the default used, and cites the source config path.
4. If `optimizer` (name) differs from what the referenced `optimization` config provides (e.g., paper says `adam` but `optimization: sgd` was supplied as input), refuse the invocation with a `BLUEPRINT_OPTIMIZER_MISMATCH` error rather than silently composing an inconsistent experiment. The orchestrator must resolve the mismatch upstream.
5. If the selected PICID scheduler/optimization config does not exist under `configs/optimization/`, refuse the invocation with a `BLUEPRINT_REFERENCES_MISSING_CONFIG` error (same class that `/verify-static` uses). Never invent a scheduler config here.
6. Do not rely on default-group fallback (e.g., letting a missing `optimization: adam` silently resolve to `default`). Every paper-specified non-batch field must appear as an explicit inlined override in the experiment YAML.

The experiment config is still a thin composition file, but now it is also the **single source of truth** for paper-stated/imputed model-specific hyperparameters and for the scheduler/check cadence used by validation. `/validate-paper` uses the primary validation profile by default: force datamodule batch sizes to `512/1024/1024`, preserve PICID scheduler/check cadence, then use paper-stated or explicitly imputed model-specific HPs for the rest. Do not bake a paper batch size or paper scheduler into the primary validation experiment YAML. They are provenance only unless the user explicitly requests exact paper-protocol training.

### Step 7 — Validate the full config

```bash
# 1. Dry-run config composition (no training, just check config resolves)
cd {project_root}
uv run python -c "
import hydra
from omegaconf import OmegaConf, DictConfig

OmegaConf.register_new_resolver('flat', lambda s: s.replace('/', '+'))
OmegaConf.register_new_resolver('uuid', lambda kind='short': 'test')
OmegaConf.register_new_resolver('sum', lambda *args: sum(args))
OmegaConf.register_new_resolver('prod', lambda *args: 1)
OmegaConf.register_new_resolver('diff', lambda a, b: a - b if not isinstance(a, str) else a)
OmegaConf.register_new_resolver('quot', lambda a, b: a // b if b != 0 else 0)
OmegaConf.register_new_resolver('mod', lambda x, y: int(x) % int(y))
OmegaConf.register_new_resolver('int_div', lambda a, b: int(a) // int(b))
OmegaConf.register_new_resolver('infer_data_dim', lambda key, dim: 10)
OmegaConf.register_new_resolver('infer_dataloader_length', lambda key: 100)

hydra.initialize(version_base='1.3', config_path='configs')
cfg = hydra.compose(config_name='run.yaml', overrides=['experiment={experiment_name}'])

# Check critical fields exist
assert cfg.datasource._target_, 'datasource._target_ missing'
assert cfg.model._target_, 'model._target_ missing'
assert cfg.loss._target_, 'loss._target_ missing'
assert cfg.task_definition.task_type, 'task_type missing'
print('Config composes successfully')
print('datasource:', cfg.datasource._target_)
print('model:', cfg.model._target_)
print('loss:', cfg.loss._target_)
print('task_type:', cfg.task_definition.task_type)
print('transforms steps:', list(cfg.transforms.keys()) if cfg.transforms else 'none')
"
```

If this fails, the error message will indicate which config group has issues. Fix and re-run.

### Step 8 — Report

```
## Experiment Configuration Complete

**Files created:**
- `configs/task_definition/{path}.yaml` — task params (seq_len={}, pred_len={}, task_type={})
- `configs/model_configs/{path}.yaml` — model+dataset+optimization binding
- `configs/experiment/{path}/base.yaml` — datasource+evaluator+task_definition
- `configs/experiment/{path}/{model_name}.yaml` — final experiment composition

**Run command:**
uv run python picid/run.py experiment={experiment_name}

**Dataset recovery:** {none, or "regenerated same-dataset wiring after {failed_dataset} failed; reason={reason}"}

**Config chain:**
experiment → base (datasource + evaluator + task_def)
           → model_configs (model + dataset + optimization)
           → transforms override
           → paper-HP overrides inlined in experiment YAML

**Paper hyperparameters (all nine fields):**
| Field | Paper value | Inlined value | Substitution |
|-------|-------------|---------------|--------------|
| optimizer | {value or NOT_SPECIFIED} | {value used} | {none or framework default + source} |
| learning_rate | ... | ... | ... |
| lr_schedule | ... | ... | ... |
| weight_decay | ... | ... | ... |
| grad_clip | ... | ... | ... |
| warmup | ... | ... | ... |
| max_epochs | ... | ... | ... |
| batch_size | ... | ... | ... |
| training_protocol_notes | ... | {propagated into comments/callbacks/readme} | ... |

**Substitutions applied** (fields that were `NOT_SPECIFIED` in the paper): list every one, or `none`.

**Evaluator:** {evaluator_config} with metrics {metric_names}
**Inverse scaling:** {enabled/disabled}, transform: {transform_name}
**Dataset class:** {dataset_class}
**Model I/O contract:** {input keys → target keys, expected batch/prediction/target shapes}
**Split contract:** {enforcement point and expected split summary}
**Training:** {yes/no}
```

## Common pitfalls

- The `#@package _global_` directive is REQUIRED at the top of experiment configs — without it, Hydra nests the config under the wrong package
- `defaults:` entries for the same experiment directory use relative paths (e.g., `concepts_n_cmapss_ds02/prognostics/base`), while cross-group overrides use `/` prefix (e.g., `/model_configs/prognostics/mlp`)
- `override /group: value` is needed when changing a default that was already set by a parent config
- `${infer_data_dim:key,dim}` resolvers are NOT available during config composition — they're registered at runtime after data loading. Use `null` in model config and resolve via model_configs
- `inverse_transform_name` must exactly match a step name in the transforms YAML
- For FitPredict models: set `requires_training: true` (the LightningModule still runs fit via training_step) and use `fit_predict_task` dataset
- Check that `data_requirements.input_tensors` lists ALL keys your model reads from the batch — missing keys cause KeyError at runtime
- Reconstruction/autoencoder wrappers must not be wired to `RULContextBatchDataset` unless the target/context sequencer lengths are explicitly validated.
- The experiment config's `override /transforms:` path is relative to `configs/transforms/` — e.g., `concepts_n_cmapss_ds02/depater2023_default` maps to `configs/transforms/concepts_n_cmapss_ds02/depater2023_default.yaml`
