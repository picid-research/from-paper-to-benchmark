---
name: verify-static
description: Run static checks and dataset executability preflight against a completed implementation blueprint, catching broken imports, missing files, wrong base classes, invalid configs, cross-file inconsistencies, and unavailable local datasets WITHOUT executing training. Use after implementation skills finish and before spending GPU time.
---

# /verify-static

Run the cheapest possible verification pass on a freshly-implemented paper.
This skill reads the implementation blueprint, identifies every file that should exist, and checks each one statically. It never trains a model. When an experiment config is available, it may run datasource/preprocessor preflight to decide whether the selected dataset can execute locally.

The goal is to surface broken imports, missing files, wrong base classes, invalid configs, cross-file inconsistencies, and dataset executability failures before any GPU time is spent.

## Input contract

The orchestrator must provide:

- **vault_dir**: Path to the vault containing `04-implementation-blueprint.json` and rendered `04-implementation-blueprint.md`
- **repo_root**: Path to the PICID repo root (where `picid/` and `configs/` live)
- **experiment_config**: current Hydra experiment config path, when available
- **dataset_candidate**: current dataset candidate name/config, when running dataset preflight

The skill reads `04-implementation-blueprint.json` first for required files, experiment config, validation context, and executable contracts. Use rendered `04-implementation-blueprint.md` only for longer audit prose such as "New files to create" tables or full config snippets when those are not sufficiently explicit in JSON.

## Procedure

Work through the checks below in order. Each check is cheap and independent — run all of them even if earlier checks fail, so the report gives a complete picture in one pass.

### Check 1 — File existence

Read `04-implementation-blueprint.json` for the implementation file list and experiment config, falling back to the rendered Markdown only when the JSON omits audit prose needed to disambiguate a path.
Build a list of every file path mentioned:
- Python files under `picid/`
- YAML configs under `configs/`
- Any `__init__.py` that needs updating

Verify each path exists on disk relative to `repo_root`.

```bash
test -f "{repo_root}/{file_path}" && echo "PASS" || echo "MISSING"
```

**Report**: list of expected files, present/missing status.

### Check 2 — Import resolution

For each NEW Python file (from the "New files to create" table), derive its module path and primary class name, then run:

```bash
uv run python -c "from {module_path} import {ClassName}; print('OK')"
```

This catches syntax errors, missing dependencies, and broken imports in one shot.

**Report**: file, import statement, pass/fail, error message if fail.

### Check 3 — Base class verification

For each NEW implementation file, verify it extends the correct base class as specified in the blueprint. The expected base classes are:

- **Data source loaders**: `AbstractDataSourceLoader` or `SingleSourceLoader`
- **Transforms**: `BaseTransform`
- **Model methods** (backbones): `nn.Module` subclass
- **Model wrappers**: `AbstractFeedForwardWrapper`, `AbstractFeedForwardTrainingWrapper`, or `AbstractFitPredictWrapper` (whichever the blueprint specified)
- **Loss functions**: `nn.Module` subclass
- **Metrics**: follow the metric interface

Use the MRO to verify:

```bash
uv run python -c "
from {module} import {Class}
import inspect
bases = [b.__name__ for b in inspect.getmro({Class})]
assert '{ExpectedBase}' in bases, f'Expected {ExpectedBase} in {bases}'
print('OK:', bases)
"
```

**Report**: file, class, expected base, actual MRO, pass/fail.

### Check 4 — Method signature verification

For each NEW implementation, verify the required methods exist with correct signatures.
Check against the base-class contracts in `.opencode/reference/` (`datasources.md`, `transforms.md`, `models.md`, `losses.md`, `evaluators.md`):

- **DataSourceLoader**: `load_data()`, `split_data()`, `get_data()`, `get_data_name()`
- **SingleSourceLoader**: `_load_data()` (the hook; parent provides the rest)
- **Transform**: `fit()`, `transform()`, `inverse_transform()`
- **Model wrapper**: has a `model` attribute, `forward()` exists
- **Loss**: `forward(pred, target)` exists

```bash
uv run python -c "
from {module} import {Class}
import inspect
for req in {required_methods_list}:
    assert hasattr({Class}, req), f'Missing method: {req}'
    sig = inspect.signature(getattr({Class}, req))
    print(f'{req}{sig}')
"
```

**Report**: class, method, signature found, pass/fail.

### Check 5 — Config YAML validation

For each NEW config file, verify:

1. YAML parses without error (`OmegaConf.load`)
2. Has a `_target_` field (unless it is a defaults-only composition file)
3. The `_target_` path resolves to an actual importable class
4. Required fields are present (check against the config patterns in `.opencode/reference/configs.md`)

```bash
uv run python -c "
from omegaconf import OmegaConf
import importlib
cfg = OmegaConf.load('{config_path}')
assert '_target_' in cfg, 'Missing _target_'
module_path, class_name = cfg._target_.rsplit('.', 1)
mod = importlib.import_module(module_path)
cls = getattr(mod, class_name)
print(f'Config OK: {cfg._target_} -> {cls}')
"
```

Use `OmegaConf.load()` here, not `hydra.compose()` — OmegaConf.load is a plain parse that does not try to resolve interpolations like `${paths.data_dir}`, which would fail outside a Hydra context.

**Report**: config file, `_target_`, importable (yes/no), error if any.

### Check 6 — Cross-file consistency (experiment config)

If an experiment config was created, verify the full composition:

1. Load the experiment config YAML
2. Check that every entry in the `defaults` list references a config file that actually exists under `configs/`
3. Check that datasource, model, transforms, loss, and evaluator configs all exist
4. Attempt a Hydra dry-run composition:

```bash
cd {repo_root} && uv run python -m picid.run experiment={experiment_name} paths=agent --cfg job 2>&1 | head -50
```

This prints the resolved config without running training. If it errors, the composition is broken.

The `--cfg job` flag must be run from `repo_root` because Hydra resolves config paths relative to the working directory.

**Report**: experiment config, composition status, errors if any.

### Check 7 — Datasource executability preflight

Run this check whenever the orchestrator is entering the experimenter flow and an experiment config exists. It is allowed to execute datasource loading and preprocessing, but it must not train a model.

If the dataset cannot be tested because the config does not yet identify a dataset, mark `SKIPPED`. If the dataset is configured but cannot execute on this machine, mark `DATASET_UNAVAILABLE` or `DATASET_EXECUTION_FAILED` instead of generic `FAIL`.

Use `DATASET_UNAVAILABLE` for:
- missing raw files or directories
- missing PHMD cache files
- invalid `paths.data_dir`
- download/cache/task retrieval errors

Use `DATASET_EXECUTION_FAILED` for:
- datasource `load_data()`/`split_data()` failures caused by malformed local data
- preprocessing failures before model or datamodule creation
- resource failures clearly caused by dataset size

```bash
uv run python -c "
import hydra
from hydra import compose, initialize_config_dir
from picid.data.preprocessing.preprocessor import PreProcessor
from picid.transforms.base.transform_manager import ConfigTransformManager

with initialize_config_dir(version_base='1.3', config_dir='{repo_root}/configs'):
    cfg = compose(config_name='run.yaml', overrides=['experiment={experiment_config}', 'paths=agent', 'logger=csv'])

loader = hydra.utils.instantiate(cfg.datasource)
manager = ConfigTransformManager(transforms_config=cfg.transforms)
preprocessor = PreProcessor(datasource=loader, transforms=manager)
preprocessor.pipeline(
    data_cache_path=None,
    data_library_part_path=None,
    transform_library_part_path=None,
    cache_preprocessed=False,
)
data = preprocessor.get_processed_data_dict(return_splits_on_first_level=True)
print('splits:', list(data.keys()))
print('train keys:', list(data['train'].keys()))
"
```

Verify shapes match what the blueprint specifies. Then instantiate the configured dataset/datamodule for at least one split and fetch one item or one batch. Check that every key in `task_definition.model.data_requirements.input_tensors` exists, that those keys satisfy `model_io_contract`, and that observed train/val/test units or condition filters satisfy `split_contract` when the paper specifies them.

**Report**: dataset candidate, datasource target, transform config, dataset class, command, status, shapes/keys found, split-contract result, model-I/O-contract result, and traceback excerpt if unavailable or failed.

## Validation

Before writing the report, verify:
- Every check produced a result (PASS, FAIL, SKIPPED, DATASET_UNAVAILABLE, or DATASET_EXECUTION_FAILED)
- Every FAIL includes the actual error message, not just "failed"
- The summary counts match the detail rows

## Report

Produce or append to `{vault_dir}/06-sanity-ladder-log.md`:

```markdown
## Static Integrity Check

**Timestamp**: {datetime}
**Blueprint**: [[04-implementation-blueprint]]

### Results

| Check | Target | Status | Detail |
|-------|--------|--------|--------|
| File existence | picid/model/methods/foo.py | PASS | exists |
| File existence | configs/model/foo.yaml | PASS | exists |
| Import | from picid.model.methods.foo import FooModel | PASS | |
| Base class | FooModel -> nn.Module | PASS | MRO: [FooModel, Module, object] |
| Base class | FooWrapper -> AbstractFeedForwardWrapper | PASS | |
| Methods | FooModel.forward | PASS | (self, x: Tensor) -> Tensor |
| Config | configs/model/foo.yaml | PASS | _target_ resolves |
| Experiment | configs/experiment/.../foo.yaml | PASS | Hydra composition OK |
| Dataset preflight | pronostia | DATASET_UNAVAILABLE | FileNotFoundError: Cache file not found... |

### Summary
- **Total checks**: {N}
- **Passed**: {N}
- **Failed**: {N}
- **Blocked (semantic mismatch)**: {N}
- **Skipped**: {N}
- **Overall STATUS**: PASS / BLOCK / FAIL

### Autofix Applied (syntactic only; empty if none)
| File | Category | Before | After |
|------|----------|--------|-------|
| {path} | trailing_whitespace / yaml_quoting / defaults_self_ordering / init_export | {snippet} | {snippet} |

### Blocked (semantic mismatches — MUST be resolved upstream, not here)
| Category | Blueprint Origin | Offending Reference | Observed State |
|----------|------------------|---------------------|----------------|
| BLUEPRINT_REFERENCES_MISSING_CONFIG | Section 6 defaults, line 4: `- override /optimization: adam` | `configs/optimization/adam.yaml` | file not found |

### Failures (if any)
{For each failure: what failed, the full error message, and a suggested fix}

### Dataset Executability
- **Current dataset candidate**: {dataset_candidate}
- **Status**: PASS / SKIPPED / DATASET_UNAVAILABLE / DATASET_EXECUTION_FAILED
- **Dataset recovery required**: yes / no
- **Reason**: {short reason}
- **Command**: `{command}`
- **Traceback excerpt**:
```text
{error_excerpt}
```
```

## Autofix Policy

This skill is allowed to apply a narrow class of **syntactic** fixes while verifying, and **MUST NOT** apply **semantic** fixes. The distinction is enforced to prevent the verifier from silently papering over blueprint–reality mismatches that should halt the workflow.

**Syntactic autofix (allowed)** — fixes that cannot change the meaning of a valid config:

- trailing whitespace, missing trailing newline, tab-to-space normalization inside YAML
- YAML quoting/unquoting that preserves the parsed value (e.g., `'true'` → `true` only when the field is typed as bool)
- re-ordering the `defaults:` list so `_self_` appears in the canonical position, without adding or removing entries
- adding a missing `__init__.py` export line when the only failure is an import from a package that does exist on disk

When a syntactic autofix is applied, it MUST be listed in the report under `autofix_applied:` with the file path, original fragment, new fragment, and category.

**Semantic autofix (forbidden)** — any edit that changes, redirects, or substitutes a referenced identifier:

- renaming or rewriting a `defaults:` entry (e.g., `- override /optimization: adam` → `- override /optimization: default`)
- changing a `_target_` value to a different importable class
- swapping, removing, or adding a defaults-list entry that is not `_self_` reordering
- inventing or choosing a substitute config file when the referenced one is missing
- editing hyperparameter values inside a referenced config

When the verifier observes a condition that would require a semantic fix, it MUST emit `STATUS = BLOCK` with one of the following categories and stop (no edit is made):

- `BLUEPRINT_REFERENCES_MISSING_CONFIG` — a defaults entry or `_target_` points at a file or class that does not exist under `configs/` or `picid/`
- `BLUEPRINT_REFERENCES_INVALID_TARGET` — the referenced class exists but fails import or does not subclass the expected base
- `BLUEPRINT_CONFIG_CONTRADICTION` — two referenced configs make mutually inconsistent claims (e.g., optimizer config vs. scheduler config assume different base optimizers)
- `BLUEPRINT_DATASET_CONTRACT_MISMATCH` — composed datasource/transforms/dataset/model I/O or split membership does not satisfy `dataset_contract`, `model_io_contract`, or `split_contract`

Each BLOCK must record the originating blueprint section/field (e.g., "Section 6 defaults, line 4: `- override /optimization: adam`") so the experimenter can route back to the correct upstream implementation skill. The experimenter resolves the block by either creating the missing config via the upstream skill or re-invoking the implementation skill with a corrected spec — never by asking the static verifier to choose a substitute.

## Behavior on failure

This skill does NOT fix implementation code. Beyond the narrow syntactic autofixes above, it reports failures with enough diagnostic information that the orchestrator can decide what to do:
- Re-invoke the implementation skill that produced the broken file
	- Trigger same-dataset recovery when the datasource preflight reports `DATASET_UNAVAILABLE` or `DATASET_EXECUTION_FAILED`
- Ask the user for help if the failure is ambiguous

A clean report (zero implementation failures, no forbidden-semantic-fix BLOCKs, and dataset preflight `PASS` or intentionally `SKIPPED`) means the implementation is safe to proceed to runtime validation or training.

If dataset preflight returns `DATASET_UNAVAILABLE` or `DATASET_EXECUTION_FAILED`, the orchestrator should repair same-dataset wiring/path/cache issues when possible, then rerun this verifier. It must not switch to another dataset. This is not a model implementation failure unless the failure is caused by paper-authored code.

## Common pitfalls

- **OmegaConf interpolations**: Expressions like `${paths.data_dir}` will fail to resolve outside a Hydra context. Use `OmegaConf.load()` for basic parsing of individual configs; only use `--cfg job` (Check 6) for full resolution.
- **Optional dependencies**: Some imports may fail because of optional packages (e.g., `tabpfn`, `autogluon`). If the error is an `ImportError` for a known optional dependency, mark the check as SKIPPED, not FAILED.
- **Hydra dry-run working directory**: The `--cfg job` command must be run from `repo_root`, not from a subdirectory.
- **Config defaults syntax**: Defaults list entries use a specific syntax (`- /group: name` or `- /group@package: name`). When checking existence in Check 6, strip the leading `/` and `@package` suffix to derive the actual file path under `configs/`.
- **`__init__.py` exports**: New modules may not be importable if the parent package's `__init__.py` was not updated. Check 2 will catch this — the suggested fix is to add the import to `__init__.py`.
- **Composition-only configs**: Some experiment configs have no `_target_` of their own — they only compose other configs via `defaults`. Skip the `_target_` assertion for these files in Check 5.
