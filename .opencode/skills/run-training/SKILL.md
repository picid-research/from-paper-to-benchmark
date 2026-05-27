   ---
name: run-training
description: Execute a full training run using the PICID/PICID CLI after sanity is accepted, monitor training health via CSV metrics, and classify success, model/training failure, or dataset executability failure. Use after verify-sanity returns PASS or WARN_CONTINUE and before result analysis.
---

# /run-training

Execute the full training run, monitor its health, and report whether it succeeded. This skill assumes static checks (`/verify-static`) passed and sanity checks (`/verify-sanity`) were accepted (`PASS` or `WARN_CONTINUE`).

If training cannot start or crashes before model-specific training because the selected dataset cannot execute on the current machine, classify the run as `DATASET_UNAVAILABLE` or `DATASET_EXECUTION_FAILED`. Do not treat that as a model training failure.

## Input contract

The orchestrator must provide:

- **vault_dir**: Path to vault (reads blueprint; writes to `07-training-log.md`)
- **experiment_config**: Hydra experiment config path (e.g., `pronostia/ptics/my_model`)
- **repo_root**: PICID repo root
- **model_paradigm**: `feedforward` | `training` | `fit_predict`
- **expected_epochs**: From blueprint/paper as the full-run ceiling when specified (e.g., 100); `NOT_SPECIFIED` means use the resolved trainer config default.
- **expected_metrics**: Dict of metric_name to expected_value from blueprint Section 7 (e.g., `{"RMSE": 12.5, "NASA_score": 0.85}`)
- **gpu_available**: Boolean
- **run_mode**: `full` | `quick`
- **training_stage**: `preflight_1epoch` | `full`. Full-mode orchestration must run `preflight_1epoch` first, repair it until it passes, then run `full`.
- **hp_profile**: `reference` | `paper` (default `reference`). Every validation profile forces datamodule batch sizes to train `512`, val `1024`, and test `1024`. `reference` is the primary validation profile: LR scheduler/check cadence is preserved from PICID, and model-specific HPs come from the paper or explicit imputations in the experiment config. `paper` uses the experiment config's paper/default non-batch HPs for auxiliary context.
- **hp_overrides**: optional Hydra override strings when a keyed official entry exists. Do not include datamodule batch-size overrides here; the training tool appends the fixed validation batch after any other overrides. For novel paper models this is commonly empty; the composed experiment config is authoritative for non-batch model/scheduler values.
- **hp_reference_source**: script/report path or note proving where `hp_overrides` came from. For novel models without keyed overrides, use a note such as `fixed validation batch 512/1024/1024 + composed PICID scheduler cadence + paper/imputed model HPs`.
- **dataset_candidate**: current dataset candidate name/config
- **epoch_budget_rationale**: REQUIRED free-text string of minimum 50 characters explaining why the chosen epoch ceiling is appropriate for THIS run — must reference the paper's `max_epochs`, the scale of the selected framework dataset, and whether early stopping is expected to terminate earlier. Missing, empty, or shorter than 50 characters → the skill refuses invocation with `MISSING_EPOCH_BUDGET_RATIONALE`.

## Normal Path

Invoke the OpenCode custom tool:

```text
PICID_training_run
```

Pass the full input contract payload. The tool enforces the sanity/fixed-batch fit gates, runs the actual `picid.run` command, writes `07-training-log.md`, appends `07-training-log.jsonl`, and returns structured JSON for the experimenter/workflow runner.

For `run_mode=full`, `training_stage=preflight_1epoch` writes `07-training-log.preflight.md` and `07-training-log.preflight.jsonl`; `training_stage=full` writes canonical `07-training-log.md` and `07-training-log.jsonl`. For `run_mode=quick`, the 1-epoch smoke result remains canonical so the existing quick-mode artifact gate still works.

Use manual shell commands only for debugging the backend itself.

## Step 0 — Preflight gates (REQUIRED; skill refuses to proceed unless all pass)

These gates run before any training command is constructed. They are mechanical — the skill refuses invocation rather than warning. The orchestrator must resolve the blocking condition and re-invoke.

### Gate 0.1 — Sanity acceptance gate

Parse the most recent structured sanity attempt from `{vault_dir}/06-sanity-ladder-log.jsonl` first. If the JSONL sidecar is absent, fall back to the most recent YAML front-matter block at `{vault_dir}/06-sanity-ladder-log.md` (produced by `/verify-sanity`). Take the last attempt (highest `attempt:` number).

- If the file does not exist: refuse with `SANITY_LOG_MISSING`.
- If neither the JSONL sidecar nor a parseable markdown attempt exists: refuse with `SANITY_LOG_UNPARSEABLE`. The orchestrator must re-run `/verify-sanity` or backfill the sidecar.
- If the latest attempt's `verdict` is `PASS`: proceed.
- If the latest attempt's `verdict` is `WARN_CONTINUE`: proceed, and preserve the authoritative `failed_checks` / `check_diagnostics` as sanity warnings in the training/evaluation narrative.
- If the latest attempt uses legacy `INVESTIGATE`: treat it like `WARN_CONTINUE`.
- For any other `verdict`: refuse with `SANITY_NOT_PASSED` and quote the authoritative `verdict`, `failed_checks`, and relevant `check_diagnostics`. The orchestrator must resolve the sanity block (typically via `/diagnose-verify-block`) before re-invoking this skill.

### Gate 0.2 — Fixed-batch fit-check gate (read-only)

Every full run (and every quick run on GPU) must be backed by a cached fixed-batch fit check keyed on `(resolved_config_hash, hardware_id)`. Compute:

- `resolved_config_hash`: sha256 of the resolved Hydra config for `experiment_config` after applying any selected `hp_profile` non-batch overrides plus the fixed validation batch overrides. For the reference profile, keyed non-batch overrides are optional; when absent, the composed experiment config plus `512/1024/1024` is hashed.
- `hardware_id`: compact string of `torch.cuda.get_device_name(0)` when `gpu_available=true`, else `cpu`.

Read `{vault_dir}/batch_fit_check.json` (a JSON array of fit-check records). If the latest record with matching `resolved_config_hash` AND `hardware_id` has `status: OK`, proceed.

If no matching OK record exists, refuse with `BATCH_FIT_CHECK_UNAVAILABLE` or `BATCH_DOES_NOT_FIT`. This skill does **not** run the fit check itself — the orchestrator must invoke `/check-batch-fit` first (once after sanity is accepted, and again after any event that changes `resolved_config_hash`, e.g. same-dataset recovery).

### Gate 0.3 — Epoch-budget-rationale gate

If `epoch_budget_rationale` is missing or `< 50` characters, refuse with `MISSING_EPOCH_BUDGET_RATIONALE`. The full rationale text MUST appear in the written `07-training-log.md`.

### Gate 0.4 — Always-write-training-log contract

Register a try/finally handler at the very start of the skill body. `{vault_dir}/07-training-log.md` MUST be written (or updated) at every skill exit, including unhandled crash. The report's front-matter `status:` field takes one of:

- `SUCCESS` — run completed, metrics collected, passed health checks
- `PARTIAL` — run completed but with health warnings (e.g., quick-mode underfitting, mild overfitting)
- `FAILED` — run completed but produced an unrecoverable training failure (NaN loss, metrics far from expected for full mode, etc.)
- `CRASHED` — Python traceback or OS-level failure prevented completion; traceback excerpt included
- `DATASET_UNAVAILABLE` — selected dataset absent/unavailable on this machine
- `DATASET_EXECUTION_FAILED` — datasource load/split/preprocessing failure on this machine

If the skill cannot collect any useful output before crashing, it still writes `07-training-log.md` with `status: CRASHED`, the invoking arguments, and the traceback. The orchestrator never needs to infer a missing log.

Required front-matter schema for `07-training-log.md`. **Values below are placeholders** — the fields are mandatory, the specific numbers are not. Fill in what this run actually produced; do not copy the example values.

```yaml
---
status: <SUCCESS | PARTIAL | FAILED | CRASHED | DATASET_UNAVAILABLE | DATASET_EXECUTION_FAILED>
experiment_config: <hydra path>
dataset_candidate: <name>
hp_profile: <reference | paper>
hp_profile_status: <KEYED_REFERENCE_OVERRIDES_WITH_FIXED_VALIDATION_BATCH | FIXED_VALIDATION_BATCH_WITH_PICID_SCHEDULER_AND_PAPER_OR_IMPUTED_MODEL_HPS | PAPER_OR_DEFAULT_PROFILE_WITH_FIXED_VALIDATION_BATCH>
is_primary_comparison: <bool>
hp_overrides:
  - <dataset-resolved non-batch Hydra override when available>
hp_reference_source: <script/report path or null>
run_mode: <full | quick>
training_stage: <preflight_1epoch | full>
max_epochs_ceiling: <int>
epochs_run: <int>
early_stopping_triggered: <bool>
early_stopping_epoch: <int | null>          # null if early stopping did not fire
batch_size_paper: <int | str | null>        # legacy compatibility field; normally null for validation runs because the fixed batch overrides paper batch
batch_size_chosen: <int | str>              # what actually ran; equals the configured train_batch_size
batch_probe_source: not_used                # legacy compatibility field; automatic probing is no longer used
batch_fit_check_source: <reused | fresh>
batch_size_configured: <int | str>
lr_paper: <float | null>                    # from blueprint §8.1; null if NOT_SPECIFIED
lr_scaled: <float>                          # legacy compatibility field; equals lr_paper because runtime autoscaling is disabled
lr_scaling_rule: none
optimizer_family: <sgd_family | adam_family | other>   # determines the scaling rule per policies.md
resolved_config_hash: <sha256 prefix>
hardware_id: <cuda-device-name | cpu>
epoch_budget_rationale: |
  <full rationale, ≥ 50 chars>
monitor_status: <not_started | running | completed | stopped | failed | exited>
stop_reason: <string | null>
monitor_events:
  - <compact lifecycle/stop event dict; no full logs or full loss arrays>
monitor_probe_summary:
  probe_count: <int>
  last_probe:
    epochs_seen: <int>
    train_loss: <float | null>
    val_loss: <float | null>
    checkpoint_created: <bool>
checkpoint_created: <bool>
test_stage_completed: <bool>
output_dir: <path | null>
metrics_csv: <path | null>
loss_curve_plot: <path | null>             # run-directory plot, if generated
vault_loss_curve_plot: <path | null>       # {vault_dir}/plots/loss_curves.png, if generated
traceback_excerpt: <string | null>          # non-null only when status=CRASHED or FAILED with an exception
timestamp: <ISO8601>
---
```

### Structured sidecar contract

After writing or updating `07-training-log.md`, append the same authoritative machine-readable payload to `{vault_dir}/07-training-log.jsonl` via `validate_paper_workflow_append_training_result`.

The JSONL object must include the same fields as the front-matter:
- `status`
- `experiment_config`
- `dataset_candidate`
- `hp_profile`
- `hp_profile_status`
- `is_primary_comparison`
- `hp_overrides`
- `hp_reference_source`
- `run_mode`
- `training_stage`
- `max_epochs_ceiling`
- `epochs_run`
- `early_stopping_triggered`
- `early_stopping_epoch`
- `batch_size_paper`
- `batch_size_chosen`
- `batch_probe_source`
- `batch_fit_check_source`
- `batch_size_configured`
- `lr_paper`
- `lr_scaled`
- `lr_scaling_rule`
- `optimizer_family`
- `resolved_config_hash`
- `hardware_id`
- `epoch_budget_rationale`
- `monitor_status`
- `stop_reason`
- `monitor_events`
- `monitor_probe_summary`
- `checkpoint_created`
- `test_stage_completed`
- `output_dir`
- `metrics_csv`
- `loss_curve_plot`
- `vault_loss_curve_plot`
- `traceback_excerpt`
- `timestamp`

`07-training-log.md` remains the narrative report; `07-training-log.jsonl` is the control-plane artifact for the workflow runner.

## Before you start

### Step 0 — Read context

1. Read `.opencode/reference/framework.md` (pipeline overview) and `.opencode/reference/patterns.md` (caching, inverse scaling, full experiment composition). If you need base-class detail, load the specific contract file from `.opencode/reference/` (`datasources.md`, `transforms.md`, `models.md`, `evaluators.md`).
2. Read `logging-guide.md` — understand where outputs land and how to read metrics
3. Read `{vault_dir}/04-implementation-blueprint.json` for the experiment config and verification protocol. Use rendered `04-implementation-blueprint.md` only for longer human-readable run-command detail if needed.
4. If `06-sanity-ladder-log.md` exists in the vault, read it to confirm sanity was accepted. Proceed on `PASS` or `WARN_CONTINUE`; stop only on categorical sanity blockers or dataset/tool failures.

### Step 0b — Choose the epoch budget

Before forming the command, determine the epoch ceiling from:

1. the paper/blueprint intent (`expected_epochs`)
2. dataset scale on the selected framework dataset (samples, windows, batches per epoch)
3. whether early stopping is enabled in the composed framework config, including its monitor and patience
4. whether this is a quick validation run or a full validation run

Use these rules:

- Treat `trainer.max_epochs` as an upper bound, not the expected actual runtime.
- For `training_stage=preflight_1epoch`, use `trainer.max_epochs=1` exactly.
- For `training_stage=full`, use the paper's `max_epochs` when specified. If the paper says `NOT_SPECIFIED`, do not add a runtime `trainer.max_epochs` override; use the resolved trainer config default.
- Full training must not use wall-clock timeout stops. The monitor may stop only for concrete instability/failure, or for a documented plateau when early stopping is absent or not firing.
- Record the rationale for the chosen budget in `07-training-log.md`.

### Step 0c — Fixed-batch fit check lives in `/check-batch-fit`

The fit-check body is owned by `.opencode/skills/check-batch-fit/SKILL.md` and is the responsibility of the orchestrator: invoke `/check-batch-fit` once after sanity is accepted, and again after any event that changes `resolved_config_hash` (same-dataset recovery, experiment re-implementation). This skill only consumes the cached OK result via Gate 0.2.

If a run OOMs even though the fixed-batch check passed, the experimenter re-invokes `/check-batch-fit` once with `force_fresh=true` to classify the hardware/config state. Do not halve or otherwise tune the batch size automatically.

### Step 0d — Enforce fixed validation batch, preserve scheduler, and do not autoscale

`/run-training` must use the validation profile's fixed datamodule batch sizes (`train=512`, `val=1024`, `test=1024`), plus the resolved experiment config's LR, optimizer, and scheduler. It must not pass `datamodule.*batch_size` overrides from a probe and must not scale LR at runtime. For the primary validation profile, fair framework validation comes from forcing `512/1024/1024`, preserving PICID's LR scheduler/check cadence in the composed config, then using paper-stated or explicitly imputed model-specific HPs. Do not refuse merely because no pre-existing official model-specific reference profile exists for a newly implemented paper model, and do not refuse because the base datamodule default is `16`. Refuse only if the composed config lacks identifiable PICID scheduler/check cadence (`PICID_SCHEDULER_UNRESOLVED`) or no matching fixed-batch fit check exists.

**Record in `07-training-log.md`:**

- Front-matter: `hp_profile`, `hp_profile_status`, `is_primary_comparison`, `hp_overrides`, `hp_reference_source`, `batch_size_configured`, `batch_size_chosen`, `batch_fit_check_source`, `lr_paper`, `lr_scaled`, `lr_scaling_rule`.
- Body section "Configured optimization profile" records the resolved batch/LR/scheduler values and notes that no runtime tuning was applied.

## Procedure

### Step 1 — Prepare the run command

The experiment config should already exist (created by the implementation skills). Construct the base command:

```bash
cd {repo_root}
uv run python -m picid.run experiment={experiment_config} \
  paths=agent \
  logger=csv \
  seed=42
```

Always use `paths=agent` for validation runs so datasets and reusable caches resolve to the shared workspace mount while artifacts stay inside the per-run slug directory. Always use `logger=csv` because WandB may not be configured and will cause the run to error. The CSV logger writes metrics to `csv_logs/version_*/metrics.csv` inside the output directory; older artifacts may use `logs/csv_logs/version_*/metrics.csv`. Accept both layouts for monitoring and post-run analysis.

Do not add a runtime `trainer.precision=...` override by default. Let the composed framework config decide unless the blueprint or the user explicitly requires a precision mode.

If a precision override is used, it must come from an explicit source:
- the paper requires it,
- the user requests it, or
- a prior failed run established that the framework default must be overridden for a concrete, recorded reason.

Add overrides based on context:

**For `training_stage=preflight_1epoch` or `run_mode=quick`** — 1-epoch pipeline gate:

- `trainer.max_epochs=1` exactly
- limit training batches aggressively (`trainer.limit_train_batches` small fraction or small integer)
- do NOT zero out validation batches — use `trainer.limit_val_batches` small-but-non-zero so validation-pipeline bugs surface
- do NOT zero out test batches — use `trainer.limit_test_batches` small-but-non-zero so test/evaluator bugs surface
- keep checkpointing enabled; checkpoint creation is part of the gate
- use the fixed validation batch sizes from the selected validation profile; `/check-batch-fit` already verified them

This stage is a precondition for full training in `run_mode=full`. Do not mark it `FAILED` on metric quality — only on hard failures or missing train/val/test/checkpoint execution.

**For `training_stage=full`** — monitored paper-scale run:

- set `trainer.max_epochs` to the paper value when specified; if `NOT_SPECIFIED`, omit this override and use the resolved trainer default
- use the configured train/val/test batch sizes from the selected `hp_profile`; for every validation profile these must be exactly `512/1024/1024`, not the paper batch size and not the base datamodule default `16`
- use the configured LR, optimizer, and warmup from the selected `hp_profile`; for `reference`, these come from paper-stated values or explicit framework/default imputations in the experiment config
- use the configured scheduler/check cadence from PICID's framework config for the primary validation profile; do not silently replace it with a paper scheduler
- do not add artificial batch limits
- early stopping is the primary termination signal; record `early_stopping_triggered` in the front-matter regardless
- launch the subprocess under the monitor loop; do not wrap it in a wall-clock timeout
- if loss plateaus and early stopping is absent or not firing, gracefully terminate and mark `PARTIAL` with `stop_reason: plateau_graceful_stop`

**For `gpu_available=false`**:
```
trainer.accelerator=cpu
trainer.devices=1
```

**For `model_paradigm=fit_predict`**:
No trainer overrides. Fit-predict models (XGBoost, TabPFN, AutoGluon, etc.) do not use the Lightning trainer loop. The run still goes through `picid.run` but the pipeline handles fit/predict internally. Monitoring steps 3a-3c below do not apply — skip to Step 4.

### Step 2 — Execute the training run

Run the command through the tool's monitored subprocess wrapper. Do not apply a wall-clock timeout to full training. Do not tee or redirect stdout/stderr to a vault file — the framework already writes a complete log to `{OUTPUT_DIR}/{experiment_group}.log`, so a vault copy is redundant and bloats the vault.

```bash
cd {repo_root} && uv run python -m picid.run experiment={experiment_config} \
  paths=agent \
  logger=csv \
  seed=42 \
  [additional overrides]
```

The output directory follows this pattern:
```
{repo_root}/artifacts/{experiment_group}/runs/{job_name}/{YYYY-MM-DD}_{HH-MM-SS}/
```

Find it after the run completes:
```bash
OUTPUT_DIR=$(ls -td {repo_root}/artifacts/*/runs/*/* 2>/dev/null | head -1)
```

### Step 3 — Monitor training health

For `training_stage=full`, poll the metrics CSV, framework log, output directory, and checkpoints while the subprocess is running. For `training_stage=preflight_1epoch`, use a shorter polling cadence and verify train/val/test/checkpoint execution after the subprocess exits.

The monitor must perform punctual probes only. Do not write full logs, stdout/stderr streams, full metrics CSV contents, or full loss arrays into `07-training-log.md` or JSONL sidecars. Keep full artifacts in the Hydra output directory and record only compact summaries: probe count, latest train/val loss, epochs seen, checkpoint/test booleans, stop reason, and a bounded lifecycle event list.

#### Step 3a — Check for crashes or errors

Read the framework log file at `{OUTPUT_DIR}/{experiment_group}.log`. Look for:

- Python tracebacks (FAIL — stop the subprocess and classify)
- CUDA out of memory errors (FAIL — stop the subprocess, re-run `/check-batch-fit` once with `force_fresh=true`; do not auto-reduce batch size)
- NaN/Inf loss warnings (FAIL — stop the subprocess; numerical instability)
- Early stopping triggered (INFO — note at which epoch)

If the run crashed, classify the crash before stopping:
- `DATASET_UNAVAILABLE`: missing raw files/directories, missing PHMD cache files, invalid `paths.data_dir`, failed PHMD download/cache/task retrieval, or similar local availability failures.
- `DATASET_EXECUTION_FAILED`: datasource load/split/preprocessing failures caused by malformed local data, or persistent dataset-size resource failure before stable model training starts.
- `FAILED`: implementation, model, optimizer, metric, or evaluator failures.

For `DATASET_UNAVAILABLE` or `DATASET_EXECUTION_FAILED`, record the dataset candidate, command, traceback excerpt, and `Dataset recovery required: yes`, then stop. The experimenter owns same-dataset recovery or reporting a named blocker.

#### Step 3b — Read training metrics

Parse `{OUTPUT_DIR}/csv_logs/version_*/metrics.csv`, falling back to `{OUTPUT_DIR}/logs/csv_logs/version_*/metrics.csv` for legacy artifacts. Prefer the newest `version_*` when more than one exists. Key columns:

- `train/loss_epoch` or `train_loss_epoch` — training loss per epoch (should decrease)
- `val/loss` or `val_loss` — validation loss per epoch (should decrease, then may plateau)
- Task-specific metrics vary by evaluator config (e.g., `val/mae`, `val/rmse`, `test/mae`)

Not every column is populated in every row — Lightning logs train and val metrics on different rows. Filter for non-empty values when reading a column.

Extract:
- **Training loss curve**: all `train/loss_epoch` or `train_loss_epoch` values
- **Validation loss curve**: all `val/loss` or `val_loss` values
- **Final test metrics**: the last row's test metric columns (populated after `trainer.test()`)
- **Best checkpoint epoch**: the epoch where the validation loss column was lowest
- **Early stopping epoch**: if early stopping triggered, which epoch

#### Step 3c — Assess training health

Apply these checks to the extracted metrics:

**Loss convergence**:
- Training loss should show an overall decreasing trend
- If training loss is flat for >30% of total epochs → WARNING (learning rate may be too low)
- If training loss increases over the last 20% of epochs → WARNING (possible overfitting or instability)
- If training loss is NaN at any point → FAIL

**Overfitting detection**:
- Compare final training loss vs final validation loss
- If train_loss < 0.1 * val_loss (train is 10x better) → WARNING: severe overfitting
- Mild overfitting (train moderately better than val) is normal and expected

**Early stopping**:
- If early stopping triggered very early (< 30% of expected epochs) → WARNING: model may be underfitting or LR too high
- If early stopping never triggered and all epochs ran → INFO: normal

### Step 4 — Evaluate final metrics

Compare the final test metrics against `expected_metrics` from the blueprint. **These are health signals, not a pass/fail gate.** Paper-validation success is judged on relative positioning against framework baselines by `/evaluate-results` against the pre-registered hypothesis in `05-paper-hypothesis.md`, not on absolute metric match with `expected_metrics`.

For each expected metric:
- Read the corresponding value from the last test row in `metrics.csv`
- Compare against the expected value
- Tolerance: within 20% of expected (papers often report best-of-N or cherry-picked results, so exact match is unrealistic)

Categorize each metric as:
- **MEETS** — within 20% of expected
- **CLOSE** — within 50% of expected
- **FAR** — more than 50% off
- **MISSING** — metric not found in CSV (wrong evaluator config?)

A FAR classification does NOT by itself trigger a `FAILED` status — for `framework_validation` runs the absolute numbers are expected to diverge due to dataset, seed, or validation-profile differences. `FAR` is reported as a health warning alongside loss-curve quality; the orchestrator decides whether to treat it as actionable based on the claim verdicts from `/evaluate-results`. Set `FAILED` only for genuine training-level failures: NaN loss, flat-line training loss over the full run, or exploding gradient trajectories that never recover.

For `run_mode=quick`: Metrics will be worse than expected because we used fewer epochs and less data. Note this in the report — the purpose of quick mode is to verify the pipeline runs end-to-end, not to hit target metrics.

### Step 5 — Collect artifacts

Record the locations of all training artifacts:

```
{OUTPUT_DIR}/
├── csv_logs/version_*/metrics.csv        # All scalar metrics with current CSV logger config
├── logs/csv_logs/version_*/metrics.csv   # Legacy accepted metrics location, if present
├── {experiment_group}.log                 # Full log
├── config_resolved.yaml                  # Resolved config
├── checkpoints/                          # Best + last checkpoints
│   ├── checkpoints/epoch{N}-val_loss{V}.ckpt  # Default callback dirpath
│   └── hparams.json                           # Config snapshot
├── eval_details/                         # Prediction arrays (if save_predictions=True)
│   ├── train/predictions.nc
│   ├── val/predictions.nc
│   └── test/predictions.nc
├── plots/                                # Evaluation plots (including loss_curves.png)
├── REPRODUCE.md                          # Reproduction guide
└── run_metadata.yaml                     # Git state
```

Check which of these actually exist — not all runs produce all artifacts (e.g., `eval_details/` requires `evaluator.save_predictions=True`).

### Step 5b — Generate loss curve plots

If `metrics.csv` exists and `model_paradigm != fit_predict`, generate a loss curve PNG:

```bash
uv run python {repo_root}/.opencode/tools/plot_training.py \
  --metrics_csv {METRICS_CSV} \
  --output_path {OUTPUT_DIR}/plots/loss_curves.png \
  [--early_stopping_epoch {early_stopping_epoch}] \
  --title "{experiment_config} — Loss Curves"

uv run python {repo_root}/.opencode/tools/plot_training.py \
  --metrics_csv {METRICS_CSV} \
  --output_path {vault_dir}/plots/loss_curves.png \
  [--early_stopping_epoch {early_stopping_epoch}] \
  --title "{experiment_config} — Loss Curves"
```

- Include `--early_stopping_epoch` only when early stopping triggered.
- `{METRICS_CSV}` is the discovered newest CSV metrics file from either accepted logger layout.
- The run-directory plot is the canonical training artifact; the vault plot is the workflow-level artifact used by `08-evaluation-report.md` and final summaries.
- If the script prints `PLOT_OK: ...`, record both paths for the report.
- If either plot prints `PLOT_FAILED: ...`, note the reason in the training log but do **not** abort the skill.

## Report

Write to `{vault_dir}/07-training-log.md`. The file MUST start with the YAML front-matter block defined in Gate 0.4 (containing `status`, `epochs_run`, `batch_size_chosen`, `resolved_config_hash`, `epoch_budget_rationale`, etc.). The human-readable markdown below is appended after the front-matter. If the skill crashes before metrics can be collected, the front-matter is still written with `status: CRASHED` and a traceback — the markdown body may be abbreviated or absent in that case.

```markdown
# Training Run Log

**Timestamp**: {datetime}
**Experiment config**: {experiment_config}
**Model paradigm**: {model_paradigm}
**Run mode**: {run_mode}
**Blueprint**: [[04-implementation-blueprint]]

## Run Configuration
- **Command**: `{full command with all overrides}`
- **Seed**: 42
- **Max epochs**: {value}
- **Epoch-budget rationale**: {paper reference, dataset scale, and early-stopping rationale}
- **Configured batch size**: {value}
- **Batch fit check**: OK / FAILED with `{vault_dir}/batch_fit_check.json` record
- **Chosen run batch size**: same as configured train batch size
- **Batch-size rationale**: dataset-resolved configured/reference profile value; no runtime tuning
- **GPU**: {yes/no}
- **Output directory**: `{OUTPUT_DIR}`

## Configured Optimization Profile

| Field | Configured value | Runtime value | Notes |
|-------|------------------|---------------|-------|
| train_batch_size | {configured} | {configured} | verified by `/check-batch-fit` |
| val_batch_size | {configured} | {configured} | verified by `/check-batch-fit` |
| test_batch_size | {configured} | {configured} | verified by `/check-batch-fit` |
| learning_rate | {configured} | {configured} | no runtime scaling |
| scheduler | {configured} | {configured} | no runtime substitution |

## Run Status
- **Completed**: yes / no (crashed)
- **Total epochs run**: {N}
- **Early stopping**: triggered at epoch {N} / not triggered
- **Early stopping config**: monitor={metric}, patience={value}, enabled={yes/no}
- **Wall time**: {if available from log}
- **Status category**: SUCCESS / PARTIAL / FAILED / DATASET_UNAVAILABLE / DATASET_EXECUTION_FAILED
- **Dataset candidate**: {dataset_candidate}
- **Dataset recovery required**: yes / no
- **Error**: {if crashed, the traceback}

## Training Health
| Indicator | Status | Detail |
|-----------|--------|--------|
| Loss convergence | OK / WARNING / FAIL | {description} |
| Overfitting | OK / WARNING | train_loss={V} vs val_loss={V} |
| NaN/Inf | OK / FAIL | {description} |
| Early stopping | INFO | {epoch or "ran all epochs"} |

## Final Metrics

| Metric | Expected | Achieved | Status | Delta |
|--------|----------|----------|--------|-------|
| {name} | {expected} | {actual} | MEETS/CLOSE/FAR/MISSING | {±%} |

### Loss Curves (summary)
- **Initial train loss**: {value}
- **Final train loss**: {value}
- **Best val loss**: {value} (epoch {N})
- **Final val loss**: {value}

![Loss Curves](plots/loss_curves.png)

## Artifacts
| Artifact | Path | Exists |
|----------|------|--------|
| Metrics CSV | {path} | yes/no |
| Loss curve plot (vault) | {vault_dir}/plots/loss_curves.png | yes/no |
| Loss curve plot (run directory) | {OUTPUT_DIR}/plots/loss_curves.png | yes/no |
| Best checkpoint | {path} | yes/no |
| Predictions (test) | {path} | yes/no |
| Resolved config | {path} | yes/no |
| Reproduction guide | {path} | yes/no |

## Verdict
- **Status**: SUCCESS / PARTIAL / FAILED / DATASET_UNAVAILABLE / DATASET_EXECUTION_FAILED
- **Reasoning**: {why this verdict}
- **Next steps**: {what to do next}
```

**Verdict rules**:
- **SUCCESS** — run completed, no health warnings, metrics MEET or are CLOSE to expected
- **PARTIAL** — run completed but with health warnings or metrics are FAR from expected. For `run_mode=quick`, PARTIAL is expected and acceptable.
- **FAILED** — run crashed, produced NaN, or metrics are completely wrong
- **DATASET_UNAVAILABLE** — selected dataset cannot be found or retrieved on this machine; orchestrator should repair same-dataset path/cache/config or stop
- **DATASET_EXECUTION_FAILED** — selected dataset loads/preprocesses unsuccessfully on this machine; orchestrator should repair same-dataset wiring or stop

## Behavior on failure

The skill does NOT fix anything. It reports with enough detail for the orchestrator to decide:

- **DATASET_UNAVAILABLE / DATASET_EXECUTION_FAILED** → attempt same-dataset recovery or report a named blocker; do not switch datasets
- **Crash / OOM** → re-check fixed-batch fit, inspect GPU memory, then escalate or require an explicit alternate profile if the fixed validation batch does not fit
- **NaN loss** → revisit model initialization, check data normalization, add gradient clipping
- **Metrics far from expected** → check hyperparameters against paper, verify data preprocessing matches paper
- **Severe overfitting** → add regularization, reduce model capacity, increase data
- **Underfitting** → increase epochs, increase LR, increase model capacity

## Common pitfalls

- Always use `logger=csv` — WandB may not be configured and will cause the run to error out immediately.
- The output directory is timestamped and changes each run. Always find the latest one after the command completes rather than hardcoding a path.
- `metrics.csv` has sparse rows — training metrics and validation metrics appear on different rows. Filter for non-empty values when extracting a specific column.
- For `fit_predict` models, there is no training loop to monitor. The model fits once and predicts — Steps 3a-3c do not apply. Metrics come from the evaluator output, not from a loss curve.
- The `--cfg job` dry-run flag (used in verify-static) is different from actually running training — do not confuse them.
- Paper-reported metrics are often best-of-many-seeds or cherry-picked. A 20% tolerance is realistic for a single-seed run; exact reproduction usually requires hyperparameter sweeps.
- For `run_mode=quick`, metrics will be significantly worse than expected — this is by design. The purpose is to verify the pipeline runs end-to-end. Do not mark the run as FAILED just because quick-mode metrics are far from expected.
- Some experiments use custom callbacks (e.g., `ResourceTracker`) that log efficiency metrics. These are informational — do not use them for pass/fail decisions.
- If early stopping patience is set very low in the config, the run may stop too early. Check the config's `callbacks.early_stopping.patience` value and note it in the report.
