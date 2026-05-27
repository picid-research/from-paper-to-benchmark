---
name: check-batch-fit
description: Check that the fixed validation batch sizes train=512, val=1024, test=1024 fit on the current hardware, without searching for or substituting a different batch size. Invoked once after sanity is accepted and whenever the resolved config hash changes, before /run-training.
---

# /check-batch-fit

Run a fixed-batch feasibility check for the exact validation batch sizes that the experiment is supposed to use. This replaces automatic batch-size tuning and Hydra-chain batch estimation: the workflow must not search upward/downward for a "best" batch size, must not inherit a base default such as `16`, and must not write a substitute training batch into the run command.

For validate-paper runs, the invariant is: force datamodule batch sizes to `train=512`, `val=1024`, `test=1024`; preserve PICID's LR scheduler/check cadence for the primary reference profile; then use paper-stated or explicitly imputed model-specific HPs for the rest. A newly implemented paper model usually has no pre-existing official model-specific HP record; that absence is not a blocker when the composed experiment carries the scheduler contract and the workflow applies the fixed validation batch overrides.

## Input contract

- **vault_dir**: path where `batch_fit_check.json` lives
- **repo_root**: PICID repo root
- **experiment_config**: Hydra experiment config (e.g., `pronostia/ptics/my_model`)
- **dataset_candidate**: current dataset name/config
- **gpu_available**: boolean
- **model_paradigm**: `feedforward` | `training` | `fit_predict`
- **hp_profile**: `reference` | `paper` (default `reference`). Every validation profile fixes datamodule batch sizes to `512/1024/1024`. `reference` is the primary validation profile: LR scheduler/check cadence is preserved from PICID, and model-specific HPs come from the paper or explicit imputations in the experiment config. `paper` uses the experiment config's paper/default non-batch HPs for auxiliary context.
- **hp_overrides**: optional explicit Hydra overrides, only when a keyed official entry exists. Do not include datamodule batch-size overrides here; the workflow applies the fixed validation batch after any other overrides so it wins. For novel paper models this is commonly empty; do not treat that as unresolved.
- **hp_reference_source**: script/report path or note proving where `hp_overrides` came from; for novel models without keyed overrides, use a note such as `fixed validation batch 512/1024/1024 + composed PICID scheduler cadence + paper/imputed model HPs`.
- **force_fresh** (optional, default `false`): if `true`, ignore any existing matching `OK` record and re-check
- **timeout_sec** (optional, default `600`): fixed-batch wall-clock budget. Increase only when logs/heartbeats show the check is healthy but slow; never use it to mask a stuck dataloader, dataset failure, or implementation bug.

## Keys

- `resolved_config_hash`: sha256 prefix of the resolved Hydra config for `experiment_config` after applying the selected `hp_profile` overrides plus the fixed validation batch overrides for `hp_profile=reference`.
- `hardware_id`: `torch.cuda.get_device_name(0)` when `gpu_available=true`, else `cpu`.

## Cache file

`{vault_dir}/batch_fit_check.json` is a JSON array. Append-only; never rewrite history.

Record schema:

```json
{
  "resolved_config_hash": "<sha256 prefix>",
  "hardware_id": "<cuda name or cpu>",
  "status": "OK",
  "train_batch_size": 512,
  "val_batch_size": 1024,
  "test_batch_size": 1024,
  "hp_profile": "reference",
  "hp_overrides": ["<optional keyed Hydra override>", "..."],
  "hp_reference_source": "scripts/paper_.../<script>.sh, report_output/..., or composed experiment note",
  "dataset_candidate": "<name>",
  "command": "<command used for the check>",
  "reason": null,
  "timestamp": "2026-04-17T12:00:00"
}
```

## Procedure

### Step 1 - Lookup

Read `{vault_dir}/batch_fit_check.json` (create it as `[]` if missing). If `force_fresh=false` and the latest matching record for `(resolved_config_hash, hardware_id)` has `status: OK`, print the record and exit 0 without re-checking.

If the latest matching record is not `OK`, rerun only when the caller passed `force_fresh=true` or the resolved config hash changed.

### Step 2 - Fixed-batch check

Use the real experiment with the selected profile. For every validation profile, apply these three Hydra overrides after any other HP overrides:

```bash
datamodule.train_batch_size=512
datamodule.val_batch_size=1024
datamodule.test_batch_size=1024
```

Do not inspect config chains or scripts to decide these batch values. The only batch-size check is whether `512/1024/1024` fits on the current hardware.

For `hp_profile=reference`, verify the composed experiment preserves the PICID scheduler/check contract:

1. inspect the nearest existing experiment/config chain for the same dataset/task to identify scheduler/checkpoint/early-stopping cadence;
2. inspect `scripts/paper_prognostics/*.sh` or `scripts/paper_diagnostics/*.sh` when present only for scheduler/provenance context, not for datamodule batch-size selection;
3. pass keyed `hp_overrides` only when a real keyed official entry exists. If none exists for the novel paper model, pass no HP overrides and rely on the composed experiment config.

For `hp_profile=paper`, pass no reference HP profile overrides, but still apply the fixed validation batch. The primary verdict is still based on the `reference` profile.

Resolve `timeout_sec` to the caller-provided positive value, otherwise `600`, then run:

```bash
timeout {timeout_sec} uv run python -m picid.run experiment={experiment_config} \
  paths=agent \
  logger=csv \
  seed=42 \
  {dataset_resolved_hp_overrides_if_reference} \
  datamodule.train_batch_size=512 \
  datamodule.val_batch_size=1024 \
  datamodule.test_batch_size=1024 \
  trainer.max_epochs=1 \
  +trainer.limit_train_batches=1 \
  +trainer.limit_val_batches=1 \
  +trainer.limit_test_batches=0 \
  ~callbacks.early_stopping \
  ~callbacks.model_checkpoint \
  enable_progress_bar=false \
  datamodule.num_workers=0 \
  datamodule.pin_memory=false
```

The command may override dataloader worker/pin-memory settings for reliability during the check. It must not override learning rate, optimizer, or scheduler beyond the selected HP profile. It must always include the fixed validation datamodule overrides above. If the composed reference config lacks identifiable PICID scheduler/check cadence, stop with `PICID_SCHEDULER_UNRESOLVED`; do not invent replacements. Do not stop merely because a novel model has no official model-specific reference HP record, and do not stop because the base datamodule config says `16`.

### Step 3 - Append record

Append exactly one record to `{vault_dir}/batch_fit_check.json`:

- `status: OK` when the command exits 0.
- `status: OOM` when the configured batch fails with an out-of-memory error.
- `status: TIMEOUT` when the resolved timeout fires.
- `status: DATASET_UNAVAILABLE` or `DATASET_EXECUTION_FAILED` when dataset loading/preprocessing fails.
- `status: FAILED` for other environment or runtime failures.

### Step 4 - Classified exit

- `OK` -> proceed to `/run-training`; the training run uses the fixed configured batch sizes.
- `DATASET_UNAVAILABLE` or `DATASET_EXECUTION_FAILED` -> same-dataset recovery or named blocker per `policies.md`.
- `OOM` -> this is not a tuning signal. Stop and report that the fixed validation batch does not fit on this hardware, or let the user opt into a different comparison profile.
- `TIMEOUT` with useful progress or evidence of a genuinely heavy model/dataset -> rerun with `force_fresh=true` and a larger timeout while staying inside the check-batch-fit attempt budget. Do not change train/val/test batch size, LR, optimizer, or scheduler.
- `TIMEOUT` with no useful progress -> classify dataset vs implementation vs environment and loop through the appropriate same-dataset repair path. Escalate only after the bounded attempts are exhausted or the reason is non-recoverable.
- `FAILED` -> classify from the traceback and route to same-dataset recovery, implementation repair, or environment escalation.

## When to re-invoke

- Once, after `/verify-sanity` returns `verdict: PASS` or `WARN_CONTINUE` and before `/run-training`.
- Again after any event that changes `resolved_config_hash`: same-dataset recovery, `/implement-experiment` re-invocation, or any edit to optimizer/trainer/datamodule sections of the experiment YAML.
- With `force_fresh=true` only when the prior check is stale or inconclusive. Do not use this skill to search for an alternate batch size.
