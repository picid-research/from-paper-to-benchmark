---
name: verify-sanity
description: Run the trimmed Karpathy-style sanity ladder (init loss → gradient flow → micro-batch memorization) on a freshly implemented model in a single in-process framework build, while distinguishing model diagnostics from categorical blockers and dataset executability failures. Use after static checks pass and before any real training.
---

# /verify-sanity

Run the standardized PICID sanity tools. Do not write one-off sanity scripts during a paper-validation run.

This skill assumes `/verify-static` already passed for the current experiment config. If static checks have not run, run them first.

## Scope and intent

The ladder follows Karpathy's stage-2 spirit: cheap, fast checks you would always run during development. The default ladder builds the framework stack **once** per invocation and runs three in-process checks against it. Total wall-clock is typically tens of seconds on toy data and under a few minutes on real datasets.

The ladder is intentionally narrow. It is a model-health diagnostic gate, not a hyperparameter probe and not a partial training run. Only categorical execution failures block training; unresolved but executable model sanity concerns are recorded as `WARN_CONTINUE`.

## Input Contract

The orchestrator must provide:

- **vault_dir**: Path to vault containing `04-implementation-blueprint.json` and rendered `04-implementation-blueprint.md`; the tool appends `06-sanity-ladder-log.md`
- **experiment_config**: Hydra experiment config path, for example `pronostia/prognostics/raw/my_model`
- **task_type**: `regression`, `classification`, or `fit_predict`
- **num_classes**: classification only
- **expected_init_loss**: regression expected first loss from the blueprint, or `"auto"`
- **repo_root**: PICID repo root
- **dataset_candidate**: current dataset candidate name/config
- **extra_overrides**: Hydra overrides for targeted reruns; include `paths=agent` unless the caller explicitly requests a different paths profile
- **timeout_sec**: optional per-check wall-clock budget (default 600s)
- **include_subset_convergence**: optional boolean (default false). Opt in only when the ladder verdict is `INVESTIGATE` and you specifically want a longer training-loss probe.

## Required Context

Before invoking the tool, read:

1. `.opencode/reference/` — the base-class contract files (`datasources.md`, `transforms.md`, `models.md`, `evaluators.md`) and `patterns.md` (fit-on-train, caching, inverse scaling)
2. `logging-guide.md`
3. `{vault_dir}/04-implementation-blueprint.json.verification_protocol` first; use rendered Markdown Section 7 only for human-readable detail

## Normal Path

Invoke the OpenCode custom tool:

```text
PICID_sanity_ladder
```

Pass:

```json
{
  "repo_root": "{repo_root}",
  "vault_dir": "{vault_dir}",
  "experiment_config": "{experiment_config}",
  "task_type": "{regression|classification|fit_predict}",
  "num_classes": "{classification only}",
  "expected_init_loss": "{number or auto}",
  "dataset_candidate": "{dataset_candidate}",
  "extra_overrides": ["paths=agent"],
  "write_log": true
}
```

The tool:

- runs the three default checks in cheapest-first order against a single shared in-process framework stack
- emits sparse `[heartbeat] check=... step=N/M loss=X elapsed=Ys` lines on stderr so the calling agent can see progress in real time without flooding context
- enforces a 600s wall-clock budget per check (configurable via `timeout_sec`); a check that times out returns a `PRECHECK_TIMEOUT` status with whatever partial results it produced rather than discarding them
- forces `logger=csv` for the opt-in subset_convergence path
- appends a consistent report to `{vault_dir}/06-sanity-ladder-log.md`
- appends the same authoritative attempt payload to `{vault_dir}/06-sanity-ladder-log.jsonl`
- returns structured JSON for decision-making

## Default Ladder

Three checks, run in order, cheapest first:

1. **`init_loss`** — single forward pass on one batch. Validates that loss computation executes and the loss is finite, then records scale mismatches against `ln(num_classes)` or `expected_init_loss` as diagnostics. NaN/Inf or a crash blocks; scale mismatch does not.
2. **`gradient_flow`** — single forward + backward. Reports dead, exploding, vanishing parameter gradients. Then runs a second backward against per-sample inputs to detect batch-dimension leakage (the `view`-vs-`permute` class of bugs). A backward crash, non-finite gradients, or no trainable parameter receiving a finite nonzero gradient blocks; partial weak/dead gradients are warnings unless they make training non-executable.
3. **`overfit_batch`** — memorize a tiny sliced micro-batch (default 4 examples) for up to 400 optimizer steps. Judge the result by task-aware end-state behavior instead of a single rigid reduction threshold:
   - regression: `PASS` when loss is near-zero or absolute prediction error is very small, warning when memorization is incomplete or absent but numerically finite
   - classification: `PASS` when train accuracy reaches 1.0 and loss is near-zero, warning when memorization is incomplete or absent but numerically finite

A `config_preflight` check runs first; if Hydra composition fails the ladder stops and reports `TOOL_INVOCATION_FAILURE` without attempting any of the three.

For `fit_predict` models all three checks are skipped with reason `fit-predict model, no gradient-based training`.

## On-Demand Individual Tools

Use these only when the experimenter has a specific reason to rerun one check:

- `PICID_sanity_init_loss` — rerun after loss/init/final-layer changes
- `PICID_sanity_overfit_batch` — rerun after model/loss/optimizer fixes
- `PICID_sanity_gradient_flow` — rerun after architecture or reshape fixes
- `PICID_sanity_zero_input` — **on-demand only, not in the default ladder.** Run when the data pipeline is suspect (transforms changed, target keys touched, or the ladder passed but real training shows the model is ignoring inputs). It trains briefly on real and zeroed inputs and compares final losses.
- `PICID_sanity_subset_convergence` — **opt-in only, not in the default ladder.** Runs a 30-epoch CLI subprocess on 10% of data. This is partial training, not a sanity check. Trigger only when the ladder passed but you want to confirm that loss continues to decrease across more steps (typically as part of debugging a hyperparameter issue, not a model issue).
- `PICID_sanity_report` — append a report from structured check results

When rerunning a targeted check, include `write_log: true` if the result should be appended to `06-sanity-ladder-log.md`.

## Why these three

- `init_loss` + `overfit_batch` + `gradient_flow` together cover Karpathy's "verify loss @ init", "overfit one batch", and "use backprop to chart dependencies" — the three checks that catch the bulk of model-wiring bugs (bad init, wrong loss, dead branches, view-vs-permute mix-ups, missing `requires_grad`).
- `subset_convergence` is *stage-3* work: it is partial training and its tail-stability criterion is really a learning-rate / optimizer probe. The first real training run already provides this signal.
- `zero_input` is valuable but situational; running it on every model wastes a full second framework build for a check that only catches data-pipeline bugs the rest of the ladder cannot see.

## Dataset Failure Classification

Before treating a failed command as a model bug, inspect the tool JSON and `06-sanity-ladder-log.md`.

Use:

- `DATASET_UNAVAILABLE` for missing raw files/directories, missing PHMD cache files, invalid `paths.data_dir`, failed PHMD download/cache/task retrieval, or equivalent local availability failures.
- `DATASET_EXECUTION_FAILED` for datasource load/split/preprocessing crashes caused by malformed local data or dataset-size resource failures before model-specific computation.

When either status occurs:

1. Stop remaining sanity checks for this dataset candidate.
2. Keep the appended `06-sanity-ladder-log.md` entry.
3. Set the legacy `fallback_trigger` field to yes to indicate dataset recovery is required.
4. Return control to the experimenter so it can repair the same dataset or stop with a named blocker.

## Decision Rules

- All applicable checks pass or are intentionally skipped → `PASS`
- One or more model diagnostics fail or return `INVESTIGATE`, but forward/backward/loss/optimizer execution remains finite and at least one trainable parameter receives a finite nonzero gradient → `WARN_CONTINUE`; record the failed checks and continue to benchmark training
- Categorical execution failure in the default ladder → `BLOCK`; fix implementation before training. Examples: check crash, NaN/Inf loss, non-finite gradients, or no trainable parameter with finite nonzero gradient.
- `PRECHECK_TIMEOUT` on a check → inspect partial metrics (`completed_steps`, `reduction_fraction`) and heartbeat progress. If the model is genuinely slow per step rather than buggy, rerun that specific check with a larger `timeout_sec` and continue within the sanity retry budget. If the timeout shows no useful progress, a stuck dataloader, or repeated non-timeout failures, route to the normal repair loop instead of escalating immediately.
- `DATASET_UNAVAILABLE` or `DATASET_EXECUTION_FAILED` → trigger same-dataset recovery or blocker, not model debugging
- Optional: only the opt-in `subset_convergence` fails → `WARN_CONTINUE`; preserve the convergence warning and let real training provide the longer-horizon signal

## Failure Pattern Guide

| Pattern | Likely cause |
| --- | --- |
| `init_loss` scale check fails with finite loss | Init, final layer, loss config, or output-target wiring warning; continue as `WARN_CONTINUE` if execution is finite |
| `init_loss` is NaN/Inf or crashes | Categorical implementation blocker |
| `init_loss` passes, `overfit_batch` is `INVESTIGATE` or finite `FAIL` | Model shows weak/incomplete memorization; inspect predictions vs targets and optimizer/loss wiring, but continue as `WARN_CONTINUE` |
| `gradient_flow` reports partial weak/dead gradients | Architecture, dead branch, bad reshape, or batch-dimension leak warning; continue unless no meaningful trainable gradient exists |
| `gradient_flow` has no finite nonzero trainable gradients | Categorical implementation blocker |
| All three produce finite diagnostic failures | Serious implementation-health warning — continue as `WARN_CONTINUE`, and do not call the model strongly validated |
| All three pass but training underperforms | Run on-demand `zero_input` to rule out data pipeline; otherwise treat as a hyperparameter or capacity issue |
| Dataset status is not `PASS` | Selected dataset cannot execute locally; repair the same dataset or stop with a named blocker |

## Report Requirements

Every attempt appended to `{vault_dir}/06-sanity-ladder-log.md` must start with a machine-readable YAML front-matter block, followed by the existing human-readable markdown body. Downstream skills (`/run-training`, `/diagnose-verify-block`) parse the block; humans read the prose.

Front-matter schema (required fields):

```yaml
---
verdict: WARN_CONTINUE          # PASS | WARN_CONTINUE | BLOCK | PRECHECK_TIMEOUT | DATASET_UNAVAILABLE | DATASET_EXECUTION_FAILED | TOOL_INVOCATION_FAILURE | ABORTED
attempt: 3                      # 1-indexed attempt number for this vault
passed_checks: [config_preflight, init_loss, gradient_flow]
failed_checks: [overfit_batch]
skipped_checks: []              # checks intentionally skipped (e.g., fit_predict)
check_diagnostics:              # per-check numeric signals; one entry for every check that ran
  overfit_batch:
    initial_loss: 0.0585
    final_loss: 0.0247
    reduction_fraction: 0.5784
    threshold: 0.90
    completed_steps: 200
    elapsed_sec: 47.3
dataset_candidate: <name>
dataset_status: PASS            # PASS | DATASET_UNAVAILABLE | DATASET_EXECUTION_FAILED
fallback_trigger: false            # legacy field: true means dataset recovery is required, not cross-dataset fallback
experiment_config: <hydra path>
timestamp: 2026-04-17T11:24:27
---
```

Rules:

- `verdict` is authoritative. `PASS` means every applicable default-ladder check passed (or was intentionally skipped). `WARN_CONTINUE` means at least one model diagnostic did not pass, but the model remains executable enough for benchmark training. `BLOCK` is reserved for categorical execution failures.
- `passed_checks` + `failed_checks` + `skipped_checks` together cover every check the tool attempted; the sets are disjoint.
- `check_diagnostics` must include the numeric signals the tool collected for each attempted check — at minimum `initial_loss`, `final_loss`, `reduction_fraction`, `threshold`, `completed_steps`, `elapsed_sec` where applicable. Include additional per-check diagnostics (gradient norms, dead-layer counts, batch-leak flags) verbatim from the tool JSON.
- `attempt` increments across successive invocations that append to the same vault log; use the count of prior `---` front-matter blocks in the file + 1.
- `dataset_status` mirrors the dataset executability classification; `fallback_trigger` is `true` iff `dataset_status != PASS`. Cross-dataset fallback is disabled.

Below the front-matter, the existing human-readable markdown body remains required and must contain:

- all check statuses: `PASS`, `FAIL`, `SKIPPED`, `PRECHECK_TIMEOUT`, `DATASET_UNAVAILABLE`, `DATASET_EXECUTION_FAILED`, or `TOOL_INVOCATION_FAILURE`
- actual loss values, gradient counts, completed steps, elapsed seconds, and diagnostics where available
- failed command and traceback excerpt for command failures
- summary verdict matching the front-matter `verdict`
- dataset candidate, dataset executability status, and fallback trigger

If the front-matter cannot be written (e.g., tool crash before results are collected), append an attempt with `verdict: ABORTED` and whichever fields are known; never leave a trailing attempt without a front-matter block.

## Structured sidecar contract

After producing the final structured result for an attempt, append the same machine-readable payload to `{vault_dir}/06-sanity-ladder-log.jsonl` via `validate_paper_workflow_append_sanity_result`.

The JSONL object must carry the same authoritative fields as the YAML front-matter:
- `verdict`
- `attempt`
- `passed_checks`
- `failed_checks`
- `skipped_checks`
- `check_diagnostics`
- `dataset_candidate`
- `dataset_status`
- `fallback_trigger`
- `experiment_config`
- `timestamp`

The markdown log remains the human-readable audit trail. The JSONL sidecar is the control-plane input for later phases.

The skill does not fix implementation code. It reports enough detail for the experimenter to choose the next implementation skill, hyperparameter adjustment, or same-dataset recovery action.
