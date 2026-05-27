# Orchestration Policies

Single source of truth for orchestration rules shared by `paper-validator`, `experimenter`, `implementation-blueprint`, and the `/implement-*`, `/verify-*`, `/run-training`, `/evaluate-results` skills. Do not restate these rules in agent or skill prose — reference this file.

## Validation modes

| Mode | When | Interpretation |
|------|------|----------------|
| `exact_reproduction` | Paper dataset and protocol exist in PICID. | Direct metric comparison valid, modulo seed/reporting differences. |
| `framework_validation` | Paper dataset exists in PICID, but exact paper protocol/details cannot be matched. | Validate implementation behavior on that same framework dataset; compare relative positioning and sanity, not raw paper numbers. |
| `benchmark_only` | No meaningful direct or relative comparison available. | Report framework behavior and limitations; do not claim paper reproduction. |

Cross-dataset fallback is disabled for this workflow. The selected validation dataset must be a direct PICID-supported dataset used by the paper. If that dataset cannot execute locally, repair the same dataset/config path when possible; otherwise stop with the named dataset blocker. Never switch to a different dataset to keep the run moving.

## Workflow status axes

The workflow must keep three statuses separate:

| Axis | Meaning | Examples |
|------|---------|----------|
| `artifact_status` | Did the phase produce its required machine-readable and human-readable artifacts? | `COMPLETE`, `MISSING`, `CORRUPT` |
| `technical_status` | Is the implementation runnable and healthy enough for benchmark validation? | `PASS`, `REPAIRABLE_BUG`, `HARD_BLOCKER` |
| `scientific_status` | What did the paper comparison say? | `VALIDATED`, `PLAUSIBLE`, `INVESTIGATE`, `INVESTIGATE_CLAIMS_DISPUTED`, `BENCHMARK_ONLY` |

Only `artifact_status` and `technical_status` may block the workflow. Scientific statuses are report verdicts and observations. A completed evaluation report with `INVESTIGATE`, `INVESTIGATE_CLAIMS_DISPUTED`, `BENCHMARK_ONLY`, metric-scale ambiguity, missing overlapping baselines, or dataset-dependent claims is still a successful evaluation phase unless it also identifies a concrete implementation/evaluator bug.

Hard blockers are conditions that prevent trustworthy benchmark execution or artifact production:
- missing/corrupt required artifacts that cannot be reconstructed
- import/config errors in generated code
- categorical sanity `BLOCK` after the diagnostic-repair budget is exhausted (for example crash, NaN/Inf loss, non-finite gradients, or no trainable parameter receiving a finite nonzero gradient)
- dataset executability failure after same-dataset recovery is exhausted
- fixed validation batch (`train=512`, `val=1024`, `test=1024`) does not fit under the selected comparison profile
- no usable training output/checkpoint/metrics for evaluation
- evaluator/runtime crash, corrupt leaderboard file, or unparseable evaluation report
- guardrail violations such as `FRAMEWORK_CHANGE_REQUIRED` or `VAULT_EDIT_ATTEMPTED`

Soft observations are evidence to preserve in the vault, not blockers:
- model sanity `WARN_CONTINUE` after unresolved diagnostic checks such as init-loss scale mismatch, partial weak/dead gradients, or failure to fully overfit a micro-batch
- contradicted paper claims after a healthy run
- normalized vs denormalized metric-scale mismatch
- paper metric/unit ambiguity
- no overlapping PICID baseline for a paper baseline
- absolute paper values under `framework_validation`
- weak framework competitiveness
- unsupported paper datasets that were intentionally excluded by the blueprint

The default success condition for `/validate-paper` is benchmark validation, not paper-claim perfection: implementation artifacts exist, static checks pass, sanity is accepted (`PASS` or `WARN_CONTINUE` with unresolved diagnostics recorded), fixed-batch feasibility is recorded, at least one scheduled benchmark run produces usable training evidence, and evaluation artifacts are written with any scientific caveats clearly recorded.

## Paper dataset selection

The paper, not the user's prior expectation, is the source of truth for dataset selection. The analysis agents must extract every dataset the paper actually trains or evaluates on, then keep only paper datasets with direct PICID support as validation targets. A direct target is a paper-named dataset whose loader/config exists in the framework inventory, currently limited to N-CMAPSS, UNIBO, NB14, and XJTU-SY when the paper uses those names or unambiguous aliases. Treat "NASA random use battery dataset", "NASA randomized/random battery dataset", and close variants as unambiguous aliases for PICID `NB14` unless the paper gives evidence that it is a different NASA battery source. Do not treat "NASA cyclic aging battery dataset", "NASA5", or "NASA11" alone as `NB14`.

Supported direct dataset families:

| Family | Datasource configs | Usual experiment roots | Notes |
|--------|--------------------|------------------------|-------|
| N-CMAPSS | `n_cmapss`, `concepts_n_cmapss*` | `concepts_n_cmapss*` | Prefer existing `concepts_n_cmapss_ds02`/multi-source configs when the paper names a supported subset. |
| UNIBO | `unibo21_bosello`, `unibo21_bosello_anomaly` | `unibo` | Battery RUL/anomaly workflows; do not use it as an NB14 substitute. |
| NB14 | `nb14_bosello` | `nb14` | Alias only for NASA randomized/random-use battery data. |
| XJTU-SY | `xjtu_sy` | `xjtu_sy` | Bearing RUL; paper condition/unit splits must be enforced explicitly. |

Unsupported paper datasets are excluded from automated validation instead of being converted into substitutes. Record each excluded dataset and the reason in the blueprint. Do not add a new datasource for an excluded dataset unless the paper explicitly requires exact reproduction and the user has asked for that scope.

When a paper has multiple direct targets, run one independent primary validation per target. Keep the run lean: one reference-profile training run per target (plus the required 1-epoch preflight gate in `run_mode=full`), no dataset sweep, no repeated seeds, and no auxiliary paper/default non-batch profile unless `hp_mode=reference_plus_paper` was explicitly selected.

## Validation run matrix

The blueprint may define a validation run matrix when the paper requires more than one planned run to judge its claims. A row is a planned benchmark work unit, not a retry and not a dataset fallback. Each row should include:
- `run_id`
- `role`: `primary`, `secondary`, or `auxiliary`
- `paper_dataset` and `framework_dataset`
- `experiment_config`
- `model_variant` or method name
- `hp_profile`
- `claims_supported`
- `required_for_benchmark_validation`

The experimenter must complete all `required_for_benchmark_validation=true` rows that pass static checks and have accepted sanity/batch-fit gates before starting optional post-training claim repair. A scientific concern in one completed row must not prevent independent planned rows from running. If a row is skipped, the final report must name the technical reason (for example no runnable config, batch-fit failure, dataset unavailable after same-dataset recovery), not merely a claim mismatch in another row.

## Dataset recovery rules

Cross-dataset fallback is disabled. Blueprints should set `fallback_allowed=false` and `dataset_fallback_candidates=[]`. Legacy fields named `fallback_trigger` in logs mean "dataset recovery/blocking path required", not permission to switch datasets.

Allowed recovery actions stay on the selected paper dataset:
1. fix an experiment/config reference to the correct existing datasource/transform/task/dataset/evaluator for the same dataset family
2. fix a paper-authored transform/model config that breaks the same dataset's executable contract
3. correct an obvious local path/cache/download invocation when the framework already supports that dataset
4. stop with a named blocker if the raw data is absent, corrupt, or the same-dataset repair budget is exhausted

**Trigger conditions (dataset executability failures):**
- `DATASET_UNAVAILABLE` — missing raw files, missing PHMD cache files, invalid `paths.data_dir`, unavailable dataset directory, PHMD download/cache/task retrieval errors
- `DATASET_EXECUTION_FAILED` — datasource `load_data()`, `split_data()`, or `PreProcessor.pipeline()` crashes from malformed local data, or dataset-size resource failures that persist after the normal batch-size/quick-run mitigation

**Non-recovery conditions (implementation failures — do NOT treat as dataset recovery):**
- import errors in newly implemented model/transform/loss/metric code
- missing generated config files or invalid Hydra composition from implementation wiring
- model output/target shape mismatches from the new implementation
- loss, forward-pass, gradient-flow, or optimization sanity failures

**Recovery steps (automatic, no user prompt):**
1. Append `## Dataset Recovery Decision` to `06-sanity-ladder-log.md`: selected dataset, failed command, error excerpt, root-cause classification, and same-dataset action.
2. If the fix is config-only, re-invoke `/implement-experiment` for the same dataset with corrected datasource/transform/dataset/task/evaluator wiring. Keep the novel model/loss/transform code unless verification proves it is broken.
3. Re-run `/verify-static` and `/verify-sanity` on the same experiment.
4. If the same dataset cannot be made executable within the retry budget, stop and report the blocker. Do not switch datasets.

## Cost-gating ladder

Spend compute only after cheaper checks pass. Order is authoritative:

1. File/input gates (Phase 0)
2. Paper segmentation and chunk index (Phase 1)
3. Blueprint completeness gate (Phase 3 readiness gate fields)
4. Static integrity + dataset preflight (`/verify-static`)
5. Runtime sanity ladder (`/verify-sanity`: init_loss → gradient_flow → overfit_batch in one in-process build). The ladder is diagnostic by default: `PASS` and `WARN_CONTINUE` are accepted for benchmark training, while categorical execution failures remain `BLOCK`.
6. Fixed validation batch + scheduler preservation check (`/check-batch-fit`) — once after sanity is accepted, cached in `{vault_dir}/batch_fit_check.json` keyed on `(resolved_config_hash, hardware_id)`; re-invoked only when the hash changes (same-dataset recovery or experiment re-composition). The validation profile always applies `datamodule.train_batch_size=512`, `datamodule.val_batch_size=1024`, and `datamodule.test_batch_size=1024` explicitly, then preserves the LR scheduler/check cadence from the selected PICID optimization/trainer/callback configs. It verifies that this fixed batch fits; it does not inspect Hydra chains to choose a replacement batch size.
7. Training (`/run-training`) — for each evaluation target, `run_mode=full` first runs `training_stage=preflight_1epoch`, a 1-epoch representative-slice gate that must exercise train, validation, test, and checkpoint creation. The experimenter repairs/retries this gate until it passes or the bounded training-repair budget is exhausted. Only then does it run `training_stage=full`: one monitored paper-scale run with the paper's `max_epochs` when specified, otherwise the resolved trainer default. Full training has no wall-clock timeout; the monitor polls metrics/logs/checkpoints and stops only for concrete instability/failure or a documented graceful plateau stop when early stopping is absent or not firing. `run_mode=quick` uses the same 1-epoch gate semantics for dev/CI.
8. Evaluation (`/evaluate-results`)

Automated runs use `logger=csv` unless the user explicitly overrides — CSV metrics are local, deterministic, and do not require WandB.

## Bounded retries

Retry budgets are tiered by cost. Cheap artifact-regeneration and static-wiring repairs get a modestly longer leash; sanity repair gets the largest pre-training budget; full-training and post-training claim-repair loops remain tighter because each iteration is expensive.

| Failure point | Action | Limit |
|---------------|--------|-------|
| Component quick check after `/implement-*` | Re-read error, adjust spec, re-invoke the same skill | 5 retries per component |
| `/verify-static` fails | Re-invoke the responsible implementation skill, rerun static | 5 static retry cycles |
| `/verify-sanity` reports `verdict: WARN_CONTINUE` | Proceed to `/check-batch-fit` and training, preserving the failed diagnostic names and metrics in `06-sanity-ladder-log.md` / `.jsonl`. | no retry required |
| `/verify-sanity` reports `verdict: BLOCK` on a categorical default-check failure | Invoke `/diagnose-verify-block` with `max_iterations=10`. This supersedes the old per-check decision tables. If repair reduces the result to `WARN_CONTINUE`, continue with warnings; if it remains `BLOCK` after budget, stop with the named blocker. | bounded by the diagnostic-repair loop budget (see below) |
| `/verify-sanity` reports `PRECHECK_TIMEOUT` | Inspect partial metrics and heartbeat progress. If the check is making real progress but the model/dataset is heavy, rerun that specific check with a larger `timeout_sec`; otherwise treat it as `BLOCK` and enter `/diagnose-verify-block`. Keep improving/rerunning within the sanity retry budget until the check passes, the timeout is proven not to be slowness, or the budget is exhausted. | counted in sanity retry budget |
| `/check-batch-fit` reports `TIMEOUT` | If logs/metrics show useful progress or slow startup from a heavy model/dataset, rerun with `force_fresh=true` and a larger timeout. If there is no useful progress, classify dataset vs implementation vs environment and loop through the matching same-dataset repair path. Do not tune batch size, LR, optimizer, or scheduler. | counted in check-batch-fit attempts |
| Training preflight `FAILED` / `CRASHED (other)` | If diagnostic signal matches sanity-repair territory (NaN loss, diverging gradient), enter `/diagnose-verify-block`. Otherwise classify dataset vs implementation vs environment and retry only with a concrete non-batch-tuning fix. Full training must not start until this gate passes. | 5 training attempts |
| Full monitored training `FAILED` / `CRASHED (other)` | Stop the active process if needed, preserve whatever artifacts exist, classify dataset vs implementation vs environment, and continue to evaluation when a usable output directory/checkpoint exists. Retry only for a concrete fix, not for open-ended tuning. | counted in training attempts |
| Training `CRASHED (OOM)` | If `/check-batch-fit` passed but full training OOMs, re-run `/check-batch-fit` with `force_fresh=true` once to classify stale hardware/config state. Do not auto-reduce the fixed validation batch; escalate or switch only via an explicit comparison profile. | counted in training attempts |
| Evaluation `IMPLEMENTATION_BUG` or evaluator/runtime failure | Return to the responsible earlier phase with evaluation diagnostic as context | bounded by phase retry limits |
| Evaluation `INVESTIGATE` | Record as a scientific observation. Re-run only if the report names a concrete technical cause (wrong dataset, wrong HP profile, missing metric extraction) and the scheduled validation matrix is complete. | no open-ended tuning |
| Evaluation `INVESTIGATE_CLAIMS_DISPUTED` | Record as a scientific observation and continue any remaining scheduled benchmark runs. Invoke `/diagnose-training-result` only after the scheduled validation matrix is complete and the contradiction plausibly indicates a fixable implementation/training defect. | bounded by the training-repair loop budget (see below) |

Ordinary sanity/training technical failures are retryable workflow states, not terminal blockers. `run_state.json` should remain `active` at the failed phase while the orchestrator repairs and retries. A sanity `WARN_CONTINUE` is a passed phase with warning metadata, not a failed phase. Completed evaluation artifacts with scientific concerns are passed phases with warning metadata, not failed phases. Use a blocking final status only after the bounded repair budget is exhausted on a categorical blocker or when the named reason is genuinely non-recoverable (for example `FRAMEWORK_CHANGE_REQUIRED`, `VAULT_EDIT_ATTEMPTED`, or a missing required artifact that cannot be reconstructed).

Timeouts outside `/verify-sanity` and `/check-batch-fit` are also recovery triggers unless the surrounding tool explicitly says otherwise. The orchestrator should classify the partial evidence, apply the smallest concrete fix, and rerun inside the relevant phase attempt budget; if logs show the operation is healthy but slow, increasing the specific timeout is allowed and preferred over declaring the phase blocked.

Do not classify a run as `TRAINING_INFEASIBLE_AT_FULL_SCALE` just because the current full-dataset plan is long or expensive. That label is reserved for true execution impossibility after the normal mitigation ladder has been exhausted (fixed-batch fit check, dataset classification, preflight repair, and bounded repair loops). A large-but-runnable full run is a cost decision, not a hard technical blocker.

If retries are exhausted, escalate with the last diagnostic. No infinite loops. No over-tuning.

## Diagnostic-repair loop budget

`/diagnose-verify-block` is the canonical resolution path for any categorical `verdict: BLOCK` from `/verify-sanity`. Non-categorical sanity diagnostics produce `WARN_CONTINUE` and do not require this loop. The loop replaces the previous fixed hyperparameter-adjustment ladder (LR/10 → optimizer swap → gradient clipping), which was brittle because it (a) treated HP-mismatch as the only cause, and (b) patched failing checks one at a time without verifying that fixes didn't regress passing checks.

- **Default budget:** `max_iterations = 10` per invocation.
- **Configurable:** the experimenter may pass a smaller budget when the root cause is obvious, but not a larger one — the 10-iteration ceiling is the cost-gating contract.
- **Not bumpable mid-loop:** once the loop starts, the budget is fixed. If 10 iterations do not reach `PASS` or reduce the result to `WARN_CONTINUE`, the skill returns `ESCALATE` and the experimenter surfaces the full iteration history to the user. Silently resuming with a new invocation after an `ESCALATE` is forbidden.
- **Per-iteration mandate:** every iteration re-runs ALL default-ladder checks (not only the previously failing ones), forms ONE global hypothesis that explains every failing check jointly, and writes its hypothesis/prediction/observation/conclusion into `06-sanity-ladder-log.md`. Per-check patching is a bug.
- **Decision-band relaxation is forbidden:** the default-ladder decision bands are the contract. A hypothesis that proposes reinterpreting a categorical `BLOCK` as `WARN_CONTINUE`, or a warning as `PASS`, without changing the implementation is not a repair — it is hiding the signal.

## Training-repair loop budget

`/diagnose-training-result` is the canonical resolution path only when the experimenter elects to repair an `INVESTIGATE_CLAIMS_DISPUTED` result after all scheduled benchmark runs have completed. One or more pre-registered claims from `05-paper-hypothesis.md` being `CONTRADICTED` after a completed training run is a scientific observation by default, not an immediate workflow blocker. The repair loop is appropriate only when the contradiction plausibly points to a fixable implementation/training defect and the user-facing benchmark result would materially improve from spending the extra training budget. It is symmetric to `/diagnose-verify-block` (same global-hypothesis discipline) but operates on the post-training signal instead of the sanity-ladder signal.

- **Signal, not metric gap.** The loop triggers on CONTRADICTED pre-registered claims, not on absolute metric distance from the paper. Success criterion is that previously-CONTRADICTED claims move to CONFIRMED (or to PLAUSIBLE within the hypothesis tolerance). Relative positioning against the PICID-reproduced baselines — which use the same fixed validation batch, LR, and scheduler policy — is the authoritative axis.
- **Default budget:** `max_iterations = 4` per invocation. The tighter budget (vs. 10 for pre-training) reflects the higher per-iteration cost: each iteration runs a full training pass.
- **Configurable:** the experimenter may pass a smaller budget, but not a larger one.
- **Not bumpable mid-loop:** once the loop starts, the budget is fixed. If 4 iterations do not restore CONFIRMED claim status, the skill returns `ESCALATE`.
- **Per-iteration mandate:** every iteration (a) forms ONE global hypothesis explaining ALL contradicted claims jointly, (b) applies a writable-file change, (c) re-accepts the sanity ladder (`PASS` or `WARN_CONTINUE`; routes through `/diagnose-verify-block` if the change introduced a categorical sanity `BLOCK` — a nested loop that consumes its own separate budget), (d) re-runs full training, and (e) re-invokes `/evaluate-results`. Iteration records are appended to `07-training-log.md`.
- **Claim-tolerance relaxation is forbidden:** lowering the margin in `05-paper-hypothesis.md` to make a CONTRADICTED claim pass is not a repair. The hypothesis is frozen at pre-registration.
- **Retraining on the test split is forbidden**, as is any change to baseline configs, metric definitions, or evaluation code.

## Fixed Batch/Scheduler and Model HP Policy

For `run_mode=full`, `/run-training` uses the batch size, learning rate, optimizer, and scheduler already selected for the experiment profile. Automatic batch-size tuning and LR autoscaling are forbidden for the primary comparison run.

- **Fixed validation batch.** Validation runs must not estimate, inherit, or inspect Hydra chains to choose datamodule batch sizes. Every validation profile uses explicit runtime/config overrides `datamodule.train_batch_size=512`, `datamodule.val_batch_size=1024`, and `datamodule.test_batch_size=1024`. These values are the fair-comparison contract for supported PICID paper datasets (including N-CMAPSS, UNIBO, NB14, and XJTU-SY) and must override composed defaults such as `16`. Never substitute a paper batch size or an ad hoc value such as 32 for a validation run.
- **Scheduler/check cadence.** Preserve the scheduler/checkpoint/early-stopping cadence from the selected PICID optimization/trainer/callback configs. Agents may inspect configs/scripts for scheduler provenance, but not for datamodule batch-size selection.
- **Model-specific HPs.** Values that are genuinely specific to the new paper model — learning rate, optimizer family, weight decay, gradient clipping, warmup, max epochs, dropout/capacity knobs, and similar fields — should follow the paper when stated. When the paper omits them, impute them from the closest existing PICID config/framework default and record a `# SUBSTITUTION:` comment plus the source. The absence of a pre-existing official model-specific HP record for a newly implemented paper model is expected and is not a blocker.
- **Scheduler exception.** Even if the paper states an LR schedule, the primary validation run preserves PICID's scheduler/check cadence unless the user explicitly asks for exact paper-protocol training. Record paper scheduler details as provenance; do not silently replace the framework scheduler for benchmark validation.
- **Profiles** live in `.opencode/reference/hparam-profiles.yaml`, but that file is only a keyed override source for stable reference entries. The fixed validation batch is enforced by the workflow/tooling, not discovered from this YAML. A missing model-specific entry means "use the composed experiment config plus fixed validation batch overrides and paper/imputed model HPs", not `REFERENCE_PROFILE_UNRESOLVED`.
- **Default mode** is `hp_mode=reference_only`, which means one primary validation run using the fixed validation batch, preserved PICID scheduler/check cadence, and paper-stated or explicitly imputed model HPs.
- **Auxiliary paper mode** is `hp_mode=reference_plus_paper`, which may additionally run a paper/default non-batch profile for context. The final verdict remains based on the primary validation profile.
- **Batch size** is fixed before Hydra composition completes: every validation profile appends `datamodule.train_batch_size=512`, `datamodule.val_batch_size=1024`, and `datamodule.test_batch_size=1024` after any other HP overrides so these values win over base datamodule defaults. The preceding `/check-batch-fit` phase only verifies that this fixed configured batch fits on the current hardware; it never chooses a larger or smaller batch.
- **Learning rate** comes from the paper when stated, otherwise from an explicit framework/default imputation in the experiment config. It is not rescaled at runtime because batch size is not changed at runtime.
- **Runtime overrides** may set keyed official/reference values when they exist, but they are optional for novel paper models. Do not halt because a composed config exposes base datamodule defaults such as `16`; the validation profile overrides them to `512/1024/1024`. Halt with `PICID_SCHEDULER_UNRESOLVED` only when the scheduler/check cadence cannot be identified from the composed framework config.
- **OOM handling** does not auto-reduce batch size. If the fixed validation batch does not fit, the run is blocked for this hardware/profile unless the user explicitly selects another profile.
- **Recorded verbatim** in `07-training-log.md` front-matter (`hp_profile_status`, `hp_overrides`, `hp_reference_source`, `batch_size_configured`, `batch_size_chosen`, `batch_fit_check_source`, `lr_paper`, `lr_scaled`, `lr_scaling_rule`) so `/evaluate-results` and `/diagnose-training-result` can read what actually ran. Legacy fields `batch_size_chosen`, `batch_probe_source`, `lr_scaled`, and `lr_scaling_rule` remain present for sidecar compatibility; `batch_probe_source` is `not_used`.

## Scope boundary for automated changes

The entire `/validate-paper` workflow operates under a strict writable-file whitelist. The whitelist is derived per-run (it depends on which files the blueprint and the implementation skills produced) and is passed explicitly into `/diagnose-verify-block` and the implementation skills. No step of the workflow — orchestrator, experimenter, diagnostic loop, or implementation skill — may edit outside its whitelist.

The fixed boundaries are:

- **Framework code is never writable.** Any file under `picid/` that existed at the paper-start commit is off-limits. If the only plausible root cause involves framework code, the workflow escalates with `FRAMEWORK_CHANGE_REQUIRED` — the user must see the framework-level signal and decide.
- **`.opencode/` is never writable by the run.** Skills, agents, policies, and reference files are part of the workflow contract, not per-paper outputs. They are only changed through explicit user-driven edits, never by a paper-validation run.
- **Vault artifacts are only appended to by the skill that owns each file.** `06-sanity-ladder-log.md` is owned jointly by `/verify-static`, `/verify-sanity`, and `/diagnose-verify-block` (the diagnostic loop's iteration records). `07-training-log.md` is owned jointly by `/run-training` (per-run front-matter + body) and `/diagnose-training-result` (training-repair iteration records appended beneath). `08-evaluation-report.md` is owned by `/evaluate-results`. In multi-target runs, the experimenter may copy completed target artifacts into target-scoped archive paths under `{vault_dir}/targets/<target_slug>/` before the next target overwrites canonical filenames; those copies are archival and must not edit artifact contents. No other skill, agent, or loop writes to any vault artifact.
- **Paper-authored new code and configs ARE writable** — specifically the paths enumerated in the blueprint's `required_new_files` plus any file added to the tree since `<paper-start-commit>` that is not under `picid/` or `{vault_dir}/`. These are the files the diagnostic loop, re-invoked implementation skills, and direct small edits (<20 lines per iteration) are allowed to touch.

The experimenter is responsible for computing the whitelist at Phase-E entry and passing it to `/diagnose-verify-block`. Whitelist violations surface as `FRAMEWORK_CHANGE_REQUIRED` or `VAULT_EDIT_ATTEMPTED` and always halt the run.

## Abort-recovery contract

A delegated skill/tool/subagent returning `Tool execution aborted`, an empty result, or an obviously incomplete result is a **recovery trigger, not a terminal verdict**. The orchestrator and experimenter must:

1. Not surface the raw abort as the final workflow outcome.
2. Inspect the vault and repo artifacts already produced.
3. Read the latest relevant artifacts before deciding:
   - `04-implementation-blueprint.md`
   - `06-sanity-ladder-log.md` if present
   - `07-training-log.md` if present
   - `08-evaluation-report.md` if present
   - any newly created code/config files the failed step was expected to write
4. Infer the last successful checkpoint and resume from there, not from scratch.
5. Classify the failure into: dataset executability, implementation/configuration, training instability, evaluation discrepancy, or orchestration/tool-invocation failure.
6. For orchestration/tool-invocation failures, continue the technical workflow using recovered diagnostics — do not stop with "tool aborted" if the underlying issue is identifiable.
7. Retry once with more explicit inputs when the abort is likely from ambiguity or missing context.
8. If a delegated skill aborts twice on the same step, continue with direct tool-based diagnosis and the remaining dependency-ordered workflow whenever possible, preserving vault traceability.
9. Only stop the overall pipeline when a required artifact is still missing after recovery attempts AND the concrete blocker preventing autonomous continuation can be named.

Return a structured final status that either reports evaluation reached, or names the exact hard blocker.
