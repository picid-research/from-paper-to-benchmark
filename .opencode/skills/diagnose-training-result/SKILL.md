---
name: diagnose-training-result
description: Optionally resolve an `INVESTIGATE_CLAIMS_DISPUTED` scientific verdict from `/evaluate-results` after the scheduled validation run matrix is complete. Runs a bounded global-hypothesis repair loop only when the contradiction plausibly indicates a fixable implementation/training defect. Each iteration reads the evaluation report + training log + pre-registered hypothesis, forms ONE critical hypothesis targeting the contradicted relative-positioning claims, applies a change inside a strict whitelist of writable files, re-runs sanity + full training + evaluation, and repeats. Terminates at `PASS` (all contradicted claims now CONFIRMED) or `ESCALATE` (iteration budget exhausted).
---

# /diagnose-training-result

Resolve an `INVESTIGATE_CLAIMS_DISPUTED` verdict by repairing the training recipe only after the experimenter has completed all required scheduled validation rows. This skill is the post-training counterpart to `/diagnose-verify-block`: same design (global-hypothesis loop, strict writable-file whitelist, threshold-relaxation forbidden) but different signal — it uses `/evaluate-results` claim verdicts from `05-paper-hypothesis.md`, not sanity-check diagnostics.

Two rules distinguish this skill from ad-hoc post-training tuning:

1. **Relative positioning is the signal, not absolute metrics.** The loop optimizes for `CONFIRMED` on the pre-registered claims that were `CONTRADICTED`. Absolute metric gaps (20% / 50% / FAR) are not the trigger and not the success criterion.
2. **One global hypothesis per iteration.** Each iteration forms a single hypothesis that jointly explains every contradicted claim. If the hypothesis implies multiple independent root causes, pick the dominant one and leave the rest for the next iteration based on observed evidence.

## Input contract

The orchestrator must provide:

- **vault_dir**: Path to vault. Reads `05-paper-hypothesis.md`, `07-training-log.md`, `08-evaluation-report.md`; iteration records append to `08-evaluation-report.md` (via re-invocation of `/evaluate-results`) and to a new section at the tail of `07-training-log.md` that this skill owns.
- **repo_root**: PICID repo root
- **experiment_config**: Hydra experiment config path
- **task_type**: `regression` | `classification` | `fit_predict`
- **dataset_candidate**: current dataset candidate name/config
- **writable_files**: REQUIRED list of absolute file paths the skill is permitted to edit (same contract as `/diagnose-verify-block`: union of blueprint's `required_new_files` and `git diff --name-only <paper-start-commit> HEAD` minus framework code and vault artifacts).
- **max_iterations**: default `4`. Not bumpable mid-loop.
- **evaluate_inputs**: the dict of inputs the experimenter passes to `/evaluate-results` (so the skill can re-invoke it with the same shape each iteration).
- **run_training_inputs**: the dict of inputs the experimenter passes to `/run-training` (so the skill can re-invoke full training each iteration; the skill itself does not invent run-mode or epoch rationale values).

## Entry preconditions

The skill refuses invocation if:

- `{vault_dir}/08-evaluation-report.md` does not exist, or its classification is not `INVESTIGATE_CLAIMS_DISPUTED`. Other verdicts (`VALIDATED`, `PLAUSIBLE`, `BENCHMARK_ONLY`, `INVESTIGATE`, `IMPLEMENTATION_BUG`) are not handled here — `IMPLEMENTATION_BUG` belongs back in Phase B, `INVESTIGATE` without disputed claims is reported to the user but not looped on.
- The experimenter has not completed every remaining `required_for_benchmark_validation=true` row in the blueprint's validation run matrix. A disputed claim in one row must not starve independent planned rows.
- The disputed claims are only `DATASET_DEPENDENT`, `UNASSESSABLE`, `UNASSESSABLE_METRIC_SCALE`, or missing-baseline observations. Those are scientific caveats, not repair signals.
- `{vault_dir}/05-paper-hypothesis.md` status is `BENCHMARK_ONLY` — no baselines to validate against, no signal for the loop to optimize.
- `{vault_dir}/07-training-log.md` front-matter `status` is not in `{SUCCESS, PARTIAL, FAILED}`. A `CRASHED` training log means training itself failed, which belongs in the experimenter's Phase F decision table, not here.
- `writable_files` is empty or missing.

## Procedure

Each iteration executes steps 1–8 below in order. After step 8, either return `PASS`/`ESCALATE`/guardrail failure, or start the next iteration.

### Step 1 — Read the current signal

Read from the vault:

1. `08-evaluation-report.md` — the "Paper Claims Validation" section, in particular every claim marked `CONTRADICTED`. Capture for each: claim text, paper's reported relation ("beats baseline X"), our reproduced result, baseline values, the specific metric the claim is phrased on.
2. `05-paper-hypothesis.md` — the pre-registered baseline table and any per-claim predictions.
3. `07-training-log.md` — front-matter (`hp_overrides`, `hp_reference_source`, `batch_size_configured`, `batch_size_chosen`, `batch_fit_check_source`, `lr_paper`, `lr_scaled`, `lr_scaling_rule`, `optimizer_family`, `epochs_run`, `early_stopping_triggered`, health-warning summary) and loss-curve body table.
4. The latest resolved config from the training output directory's `config_resolved.yaml` when needed to resolve override paths.

### Step 2 — Diagnose from the post-training signal

Build a single diagnostic block:

| Contradicted claim | Paper relation | Our metric | Baseline metric | Gap | Plausible root causes |
|--------------------|----------------|------------|-----------------|-----|------------------------|
| "model beats MLP" | model < MLP (lower better) | 12.8 | 10.4 | -2.4 (worse) | scaled LR too aggressive / undertraining / capacity mismatch / regularization wrong |

Cross-reference with `07-training-log.md`:

- Was the loss curve healthy, flat, diverging, or noisy?
- Did early stopping trigger early (under-training) or never trigger (possibly over-training)?
- Do the resolved batch size, LR, optimizer, and scheduler match the intended comparison profile?
- Are the framework baselines run under the same official/reference profile? (They should be — if not, the comparison is unfair and the right fix is upstream in baseline retrieval, not here.)

### Step 3 — Classify each contradicted claim

For each contradicted claim, pick exactly one dominant hypothesis class:

- `LR_TOO_AGGRESSIVE` — configured LR likely exceeds the stable range for this optimizer/model.
- `UNDER_TRAINING` — early stopping or epoch ceiling cut the run before convergence; loss curve was still decreasing at the end.
- `OVER_TRAINING` — train loss bottomed out but val loss rose; need stronger regularization or fewer epochs.
- `CAPACITY_MISMATCH` — model is too small (val loss plateaus high) or too large (severe overfitting); check vs baseline architectures.
- `REGULARIZATION_MISMATCH` — weight_decay, dropout, or grad_clip is inconsistent with the selected comparison profile.
- `OPTIMIZER_FAMILY_MISMATCH` — paper/profile uses SGD but blueprint inlined Adam (or vice versa); dynamics are wrong.
- `UNKNOWN` — signal is inconclusive; additional probe (e.g., one-epoch LR sweep) needed before hypothesis.

These classifications inform but do not constrain Step 4's single global hypothesis.

### Step 4 — Form ONE global hypothesis

```yaml
iteration: <n>
global_hypothesis: <one sentence; the single root-cause theory that explains every CONTRADICTED claim>
target_change:
  file: <absolute path; MUST be in writable_files>
  field_or_symbol: <specific override path / function name / constant>
  from: <current value>
  to: <proposed value>
predicted_effect:
  <claim id or short text>: <prediction, e.g., "CONFIRMED — model metric drops below MLP baseline">
  <claim id>: <prediction for every currently-CONFIRMED claim too: must be "stays CONFIRMED">
falsification_criterion: <specific observable outcome that would refute this hypothesis>
rationale: <why this single change is expected to flip every CONTRADICTED claim to CONFIRMED without regressing any CONFIRMED one>
```

If the hypothesis requires editing a file NOT in `writable_files`, abort the iteration with `FRAMEWORK_CHANGE_REQUIRED`. No edit is made.

### Step 5 — Apply the change

Allowed change kinds per iteration (all must target `writable_files`):

- Edit the experiment YAML's inlined non-batch HP overrides (`optimization.lr`, `optimization.optimizer.weight_decay`, `trainer.gradient_clip_val`, scheduler settings). Do not tune `datamodule.train_batch_size`, `datamodule.val_batch_size`, or `datamodule.test_batch_size`; validation runs must remain fixed at train `512`, val `1024`, and test `1024`.
- Re-invoke `/implement-model` or `/implement-loss` with an updated spec targeting a capacity / regularization change.
- Direct small edits (< 20 lines per iteration) to paper-authored Python files in `writable_files`.

Record the exact diff (file path, before/after snippet).

### Step 6 — Re-accept sanity gate

Invoke `/verify-sanity` on the changed configuration. Parse the new front-matter. If `verdict` is `PASS` or `WARN_CONTINUE`, continue; preserve any `WARN_CONTINUE` diagnostics in the iteration record. If `verdict` is anything else:

- Route into `/diagnose-verify-block` with `max_iterations=10` and the SAME `writable_files`. This prevents the post-training loop from leaving the model in a sanity-broken state.
- If `/diagnose-verify-block` returns `PASS` or `WARN_CONTINUE`, continue to Step 7.
- If `/diagnose-verify-block` returns `ESCALATE` or `FRAMEWORK_CHANGE_REQUIRED`, abort this skill's loop with the same verdict and surface the sanity-iteration history alongside the training-iteration history.

### Step 7 — Re-run full training

Re-invoke `/run-training` with `run_mode=full` and the `run_training_inputs` (including a fresh `epoch_budget_rationale` that references this iteration's change). `/run-training` must use the resolved config values directly; do not pass autoscale or batch-tuning inputs.

Parse the new `07-training-log.md` front-matter:

- `status: SUCCESS | PARTIAL` → continue to Step 8.
- `status: FAILED | CRASHED | DATASET_*` → this iteration's change broke something more fundamental than the relative positioning. Record the outcome, return control to the experimenter with an `ESCALATE` verdict and the iteration history — do not continue looping.

### Step 8 — Re-invoke `/evaluate-results` and record the iteration

Re-invoke `/evaluate-results` with `evaluate_inputs`. Parse the new `08-evaluation-report.md` classification and Paper Claims Validation section.

Append an iteration record as a new `## Iteration <n>` section at the tail of `07-training-log.md` (this skill owns this specific append; no other writer touches the file after initial training):

```markdown
## Iteration <n> — CONFIRMED / REFUTED / PARTIAL

**Hypothesis:** <from step 4>
**Target change:** <file : field, from → to>
**Sanity acceptance:** PASS / WARN_CONTINUE (via /diagnose-verify-block if applicable)

**Predicted vs Observed claim verdicts:**

| Claim | Predicted | Observed | Regression from prior CONFIRMED? |
|-------|-----------|----------|----------------------------------|
| ... | CONFIRMED | CONFIRMED | no |

**Falsifier observed?** <yes/no, with specific observation>

**Evaluation verdict:** VALIDATED / PLAUSIBLE / INVESTIGATE_CLAIMS_DISPUTED / ...

**Diff applied:**
```diff
<actual diff>
```
```

Conclusion rules:

- `CONFIRMED` — every predicted claim outcome matches observation AND no previously CONFIRMED claim regressed AND the evaluate-results verdict is `VALIDATED` or `PLAUSIBLE`.
- `REFUTED` — falsification_criterion fired, OR the observation diverges materially from prediction.
- `PARTIAL` — some claims flipped to CONFIRMED but not all; narrow the next iteration's hypothesis to the residual.

If `CONFIRMED` → return `PASS`.
Else if iteration count `== max_iterations` → return `ESCALATE` with full iteration history.
Else → next iteration.

## Forbidden actions (hard guardrails)

The skill refuses any of these; the hypothesis that implies them aborts with `FRAMEWORK_CHANGE_REQUIRED` or `VAULT_EDIT_ATTEMPTED`:

- Edit framework code under `picid/` that existed at the paper-start commit.
- Edit any `.opencode/` file.
- Edit any vault artifact except the specific append operations defined in Step 8 (to `07-training-log.md`) and the `/evaluate-results` / `/verify-sanity` / `/diagnose-verify-block` re-invocations that own their own artifacts.
- Relax a claim's tolerance in `05-paper-hypothesis.md` to make a CONTRADICTED claim trivially CONFIRMED. The pre-registered predictions are the contract.
- Retrain on the test split or leak test data through the training pipeline.
- Increase `max_iterations` mid-loop.

## Termination and return values

- `PASS` — all previously CONTRADICTED claims are now CONFIRMED AND the evaluate-results verdict is `VALIDATED` or `PLAUSIBLE`. Orchestrator proceeds to Phase H.
- `ESCALATE` — budget exhausted without `PASS`. Orchestrator surfaces the full iteration history to the user. Do not silently retry or bump budget.
- `FRAMEWORK_CHANGE_REQUIRED` — only plausible hypothesis involves editing framework code. User intervention.
- `VAULT_EDIT_ATTEMPTED` — internal guardrail violation. User intervention.
- `SANITY_UNRECOVERABLE` — `/diagnose-verify-block` could not restore `PASS` or `WARN_CONTINUE` during Step 6, and no further hypothesis is possible inside the writable-files whitelist. User intervention.

## Worked example (illustrative; NOT prescriptive)

A paper reports its model beats MLP and matches Transformer on RUL. Our full run produces:

- Model RMSE: 12.8 (baseline MLP: 10.4, baseline Transformer: 11.9)
- Claim "beats MLP" → CONTRADICTED. Claim "matches Transformer" → CONTRADICTED.
- Training log shows `lr_paper=0.001`, `lr_scaled=0.001` (`lr_scaling_rule=none`), loss curve descended but plateaued at epoch 30 of 100, early stopping never fired.

Possible iteration path:

| Iter | Global hypothesis | Target change | Prediction | Falsifier | Conclusion |
|------|-------------------|---------------|------------|-----------|------------|
| 1 | configured LR is too aggressive for this model under the reference batch | experiment YAML: lower `optimization.lr` within the selected profile's allowed repair scope | both claims flip to CONFIRMED | model RMSE worse than 12.8 | CONFIRMED / REFUTED |
| 2 (if 1 REFUTED) | under-training — loss still decreasing, need more epochs | raise epoch ceiling, ensure early stopping enabled | both claims flip to CONFIRMED | val loss plateau identical | ... |
| 3 (if 1 CONFIRMED) | — not needed, skill returns PASS | | | | |

## Common pitfalls

- **Chasing absolute metrics.** "Our model got 12.8 but paper said 10.2" is not the signal this loop responds to. CONTRADICTED / CONFIRMED against the relative baseline is.
- **Per-claim hypotheses.** One hypothesis must explain all CONTRADICTED claims jointly.
- **Relaxing tolerances.** Changing the claim predictions in `05-paper-hypothesis.md` is forbidden. If you think a claim was badly pre-registered, escalate — don't quietly edit.
- **Skipping the sanity re-acceptance.** Step 6 is not optional. A change that fixes relative positioning but introduces a categorical sanity `BLOCK` is still a broken model; `WARN_CONTINUE` must be recorded, not hidden.
- **Unbounded tuning.** The 3-iteration budget is tight on purpose. Exhaustion escalates to the user.
