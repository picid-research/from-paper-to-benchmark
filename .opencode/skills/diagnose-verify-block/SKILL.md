---
name: diagnose-verify-block
description: Resolve a categorical `verdict: BLOCK` from `/verify-sanity` by running a bounded global-hypothesis repair loop. Each iteration re-runs all default-ladder checks, aggregates diagnostics, forms ONE global hypothesis covering all failing checks jointly, applies a change inside a strict whitelist of writable files, and re-runs. Terminates at `PASS`, `WARN_CONTINUE`, or `ESCALATE`.
---

# /diagnose-verify-block

Resolve a categorical `/verify-sanity` BLOCK by repairing the implementation. Non-categorical model sanity failures are `WARN_CONTINUE` and do not enter this loop unless a later training/evaluation signal makes them actionable. This skill replaces the deprecated fixed hyperparameter-adjustment ladder (LR/10 → optimizer swap → gradient clipping).

Two design rules distinguish this skill from ad-hoc debugging:

1. **Global hypothesis, not per-check patching.** Each iteration re-runs ALL default-ladder checks, then forms a single hypothesis that explains EVERY failing check jointly. This prevents the "fix check A, break check B" failure mode — an iteration's change must not regress any previously passing check.
2. **Strict writable-file whitelist.** The skill never edits framework code (anything in `picid/` that existed before the paper run) and never edits vault artifacts. Changes are restricted to files derived from the blueprint's `required_new_files` plus the current paper's experiment/model/loss/transform configs.

## Input contract

The orchestrator must provide:

- **vault_dir**: Path to vault (reads `04-implementation-blueprint.json` for the machine contract and rendered `04-implementation-blueprint.md` only for audit prose, plus `06-sanity-ladder-log.md`; the skill appends new attempts to `06-sanity-ladder-log.md` via re-invocation of `/verify-sanity`)
- **repo_root**: PICID repo root
- **experiment_config**: current Hydra experiment config path
- **task_type**: `regression` | `classification` | `fit_predict`
- **num_classes**: classification only
- **expected_init_loss**: regression expected first loss, or `"auto"`
- **dataset_candidate**: current dataset candidate name/config
- **writable_files**: REQUIRED list of absolute file paths the skill is permitted to edit during this invocation. The experimenter computes this per-run as the union of:
  - every path in `04-implementation-blueprint.json.required_new_files`
  - every file in `git diff --name-only <paper-start-commit> HEAD` that is NOT under `picid/` as it existed at `<paper-start-commit>`
  The skill refuses any edit that falls outside this whitelist.
- **max_iterations**: default `10`. The iteration budget for this invocation; not bumpable mid-loop.

## Entry preconditions

The skill refuses invocation if:

- Neither `{vault_dir}/06-sanity-ladder-log.jsonl` nor a parseable latest attempt in `{vault_dir}/06-sanity-ladder-log.md` exists (the sanity log is the diagnostic source; without it the loop has no starting signal).
- The latest authoritative sanity `verdict` is `PASS` (nothing to repair).
- The latest authoritative sanity `verdict` is `WARN_CONTINUE` (diagnostics are already non-blocking; the experimenter should continue and preserve the warning).
- The latest authoritative sanity `verdict` is `DATASET_UNAVAILABLE` or `DATASET_EXECUTION_FAILED` (dataset problem — the experimenter must trigger fallback per policies.md, not this skill).
- `writable_files` is empty or missing.

## Procedure

### Iteration structure

The skill runs up to `max_iterations` iterations. Each iteration executes steps 1–7 below, in order. Between iterations the skill appends a structured iteration record to `{vault_dir}/06-sanity-ladder-log.md` (via `/verify-sanity` with `write_log: true` on step 6, plus a dedicated iteration block on step 7).

#### Step 1 — Re-run ALL default-ladder checks

Invoke `/verify-sanity` with the full default ladder (`init_loss`, `gradient_flow`, `overfit_batch`). Do NOT scope to the previously failing check — a change from the previous iteration may have regressed a previously passing check, and the loop's correctness depends on catching that.

Parse the freshly appended structured attempt from `06-sanity-ladder-log.jsonl` first, else the markdown front-matter.

- If `verdict: PASS` → loop terminates at this iteration; skill returns `PASS` to the experimenter.
- If `verdict: WARN_CONTINUE` → loop terminates at this iteration; skill returns `WARN_CONTINUE` to the experimenter so training can proceed with the unresolved diagnostics recorded.
- If `verdict: DATASET_UNAVAILABLE` or `DATASET_EXECUTION_FAILED` → abort the loop with `DATASET_RECOVERY_REQUIRED` and return to the experimenter (the dataset problem is out of scope for this skill).
- If `verdict: BLOCK` → continue to Step 2.

#### Step 2 — Aggregate diagnostics across all failing checks

Read `check_diagnostics` from the authoritative structured attempt. Build a single table listing, for each failed check, the numeric signals the tool provided. At minimum:

| Check | Signal | Value | Threshold | Gap |
|-------|--------|-------|-----------|-----|
| init_loss | observed_loss | ... | expected ± tol | ... |
| gradient_flow | dead_layers | N | 0 | N |
| gradient_flow | batch_leak_detected | true/false | false | ... |
| overfit_batch | final_accuracy / final_mae_ratio | ... | PASS / INVESTIGATE band | ... |
| overfit_batch | best_reduction_fraction | ... | diagnostic only | ... |
| overfit_batch | completed_steps | 400 | — | — |

Also list, for every check that CURRENTLY PASSES, the numeric signal you will monitor in the next re-run to confirm the hypothesis did not regress it.

#### Step 3 — Classify each failure

For each failing check, pick exactly one classification:

- `IMPL_BUG` — likely a forward-pass, reshape, masking, or backward wiring defect in the paper-authored model/loss/wrapper code
- `HP_MISMATCH` — likely an optimizer, LR, weight decay, gradient clipping, or scheduler value that is wrong for this model
- `LOSS_COMPOSITION` — likely a composite/multi-term loss whose weights, auxiliary terms, or masking suppress gradient signal for the overfit check
- `DATA_PIPELINE` — likely a transform/collate/batch-construction bug; consider if `zero_input` should be run on-demand to confirm

The classifications per check are independent inputs to Step 4. They do not constrain the final hypothesis — multiple classifications may collapse into one root cause.

#### Step 4 — Form ONE global hypothesis

Write a hypothesis block that has exactly these five fields. The block must explain every failing check JOINTLY — not one hypothesis per failing check.

```yaml
iteration: <n>
global_hypothesis: <one sentence; the single root-cause theory>
target_change:
  file: <absolute path; MUST be in writable_files>
  field_or_symbol: <specific override path / function name / constant>
  from: <current value>
  to: <proposed value>
predicted_effect:
  <check_name>: <numeric prediction, e.g., "micro-batch fit reaches PASS band">
  <check_name>: <numeric prediction for each default-ladder check, including the ones currently passing — prediction for passing checks must be "stays PASS">
falsification_criterion: <a specific observable outcome that would refute this hypothesis; may NOT be post-hoc rationalization>
rationale: <why this single change is expected to resolve every failing check jointly>
```

**If the hypothesis requires editing a file NOT in `writable_files`** (e.g., a framework Python file, a vault markdown, or a file not touched by the paper's implementation), the skill aborts this iteration with `FRAMEWORK_CHANGE_REQUIRED` and escalates to the user. No edit is made.

#### Step 5 — Apply the change

Allowed change kinds:

1. Edit a Hydra config file inside `writable_files` (e.g., experiment YAML, model config, loss config).
2. Re-invoke `/implement-model` or `/implement-loss` with an updated spec derived from the hypothesis. The re-invocation's outputs must themselves be within `writable_files`.
3. Direct small edits (< 20 lines total per iteration) to paper-authored Python files inside `writable_files` (model, loss, wrapper, transform).

Record the exact diff that was applied (file path, lines changed, before/after snippet).

#### Step 6 — Re-run ALL default-ladder checks

Invoke `/verify-sanity` again with the full default ladder and `write_log: true`. Parse the freshly appended structured attempt from `06-sanity-ladder-log.jsonl` first, else the markdown front-matter.

#### Step 7 — Record iteration outcome

Append a block to `{vault_dir}/06-sanity-ladder-log.md` AFTER the attempt that `/verify-sanity` just wrote, with this schema:

```markdown
## Iteration <n> — <CONFIRMED | REFUTED | PARTIAL>

**Hypothesis:** <from step 4>
**Target change:** <file : field, from → to>
**Predicted vs Observed:**

| Check | Predicted | Observed | Regression from prior PASS? |
|-------|-----------|----------|------------------------------|
| init_loss | stays PASS | PASS | no |
| gradient_flow | stays PASS | PASS | no |
| overfit_batch | reaches PASS band | INVESTIGATE | n/a |

**Conclusion:** CONFIRMED / REFUTED / PARTIAL

**Falsifier observed?** <yes/no, with the specific observation>

**Diff applied:**
```diff
<actual diff>
```
```

Conclusion rules:
- `CONFIRMED` — every predicted effect matches observation AND no previously passing check regressed
- `REFUTED` — the falsification_criterion fired, OR the observation diverges materially from every predicted effect
- `PARTIAL` — some predicted effects match, some do not; progress is being made but the root cause is broader than the hypothesis

After recording:

- If the latest attempt's `verdict: PASS` → skill returns `PASS` and exits.
- If the latest attempt's `verdict: WARN_CONTINUE` → skill returns `WARN_CONTINUE` and exits.
- If iteration count `== max_iterations` → skill returns `ESCALATE` with a summary of every iteration's hypothesis/conclusion and the final diagnostic table.
- Otherwise → return to Step 1 for the next iteration. Subsequent hypotheses MUST take into account the prior iterations' evidence (a REFUTED hypothesis should not be re-proposed without new diagnostics; a PARTIAL should narrow the next hypothesis to the unexplained residual).

## Forbidden actions (hard guardrails)

The skill refuses to perform any of these and, if the hypothesis implies them, aborts with `FRAMEWORK_CHANGE_REQUIRED` or `VAULT_EDIT_ATTEMPTED`:

- Edit any path under `picid/` that existed at the paper-start commit (framework code).
- Edit any `.opencode/` skill, agent, policy, or reference file.
- Edit any vault artifact (`{vault_dir}/00-…` through `08-…`). `06-sanity-ladder-log.md` is **appended to** by `/verify-sanity` on each iteration and by this skill's Step 7 iteration record — those specific append operations are the only vault writes this skill performs.
- Relax the sanity decision bands themselves. `WARN_CONTINUE` is emitted by `/verify-sanity` only when the check remains executable; manually re-labeling a categorical `BLOCK` is hiding the signal.
- Increase `max_iterations` mid-loop.

## Termination and return values

The skill returns exactly one of:

- `PASS` — every default-ladder check now passes; the orchestrator may proceed to `/run-training`.
- `WARN_CONTINUE` — no categorical sanity blocker remains, but one or more model diagnostics still failed. The orchestrator proceeds to `/check-batch-fit` and `/run-training` while preserving the warning.
- `ESCALATE` — iteration budget exhausted without `PASS` or `WARN_CONTINUE`. The orchestrator MUST surface the full iteration history to the user and not silently retry.
- `FRAMEWORK_CHANGE_REQUIRED` — the only plausible hypothesis involves editing framework code. User intervention required.
- `DATASET_RECOVERY_REQUIRED` — during an iteration, the sanity check reported a dataset availability/execution failure. The experimenter repairs the same dataset or stops with a named blocker per policies.md.
- `VAULT_EDIT_ATTEMPTED` — an internal guardrail violation; should never occur in practice. User intervention required.

## Worked example (illustrative; NOT prescriptive)

The hypothesis below is just one possible iteration path for a composite-loss model whose `overfit_batch` never reaches the PASS band while `init_loss` and `gradient_flow` pass. The skill generates hypotheses from whichever diagnostics the current paper exposes; do not hard-code this path.

| Iter | Global hypothesis | Target change | Prediction (all three checks) | Falsifier | Conclusion |
|------|-------------------|---------------|-------------------------------|-----------|------------|
| 1 | An auxiliary loss term dominates the gradient and prevents micro-batch memorization | experiment YAML: `loss.aux_weight: 1.0 → 0.0` | init_loss stays PASS; gradient_flow stays PASS; overfit_batch reaches PASS band | overfit remains FAIL | CONFIRMED / REFUTED |
| 2 (if 1 REFUTED) | LR is too low for the micro-batch memorization probe | experiment YAML: `optimization.lr *= 10` | same predictions | overfit remains FAIL | ... |
| 3 (if 1 CONFIRMED) | Aux-loss branch is not receiving gradient in the first place | re-invoke `/implement-model` with fix spec for aux branch | same predictions with aux_weight restored | baseline overfit flat | ... |

## Common pitfalls

- **Per-check hypotheses.** "Fix overfit by lowering LR, fix init_loss by rescaling targets" is two hypotheses. The loop requires ONE. If you genuinely have two independent root causes, pick the likelier one first and leave the other for the next iteration after observing which passes and which doesn't.
- **Post-hoc falsification.** The `falsification_criterion` must be written BEFORE the re-run. Rewriting it afterwards defeats the whole purpose.
- **Silent regression.** The loop only works if every iteration re-runs every check. Scoping to the previously failing check is a bug — don't do it.
- **Scope creep into framework code.** Tempting, but forbidden. If the root cause really is in framework code, the paper's implementation is also exposing a framework issue that the user must see and decide on.
- **Decision-band relaxation.** "call this PASS even though a diagnostic still clearly fails" is not a repair. If the failure is non-categorical, `/verify-sanity` emits `WARN_CONTINUE`; if it is categorical, the loop must repair it or escalate.
