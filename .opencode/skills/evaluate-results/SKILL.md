---
name: evaluate-results
description: After training completes, assess whether the implemented model behaves correctly and is competitive within the PICID framework, AND judge each pre-registered paper claim from 05-paper-hypothesis.md. Compares achieved metrics against PICID's frozen internal baselines (report_output/*/results.nc via the PICID_baselines tool) and the paper's reported results using magnitude sanity, relative positioning, behavioral consistency, and a claim-by-claim verdict — accounting for the fact that the training dataset is often different from the paper's dataset.
---

# /evaluate-results

Assess whether a trained model behaves correctly and is competitive within PICID.

The paper's dataset is often NOT the same as the datasets available in PICID. We train
on PICID's existing datasets, which may differ from what the paper used. This means direct
numeric comparison with paper results is usually invalid. Instead, evaluation uses three axes:

1. **Magnitude sanity** — are results in a reasonable range for this domain/task?
2. **Relative positioning** — does the model beat/match baselines by similar margins as the paper claims?
3. **Behavioral consistency** — does the model show the same strengths/weaknesses the paper describes?

Evaluation is a report-writing phase, not a workflow blocker by default. Keep these statuses separate:

- `artifact_status`: whether `08-evaluation-report.md` and `08-evaluation-report.json` were written completely.
- `technical_status`: whether the implementation/evaluator appears healthy (`PASS`) or the result points to a concrete code/config/evaluator problem (`IMPLEMENTATION_BUG` / `EVALUATOR_ERROR`).
- `scientific_status`: the paper-comparison verdict (`VALIDATED`, `PLAUSIBLE`, `INVESTIGATE`, `INVESTIGATE_CLAIMS_DISPUTED`, `BENCHMARK_ONLY`).

A completed report with `INVESTIGATE`, `INVESTIGATE_CLAIMS_DISPUTED`, metric-scale ambiguity, missing overlapping baselines, or `BENCHMARK_ONLY` is still a successful evaluation artifact unless `technical_status` is a bug/error.

## Input contract

The orchestrator provides:

- **vault_dir**: path to vault (reads blueprint + training log, writes `08-evaluation-report.md`, updates `00-paper-hub.md`)
- **repo_root**: PICID repo root
- **experiment_config**: Hydra experiment config used for training
- **training_output_dir**: path to the completed training run's output directory (from `07-training-log.md`)
- **task_type**: one of `prognostics` | `diagnostics` | `anomaly_detection` | `forecasting` | `state_forecasting`
- **dataset_used**: which PICID dataset was trained on (e.g., `pronostia`, `n_cmapss`)
- **paper_dataset**: what dataset the paper used (may differ from `dataset_used`)
- **datasets_are_same**: boolean — whether `paper_dataset == dataset_used`
- **dataset_recovery**: optional dict with `occurred`, `dataset`, `reason`, and same-dataset action. Cross-dataset fallback is disabled.
- **comparison_mode**: `exact_reproduction` | `framework_validation` | `benchmark_only`; do not change comparison mode by switching datasets
- **paper_baselines**: dict of `baseline_name -> metrics` as reported in the paper (e.g., `{"MLP": {"RMSE": 18.5}, "LSTM": {"RMSE": 14.2}, "proposed": {"RMSE": 12.5}}`)
- **framework_baselines**: list of experiment configs for existing PICID models trained on the same `dataset_used` (e.g., `["pronostia/prognostics/mlp", "pronostia/prognostics/cnn1d"]`) — used only as a fallback when `report_output/` has no matching folder
- **run_seeds**: list of seeds used (e.g., `[42, 123, 7]`)
- **hypothesis_path**: path to the pre-registered hypothesis, typically `{vault_dir}/05-paper-hypothesis.md`. Required for full claims-validation output; when absent, the skill degrades to the three-axis eval only and prints a warning.
- **report_output_root**: optional override for `<repo_root>/report_output`

## Procedure

### Step 0 — Read framework context

```
Read `.opencode/reference/evaluators.md` (evaluator + metric contracts) and the §4.5/§4.6 entries in `.opencode/reference/inventory.md` (existing metrics and evaluators).
Read logging-guide.md for where predictions and metrics are stored.
Read {vault_dir}/03-algorithmic-spec.json.training_hyperparameters to distinguish paper-stated values from `NOT_SPECIFIED` omissions.
Read {vault_dir}/04-implementation-blueprint.json.verification_protocol.
Read {vault_dir}/04-implementation-blueprint.json.paper_hyperparameters and the readiness-gate fields to collect config substitutions, framework defaults, dataset target/exclusion decisions, and any `# SUBSTITUTION:` notes carried into the experiment config. Use the rendered Markdown audit files only for human-readable source snippets.
Read {vault_dir}/07-training-log.md for training results and output directories.
```

### Step 1 — Collect achieved metrics

Read metrics from `07-training-log.md` (already extracted by the run-training skill).

If multiple seeds were run:
- Collect metrics from each run
- Compute mean +/- std per metric
- Use mean for all comparisons

For regression-like tasks (`prognostics`, `forecasting`, `state_forecasting`), carry both RMSE and MAE through the evaluation whenever both are logged. Do not reduce the evaluation to a single primary metric unless one counterpart is genuinely unavailable; in that case, record which metric is missing.

If predictions exist (`eval_details/test/predictions.nc`), note they're available
for deeper analysis but don't recompute unless needed.

### Step 2 — Collect framework baseline metrics

PICID has a frozen internal leaderboard at `report_output/<dataset_folder>/results.nc` produced by `picid_report`. That is the **preferred source** — it is consistent across papers and never re-computed.

**Approach A (preferred): load from report_output via the PICID_baselines tool**

Invoke the OpenCode tool `load_baselines` (backed by `.opencode/tools/PICID_baselines/`):

```
load_baselines(
  dataset = dataset_used,
  task_type = task_type,
  subtask = subtask,                       # only for hsf15 / concepts_n_cmapss variants
  metric_keys = <same keys used in 05-paper-hypothesis.md>,
  report_output_root = report_output_root, # optional override
)
```

For regression-like tasks, the metric key list should include the RMSE/MAE pair with the same split and scale, for example `["test/rmse_denormalized", "test/mae_denormalized"]` or `["test/rmse_normalized", "test/mae_normalized"]`.

Act on the returned `status`:

| Status | Action |
|--------|--------|
| `OK` | Use the `rows` directly — no re-running of configs. Build the framework leaderboard below. |
| `METRIC_MISSING` | Retry once with the intersection of requested and `available_metric_keys`. Record which variant was chosen. |
| `FOLDER_NOT_FOUND` | Fall back to Approach B. |
| `CORRUPT_NETCDF` | Stop and report the tool's `reason` field — something is wrong with the leaderboard file itself. |

**Approach B (fallback): existing run artifacts**

Use this only when Approach A returned `FOLDER_NOT_FOUND`.

```bash
# Find most recent run for this baseline
find {repo_root}/artifacts/ -path "*/{baseline_experiment}/**/metrics.csv" -type f | sort | tail -1
```
Parse the CSV for test metrics.

Re-running baselines is allowed only with explicit experimenter approval:
```bash
uv run python -m picid.run experiment={baseline_experiment} logger=csv seed=42
```

**Approach C: Leaderboard scripts**

Look in `scripts/leaderboard/` or `picid_report/` for any pre-aggregated summaries beyond `results.nc`.

Build the framework leaderboard table:

| Model | RMSE | MAE | {other metrics} |
|-------|------|-----|-----------------|
| MLP | 18.5 | 14.1 | ... |
| CNN1D | 15.2 | 11.8 | ... |
| NEW_MODEL | 13.1 | 9.2 | ... |

If no framework baselines exist for this dataset, note "First model on this dataset —
no framework comparison possible" and skip to Step 4.

### Step 2b — Generate baseline comparison plot

Once the framework leaderboard table is assembled, generate baseline comparison plots to embed in the report.

For regression-like tasks, create one plot for RMSE and one plot for MAE when both metrics are available. For non-regression tasks, create the usual plot for the selected primary metric.

Construct an inline JSON payload from the leaderboard rows for each plotted metric:

```json
{
  "baselines": {"MLP": 14.2, "CNN1D": 15.8, ...},
  "new_model_name": "{new_model_class_name}",
  "new_model_value": {achieved_mean_for_this_metric}
}
```

Then run:

```bash
mkdir -p {vault_dir}/plots
uv run python {repo_root}/.opencode/tools/plot_baselines.py \
  --input_json '<json payload>' \
  --metric_key "{metric_key}" \
  --lower_is_better {true|false} \
  --output_path {vault_dir}/plots/baseline_comparison_{metric_slug}.png \
  --title "{dataset_used} / {task_type} — {metric_key} Baseline Comparison"
```

- Set `--lower_is_better false` for metrics like accuracy, R², AUC.
- Set `--lower_is_better true` for RMSE and MAE.
- If the script prints `PLOT_OK: ...`, record the path for the report.
- If it prints `PLOT_FAILED: ...`, note the reason in the report but continue.
- Skip this step when `status == FIRST_ENTRY` (no baselines to compare against).

### Step 2.5 — Load the pre-registered hypothesis

Open `hypothesis_path` (typically `{vault_dir}/05-paper-hypothesis.md`). If the file is missing, emit a warning, set `hypothesis_status = ABSENT`, and skip the claims-validation section — the three-axis eval still runs.

When present, parse these elements:

1. **Hypothesis status** from the top header — one of `PRE_REGISTERED | BENCHMARK_ONLY`. Stash for the final report.
2. **Context block** — `paper_dataset`, `framework dataset used`, `datasets_are_same`, `comparison_mode`, `metric keys under evaluation`, `baseline source`. Validate:
   - Metric keys match what you used in Step 2. If not, use the hypothesis's keys (it is the authoritative pre-registration) and re-query the baseline tool if necessary.
   - `framework dataset used` in the hypothesis matches the training `dataset_used`. If not, set `hypothesis_status = STALE_DATASET_MISMATCH`, skip the claims-validation section, and record the mismatch in the final report's Discrepancy Analysis.
3. **Pre-Registered Claims** — a list under the `## Pre-Registered Claims` heading. Each `### C<N>:` block carries:
   - Claim text (verbatim paper text on the `### C<N>` line).
   - `Type:` one of `relative_gap | absolute_value | behavioral`.
   - `Paper baseline` / `PICID baseline` mapping.
   - `Numerical prediction` — an expression over achieved metrics and baseline metrics.
   - `Confirmation rule` — the threshold and the UNASSESSABLE condition.

Build an in-memory list `claims = [{id, text, type, paper_baseline, PICID_baseline, prediction, rule}, ...]`.

When judging claims, use these verdicts:

| Verdict | Meaning |
|---------|---------|
| `CONFIRMED` | The observed value satisfies the pre-registered rule on the selected metric. |
| `CONTRADICTED` | The observed value directly violates an assessable relative-positioning or same-dataset rule. |
| `DATASET_DEPENDENT` | The claim is absolute or protocol-specific and the validation is `framework_validation` or otherwise cross-dataset. |
| `UNASSESSABLE` | Required data, metric, baseline, or logged behavior is unavailable. |
| `UNASSESSABLE_METRIC_SCALE` | The paper value and PICID metric key are on incompatible or ambiguous scales (for example normalized vs denormalized) and no reliable conversion is pre-registered. |
| `UNASSESSABLE_NO_OVERLAPPING_BASELINE` | The paper baseline has no PICID counterpart for relative-gap validation. |

Metric-scale mismatches are observations, not claim contradictions. If the paper reports normalized values but the frozen hypothesis or framework metric is `test/rmse_denormalized` (or vice versa), do not mark an absolute-value claim `CONTRADICTED` unless an explicit conversion was pre-registered before training. Mark it `UNASSESSABLE_METRIC_SCALE` or `DATASET_DEPENDENT` and explain the mismatch.

### Step 3 — Three-axis evaluation

#### Axis 1: Magnitude sanity

Are the absolute metric values in a reasonable range for this domain and task?

This is a coarse check — not a precision comparison:
- For RUL prediction on bearing data, RMSE in the range 5-50 is typical
- For RUL on turbofan (N-CMAPSS), RMSE in the range 10-30 is typical
- For classification, accuracy should be well above random chance (1/C)
- For forecasting, error metrics should be lower than naive persistence baseline

If `datasets_are_same=true`, this becomes a direct comparison: achieved metrics
should be in the same ORDER OF MAGNITUDE as paper values.

If `datasets_are_same=false`, compare against the range of results that existing
framework baselines achieve on `dataset_used` — the new model should be in a similar
range, not orders of magnitude off.

**Verdict**: `REASONABLE` | `SUSPICIOUS` (way too high/low) | `BROKEN` (NaN, zero, or absurd values)

#### Axis 2: Relative positioning (the key evaluation)

This is the most informative check when datasets differ.

The paper reports relative performance against baselines. The core idea: if the paper
claims "our method beats MLP by 32%", and we also have an MLP baseline in PICID,
check whether the new model shows a similar relative improvement over our MLP.

**Procedure:**

1. Identify overlapping baselines between the paper's table and PICID's available models:
   - MLP -> `picid/model/methods/mlp.py`
   - CNN/1D-CNN -> `picid/model/methods/cnn_1d.py`
   - LSTM -> `picid/baselines/lstm_model/`
   - Transformer variants -> PatchTST, Crossformer, etc. in `picid/baselines/`
   - Linear models -> `picid/baselines/linear_model/`

2. For each overlapping baseline, compute:
   ```
   relative_gap_paper = (paper_proposed - paper_baseline) / paper_baseline
   relative_gap_ours  = (our_model - our_baseline) / our_baseline
   consistency = abs(relative_gap_paper - relative_gap_ours) < 0.15
   ```

3. Use a threshold of 15 percentage points (e.g., paper claims 30% improvement,
   we see 20% improvement -> difference is 10pp -> CONSISTENT).

If no baselines overlap between paper and framework, skip this axis and note
"No overlapping baselines for relative comparison."

**Verdict**: `CONSISTENT` | `PARTIALLY_CONSISTENT` | `INCONSISTENT` | `NOT_ASSESSABLE`

#### Axis 3: Behavioral consistency

Check qualitative claims from the paper (extracted in the blueprint) against observed behavior:

- "Model performs better on longer sequences" -> check if metrics improve with
  sequence length (if per-unit data available with varying lengths)
- "Model is robust to noise" -> harder to test, note as unchecked
- "Model converges faster than baselines" -> compare epochs-to-convergence from
  training logs
- "Model has lower complexity / fewer parameters" -> check model parameter count

Not all behavioral claims can be verified. List which ones were checked and which
were not assessable.

**Verdict**: `CONSISTENT` | `PARTIALLY_CONSISTENT` | `NOT_ASSESSABLE`

### Step 4 — Final classification

Combine the three axes:

| Magnitude | Relative Positioning | Behavioral | Classification |
|-----------|---------------------|------------|----------------|
| REASONABLE | CONSISTENT | CONSISTENT | **VALIDATED** — implementation appears correct |
| REASONABLE | CONSISTENT | NOT_ASSESSABLE | **VALIDATED** — high confidence |
| REASONABLE | PARTIALLY_CONSISTENT | any | **PLAUSIBLE** — likely correct, minor discrepancies |
| REASONABLE | INCONSISTENT | any | **INVESTIGATE** — model works but doesn't match paper's relative claims |
| REASONABLE | NOT_ASSESSABLE | any | **BENCHMARK_ONLY** — works on our data, can't compare to paper |
| SUSPICIOUS | any | any | **INVESTIGATE** — results are unusual for this domain |
| BROKEN | any | any | **IMPLEMENTATION_BUG** — go back to verify-sanity |

**Claim-driven scientific verdict:** if the Paper Claims Validation step returns any `CONTRADICTED` verdicts and the three-axis classification would otherwise be `VALIDATED` or `PLAUSIBLE`, set `scientific_status` / `Classification` to `INVESTIGATE_CLAIMS_DISPUTED`. This is a completed evaluation with disputed scientific evidence, not an implementation failure by itself. `DATASET_DEPENDENT`, `UNASSESSABLE`, `UNASSESSABLE_METRIC_SCALE`, and `UNASSESSABLE_NO_OVERLAPPING_BASELINE` verdicts do not trigger this downgrade — they are expected whenever datasets, metrics, or baselines do not align.

Set `technical_status=IMPLEMENTATION_BUG` only when the achieved metrics or artifacts indicate a concrete runnable-code problem (NaN/Inf, absurd zero/constant outputs, missing test metrics after training reported success, evaluator crash, or a result that contradicts already-accepted sanity in a way that requires returning to implementation). Weak competitiveness or paper-claim disagreement alone is `technical_status=PASS`.

Additional classification based on framework competitiveness:
- **COMPETITIVE**: new model is in top 50% of framework baselines on this dataset
- **MIDDLE**: between 50th and 75th percentile
- **NOT_COMPETITIVE**: bottom 25%
- **FIRST_ENTRY**: no baselines to compare against

### Step 5 — Write to vault

#### Create `{vault_dir}/08-evaluation-report.md`:

```markdown
# Evaluation Report

**Timestamp**: {datetime}
**Paper**: [[00-paper-hub]]
**Blueprint**: [[04-implementation-blueprint]]
**Training**: [[07-training-log]]

## Context
- **Paper's dataset**: {paper_dataset}
- **Framework dataset used**: {dataset_used}
- **Same dataset**: {yes/no}
- **Comparison mode**: {comparison_mode}
- **Task type**: {task_type}

## Paper Evidence vs Framework Choices

Summarize what was directly stated by the paper versus what the validation workflow had to impute, substitute, or inherit from PICID. Build this section by going back to `03-algorithmic-spec.json.training_hyperparameters`, `04-implementation-blueprint.json.paper_hyperparameters` / readiness-gate fields, and the final training log.

### Clearly stated in the paper
| Element | Paper-stated value | Source |
| ------- | ------------------ | ------ |
| dataset(s) | {value} | {section/table/ref} |
| optimizer | {value or "not stated"} | {source or NOT_SPECIFIED} |
| learning_rate | {value or "not stated"} | {source or NOT_SPECIFIED} |
| max_epochs | {value or "not stated"} | {source or NOT_SPECIFIED} |
| batch_size | {value or "not stated"} | {source or NOT_SPECIFIED} |
| protocol notes | {value or "not stated"} | {source or NOT_SPECIFIED} |

### Imputed, substituted, or framework-provided
| Element | Value actually used | Why it was needed | Source |
| ------- | ------------------- | ----------------- | ------ |
| dataset target/exclusion | {value} | {direct PICID match / unsupported paper dataset excluded / same-dataset recovery} | {blueprint/training log} |
| trainer max_epochs | {value} | {paper value used OR trainer default because NOT_SPECIFIED} | {blueprint/training log} |
| validation batch/scheduler/profile | train=512, val=1024, test=1024; {scheduler/profile value} | {fixed validation batch enforced for fair comparison; PICID scheduler/check cadence preserved; model HPs paper-stated or imputed because primary hp_mode=reference_only} | {training log, hp_reference_source, or experiment config comments} |
| preprocessing/windowing defaults | {value} | {paper omitted or framework standard} | {blueprint/config} |

If there were no imputations or substitutions, state that explicitly. Do not hide a `NOT_SPECIFIED` paper field by only reporting the final value.

## Dataset Recovery

- **Recovery occurred**: {yes/no}
- **Failed dataset**: {failed_dataset or "none"}
- **Final dataset used**: {dataset_used}
- **Reason**: {reason or "not applicable"}
- **Candidate rank**: {candidate_rank or "not applicable"}
- **Impact**: {State whether same-dataset recovery changed protocol comparability.}

## Axis 1: Magnitude Sanity

**Verdict**: {REASONABLE | SUSPICIOUS | BROKEN}

| Metric | Achieved (mean+/-std) | Typical Range for Domain | Assessment |
|--------|----------------------:|--------------------------|------------|
| RMSE   | {val}+/-{std}         | {low}-{high}             | {verdict}  |
| MAE    | {val}+/-{std}         | {low}-{high}             | {verdict}  |

{If datasets_are_same: also show direct comparison with paper values}

## Axis 2: Relative Positioning

**Verdict**: {CONSISTENT | PARTIALLY_CONSISTENT | INCONSISTENT | NOT_ASSESSABLE}

### Overlapping baselines found
| Baseline | Paper's Dataset | Our Dataset | Overlap |
|----------|:---------------:|:-----------:|:-------:|
| MLP      | Y (paper Table 2) | Y (framework) | Y    |
| LSTM     | Y (paper Table 2) | N              | N    |
| CNN1D    | N                  | Y (framework)  | N    |

### Relative gap comparison (overlapping baselines only)
| vs Baseline | Paper's Gap | Our Gap | Difference | Consistent? |
|-------------|-------------|---------|------------|-------------|
| vs MLP      | -32% RMSE   | -28% RMSE | 4pp     | Y           |

### Framework leaderboard ({dataset_used}, {task_type})
| Rank | Model | {metric1} | {metric2} |
|------|-------|-----------|-----------|
| 1    | {best} | {val}    | {val}     |
| ...  |        |          |           |
| {N}  | **{new_model}** | **{val}** | **{val}** |

![Baseline Comparison](plots/baseline_comparison.png)

**Competitiveness**: {COMPETITIVE | MIDDLE | NOT_COMPETITIVE | FIRST_ENTRY}

## Training Curves

Embed the vault-level loss plot produced by `/run-training` when it exists:

![Train and Validation Loss](plots/loss_curves.png)

If `plots/loss_curves.png` is missing, state the missing artifact explicitly and point to the `07-training-log.md` artifact section for the plotting failure reason.

## Axis 3: Behavioral Consistency

**Verdict**: {CONSISTENT | PARTIALLY_CONSISTENT | NOT_ASSESSABLE}

| Paper Claim | Checked? | Result |
|-------------|----------|--------|
| "Faster convergence than baselines" | Y | {confirmed/not confirmed} |
| "Better on long sequences" | N (no per-unit length data) | NOT_ASSESSED |
| "Lower parameter count" | Y | {value vs baselines} |

## Paper Claims Validation

Pre-registered from [[05-paper-hypothesis]] ({hypothesis_status}).
Baseline source: `report_output/{resolved_folder}/results.nc`.

| Claim | Type | Prediction | Observed | Verdict | Justification |
|-------|------|------------|----------|---------|---------------|
| C1 | relative_gap | our_gap vs lstm ≤ -0.20 on test/rmse_denormalized | -0.23 (paper: -0.21) | CONFIRMED | within 3 pp of paper's claim |
| C2 | absolute_value | test/rmse_denormalized < 1.5 | 0.12 | DATASET_DEPENDENT | paper's target on NASA; we trained on NB14 (framework_validation) |
| C3 | behavioral | faster convergence than baselines | — | UNASSESSABLE | PICID does not log epochs-to-best per baseline |
| C4 | absolute_value | test/rmse_denormalized comparable to paper normalized RMSE | — | UNASSESSABLE_METRIC_SCALE | paper and framework metric scales differ and no pre-registered conversion exists |

**Summary**: CONFIRMED {n} / CONTRADICTED {n} / DATASET_DEPENDENT {n} / UNASSESSABLE {n} / UNASSESSABLE_METRIC_SCALE {n} / UNASSESSABLE_NO_OVERLAPPING_BASELINE {n} of {total} pre-registered claims.

{If hypothesis_status is ABSENT or STALE_DATASET_MISMATCH, replace the table with a one-line explanation; do not fabricate verdicts.}

## Final Verdict

### Artifact Status: COMPLETE
### Technical Status: {PASS | IMPLEMENTATION_BUG | EVALUATOR_ERROR}
### Classification: {VALIDATED | PLAUSIBLE | INVESTIGATE | INVESTIGATE_CLAIMS_DISPUTED | BENCHMARK_ONLY | IMPLEMENTATION_BUG}
### Competitiveness: {COMPETITIVE | MIDDLE | NOT_COMPETITIVE | FIRST_ENTRY}
### Paper Claims: CONFIRMED {n} / CONTRADICTED {n} / DATASET_DEPENDENT {n} / UNASSESSABLE {n} / UNASSESSABLE_METRIC_SCALE {n} / UNASSESSABLE_NO_OVERLAPPING_BASELINE {n}

### Summary
{3-5 sentences: what was implemented, on which dataset, how it performs relative
to baselines, whether the relative positioning matches the paper's claims, and
what this means for the framework}

### Discrepancy Analysis (if not fully validated)
Why results might differ from paper:
- {if recovery occurred: "Dataset recovery: selected dataset {dataset_used} needed same-dataset repair because {reason}."}
- {e.g., "Different dataset: paper uses {X}, we tested on {Y} — domain gap expected"}
- {e.g., "Paper reports best-of-N, we report mean+/-std across seeds"}
- {e.g., "Paper's LSTM may differ from our LSTM implementation"}
- {e.g., "Paper uses custom preprocessing not replicated in our transform pipeline"}
- {e.g., "Paper evaluates on different split — unit assignment may not match"}

### Recommendations
- {e.g., "Model is validated — register as baseline for {dataset}/{task}"}
- {e.g., "Relative positioning matches paper — implementation appears correct"}
- {e.g., "Model underperforms on our data — may need dataset-specific tuning"}
- {e.g., "Cannot assess relative positioning — consider implementing paper's MLP baseline for controlled comparison"}
- {e.g., "Consider acquiring paper's dataset for direct comparison"}

### Files Produced
- Model checkpoint: {path}
- Predictions: {path or "not saved"}
- Experiment config: {path}
- Full evaluation report: this file
```

#### Update `{vault_dir}/00-paper-hub.md`:

Add/update the pipeline status and final verdict:

```markdown
## Pipeline Status
- [x] 00-paper-hub
- [x] 01-chunk-index
- [x] 02-conceptual-analysis
- [x] 03-algorithmic-spec
- [x] 04-implementation-blueprint
- [x] 05-paper-hypothesis
- [x] 06-sanity-ladder-log
- [x] 07-training-log
- [x] 08-evaluation-report

## Final Verdict: {VALIDATED | PLAUSIBLE | INVESTIGATE | INVESTIGATE_CLAIMS_DISPUTED | BENCHMARK_ONLY | IMPLEMENTATION_BUG}
## Technical Status: {PASS | IMPLEMENTATION_BUG | EVALUATOR_ERROR}
## Competitiveness: {ranking on framework leaderboard; for regression-like tasks report both RMSE and MAE rankings when available}
## Dataset Used: {dataset_used} ({comparison_mode})
## Dataset Recovery: {none, or "{dataset_used}: {reason}"}
## Paper Claims Summary: CONFIRMED {n} / CONTRADICTED {n} / DATASET_DEPENDENT {n} / UNASSESSABLE {n} / UNASSESSABLE_METRIC_SCALE {n} / UNASSESSABLE_NO_OVERLAPPING_BASELINE {n}  (from [[05-paper-hypothesis]])
## Key Result: {1-line summary, e.g., "Ranked 2/5 on PRONOSTIA/prognostics. Relative gap vs MLP matches paper claims (-28% vs paper's -32%). Model appears correctly implemented."}
```

### Step 6 — Report to experimenter

```
## Evaluation Complete

- **Magnitude**: {REASONABLE} — results in expected range for {domain}
- **Relative positioning**: {CONSISTENT} — relative gaps match paper's claims
  (checked against {N} overlapping baselines)
- **Competitiveness**: ranked {N}/{total} on {dataset} ({metric}); for regression-like tasks include both RMSE and MAE ranks when available
- **Classification**: {VALIDATED}
- **Action**: {register as baseline | investigate | re-implement}
```

### Structured sidecar

After writing `08-evaluation-report.md`, call `validate_paper_workflow_write_evaluation_sidecar` to create `{vault_dir}/08-evaluation-report.json`.

The sidecar must contain:
- `status` — the final classification (`VALIDATED | PLAUSIBLE | INVESTIGATE | INVESTIGATE_CLAIMS_DISPUTED | BENCHMARK_ONLY | IMPLEMENTATION_BUG`)
- `artifact_status` — `COMPLETE` when both markdown and JSON artifacts were written
- `technical_status` — `PASS | IMPLEMENTATION_BUG | EVALUATOR_ERROR`
- `scientific_status` — same value as `status` unless the report chooses to keep a separate scientific label
- `blocking` — boolean. Use `false` for completed scientific observations (`INVESTIGATE`, `INVESTIGATE_CLAIMS_DISPUTED`, metric-scale ambiguity, benchmark-only); use `true` only for technical/evaluator failures that require upstream repair.
- `observations` — compact list of non-blocking caveats such as metric-scale mismatch, missing baselines, dataset-dependent claims, or disputed claims
- `comparison_mode`
- `dataset_used`
- `paper_dataset`
- `hypothesis_status`
- `paper_claims_summary` — at minimum confirmed / contradicted / dataset_dependent / unassessable counts
- `timestamp`

The markdown report is for humans. The JSON sidecar is the machine-readable workflow artifact.

## Special cases

### Datasets ARE the same (rare but ideal)
When `datasets_are_same=true`, Axis 1 becomes a DIRECT comparison — achieved metrics
should be within reasonable tolerance of paper values (accounting for seed variance and
possible best-of-N vs mean reporting differences). Report both direct and relative comparisons.

### No overlapping baselines
If no baselines overlap between paper and framework:
- Skip Axis 2 entirely
- Verdict defaults to `BENCHMARK_ONLY`
- Recommend: "Consider implementing one of the paper's baselines (e.g., MLP or LSTM)
  on the same PICID dataset to enable relative comparison"

### Paper doesn't report baseline comparisons
Some papers only report their own results without baselines:
- Skip relative positioning
- Rely on magnitude sanity + framework leaderboard only
- Note limitation in report

### Fit-predict models
Same evaluation logic — metrics are metrics regardless of training paradigm.

### Multiple datasets
Invoke this skill once per dataset. Each produces its own evaluation report section.

## Common pitfalls

- **Metric direction matters**: lower RMSE = better, higher R2 = better. Always check
  direction before computing relative gaps. See `.opencode/reference/inventory.md` §4.5.
- **Relative gap formula**: use `(model - baseline) / baseline`. For "lower is better"
  metrics, a negative gap means improvement. Be consistent.
- **Paper results are often cherry-picked** (best run) — our mean+/-std will look worse
  even if implementation is identical. ALWAYS note this in discrepancy analysis.
- **Seed variance**: different random seeds cause 5-15% variance on small test sets —
  this is noise, not bugs.
- **"MLP" in the paper may not be the same MLP** as in PICID (different hidden sizes,
  depth, activation). Relative comparisons are approximate — the 15pp threshold accounts for this.
- **Framework leaderboard is only valid** if all models were trained under the same
  conditions (same dataset config, same split, same preprocessing pipeline).
- **Per-unit metrics can vary wildly** even when aggregate metrics match.
- **Metric units**: some papers normalize metrics differently (percentage vs absolute,
  RMSE vs MSE). Always verify units match before comparing.
