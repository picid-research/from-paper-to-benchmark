---
name: paper-hypothesis
description: After the implementation blueprint is finalized and BEFORE training runs, extract the paper's quantitative claims and pre-register a comparison table against PICID's internal baselines (parsed from report_output/*/results.nc via the PICID_baselines tool). Writes vault/paper/05-paper-hypothesis.md, which /evaluate-results later consumes to judge each claim as CONFIRMED / CONTRADICTED / DATASET_DEPENDENT / UNASSESSABLE / UNASSESSABLE_METRIC_SCALE against the actual trained-model metrics. This is the pre-registration step — don't author it after seeing training results.
---

# /paper-hypothesis

Pre-register the paper's quantitative claims and the expected comparison against PICID's frozen internal leaderboard, BEFORE training runs.

This skill exists because PICID already has a consistent, reproducible baseline table for each dataset (in `report_output/<folder>/results.nc`, produced by `picid_report`). The paper's claims should be recorded up front, mapped to those baselines, and given explicit numerical confirmation rules — so the post-training `/evaluate-results` step is a verdict on a fixed hypothesis, not a retroactive narrative.

## Input contract

The orchestrator must provide:

- **vault_dir**: path to vault (reads `02`, `03`, `04`; writes `05-paper-hypothesis.md`; updates `00-paper-hub.md` checklist)
- **repo_root**: PICID repo root
- **dataset_used**: the PICID dataset key the experimenter will train on AFTER any fallback (e.g., `nb14`, `hsf15`, `concepts_n_cmapss`)
- **task_type**: one of `prognostics | diagnostics | anomaly_detection | forecasting | state_forecasting`
- **subtask**: optional — required for datasets with multiple folders (e.g., `hsf15` subtasks: `pump | cooler | valve | accumulator`; `concepts_n_cmapss` prognostics: `default | ds02`)
- **paper_dataset**: what the paper used (e.g., `NASA Battery`, `FEMTO-PRONOSTIA`)
- **datasets_are_same**: boolean — `true` only if the paper's dataset matches `dataset_used`
- **comparison_mode**: `exact_reproduction | framework_validation | benchmark_only` (from blueprint)

## Procedure

### Step 0 — Read context

1. Read `{vault_dir}/02-conceptual-analysis.json` to extract the paper's main contributions and qualitative claims (what the method is supposed to do better than baselines). Use the Markdown only for source snippets if needed.
2. Read `{vault_dir}/03-algorithmic-spec.json` to confirm which metrics the paper reports on and their units.
3. Read `{vault_dir}/04-implementation-blueprint.json` — **this is the authoritative source for `verification_protocol` / `metric_keys`**, task type, dataset, and comparison context. Use the rendered Markdown only for human-readable snippets.

### Step 1 — Derive the metric keys to compare on

The baselines tool takes `metric_keys` from the NetCDF `metric_key` dimension (e.g., `test/rmse_denormalized`, `test/f1`).

Apply this hybrid rule:

1. If blueprint §7 explicitly names metric keys (prefer the ones using the `test/` prefix), use them verbatim.
   - For regression-like tasks (`prognostics`, `forecasting`, `state_forecasting`), always carry both RMSE and MAE when the corresponding key exists. If §7 names only RMSE, add the matching MAE key with the same split/scale (`test/rmse_denormalized` -> `test/mae_denormalized`, `test/rmse_normalized` -> `test/mae_normalized`). If §7 names only MAE, add the matching RMSE key. If the counterpart is missing from `available_metric_keys`, record it under `missing_metric_keys` and continue with the available one.
2. Otherwise, fall back to the task-type defaults:
   - `prognostics`: `["test/rmse_denormalized", "test/mae_denormalized"]`
   - `diagnostics`: `["test/accuracy", "test/f1"]`
   - `forecasting` / `state_forecasting`: `["test/rmse_normalized", "test/mae_normalized"]`
   - `anomaly_detection`: `["test/accuracy", "test/f1"]` — note in the report if the paper reports AUROC and we cannot find it.
3. Metric-key drift matters (`_denormalized` vs `_normalized`; `test/` vs `test_best_rerun/`). Record the chosen variant in the hypothesis file. If the paper's reported values are on a different scale than the selected PICID metric and no explicit conversion is available before training, pre-register the affected absolute-value claims as scale-ambiguous so `/evaluate-results` can mark them `UNASSESSABLE_METRIC_SCALE` rather than `CONTRADICTED`.

### Step 2 — Query PICID internal baselines

Invoke the `PICID_baselines` tool (the OpenCode wrapper `load_baselines`):

```
load_baselines(
  dataset = dataset_used,
  task_type = task_type,
  subtask = subtask,           # only when applicable
  metric_keys = <from Step 1>,
)
```

Handle the tool's `status` field explicitly:

| Status | Action |
|--------|--------|
| `OK` | Build the baseline table from `rows`. Hypothesis status = `PRE_REGISTERED`. |
| `METRIC_MISSING` | Retry once with the intersection of requested keys and `available_metric_keys`. If still empty, tag each affected claim `UNASSESSABLE_METRIC_MISSING` and include `available_metric_keys` in the report for future reference. |
| `FOLDER_NOT_FOUND` | Hypothesis status = `BENCHMARK_ONLY`. All claims become `UNASSESSABLE_NO_INTERNAL_BASELINE`. **Do not abort** — the experimenter still needs to proceed with training. |
| `CORRUPT_NETCDF` | Abort the skill with the `reason` field. The experimenter will escalate. |

### Step 3 — Extract paper claims

From `02-conceptual-analysis.json` and `03-algorithmic-spec.json`, enumerate the paper's **quantitative** and **behavioral** claims. Tag each with a stable ID (`C1`, `C2`, ...) and a type:

| Type | Meaning | Confirmation rule |
|------|---------|-------------------|
| `relative_gap` | "Method X beats baseline Y by N% on metric M" | CONFIRMED if our gap vs our Y on the same metric is within 15pp of paper's gap; CONTRADICTED if the sign flips or the difference exceeds 30pp. For regression-like tasks, register separate RMSE and MAE checks when both metrics are available, even if the paper emphasizes only one. |
| `absolute_value` | "RMSE < 1.5" / "Accuracy > 0.95" | Directly judged only when `datasets_are_same=true` and metric scale/unit matches or has a pre-registered conversion. When datasets differ, this claim becomes `DATASET_DEPENDENT`; when scales differ, it becomes `UNASSESSABLE_METRIC_SCALE`. |
| `behavioral` | "Faster convergence", "Better on long sequences", "Lower parameter count" | Usually `UNASSESSABLE` unless the claim maps to an aggregate metric we log (parameter count, training-epochs-to-best). Record what would be needed to judge it. |

For each claim, record:
- The verbatim paper text and the chunk/section reference.
- Mapped paper baseline (if any) — the baseline the paper compared against.
- Mapped PICID baseline short name (from Step 2's rows) — the baseline the verdict will actually use.
- The exact numerical prediction to evaluate post-training.

If a paper claim has no PICID counterpart (paper compared to `TransformerXL`; PICID has no `TransformerXL` row) — keep the claim but mark it `UNASSESSABLE_NO_OVERLAPPING_BASELINE`.

### Step 4 — Write 05-paper-hypothesis.md

Write `{vault_dir}/05-paper-hypothesis.md` using this structure. Fill in every placeholder. Do not rearrange headings — `/evaluate-results` parses this file.

```markdown
# Paper Hypothesis (Pre-Registered)

**Timestamp**: {iso_timestamp}
**Status**: PRE_REGISTERED | BENCHMARK_ONLY
**Paper**: [[00-paper-hub]]
**Blueprint**: [[04-implementation-blueprint]]

## Context
- **Paper dataset**: {paper_dataset}
- **Framework dataset used**: {dataset_used}{/subtask if any}
- **Datasets are same**: {yes|no}
- **Comparison mode**: {comparison_mode}
- **Metric keys under evaluation**: {comma-separated list}
- **Baseline source**: `report_output/{resolved_folder}/results.nc` (project `{project_name}`)

## Paper's Reported Baselines ({paper_dataset})
{Extracted from the paper's comparison table. Cite the chunk/table.}

| Model | {metric_1} | {metric_2} | Source |
|-------|-----------:|-----------:|--------|
| ...   | ...        | ...        | paper Table N |

## PICID Internal Baselines ({dataset_used})
{One row per PICID model returned by PICID_baselines. Use the short_name and format `mean ± std (n=...)`.}

| Model (short) | {metric_key_1} mean ± std (n) | {metric_key_2} mean ± std (n) |
|---------------|------------------------------:|------------------------------:|
| lstm          | 56.57 ± 3.20 (n=5)            | 41.13 ± 2.34 (n=5)            |
| ...           | ...                           | ...                           |

## Pre-Registered Claims

### C1: "{verbatim paper claim}"
- **Type**: relative_gap | absolute_value | behavioral
- **Paper source**: {section/table reference}
- **Paper baseline**: {paper's Y model}
- **PICID baseline**: {short_name from the table above, or "UNASSESSABLE_NO_OVERLAPPING_BASELINE"}
- **Numerical prediction**: {exact expression, e.g. `(our_model_rmse - lstm_rmse) / lstm_rmse <= -0.20 on test/rmse_denormalized`}
- **Confirmation rule**: CONFIRMED if our relative gap is within 15pp of paper's gap ({paper_gap:.3f}); CONTRADICTED if the sign flips or the difference exceeds 30pp; UNASSESSABLE if the mapped baseline or metric is missing at verdict time.
- **Metric scale note**: {`MATCHED` | `PAPER_NORMALIZED_PICID_DENORMALIZED` | `PAPER_DENORMALIZED_PICID_NORMALIZED` | `UNKNOWN`; include conversion rule or "no conversion available"}

### C2: "..."
{...}

## Cross-Dataset Caveats
{Only include this section when datasets_are_same=false. Otherwise omit.}

- Paper's absolute numbers are on `{paper_dataset}`; our internal baselines are on `{dataset_used}`. Absolute-value claims are therefore tagged `DATASET_DEPENDENT` at verdict time and will be judged only by magnitude sanity.
- Relative-positioning claims remain the primary axis — we compare our (trained_model vs PICID baseline) gap to the paper's (proposed vs paper baseline) gap.
- Behavioral claims remain `UNASSESSABLE` unless they map to aggregate metrics available in PICID.
- Metric-scale mismatches are preserved as observations. If the paper reports normalized values but the selected PICID metric is denormalized, or vice versa, affected absolute-value claims are `UNASSESSABLE_METRIC_SCALE` unless this file records a concrete conversion rule before training.

## Unassessable Claims
{Claims that cannot be judged from PICID metrics. Each entry should say what would be needed to judge it.}

- {claim} — requires {what's missing}

## Tool-derived Metadata (do not edit)
- `resolved_folder`: {folder_name}
- `available_metric_keys_count`: {len}
- `n_PICID_models`: {n}
- `missing_metric_keys`: {list or "none"}
- `metric_scale_status`: {MATCHED | MIXED | UNKNOWN}
```

### Step 5 — Update the hub checklist

Open `{vault_dir}/00-paper-hub.md`. In the "Pipeline Status" (or equivalent) checkbox block, mark `- [x] 05-paper-hypothesis`. If the hub has no such entry, insert it between `- [x] 04-implementation-blueprint` and `- [ ] 06-sanity-ladder-log` (or the next existing slot).

Also add a one-line summary under the pipeline status block:

```
## Hypothesis Pre-Registration
- Status: PRE_REGISTERED | BENCHMARK_ONLY
- Baseline source: report_output/{resolved_folder}/results.nc
- Claims: N total ({n_relative_gap} relative_gap, {n_absolute_value} absolute_value, {n_behavioral} behavioral)
```

## Output

- `{vault_dir}/05-paper-hypothesis.md` — the pre-registered hypothesis.
- `{vault_dir}/00-paper-hub.md` — checkbox flipped, one-line summary appended.

## Failure modes

| Failure | What to write | What to return |
|---------|---------------|----------------|
| No claims found in 02/04 (very minimal paper) | Write `05` with an empty "Pre-Registered Claims" section and a note in "Unassessable Claims" explaining the paper didn't report comparative quantitative claims. Status `PRE_REGISTERED`. | Success. `/evaluate-results` will see an empty claims list and skip the validation section. |
| `FOLDER_NOT_FOUND` | Write `05` with status `BENCHMARK_ONLY`, empty PICID baseline table, all claims tagged `UNASSESSABLE_NO_INTERNAL_BASELINE`. | Success. Experimenter proceeds with training. |
| `METRIC_MISSING` post-retry | Write `05` with requested keys shown and affected claims tagged `UNASSESSABLE_METRIC_MISSING`. Include `available_metric_keys` in the report so a human can pick the right ones. | Success. |
| `CORRUPT_NETCDF` | Do NOT write a partial `05`. | Raise with the tool's `reason` field so the experimenter can classify and retry. |

## Never do

- Never author `05-paper-hypothesis.md` AFTER training has produced results — that defeats the pre-registration purpose. If the experimenter's `run_mode` has already advanced past Phase F, refuse with "hypothesis must precede training."
- Never use the training-run's own metrics as baseline numbers. Baselines come only from `report_output/*/results.nc`.
- Never re-run baseline experiments. The internal leaderboard is frozen; re-running would invalidate cross-paper comparability.
- Never invent paper claims. If the paper's comparison table is missing or unclear, record that in "Unassessable Claims" and leave the claims list empty rather than fabricating predictions.

## Common pitfalls

- **Metric-key drift**: `test/rmse_denormalized` vs `test/rmse_normalized` is a real difference — paper's absolute numbers are often denormalized while their relative gaps are computed on the same scale. Pick one variant, record the choice, and stick to it at verdict time. If the chosen variant does not match the paper's absolute-value scale and no conversion is available, record that up front; later evaluation must treat it as unassessable scale ambiguity, not as a contradiction.
- **Late dataset mismatch**: cross-dataset fallback is disabled. If `dataset_used` changes after this skill runs, `05` becomes stale and the run should surface the mismatch rather than silently validating on a substitute dataset.
- **"LSTM" ambiguity**: the paper's `LSTM` may not be the same architecture as PICID's `baselines.lstm_model.LSTM_Forecaster` (different hidden sizes, depth). Document this caveat under the mapped baseline — the 15pp tolerance is there precisely to absorb this.
- **Best-of-N vs mean±std**: papers often report a best run; PICID's `results.nc` stores `mean`/`std` over seeds. Note this when writing the confirmation rule.
