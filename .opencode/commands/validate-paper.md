---
description: Validate a research paper by generating an implementation blueprint, implementing novel components, and evaluating them in PICID.
agent: paper-validator
---

Run the PICID paper validation workflow.

This workflow is stateful. The orchestrator must keep `{vault_dir}/run_state.json`
up to date via the `validate_paper_workflow_*` tools and persist structured
sidecars for the key machine-consumed artifacts:
- `02-conceptual-analysis.json`
- `03-algorithmic-spec.json`
- `04-implementation-blueprint.json`
- `06-sanity-ladder-log.jsonl`
- `batch_fit_check.json`
- `07-training-log.jsonl`
- `08-evaluation-report.json`
- `session_stats.json`

This command is unattended by default. The orchestrator and subagents must proceed autonomously using the blueprint, policies, scheduled validation run matrix, and bounded-retry rules without asking the user for intermediate decisions. Cross-dataset fallback is disabled: selected datasets must be direct PICID-supported paper datasets and must be repaired in place or reported as named blockers. Scientific caveats such as disputed paper claims, metric-scale mismatch, missing overlapping baselines, and benchmark-only status must be recorded as observations in the vault rather than treated as blockers.

Arguments:
- `$1`: paper directory containing the PDF or converted markdown. Default: `vault/paper`
- `$2`: vault output directory. Default: same as `$1`
- `$3`: run mode. One of `blueprint-only`, `quick` (1-epoch smoke test that exercises train/val/test/checkpointing for each direct PICID paper dataset), or `full` (repair-looped 1-epoch preflight, then one monitored paper-scale run per direct PICID paper dataset without wall-clock timeout interruption). Default: `full`
- `$4`: hyperparameter mode. One of `reference_only` (default; force validation datamodule batch sizes to train `512`, val `1024`, test `1024`, preserve PICID LR scheduler/check cadence, and use paper-stated or explicitly imputed model HPs) or `reference_plus_paper` (also run the paper/default HP profile as auxiliary context).

Use the current working directory as the PICID repo root. Follow `.opencode/reference/README.md` (and the section files it points to) for implementation contracts, and `logging-guide.md` for training artifact handling.
