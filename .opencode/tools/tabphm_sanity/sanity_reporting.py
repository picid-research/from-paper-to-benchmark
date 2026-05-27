from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from sanity_base import (
    BLOCK,
    DATASET_EXECUTION_FAILED,
    DATASET_UNAVAILABLE,
    FAIL,
    INVESTIGATE,
    PASS,
    PRECHECK_TIMEOUT,
    PROCEED,
    SKIPPED,
    TOOL_INVOCATION_FAILURE,
    WARN_CONTINUE,
    CheckResult,
    json_default,
)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_hard_model_failure(check: CheckResult) -> bool:
    """Return true only for sanity failures that make benchmark training unsafe."""
    if check.status != FAIL:
        return False
    if check.command or check.traceback_excerpt:
        return True

    metrics = check.metrics or {}
    if check.name == "init_loss":
        first_loss = _float_or_none(metrics.get("first_loss"))
        return first_loss is not None and not math.isfinite(first_loss)

    if check.name == "gradient_flow":
        trainable_count = _float_or_none(metrics.get("trainable_parameter_count"))
        if trainable_count == 0:
            return True
        norms = metrics.get("gradient_norms") or {}
        if not isinstance(norms, dict):
            return False
        values = [
            value for value in (_float_or_none(item) for item in norms.values())
            if value is not None
        ]
        if any(not math.isfinite(value) for value in values):
            return True
        if values:
            return not any(math.isfinite(value) and value > 0.0 for value in values)
        dead_layers = metrics.get("dead_layers") or []
        return bool(dead_layers)

    if check.name == "overfit_batch":
        for key in ("initial_loss", "final_loss", "best_loss"):
            value = _float_or_none(metrics.get(key))
            if value is not None and not math.isfinite(value):
                return True
        return False

    return False


def summarize_verdict(checks: list[CheckResult]) -> str:
    for check in checks:
        if check.status in (DATASET_UNAVAILABLE, DATASET_EXECUTION_FAILED):
            return check.status
    for check in checks:
        if check.status == TOOL_INVOCATION_FAILURE:
            return TOOL_INVOCATION_FAILURE
    for check in checks:
        if check.status == PRECHECK_TIMEOUT:
            return PRECHECK_TIMEOUT
    by_name = {check.name: check.status for check in checks}
    if all(status in (PASS, SKIPPED) for status in by_name.values()):
        return PROCEED
    if any(_is_hard_model_failure(check) for check in checks):
        return BLOCK
    if any(status in (FAIL, INVESTIGATE) for status in by_name.values()):
        return WARN_CONTINUE
    return BLOCK


def _next_attempt_id(vault_dir: str | Path | None, experiment_config: str) -> int:
    if not vault_dir:
        return 1
    path = Path(vault_dir).expanduser().resolve() / "06-sanity-ladder-log.md"
    if not path.exists():
        return 1
    max_id = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            match = re.search(r"\*\*Attempt\*\*:\s*(\d+)", line)
            if match:
                max_id = max(max_id, int(match.group(1)))
    return max_id + 1


def render_markdown(
    result: dict[str, Any],
    vault_dir: str | Path | None,
    experiment_config: str,
    task_type: str,
    dataset_candidate: str | None,
) -> str:
    attempt_id = _next_attempt_id(vault_dir, experiment_config)
    check_diagnostics = {
        check.get("name"): (check.get("metrics") or {}) for check in result.get("checks", [])
    }
    frontmatter = {
        "verdict": PASS if result.get("verdict") == PROCEED else result.get("verdict"),
        "attempt": attempt_id,
        "passed_checks": [
            check.get("name") for check in result.get("checks", []) if check.get("status") == PASS
        ],
        "failed_checks": [
            check.get("name")
            for check in result.get("checks", [])
            if check.get("status") not in (PASS, SKIPPED)
        ],
        "skipped_checks": [
            check.get("name") for check in result.get("checks", []) if check.get("status") == SKIPPED
        ],
        "check_diagnostics": check_diagnostics,
        "dataset_candidate": dataset_candidate,
        "dataset_status": result.get("dataset_executability", PASS),
        "fallback_trigger": bool(result.get("fallback_trigger")),
        "experiment_config": experiment_config,
        "timestamp": datetime.now().astimezone().replace(microsecond=0).isoformat(),
    }
    supersedes_note = ""
    if attempt_id > 1:
        supersedes_note = f"**Supersedes**: Attempt {attempt_id - 1} (prior results for this experiment config are superseded by this run)"

    lines = [
        "---",
        yaml.safe_dump(frontmatter, sort_keys=False).strip(),
        "---",
        "",
        "",
        "## Sanity Ladder -- Runtime Checks",
        "",
        f"**Timestamp**: {datetime.now().isoformat(timespec='seconds')}",
        f"**Experiment config**: `{experiment_config}`",
        f"**Task type**: `{task_type}`",
        f"**Attempt**: {attempt_id}",
    ]
    if supersedes_note:
        lines.append(supersedes_note)
    if vault_dir:
        lines.append("**Blueprint**: [[04-implementation-blueprint]]")
    lines.append("")

    for check in result.get("checks", []):
        lines.extend(
            [
                f"### Check: {check.get('name')}",
                f"- **Status**: {check.get('status')}",
                f"- **Diagnostic**: {check.get('diagnostic') or ''}",
            ]
        )
        metrics = check.get("metrics") or {}
        for key, value in metrics.items():
            if key == "gradient_norms":
                lines.append(f"- **{key}**: {len(value)} parameter tensors recorded")
            else:
                lines.append(
                    f"- **{key}**: `{json.dumps(value, default=json_default)}`"
                )
        run_dirs = check.get("run_dirs") or []
        if run_dirs:
            lines.append(f"- **Run directory**: `{run_dirs[-1]}`")
        if check.get("traceback_excerpt"):
            lines.extend(
                [
                    "- **Traceback excerpt**:",
                    "```text",
                    check["traceback_excerpt"],
                    "```",
                ]
            )
        lines.append("")

    checks = result.get("checks", [])
    passed = sum(1 for check in checks if check.get("status") == PASS)
    investigate = sum(1 for check in checks if check.get("status") == INVESTIGATE)
    failed = sum(
        1
        for check in checks
        if check.get("status")
        in {
            FAIL,
            PRECHECK_TIMEOUT,
            DATASET_UNAVAILABLE,
            DATASET_EXECUTION_FAILED,
            TOOL_INVOCATION_FAILURE,
        }
    )
    skipped = sum(1 for check in checks if check.get("status") == SKIPPED)
    lines.extend(
        [
            "### Summary",
            f"- **Checks passed**: {passed}/{len(checks)}",
            f"- **Checks investigate**: {investigate}/{len(checks)}",
            f"- **Checks failed**: {failed}/{len(checks)}",
            f"- **Checks skipped**: {skipped}/{len(checks)}",
            f"- **Verdict**: {result.get('verdict')}",
            f"- **Failure pattern interpretation**: {result.get('failure_pattern_interpretation', '')}",
            "",
            "### Dataset Executability",
            f"- **Dataset candidate**: {dataset_candidate or 'unknown'}",
            f"- **Status**: {result.get('dataset_executability', PASS)}",
            f"- **Dataset recovery required**: {'yes' if result.get('fallback_trigger') else 'no'}",
            f"- **Failed command**: `{result.get('failed_command') or ''}`",
        ]
    )
    if result.get("traceback_excerpt"):
        lines.extend(
            ["- **Traceback excerpt**:", "```text", result["traceback_excerpt"], "```"]
        )
    return "\n".join(lines) + "\n"


def append_report(markdown: str, vault_dir: str | Path | None) -> str | None:
    if not vault_dir:
        return None
    path = Path(vault_dir).expanduser().resolve() / "06-sanity-ladder-log.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(markdown)
    return str(path)


def append_sidecar(result: dict[str, Any], vault_dir: str | Path | None) -> str | None:
    if not vault_dir:
        return None
    path = Path(vault_dir).expanduser().resolve() / "06-sanity-ladder-log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result, default=json_default, sort_keys=True))
        handle.write("\n")
    return str(path)


def interpret_failure(checks: list[CheckResult], verdict: str) -> str:
    if verdict == PROCEED:
        return "All applicable sanity checks passed."
    if verdict == INVESTIGATE:
        return (
            "Sanity checks show meaningful signal but not a clean enough pass. "
            "Inspect the nuanced diagnostics before training."
        )
    if verdict == WARN_CONTINUE:
        return (
            "One or more model sanity diagnostics did not pass, but no categorical "
            "execution blocker was detected. Preserve the warning and continue to "
            "benchmark training."
        )
    if verdict in (DATASET_UNAVAILABLE, DATASET_EXECUTION_FAILED):
        return "Selected dataset could not execute locally; experimenter should repair the same dataset or stop with a named blocker."
    if verdict == TOOL_INVOCATION_FAILURE:
        return "Tool invocation failed (e.g. Hydra config composition error). This is a tooling/config issue, not a model failure. Fix the config and rerun."
    if verdict == PRECHECK_TIMEOUT:
        return "A check timed out. This may indicate a slow dataset or long preprocessing, not a model bug. Consider increasing timeout_sec."
    failed = {check.name for check in checks if check.status == FAIL}
    if failed == {"subset_convergence"}:
        return (
            "Likely hyperparameter issue; try learning-rate or optimizer adjustments."
        )
    if "zero_input" in failed:
        return "Data pipeline may not be feeding informative inputs to the model."
    if "gradient_flow" in failed:
        return "Architecture or reshape bug likely affected gradient flow."
    if "overfit_batch" in failed:
        return "Forward pass, loss, optimizer, or backpropagation may be broken."
    if "init_loss" in failed:
        return (
            "Initialization, final layer, loss, or output-target wiring may be wrong."
        )
    return "One or more sanity checks failed; inspect check diagnostics."


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (bool, int, float, str)) or value is None:
            compact[key] = value
            continue
        if key == "gradient_norms" and isinstance(value, dict):
            compact["gradient_parameter_tensors"] = len(value)
            numeric = [
                float(item) for item in value.values() if isinstance(item, (int, float))
            ]
            if numeric:
                compact["gradient_norm_max"] = max(numeric)
                nonzero = [item for item in numeric if item > 0]
                if nonzero:
                    compact["gradient_norm_min_nonzero"] = min(nonzero)
            continue
        if isinstance(value, list):
            compact[f"{key}_count"] = len(value)
            if (
                value
                and len(value) <= 3
                and all(
                    isinstance(item, (bool, int, float, str)) or item is None
                    for item in value
                )
            ):
                compact[f"{key}_preview"] = value
            continue
        if isinstance(value, dict):
            compact[f"{key}_count"] = len(value)
    return compact


def ladder_result(
    checks: list[CheckResult],
    experiment_config: str,
    task_type: str,
    dataset_candidate: str | None,
    vault_dir: str | Path | None,
    write_log: bool,
    compact: bool = False,
) -> dict[str, Any]:
    verdict = summarize_verdict(checks)
    dataset_check = next(
        (
            check
            for check in checks
            if check.status in (DATASET_UNAVAILABLE, DATASET_EXECUTION_FAILED)
        ),
        None,
    )
    result = {
        "experiment_config": experiment_config,
        "task_type": task_type,
        "dataset_candidate": dataset_candidate,
        "verdict": PASS if verdict == PROCEED else verdict,
        "checks": [check.to_dict() for check in checks],
        "dataset_executability": dataset_check.status if dataset_check else PASS,
        "fallback_trigger": dataset_check is not None,
        "failed_command": " ".join(dataset_check.command or [])
        if dataset_check
        else None,
        "traceback_excerpt": dataset_check.traceback_excerpt if dataset_check else "",
        "failure_pattern_interpretation": interpret_failure(checks, verdict),
    }
    if write_log:
        markdown = render_markdown(
            result, vault_dir, experiment_config, task_type, dataset_candidate
        )
        result["markdown_appended_path"] = append_report(markdown, vault_dir)
        result["jsonl_appended_path"] = append_sidecar(
            {
                "verdict": result["verdict"],
                "attempt": _next_attempt_id(vault_dir, experiment_config) - 1,
                "passed_checks": [
                    check.name for check in checks if check.status == PASS
                ],
                "failed_checks": [
                    check.name for check in checks if check.status not in (PASS, SKIPPED)
                ],
                "skipped_checks": [
                    check.name for check in checks if check.status == SKIPPED
                ],
                "check_diagnostics": {check.name: check.metrics for check in checks},
                "dataset_candidate": dataset_candidate,
                "dataset_status": result["dataset_executability"],
                "fallback_trigger": result["fallback_trigger"],
                "experiment_config": experiment_config,
                "timestamp": datetime.now().astimezone().replace(microsecond=0).isoformat(),
            },
            vault_dir,
        )
    if compact:
        compact_checks = []
        for check in checks:
            compact_check = {
                "name": check.name,
                "status": check.status,
                "diagnostic": check.diagnostic,
            }
            if check.metrics:
                compact_metrics = _compact_metrics(check.metrics)
                if compact_metrics:
                    compact_check["metrics"] = compact_metrics
            compact_checks.append(compact_check)
        compact_result = {
            "experiment_config": result["experiment_config"],
            "task_type": result["task_type"],
            "dataset_candidate": result["dataset_candidate"],
            "verdict": result["verdict"],
            "checks": compact_checks,
            "dataset_executability": result["dataset_executability"],
            "fallback_trigger": result["fallback_trigger"],
            "failure_pattern_interpretation": result["failure_pattern_interpretation"],
        }
        if result.get("markdown_appended_path"):
            compact_result["markdown_appended_path"] = result["markdown_appended_path"]
        if result.get("jsonl_appended_path"):
            compact_result["jsonl_appended_path"] = result["jsonl_appended_path"]
        if result.get("failed_command"):
            compact_result["failed_command"] = result["failed_command"]
        if result.get("traceback_excerpt"):
            compact_result["traceback_excerpt"] = result["traceback_excerpt"]
        return compact_result
    return result
