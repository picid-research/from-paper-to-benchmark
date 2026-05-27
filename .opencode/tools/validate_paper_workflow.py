"""Deterministic state and artifact management for the validate-paper workflow.

Usage:
    uv run python .opencode/tools/validate_paper_workflow.py <command> --input-json '<payload>'

This backend is intentionally narrow: it owns workflow state, fixed phase order,
artifact gate checks, and structured sidecars. LLM prompts may still perform the
judgment-heavy steps, but they should use this tool as the control plane.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

RESULT_MARKER = "VALIDATE_PAPER_WORKFLOW_RESULT_JSON="
STATE_FILE_NAME = "run_state.json"

PHASE_ORDER = [
    "input_check",
    "process_paper",
    "chunk_index",
    "conceptual_analysis",
    "algorithmic_spec",
    "implementation_blueprint",
    "paper_hypothesis",
    "implement_components",
    "verify_static",
    "verify_sanity",
    "check_batch_fit",
    "run_training",
    "evaluate_results",
]

PHASE_DEPENDENCIES = {
    "input_check": [],
    "process_paper": ["input_check"],
    "chunk_index": ["process_paper"],
    "conceptual_analysis": ["chunk_index"],
    "algorithmic_spec": ["chunk_index"],
    "implementation_blueprint": ["conceptual_analysis", "algorithmic_spec"],
    "paper_hypothesis": ["implementation_blueprint"],
    "implement_components": ["paper_hypothesis"],
    "verify_static": ["implement_components"],
    "verify_sanity": ["verify_static"],
    "check_batch_fit": ["verify_sanity"],
    "run_training": ["check_batch_fit"],
    "evaluate_results": ["run_training"],
}

DEFAULT_RELATIVE_ARTIFACTS = {
    "chunk_index": ["00-paper-hub.md", "01-chunk-index.md"],
    "conceptual_analysis": ["02-conceptual-analysis.md", "02-conceptual-analysis.json"],
    "algorithmic_spec": ["03-algorithmic-spec.md", "03-algorithmic-spec.json"],
    "implementation_blueprint": ["04-implementation-blueprint.md", "04-implementation-blueprint.json"],
    "paper_hypothesis": ["05-paper-hypothesis.md"],
    "verify_static": ["06-sanity-ladder-log.md"],
    "verify_sanity": ["06-sanity-ladder-log.md", "06-sanity-ladder-log.jsonl"],
    "check_batch_fit": ["batch_fit_check.json"],
    "run_training": ["07-training-log.md", "07-training-log.jsonl"],
    "evaluate_results": ["08-evaluation-report.md", "08-evaluation-report.json"],
}

DEFAULT_PHASE_ATTEMPT_LIMITS: dict[str, int | dict[str, int]] = {
    "algorithmic_spec": 3,
    "paper_hypothesis": 3,
    "verify_static": 6,
    "check_batch_fit": 3,
    "run_training": {
        "quick": 5,
        "full": 8,
    },
    "evaluate_results": 5,
}

BLUEPRINT_REQUIRED_KEYS = {
    "schema_version",
    "paper_dataset",
    "evaluation_targets",
    "excluded_paper_datasets",
    "framework_dataset_used",
    "datasets_are_same",
    "comparison_mode",
    "fallback_allowed",
    "dataset_fallback_candidates",
    "experiment_config",
    "task_type",
    "required_new_files",
    "verification_protocol",
    "paper_hyperparameters",
    "dataset_contract",
    "model_io_contract",
    "split_contract",
}

CONCEPTUAL_REQUIRED_KEYS = {
    "schema_version",
    "paper_summary",
    "structure_map",
    "novel_components",
    "reused_components",
    "configured_components",
    "dataset_mapping",
    "direct_evaluation_targets",
    "excluded_paper_datasets",
    "integration_roadmap",
}

ALGORITHMIC_REQUIRED_KEYS = {
    "schema_version",
    "algorithms",
    "equations",
    "architectures",
    "losses",
    "training_hyperparameters",
    "data_processing",
    "reference_implementations",
}

REQUIRED_TRAINING_HP_ROWS = (
    "optimizer",
    "learning_rate",
    "lr_schedule",
    "weight_decay",
    "grad_clip",
    "warmup",
    "max_epochs",
    "batch_size",
    "training_protocol_notes",
)

SANITY_REQUIRED_KEYS = {
    "verdict",
    "attempt",
    "passed_checks",
    "failed_checks",
    "skipped_checks",
    "check_diagnostics",
    "dataset_candidate",
    "dataset_status",
    "fallback_trigger",
    "experiment_config",
    "timestamp",
}

TRAINING_REQUIRED_KEYS = {
    "status",
    "experiment_config",
    "dataset_candidate",
    "run_mode",
    "max_epochs_ceiling",
    "epochs_run",
    "early_stopping_triggered",
    "early_stopping_epoch",
    "batch_size_paper",
    "batch_size_chosen",
    "batch_probe_source",
    "lr_paper",
    "lr_scaled",
    "lr_scaling_rule",
    "optimizer_family",
    "resolved_config_hash",
    "hardware_id",
    "epoch_budget_rationale",
    "traceback_excerpt",
    "timestamp",
}

EVALUATION_REQUIRED_KEYS = {
    "status",
    "comparison_mode",
    "dataset_used",
    "paper_dataset",
    "hypothesis_status",
    "paper_claims_summary",
    "timestamp",
}


class WorkflowError(RuntimeError):
    """Raised when a workflow transition or artifact validation fails."""


@dataclass
class PhaseState:
    """Per-phase execution record."""

    name: str
    status: str = "pending"
    attempts: int = 0
    artifact_paths: list[str] = field(default_factory=list)
    started_at: str | None = None
    completed_at: str | None = None
    last_error_code: str | None = None
    last_error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowState:
    """Canonical state for one validate-paper run."""

    schema_version: int
    run_id: str
    repo_root: str
    paper_dir: str
    vault_dir: str
    run_mode: str
    overall_status: str
    current_phase: str
    started_at: str | None = None
    ended_at: str | None = None
    hp_mode: str = "reference_only"
    selected_dataset: str | None = None
    comparison_mode: str | None = None
    final_status: str | None = None
    blocker: dict[str, Any] | None = None
    dataset_history: list[dict[str, Any]] = field(default_factory=list)
    phase_order: list[str] = field(default_factory=lambda: list(PHASE_ORDER))
    phases: dict[str, PhaseState] = field(default_factory=dict)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def ensure_run_mode(run_mode: str) -> str:
    if run_mode not in {"blueprint-only", "quick", "full"}:
        raise WorkflowError(f"Unsupported run_mode: {run_mode}")
    return run_mode


def ensure_phase_name(phase: str) -> str:
    if phase not in PHASE_ORDER:
        raise WorkflowError(f"Unknown phase: {phase}")
    return phase


def state_path(vault_dir: Path) -> Path:
    return vault_dir / STATE_FILE_NAME


def phase_to_dict(phase: PhaseState) -> dict[str, Any]:
    return asdict(phase)


def state_to_dict(state: WorkflowState) -> dict[str, Any]:
    payload = asdict(state)
    payload["phases"] = {name: phase_to_dict(phase) for name, phase in state.phases.items()}
    return payload


def load_state(vault_dir: Path) -> WorkflowState:
    path = state_path(vault_dir)
    if not path.is_file():
        raise WorkflowError(f"Workflow state not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "phase_order" in data:
        data["phase_order"] = [
            "check_batch_fit" if name == "probe_batch_size" else name
            for name in data["phase_order"]
        ]
    phases = {}
    for name, phase_data in data.get("phases", {}).items():
        if name == "probe_batch_size":
            name = "check_batch_fit"
        phase_payload = dict(phase_data)
        phase_payload.pop("name", None)
        phases[name] = PhaseState(name=name, **phase_payload)
    for name in PHASE_ORDER:
        phases.setdefault(name, PhaseState(name=name))
    data["phases"] = phases
    data.setdefault("started_at", earliest_phase_started_at(phases))
    data.setdefault("ended_at", None)
    return WorkflowState(**data)


def write_state(state: WorkflowState) -> Path:
    path = state_path(Path(state.vault_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state_to_dict(state), indent=2, sort_keys=True), encoding="utf-8")
    return path


def earliest_phase_started_at(phases: dict[str, PhaseState]) -> str | None:
    timestamps = [phase.started_at for phase in phases.values() if phase.started_at]
    return min(timestamps) if timestamps else None


def mark_run_ended(state: WorkflowState, timestamp: str | None = None) -> None:
    state.ended_at = state.ended_at or timestamp or utc_now()


def ensure_hp_mode(hp_mode: str) -> str:
    if hp_mode not in {"reference_only", "reference_plus_paper"}:
        raise WorkflowError(f"Unsupported hp_mode: {hp_mode}")
    return hp_mode


def make_initial_state(repo_root: Path, paper_dir: Path, vault_dir: Path, run_mode: str, hp_mode: str = "reference_only") -> WorkflowState:
    mode = ensure_run_mode(run_mode)
    hp_mode = ensure_hp_mode(hp_mode)
    phases = {name: PhaseState(name=name) for name in PHASE_ORDER}
    if mode == "blueprint-only":
        for name in ["implement_components", "verify_static", "verify_sanity", "check_batch_fit", "run_training", "evaluate_results"]:
            phases[name].status = "skipped"
    return WorkflowState(
        schema_version=1,
        run_id=str(uuid.uuid4()),
        repo_root=str(repo_root),
        paper_dir=str(paper_dir),
        vault_dir=str(vault_dir),
        run_mode=mode,
        hp_mode=hp_mode,
        overall_status="active",
        current_phase="input_check",
        started_at=utc_now(),
        phases=phases,
    )


def phase_dependencies_satisfied(state: WorkflowState, phase: str) -> bool:
    deps = PHASE_DEPENDENCIES[phase]
    for dep in deps:
        dep_state = state.phases[dep]
        if dep_state.status not in {"passed", "skipped"}:
            return False
    return True


def next_pending_phase(state: WorkflowState) -> str:
    for name in state.phase_order:
        phase = state.phases[name]
        if phase.status == "pending" and phase_dependencies_satisfied(state, name):
            return name
    if state.run_mode == "blueprint-only" and state.phases["paper_hypothesis"].status == "passed":
        return "done"
    if state.phases["evaluate_results"].status == "passed":
        return "done"
    return state.current_phase


def resolve_artifact_paths(
    state: WorkflowState,
    phase: str,
    explicit_paths: list[str] | None = None,
) -> list[Path]:
    if explicit_paths:
        return [Path(path_str) for path_str in explicit_paths]
    relative_paths = DEFAULT_RELATIVE_ARTIFACTS.get(phase, [])
    vault_dir = Path(state.vault_dir)
    return [vault_dir / rel for rel in relative_paths]


def validate_artifacts(paths: list[Path]) -> list[str]:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise WorkflowError(f"Missing required artifacts: {missing}")
    return [str(path) for path in paths]


def _coerce_positive_int(value: Any, *, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise WorkflowError(f"{label} must be a positive integer, got: {value!r}") from exc
    if parsed <= 0:
        raise WorkflowError(f"{label} must be a positive integer, got: {parsed}")
    return parsed


def phase_attempt_limit(
    state: WorkflowState,
    phase_name: str,
    *,
    override: Any | None = None,
) -> int | None:
    if override is not None:
        return _coerce_positive_int(override, label=f"{phase_name} max_attempts")
    phase = state.phases[phase_name]
    metadata_limit = phase.metadata.get("max_attempts")
    if metadata_limit is not None:
        return _coerce_positive_int(metadata_limit, label=f"{phase_name} max_attempts")
    raw_limit = DEFAULT_PHASE_ATTEMPT_LIMITS.get(phase_name)
    if raw_limit is None:
        return None
    if isinstance(raw_limit, dict):
        limit = raw_limit.get(state.run_mode)
        if limit is None:
            limit = raw_limit.get("default")
        return _coerce_positive_int(limit, label=f"{phase_name} max_attempts") if limit is not None else None
    return _coerce_positive_int(raw_limit, label=f"{phase_name} max_attempts")


def _set_phase_budget_metadata(phase: PhaseState, limit: int | None) -> None:
    if limit is None:
        phase.metadata.pop("max_attempts", None)
        phase.metadata.pop("remaining_attempts", None)
        phase.metadata.pop("retry_budget_exhausted", None)
        return
    phase.metadata["max_attempts"] = limit
    phase.metadata["remaining_attempts"] = max(limit - phase.attempts, 0)
    if phase.attempts < limit:
        phase.metadata.pop("retry_budget_exhausted", None)


def _enforce_retry_budget(state: WorkflowState, phase_name: str) -> None:
    phase = state.phases[phase_name]
    limit = phase_attempt_limit(state, phase_name)
    _set_phase_budget_metadata(phase, limit)
    if (
        limit is None
        or phase.status != "failed"
        or not bool(phase.metadata.get("retryable"))
        or phase.attempts < limit
    ):
        return
    phase.metadata["retryable"] = False
    phase.metadata["retry_budget_exhausted"] = True
    detail = f"Retry budget exhausted for {phase_name}: attempts={phase.attempts}, max_attempts={limit}"
    if phase.last_error_message:
        if detail not in phase.last_error_message:
            phase.last_error_message = f"{phase.last_error_message} [{detail}]"
    else:
        phase.last_error_message = detail
    if not phase.last_error_code:
        phase.last_error_code = "RETRY_BUDGET_EXHAUSTED"


def apply_attempt_budgets(state: WorkflowState) -> None:
    for phase_name in state.phase_order:
        _enforce_retry_budget(state, phase_name)


def validate_required_keys(payload: dict[str, Any], required_keys: set[str], label: str) -> None:
    missing = sorted(key for key in required_keys if key not in payload)
    if missing:
        raise WorkflowError(f"{label} missing required keys: {missing}")


def _is_placeholder(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    raw = value.strip()
    if not raw:
        return True
    if raw == "NOT_SPECIFIED":
        return False
    lowered = raw.lower()
    if lowered in {"todo", "tbd", "unknown", "n/a", "none"}:
        return True
    return bool(re.fullmatch(r"\[[^\]]*\]|\.\.\.|<[^>]+>", raw))


def _coerce_training_hparams(value: Any, *, label: str) -> dict[str, dict[str, Any]]:
    if isinstance(value, dict):
        if all(param in value for param in REQUIRED_TRAINING_HP_ROWS):
            rows = value
        elif "rows" in value:
            rows = value["rows"]
        else:
            rows = value
    else:
        rows = value

    normalized: dict[str, dict[str, Any]] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                raise WorkflowError(f"{label} training_hyperparameters rows must be objects")
            name = row.get("parameter") or row.get("Parameter")
            if not name:
                raise WorkflowError(f"{label} training_hyperparameters row missing parameter")
            normalized[str(name).strip()] = {
                "value": row.get("value", row.get("Value")),
                "source_location": row.get("source_location", row.get("Source Location")),
                "category": row.get("category", row.get("Category")),
                "framework_default_available": row.get(
                    "framework_default_available",
                    row.get("Framework Default Available"),
                ),
            }
    elif isinstance(rows, dict):
        for name, raw_row in rows.items():
            if isinstance(raw_row, dict):
                normalized[str(name).strip()] = {
                    "value": raw_row.get("value", raw_row.get("Value")),
                    "source_location": raw_row.get(
                        "source_location",
                        raw_row.get("Source Location"),
                    ),
                    "category": raw_row.get("category", raw_row.get("Category")),
                    "framework_default_available": raw_row.get(
                        "framework_default_available",
                        raw_row.get("Framework Default Available"),
                    ),
                }
            else:
                normalized[str(name).strip()] = {
                    "value": raw_row,
                    "source_location": None,
                    "category": None,
                    "framework_default_available": None,
                }
    else:
        raise WorkflowError(f"{label} training_hyperparameters must be a mapping or row list")

    missing = [name for name in REQUIRED_TRAINING_HP_ROWS if name not in normalized]
    if missing:
        raise WorkflowError(f"{label} missing required training hyperparameter rows: {missing}")

    invalid = [
        name
        for name in REQUIRED_TRAINING_HP_ROWS
        if _is_placeholder(normalized[name].get("value"))
    ]
    if invalid:
        raise WorkflowError(
            f"{label} has empty or placeholder training hyperparameter values: {invalid}. "
            "Use concrete paper values or the literal NOT_SPECIFIED."
        )

    invalid_sources = [
        name
        for name in REQUIRED_TRAINING_HP_ROWS
        if _is_placeholder(normalized[name].get("source_location"))
    ]
    if invalid_sources:
        raise WorkflowError(
            f"{label} has empty or placeholder training hyperparameter source locations: "
            f"{invalid_sources}. Use a paper location or the literal NOT_SPECIFIED."
        )

    ordered = {name: normalized[name] for name in REQUIRED_TRAINING_HP_ROWS}
    for name, row in normalized.items():
        if name not in ordered:
            ordered[name] = row
    return ordered


def validate_conceptual_payload(payload: dict[str, Any]) -> None:
    validate_required_keys(payload, CONCEPTUAL_REQUIRED_KEYS, "Conceptual analysis sidecar")


def validate_algorithmic_payload(payload: dict[str, Any]) -> None:
    validate_required_keys(payload, ALGORITHMIC_REQUIRED_KEYS, "Algorithmic spec sidecar")
    payload["training_hyperparameters"] = _coerce_training_hparams(
        payload["training_hyperparameters"],
        label="Algorithmic spec sidecar",
    )


def validate_blueprint_payload(payload: dict[str, Any]) -> None:
    validate_required_keys(payload, BLUEPRINT_REQUIRED_KEYS, "Blueprint sidecar")
    payload["paper_hyperparameters"] = _coerce_training_hparams(
        payload["paper_hyperparameters"],
        label="Blueprint sidecar",
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_ready(payload), sort_keys=True))
        handle.write("\n")


def _json_ready(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _coerce_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    return None


def mark_overall_status(state: WorkflowState) -> None:
    retryable_failed_phase = next(
        (
            name
            for name in state.phase_order
            if state.phases[name].status == "failed"
            and bool(state.phases[name].metadata.get("retryable"))
        ),
        None,
    )
    terminal_failed_phase = next(
        (
            name
            for name in state.phase_order
            if state.phases[name].status == "failed"
            and not bool(state.phases[name].metadata.get("retryable"))
        ),
        None,
    )
    if state.run_mode == "blueprint-only" and state.phases["paper_hypothesis"].status == "passed":
        state.blocker = None
        state.overall_status = "completed"
        state.final_status = "blueprint_ready"
        state.current_phase = "done"
        mark_run_ended(state, state.phases["paper_hypothesis"].completed_at)
        return
    if state.phases["evaluate_results"].status == "passed":
        state.blocker = None
        state.overall_status = "completed"
        state.final_status = "evaluation_complete"
        state.current_phase = "done"
        mark_run_ended(state, state.phases["evaluate_results"].completed_at)
        return
    if terminal_failed_phase is not None:
        state.overall_status = "blocked"
        state.final_status = None
        if state.blocker and state.blocker.get("phase"):
            state.current_phase = str(state.blocker["phase"])
        else:
            state.current_phase = terminal_failed_phase
        mark_run_ended(state, state.phases[terminal_failed_phase].completed_at)
        return
    if retryable_failed_phase is not None:
        state.blocker = None
        state.overall_status = "active"
        state.final_status = None
        state.ended_at = None
        state.current_phase = retryable_failed_phase
        return
    state.blocker = None
    state.overall_status = "active"
    state.final_status = None
    state.ended_at = None
    state.current_phase = next_pending_phase(state)


def _parse_table(section_text: str) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in section_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        parts = [part.strip() for part in stripped.strip("|").split("|")]
        if len(parts) < 2:
            continue
        if parts[0] in {"Field", "Parameter"} or set(parts[0]) == {"-"}:
            continue
        rows[parts[0].strip("` ")] = parts[1].strip()
    return rows


def _parse_markdown_table_rows(section_text: str) -> list[dict[str, str]]:
    headers: list[str] | None = None
    rows: list[dict[str, str]] = []
    for line in section_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        parts = [part.strip() for part in stripped.strip("|").split("|")]
        if not parts:
            continue
        if headers is None:
            headers = parts
            continue
        if all(set(part) <= {"-", ":"} for part in parts):
            continue
        if headers and len(parts) >= len(headers):
            rows.append(dict(zip(headers, parts, strict=False)))
    return rows


def _training_hparams_from_markdown(section_text: str) -> dict[str, dict[str, Any]]:
    rows = _parse_markdown_table_rows(section_text)
    selected = [
        row
        for row in rows
        if (row.get("Parameter") or row.get("parameter") or "").strip()
        in REQUIRED_TRAINING_HP_ROWS
    ]
    return _coerce_training_hparams(selected, label="Markdown training hyperparameters")


def _extract_section(markdown: str, heading: str) -> str:
    marker = f"## {heading}"
    start = markdown.find(marker)
    if start < 0:
        return ""
    rest = markdown[start:]
    lines = rest.splitlines()
    collected: list[str] = []
    first = True
    for line in lines:
        if not first and line.startswith("## "):
            break
        collected.append(line)
        first = False
    return "\n".join(collected).strip()


def _parse_listish(value: str) -> list[str]:
    raw = value.strip()
    if raw in {"", "[]"}:
        return []
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip("`'\"") for item in inner.split(",") if item.strip()]
    return [raw.strip("`")]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _strip_writer_only_fields(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    clean = dict(payload)
    sections = clean.pop("markdown_sections", {}) or {}
    clean.pop("markdown", None)
    if not isinstance(sections, dict):
        raise WorkflowError("markdown_sections must be an object when provided")
    return clean, {str(key): str(value) for key, value in sections.items()}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")


def _section_from_payload(sections: dict[str, str], key: str, fallback: str = "") -> str:
    value = sections.get(key)
    if value:
        return value.strip()
    return fallback.strip()


def _render_conceptual_markdown(payload: dict[str, Any], sections: dict[str, str]) -> str:
    return "\n\n".join(
        part
        for part in (
            "# Conceptual Analysis",
            "> Machine contract: [[02-conceptual-analysis.json]]. Companion: [[03-algorithmic-spec]].",
            "## 1. Structure Map\n" + _section_from_payload(sections, "structure_map", str(payload.get("structure_map", ""))),
            "## 2. Novelty Assessment\n" + _section_from_payload(sections, "novelty_assessment", str(payload.get("novel_components", ""))),
            "## 3. Dataset Mapping\n" + _section_from_payload(sections, "dataset_mapping", str(payload.get("dataset_mapping", ""))),
            "## 4. Method Decomposition\n" + _section_from_payload(sections, "method_decomposition"),
            "## 5. Integration Roadmap\n" + _section_from_payload(sections, "integration_roadmap", str(payload.get("integration_roadmap", ""))),
        )
        if part.strip()
    ) + "\n"


def _render_hparam_table(rows: dict[str, dict[str, Any]]) -> str:
    lines = [
        "| Parameter | Value | Source Location | Category | Framework Default Available |",
        "|-----------|-------|-----------------|----------|-----------------------------|",
    ]
    for name in rows:
        row = rows[name]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    *[
                        str(row.get(key) or "NOT_SPECIFIED")
                        for key in (
                            "value",
                            "source_location",
                            "category",
                            "framework_default_available",
                        )
                    ],
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def _render_algorithmic_markdown(payload: dict[str, Any], sections: dict[str, str]) -> str:
    hparams = _coerce_training_hparams(
        payload["training_hyperparameters"],
        label="Algorithmic spec markdown render",
    )
    return "\n\n".join(
        part
        for part in (
            "# Algorithmic Specification",
            "> Machine contract: [[03-algorithmic-spec.json]]. Companion: [[02-conceptual-analysis]].",
            "## 1. Algorithm Extraction\n" + _section_from_payload(sections, "algorithms", str(payload.get("algorithms", ""))),
            "## 2. Equation Inventory\n" + _section_from_payload(sections, "equations", str(payload.get("equations", ""))),
            "## 3. Architecture Specification\n" + _section_from_payload(sections, "architectures", str(payload.get("architectures", ""))),
            "## 4. Loss Functions\n" + _section_from_payload(sections, "losses", str(payload.get("losses", ""))),
            "## 5. Training Hyperparameters (REQUIRED — complete table)\n" + _render_hparam_table(hparams),
            "## 6. Data Processing Specification\n" + _section_from_payload(sections, "data_processing", str(payload.get("data_processing", ""))),
            "## 7. Reference Implementations\n" + _section_from_payload(sections, "reference_implementations", str(payload.get("reference_implementations", ""))),
        )
        if part.strip()
    ) + "\n"


def _render_blueprint_markdown(payload: dict[str, Any], sections: dict[str, str]) -> str:
    hparams = _coerce_training_hparams(
        payload["paper_hyperparameters"],
        label="Blueprint markdown render",
    )
    readiness_rows = [
        ("experiment_config", payload.get("experiment_config")),
        ("task_type", payload.get("task_type")),
        ("paper_dataset", payload.get("paper_dataset")),
        ("evaluation_targets", payload.get("evaluation_targets")),
        ("excluded_paper_datasets", payload.get("excluded_paper_datasets")),
        ("framework_dataset_used", payload.get("framework_dataset_used")),
        ("datasets_are_same", payload.get("datasets_are_same")),
        ("comparison_mode", payload.get("comparison_mode")),
        ("fallback_allowed", payload.get("fallback_allowed")),
        ("dataset_fallback_candidates", payload.get("dataset_fallback_candidates")),
        ("required_new_files", payload.get("required_new_files")),
        ("validation_run_matrix", payload.get("validation_run_matrix", [])),
        ("dataset_contract", payload.get("dataset_contract")),
        ("model_io_contract", payload.get("model_io_contract")),
        ("split_contract", payload.get("split_contract")),
    ]
    readiness = "\n".join(
        ["| Field | Value |", "|-------|-------|"]
        + [f"| {key} | `{value}` |" for key, value in readiness_rows]
    )
    return "\n\n".join(
        part
        for part in (
            "# Implementation Blueprint",
            "> Machine contract: [[04-implementation-blueprint.json]]. Sources: [[02-conceptual-analysis.json]], [[03-algorithmic-spec.json]].",
            "## 0. Platform & Accelerator\n" + _section_from_payload(sections, "platform"),
            "## 1. Integration Summary\n" + _section_from_payload(sections, "integration_summary"),
            "## 2. Dataset Specification\n" + _section_from_payload(sections, "dataset_specification"),
            "## 3. Transform Specification\n" + _section_from_payload(sections, "transform_specification"),
            "## 4. Model Specification\n" + _section_from_payload(sections, "model_specification"),
            "## 5. Custom Loss\n" + _section_from_payload(sections, "custom_loss"),
            "## 6. Experiment Config\n" + _section_from_payload(sections, "experiment_config_yaml", str(payload.get("experiment_config", ""))),
            "## 7. Verification Protocol\n" + _section_from_payload(sections, "verification_protocol", str(payload.get("verification_protocol", ""))),
            "## 8.1 Training Hyperparameters (from paper — REQUIRED)\n" + _render_hparam_table(hparams),
            "## 8. Readiness Gate\n" + readiness,
            "## 9. Staged Build Plan\n" + _section_from_payload(sections, "staged_build_plan"),
        )
        if part.strip()
    ) + "\n"


def _parse_boolish(value: str) -> bool:
    normalized = value.strip().strip("`").lower()
    return normalized in {"true", "yes", "y"}


def _frontmatter_blocks(markdown: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    lines = markdown.splitlines()
    idx = 0
    while idx < len(lines):
        if lines[idx].strip() != "---":
            idx += 1
            continue
        idx += 1
        payload_lines: list[str] = []
        while idx < len(lines) and lines[idx].strip() != "---":
            payload_lines.append(lines[idx])
            idx += 1
        if idx < len(lines) and lines[idx].strip() == "---":
            raw = "\n".join(payload_lines)
            loaded = yaml.safe_load(raw) or {}
            if isinstance(loaded, dict):
                blocks.append(loaded)
        idx += 1
    return blocks


def _parse_claim_summary(text: str) -> dict[str, int]:
    summary = {
        "confirmed": 0,
        "contradicted": 0,
        "dataset_dependent": 0,
        "unassessable": 0,
        "unassessable_metric_scale": 0,
        "unassessable_no_overlapping_baseline": 0,
    }
    match = re.search(
        r"CONFIRMED\s+(\d+)\s*/\s*CONTRADICTED\s+(\d+)\s*/\s*DATASET_DEPENDENT\s+(\d+)\s*/\s*UNASSESSABLE\s+(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        summary.update(
            {
                "confirmed": int(match.group(1)),
                "contradicted": int(match.group(2)),
                "dataset_dependent": int(match.group(3)),
                "unassessable": int(match.group(4)),
            }
        )
    metric_scale_match = re.search(
        r"UNASSESSABLE_METRIC_SCALE\s+(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if metric_scale_match:
        summary["unassessable_metric_scale"] = int(metric_scale_match.group(1))
    no_baseline_match = re.search(
        r"UNASSESSABLE_NO_OVERLAPPING_BASELINE\s+(\d+)",
        text,
        flags=re.IGNORECASE,
    )
    if no_baseline_match:
        summary["unassessable_no_overlapping_baseline"] = int(no_baseline_match.group(1))
    return summary


def _conceptual_sidecar_from_markdown(vault_dir: Path) -> Path | None:
    md_path = vault_dir / "02-conceptual-analysis.md"
    json_path = vault_dir / "02-conceptual-analysis.json"
    if not md_path.is_file() or json_path.is_file():
        return json_path if json_path.is_file() else None
    markdown = md_path.read_text(encoding="utf-8")
    payload = {
        "schema_version": 1,
        "vault_dir": str(vault_dir),
        "markdown_path": str(md_path),
        "source_markdown_sha256": _sha256_text(markdown),
        "paper_summary": _extract_section(markdown, "1. Structure Map"),
        "structure_map": _extract_section(markdown, "1. Structure Map"),
        "novel_components": _extract_section(markdown, "2. Novelty Assessment"),
        "reused_components": _extract_section(markdown, "2. Novelty Assessment"),
        "configured_components": _extract_section(markdown, "2. Novelty Assessment"),
        "dataset_mapping": _extract_section(markdown, "3. Dataset Mapping"),
        "direct_evaluation_targets": [],
        "excluded_paper_datasets": [],
        "integration_roadmap": _extract_section(markdown, "5. Integration Roadmap"),
    }
    validate_conceptual_payload(payload)
    _write_json(json_path, payload)
    return json_path


def _algorithmic_sidecar_from_markdown(vault_dir: Path) -> Path | None:
    md_path = vault_dir / "03-algorithmic-spec.md"
    json_path = vault_dir / "03-algorithmic-spec.json"
    if not md_path.is_file() or json_path.is_file():
        return json_path if json_path.is_file() else None
    markdown = md_path.read_text(encoding="utf-8")
    hparam_section = _extract_section(markdown, "5. Training Hyperparameters")
    if not hparam_section:
        hparam_section = _extract_section(markdown, "5. Training Hyperparameters (REQUIRED — complete table)")
    payload = {
        "schema_version": 1,
        "vault_dir": str(vault_dir),
        "markdown_path": str(md_path),
        "source_markdown_sha256": _sha256_text(markdown),
        "algorithms": _extract_section(markdown, "1. Algorithm Extraction"),
        "equations": _extract_section(markdown, "2. Equation Inventory"),
        "architectures": _extract_section(markdown, "3. Architecture Specification"),
        "losses": _extract_section(markdown, "4. Loss Functions"),
        "training_hyperparameters": _training_hparams_from_markdown(hparam_section),
        "data_processing": _extract_section(markdown, "6. Data Processing Specification"),
        "reference_implementations": _extract_section(markdown, "7. Reference Implementations"),
    }
    validate_algorithmic_payload(payload)
    _write_json(json_path, payload)
    return json_path


def _blueprint_sidecar_from_markdown(vault_dir: Path) -> Path | None:
    md_path = vault_dir / "04-implementation-blueprint.md"
    json_path = vault_dir / "04-implementation-blueprint.json"
    if not md_path.is_file() or json_path.is_file():
        return json_path if json_path.is_file() else None
    markdown = md_path.read_text(encoding="utf-8")
    readiness = _parse_table(_extract_section(markdown, "8. Readiness Gate"))
    verification_protocol = _extract_section(markdown, "7. Verification Protocol")
    if not readiness:
        return None
    hparam_section = _extract_section(
        markdown,
        "8.1 Training Hyperparameters (from paper — REQUIRED)",
    )
    payload = {
        "schema_version": 1,
        "vault_dir": str(vault_dir),
        "markdown_path": str(md_path),
        "source_markdown_sha256": _sha256_text(markdown),
        "paper_dataset": readiness.get("paper_dataset"),
        "evaluation_targets": _parse_listish(readiness.get("evaluation_targets", "[]")),
        "excluded_paper_datasets": _parse_listish(
            readiness.get("excluded_paper_datasets", "[]")
        ),
        "framework_dataset_used": readiness.get("framework_dataset_used"),
        "datasets_are_same": _parse_boolish(readiness.get("datasets_are_same", "false")),
        "comparison_mode": readiness.get("comparison_mode", "").strip("`"),
        "fallback_allowed": _parse_boolish(readiness.get("fallback_allowed", "true")),
        "dataset_fallback_candidates": _parse_listish(
            readiness.get("dataset_fallback_candidates", "[]")
        ),
        "experiment_config": readiness.get("experiment_config", "").strip("`"),
        "task_type": readiness.get("task_type", "").strip("`"),
        "required_new_files": _parse_listish(readiness.get("required_new_files", "[]")),
        "verification_protocol": verification_protocol,
        "paper_hyperparameters": _training_hparams_from_markdown(hparam_section),
        "dataset_contract": readiness.get("dataset_contract", "").strip(),
        "model_io_contract": readiness.get("model_io_contract", "").strip(),
        "split_contract": readiness.get("split_contract", "").strip(),
        "validation_run_matrix": _parse_listish(
            readiness.get("validation_run_matrix", "[]")
        ),
    }
    if any(
        payload.get(key) is None or payload.get(key) == ""
        for key in BLUEPRINT_REQUIRED_KEYS
        if key != "verification_protocol"
    ):
        return None
    validate_blueprint_payload(payload)
    _write_json(json_path, payload)
    return json_path


def _sanity_sidecar_from_markdown(vault_dir: Path) -> Path | None:
    md_path = vault_dir / "06-sanity-ladder-log.md"
    jsonl_path = vault_dir / "06-sanity-ladder-log.jsonl"
    if not md_path.is_file() or jsonl_path.is_file():
        return jsonl_path if jsonl_path.is_file() else None
    markdown = md_path.read_text(encoding="utf-8")
    blocks = _frontmatter_blocks(markdown)
    if not blocks:
        blocks = _legacy_sanity_blocks(markdown)
    if not blocks:
        return None
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for block in blocks:
            if not all(key in block for key in SANITY_REQUIRED_KEYS):
                return None
            handle.write(json.dumps(block, sort_keys=True))
            handle.write("\n")
    return jsonl_path


def _legacy_sanity_blocks(markdown: str) -> list[dict[str, Any]]:
    raw_blocks = markdown.split("## Sanity Ladder -- Runtime Checks")
    parsed: list[dict[str, Any]] = []
    for raw in raw_blocks[1:]:
        block = raw.strip()
        if not block:
            continue
        attempt_match = re.search(r"\*\*Attempt\*\*:\s*(\d+)", block)
        experiment_match = re.search(r"\*\*Experiment config\*\*:\s*`([^`]+)`", block)
        dataset_candidate_match = re.search(
            r"\*\*Dataset candidate\*\*:\s*([^\n]+)",
            block,
        )
        dataset_status_match = re.search(
            r"### Dataset Executability.*?\*\*Status\*\*:\s*([A-Z_]+)",
            block,
            flags=re.DOTALL,
        )
        fallback_match = re.search(
            r"\*\*(?:Fallback trigger|Dataset recovery required)\*\*:\s*(yes|no)",
            block,
            flags=re.IGNORECASE,
        )
        verdict_match = re.search(r"\*\*Verdict\*\*:\s*([A-Z_]+)", block)
        timestamp_match = re.search(r"\*\*Timestamp\*\*:\s*([^\n]+)", block)
        checks: list[tuple[str, str, dict[str, Any]]] = []
        for check_match in re.finditer(
            r"### Check:\s*([^\n]+)\n(.*?)(?=\n### Check:|\n### Summary|\Z)",
            block,
            flags=re.DOTALL,
        ):
            check_name = check_match.group(1).strip()
            check_body = check_match.group(2)
            status_match = re.search(r"\*\*Status\*\*:\s*([A-Z_]+)", check_body)
            status = status_match.group(1) if status_match else "UNKNOWN"
            metrics: dict[str, Any] = {}
            for metric_match in re.finditer(
                r"- \*\*([^*]+)\*\*:\s*`([^`]+)`",
                check_body,
            ):
                key = metric_match.group(1).strip()
                raw_value = metric_match.group(2)
                try:
                    metrics[key] = json.loads(raw_value)
                except json.JSONDecodeError:
                    metrics[key] = raw_value
            checks.append((check_name, status, metrics))

        if not (attempt_match and experiment_match and verdict_match and timestamp_match):
            continue

        passed_checks = [name for name, status, _ in checks if status == "PASS"]
        skipped_checks = [name for name, status, _ in checks if status == "SKIPPED"]
        failed_checks = [
            name for name, status, _ in checks if status not in {"PASS", "SKIPPED"}
        ]
        verdict = verdict_match.group(1).strip()
        if verdict == "PROCEED":
            verdict = "PASS"
        parsed.append(
            {
                "verdict": verdict,
                "attempt": int(attempt_match.group(1)),
                "passed_checks": passed_checks,
                "failed_checks": failed_checks,
                "skipped_checks": skipped_checks,
                "check_diagnostics": {name: metrics for name, _, metrics in checks},
                "dataset_candidate": (
                    dataset_candidate_match.group(1).strip()
                    if dataset_candidate_match
                    else None
                ),
                "dataset_status": (
                    dataset_status_match.group(1).strip()
                    if dataset_status_match
                    else "PASS"
                ),
                "fallback_trigger": (
                    fallback_match.group(1).strip().lower() == "yes"
                    if fallback_match
                    else False
                ),
                "experiment_config": experiment_match.group(1).strip(),
                "timestamp": timestamp_match.group(1).strip(),
            }
        )
    return parsed


def _training_sidecar_from_markdown(vault_dir: Path) -> Path | None:
    md_path = vault_dir / "07-training-log.md"
    jsonl_path = vault_dir / "07-training-log.jsonl"
    if not md_path.is_file() or jsonl_path.is_file():
        return jsonl_path if jsonl_path.is_file() else None
    markdown = md_path.read_text(encoding="utf-8")
    blocks = _frontmatter_blocks(markdown)
    if not blocks:
        return None
    latest = blocks[-1]
    if not all(key in latest for key in TRAINING_REQUIRED_KEYS):
        return None
    with jsonl_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_ready(latest), sort_keys=True))
        handle.write("\n")
    return jsonl_path


def _latest_jsonl_record(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return records[-1] if records else None


def _latest_sanity_record(vault_dir: Path) -> dict[str, Any] | None:
    jsonl_path = vault_dir / "06-sanity-ladder-log.jsonl"
    latest = _latest_jsonl_record(jsonl_path)
    if latest is not None:
        return latest
    md_path = vault_dir / "06-sanity-ladder-log.md"
    if not md_path.is_file():
        return None
    markdown = md_path.read_text(encoding="utf-8")
    blocks = _frontmatter_blocks(markdown) or _legacy_sanity_blocks(markdown)
    return blocks[-1] if blocks else None


def _latest_training_record(vault_dir: Path) -> dict[str, Any] | None:
    jsonl_path = vault_dir / "07-training-log.jsonl"
    latest = _latest_jsonl_record(jsonl_path)
    if latest is not None:
        return latest
    md_path = vault_dir / "07-training-log.md"
    if not md_path.is_file():
        return None
    blocks = _frontmatter_blocks(md_path.read_text(encoding="utf-8"))
    return blocks[-1] if blocks else None


def _latest_batch_fit_record(vault_dir: Path) -> dict[str, Any] | None:
    path = vault_dir / "batch_fit_check.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"status": "BATCH_FIT_CHECK_UNPARSEABLE"}
    if isinstance(payload, list):
        records = [record for record in payload if isinstance(record, dict)]
        return records[-1] if records else None
    if isinstance(payload, dict):
        records = payload.get("records")
        if isinstance(records, list):
            dict_records = [record for record in records if isinstance(record, dict)]
            return dict_records[-1] if dict_records else None
        return payload
    return {"status": "BATCH_FIT_CHECK_UNPARSEABLE"}


def infer_phase_status_from_artifacts(vault_dir: Path, phase_name: str) -> tuple[str | None, dict[str, Any]]:
    if phase_name == "verify_sanity":
        latest = _latest_sanity_record(vault_dir)
        if latest is None:
            return None, {}
        verdict = str(latest.get("verdict", "")).strip().upper()
        metadata = {"latest_verdict": verdict}
        if verdict in {"PASS", "WARN_CONTINUE", "INVESTIGATE"}:
            if verdict != "PASS":
                metadata["sanity_warning"] = True
            return "passed", metadata
        metadata["retryable"] = verdict in {
            "ABORTED",
            "BLOCK",
            "INVESTIGATE",
            "DATASET_EXECUTION_FAILED",
            "DATASET_UNAVAILABLE",
            "PRECHECK_TIMEOUT",
            "TOOL_INVOCATION_FAILURE",
        }
        return "failed", {
            **metadata,
            "reason_code": verdict or "SANITY_RESULT_UNPARSEABLE",
        }
    if phase_name == "check_batch_fit":
        latest = _latest_batch_fit_record(vault_dir)
        if latest is None:
            return None, {}
        status = str(latest.get("status", "")).strip().upper()
        metadata = {"latest_status": status}
        for key in (
            "resolved_config_hash",
            "hardware_id",
            "timeout_sec",
            "force_fresh",
            "dataset_candidate",
        ):
            if key in latest:
                metadata[key] = latest[key]
        if status == "OK":
            return "passed", metadata
        metadata["retryable"] = status in {
            "DATASET_EXECUTION_FAILED",
            "DATASET_UNAVAILABLE",
            "FAILED",
            "TIMEOUT",
        }
        if status == "TIMEOUT":
            metadata["recovery_action"] = (
                "inspect progress; increase timeout if healthy-but-slow, otherwise repair same-dataset wiring or stop"
            )
        return "failed", {
            **metadata,
            "reason_code": status or "BATCH_FIT_CHECK_UNPARSEABLE",
        }
    if phase_name == "run_training":
        latest = _latest_training_record(vault_dir)
        if latest is None:
            return None, {}
        status = str(latest.get("status", "")).strip().upper()
        metadata = {"latest_status": status}
        if status in {"SUCCESS", "PARTIAL"}:
            return "passed", metadata
        metadata["retryable"] = status in {
            "CRASHED",
            "DATASET_EXECUTION_FAILED",
            "DATASET_UNAVAILABLE",
            "FAILED",
        }
        return "failed", {
            **metadata,
            "reason_code": status or "TRAINING_RESULT_UNPARSEABLE",
        }
    if phase_name == "evaluate_results":
        json_path = vault_dir / "08-evaluation-report.json"
        if not json_path.is_file():
            return None, {}
        latest = json.loads(json_path.read_text(encoding="utf-8"))
        status = str(latest.get("status", "")).strip().upper()
        artifact_status = str(latest.get("artifact_status", "COMPLETE")).strip().upper()
        technical_status = str(latest.get("technical_status", "PASS")).strip().upper()
        blocking = _coerce_optional_bool(latest.get("blocking"))
        metadata = {
            "latest_status": status,
            "artifact_status": artifact_status,
            "technical_status": technical_status,
            "scientific_status": str(latest.get("scientific_status", status)).strip().upper(),
        }
        technical_failure_statuses = {
            "CORRUPT_NETCDF",
            "EVALUATION_ERROR",
            "EVALUATOR_ERROR",
            "IMPLEMENTATION_BUG",
            "MISSING_METRICS",
            "REPORT_UNPARSEABLE",
        }
        technical_failure_markers = {
            "CORRUPT",
            "EVALUATOR_ERROR",
            "HARD_BLOCKER",
            "IMPLEMENTATION_BUG",
            "MISSING",
            "REPAIRABLE_BUG",
        }
        if (
            blocking is True
            or status in technical_failure_statuses
            or artifact_status in {"MISSING", "CORRUPT"}
            or technical_status in technical_failure_markers
        ):
            metadata["retryable"] = status in {
                "EVALUATION_ERROR",
                "EVALUATOR_ERROR",
                "IMPLEMENTATION_BUG",
                "MISSING_METRICS",
            } or technical_status in {
                "EVALUATOR_ERROR",
                "IMPLEMENTATION_BUG",
                "REPAIRABLE_BUG",
            }
            return "failed", {
                **metadata,
                "reason_code": status or technical_status or "EVALUATION_RESULT_UNPARSEABLE",
            }
        if status in {
            "BENCHMARK_ONLY",
            "INVESTIGATE",
            "INVESTIGATE_CLAIMS_DISPUTED",
            "OK",
            "PLAUSIBLE",
            "STALE_DATASET_MISMATCH",
            "VALIDATED",
        } or (blocking is False and artifact_status == "COMPLETE"):
            return "passed", metadata
        metadata["retryable"] = False
        return "failed", {
            **metadata,
            "reason_code": status or "EVALUATION_RESULT_UNPARSEABLE",
        }
    return None, {}


def _evaluation_sidecar_from_markdown(vault_dir: Path) -> Path | None:
    md_path = vault_dir / "08-evaluation-report.md"
    json_path = vault_dir / "08-evaluation-report.json"
    if not md_path.is_file() or json_path.is_file():
        return json_path if json_path.is_file() else None

    blueprint_json = vault_dir / "04-implementation-blueprint.json"
    if not blueprint_json.is_file():
        blueprint_json = _blueprint_sidecar_from_markdown(vault_dir) or blueprint_json
    if not blueprint_json.is_file():
        return None

    blueprint = json.loads(blueprint_json.read_text(encoding="utf-8"))
    markdown = md_path.read_text(encoding="utf-8")
    classification_match = re.search(
        r"^### Classification:\s*(.+?)\s*$",
        markdown,
        flags=re.MULTILINE,
    )
    artifact_status_match = re.search(
        r"^### Artifact Status:\s*(.+?)\s*$",
        markdown,
        flags=re.MULTILINE,
    )
    technical_status_match = re.search(
        r"^### Technical Status:\s*(.+?)\s*$",
        markdown,
        flags=re.MULTILINE,
    )
    hypothesis_match = re.search(
        r"Pre-registered from \[\[05-paper-hypothesis\]\] \(([^)]+)\)",
        markdown,
    )
    summary_match = re.search(
        r"^\*\*Summary\*\*:\s*(CONFIRMED.+)$",
        markdown,
        flags=re.MULTILINE,
    )
    payload = {
        "vault_dir": str(vault_dir),
        "status": classification_match.group(1).strip() if classification_match else "",
        "artifact_status": (
            artifact_status_match.group(1).strip() if artifact_status_match else "COMPLETE"
        ),
        "technical_status": (
            technical_status_match.group(1).strip() if technical_status_match else "PASS"
        ),
        "scientific_status": classification_match.group(1).strip() if classification_match else "",
        "blocking": (
            (technical_status_match.group(1).strip().upper() if technical_status_match else "PASS")
            not in {"PASS", "OK"}
        ),
        "observations": [],
        "comparison_mode": blueprint.get("comparison_mode", ""),
        "dataset_used": blueprint.get("framework_dataset_used", ""),
        "paper_dataset": blueprint.get("paper_dataset", ""),
        "hypothesis_status": hypothesis_match.group(1).strip() if hypothesis_match else "ABSENT",
        "paper_claims_summary": _parse_claim_summary(summary_match.group(1) if summary_match else markdown),
        "timestamp": utc_now(),
    }
    if any(
        payload.get(key) is None or payload.get(key) == ""
        for key in ("status", "comparison_mode", "dataset_used", "paper_dataset")
    ):
        return None
    validate_required_keys(payload, EVALUATION_REQUIRED_KEYS, "Evaluation sidecar")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return json_path


def backfill_sidecars(vault_dir: Path) -> dict[str, str]:
    created: dict[str, str] = {}
    for key, creator in (
        ("02-conceptual-analysis.json", _conceptual_sidecar_from_markdown),
        ("03-algorithmic-spec.json", _algorithmic_sidecar_from_markdown),
        ("04-implementation-blueprint.json", _blueprint_sidecar_from_markdown),
        ("06-sanity-ladder-log.jsonl", _sanity_sidecar_from_markdown),
        ("07-training-log.jsonl", _training_sidecar_from_markdown),
        ("08-evaluation-report.json", _evaluation_sidecar_from_markdown),
    ):
        try:
            path = creator(vault_dir)
        except WorkflowError:
            path = None
        if path is not None and path.exists():
            created[key] = str(path)
    return created


def command_init_run(payload: dict[str, Any]) -> dict[str, Any]:
    repo_root = Path(payload["repo_root"]).resolve()
    paper_dir = Path(payload["paper_dir"]).resolve()
    vault_dir = Path(payload.get("vault_dir") or payload["paper_dir"]).resolve()
    run_mode = ensure_run_mode(payload.get("run_mode", "full"))
    hp_mode = ensure_hp_mode(payload.get("hp_mode", "reference_only"))
    state_file = state_path(vault_dir)
    if state_file.exists():
        state = load_state(vault_dir)
        apply_attempt_budgets(state)
        write_state(state)
        return {"status": "EXISTS", "state_path": str(state_file), "state": state_to_dict(state)}
    state = make_initial_state(repo_root=repo_root, paper_dir=paper_dir, vault_dir=vault_dir, run_mode=run_mode, hp_mode=hp_mode)
    apply_attempt_budgets(state)
    write_state(state)
    return {"status": "CREATED", "state_path": str(state_file), "state": state_to_dict(state)}


def command_get_status(payload: dict[str, Any]) -> dict[str, Any]:
    vault_dir = Path(payload["vault_dir"]).resolve()
    backfill_sidecars(vault_dir)
    state = load_state(vault_dir)
    apply_attempt_budgets(state)
    mark_overall_status(state)
    write_state(state)
    return {
        "status": state.overall_status.upper(),
        "current_phase": state.current_phase,
        "final_status": state.final_status,
        "blocker": state.blocker,
        "state_path": str(state_path(Path(state.vault_dir))),
        "state": state_to_dict(state),
    }


def command_start_phase(payload: dict[str, Any]) -> dict[str, Any]:
    state = load_state(Path(payload["vault_dir"]).resolve())
    phase_name = ensure_phase_name(payload["phase"])
    if not phase_dependencies_satisfied(state, phase_name):
        raise WorkflowError(f"Phase dependencies not satisfied for {phase_name}")
    phase = state.phases[phase_name]
    limit = phase_attempt_limit(state, phase_name, override=payload.get("max_attempts"))
    _set_phase_budget_metadata(phase, limit)
    if limit is not None and phase.attempts >= limit:
        raise WorkflowError(
            f"Retry budget exhausted for {phase_name}: attempts={phase.attempts}, max_attempts={limit}"
        )
    phase.attempts += 1
    phase.status = "running"
    phase.started_at = utc_now()
    phase.last_error_code = None
    phase.last_error_message = None
    _set_phase_budget_metadata(phase, limit)
    state.current_phase = phase_name
    state.overall_status = "active"
    state.final_status = None
    state.ended_at = None
    state.blocker = None
    write_state(state)
    return {"status": "RUNNING", "phase": phase_name, "attempt": phase.attempts, "state": state_to_dict(state)}


def command_complete_phase(payload: dict[str, Any]) -> dict[str, Any]:
    vault_dir = Path(payload["vault_dir"]).resolve()
    backfill_sidecars(vault_dir)
    state = load_state(vault_dir)
    phase_name = ensure_phase_name(payload["phase"])
    phase = state.phases[phase_name]
    artifact_paths = resolve_artifact_paths(state, phase_name, payload.get("artifact_paths"))
    validated_paths = validate_artifacts(artifact_paths)
    phase.status = "passed"
    phase.completed_at = utc_now()
    phase.artifact_paths = validated_paths
    phase.last_error_code = None
    phase.last_error_message = None
    phase.metadata.update(payload.get("metadata") or {})
    phase.metadata["retryable"] = False
    if "selected_dataset" in phase.metadata:
        state.selected_dataset = str(phase.metadata["selected_dataset"])
    if "comparison_mode" in phase.metadata:
        state.comparison_mode = str(phase.metadata["comparison_mode"])
    if "dataset_transition" in payload:
        state.dataset_history.append(payload["dataset_transition"])
    apply_attempt_budgets(state)
    mark_overall_status(state)
    write_state(state)
    return {"status": "PASSED", "phase": phase_name, "next_phase": state.current_phase, "state": state_to_dict(state)}


def command_fail_phase(payload: dict[str, Any]) -> dict[str, Any]:
    state = load_state(Path(payload["vault_dir"]).resolve())
    phase_name = ensure_phase_name(payload["phase"])
    phase = state.phases[phase_name]
    retryable = bool(payload.get("retryable", False))
    limit = phase_attempt_limit(state, phase_name, override=payload.get("max_attempts"))
    _set_phase_budget_metadata(phase, limit)
    phase.status = "failed"
    phase.completed_at = utc_now()
    phase.last_error_code = payload.get("reason_code")
    phase.last_error_message = payload.get("message")
    phase.metadata.update(payload.get("metadata") or {})
    phase.metadata["retryable"] = retryable
    _enforce_retry_budget(state, phase_name)
    retryable = bool(phase.metadata.get("retryable"))
    if retryable:
        state.blocker = None
        state.overall_status = "active"
        state.final_status = None
        state.ended_at = None
        state.current_phase = phase_name
    else:
        state.blocker = {
            "phase": phase_name,
            "reason_code": payload.get("reason_code"),
            "message": payload.get("message"),
            "timestamp": phase.completed_at,
        }
        state.overall_status = "blocked"
        state.final_status = None
        mark_run_ended(state, phase.completed_at)
        state.current_phase = phase_name
    write_state(state)
    return {"status": "FAILED", "phase": phase_name, "state": state_to_dict(state)}


def command_sync_from_artifacts(payload: dict[str, Any]) -> dict[str, Any]:
    vault_dir = Path(payload["vault_dir"]).resolve()
    created_sidecars = backfill_sidecars(vault_dir)
    try:
        state = load_state(vault_dir)
    except WorkflowError:
        repo_root = Path(payload["repo_root"]).resolve()
        paper_dir = Path(payload["paper_dir"]).resolve()
        state = make_initial_state(
            repo_root=repo_root,
            paper_dir=paper_dir,
            vault_dir=vault_dir,
            run_mode=payload.get("run_mode", "full"),
            hp_mode=payload.get("hp_mode", "reference_only"),
        )

    state.blocker = None
    for phase_name in PHASE_ORDER:
        phase = state.phases[phase_name]
        if phase.status in {"passed", "skipped"}:
            continue
        artifact_paths = resolve_artifact_paths(state, phase_name)
        inferred_status, metadata = infer_phase_status_from_artifacts(vault_dir, phase_name)
        existing_artifacts = [str(path) for path in artifact_paths if path.exists()]
        all_artifacts_present = bool(artifact_paths) and len(existing_artifacts) == len(artifact_paths)
        if all_artifacts_present or inferred_status == "failed":
            phase.status = inferred_status or "passed"
            phase.completed_at = phase.completed_at or utc_now()
            phase.artifact_paths = existing_artifacts
            phase.metadata.update(metadata)
            if phase.status == "failed":
                phase.last_error_code = str(
                    metadata.get("reason_code") or f"{phase_name.upper()}_FAILED"
                )
                phase.last_error_message = f"{phase_name} artifact indicates failure"
                if state.blocker is None and not bool(metadata.get("retryable")):
                    state.blocker = {
                        "phase": phase_name,
                        "reason_code": phase.last_error_code,
                        "message": phase.last_error_message,
                        "timestamp": phase.completed_at,
                    }

    apply_attempt_budgets(state)
    mark_overall_status(state)
    write_state(state)
    return {
        "status": "SYNCED",
        "current_phase": state.current_phase,
        "created_sidecars": created_sidecars,
        "state": state_to_dict(state),
    }


def command_write_blueprint_sidecar(payload: dict[str, Any]) -> dict[str, Any]:
    clean, sections = _strip_writer_only_fields(payload)
    validate_blueprint_payload(clean)
    vault_dir = Path(clean["vault_dir"]).resolve()
    json_path = vault_dir / "04-implementation-blueprint.json"
    md_path = vault_dir / "04-implementation-blueprint.md"
    _write_json(json_path, clean)
    rendered = _render_blueprint_markdown(clean, sections)
    md_path.write_text(rendered, encoding="utf-8")
    return {"status": "WROTE", "artifact_path": str(json_path), "markdown_path": str(md_path)}


def command_write_conceptual_sidecar(payload: dict[str, Any]) -> dict[str, Any]:
    clean, sections = _strip_writer_only_fields(payload)
    validate_conceptual_payload(clean)
    vault_dir = Path(clean["vault_dir"]).resolve()
    json_path = vault_dir / "02-conceptual-analysis.json"
    md_path = vault_dir / "02-conceptual-analysis.md"
    _write_json(json_path, clean)
    rendered = _render_conceptual_markdown(clean, sections)
    md_path.write_text(rendered, encoding="utf-8")
    return {"status": "WROTE", "artifact_path": str(json_path), "markdown_path": str(md_path)}


def command_write_algorithmic_sidecar(payload: dict[str, Any]) -> dict[str, Any]:
    clean, sections = _strip_writer_only_fields(payload)
    validate_algorithmic_payload(clean)
    vault_dir = Path(clean["vault_dir"]).resolve()
    json_path = vault_dir / "03-algorithmic-spec.json"
    md_path = vault_dir / "03-algorithmic-spec.md"
    _write_json(json_path, clean)
    rendered = _render_algorithmic_markdown(clean, sections)
    md_path.write_text(rendered, encoding="utf-8")
    return {"status": "WROTE", "artifact_path": str(json_path), "markdown_path": str(md_path)}


def command_append_sanity_result(payload: dict[str, Any]) -> dict[str, Any]:
    validate_required_keys(payload, SANITY_REQUIRED_KEYS, "Sanity result")
    path = Path(payload["vault_dir"]).resolve() / "06-sanity-ladder-log.jsonl"
    append_jsonl(path, payload)
    return {"status": "APPENDED", "artifact_path": str(path)}


def command_append_training_result(payload: dict[str, Any]) -> dict[str, Any]:
    validate_required_keys(payload, TRAINING_REQUIRED_KEYS, "Training result")
    path = Path(payload["vault_dir"]).resolve() / "07-training-log.jsonl"
    append_jsonl(path, payload)
    return {"status": "APPENDED", "artifact_path": str(path)}


def command_write_evaluation_sidecar(payload: dict[str, Any]) -> dict[str, Any]:
    validate_required_keys(payload, EVALUATION_REQUIRED_KEYS, "Evaluation sidecar")
    path = Path(payload["vault_dir"]).resolve() / "08-evaluation-report.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"status": "WROTE", "artifact_path": str(path)}


COMMANDS = {
    "init_run": command_init_run,
    "get_status": command_get_status,
    "start_phase": command_start_phase,
    "complete_phase": command_complete_phase,
    "fail_phase": command_fail_phase,
    "sync_from_artifacts": command_sync_from_artifacts,
    "write_conceptual_sidecar": command_write_conceptual_sidecar,
    "write_algorithmic_sidecar": command_write_algorithmic_sidecar,
    "write_blueprint_sidecar": command_write_blueprint_sidecar,
    "append_sanity_result": command_append_sanity_result,
    "append_training_result": command_append_training_result,
    "write_evaluation_sidecar": command_write_evaluation_sidecar,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument("--input-json", required=True, help="JSON payload for the selected command")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = json.loads(args.input_json)
    try:
        result = COMMANDS[args.command](payload)
    except WorkflowError as exc:
        result = {"status": "ERROR", "reason": str(exc), "command": args.command}
        print(f"{RESULT_MARKER}{json.dumps(result, sort_keys=True)}")
        return 1
    print(f"{RESULT_MARKER}{json.dumps(result, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
