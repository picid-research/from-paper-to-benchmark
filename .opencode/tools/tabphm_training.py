"""Execute a PICID training run and write structured training artifacts.

Usage:
    uv run python .opencode/tools/PICID_training.py run --input-json '<payload>'
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

TOOL_DIR = Path(__file__).resolve().parent
SANITY_DIR = TOOL_DIR / "PICID_sanity"
if str(TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(TOOL_DIR))
if str(SANITY_DIR) not in sys.path:
    sys.path.insert(0, str(SANITY_DIR))

from picid.utils.hash_utils import hash_config  # noqa: E402

from validate_paper_workflow import _frontmatter_blocks, _legacy_sanity_blocks  # noqa: E402
from sanity_base import (  # noqa: E402
    INVESTIGATE,
    PASS,
    WARN_CONTINUE,
    base_overrides,
    classify_dataset_failure,
    traceback_excerpt,
)

ACCEPTED_SANITY_VERDICTS = {PASS, WARN_CONTINUE, INVESTIGATE}

RESULT_MARKER = "PICID_TRAINING_RESULT_JSON="
SUCCESS_STATUSES = {"SUCCESS", "PARTIAL"}
PROFILE_PATH = TOOL_DIR.parent / "reference" / "hparam-profiles.yaml"
FIXED_VALIDATION_BATCH_OVERRIDES = (
    "datamodule.train_batch_size=512",
    "datamodule.val_batch_size=1024",
    "datamodule.test_batch_size=1024",
)
FIXED_VALIDATION_BATCH_KEYS = {
    "datamodule.train_batch_size",
    "datamodule.val_batch_size",
    "datamodule.test_batch_size",
}
DEFAULT_MONITOR_INTERVAL_SEC = 60.0
DEFAULT_PLATEAU_PATIENCE_EPOCHS = 20
DEFAULT_PLATEAU_MIN_EPOCHS = 10
PLATEAU_REL_IMPROVEMENT = 1e-3
MAX_MONITOR_EVENTS = 20
MAX_TEXT_EXCERPT_CHARS = 4000
RUN_ROOT_MARKERS = ("REPRODUCE.md", "run_metadata.yaml", "config_resolved.yaml", ".hydra")
TRAIN_LOSS_KEYS = ("train/loss_epoch", "train_loss_epoch", "train/loss", "train_loss")
VAL_LOSS_KEYS = ("val/loss", "val_loss")
METRIC_KEY_ALIASES = {
    "train_loss_epoch": TRAIN_LOSS_KEYS,
    "train/loss_epoch": TRAIN_LOSS_KEYS,
    "train_loss": TRAIN_LOSS_KEYS,
    "train/loss": TRAIN_LOSS_KEYS,
    "val_loss": VAL_LOSS_KEYS,
    "val/loss": VAL_LOSS_KEYS,
}
TEST_LOG_MARKERS = (
    "test/loss",
    "test/mae",
    "test/rmse",
    "trainer.test",
    "testing dataloader",
    "test stage",
)


@dataclass
class BatchFitCheckRecord:
    resolved_config_hash: str
    hardware_id: str
    status: str
    train_batch_size: int | str | None
    val_batch_size: int | str | None
    test_batch_size: int | str | None
    timestamp: str


@dataclass
class TrainingInvocationError(RuntimeError):
    reason_code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.reason_code}: {self.detail}"


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def payload_arg(payload: dict[str, Any], key: str, default: Any = None) -> Any:
    value = payload.get(key, default)
    return default if value is None else value


def load_hparam_profiles(repo_root: Path) -> dict[str, Any]:
    path = repo_root / ".opencode" / "reference" / "hparam-profiles.yaml"
    if not path.is_file():
        path = PROFILE_PATH
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    profiles = data.get("profiles", {})
    if not isinstance(profiles, dict):
        raise RuntimeError(f"Invalid hparam profiles file: {path}")
    return profiles


def hydra_cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def normalize_runtime_overrides(raw_overrides: Any) -> list[str]:
    if raw_overrides in (None, "", []):
        return []
    if isinstance(raw_overrides, list):
        overrides = raw_overrides
    elif isinstance(raw_overrides, dict):
        overrides = [f"{key}={hydra_cli_value(value)}" for key, value in raw_overrides.items()]
    else:
        raise TrainingInvocationError(
            "INVALID_HP_OVERRIDES",
            "hp_overrides must be a list of Hydra override strings or a mapping",
        )
    if not all(isinstance(item, str) and item.strip() for item in overrides):
        raise TrainingInvocationError(
            "INVALID_HP_OVERRIDES",
            "hp_overrides entries must be non-empty Hydra override strings",
        )
    return [item.strip() for item in overrides]


def override_key(override: str) -> str:
    return override.split("=", 1)[0].strip().lstrip("+~")


def enforce_fixed_validation_batch(overrides: list[str]) -> list[str]:
    filtered = [
        item
        for item in overrides
        if override_key(item) not in FIXED_VALIDATION_BATCH_KEYS
    ]
    return [*filtered, *FIXED_VALIDATION_BATCH_OVERRIDES]


def cli_overrides(overrides: dict[str, Any], profile_name: str) -> list[str]:
    if not isinstance(overrides, dict):
        raise TrainingInvocationError(
            "INVALID_HP_PROFILE",
            f"Profile {profile_name!r} has invalid overrides",
        )
    return [f"{key}={value}" for key, value in overrides.items()]


def normalized_experiment_config(experiment_config: str) -> str:
    value = experiment_config.strip().removeprefix("experiment=").strip("/")
    return value.removesuffix(".yaml").removesuffix(".yml")


def reference_profile_lookup_groups(experiment_config: str) -> list[tuple[str, list[str]]]:
    normalized = normalized_experiment_config(experiment_config)
    parts = [part for part in normalized.split("/") if part]
    groups: list[tuple[str, list[str]]] = [("by_experiment", [normalized])]
    if len(parts) >= 2:
        model_name = parts[-1]
        scope_prefixes = ["/".join(parts[:idx]) for idx in range(len(parts) - 1, 0, -1)]
        groups.append(
            (
                "by_scope_model",
                [f"{scope}/{model_name}" for scope in scope_prefixes],
            )
        )
        groups.append(("by_scope", scope_prefixes))
        groups.append(("by_dataset", [parts[0]]))
    return groups


def profile_entry_overrides(entry: Any, profile_name: str, key: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise TrainingInvocationError(
            "INVALID_HP_PROFILE",
            f"Profile {profile_name!r} entry {key!r} must be a mapping",
        )
    overrides = entry.get("overrides", entry)
    if not isinstance(overrides, dict):
        raise TrainingInvocationError(
            "INVALID_HP_PROFILE",
            f"Profile {profile_name!r} entry {key!r} has invalid overrides",
        )
    return overrides


def resolved_reference_overrides(profile: dict[str, Any], experiment_config: str) -> dict[str, Any]:
    for section, keys in reference_profile_lookup_groups(experiment_config):
        mapping = profile.get(section) or {}
        if not isinstance(mapping, dict):
            raise TrainingInvocationError(
                "INVALID_HP_PROFILE",
                f"Reference profile section {section!r} must be a mapping",
        )
        for key in keys:
            if key in mapping:
                return profile_entry_overrides(mapping[key], "reference", f"{section}:{key}")

    return {}


def profile_overrides(
    repo_root: Path,
    hp_profile: str,
    experiment_config: str,
    model_paradigm: str | None = None,
    runtime_overrides: Any = None,
) -> list[str]:
    explicit_overrides = normalize_runtime_overrides(runtime_overrides)
    if explicit_overrides:
        if hp_profile in {"reference", "paper"}:
            return enforce_fixed_validation_batch(explicit_overrides)
        return explicit_overrides
    profiles = load_hparam_profiles(repo_root)
    profile = profiles.get(hp_profile)
    if not isinstance(profile, dict):
        raise TrainingInvocationError(
            "UNKNOWN_HP_PROFILE",
            f"Unknown hp_profile={hp_profile!r}; available={sorted(profiles)}",
        )
    if hp_profile == "reference":
        return enforce_fixed_validation_batch(
            cli_overrides(
                resolved_reference_overrides(profile, experiment_config),
                hp_profile,
            )
        )
    overrides = profile.get("overrides") or {}
    return enforce_fixed_validation_batch(cli_overrides(overrides, hp_profile))


def hp_profile_status(hp_profile: str, normalized_hp_overrides: list[str]) -> str:
    if hp_profile == "reference" and normalized_hp_overrides:
        return "KEYED_REFERENCE_OVERRIDES_WITH_FIXED_VALIDATION_BATCH"
    if hp_profile == "reference":
        return "FIXED_VALIDATION_BATCH_WITH_PICID_SCHEDULER_AND_PAPER_OR_IMPUTED_MODEL_HPS"
    return "PAPER_OR_DEFAULT_PROFILE_WITH_FIXED_VALIDATION_BATCH"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--input-json", required=True)
    return parser.parse_args(argv)


def ensure_repo_root(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve()


def load_latest_sanity(vault_dir: Path) -> dict[str, Any]:
    jsonl_path = vault_dir / "06-sanity-ladder-log.jsonl"
    if jsonl_path.is_file():
        records = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if records:
            return records[-1]

    md_path = vault_dir / "06-sanity-ladder-log.md"
    if md_path.is_file():
        markdown = md_path.read_text(encoding="utf-8")
        blocks = _frontmatter_blocks(markdown) or _legacy_sanity_blocks(markdown)
        if blocks:
            latest = blocks[-1]
            if latest.get("verdict") == "PROCEED":
                latest["verdict"] = PASS
            return latest
    raise RuntimeError("SANITY_LOG_UNPARSEABLE")


def load_latest_preflight(vault_dir: Path, hp_profile: str) -> dict[str, Any] | None:
    suffix = ".preflight" if hp_profile == "reference" else f".preflight.{hp_profile}"
    jsonl_path = vault_dir / f"07-training-log{suffix}.jsonl"
    if jsonl_path.is_file():
        records = [
            json.loads(line)
            for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if records:
            return records[-1]
    md_path = vault_dir / f"07-training-log{suffix}.md"
    if md_path.is_file():
        blocks = _frontmatter_blocks(md_path.read_text(encoding="utf-8"))
        if blocks:
            return blocks[-1]
    return None


def compose_resolved_config(
    repo_root: Path,
    experiment_config: str,
    hp_profile: str = "reference",
    model_paradigm: str | None = None,
    hp_overrides: Any = None,
) -> dict[str, Any]:
    with initialize_config_dir(version_base="1.3", config_dir=str(repo_root / "configs")):
        cfg = compose(
            config_name="run.yaml",
            overrides=[
                f"experiment={experiment_config}",
                "paths=agent",
                "logger=csv",
                *profile_overrides(
                    repo_root,
                    hp_profile,
                    experiment_config,
                    model_paradigm,
                    hp_overrides,
                ),
            ],
        )
    return OmegaConf.to_container(cfg, resolve=True)


def compute_resolved_config_hash(
    repo_root: Path,
    experiment_config: str,
    hp_profile: str = "reference",
    model_paradigm: str | None = None,
    hp_overrides: Any = None,
) -> str:
    resolved_cfg = compose_resolved_config(
        repo_root,
        experiment_config,
        hp_profile,
        model_paradigm,
        hp_overrides,
    )
    return hash_config(resolved_cfg)[:12]


def hardware_id(gpu_available: bool) -> str:
    if not gpu_available:
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return "cpu"


def load_batch_fit_check(vault_dir: Path, resolved_hash: str, hw_id: str) -> BatchFitCheckRecord:
    path = vault_dir / "batch_fit_check.json"
    if not path.is_file():
        raise TrainingInvocationError(
            "BATCH_FIT_CHECK_UNAVAILABLE",
            f"Missing fixed-batch fit check artifact: {path}",
        )
    records = json.loads(path.read_text(encoding="utf-8"))
    for record in reversed(records):
        if (
            record.get("resolved_config_hash") == resolved_hash
            and record.get("hardware_id") == hw_id
        ):
            status = str(record.get("status", "")).upper()
            if status not in {"OK", "PASS"}:
                raise TrainingInvocationError(
                    "BATCH_DOES_NOT_FIT",
                    str(record.get("reason", status)),
                )
            return BatchFitCheckRecord(
                resolved_config_hash=record["resolved_config_hash"],
                hardware_id=record["hardware_id"],
                status=status,
                train_batch_size=record.get("train_batch_size"),
                val_batch_size=record.get("val_batch_size"),
                test_batch_size=record.get("test_batch_size"),
                timestamp=record.get("timestamp", ""),
            )
    raise TrainingInvocationError(
        "BATCH_FIT_CHECK_UNAVAILABLE",
        f"No fixed-batch fit check matches resolved_config_hash={resolved_hash}, hardware_id={hw_id}",
    )


def read_metrics_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric_key_candidates(key: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(key, str):
        return METRIC_KEY_ALIASES.get(key, (key,))
    candidates: list[str] = []
    for item in key:
        for candidate in metric_key_candidates(item):
            if candidate not in candidates:
                candidates.append(candidate)
    return tuple(candidates)


def numeric_series(rows: list[dict[str, str]], key: str | list[str] | tuple[str, ...]) -> list[float]:
    keys = metric_key_candidates(key)
    values: list[float] = []
    for row in rows:
        raw = next((row.get(candidate) for candidate in keys if row.get(candidate) not in (None, "")), None)
        if raw in (None, ""):
            continue
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return values


def parse_expected_epochs(value: Any) -> int | None:
    if value in (None, "", "NOT_SPECIFIED"):
        return None
    if isinstance(value, str) and value.strip().upper() == "NOT_SPECIFIED":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def trainer_default_max_epochs(resolved_cfg: dict[str, Any]) -> int | None:
    trainer_cfg = resolved_cfg.get("trainer", {})
    if not isinstance(trainer_cfg, dict):
        return None
    return parse_expected_epochs(trainer_cfg.get("max_epochs"))


def _path_from_hydra_value(value: str, repo_root: Path) -> Path | None:
    cleaned = value.strip().strip("'\"")
    if not cleaned or "${" in cleaned:
        return None
    path = Path(cleaned).expanduser()
    return path if path.is_absolute() else (repo_root / path).resolve()


def explicit_output_dir_from_command(command: list[str], repo_root: Path) -> Path | None:
    for arg in reversed(command):
        if "=" not in arg:
            continue
        raw_key, raw_value = arg.split("=", 1)
        key = raw_key.lstrip("+")
        if key in {"paths.output_dir", "hydra.run.dir"}:
            return _path_from_hydra_value(raw_value, repo_root)
    return None


def looks_like_run_root(path: Path) -> bool:
    return any((path / marker).exists() for marker in RUN_ROOT_MARKERS)


def run_root_activity_time(path: Path) -> datetime:
    timestamps = [path.stat().st_mtime]
    for marker in RUN_ROOT_MARKERS:
        marker_path = path / marker
        if marker_path.exists():
            timestamps.append(marker_path.stat().st_mtime)
    return datetime.fromtimestamp(max(timestamps), UTC)


def output_dir_from_resolved_config(run_dir: Path) -> Path | None:
    cfg = load_resolved_run_config(run_dir)
    paths_cfg = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    if not isinstance(paths_cfg, dict):
        return None
    value = paths_cfg.get("output_dir")
    if not isinstance(value, str):
        return None
    return _path_from_hydra_value(value, run_dir)


def latest_run_dir(
    repo_root: Path,
    started_after: datetime,
    explicit_output_dir: Path | None = None,
) -> Path | None:
    if explicit_output_dir is not None:
        return explicit_output_dir if explicit_output_dir.is_dir() else None

    artifacts_dir = repo_root / "artifacts"
    if not artifacts_dir.exists():
        return None
    candidates_by_path: dict[Path, Path] = {}
    for marker in RUN_ROOT_MARKERS:
        for marker_path in artifacts_dir.rglob(marker):
            run_root = marker_path.parent
            if marker_path.name == ".hydra" and marker_path.is_dir():
                run_root = marker_path.parent
            if run_root.is_dir() and looks_like_run_root(run_root):
                resolved_output_dir = output_dir_from_resolved_config(run_root)
                if resolved_output_dir and resolved_output_dir.is_dir():
                    run_root = resolved_output_dir
                candidates_by_path[run_root] = run_root
    candidates = [
        path
        for path in candidates_by_path.values()
        if run_root_activity_time(path) >= started_after
    ]
    candidates.sort(key=lambda path: run_root_activity_time(path))
    return candidates[-1] if candidates else None


def metrics_version_key(path: Path) -> tuple[int, float, str]:
    match = re.fullmatch(r"version_(\d+)", path.parent.name)
    version = int(match.group(1)) if match else -1
    return (version, path.stat().st_mtime, str(path))


def find_metrics_csv(run_dir: Path | None) -> Path | None:
    if not run_dir:
        return None
    candidates = [
        path
        for pattern in (
            "csv_logs/version_*/metrics.csv",
            "logs/csv_logs/version_*/metrics.csv",
        )
        for path in run_dir.glob(pattern)
        if path.is_file()
    ]
    candidates.sort(key=metrics_version_key)
    return candidates[-1] if candidates else None


def find_framework_log(run_dir: Path | None) -> Path | None:
    if not run_dir:
        return None
    candidates = sorted(run_dir.glob("*.log"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def load_resolved_run_config(run_dir: Path | None) -> dict[str, Any]:
    if not run_dir:
        return {}
    path = run_dir / "config_resolved.yaml"
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _configured_checkpoint_dirs(run_dir: Path) -> list[Path]:
    cfg = load_resolved_run_config(run_dir)
    dirs: list[Path] = []
    callbacks_cfg = cfg.get("callbacks", {}) if isinstance(cfg, dict) else {}
    if isinstance(callbacks_cfg, dict):
        checkpoint_cfg = callbacks_cfg.get("model_checkpoint", {})
        if isinstance(checkpoint_cfg, dict) and isinstance(checkpoint_cfg.get("dirpath"), str):
            configured = _path_from_hydra_value(checkpoint_cfg["dirpath"], run_dir)
            if configured:
                dirs.append(configured)
    paths_cfg = cfg.get("paths", {}) if isinstance(cfg, dict) else {}
    if isinstance(paths_cfg, dict) and isinstance(paths_cfg.get("ckpt_dir"), str):
        configured = _path_from_hydra_value(paths_cfg["ckpt_dir"], run_dir)
        if configured:
            dirs.append(configured)
    dirs.extend([run_dir / "checkpoints", run_dir / "checkpoints" / "checkpoints"])
    deduped: list[Path] = []
    for path in dirs:
        if path not in deduped:
            deduped.append(path)
    return deduped


def find_checkpoint_files(run_dir: Path | None) -> list[Path]:
    if not run_dir or not run_dir.is_dir():
        return []
    checkpoints: set[Path] = set()
    for directory in _configured_checkpoint_dirs(run_dir):
        if directory.is_dir():
            checkpoints.update(path for path in directory.rglob("*.ckpt") if path.is_file())
    checkpoints.update(path for path in run_dir.rglob("*.ckpt") if path.is_file())
    return sorted(checkpoints, key=lambda path: (path.stat().st_mtime, str(path)))


def checkpoint_created(run_dir: Path | None) -> bool:
    return bool(find_checkpoint_files(run_dir))


def test_stage_completed(run_dir: Path | None, test_metrics: dict[str, float]) -> bool:
    if test_metrics:
        return True
    if not run_dir:
        return False
    metrics_csv = find_metrics_csv(run_dir)
    if metrics_csv and extract_test_metrics(read_metrics_csv(metrics_csv)):
        return True
    eval_test_dir = run_dir / "eval_details" / "test"
    if eval_test_dir.exists():
        return True
    log_text = safe_read_text(find_framework_log(run_dir), limit_chars=MAX_TEXT_EXCERPT_CHARS).lower()
    return any(marker in log_text for marker in TEST_LOG_MARKERS)


def preflight_evidence(
    run_dir: Path | None,
    rows: list[dict[str, str]],
    test_metrics: dict[str, float],
    returncode: int | None,
) -> dict[str, bool]:
    if run_dir and not rows:
        metrics_csv = find_metrics_csv(run_dir)
        rows = read_metrics_csv(metrics_csv) if metrics_csv else []
    effective_test_metrics = test_metrics or extract_test_metrics(rows)
    return {
        "returncode_zero": returncode == 0,
        "train_stage_completed": bool(numeric_series(rows, TRAIN_LOSS_KEYS)),
        "val_stage_completed": bool(numeric_series(rows, VAL_LOSS_KEYS)),
        "test_stage_completed": test_stage_completed(run_dir, effective_test_metrics),
        "checkpoint_created": checkpoint_created(run_dir),
    }


def preflight_gate_passed(evidence: dict[str, bool]) -> bool:
    return all(
        evidence.get(key, False)
        for key in (
            "returncode_zero",
            "train_stage_completed",
            "val_stage_completed",
            "test_stage_completed",
            "checkpoint_created",
        )
    )


def safe_read_text(path: Path | None, limit_chars: int | None = None) -> str:
    if not path or not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if limit_chars and len(text) > limit_chars:
        return text[-limit_chars:]
    return text


def tail_file_text(handle: Any, limit_chars: int = MAX_TEXT_EXCERPT_CHARS) -> str:
    handle.seek(0)
    text = handle.read()
    return text[-limit_chars:] if len(text) > limit_chars else text


def compact_event(events: list[dict[str, Any]], event: dict[str, Any]) -> None:
    events.append(event)
    if len(events) > MAX_MONITOR_EVENTS:
        del events[1 : 1 + (len(events) - MAX_MONITOR_EVENTS)]


def has_nonfinite(values: list[float]) -> bool:
    return any(not math.isfinite(value) for value in values)


def exploding_loss(values: list[float]) -> bool:
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) < 3:
        return False
    baseline = min(abs(value) for value in finite[:-1] if value != 0) if any(value != 0 for value in finite[:-1]) else 0
    if baseline <= 0:
        return False
    return finite[-1] > baseline * 100 and finite[-1] > finite[-2] * 2


def plateau_detected(values: list[float], patience_epochs: int) -> bool:
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) < max(DEFAULT_PLATEAU_MIN_EPOCHS, patience_epochs):
        return False
    window = finite[-patience_epochs:]
    first = window[0]
    best = min(window)
    denom = max(abs(first), 1e-12)
    rel_improvement = (first - best) / denom
    return rel_improvement >= 0 and rel_improvement < PLATEAU_REL_IMPROVEMENT


def extract_test_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    test_metrics: dict[str, float] = {}
    for row in reversed(rows):
        for key, value in row.items():
            if key and key.startswith("test") and value not in (None, ""):
                try:
                    test_metrics[key] = float(value)
                except ValueError:
                    continue
        if test_metrics:
            break
    return test_metrics


def strong_live_dataset_failure(text: str) -> str | None:
    lowered = text.lower()
    if not any(
        marker in lowered
        for marker in (
            "filenotfounderror",
            "no such file or directory",
            "paths.data_dir",
            "phmd",
            "download",
            "memoryerror",
        )
    ):
        return None
    return classify_dataset_failure(text)


def classify_status(returncode: int, stdout: str, stderr: str) -> str:
    combined = "\n".join(part for part in (stdout, stderr) if part)
    if returncode == 0:
        return "SUCCESS"
    dataset_status = classify_dataset_failure(combined)
    if dataset_status:
        return dataset_status
    return "CRASHED"


def early_stopping_epoch(log_text: str) -> int | None:
    match = re.search(r"Epoch (\d+): .*early stopping", log_text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def plot_loss_curves(repo_root: Path, metrics_csv: Path, output_path: Path, early_epoch: int | None) -> tuple[bool, str]:
    command = [
        "uv",
        "run",
        "python",
        str(repo_root / ".opencode" / "tools" / "plot_training.py"),
        "--metrics_csv",
        str(metrics_csv),
        "--output_path",
        str(output_path),
        "--title",
        "Training Loss Curves",
    ]
    if early_epoch is not None:
        command.extend(["--early_stopping_epoch", str(early_epoch)])
    proc = subprocess.run(command, cwd=repo_root, capture_output=True, text=True)
    output = (proc.stdout or "") + (proc.stderr or "")
    return "PLOT_OK:" in output, output.strip()


def build_command(
    repo_root: Path,
    experiment_config: str,
    hp_profile: str,
    hp_overrides: Any,
    run_mode: str,
    training_stage: str,
    gpu_available: bool,
    model_paradigm: str,
    max_epochs: int | None,
) -> list[str]:
    overrides = profile_overrides(
        repo_root,
        hp_profile,
        experiment_config,
        model_paradigm,
        hp_overrides,
    )
    if model_paradigm != "fit_predict":
        if training_stage == "preflight_1epoch" or run_mode == "quick":
            overrides.extend(
                [
                    "trainer.max_epochs=1",
                    "+trainer.limit_train_batches=1",
                    "+trainer.limit_val_batches=1",
                    "+trainer.limit_test_batches=1",
                ]
            )
        else:
            if max_epochs is not None:
                overrides.append(f"trainer.max_epochs={max_epochs}")
    if not gpu_available:
        overrides.extend(["trainer.accelerator=cpu", "trainer.devices=1"])
    return [
        "uv",
        "run",
        "python",
        "-m",
        "picid.run",
        f"experiment={experiment_config}",
        *base_overrides(["paths=agent", "seed=42", *overrides]),
    ]


def profile_suffix(frontmatter: dict[str, Any]) -> str:
    parts: list[str] = []
    if (
        str(frontmatter.get("training_stage") or "") == "preflight_1epoch"
        and str(frontmatter.get("run_mode") or "full") == "full"
    ):
        parts.append("preflight")
    hp_profile = str(frontmatter.get("hp_profile") or "reference")
    if hp_profile != "reference":
        parts.append(hp_profile)
    return "" if not parts else "." + ".".join(parts)


def write_training_log(vault_dir: Path, frontmatter: dict[str, Any], body: str) -> Path:
    path = vault_dir / f"07-training-log{profile_suffix(frontmatter)}.md"
    content = ["---", yaml.safe_dump(frontmatter, sort_keys=False).strip(), "---", "", body.strip(), ""]
    path.write_text("\n".join(content), encoding="utf-8")
    return path


def append_training_sidecar(vault_dir: Path, payload: dict[str, Any]) -> Path:
    path = vault_dir / f"07-training-log{profile_suffix(payload)}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")
    return path


def base_frontmatter(
    *,
    experiment_config: str,
    dataset_candidate: str | None,
    run_mode: str,
    training_stage: str,
    max_epochs: int | None,
    hw_id: str,
    epoch_budget_rationale: str,
) -> dict[str, Any]:
    return {
        "status": "FAILED",
        "experiment_config": experiment_config,
        "dataset_candidate": dataset_candidate,
        "hp_profile": None,
        "hp_profile_status": None,
        "is_primary_comparison": None,
        "run_mode": run_mode,
        "training_stage": training_stage,
        "max_epochs_ceiling": max_epochs,
        "epochs_run": 0,
        "early_stopping_triggered": False,
        "early_stopping_epoch": None,
        "batch_size_paper": None,
        "batch_size_chosen": None,
        "batch_probe_source": None,
        "batch_fit_check_source": None,
        "batch_size_configured": None,
        "hp_overrides": None,
        "hp_reference_source": None,
        "lr_paper": None,
        "lr_scaled": None,
        "lr_scaling_rule": "none",
        "optimizer_family": "other",
        "resolved_config_hash": None,
        "hardware_id": hw_id,
        "epoch_budget_rationale": epoch_budget_rationale,
        "monitor_status": "not_started",
        "stop_reason": None,
        "monitor_events": [],
        "monitor_probe_summary": {},
        "checkpoint_created": False,
        "test_stage_completed": False,
        "output_dir": None,
        "metrics_csv": None,
        "loss_curve_plot": None,
        "vault_loss_curve_plot": None,
        "traceback_excerpt": None,
        "timestamp": utc_now(),
    }


def failure_body(
    *,
    frontmatter: dict[str, Any],
    reason_code: str,
    detail: str,
    command: list[str] | None = None,
    run_dir: Path | None = None,
    metrics_csv: Path | None = None,
) -> str:
    body_lines = [
        "# Training Run Log",
        "",
        f"**Timestamp**: {frontmatter['timestamp']}",
        f"**Experiment config**: {frontmatter['experiment_config']}",
        f"**HP profile**: {frontmatter.get('hp_profile')}",
        f"**HP profile status**: {frontmatter.get('hp_profile_status')}",
        f"**HP reference source**: {frontmatter.get('hp_reference_source')}",
        f"**Run mode**: {frontmatter['run_mode']}",
        f"**Dataset candidate**: {frontmatter['dataset_candidate']}",
        f"**Status**: {frontmatter['status']}",
        f"**Reason code**: {reason_code}",
        f"**Detail**: {detail}",
    ]
    if command:
        body_lines.append(f"**Command**: `{' '.join(command)}`")
    body_lines.extend(
        [
            f"**Output directory**: `{run_dir}`" if run_dir else "**Output directory**: not found",
            f"**Metrics CSV**: `{metrics_csv}`" if metrics_csv else "**Metrics CSV**: not found",
        ]
    )
    if frontmatter.get("traceback_excerpt"):
        body_lines.extend(
            [
                "",
                "## Traceback Excerpt",
                "```text",
                str(frontmatter["traceback_excerpt"]),
                "```",
            ]
        )
    return "\n".join(body_lines)


def terminate_process(proc: subprocess.Popen[Any], reason: str, grace_sec: float = 120.0) -> str:
    if proc.poll() is not None:
        return "already_exited"
    proc.terminate()
    try:
        proc.wait(timeout=grace_sec)
        return "terminated"
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=20)
        return "killed"


def run_monitored_command(
    *,
    repo_root: Path,
    command: list[str],
    training_stage: str,
    model_paradigm: str,
    monitor_interval_sec: float,
    plateau_patience_epochs: int,
) -> dict[str, Any]:
    started_at = datetime.now(UTC)
    events: list[dict[str, Any]] = [
        {
            "timestamp": utc_now(),
            "event": "process_started",
            "training_stage": training_stage,
        }
    ]
    stop_reason: str | None = None
    monitor_status = "running"
    run_dir: Path | None = None
    metrics_csv: Path | None = None
    rows: list[dict[str, str]] = []
    output_excerpt = ""
    log_excerpt = ""
    returncode: int | None = None
    probe_count = 0
    explicit_output_dir = explicit_output_dir_from_command(command, repo_root)
    last_probe: dict[str, Any] = {
        "epochs_seen": 0,
        "train_loss": None,
        "val_loss": None,
        "checkpoint_created": False,
    }

    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout_file, tempfile.TemporaryFile(
        mode="w+", encoding="utf-8"
    ) as stderr_file:
        proc = subprocess.Popen(
            command,
            cwd=repo_root,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
        )
        last_epoch_count = -1
        while True:
            returncode = proc.poll()
            current_run_dir = latest_run_dir(repo_root, started_at, explicit_output_dir)
            if current_run_dir and current_run_dir != run_dir:
                run_dir = current_run_dir
                metrics_csv = find_metrics_csv(run_dir)
                compact_event(
                    events,
                    {
                        "timestamp": utc_now(),
                        "event": "output_dir_detected",
                        "output_dir": str(run_dir),
                    }
                )
            if run_dir:
                metrics_csv = find_metrics_csv(run_dir)
                if metrics_csv and metrics_csv.is_file():
                    rows = read_metrics_csv(metrics_csv)
                    train_losses = numeric_series(rows, TRAIN_LOSS_KEYS)
                    val_losses = numeric_series(rows, VAL_LOSS_KEYS)
                    epoch_count = len(train_losses)
                    probe_count += 1
                    last_probe = {
                        "epochs_seen": epoch_count,
                        "train_loss": train_losses[-1] if train_losses else None,
                        "val_loss": val_losses[-1] if val_losses else None,
                        "checkpoint_created": checkpoint_created(run_dir),
                    }
                    if epoch_count != last_epoch_count:
                        last_epoch_count = epoch_count
                        compact_event(
                            events,
                            {
                                "timestamp": utc_now(),
                                "event": "epoch_probe",
                                "epochs_seen": epoch_count,
                                "train_loss": train_losses[-1] if train_losses else None,
                                "val_loss": val_losses[-1] if val_losses else None,
                            }
                        )
                    combined_losses = train_losses + val_losses
                    if has_nonfinite(combined_losses):
                        stop_reason = "nonfinite_loss"
                        monitor_status = "failed"
                    elif exploding_loss(train_losses) or exploding_loss(val_losses):
                        stop_reason = "exploding_loss"
                        monitor_status = "failed"
                    elif (
                        training_stage == "full"
                        and model_paradigm != "fit_predict"
                        and returncode is None
                        and (
                            plateau_detected(val_losses, plateau_patience_epochs)
                            or plateau_detected(train_losses, plateau_patience_epochs)
                        )
                    ):
                        stop_reason = "plateau_graceful_stop"
                        monitor_status = "stopped"
                    if stop_reason:
                        action = terminate_process(proc, stop_reason)
                        returncode = proc.returncode
                        compact_event(
                            events,
                            {
                                "timestamp": utc_now(),
                                "event": "monitor_stop",
                                "reason": stop_reason,
                                "action": action,
                            }
                        )
                        break

                log_tail = safe_read_text(find_framework_log(run_dir), limit_chars=MAX_TEXT_EXCERPT_CHARS)
                dataset_status = strong_live_dataset_failure(log_tail)
                if dataset_status:
                    stop_reason = dataset_status
                    monitor_status = "failed"
                    action = terminate_process(proc, dataset_status)
                    returncode = proc.returncode
                    compact_event(
                        events,
                        {
                            "timestamp": utc_now(),
                            "event": "monitor_stop",
                            "reason": dataset_status,
                            "action": action,
                        }
                    )
                    break
                lowered_log_tail = log_tail.lower()
                if "traceback (most recent call last)" in lowered_log_tail:
                    stop_reason = "traceback_detected"
                    monitor_status = "failed"
                    action = terminate_process(proc, stop_reason)
                    returncode = proc.returncode
                    compact_event(
                        events,
                        {
                            "timestamp": utc_now(),
                            "event": "monitor_stop",
                            "reason": stop_reason,
                            "action": action,
                        }
                    )
                    break
                if "cuda out of memory" in lowered_log_tail or "outofmemoryerror" in lowered_log_tail:
                    stop_reason = "oom"
                    monitor_status = "failed"
                    action = terminate_process(proc, stop_reason)
                    returncode = proc.returncode
                    compact_event(
                        events,
                        {
                            "timestamp": utc_now(),
                            "event": "monitor_stop",
                            "reason": stop_reason,
                            "action": action,
                        }
                    )
                    break

            if returncode is not None:
                monitor_status = "completed" if returncode == 0 else "exited"
                break
            time.sleep(max(0.1, monitor_interval_sec))

        output_excerpt = "\n".join(
            part for part in (tail_file_text(stdout_file), tail_file_text(stderr_file)) if part
        )[-MAX_TEXT_EXCERPT_CHARS:]

    if run_dir:
        metrics_csv = find_metrics_csv(run_dir)
        rows = read_metrics_csv(metrics_csv) if metrics_csv else rows
        log_excerpt = safe_read_text(find_framework_log(run_dir), limit_chars=MAX_TEXT_EXCERPT_CHARS)
    compact_event(
        events,
        {
            "timestamp": utc_now(),
            "event": "process_finished",
            "returncode": returncode,
            "monitor_status": monitor_status,
            "stop_reason": stop_reason,
        }
    )
    return {
        "returncode": returncode,
        "output_excerpt": output_excerpt,
        "log_excerpt": log_excerpt,
        "started_at": started_at,
        "run_dir": run_dir,
        "metrics_csv": metrics_csv,
        "rows": rows,
        "monitor_status": monitor_status,
        "stop_reason": stop_reason,
        "monitor_events": events,
        "monitor_probe_summary": {
            "probe_count": probe_count,
            "last_probe": last_probe,
        },
    }


def run_training(payload: dict[str, Any]) -> dict[str, Any]:
    repo_root = ensure_repo_root(payload_arg(payload, "repo_root", "."))
    vault_dir = Path(payload["vault_dir"]).expanduser().resolve()
    experiment_config = payload["experiment_config"]
    model_paradigm = payload_arg(payload, "model_paradigm", "feedforward")
    gpu_available = bool(payload_arg(payload, "gpu_available", False))
    run_mode = payload_arg(payload, "run_mode", "full")
    training_stage = str(
        payload_arg(
            payload,
            "training_stage",
            "preflight_1epoch" if run_mode == "quick" else "full",
        )
    )
    hp_profile = str(payload_arg(payload, "hp_profile", "reference")).strip() or "reference"
    hp_overrides = payload_arg(payload, "hp_overrides")
    hp_reference_source = payload_arg(payload, "hp_reference_source")
    dataset_candidate = payload_arg(payload, "dataset_candidate")
    expected_epochs = payload_arg(payload, "expected_epochs")
    epoch_budget_rationale = str(payload_arg(payload, "epoch_budget_rationale", "")).strip()
    monitor_interval_sec = float(
        payload_arg(
            payload,
            "monitor_interval_sec",
            5.0 if training_stage == "preflight_1epoch" else DEFAULT_MONITOR_INTERVAL_SEC,
        )
    )
    plateau_patience_epochs = int(
        payload_arg(payload, "plateau_patience_epochs", DEFAULT_PLATEAU_PATIENCE_EPOCHS)
    )
    expected_epochs_parsed = parse_expected_epochs(expected_epochs)
    max_epochs = 1 if training_stage == "preflight_1epoch" else expected_epochs_parsed
    hw_id = hardware_id(gpu_available)
    frontmatter = base_frontmatter(
        experiment_config=experiment_config,
        dataset_candidate=dataset_candidate,
        run_mode=run_mode,
        training_stage=training_stage,
        max_epochs=max_epochs,
        hw_id=hw_id,
        epoch_budget_rationale=epoch_budget_rationale,
    )
    frontmatter["hp_profile"] = hp_profile
    frontmatter["is_primary_comparison"] = hp_profile == "reference"
    frontmatter["hp_reference_source"] = hp_reference_source
    normalized_hp_overrides: list[str] = []
    command: list[str] | None = None
    run_dir: Path | None = None
    metrics_csv: Path | None = None
    test_metrics: dict[str, float] = {}
    returncode: int | None = None

    try:
        if training_stage not in {"preflight_1epoch", "full"}:
            raise TrainingInvocationError(
                "INVALID_TRAINING_STAGE",
                "training_stage must be preflight_1epoch or full",
            )
        normalized_hp_overrides = normalize_runtime_overrides(hp_overrides)
        frontmatter["hp_overrides"] = normalized_hp_overrides
        frontmatter["hp_profile_status"] = hp_profile_status(
            hp_profile,
            normalized_hp_overrides,
        )
        if len(epoch_budget_rationale) < 50:
            raise TrainingInvocationError(
                "MISSING_EPOCH_BUDGET_RATIONALE",
                "epoch_budget_rationale must be at least 50 characters",
            )

        sanity = load_latest_sanity(vault_dir)
        sanity_verdict = str(sanity.get("verdict", "")).strip().upper()
        if sanity_verdict not in ACCEPTED_SANITY_VERDICTS:
            raise TrainingInvocationError(
                "SANITY_NOT_PASSED",
                f"latest sanity verdict is {sanity.get('verdict')}",
            )
        if run_mode == "full" and training_stage == "full":
            preflight = load_latest_preflight(vault_dir, hp_profile)
            if not preflight or str(preflight.get("status", "")).upper() != "SUCCESS":
                raise TrainingInvocationError(
                    "PREFLIGHT_1EPOCH_NOT_PASSED",
                    "Full training requires a successful 1-epoch train/val/test/checkpoint preflight",
                )

        resolved_hash = compute_resolved_config_hash(
            repo_root, experiment_config, hp_profile, model_paradigm, hp_overrides
        )
        frontmatter["resolved_config_hash"] = resolved_hash
        fit_check = load_batch_fit_check(vault_dir, resolved_hash, hw_id)
        resolved_cfg = compose_resolved_config(
            repo_root, experiment_config, hp_profile, model_paradigm, hp_overrides
        )
        if training_stage == "full" and max_epochs is None:
            max_epochs = trainer_default_max_epochs(resolved_cfg)
            frontmatter["max_epochs_ceiling"] = max_epochs

        command = build_command(
            repo_root=repo_root,
            experiment_config=experiment_config,
            hp_profile=hp_profile,
            hp_overrides=hp_overrides,
            run_mode=run_mode,
            training_stage=training_stage,
            gpu_available=gpu_available,
            model_paradigm=model_paradigm,
            max_epochs=max_epochs,
        )
        monitor_result = run_monitored_command(
            repo_root=repo_root,
            command=command,
            training_stage=training_stage,
            model_paradigm=model_paradigm,
            monitor_interval_sec=monitor_interval_sec,
            plateau_patience_epochs=plateau_patience_epochs,
        )
        returncode = monitor_result["returncode"]
        run_dir = monitor_result["run_dir"]
        metrics_csv = monitor_result["metrics_csv"]
        rows = monitor_result["rows"]
        rows = read_metrics_csv(metrics_csv) if metrics_csv else []
        train_losses = numeric_series(rows, TRAIN_LOSS_KEYS)
        val_losses = numeric_series(rows, VAL_LOSS_KEYS)
        test_metrics = extract_test_metrics(rows)
        datamodule_cfg = (
            resolved_cfg.get("datamodule", {})
            if isinstance(resolved_cfg.get("datamodule"), dict)
            else {}
        )
        batch_size_configured = datamodule_cfg.get("train_batch_size")
        optimization_cfg = (
            resolved_cfg.get("optimization", {})
            if isinstance(resolved_cfg.get("optimization"), dict)
            else {}
        )
        lr_paper = optimization_cfg.get("lr")

        framework_log_text = monitor_result.get("log_excerpt") or safe_read_text(
            find_framework_log(run_dir), limit_chars=MAX_TEXT_EXCERPT_CHARS
        )
        combined_output = "\n".join(
            part
            for part in (
                monitor_result.get("output_excerpt") or "",
                framework_log_text,
            )
            if part
        )
        early_epoch = early_stopping_epoch(combined_output)
        plot_path = run_dir / "plots" / "loss_curves.png" if run_dir else None
        vault_plot_path = vault_dir / "plots" / "loss_curves.png" if run_dir else None
        plot_ok = False
        plot_message = ""
        vault_plot_ok = False
        vault_plot_message = ""
        if metrics_csv and metrics_csv.is_file() and plot_path and model_paradigm != "fit_predict":
            plot_ok, plot_message = plot_loss_curves(repo_root, metrics_csv, plot_path, early_epoch)
            if vault_plot_path:
                vault_plot_ok, vault_plot_message = plot_loss_curves(
                    repo_root, metrics_csv, vault_plot_path, early_epoch
                )

        stop_reason = monitor_result.get("stop_reason")
        monitor_status = monitor_result.get("monitor_status")
        dataset_status = classify_dataset_failure(combined_output)
        if stop_reason in {"DATASET_UNAVAILABLE", "DATASET_EXECUTION_FAILED"}:
            status = str(stop_reason)
        elif dataset_status and returncode not in (0, None):
            status = dataset_status
        elif stop_reason == "plateau_graceful_stop":
            status = "PARTIAL"
        elif stop_reason in {"nonfinite_loss", "exploding_loss", "traceback_detected"}:
            status = "FAILED"
        elif stop_reason == "oom":
            status = "CRASHED"
        else:
            status = classify_status(int(returncode or 0), monitor_result.get("output_excerpt") or "", framework_log_text)
        gate_evidence = preflight_evidence(run_dir, rows, test_metrics, returncode)
        if (
            status == "SUCCESS"
            and training_stage == "preflight_1epoch"
            and not preflight_gate_passed(gate_evidence)
        ):
            status = "FAILED"
            stop_reason = stop_reason or "preflight_missing_required_evidence"

        frontmatter.update(
            {
                "status": status,
                "hp_profile": hp_profile,
                "is_primary_comparison": hp_profile == "reference",
                "hp_overrides": normalized_hp_overrides,
                "hp_reference_source": hp_reference_source,
                "epochs_run": len(train_losses),
                "early_stopping_triggered": early_epoch is not None,
                "early_stopping_epoch": early_epoch,
                "batch_size_paper": None,
                "batch_size_chosen": batch_size_configured,
                "batch_probe_source": "not_used",
                "batch_fit_check_source": "reused",
                "batch_size_configured": batch_size_configured,
                "lr_paper": lr_paper,
                "lr_scaled": lr_paper,
                "lr_scaling_rule": "none",
                "optimizer_family": "other",
                "monitor_status": monitor_status,
                "stop_reason": stop_reason,
                "monitor_events": monitor_result.get("monitor_events", []),
                "monitor_probe_summary": monitor_result.get("monitor_probe_summary", {}),
                "preflight_evidence": gate_evidence if training_stage == "preflight_1epoch" else {},
                "checkpoint_created": gate_evidence["checkpoint_created"],
                "test_stage_completed": gate_evidence["test_stage_completed"],
                "output_dir": str(run_dir) if run_dir else None,
                "metrics_csv": str(metrics_csv) if metrics_csv else None,
                "loss_curve_plot": str(plot_path) if plot_ok and plot_path else None,
                "vault_loss_curve_plot": str(vault_plot_path) if vault_plot_ok and vault_plot_path else None,
                "traceback_excerpt": traceback_excerpt(combined_output) or None,
                "timestamp": utc_now(),
            }
        )

        body_lines = [
            "# Training Run Log",
            "",
            f"**Timestamp**: {frontmatter['timestamp']}",
            f"**Experiment config**: {experiment_config}",
            f"**HP profile**: {hp_profile}",
            f"**HP profile status**: {frontmatter['hp_profile_status']}",
            f"**HP reference source**: {hp_reference_source}",
            f"**Run mode**: {run_mode}",
            f"**Training stage**: {training_stage}",
            f"**Command**: `{' '.join(command)}`",
            f"**Output directory**: `{run_dir}`" if run_dir else "**Output directory**: not found",
            "",
            "## Run Status",
            f"- **Status category**: {status}",
            f"- **Dataset candidate**: {dataset_candidate}",
            f"- **Monitor status**: {monitor_status}",
            f"- **Stop reason**: {stop_reason or 'none'}",
            f"- **Monitor probes**: {monitor_result.get('monitor_probe_summary', {}).get('probe_count', 0)}",
            f"- **Early stopping**: {'triggered' if early_epoch is not None else 'not triggered'}",
            f"- **Batch fit check**: {fit_check.status} ({fit_check.train_batch_size}/{fit_check.val_batch_size}/{fit_check.test_batch_size})",
            f"- **Checkpoint created**: {frontmatter['checkpoint_created']}",
            f"- **Test stage completed**: {frontmatter['test_stage_completed']}",
            "",
            "## Loss Summary",
            f"- **Initial train loss**: {train_losses[0] if train_losses else 'n/a'}",
            f"- **Final train loss**: {train_losses[-1] if train_losses else 'n/a'}",
            f"- **Final val loss**: {val_losses[-1] if val_losses else 'n/a'}",
            "",
            "## Final Test Metrics",
        ]
        if test_metrics:
            for key, value in sorted(test_metrics.items()):
                body_lines.append(f"- **{key}**: {value}")
        else:
            body_lines.append("- No test metrics found.")
        body_lines.extend(
            [
                "",
                "## Configured Optimization Profile",
                f"- **Profile status**: {frontmatter['hp_profile_status']}",
                f"- **HP overrides**: `{normalized_hp_overrides}`",
                f"- **Train batch size**: {datamodule_cfg.get('train_batch_size')}",
                f"- **Val batch size**: {datamodule_cfg.get('val_batch_size')}",
                f"- **Test batch size**: {datamodule_cfg.get('test_batch_size')}",
                f"- **Learning rate**: {lr_paper}",
                f"- **Scheduler**: {optimization_cfg.get('scheduler', {}).get('_target_', 'none') if isinstance(optimization_cfg.get('scheduler'), dict) else 'none'}",
                "- **Runtime tuning**: none",
                "",
                "## Monitor Probes",
                f"- **Summary**: `{monitor_result.get('monitor_probe_summary', {})}`",
                "- **Event log**: compact lifecycle/stop events only; full logs stay in the run directory.",
            ]
        )
        for event in monitor_result.get("monitor_events", []):
            body_lines.append(f"- `{event}`")
        body_lines.extend(
            [
                "",
                "## Artifacts",
                f"- **Metrics CSV**: `{metrics_csv}`" if metrics_csv else "- **Metrics CSV**: not found",
                f"- **Loss curve plot (run directory)**: `{plot_path}` ({'OK' if plot_ok else plot_message or 'not generated'})"
                if plot_path
                else "- **Loss curve plot (run directory)**: not generated",
                f"- **Loss curve plot (vault)**: `{vault_plot_path}` ({'OK' if vault_plot_ok else vault_plot_message or 'not generated'})"
                if vault_plot_path
                else "- **Loss curve plot (vault)**: not generated",
                f"- **Resolved config**: `{run_dir / 'config_resolved.yaml'}`" if run_dir else "- **Resolved config**: not found",
                f"- **Reproduction guide**: `{run_dir / 'REPRODUCE.md'}`" if run_dir else "- **Reproduction guide**: not found",
            ]
        )
        body = "\n".join(body_lines)
    except TrainingInvocationError as exc:
        frontmatter.update(
            {
                "status": "FAILED",
                "traceback_excerpt": exc.detail,
                "timestamp": utc_now(),
            }
        )
        body = failure_body(
            frontmatter=frontmatter,
            reason_code=exc.reason_code,
            detail=exc.detail,
            command=command,
            run_dir=run_dir,
            metrics_csv=metrics_csv,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        frontmatter.update(
            {
                "status": "CRASHED",
                "traceback_excerpt": traceback_excerpt(tb) or str(exc),
                "timestamp": utc_now(),
            }
        )
        body = failure_body(
            frontmatter=frontmatter,
            reason_code=type(exc).__name__,
            detail=str(exc),
            command=command,
            run_dir=run_dir,
            metrics_csv=metrics_csv,
        )

    log_path = write_training_log(vault_dir, frontmatter, body)
    jsonl_path = append_training_sidecar(vault_dir, frontmatter)

    return {
        **frontmatter,
        "output_dir": str(run_dir) if run_dir else None,
        "metrics_csv": str(metrics_csv) if metrics_csv else None,
        "test_metrics": test_metrics,
        "markdown_path": str(log_path),
        "jsonl_path": str(jsonl_path),
        "returncode": returncode,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = json.loads(args.input_json)
    if args.command != "run":
        result = {"status": "ERROR", "reason": f"Unknown command: {args.command}"}
        print(RESULT_MARKER + json.dumps(result, sort_keys=True))
        return 1
    result = run_training(payload)
    print(RESULT_MARKER + json.dumps(result, sort_keys=True))
    return 0 if result.get("status") in SUCCESS_STATUSES else 1


if __name__ == "__main__":
    raise SystemExit(main())
