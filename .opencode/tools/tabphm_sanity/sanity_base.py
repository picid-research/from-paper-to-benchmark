from __future__ import annotations

import csv
import math
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


PASS = "PASS"
FAIL = "FAIL"
SKIPPED = "SKIPPED"
DATASET_UNAVAILABLE = "DATASET_UNAVAILABLE"
DATASET_EXECUTION_FAILED = "DATASET_EXECUTION_FAILED"
PRECHECK_TIMEOUT = "PRECHECK_TIMEOUT"
TOOL_INVOCATION_FAILURE = "TOOL_INVOCATION_FAILURE"
PROCEED = PASS
INVESTIGATE = "INVESTIGATE"
WARN_CONTINUE = "WARN_CONTINUE"
BLOCK = "BLOCK"

DEFAULT_TIMEOUT_SEC = 600


class CheckTimeoutError(Exception):
    """Raised when a sanity check exceeds its wall-clock budget."""


@contextmanager
def time_limit(seconds: int | None):
    """SIGALRM-based wall-clock cap for in-process checks.

    Yields control immediately if `seconds` is None or non-positive. On Linux only
    (the project's only supported platform). When the alarm fires, raises
    `CheckTimeoutError` from inside whatever code is running.
    """
    if not seconds or seconds <= 0:
        yield
        return

    def _on_alarm(signum, frame):
        raise CheckTimeoutError(f"check exceeded {seconds}s budget")

    previous = signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def emit_heartbeat(check: str, **fields: Any) -> None:
    """Print a single sparse heartbeat line to stderr.

    Format: `[heartbeat] check=<name> k1=v1 k2=v2 ...`. Stderr only, flushed
    immediately so the calling agent sees it as the check progresses. Never
    write framework / Lightning / Hydra logs through this channel — heartbeats
    must stay one-line and low-volume.
    """
    pieces = [f"[heartbeat] check={check}"]
    for key, value in fields.items():
        if isinstance(value, float):
            pieces.append(f"{key}={value:.4g}")
        else:
            pieces.append(f"{key}={value}")
    print(" ".join(pieces), file=sys.stderr, flush=True)


DATASET_UNAVAILABLE_MARKERS = (
    "FileNotFoundError",
    "No such file or directory",
    "does not exist",
    "not found",
    "missing",
    "paths.data_dir",
    "data_dir",
    "cache file",
    "cache path",
    "PHMD",
    "download",
    "dataset unavailable",
)

DATASET_EXECUTION_MARKERS = (
    "load_data",
    "split_data",
    "PreProcessor",
    "preprocessor",
    "datasource",
    "malformed",
    "MemoryError",
    "out of memory",
    "resource exhausted",
    "Target and context sequencers must have the same length",
    "RULContextBatchDataset",
    "HydraConcatDataset",
    "KeyError: 'column_map'",
)

TOOL_INVOCATION_MARKERS = (
    "Could not override '",
    "Multiple values for experiment",
    "Error executing job with overrides",
    "Early stopping conditioned on metric `val/loss` which is not available",
    "ModelCheckpoint",
)


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_sec: float
    run_dir: str | None = None
    metrics_path: str | None = None
    timed_out: bool = False


@dataclass
class CheckResult:
    name: str
    status: str
    diagnostic: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    run_dirs: list[str] = field(default_factory=list)
    command: list[str] | None = None
    traceback_excerpt: str = ""
    dataset_executability: str = PASS
    fallback_trigger: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "diagnostic": self.diagnostic,
            "metrics": self.metrics,
            "run_dirs": self.run_dirs,
            "command": self.command,
            "traceback_excerpt": self.traceback_excerpt,
            "dataset_executability": self.dataset_executability,
            "fallback_trigger": self.fallback_trigger,
        }


def json_default(value: Any) -> Any:
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
    except Exception:
        pass
    if isinstance(value, Path):
        return str(value)
    return str(value)


def ensure_project_root(repo_root: str | Path) -> Path:
    root = Path(repo_root).expanduser().resolve()
    os.environ.setdefault("PROJECT_ROOT", str(root))
    return root


_TRAINER_APPEND_KEYS = frozenset(
    {
        "trainer.limit_train_batches",
        "trainer.limit_val_batches",
        "trainer.limit_test_batches",
        "trainer.overfit_batches",
    }
)


def _fix_override_prefix(override: str) -> str:
    if override.startswith(("+", "~", "++")):
        return override
    key = override.split("=", 1)[0]
    return f"+{override}" if key in _TRAINER_APPEND_KEYS else override


def base_overrides(extra_overrides: Iterable[str] | None = None) -> list[str]:
    overrides = [_fix_override_prefix(o) for o in (extra_overrides or [])]
    if not any(o.startswith("logger=") or o.startswith("+logger=") for o in overrides):
        overrides.insert(0, "logger=csv")
    return overrides


def build_picid_command(experiment_config: str, overrides: Iterable[str]) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "-m",
        "picid.run",
        f"experiment={experiment_config}",
        *overrides,
    ]


def list_metrics(repo_root: Path) -> list[Path]:
    artifacts = repo_root / "artifacts"
    if not artifacts.exists():
        return []
    return sorted(
        artifacts.rglob("csv_logs/version_*/metrics.csv"),
        key=lambda p: p.stat().st_mtime,
    )


def latest_metrics_since(repo_root: Path, since: float) -> Path | None:
    candidates = [p for p in list_metrics(repo_root) if p.stat().st_mtime >= since]
    if candidates:
        return candidates[-1]
    all_metrics = list_metrics(repo_root)
    return all_metrics[-1] if all_metrics else None


def run_dir_from_metrics(metrics_path: Path | None) -> str | None:
    if metrics_path is None:
        return None
    try:
        return str(metrics_path.parents[3])
    except IndexError:
        return str(metrics_path.parent)


def run_picid(
    repo_root: Path,
    experiment_config: str,
    overrides: Iterable[str],
    timeout_sec: int | None,
) -> CommandResult:
    def _coerce_text(value: str | bytes | None) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

    start = time.time()
    cmd = build_picid_command(experiment_config, overrides)
    env = os.environ.copy()
    env.setdefault("PROJECT_ROOT", str(repo_root))
    env.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    try:
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            command=cmd,
            returncode=124,
            stdout=_coerce_text(exc.stdout),
            stderr=_coerce_text(exc.stderr)
            + f"\n[TIMEOUT] Process exceeded {timeout_sec}s limit.",
            duration_sec=time.time() - start,
            timed_out=True,
        )
    metrics_path = latest_metrics_since(repo_root, start)
    return CommandResult(
        command=cmd,
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        duration_sec=time.time() - start,
        run_dir=run_dir_from_metrics(metrics_path),
        metrics_path=str(metrics_path) if metrics_path else None,
    )


def traceback_excerpt(text: str, limit: int = 4000) -> str:
    text = text.strip()
    if not text:
        return ""
    lines = text.splitlines()
    interesting: list[str] = []
    capture = False
    for line in lines:
        if "Traceback" in line:
            capture = True
        if capture:
            interesting.append(line)
    excerpt = "\n".join(interesting[-80:] if interesting else lines[-80:])
    return excerpt[-limit:]


def classify_dataset_failure(text: str) -> str | None:
    lowered = text.lower()
    for marker in DATASET_UNAVAILABLE_MARKERS:
        if marker.lower() in lowered:
            return DATASET_UNAVAILABLE
    for marker in DATASET_EXECUTION_MARKERS:
        if marker.lower() in lowered:
            return DATASET_EXECUTION_FAILED
    return None


def classify_tool_invocation_failure(text: str) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in TOOL_INVOCATION_MARKERS)


def failed_command_result(name: str, result: CommandResult) -> CheckResult:
    if result.timed_out:
        return CheckResult(
            name=name,
            status=PRECHECK_TIMEOUT,
            diagnostic=(
                f"Command timed out after {result.duration_sec:.0f}s. "
                "This is a timeout, not a confirmed failure — the operation may simply be slow. "
                "Consider increasing timeout_sec."
            ),
            run_dirs=[result.run_dir] if result.run_dir else [],
            command=result.command,
            traceback_excerpt=traceback_excerpt(result.stderr),
        )
    combined = f"{result.stdout}\n{result.stderr}"
    if classify_tool_invocation_failure(combined):
        return CheckResult(
            name=name,
            status=TOOL_INVOCATION_FAILURE,
            diagnostic="Runtime command failed due to tool/config invocation issues, not a confirmed dataset or model failure. Inspect traceback and command overrides/callbacks.",
            run_dirs=[result.run_dir] if result.run_dir else [],
            command=result.command,
            traceback_excerpt=traceback_excerpt(combined),
        )
    dataset_status = classify_dataset_failure(combined)
    if dataset_status is not None:
        return CheckResult(
            name=name,
            status=dataset_status,
            diagnostic=f"Runtime command failed before a usable sanity result: {dataset_status}.",
            run_dirs=[result.run_dir] if result.run_dir else [],
            command=result.command,
            traceback_excerpt=traceback_excerpt(combined),
            dataset_executability=dataset_status,
            fallback_trigger=True,
        )
    return CheckResult(
        name=name,
        status=FAIL,
        diagnostic="Runtime command failed. Inspect traceback for implementation or config errors.",
        run_dirs=[result.run_dir] if result.run_dir else [],
        command=result.command,
        traceback_excerpt=traceback_excerpt(combined),
    )


def read_metric_series(
    metrics_path: str | Path | None, names: Iterable[str]
) -> list[float]:
    if metrics_path is None:
        return []
    path = Path(metrics_path)
    if not path.exists():
        return []
    wanted = list(names)
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for name in wanted:
                raw = row.get(name)
                if raw in (None, ""):
                    continue
                try:
                    values.append(float(raw))
                    break
                except ValueError:
                    continue
    return values


def loss_step_series(metrics_path: str | Path | None) -> list[float]:
    return read_metric_series(
        metrics_path, ("train_loss_step", "train/loss_step", "train_loss", "train/loss")
    )


def loss_epoch_series(metrics_path: str | Path | None) -> list[float]:
    return read_metric_series(
        metrics_path,
        ("train_loss_epoch", "train/loss_epoch", "train_loss", "train/loss"),
    )


def is_bad_loss(value: float) -> bool:
    return math.isnan(value) or math.isinf(value) or value == 0.0
