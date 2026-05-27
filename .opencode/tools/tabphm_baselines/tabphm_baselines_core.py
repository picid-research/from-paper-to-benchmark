"""Load PICID internal baselines from report_output/*/results.nc.

Reads the frozen leaderboard NetCDFs produced by picid_report and returns
structured rows per (model, metric_key). Used by the /paper-hypothesis and
/evaluate-results skills so paper-claim validation has a consistent yardstick
that is NOT recomputed every run.

Public surface:
    - load_baselines(...)
    - resolve_folder(...)
    - list_datasets(...)
    - short_name_for(...)

The tool emits two sparse `[heartbeat]` lines on stderr per load call (start
and done) and never prints framework logs.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


STATUS_OK = "OK"
STATUS_FOLDER_NOT_FOUND = "FOLDER_NOT_FOUND"
STATUS_METRIC_MISSING = "METRIC_MISSING"
STATUS_CORRUPT_NETCDF = "CORRUPT_NETCDF"

DEFAULT_REPORT_OUTPUT_SUBDIR = "report_output"
DATASET_MAP_FILENAME = "dataset_map.yaml"


def emit_heartbeat(phase: str, **fields: Any) -> None:
    pieces = [f"[heartbeat] load phase={phase}"]
    for key, value in fields.items():
        pieces.append(f"{key}={value}")
    print(" ".join(pieces), file=sys.stderr, flush=True)


def _dataset_map_path() -> Path:
    return Path(__file__).with_name(DATASET_MAP_FILENAME)


def _load_dataset_map() -> dict[str, Any]:
    path = _dataset_map_path()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _report_output_root(repo_root: Path, override: str | Path | None) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
    return (repo_root / DEFAULT_REPORT_OUTPUT_SUBDIR).resolve()


@dataclass
class ResolvedFolder:
    status: str
    folder: Path | None = None
    candidates: list[str] | None = None
    reason: str = ""


def resolve_folder(
    repo_root: Path,
    dataset: str,
    task_type: str,
    subtask: str | None,
    report_output_root: str | Path | None,
) -> ResolvedFolder:
    root = _report_output_root(repo_root, report_output_root)
    if not root.exists():
        return ResolvedFolder(
            status=STATUS_FOLDER_NOT_FOUND,
            reason=f"report_output root does not exist: {root}",
        )

    mapping = _load_dataset_map()
    entry = mapping.get(dataset)
    if entry is not None:
        task_entry = entry.get(task_type) if isinstance(entry, dict) else None
        folder_name: str | None = None
        if isinstance(task_entry, str):
            folder_name = task_entry
        elif isinstance(task_entry, dict):
            key = subtask if subtask else "default"
            folder_name = task_entry.get(key) or task_entry.get("default")
        if folder_name:
            candidate = root / folder_name
            if candidate.exists():
                return ResolvedFolder(status=STATUS_OK, folder=candidate)
            return ResolvedFolder(
                status=STATUS_FOLDER_NOT_FOUND,
                reason=f"mapped folder missing on disk: {candidate}",
            )

    glob_pattern = f"*{dataset}*{task_type}*"
    glob_matches = sorted(p for p in root.glob(glob_pattern) if p.is_dir())
    if len(glob_matches) == 1:
        return ResolvedFolder(status=STATUS_OK, folder=glob_matches[0])
    if len(glob_matches) > 1:
        return ResolvedFolder(
            status=STATUS_FOLDER_NOT_FOUND,
            candidates=[p.name for p in glob_matches],
            reason=(
                f"ambiguous glob for '{glob_pattern}' matched {len(glob_matches)} folders; "
                "add an explicit entry to dataset_map.yaml"
            ),
        )
    return ResolvedFolder(
        status=STATUS_FOLDER_NOT_FOUND,
        reason=(
            f"no folder in {root} matches dataset='{dataset}' task_type='{task_type}' "
            f"subtask='{subtask}' (neither dataset_map.yaml nor glob '{glob_pattern}')"
        ),
    )


_SUFFIX_STRIP = re.compile(r"(_Wrapper|_Forecaster|Wrapper|Forecaster)$")
_SPLIT_LOWER_UPPER = re.compile(r"(?<=[a-z])(?=[A-Z])")
_SPLIT_ACRONYM_WORD = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")


def short_name_for(model_str: str) -> str:
    """Derive a short, stable baseline name from a dotted-class path.

    Keeps acronym clumps intact (`LSTM` -> `lstm`, not `l_s_t_m`) and only
    splits at camelCase boundaries (`FitPredictXGBoost` -> `fit_predict_xg_boost`).

    Examples:
        'baselines.lstm_model.LSTM_Forecaster' -> 'lstm'
        'MLPWrapper' -> 'mlp'
        'baselines.stats.StatisticalBaselineWrapper (linear)' -> 'statistical_baseline_linear'
    """
    raw = model_str.strip()
    paren_suffix = ""
    if raw.endswith(")") and "(" in raw:
        head, tail = raw.rsplit("(", 1)
        paren_suffix = "_" + tail[:-1].strip().lower().replace(" ", "_")
        raw = head.rstrip()
    last = raw.rsplit(".", 1)[-1]
    last = _SUFFIX_STRIP.sub("", last)
    last = _SPLIT_LOWER_UPPER.sub("_", last)
    last = _SPLIT_ACRONYM_WORD.sub("_", last)
    last = last.lower()
    last = re.sub(r"_+", "_", last).strip("_")
    return f"{last}{paren_suffix}"


def list_datasets(
    repo_root: Path,
    report_output_root: str | Path | None = None,
) -> list[str]:
    root = _report_output_root(repo_root, report_output_root)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "results.nc").exists())


def _scalar(value: Any) -> Any:
    try:
        import numpy as np  # local import to keep module import cheap
    except Exception:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist() if value.shape else value.item()
    return value


def _stringify(value: Any) -> str:
    value = _scalar(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def load_baselines(
    repo_root: str | Path,
    dataset: str,
    task_type: str,
    metric_keys: Iterable[str],
    subtask: str | None = None,
    models_filter: str | None = None,
    report_output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load baseline rows for `dataset`/`task_type` from results.nc.

    Returns a plain dict suitable for JSON serialization. On any recoverable
    failure the `status` field is set to a non-OK value and `rows` is empty.
    """
    repo = Path(repo_root).expanduser().resolve()
    wanted = list(dict.fromkeys(metric_keys))  # preserve order, drop dupes
    emit_heartbeat(
        "start",
        dataset=dataset,
        task_type=task_type,
        subtask=subtask or "-",
        n_metric_keys=len(wanted),
    )

    resolved = resolve_folder(repo, dataset, task_type, subtask, report_output_root)
    if resolved.status != STATUS_OK or resolved.folder is None:
        emit_heartbeat("done", status=resolved.status, n_models=0, n_metrics=0)
        return {
            "status": resolved.status,
            "resolved_folder": None,
            "candidates": resolved.candidates or [],
            "reason": resolved.reason,
            "project_name": None,
            "dataset": None,
            "available_metric_keys": [],
            "reporting_metrics": [],
            "n_models": 0,
            "rows": [],
            "requested_metric_keys": wanted,
            "missing_metric_keys": list(wanted),
            "warnings": [],
        }

    nc_path = resolved.folder / "results.nc"
    if not nc_path.exists():
        emit_heartbeat("done", status=STATUS_FOLDER_NOT_FOUND, n_models=0, n_metrics=0)
        return {
            "status": STATUS_FOLDER_NOT_FOUND,
            "resolved_folder": resolved.folder.name,
            "candidates": [],
            "reason": f"results.nc missing inside {resolved.folder}",
            "project_name": None,
            "dataset": None,
            "available_metric_keys": [],
            "reporting_metrics": [],
            "n_models": 0,
            "rows": [],
            "requested_metric_keys": wanted,
            "missing_metric_keys": list(wanted),
            "warnings": [],
        }

    try:
        import xarray as xr
    except ImportError as exc:
        emit_heartbeat("done", status=STATUS_CORRUPT_NETCDF, n_models=0, n_metrics=0)
        return {
            "status": STATUS_CORRUPT_NETCDF,
            "resolved_folder": resolved.folder.name,
            "reason": f"xarray not available: {exc}",
            "project_name": None,
            "dataset": None,
            "available_metric_keys": [],
            "reporting_metrics": [],
            "n_models": 0,
            "rows": [],
            "requested_metric_keys": wanted,
            "missing_metric_keys": list(wanted),
            "warnings": [],
        }

    try:
        ds = xr.open_dataset(nc_path)
    except Exception as exc:
        emit_heartbeat("done", status=STATUS_CORRUPT_NETCDF, n_models=0, n_metrics=0)
        return {
            "status": STATUS_CORRUPT_NETCDF,
            "resolved_folder": resolved.folder.name,
            "reason": f"failed to open {nc_path}: {exc!r}",
            "project_name": None,
            "dataset": None,
            "available_metric_keys": [],
            "reporting_metrics": [],
            "n_models": 0,
            "rows": [],
            "requested_metric_keys": wanted,
            "missing_metric_keys": list(wanted),
            "warnings": [],
        }

    try:
        warnings: list[str] = []
        available_metric_keys = [_stringify(v) for v in ds["metric_key"].values.tolist()]
        all_models = [_stringify(v) for v in ds["model"].values.tolist()]
        datasets_in_file = [_stringify(v) for v in ds["dataset"].values.tolist()]
        dataset_label = datasets_in_file[0] if datasets_in_file else None
        project_name = _stringify(ds.attrs.get("project_name", resolved.folder.name))
        reporting_metrics = [
            _stringify(m) for m in list(ds.attrs.get("reporting_metrics", []) or [])
        ]

        kept_models: list[str] = []
        for m in all_models:
            if models_filter and models_filter not in m:
                continue
            kept_models.append(m)

        missing = [k for k in wanted if k not in available_metric_keys]
        present_keys = [k for k in wanted if k in available_metric_keys]

        mean_da = ds["mean"] if "mean" in ds else None
        std_da = ds["std"] if "std" in ds else None
        n_da = ds["n"] if "n" in ds else None
        opt_value_da = ds["opt_value"] if "opt_value" in ds else None
        opt_metric_da = ds["opt_metric"] if "opt_metric" in ds else None
        sort_metric_da = ds["sort_metric"] if "sort_metric" in ds else None

        def _pick(da, **sel):
            if da is None:
                return None
            try:
                value = _scalar(da.sel(**sel).values)
            except Exception:
                return None
            if isinstance(value, float):
                import math as _m
                if _m.isnan(value):
                    return None
            return value

        rows: list[dict[str, Any]] = []
        for model in kept_models:
            opt_value = _pick(opt_value_da, dataset=dataset_label, model=model)
            opt_metric = _pick(opt_metric_da, dataset=dataset_label, model=model)
            sort_metric = _pick(sort_metric_da, dataset=dataset_label, model=model)
            for key in present_keys:
                row = {
                    "model": model,
                    "short_name": short_name_for(model),
                    "metric_key": key,
                    "mean": _pick(mean_da, dataset=dataset_label, model=model, metric_key=key),
                    "std": _pick(std_da, dataset=dataset_label, model=model, metric_key=key),
                    "n": _pick(n_da, dataset=dataset_label, model=model, metric_key=key),
                    "opt_value": opt_value,
                    "opt_metric": opt_metric,
                    "sort_metric": sort_metric,
                }
                rows.append(row)

        if present_keys:
            status = STATUS_OK if not missing else STATUS_METRIC_MISSING
        else:
            status = STATUS_METRIC_MISSING

        result = {
            "status": status,
            "resolved_folder": resolved.folder.name,
            "candidates": [],
            "reason": "" if status == STATUS_OK else f"missing metric keys: {missing}",
            "project_name": project_name,
            "dataset": dataset_label,
            "available_metric_keys": available_metric_keys,
            "reporting_metrics": reporting_metrics,
            "n_models": len(kept_models),
            "rows": rows,
            "requested_metric_keys": wanted,
            "missing_metric_keys": missing,
            "warnings": warnings,
        }
    finally:
        ds.close()

    emit_heartbeat(
        "done",
        status=result["status"],
        n_models=result["n_models"],
        n_metrics=len(present_keys) if present_keys else 0,
    )
    return result
