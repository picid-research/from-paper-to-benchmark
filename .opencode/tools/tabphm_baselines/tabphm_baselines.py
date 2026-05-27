"""CLI entrypoint for the PICID baselines tool.

OpenCode tool wrapper passes a JSON payload via --input-json and reads one
marker-prefixed JSON result from stdout. Stderr is inherited so [heartbeat]
lines surface in the calling session.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).parent))

from PICID_baselines_core import (  # noqa: E402
    list_datasets,
    load_baselines,
    resolve_folder,
    short_name_for,
)


RESULT_MARKER = "PICID_BASELINES_RESULT_JSON="


def _finish(result: dict[str, Any]) -> None:
    print(RESULT_MARKER + json.dumps(result, sort_keys=True))


def _payload_repo_root(payload: dict[str, Any]) -> Path:
    return Path(payload.get("repo_root") or ".").expanduser().resolve()


def _run_load(payload: dict[str, Any]) -> dict[str, Any]:
    dataset = payload.get("dataset")
    task_type = payload.get("task_type")
    metric_keys = payload.get("metric_keys") or []
    if not dataset or not task_type:
        raise ValueError("load requires 'dataset' and 'task_type'")
    if not isinstance(metric_keys, list) or not metric_keys:
        raise ValueError("load requires a non-empty 'metric_keys' list")
    return load_baselines(
        repo_root=_payload_repo_root(payload),
        dataset=str(dataset),
        task_type=str(task_type),
        subtask=payload.get("subtask"),
        metric_keys=[str(m) for m in metric_keys],
        models_filter=payload.get("models_filter"),
        report_output_root=payload.get("report_output_root"),
    )


def _run_resolve_folder(payload: dict[str, Any]) -> dict[str, Any]:
    dataset = payload.get("dataset")
    task_type = payload.get("task_type")
    if not dataset or not task_type:
        raise ValueError("resolve_folder requires 'dataset' and 'task_type'")
    resolved = resolve_folder(
        _payload_repo_root(payload),
        str(dataset),
        str(task_type),
        payload.get("subtask"),
        payload.get("report_output_root"),
    )
    return {
        "status": resolved.status,
        "folder": resolved.folder.name if resolved.folder else None,
        "candidates": resolved.candidates or [],
        "reason": resolved.reason,
    }


def _run_list_datasets(payload: dict[str, Any]) -> dict[str, Any]:
    folders = list_datasets(
        _payload_repo_root(payload),
        payload.get("report_output_root"),
    )
    return {"status": "OK", "folders": folders}


def _run_short_name(payload: dict[str, Any]) -> dict[str, Any]:
    model = payload.get("model")
    if not model:
        raise ValueError("short_name requires 'model'")
    return {"status": "OK", "model": model, "short_name": short_name_for(str(model))}


def run_command(command: str, payload: dict[str, Any]) -> dict[str, Any]:
    if command == "load":
        return _run_load(payload)
    if command == "resolve_folder":
        return _run_resolve_folder(payload)
    if command == "list_datasets":
        return _run_list_datasets(payload)
    if command == "short_name":
        return _run_short_name(payload)
    raise ValueError(f"Unknown command: {command}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query PICID internal baselines.")
    parser.add_argument(
        "command",
        choices=["load", "resolve_folder", "list_datasets", "short_name"],
    )
    parser.add_argument("--input-json", required=True)
    args = parser.parse_args()
    payload = json.loads(args.input_json)
    result = run_command(args.command, payload)
    _finish(result)


if __name__ == "__main__":
    main()
