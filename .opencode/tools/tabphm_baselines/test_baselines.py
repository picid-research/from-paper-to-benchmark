"""Smoke tests for the PICID_baselines tool.

The tests depend on real `report_output/*/results.nc` files sitting in the repo.
If the NB14 folder is missing the tests are skipped at collection time — this
keeps the tests useful for developer machines while not blocking CI on a repo
that chose to ship without the frozen leaderboard.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from PICID_baselines_core import (  # noqa: E402
    STATUS_FOLDER_NOT_FOUND,
    STATUS_METRIC_MISSING,
    STATUS_OK,
    load_baselines,
    resolve_folder,
    short_name_for,
)


REPO_ROOT = THIS_DIR.parents[2]
REPORT_OUTPUT_ROOT = REPO_ROOT / "report_output"
NB14_FOLDER = REPORT_OUTPUT_ROOT / "16_03_2026_nb14_prognostics_combined"


requires_nb14 = pytest.mark.skipif(
    not (NB14_FOLDER / "results.nc").exists(),
    reason="NB14 report_output results.nc not available on this machine",
)


def test_short_name_basic():
    assert short_name_for("baselines.lstm_model.LSTM_Forecaster") == "lstm"
    assert short_name_for("MLPWrapper") == "mlp"
    assert short_name_for("baselines.crossformer_model.Crossformer_Forecaster") == "crossformer"


def test_short_name_parenthesized_suffix_keeps_distinct():
    a = short_name_for("baselines.stats.StatisticalBaselineWrapper(linear)")
    b = short_name_for("baselines.stats.StatisticalBaselineWrapper(exponential)")
    assert a == "statistical_baseline_linear"
    assert b == "statistical_baseline_exponential"
    assert a != b


def test_resolve_folder_maps_nb14():
    resolved = resolve_folder(
        REPO_ROOT,
        dataset="nb14",
        task_type="prognostics",
        subtask=None,
        report_output_root=REPORT_OUTPUT_ROOT,
    )
    assert resolved.status in (STATUS_OK, STATUS_FOLDER_NOT_FOUND)
    if resolved.status == STATUS_OK:
        assert resolved.folder is not None
        assert resolved.folder.name == "16_03_2026_nb14_prognostics_combined"


def test_resolve_folder_unknown_returns_folder_not_found():
    resolved = resolve_folder(
        REPO_ROOT,
        dataset="does_not_exist_xyz",
        task_type="prognostics",
        subtask=None,
        report_output_root=REPORT_OUTPUT_ROOT,
    )
    assert resolved.status == STATUS_FOLDER_NOT_FOUND


@requires_nb14
def test_load_baselines_nb14_ok():
    result = load_baselines(
        repo_root=REPO_ROOT,
        dataset="nb14",
        task_type="prognostics",
        metric_keys=["test/rmse_denormalized", "test/mae_denormalized"],
        report_output_root=REPORT_OUTPUT_ROOT,
    )
    assert result["status"] == STATUS_OK
    assert result["resolved_folder"] == "16_03_2026_nb14_prognostics_combined"
    assert result["n_models"] >= 5
    short_names = {row["short_name"] for row in result["rows"]}
    assert "lstm" in short_names
    assert any(row["metric_key"] == "test/rmse_denormalized" for row in result["rows"])
    assert "reporting_metrics" in result
    assert result["missing_metric_keys"] == []


@requires_nb14
def test_load_baselines_nb14_missing_metric():
    result = load_baselines(
        repo_root=REPO_ROOT,
        dataset="nb14",
        task_type="prognostics",
        metric_keys=["test/does_not_exist"],
        report_output_root=REPORT_OUTPUT_ROOT,
    )
    assert result["status"] == STATUS_METRIC_MISSING
    assert "test/does_not_exist" in result["missing_metric_keys"]
    assert result["available_metric_keys"]  # still populated


@requires_nb14
def test_load_baselines_matches_direct_xarray_read():
    import xarray as xr

    result = load_baselines(
        repo_root=REPO_ROOT,
        dataset="nb14",
        task_type="prognostics",
        metric_keys=["test/rmse_denormalized"],
        report_output_root=REPORT_OUTPUT_ROOT,
    )
    assert result["status"] == STATUS_OK
    sample = result["rows"][0]
    ds = xr.open_dataset(NB14_FOLDER / "results.nc")
    try:
        expected_mean = ds["mean"].sel(
            dataset=result["dataset"],
            model=sample["model"],
            metric_key=sample["metric_key"],
        ).item()
    finally:
        ds.close()
    assert sample["mean"] == pytest.approx(expected_mean, rel=1e-9, abs=1e-12)


def test_load_baselines_unknown_dataset_is_folder_not_found():
    result = load_baselines(
        repo_root=REPO_ROOT,
        dataset="does_not_exist_xyz",
        task_type="prognostics",
        metric_keys=["test/rmse_denormalized"],
        report_output_root=REPORT_OUTPUT_ROOT,
    )
    assert result["status"] == STATUS_FOLDER_NOT_FOUND
    assert result["rows"] == []


@requires_nb14
def test_cli_load_emits_exactly_two_heartbeats(tmp_path):
    """The tool must emit two and only two `[heartbeat]` lines per load call."""
    script = THIS_DIR / "PICID_baselines.py"
    payload = {
        "repo_root": str(REPO_ROOT),
        "dataset": "nb14",
        "task_type": "prognostics",
        "metric_keys": ["test/rmse_denormalized"],
        "report_output_root": str(REPORT_OUTPUT_ROOT),
    }
    proc = subprocess.run(
        ["uv", "run", "python", str(script), "load", "--input-json", json.dumps(payload)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    heartbeat_lines = [ln for ln in proc.stderr.splitlines() if ln.startswith("[heartbeat]")]
    assert len(heartbeat_lines) == 2, proc.stderr
    marker = "PICID_BASELINES_RESULT_JSON="
    assert marker in proc.stdout
    parsed = json.loads(proc.stdout.split(marker, 1)[1].strip())
    assert parsed["status"] == "OK"
