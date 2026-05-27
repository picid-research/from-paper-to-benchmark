"""Tests for PICID sanity tool fixes.

Covers:
1. Hydra override syntax (experiment= vs +experiment=) -- still used by the
   opt-in subset_convergence CLI path.
2. Subprocess timeout handling -- still used by subset_convergence.
3. Config preflight failure detection.
4. Report attempt tracking / supersession.
5. Failed command result classification.
6. New post-trim utilities: DEFAULT_TIMEOUT_SEC, _resolve_timeout, time_limit
   SIGALRM context manager, emit_heartbeat stderr format, stack cache reuse.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path
from unittest import mock

import pytest

# Ensure the module is importable from this directory
import sys

sys.path.insert(0, str(Path(__file__).parent))

from PICID_sanity_core import (
    BLOCK,
    DATASET_EXECUTION_FAILED,
    DATASET_UNAVAILABLE,
    DEFAULT_TIMEOUT_SEC,
    FAIL,
    INVESTIGATE,
    PASS,
    PRECHECK_TIMEOUT,
    PROCEED,
    SKIPPED,
    TOOL_INVOCATION_FAILURE,
    WARN_CONTINUE,
    CheckResult,
    CheckTimeoutError,
    CommandResult,
    _fix_override_prefix,
    _next_attempt_id,
    _resolve_timeout,
    base_overrides,
    build_picid_command,
    check_overfit_batch,
    classify_dataset_failure,
    emit_heartbeat,
    failed_command_result,
    interpret_failure,
    render_markdown,
    summarize_verdict,
    time_limit,
    traceback_excerpt,
)
from sanity_framework import (
    _STACK_CACHE,
    _cache_key,
    clear_stack_cache,
)
from sanity_reporting import ladder_result


# ---------------------------------------------------------------------------
# 1. Hydra override syntax
# ---------------------------------------------------------------------------


class TestBuildPicidCommand:
    def test_uses_override_syntax_not_plus(self):
        """build_picid_command must use 'experiment=' (override), not '+experiment=' (add)."""
        cmd = build_picid_command(
            "nb14/prognostics/combined/detransformer", ["logger=csv"]
        )
        experiment_arg = [arg for arg in cmd if "experiment=" in arg]
        assert len(experiment_arg) == 1
        assert experiment_arg[0] == "experiment=nb14/prognostics/combined/detransformer"
        assert not experiment_arg[0].startswith("+")

    def test_includes_uv_run_prefix(self):
        cmd = build_picid_command("test/config", [])
        assert cmd[:5] == ["uv", "run", "python", "-m", "picid.run"]

    def test_appends_overrides(self):
        overrides = ["trainer.max_epochs=10", "logger=csv"]
        cmd = build_picid_command("test/config", overrides)
        assert "trainer.max_epochs=10" in cmd
        assert "logger=csv" in cmd

    def test_no_plus_prefix_in_any_experiment_arg(self):
        """Ensure no +experiment= appears even with complex config paths."""
        configs = [
            "pronostia/prognostics/raw/my_model",
            "nb14/prognostics/combined/detransformer",
            "unibo/prognostics/combined/detransformer",
        ]
        for cfg in configs:
            cmd = build_picid_command(cfg, ["logger=csv"])
            for arg in cmd:
                if "experiment=" in arg:
                    assert not arg.startswith("+"), (
                        f"Found +experiment= in command for {cfg}: {arg}"
                    )


# ---------------------------------------------------------------------------
# 1b. Hydra override prefix auto-detection
# ---------------------------------------------------------------------------


class TestFixOverridePrefix:
    """Trainer throttling keys must use + append syntax; others must not."""

    def test_limit_train_batches_gets_plus(self):
        assert (
            _fix_override_prefix("trainer.limit_train_batches=1")
            == "+trainer.limit_train_batches=1"
        )

    def test_limit_val_batches_gets_plus(self):
        assert (
            _fix_override_prefix("trainer.limit_val_batches=0")
            == "+trainer.limit_val_batches=0"
        )

    def test_limit_test_batches_gets_plus(self):
        assert (
            _fix_override_prefix("trainer.limit_test_batches=0.5")
            == "+trainer.limit_test_batches=0.5"
        )

    def test_overfit_batches_gets_plus(self):
        assert (
            _fix_override_prefix("trainer.overfit_batches=1")
            == "+trainer.overfit_batches=1"
        )

    def test_already_prefixed_not_doubled(self):
        assert (
            _fix_override_prefix("+trainer.limit_train_batches=1")
            == "+trainer.limit_train_batches=1"
        )

    def test_normal_override_not_prefixed(self):
        assert _fix_override_prefix("trainer.max_epochs=10") == "trainer.max_epochs=10"

    def test_logger_not_prefixed(self):
        assert _fix_override_prefix("logger=csv") == "logger=csv"

    def test_enable_progress_bar_not_prefixed(self):
        assert (
            _fix_override_prefix("enable_progress_bar=false")
            == "enable_progress_bar=false"
        )

    def test_datamodule_not_prefixed(self):
        assert (
            _fix_override_prefix("datamodule.train_batch_size=8")
            == "datamodule.train_batch_size=8"
        )

    def test_tilde_prefix_preserved(self):
        assert _fix_override_prefix("~trainer.callbacks") == "~trainer.callbacks"


class TestBaseOverridesWithPrefix:
    """base_overrides should auto-fix trainer throttling keys."""

    def test_init_loss_overrides(self):
        result = base_overrides(
            [
                "trainer.max_epochs=1",
                "trainer.limit_train_batches=1",
                "trainer.limit_val_batches=0",
                "trainer.limit_test_batches=0",
                "enable_progress_bar=false",
            ]
        )
        assert "+trainer.limit_train_batches=1" in result
        assert "+trainer.limit_val_batches=0" in result
        assert "+trainer.limit_test_batches=0" in result
        assert "trainer.max_epochs=1" in result  # no +
        assert "enable_progress_bar=false" in result  # no +

    def test_overfit_batch_overrides(self):
        result = base_overrides(
            [
                "trainer.max_epochs=200",
                "trainer.overfit_batches=1",
                "trainer.limit_val_batches=0",
                "enable_progress_bar=false",
                "datamodule.train_batch_size=8",
            ]
        )
        assert "+trainer.overfit_batches=1" in result
        assert "+trainer.limit_val_batches=0" in result
        assert "trainer.max_epochs=200" in result  # no +
        assert "datamodule.train_batch_size=8" in result  # no +

    def test_subset_convergence_overrides(self):
        result = base_overrides(
            [
                "trainer.max_epochs=30",
                "trainer.limit_train_batches=0.1",
                "trainer.limit_val_batches=0.1",
                "trainer.limit_test_batches=0",
                "enable_progress_bar=false",
            ]
        )
        assert "+trainer.limit_train_batches=0.1" in result
        assert "+trainer.limit_val_batches=0.1" in result
        assert "+trainer.limit_test_batches=0" in result
        assert "trainer.max_epochs=30" in result  # no +

    def test_end_to_end_init_loss_command(self):
        """Full command for init_loss should have correct prefix on all overrides."""
        overrides = base_overrides(
            [
                "trainer.max_epochs=1",
                "trainer.limit_train_batches=1",
                "trainer.limit_val_batches=0",
                "trainer.limit_test_batches=0",
                "enable_progress_bar=false",
            ]
        )
        cmd = build_picid_command("nb14/prognostics/combined/detransformer", overrides)
        cmd_str = " ".join(cmd)
        # experiment uses plain override
        assert "experiment=nb14" in cmd_str
        assert "+experiment=" not in cmd_str
        # throttling keys use + append
        assert "+trainer.limit_train_batches=1" in cmd_str
        assert "+trainer.limit_val_batches=0" in cmd_str
        assert "+trainer.limit_test_batches=0" in cmd_str
        # max_epochs uses plain override
        assert " trainer.max_epochs=1" in cmd_str


# ---------------------------------------------------------------------------
# 2. Timeout handling
# ---------------------------------------------------------------------------


class TestTimeoutHandling:
    def test_command_result_timed_out_field(self):
        """CommandResult should have a timed_out field."""
        result = CommandResult(
            command=["echo"],
            returncode=124,
            stdout="",
            stderr="",
            duration_sec=600.0,
            timed_out=True,
        )
        assert result.timed_out is True

    def test_command_result_default_not_timed_out(self):
        result = CommandResult(
            command=["echo"],
            returncode=0,
            stdout="ok",
            stderr="",
            duration_sec=1.0,
        )
        assert result.timed_out is False

    def test_failed_command_result_timeout_produces_precheck_timeout(self):
        """When a command times out, failed_command_result should return PRECHECK_TIMEOUT, not FAIL."""
        result = CommandResult(
            command=["uv", "run", "python", "-m", "picid.run", "experiment=test"],
            returncode=124,
            stdout="",
            stderr="",
            duration_sec=600.0,
            timed_out=True,
        )
        check = failed_command_result("init_loss", result)
        assert check.status == PRECHECK_TIMEOUT
        assert "timed out" in check.diagnostic.lower()
        assert check.name == "init_loss"

    def test_failed_command_result_normal_failure_still_works(self):
        """Non-timeout failures should still produce FAIL."""
        result = CommandResult(
            command=["uv", "run", "python", "-m", "picid.run"],
            returncode=1,
            stdout="",
            stderr="RuntimeError: some model bug",
            duration_sec=5.0,
        )
        check = failed_command_result("overfit_batch", result)
        assert check.status == FAIL

    def test_failed_command_result_dataset_unavailable(self):
        """Dataset-related failures should still be classified correctly."""
        result = CommandResult(
            command=["uv", "run", "python", "-m", "picid.run"],
            returncode=1,
            stdout="",
            stderr="FileNotFoundError: No such file or directory: '/data/nb14/raw'",
            duration_sec=3.0,
        )
        check = failed_command_result("init_loss", result)
        assert check.status == DATASET_UNAVAILABLE
        assert check.fallback_trigger is True

    def test_failed_command_result_dataset_execution_failed(self):
        result = CommandResult(
            command=["uv", "run", "python", "-m", "picid.run"],
            returncode=1,
            stdout="",
            stderr="MemoryError during load_data preprocessing",
            duration_sec=60.0,
        )
        check = failed_command_result("init_loss", result)
        assert check.status == DATASET_EXECUTION_FAILED
        assert check.fallback_trigger is True

    def test_timeout_not_marked_as_dataset_failure(self):
        """Timeout should NOT be classified as dataset failure — it's a separate status."""
        result = CommandResult(
            command=["uv", "run", "python", "-m", "picid.run"],
            returncode=124,
            stdout="",
            stderr="",
            duration_sec=600.0,
            timed_out=True,
        )
        check = failed_command_result("init_loss", result)
        assert check.status == PRECHECK_TIMEOUT
        assert check.fallback_trigger is False

    def test_run_picid_timeout_decodes_bytes(self):
        from sanity_base import run_picid

        timeout = subprocess.TimeoutExpired(
            cmd=["uv", "run", "python", "-m", "picid.run"],
            timeout=1,
            output=b"partial stdout",
            stderr=b"partial stderr",
        )

        with mock.patch("sanity_base.subprocess.run", side_effect=timeout):
            result = run_picid(Path("/PICID"), "test/config", [], 1)

        assert result.timed_out is True
        assert result.returncode == 124
        assert result.stdout == "partial stdout"
        assert "partial stderr" in result.stderr
        assert "[TIMEOUT]" in result.stderr


# ---------------------------------------------------------------------------
# 3. Config preflight / summarize_verdict
# ---------------------------------------------------------------------------


class TestSummarizeVerdict:
    def test_all_pass(self):
        checks = [
            CheckResult(name="init_loss", status=PASS),
            CheckResult(name="overfit_batch", status=PASS),
        ]
        assert summarize_verdict(checks) == PROCEED

    def test_all_skipped(self):
        checks = [CheckResult(name="init_loss", status=SKIPPED)]
        assert summarize_verdict(checks) == PROCEED

    def test_dataset_unavailable_takes_priority(self):
        checks = [
            CheckResult(name="init_loss", status=DATASET_UNAVAILABLE),
            CheckResult(name="overfit_batch", status=PASS),
        ]
        assert summarize_verdict(checks) == DATASET_UNAVAILABLE

    def test_tool_invocation_failure(self):
        checks = [
            CheckResult(name="config_preflight", status=TOOL_INVOCATION_FAILURE),
        ]
        assert summarize_verdict(checks) == TOOL_INVOCATION_FAILURE

    def test_precheck_timeout(self):
        checks = [
            CheckResult(name="init_loss", status=PRECHECK_TIMEOUT),
            CheckResult(name="overfit_batch", status=PASS),
        ]
        assert summarize_verdict(checks) == PRECHECK_TIMEOUT

    def test_dataset_failure_has_priority_over_tool_invocation(self):
        checks = [
            CheckResult(name="init_loss", status=DATASET_UNAVAILABLE),
            CheckResult(name="config_preflight", status=TOOL_INVOCATION_FAILURE),
        ]
        assert summarize_verdict(checks) == DATASET_UNAVAILABLE

    def test_tool_invocation_has_priority_over_timeout(self):
        checks = [
            CheckResult(name="config_preflight", status=TOOL_INVOCATION_FAILURE),
            CheckResult(name="init_loss", status=PRECHECK_TIMEOUT),
        ]
        assert summarize_verdict(checks) == TOOL_INVOCATION_FAILURE

    def test_only_subset_convergence_fails_returns_warn_continue(self):
        checks = [
            CheckResult(name="init_loss", status=PASS),
            CheckResult(name="zero_input", status=PASS),
            CheckResult(name="overfit_batch", status=PASS),
            CheckResult(name="gradient_flow", status=PASS),
            CheckResult(name="subset_convergence", status=FAIL),
        ]
        assert summarize_verdict(checks) == WARN_CONTINUE

    def test_overfit_batch_investigate_returns_warn_continue(self):
        checks = [
            CheckResult(name="init_loss", status=PASS),
            CheckResult(name="gradient_flow", status=PASS),
            CheckResult(name="overfit_batch", status=INVESTIGATE),
        ]
        assert summarize_verdict(checks) == WARN_CONTINUE

    def test_noncatastrophic_core_check_failure_returns_warn_continue(self):
        checks = [
            CheckResult(name="init_loss", status=FAIL),
            CheckResult(name="overfit_batch", status=PASS),
        ]
        assert summarize_verdict(checks) == WARN_CONTINUE

    def test_no_meaningful_gradient_flow_returns_block(self):
        checks = [
            CheckResult(name="init_loss", status=PASS),
            CheckResult(
                name="gradient_flow",
                status=FAIL,
                metrics={
                    "gradient_norms": {"layer.weight": 0.0},
                    "dead_layers": ["layer.weight"],
                },
            ),
            CheckResult(name="overfit_batch", status=PASS),
        ]
        assert summarize_verdict(checks) == BLOCK


class TestInterpretFailure:
    def test_proceed(self):
        msg = interpret_failure([], PROCEED)
        assert "passed" in msg.lower()

    def test_tool_invocation(self):
        msg = interpret_failure([], TOOL_INVOCATION_FAILURE)
        assert "tool" in msg.lower() or "config" in msg.lower()

    def test_precheck_timeout(self):
        msg = interpret_failure([], PRECHECK_TIMEOUT)
        assert "timeout" in msg.lower() or "timed out" in msg.lower()

    def test_investigate(self):
        msg = interpret_failure([], INVESTIGATE)
        assert "inspect" in msg.lower() or "nuanced" in msg.lower()

    def test_warn_continue(self):
        msg = interpret_failure([], WARN_CONTINUE)
        assert "continue" in msg.lower()

    def test_dataset_unavailable(self):
        msg = interpret_failure([], DATASET_UNAVAILABLE)
        assert "dataset" in msg.lower()


# ---------------------------------------------------------------------------
# 4. Report attempt tracking
# ---------------------------------------------------------------------------


class TestAttemptTracking:
    def test_first_attempt_returns_1(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert _next_attempt_id(tmpdir, "test/config") == 1

    def test_no_vault_dir_returns_1(self):
        assert _next_attempt_id(None, "test/config") == 1

    def test_increments_after_existing_attempt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "06-sanity-ladder-log.md"
            log_path.write_text(
                textwrap.dedent("""\
                ## Sanity Ladder -- Runtime Checks

                **Timestamp**: 2026-04-13T10:00:00
                **Experiment config**: `test/config`
                **Attempt**: 1

                ### Check: init_loss
                - **Status**: FAIL
            """)
            )
            assert _next_attempt_id(tmpdir, "test/config") == 2

    def test_increments_to_3_after_two_attempts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "06-sanity-ladder-log.md"
            log_path.write_text(
                textwrap.dedent("""\
                ## Sanity Ladder -- Runtime Checks
                **Attempt**: 1
                test/config

                ## Sanity Ladder -- Runtime Checks
                **Attempt**: 2
                test/config
            """)
            )
            assert _next_attempt_id(tmpdir, "test/config") == 3

    def test_render_markdown_includes_attempt_id(self):
        result = {
            "checks": [{"name": "init_loss", "status": "PASS", "diagnostic": "ok"}],
            "verdict": "PROCEED",
            "dataset_executability": "PASS",
            "fallback_trigger": False,
            "failed_command": None,
            "traceback_excerpt": "",
            "failure_pattern_interpretation": "All passed.",
        }
        md = render_markdown(result, None, "test/config", "regression", None)
        assert "**Attempt**: 1" in md

    def test_render_markdown_supersedes_note_on_second_attempt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "06-sanity-ladder-log.md"
            log_path.write_text("**Attempt**: 1\ntest/config\n")
            result = {
                "checks": [{"name": "init_loss", "status": "PASS", "diagnostic": "ok"}],
                "verdict": "PROCEED",
                "dataset_executability": "PASS",
                "fallback_trigger": False,
                "failed_command": None,
                "traceback_excerpt": "",
                "failure_pattern_interpretation": "All passed.",
            }
            md = render_markdown(result, tmpdir, "test/config", "regression", None)
            assert "**Attempt**: 2" in md
            assert "Supersedes" in md

    def test_render_markdown_counts_investigate_checks(self):
        result = {
            "checks": [
                {"name": "overfit_batch", "status": INVESTIGATE, "diagnostic": "nuanced"}
            ],
            "verdict": INVESTIGATE,
            "dataset_executability": "PASS",
            "fallback_trigger": False,
            "failed_command": None,
            "traceback_excerpt": "",
            "failure_pattern_interpretation": "Inspect nuanced diagnostics.",
        }
        md = render_markdown(result, None, "test/config", "regression", None)
        assert "**Checks investigate**: 1/1" in md


class TestCompactResults:
    def test_compact_ladder_result_summarizes_nested_metrics(self):
        check = CheckResult(
            name="gradient_flow",
            status=PASS,
            diagnostic="ok",
            metrics={
                "batch_dimension_leak": False,
                "batch_dimension_leak_details": [
                    {"key": "features", "sample_idx": 0, "grad_norm": 1.0},
                    {"key": "features", "sample_idx": 1, "grad_norm": 0.0},
                ],
                "gradient_norms": {
                    "layer1.weight": 2.0,
                    "layer1.bias": 0.5,
                },
                "dead_layers": ["layer2.weight"],
            },
        )

        result = ladder_result(
            checks=[check],
            experiment_config="test/config",
            task_type="regression",
            dataset_candidate="dummy",
            vault_dir=None,
            write_log=False,
            compact=True,
        )

        metrics = result["checks"][0]["metrics"]
        assert metrics["batch_dimension_leak"] is False
        assert metrics["batch_dimension_leak_details_count"] == 2
        assert metrics["gradient_parameter_tensors"] == 2
        assert metrics["gradient_norm_max"] == 2.0
        assert metrics["gradient_norm_min_nonzero"] == 0.5
        assert metrics["dead_layers_count"] == 1
        assert metrics["dead_layers_preview"] == ["layer2.weight"]

    def test_ladder_result_writes_jsonl_sidecar_when_write_log_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            check = CheckResult(
                name="init_loss",
                status=PASS,
                diagnostic="ok",
                metrics={"first_loss": 0.5},
            )
            result = ladder_result(
                checks=[check],
                experiment_config="test/config",
                task_type="regression",
                dataset_candidate="dummy",
                vault_dir=tmpdir,
                write_log=True,
                compact=True,
            )

            jsonl_path = Path(tmpdir) / "06-sanity-ladder-log.jsonl"
            assert jsonl_path.exists()
            lines = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
            assert len(lines) == 1
            assert lines[0]["verdict"] == PASS
            assert lines[0]["passed_checks"] == ["init_loss"]
            assert Path(result["jsonl_appended_path"]).resolve() == jsonl_path.resolve()


# ---------------------------------------------------------------------------
# 4b. Micro-batch overfit classification
# ---------------------------------------------------------------------------


class TestMicroOverfitClassification:
    @mock.patch("PICID_sanity_core.manual_overfit_single_batch")
    def test_regression_passes_on_clean_memorization(self, overfit_mock):
        overfit_mock.return_value = {
            "skipped": False,
            "timed_out": False,
            "initial_loss": 1.0,
            "best_loss": 0.01,
            "final_loss": 0.02,
            "losses": [1.0, 0.2, 0.02],
            "completed_steps": 40,
            "max_steps": 400,
            "elapsed_sec": 2.0,
            "micro_batch_size": 4,
            "full_batch_size": 32,
            "final_mae": 0.01,
            "final_mae_ratio": 0.02,
            "final_max_abs_error": 0.03,
            "final_max_abs_error_ratio": 0.05,
            "prediction_preview": [0.1, 0.2],
            "target_preview": [0.1, 0.2],
        }

        check = check_overfit_batch(
            repo_root=Path("/repo"),
            experiment_config="demo/exp",
            task_type="regression",
            extra_overrides=[],
            timeout_sec=60,
        )

        assert check.status == PASS
        assert check.metrics["micro_batch_size"] == 4
        assert check.metrics["best_reduction_fraction"] == pytest.approx(0.99)

    @mock.patch("PICID_sanity_core.manual_overfit_single_batch")
    def test_regression_can_return_investigate(self, overfit_mock):
        overfit_mock.return_value = {
            "skipped": False,
            "timed_out": False,
            "initial_loss": 1.0,
            "best_loss": 0.06,
            "final_loss": 0.12,
            "losses": [1.0, 0.4, 0.12],
            "completed_steps": 120,
            "max_steps": 400,
            "elapsed_sec": 5.0,
            "micro_batch_size": 4,
            "full_batch_size": 32,
            "final_mae": 0.08,
            "final_mae_ratio": 0.12,
            "final_max_abs_error": 0.18,
            "final_max_abs_error_ratio": 0.30,
            "prediction_preview": [0.2, 0.3],
            "target_preview": [0.1, 0.4],
        }

        check = check_overfit_batch(
            repo_root=Path("/repo"),
            experiment_config="demo/exp",
            task_type="regression",
            extra_overrides=[],
            timeout_sec=60,
        )

        assert check.status == INVESTIGATE

    @mock.patch("PICID_sanity_core.manual_overfit_single_batch")
    def test_classification_fails_when_far_from_memorization(self, overfit_mock):
        overfit_mock.return_value = {
            "skipped": False,
            "timed_out": False,
            "initial_loss": 1.2,
            "best_loss": 0.8,
            "final_loss": 0.9,
            "losses": [1.2, 1.0, 0.9],
            "completed_steps": 150,
            "max_steps": 400,
            "elapsed_sec": 6.0,
            "micro_batch_size": 4,
            "full_batch_size": 32,
            "final_accuracy": 0.5,
            "prediction_preview": [1, 0, 1, 0],
            "target_preview": [1, 1, 0, 0],
        }

        check = check_overfit_batch(
            repo_root=Path("/repo"),
            experiment_config="demo/exp",
            task_type="classification",
            extra_overrides=[],
            timeout_sec=60,
        )

        assert check.status == FAIL


# ---------------------------------------------------------------------------
# 5. Classify dataset failure
# ---------------------------------------------------------------------------


class TestClassifyDatasetFailure:
    def test_file_not_found(self):
        assert (
            classify_dataset_failure("FileNotFoundError: /data/raw")
            == DATASET_UNAVAILABLE
        )

    def test_no_such_file(self):
        assert (
            classify_dataset_failure("No such file or directory") == DATASET_UNAVAILABLE
        )

    def test_memory_error(self):
        assert (
            classify_dataset_failure("MemoryError during preprocessing")
            == DATASET_EXECUTION_FAILED
        )

    def test_load_data_failure(self):
        assert (
            classify_dataset_failure("Error in load_data()") == DATASET_EXECUTION_FAILED
        )

    def test_dataset_contract_mismatch_failure(self):
        assert (
            classify_dataset_failure(
                "ValueError: Target and context sequencers must have the same length"
            )
            == DATASET_EXECUTION_FAILED
        )

    def test_column_map_preprocessing_failure(self):
        assert classify_dataset_failure("KeyError: 'column_map'") == DATASET_EXECUTION_FAILED

    def test_normal_error_not_classified(self):
        assert classify_dataset_failure("RuntimeError: shape mismatch") is None

    def test_empty_text(self):
        assert classify_dataset_failure("") is None


# ---------------------------------------------------------------------------
# 6. Base overrides
# ---------------------------------------------------------------------------


class TestBaseOverrides:
    def test_adds_csv_logger_by_default(self):
        result = base_overrides([])
        assert "logger=csv" in result

    def test_does_not_duplicate_logger(self):
        result = base_overrides(["logger=wandb"])
        assert "logger=csv" not in result
        assert "logger=wandb" in result

    def test_does_not_duplicate_plus_logger(self):
        result = base_overrides(["+logger=csv"])
        assert result.count("logger=csv") == 0  # only +logger=csv
        assert "+logger=csv" in result


# ---------------------------------------------------------------------------
# 7. CheckResult dataclass
# ---------------------------------------------------------------------------


class TestCheckResult:
    def test_to_dict_roundtrip(self):
        check = CheckResult(
            name="init_loss",
            status=PASS,
            diagnostic="ok",
            metrics={"loss": 0.5},
        )
        d = check.to_dict()
        assert d["name"] == "init_loss"
        assert d["status"] == PASS
        assert d["metrics"]["loss"] == 0.5

    def test_default_dataset_executability(self):
        check = CheckResult(name="test", status=PASS)
        assert check.dataset_executability == PASS
        assert check.fallback_trigger is False


# ---------------------------------------------------------------------------
# 8. Traceback excerpt
# ---------------------------------------------------------------------------


class TestTracebackExcerpt:
    def test_extracts_traceback(self):
        text = "INFO: starting\nTraceback (most recent call last):\n  File 'x.py'\nRuntimeError: boom"
        result = traceback_excerpt(text)
        assert "Traceback" in result
        assert "RuntimeError" in result

    def test_empty_input(self):
        assert traceback_excerpt("") == ""

    def test_no_traceback_returns_tail(self):
        text = "some log output\nanother line"
        result = traceback_excerpt(text)
        assert "some log" in result


# ---------------------------------------------------------------------------
# 9. Default timeout + _resolve_timeout
# ---------------------------------------------------------------------------


class TestResolveTimeout:
    def test_default_is_600(self):
        assert DEFAULT_TIMEOUT_SEC == 600

    def test_none_resolves_to_default(self):
        assert _resolve_timeout(None) == 600

    def test_zero_resolves_to_default(self):
        assert _resolve_timeout(0) == 600

    def test_negative_resolves_to_default(self):
        assert _resolve_timeout(-5) == 600

    def test_positive_passes_through(self):
        assert _resolve_timeout(60) == 60
        assert _resolve_timeout(300) == 300


# ---------------------------------------------------------------------------
# 10. time_limit SIGALRM context manager
# ---------------------------------------------------------------------------


class TestTimeLimit:
    def test_no_op_when_seconds_is_none(self):
        with time_limit(None):
            time.sleep(0.05)

    def test_no_op_when_seconds_is_zero(self):
        with time_limit(0):
            time.sleep(0.05)

    def test_completes_under_budget(self):
        with time_limit(5):
            time.sleep(0.05)

    def test_fires_when_busy_loop_exceeds_budget(self):
        start = time.monotonic()
        with pytest.raises(CheckTimeoutError):
            with time_limit(1):
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    pass
        elapsed = time.monotonic() - start
        assert elapsed < 3, f"alarm should have interrupted within ~1s, took {elapsed:.2f}s"

    def test_alarm_cleared_after_exit(self):
        import signal as _signal

        with time_limit(10):
            pass
        # If the alarm wasn't cleared, a leftover would fire later. Verify
        # signal.alarm(0) returns 0 (= no pending alarm).
        assert _signal.alarm(0) == 0


# ---------------------------------------------------------------------------
# 11. emit_heartbeat stderr format
# ---------------------------------------------------------------------------


class TestEmitHeartbeat:
    def test_writes_to_stderr_with_check_field(self, capsys):
        emit_heartbeat("init_loss", step=5, loss=2.341)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.startswith("[heartbeat] check=init_loss")
        assert "step=5" in captured.err
        assert "loss=2.341" in captured.err

    def test_no_fields_still_emits_check(self, capsys):
        emit_heartbeat("gradient_flow")
        captured = capsys.readouterr()
        assert captured.err.strip() == "[heartbeat] check=gradient_flow"

    def test_float_formatting_compact(self, capsys):
        emit_heartbeat("overfit_batch", loss=0.000123456789)
        captured = capsys.readouterr()
        # 4 significant digits via %.4g
        assert "loss=0.0001235" in captured.err or "loss=1.235e-04" in captured.err


# ---------------------------------------------------------------------------
# 12. Stack cache reuse
# ---------------------------------------------------------------------------


class TestStackCache:
    def setup_method(self):
        clear_stack_cache()

    def teardown_method(self):
        clear_stack_cache()

    def test_cache_key_is_order_invariant(self):
        a = _cache_key("exp/x", ["b=1", "a=2"])
        b = _cache_key("exp/x", ["a=2", "b=1"])
        assert a == b

    def test_cache_key_differs_by_experiment(self):
        a = _cache_key("exp/x", ["a=1"])
        b = _cache_key("exp/y", ["a=1"])
        assert a != b

    def test_cache_key_differs_by_overrides(self):
        a = _cache_key("exp/x", ["a=1"])
        b = _cache_key("exp/x", ["a=2"])
        assert a != b

    def test_cache_hit_returns_same_object(self):
        from sanity_framework import get_or_build_framework_stack

        sentinel = {"cfg": object(), "model": object()}
        key = _cache_key("exp/test", [])
        _STACK_CACHE[key] = sentinel

        with mock.patch("sanity_framework.build_framework_stack") as builder:
            result = get_or_build_framework_stack(
                Path("/repo"),
                "exp/test",
                [],
                ensure_project_root_fn=lambda x: Path(x),
                base_overrides_fn=lambda o: list(o or []),
            )
        assert result is sentinel
        builder.assert_not_called()

    def test_cache_miss_invokes_builder(self):
        from sanity_framework import get_or_build_framework_stack

        built = {"cfg": object(), "model": object()}
        with mock.patch(
            "sanity_framework.build_framework_stack", return_value=built
        ) as builder:
            result = get_or_build_framework_stack(
                Path("/repo"),
                "exp/missing",
                [],
                ensure_project_root_fn=lambda x: Path(x),
                base_overrides_fn=lambda o: list(o or []),
            )
        assert result is built
        builder.assert_called_once()
        # Subsequent call must hit the cache, not rebuild.
        with mock.patch("sanity_framework.build_framework_stack") as builder2:
            again = get_or_build_framework_stack(
                Path("/repo"),
                "exp/missing",
                [],
                ensure_project_root_fn=lambda x: Path(x),
                base_overrides_fn=lambda o: list(o or []),
            )
        assert again is built
        builder2.assert_not_called()


# ---------------------------------------------------------------------------
# 13. Ladder shape (3 default checks; subset_convergence opt-in)
# ---------------------------------------------------------------------------


class TestLadderShape:
    def test_default_ladder_excludes_subset_convergence(self):
        from PICID_sanity import common

        payload = {"experiment_config": "x", "repo_root": "."}
        values = common(payload)
        assert values["include_subset_convergence"] is False

    def test_opt_in_subset_convergence_flag(self):
        from PICID_sanity import common

        payload = {
            "experiment_config": "x",
            "repo_root": ".",
            "include_subset_convergence": True,
        }
        values = common(payload)
        assert values["include_subset_convergence"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
