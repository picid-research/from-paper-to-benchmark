import json

import pytest

import validate_paper_workflow as workflow


def _write_evaluation_sidecar(vault_dir, **overrides):
    payload = {
        "status": "VALIDATED",
        "artifact_status": "COMPLETE",
        "technical_status": "PASS",
        "scientific_status": "VALIDATED",
        "blocking": False,
        "comparison_mode": "framework_validation",
        "dataset_used": "nb14",
        "paper_dataset": "NASA random use battery",
        "hypothesis_status": "PRE_REGISTERED",
        "paper_claims_summary": {
            "confirmed": 1,
            "contradicted": 0,
            "dataset_dependent": 0,
            "unassessable": 0,
        },
        "timestamp": "2026-04-26T00:00:00+00:00",
    }
    payload.update(overrides)
    path = vault_dir / "08-evaluation-report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_claims_disputed_evaluation_is_completed_observation(tmp_path):
    _write_evaluation_sidecar(
        tmp_path,
        status="INVESTIGATE_CLAIMS_DISPUTED",
        scientific_status="INVESTIGATE_CLAIMS_DISPUTED",
        paper_claims_summary={
            "confirmed": 1,
            "contradicted": 1,
            "dataset_dependent": 0,
            "unassessable": 0,
            "unassessable_metric_scale": 0,
            "unassessable_no_overlapping_baseline": 0,
        },
    )

    status, metadata = workflow.infer_phase_status_from_artifacts(
        tmp_path,
        "evaluate_results",
    )

    assert status == "passed"
    assert metadata["latest_status"] == "INVESTIGATE_CLAIMS_DISPUTED"
    assert metadata["technical_status"] == "PASS"


def test_stale_or_metric_ambiguous_evaluation_is_completed_observation(tmp_path):
    _write_evaluation_sidecar(
        tmp_path,
        status="STALE_DATASET_MISMATCH",
        scientific_status="STALE_DATASET_MISMATCH",
        observations=["UNASSESSABLE_METRIC_SCALE"],
    )

    status, metadata = workflow.infer_phase_status_from_artifacts(
        tmp_path,
        "evaluate_results",
    )

    assert status == "passed"
    assert metadata["latest_status"] == "STALE_DATASET_MISMATCH"


def test_implementation_bug_evaluation_remains_retryable_failure(tmp_path):
    _write_evaluation_sidecar(
        tmp_path,
        status="IMPLEMENTATION_BUG",
        technical_status="IMPLEMENTATION_BUG",
        scientific_status="IMPLEMENTATION_BUG",
        blocking=True,
    )

    status, metadata = workflow.infer_phase_status_from_artifacts(
        tmp_path,
        "evaluate_results",
    )

    assert status == "failed"
    assert metadata["reason_code"] == "IMPLEMENTATION_BUG"
    assert metadata["retryable"] is True


def test_batch_fit_timeout_remains_retryable_failure(tmp_path):
    (tmp_path / "batch_fit_check.json").write_text(
        json.dumps(
            [
                {
                    "status": "TIMEOUT",
                    "resolved_config_hash": "abc123",
                    "hardware_id": "cpu",
                    "timeout_sec": 600,
                }
            ]
        ),
        encoding="utf-8",
    )

    status, metadata = workflow.infer_phase_status_from_artifacts(
        tmp_path,
        "check_batch_fit",
    )

    assert status == "failed"
    assert metadata["reason_code"] == "TIMEOUT"
    assert metadata["retryable"] is True
    assert metadata["timeout_sec"] == 600


def test_warn_continue_sanity_phase_passes_with_warning(tmp_path):
    (tmp_path / "06-sanity-ladder-log.jsonl").write_text(
        json.dumps(
            {
                "verdict": "WARN_CONTINUE",
                "attempt": 1,
                "passed_checks": ["config_preflight"],
                "failed_checks": ["overfit_batch"],
                "skipped_checks": [],
                "check_diagnostics": {},
                "dataset_candidate": "nb14",
                "dataset_status": "PASS",
                "fallback_trigger": False,
                "experiment_config": "nb14/prognostics/example",
                "timestamp": "2026-04-26T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    status, metadata = workflow.infer_phase_status_from_artifacts(
        tmp_path,
        "verify_sanity",
    )

    assert status == "passed"
    assert metadata["latest_verdict"] == "WARN_CONTINUE"
    assert metadata["sanity_warning"] is True


def test_batch_fit_oom_is_non_retryable_failure(tmp_path):
    (tmp_path / "batch_fit_check.json").write_text(
        json.dumps([{"status": "OOM", "resolved_config_hash": "abc123"}]),
        encoding="utf-8",
    )

    status, metadata = workflow.infer_phase_status_from_artifacts(
        tmp_path,
        "check_batch_fit",
    )

    assert status == "failed"
    assert metadata["reason_code"] == "OOM"
    assert metadata["retryable"] is False


def test_batch_fit_ok_passes(tmp_path):
    (tmp_path / "batch_fit_check.json").write_text(
        json.dumps([{"status": "OK", "resolved_config_hash": "abc123"}]),
        encoding="utf-8",
    )

    status, metadata = workflow.infer_phase_status_from_artifacts(
        tmp_path,
        "check_batch_fit",
    )

    assert status == "passed"
    assert metadata["latest_status"] == "OK"


def test_claim_summary_parser_keeps_extended_unassessable_counts():
    text = (
        "CONFIRMED 2 / CONTRADICTED 1 / DATASET_DEPENDENT 3 / "
        "UNASSESSABLE 4 / UNASSESSABLE_METRIC_SCALE 5 / "
        "UNASSESSABLE_NO_OVERLAPPING_BASELINE 6"
    )

    summary = workflow._parse_claim_summary(text)

    assert summary == {
        "confirmed": 2,
        "contradicted": 1,
        "dataset_dependent": 3,
        "unassessable": 4,
        "unassessable_metric_scale": 5,
        "unassessable_no_overlapping_baseline": 6,
    }


def _valid_hparams():
    return {
        "optimizer": {"value": "Adam", "source_location": "Table 1", "category": "optimization", "framework_default_available": "yes"},
        "learning_rate": {"value": "1e-3", "source_location": "Table 1", "category": "optimization", "framework_default_available": "yes"},
        "lr_schedule": {"value": "NOT_SPECIFIED", "source_location": "NOT_SPECIFIED", "category": "optimization", "framework_default_available": "yes"},
        "weight_decay": {"value": "NOT_SPECIFIED", "source_location": "NOT_SPECIFIED", "category": "optimization", "framework_default_available": "yes"},
        "grad_clip": {"value": "NOT_SPECIFIED", "source_location": "NOT_SPECIFIED", "category": "optimization", "framework_default_available": "yes"},
        "warmup": {"value": "NOT_SPECIFIED", "source_location": "NOT_SPECIFIED", "category": "optimization", "framework_default_available": "yes"},
        "max_epochs": {"value": "100", "source_location": "Table 1", "category": "training", "framework_default_available": "yes"},
        "batch_size": {"value": "64", "source_location": "Table 1", "category": "training", "framework_default_available": "yes"},
        "training_protocol_notes": {"value": "early stopping on validation loss", "source_location": "Section 4", "category": "training", "framework_default_available": "n/a"},
    }


def test_algorithmic_sidecar_rejects_missing_required_hparam(tmp_path):
    payload = {
        "vault_dir": str(tmp_path),
        "schema_version": 1,
        "algorithms": [],
        "equations": [],
        "architectures": [],
        "losses": [],
        "training_hyperparameters": {
            key: value
            for key, value in _valid_hparams().items()
            if key != "warmup"
        },
        "data_processing": {},
        "reference_implementations": {},
    }

    with pytest.raises(workflow.WorkflowError, match="warmup"):
        workflow.command_write_algorithmic_sidecar(payload)


def test_blueprint_writer_strips_render_only_sections_and_writes_markdown(tmp_path):
    hparams = _valid_hparams()
    hparams["dropout"] = {
        "value": "0.1",
        "source_location": "Table 1",
        "category": "model",
        "framework_default_available": "yes",
    }
    payload = {
        "vault_dir": str(tmp_path),
        "schema_version": 1,
        "paper_dataset": "NASA random use battery",
        "evaluation_targets": ["NB14"],
        "excluded_paper_datasets": [],
        "framework_dataset_used": "nb14",
        "datasets_are_same": True,
        "comparison_mode": "exact_reproduction",
        "fallback_allowed": False,
        "dataset_fallback_candidates": [],
        "experiment_config": "nb14/prognostics/example",
        "task_type": "prognostics",
        "required_new_files": ["configs/experiment/nb14/prognostics/example.yaml"],
        "verification_protocol": {"metrics": ["test/rmse_denormalized"]},
        "paper_hyperparameters": hparams,
        "dataset_contract": {"dataset": "nb14"},
        "model_io_contract": {"input": "features"},
        "split_contract": {"split": "paper"},
        "markdown_sections": {"integration_summary": "Rendered from the validated payload."},
    }

    result = workflow.command_write_blueprint_sidecar(payload)
    sidecar = json.loads((tmp_path / "04-implementation-blueprint.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "04-implementation-blueprint.md").read_text(encoding="utf-8")

    assert result["status"] == "WROTE"
    assert "markdown_sections" not in sidecar
    assert sidecar["paper_hyperparameters"]["warmup"]["value"] == "NOT_SPECIFIED"
    assert sidecar["paper_hyperparameters"]["dropout"]["value"] == "0.1"
    assert "Rendered from the validated payload." in markdown
    assert "| dropout | 0.1 | Table 1 | model | yes |" in markdown
