"""Reusable sanity checks for the PICID paper-validation workflow.

This module now focuses on check orchestration and re-exports shared helpers from
smaller tool-side modules to keep the implementation maintainable.
"""

from __future__ import annotations

import math
import statistics
import traceback
from pathlib import Path
from typing import Any

from sanity_base import (
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
    base_overrides,
    build_picid_command,
    classify_dataset_failure,
    classify_tool_invocation_failure,
    emit_heartbeat,
    ensure_project_root,
    failed_command_result,
    is_bad_loss,
    json_default,
    loss_epoch_series,
    loss_step_series,
    read_metric_series,
    run_picid,
    run_dir_from_metrics,
    time_limit,
    traceback_excerpt,
)
from sanity_framework import (
    _clone_value,
    _first_tensor_output,
    _mark_inputs_require_grad,
    _optimizer_from_config,
    build_framework_stack,
    clear_stack_cache,
    compose_cfg,
    get_or_build_framework_stack,
    manual_overfit_single_batch,
    manual_train_loss,
    tool_failure_result,
)
from sanity_reporting import (
    _next_attempt_id,
    append_report,
    interpret_failure,
    ladder_result,
    render_markdown,
    summarize_verdict,
)


def _resolve_timeout(timeout_sec: int | None) -> int:
    return int(timeout_sec) if timeout_sec and timeout_sec > 0 else DEFAULT_TIMEOUT_SEC


def _timeout_check(name: str, elapsed: float, budget: int) -> CheckResult:
    return CheckResult(
        name=name,
        status=PRECHECK_TIMEOUT,
        diagnostic=(
            f"In-process check exceeded {budget}s budget after {elapsed:.1f}s. "
            "If the model is genuinely slow per step, increase timeout_sec for this check."
        ),
        metrics={"elapsed_sec": elapsed, "timeout_sec": budget},
    )


def _loss_reduction(initial: float | None, value: float | None) -> float | None:
    if initial in (None, 0.0) or value is None:
        return None
    return (float(initial) - float(value)) / max(abs(float(initial)), 1e-12)


def _micro_overfit_verdict(task_type: str, run: dict[str, Any]) -> tuple[str, str]:
    initial = run.get("initial_loss")
    final = run.get("final_loss")
    best = run.get("best_loss", final)
    final_reduction = _loss_reduction(initial, final)
    best_reduction = _loss_reduction(initial, best)

    if run.get("timed_out") and (best_reduction is None or best_reduction < 0.90):
        return (
            PRECHECK_TIMEOUT,
            (
                f"Micro-batch memorization timed out after {run.get('completed_steps')}/"
                f"{run.get('max_steps')} steps before showing strong overfit signal. "
                "Increase timeout_sec if the step rate is genuinely slow."
            ),
        )

    if task_type == "classification":
        accuracy = run.get("final_accuracy")
        if accuracy is not None and accuracy >= 0.999 and final is not None:
            if final <= max(0.05, float(initial) * 0.10):
                return (
                    PASS,
                    "Micro-batch classification probe reached perfect accuracy with near-zero training loss.",
                )
        if accuracy is not None and (
            accuracy >= 0.85 or (best_reduction is not None and best_reduction >= 0.90)
        ):
            return (
                INVESTIGATE,
                "Micro-batch classification probe shows strong but incomplete memorization.",
            )
        return (
            FAIL,
            "Micro-batch classification probe did not get close to full memorization.",
        )

    mae_ratio = run.get("final_mae_ratio")
    max_abs_ratio = run.get("final_max_abs_error_ratio")
    if final is not None and initial is not None:
        if final <= max(1e-4, float(initial) * 0.05):
            return (
                PASS,
                "Micro-batch regression probe drove the training loss close to zero.",
            )
    if mae_ratio is not None and max_abs_ratio is not None:
        if mae_ratio <= 0.05 and max_abs_ratio <= 0.20:
            return (
                PASS,
                "Micro-batch regression probe predictions align closely with the targets.",
            )
        if mae_ratio <= 0.20 or (best_reduction is not None and best_reduction >= 0.90):
            return (
                INVESTIGATE,
                "Micro-batch regression probe shows strong but not yet clean memorization.",
            )
    elif best_reduction is not None and best_reduction >= 0.90:
        return (
            INVESTIGATE,
            "Micro-batch probe reduced loss strongly, but prediction-target alignment metrics were incomplete.",
        )
    return (
        FAIL,
        "Micro-batch regression probe stayed far from target-level memorization.",
    )


def check_init_loss(
    repo_root: Path,
    experiment_config: str,
    task_type: str,
    num_classes: int | None,
    expected_init_loss: float | str | None,
    extra_overrides: list[str],
    timeout_sec: int | None,
) -> CheckResult:
    """In-process loss-at-init: build (or reuse) the framework stack, take one
    batch, run a single forward pass, validate the loss. No CLI subprocess, no
    metrics.csv parsing — everything happens in this Python process so it shares
    the cached stack with the other ladder checks."""
    import time as _time

    import torch

    budget = _resolve_timeout(timeout_sec)
    start = _time.monotonic()
    try:
        with time_limit(budget):
            emit_heartbeat("init_loss", phase="start", budget=budget)
            stack = get_or_build_framework_stack(
                repo_root,
                experiment_config,
                extra_overrides,
                ensure_project_root,
                base_overrides,
            )
            model = stack["model"]
            cfg = stack["cfg"]
            if model.__class__.__name__ == "FitPredictWrapperLightningModule" or (
                not cfg.task_definition.get("requires_training", True)
            ):
                return CheckResult(
                    name="init_loss",
                    status=SKIPPED,
                    diagnostic="fit-predict model, no gradient-based training",
                )
            model.train()
            batch = _clone_value(next(iter(stack["datamodule"].train_dataloader())))
            with torch.no_grad():
                model_out = model.model_step(batch, stage="train")
            loss_tensor = model_out["loss"]
            if torch.is_tensor(loss_tensor) and loss_tensor.ndim > 0:
                loss_tensor = loss_tensor.mean()
            first = float(loss_tensor.detach().cpu())
            emit_heartbeat("init_loss", phase="forward_done", loss=first)
    except CheckTimeoutError:
        return _timeout_check("init_loss", _time.monotonic() - start, budget)
    except Exception:
        tb = traceback.format_exc()
        tool_failure = classify_tool_invocation_failure(tb)
        return CheckResult(
            **tool_failure_result("init_loss", tb, tool_flag=tool_failure)
        )

    metrics: dict[str, Any] = {"first_loss": first, "elapsed_sec": _time.monotonic() - start}
    status = PASS
    diagnostic = "Initial loss is finite and non-zero."
    if is_bad_loss(first):
        status = FAIL
        diagnostic = "Initial loss is NaN, Inf, or exactly zero."
    elif task_type == "classification" and num_classes:
        expected = math.log(num_classes)
        rel_error = abs(first - expected) / max(abs(expected), 1e-12)
        metrics.update({"expected": expected, "relative_error": rel_error})
        if rel_error > 0.20:
            status = FAIL
            diagnostic = "Classification init loss is outside 20% of ln(num_classes)."
    elif expected_init_loss not in (None, "auto"):
        expected = float(expected_init_loss)
        rel_error = abs(first - expected) / max(abs(expected), 1e-12)
        metrics.update({"expected": expected, "relative_error": rel_error})
        if rel_error > 0.50:
            status = FAIL
            diagnostic = (
                "Regression init loss is outside 50% of the blueprint expectation."
            )

    return CheckResult(
        name="init_loss",
        status=status,
        diagnostic=diagnostic,
        metrics=metrics,
    )


def check_overfit_batch(
    repo_root: Path,
    experiment_config: str,
    task_type: str,
    extra_overrides: list[str],
    timeout_sec: int | None,
    micro_batch_size: int = 4,
    max_steps: int = 400,
) -> CheckResult:
    import time as _time

    budget = _resolve_timeout(timeout_sec)
    start = _time.monotonic()
    try:
        with time_limit(budget):
            emit_heartbeat("overfit_batch", phase="start", budget=budget)
            run = manual_overfit_single_batch(
                repo_root,
                experiment_config,
                extra_overrides,
                ensure_project_root,
                base_overrides,
                task_type=task_type,
                micro_batch_size=micro_batch_size,
                max_steps=max_steps,
            )
    except CheckTimeoutError:
        return _timeout_check("overfit_batch", _time.monotonic() - start, budget)
    except Exception:
        tb = traceback.format_exc()
        tool_failure = classify_tool_invocation_failure(tb)
        return CheckResult(
            **tool_failure_result("overfit_batch", tb, tool_flag=tool_failure)
        )

    if run.get("skipped"):
        return CheckResult(
            name="overfit_batch",
            status=SKIPPED,
            diagnostic=run.get("reason") or "micro-batch overfit check skipped",
            metrics=run,
        )

    losses = run.get("losses", [])
    if len(losses) < 2:
        if run.get("timed_out"):
            return _timeout_check("overfit_batch", run.get("elapsed_sec", 0.0), budget)
        return CheckResult(
            name="overfit_batch",
            status=FAIL,
            diagnostic="Need at least two loss values to evaluate overfit behavior.",
            metrics={"losses_found": len(losses)},
        )

    initial = float(run["initial_loss"])
    final = float(run["final_loss"])
    best = float(run.get("best_loss", final))
    reduction = _loss_reduction(initial, final)
    best_reduction = _loss_reduction(initial, best)
    status, diagnostic = _micro_overfit_verdict(task_type, run)
    return CheckResult(
        name="overfit_batch",
        status=status,
        diagnostic=diagnostic,
        metrics={
            "initial_loss": initial,
            "final_loss": final,
            "best_loss": best,
            "reduction_fraction": reduction,
            "best_reduction_fraction": best_reduction,
            "num_loss_points": len(losses),
            "completed_steps": run.get("completed_steps"),
            "max_steps": run.get("max_steps"),
            "elapsed_sec": run.get("elapsed_sec"),
            "timed_out": bool(run.get("timed_out")),
            "micro_batch_size": run.get("micro_batch_size"),
            "full_batch_size": run.get("full_batch_size"),
            "final_accuracy": run.get("final_accuracy"),
            "final_mae": run.get("final_mae"),
            "final_mae_ratio": run.get("final_mae_ratio"),
            "final_max_abs_error": run.get("final_max_abs_error"),
            "final_max_abs_error_ratio": run.get("final_max_abs_error_ratio"),
            "prediction_preview": run.get("prediction_preview"),
            "target_preview": run.get("target_preview"),
        },
    )


def check_subset_convergence(
    repo_root: Path,
    experiment_config: str,
    extra_overrides: list[str],
    timeout_sec: int | None,
) -> CheckResult:
    """Opt-in only. Not part of the default ladder.

    Runs a real CLI training subprocess (30 epochs on 10% of data) and inspects
    the loss series for monotone decrease + tail stability. The ladder caller
    must explicitly set `include_subset_convergence: true` to run this — it is
    expensive (minutes), it duplicates signal already provided by `overfit_batch`,
    and the tail-stability check is really a hyperparameter probe rather than a
    model-correctness probe.
    """
    overrides = base_overrides(
        [
            "trainer.max_epochs=30",
            "trainer.limit_train_batches=0.1",
            "trainer.limit_val_batches=0.1",
            "trainer.limit_test_batches=0",
            "enable_progress_bar=false",
            *extra_overrides,
        ]
    )
    result = run_picid(repo_root, experiment_config, overrides, timeout_sec)
    if result.returncode != 0:
        return failed_command_result("subset_convergence", result)

    losses = loss_epoch_series(result.metrics_path) or loss_step_series(
        result.metrics_path
    )
    if len(losses) < 10:
        return CheckResult(
            name="subset_convergence",
            status=FAIL,
            diagnostic="Need at least ten loss values to evaluate convergence trend.",
            run_dirs=[result.run_dir] if result.run_dir else [],
            command=result.command,
            metrics={"metrics_path": result.metrics_path, "losses_found": len(losses)},
        )

    first_avg = statistics.fmean(losses[:5])
    last_avg = statistics.fmean(losses[-5:])
    tail = losses[-10:]
    tail_mean = statistics.fmean(tail)
    tail_std = statistics.pstdev(tail) if len(tail) > 1 else 0.0
    status = (
        PASS
        if last_avg < first_avg and tail_std < 0.5 * max(abs(tail_mean), 1e-12)
        else FAIL
    )
    diagnostic = (
        "Subset training loss decreases and the tail is not wildly oscillatory."
        if status == PASS
        else "Subset training loss does not show a stable decreasing trend."
    )
    return CheckResult(
        name="subset_convergence",
        status=status,
        diagnostic=diagnostic,
        metrics={
            "epochs_1_5_avg_loss": first_avg,
            "last_5_avg_loss": last_avg,
            "last_10_std": tail_std,
            "num_loss_points": len(losses),
            "metrics_path": result.metrics_path,
        },
        run_dirs=[result.run_dir] if result.run_dir else [],
        command=result.command,
    )


def preflight_config(
    repo_root: Path, experiment_config: str, extra_overrides: list[str]
) -> CheckResult:
    try:
        compose_cfg(repo_root, experiment_config, extra_overrides, base_overrides)
        return CheckResult(
            name="config_preflight",
            status=PASS,
            diagnostic="Hydra config composition succeeded.",
        )
    except Exception:
        tb = traceback.format_exc()
        return CheckResult(
            name="config_preflight",
            status=TOOL_INVOCATION_FAILURE,
            diagnostic="Hydra config composition failed. This is a config/tool invocation error, not a model failure. Fix the experiment config before retrying sanity checks.",
            traceback_excerpt=traceback_excerpt(tb),
        )


def check_zero_input(
    repo_root: Path,
    experiment_config: str,
    extra_overrides: list[str],
    zero_keys: list[str] | None,
    timeout_sec: int | None = None,
) -> CheckResult:
    """On-demand only. Not part of the default 3-check ladder.

    Compares manual training on real vs zeroed inputs to detect data-pipeline
    bugs. Skip unless the pipeline is suspect — the ladder's other checks already
    catch most model-wiring bugs cheaper.
    """
    import time as _time

    budget = _resolve_timeout(timeout_sec)
    start = _time.monotonic()
    try:
        with time_limit(budget):
            emit_heartbeat("zero_input", phase="real_start", budget=budget)
            real = manual_train_loss(
                repo_root,
                experiment_config,
                extra_overrides,
                False,
                zero_keys,
                ensure_project_root,
                base_overrides,
            )
            emit_heartbeat("zero_input", phase="zero_start")
            zero = manual_train_loss(
                repo_root,
                experiment_config,
                extra_overrides,
                True,
                zero_keys,
                ensure_project_root,
                base_overrides,
            )
    except CheckTimeoutError:
        return _timeout_check("zero_input", _time.monotonic() - start, budget)
    except Exception:
        tb = traceback.format_exc()
        tool_failure = classify_tool_invocation_failure(tb)
        return CheckResult(
            **tool_failure_result("zero_input", tb, tool_flag=tool_failure)
        )

    if real.get("skipped") or zero.get("skipped"):
        return CheckResult(
            name="zero_input",
            status=SKIPPED,
            diagnostic=real.get("reason")
            or zero.get("reason")
            or "zero-input check skipped",
            metrics={"real": real, "zero": zero},
        )

    real_final = real.get("final_loss")
    zero_final = zero.get("final_loss")
    if real_final is None or zero_final is None:
        status = FAIL
        diagnostic = "No final loss was produced by the manual training loop."
    else:
        status = PASS if real_final < zero_final * 0.95 else FAIL
        diagnostic = (
            "Real inputs train better than zeroed inputs."
            if status == PASS
            else "Real inputs do not clearly outperform zeroed inputs."
        )

    return CheckResult(
        name="zero_input",
        status=status,
        diagnostic=diagnostic,
        metrics={
            "real_initial_loss": real.get("initial_loss"),
            "real_final_loss": real_final,
            "zero_initial_loss": zero.get("initial_loss"),
            "zero_final_loss": zero_final,
            "delta_real_minus_zero": None
            if real_final is None or zero_final is None
            else real_final - zero_final,
        },
    )


def check_gradient_flow(
    repo_root: Path,
    experiment_config: str,
    extra_overrides: list[str],
    zero_keys: list[str] | None,
    timeout_sec: int | None = None,
) -> CheckResult:
    import time as _time

    import torch

    budget = _resolve_timeout(timeout_sec)
    start = _time.monotonic()
    try:
        with time_limit(budget):
            emit_heartbeat("gradient_flow", phase="start", budget=budget)
            stack = get_or_build_framework_stack(
                repo_root,
                experiment_config,
                extra_overrides,
                ensure_project_root,
                base_overrides,
            )
            model = stack["model"]
            if model.__class__.__name__ == "FitPredictWrapperLightningModule":
                return CheckResult(
                    name="gradient_flow",
                    status=SKIPPED,
                    diagnostic="fit-predict model, no gradient-based training",
                )
            model.train()
            batch = next(iter(stack["datamodule"].train_dataloader()))
            model.zero_grad(set_to_none=True)
            model_out = model.model_step(_clone_value(batch), stage="train")
            loss = model_out["loss"]
            if torch.is_tensor(loss) and loss.ndim > 0:
                loss = loss.mean()
            loss.backward()
            emit_heartbeat("gradient_flow", phase="backward_done")
    except CheckTimeoutError:
        return _timeout_check("gradient_flow", _time.monotonic() - start, budget)
    except Exception:
        tb = traceback.format_exc()
        tool_failure = classify_tool_invocation_failure(tb)
        return CheckResult(
            **tool_failure_result("gradient_flow", tb, tool_flag=tool_failure)
        )

    dead: list[str] = []
    exploding: list[str] = []
    vanishing: list[str] = []
    norms: dict[str, float] = {}
    trainable_parameter_count = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        trainable_parameter_count += 1
        if param.grad is None:
            dead.append(name)
            continue
        norm = float(param.grad.detach().norm().cpu())
        norms[name] = norm
        if norm == 0.0:
            dead.append(name)
        elif norm > 1000:
            exploding.append(name)
        elif norm < 1e-7:
            vanishing.append(name)

    leak = None
    leak_details: list[dict[str, Any]] = []
    try:
        model.zero_grad(set_to_none=True)
        leak_batch = _clone_value(batch)
        selected_zero_keys = set(zero_keys) if zero_keys else None
        marked = _mark_inputs_require_grad(leak_batch, selected_zero_keys)
        model_out = model.model_step(leak_batch, stage="train")
        target = _first_tensor_output(model_out)
        if target is not None:
            sample = target[0]
            while hasattr(sample, "ndim") and sample.ndim > 0:
                sample = sample.sum()
            target = sample
            target.backward()
            for key_name, tensor in marked:
                grad = tensor.grad
                if grad is None:
                    continue
                sample_norms = (
                    grad.reshape(grad.shape[0], -1).norm(dim=1).detach().cpu().tolist()
                )
                leak_details.extend(
                    {"key": key_name, "sample_idx": idx, "grad_norm": value}
                    for idx, value in enumerate(sample_norms)
                )
                if any(value > 1e-8 for value in sample_norms[1:]):
                    leak = True
            if leak is None:
                leak = False
    except Exception:
        leak = None

    status = (
        PASS
        if trainable_parameter_count > 0
        and not dead
        and not exploding
        and not vanishing
        and leak is not True
        else FAIL
    )
    diagnostic = (
        "No dead/exploding/vanishing gradients or batch leak detected."
        if status == PASS
        else "Gradient-flow audit found dead/exploding/vanishing gradients or batch leakage."
    )
    return CheckResult(
        name="gradient_flow",
        status=status,
        diagnostic=diagnostic,
        metrics={
            "dead_layers": dead,
            "exploding_layers": exploding,
            "vanishing_layers": vanishing,
            "batch_dimension_leak": leak,
            "batch_dimension_leak_details": leak_details,
            "gradient_norms": norms,
            "trainable_parameter_count": trainable_parameter_count,
        },
    )


def skip_gradient_check(name: str) -> CheckResult:
    return CheckResult(
        name=name,
        status=SKIPPED,
        diagnostic="fit-predict model, no gradient-based training",
    )
