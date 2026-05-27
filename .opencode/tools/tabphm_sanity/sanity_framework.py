from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from sanity_base import (
    CheckTimeoutError,
    classify_dataset_failure,
    emit_heartbeat,
    traceback_excerpt,
)


_STACK_CACHE: dict[tuple, dict[str, Any]] = {}


def _cache_key(experiment_config: str, extra_overrides: list[str] | None) -> tuple:
    return (experiment_config, tuple(sorted(extra_overrides or [])))


def clear_stack_cache() -> None:
    """Drop any cached framework stacks. Used by tests; the cache is
    process-local so it is also implicitly cleared between tool invocations."""
    _STACK_CACHE.clear()


def compose_cfg(
    repo_root: Path,
    experiment_config: str,
    extra_overrides: list[str],
    base_overrides_fn,
):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    config_dir = str(repo_root / "configs")
    try:
        with initialize_config_dir(version_base="1.3", config_dir=config_dir):
            return compose(
                config_name="run.yaml",
                overrides=[
                    f"experiment={experiment_config}",
                    *base_overrides_fn(extra_overrides),
                ],
            )
    except Exception:
        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base="1.3", config_dir=config_dir):
            return compose(
                config_name="run.yaml",
                overrides=[
                    f"+experiment={experiment_config}",
                    *base_overrides_fn(extra_overrides),
                ],
            )


def build_framework_stack(
    repo_root: Path,
    experiment_config: str,
    extra_overrides: list[str] | None,
    ensure_project_root_fn,
    base_overrides_fn,
) -> dict[str, Any]:
    repo_root = ensure_project_root_fn(repo_root)
    emit_heartbeat("stack_build", phase="compose_cfg")
    cfg = compose_cfg(
        repo_root, experiment_config, extra_overrides or [], base_overrides_fn
    )

    import hydra
    from omegaconf import OmegaConf
    from picid.data.preprocessing.preprocessor import PreProcessor
    from picid.run import (
        create_lightning_module,
        register_data_dim_resolver,
        register_infer_dataloader_length_resolver,
    )
    from picid.transforms.base.multisource import InverseTransformMixin
    from picid.transforms.base.transform_manager import ConfigTransformManager

    emit_heartbeat("stack_build", phase="datasource")
    dataset_source = hydra.utils.instantiate(cfg.datasource)
    transforms_manager = ConfigTransformManager(transforms_config=cfg.transforms)
    preprocessor = PreProcessor(
        datasource=dataset_source, transforms=transforms_manager
    )

    use_cache = (
        cfg.cache.use_cache_after_loading or cfg.cache.use_cache_after_transfroms
    )
    data_cache_path = cfg.paths.cache_path if use_cache else None
    data_library_part_path = repo_root / "picid/data/datasources" if use_cache else None
    transform_library_part_path = repo_root / "picid/transforms" if use_cache else None

    emit_heartbeat("stack_build", phase="preprocess")
    preprocessor.pipeline(
        data_cache_path=data_cache_path,
        data_library_part_path=data_library_part_path,
        transform_library_part_path=transform_library_part_path,
        cache_preprocessed=cfg.cache.use_cache_after_transfroms,
    )
    data_dict = preprocessor.get_processed_data_dict(return_splits_on_first_level=True)
    meta_data_dict = preprocessor.get_meta_data_dict()
    if cfg.cache.use_cache_after_transfroms:
        transforms_manager = preprocessor.get_cached_transform_manager()

    input_keys = cfg.task_definition.model.data_requirements.input_tensors
    datasets = {}
    for split in ["train", "val", "test"]:
        init_cfg = cfg.dataset
        if (
            overrides := getattr(cfg.dataset, "_overrides", None)
        ) is not None and split in overrides:
            init_cfg = OmegaConf.merge(cfg.dataset, overrides[split])
        split_meta = meta_data_dict.copy()
        split_meta["current_data_split"] = split
        datasets[split] = hydra.utils.instantiate(
            init_cfg,
            data_dict={key: data_dict[split][key] for key in input_keys},
            meta_data_dict=split_meta,
        )

    datamodule = hydra.utils.instantiate(
        cfg.datamodule,
        dataset_train=datasets["train"],
        dataset_val=datasets["val"],
        dataset_test=datasets["test"],
    )

    loader_lengths = {
        "train": len(datamodule.train_dataloader()),
        "val": len(datamodule.val_dataloader()),
        "test": len(datamodule.test_dataloader()),
    }
    register_infer_dataloader_length_resolver(loader_lengths)
    register_data_dim_resolver(data=data_dict["train"])

    class DummyEvaluator:
        effective_pred_len = None

        def update(self, *args, **kwargs):
            return None

        def reset(self):
            return None

        def compute(self, *args, **kwargs):
            return {}

    evaluators = {split: DummyEvaluator() for split in ["train", "val", "test"]}
    for split in ["train", "val", "test"]:
        if cfg.evaluator and cfg.evaluator.get(split) is not None:
            inverse_required = cfg.evaluator[split].get("apply_inverse_scaling", False)
            inverse_key = cfg.evaluator[split].get("inverse_transform_key", None)
            inverse_name = cfg.evaluator[split].get("inverse_transform_name", None)
            if inverse_required and inverse_key is not None:
                transforms_manager.get_inverter_for_key_with_name(inverse_key)
            elif inverse_required and inverse_name is not None:
                inverse_transform = transforms_manager.get_transform(
                    inverse_name
                ).transform_instance
                assert isinstance(inverse_transform, InverseTransformMixin)

    emit_heartbeat("stack_build", phase="model")
    loss = hydra.utils.instantiate(cfg.loss)
    model = create_lightning_module(
        cfg=cfg, datamodule=datamodule, evaluators=evaluators, loss=loss
    )
    emit_heartbeat("stack_build", phase="ready")
    return {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "data_dict": data_dict,
        "loader_lengths": loader_lengths,
    }


def get_or_build_framework_stack(
    repo_root: Path,
    experiment_config: str,
    extra_overrides: list[str] | None,
    ensure_project_root_fn,
    base_overrides_fn,
) -> dict[str, Any]:
    """Cached variant. All ladder checks share one stack per
    (experiment_config, sorted(overrides)) key, paid once per ladder run."""
    key = _cache_key(experiment_config, extra_overrides)
    cached = _STACK_CACHE.get(key)
    if cached is not None:
        emit_heartbeat("stack_build", phase="cache_hit")
        return cached
    stack = build_framework_stack(
        repo_root,
        experiment_config,
        extra_overrides,
        ensure_project_root_fn,
        base_overrides_fn,
    )
    _STACK_CACHE[key] = stack
    return stack


def _clone_value(value: Any) -> Any:
    try:
        import torch

        if torch.is_tensor(value):
            return value.detach().clone()
    except Exception:
        pass
    if isinstance(value, dict):
        cloned = value.__class__()
        for key, child in value.items():
            cloned[key] = _clone_value(child)
        return cloned
    if isinstance(value, (list, tuple)):
        return value.__class__(_clone_value(v) for v in value)
    return value


def _infer_batch_size(value: Any) -> int | None:
    try:
        import torch

        if torch.is_tensor(value) and value.ndim > 0:
            return int(value.shape[0])
    except Exception:
        pass
    if isinstance(value, dict):
        for child in value.values():
            found = _infer_batch_size(child)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for child in value:
            found = _infer_batch_size(child)
            if found is not None:
                return found
    return None


def _slice_batch(value: Any, sample_count: int) -> Any:
    try:
        import torch

        if torch.is_tensor(value):
            if value.ndim > 0 and value.shape[0] >= sample_count:
                return value[:sample_count]
            return value
    except Exception:
        pass
    if isinstance(value, dict):
        sliced = value.__class__()
        for key, child in value.items():
            sliced[key] = _slice_batch(child, sample_count)
        return sliced
    if isinstance(value, list):
        return [_slice_batch(child, sample_count) for child in value]
    if isinstance(value, tuple):
        return tuple(_slice_batch(child, sample_count) for child in value)
    return value


def _should_zero_key(key: str, zero_keys: set[str] | None) -> bool:
    if zero_keys is not None:
        return key in zero_keys
    lowered = key.lower()
    if any(token in lowered for token in ("target", "rul", "fault", "label", "y_")):
        return False
    return any(
        token in lowered for token in ("feature", "context", "input", "x_", "time")
    )


def zero_batch(
    batch: Any, zero_keys: set[str] | None = None, parent_key: str = ""
) -> Any:
    import torch

    cloned = _clone_value(batch)

    def mutate(obj: Any, key_name: str = "") -> Any:
        if torch.is_tensor(obj):
            if obj.is_floating_point() and _should_zero_key(
                key_name or parent_key, zero_keys
            ):
                return torch.zeros_like(obj)
            return obj
        if isinstance(obj, dict):
            for key, value in list(obj.items()):
                obj[key] = mutate(value, str(key))
            return obj
        if isinstance(obj, list):
            return [mutate(value, key_name) for value in obj]
        if isinstance(obj, tuple):
            return tuple(mutate(value, key_name) for value in obj)
        return obj

    return mutate(cloned, parent_key)


def _optimizer_from_config(configured: Any):
    if isinstance(configured, dict):
        return configured.get("optimizer")
    if isinstance(configured, (list, tuple)) and configured:
        first = configured[0]
        if isinstance(first, dict):
            return first.get("optimizer")
        return first
    return configured


def manual_train_loss(
    repo_root: Path,
    experiment_config: str,
    extra_overrides: list[str],
    zero_inputs: bool,
    zero_keys: list[str] | None,
    ensure_project_root_fn,
    base_overrides_fn,
    max_epochs: int = 5,
    max_batches: int = 10,
    heartbeat_every_steps: int = 5,
    heartbeat_every_seconds: float = 5.0,
) -> dict[str, Any]:
    import torch

    stack = get_or_build_framework_stack(
        repo_root,
        experiment_config,
        extra_overrides,
        ensure_project_root_fn,
        base_overrides_fn,
    )
    model = stack["model"]
    cfg = stack["cfg"]
    if not cfg.task_definition.get("requires_training", True):
        return {"skipped": True, "reason": "task_definition.requires_training is false"}
    if model.__class__.__name__ == "FitPredictWrapperLightningModule":
        return {
            "skipped": True,
            "reason": "fit-predict model, no gradient-based training",
        }
    model.train()
    optimizer = _optimizer_from_config(model.configure_optimizers())
    if optimizer is None:
        return {"skipped": True, "reason": "model returned no optimizer"}

    selected_zero_keys = set(zero_keys) if zero_keys else None
    losses: list[float] = []
    timed_out = False
    check_label = "zero_input_zero" if zero_inputs else "zero_input_real"
    start = time.monotonic()
    last_heartbeat = start
    step_idx = 0
    try:
        for _epoch in range(max_epochs):
            loader = stack["datamodule"].train_dataloader()
            for batch_idx, batch in enumerate(loader):
                if batch_idx >= max_batches:
                    break
                batch_for_step = (
                    zero_batch(batch, selected_zero_keys)
                    if zero_inputs
                    else _clone_value(batch)
                )
                optimizer.zero_grad(set_to_none=True)
                model_out = model.model_step(batch_for_step, stage="train")
                loss = model_out["loss"]
                if torch.is_tensor(loss) and loss.ndim > 0:
                    loss = loss.mean()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                step_idx += 1
                now = time.monotonic()
                if (
                    step_idx % heartbeat_every_steps == 0
                    or now - last_heartbeat >= heartbeat_every_seconds
                ):
                    emit_heartbeat(
                        check_label,
                        step=step_idx,
                        loss=losses[-1],
                        elapsed=now - start,
                    )
                    last_heartbeat = now
    except CheckTimeoutError:
        timed_out = True
    return {
        "skipped": False,
        "timed_out": timed_out,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "losses": losses,
        "elapsed_sec": time.monotonic() - start,
    }


def manual_overfit_single_batch(
    repo_root: Path,
    experiment_config: str,
    extra_overrides: list[str],
    ensure_project_root_fn,
    base_overrides_fn,
    task_type: str,
    micro_batch_size: int = 4,
    max_steps: int = 400,
    heartbeat_every_steps: int = 10,
    heartbeat_every_seconds: float = 5.0,
) -> dict[str, Any]:
    import torch

    stack = get_or_build_framework_stack(
        repo_root,
        experiment_config,
        extra_overrides,
        ensure_project_root_fn,
        base_overrides_fn,
    )
    model = stack["model"]
    cfg = stack["cfg"]
    if not cfg.task_definition.get("requires_training", True):
        return {"skipped": True, "reason": "task_definition.requires_training is false"}
    if model.__class__.__name__ == "FitPredictWrapperLightningModule":
        return {
            "skipped": True,
            "reason": "fit-predict model, no gradient-based training",
        }
    model.train()
    optimizer = _optimizer_from_config(model.configure_optimizers())
    if optimizer is None:
        return {"skipped": True, "reason": "model returned no optimizer"}

    batch = _clone_value(next(iter(stack["datamodule"].train_dataloader())))
    full_batch_size = _infer_batch_size(batch)
    effective_micro_batch = max(1, int(micro_batch_size))
    if full_batch_size is not None:
        effective_micro_batch = min(effective_micro_batch, full_batch_size)
    batch = _slice_batch(batch, effective_micro_batch)
    losses: list[float] = []
    timed_out = False
    start = time.monotonic()
    last_heartbeat = start
    completed_steps = 0
    try:
        for step in range(max_steps):
            optimizer.zero_grad(set_to_none=True)
            model_out = model.model_step(_clone_value(batch), stage="train")
            loss = model_out["loss"]
            if torch.is_tensor(loss) and loss.ndim > 0:
                loss = loss.mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            completed_steps = step + 1
            now = time.monotonic()
            if (
                completed_steps % heartbeat_every_steps == 0
                or now - last_heartbeat >= heartbeat_every_seconds
            ):
                emit_heartbeat(
                    "overfit_batch",
                    step=f"{completed_steps}/{max_steps}",
                    loss=losses[-1],
                    elapsed=now - start,
                )
                last_heartbeat = now
    except CheckTimeoutError:
        timed_out = True
    final_loss = losses[-1] if losses else None
    best_loss = min(losses) if losses else None
    final_metrics: dict[str, Any] = {}
    if losses:
        with torch.no_grad():
            final_out = model.model_step(_clone_value(batch), stage="train")
            loss = final_out["loss"]
            if torch.is_tensor(loss) and loss.ndim > 0:
                loss = loss.mean()
            final_loss = float(loss.detach().cpu())
            predictions, targets = _prediction_target_pair(final_out)
            final_metrics = _task_metrics(task_type, predictions, targets)
    return {
        "skipped": False,
        "timed_out": timed_out,
        "micro_batch_size": effective_micro_batch,
        "full_batch_size": full_batch_size,
        "completed_steps": completed_steps,
        "max_steps": max_steps,
        "initial_loss": losses[0] if losses else None,
        "best_loss": best_loss,
        "final_loss": final_loss,
        "losses": losses,
        "elapsed_sec": time.monotonic() - start,
        **final_metrics,
    }


def _first_tensor_output(output: Any):
    import torch

    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        for key in ("predictions", "prediction", "outputs", "output", "logits"):
            value = output.get(key)
            if torch.is_tensor(value):
                return value
        for value in output.values():
            found = _first_tensor_output(value)
            if found is not None:
                return found
    if isinstance(output, (list, tuple)):
        for value in output:
            found = _first_tensor_output(value)
            if found is not None:
                return found
    return None


def _prediction_target_pair(output: Any):
    import torch

    if isinstance(output, dict):
        predictions = output.get("predictions")
        targets = output.get("targets")
        if torch.is_tensor(predictions) and torch.is_tensor(targets):
            return predictions, targets
    return None, None


def _tensor_preview(tensor, limit: int = 8) -> list[float | int]:
    flat = tensor.detach().reshape(-1).cpu()
    preview = flat[:limit].tolist()
    normalized: list[float | int] = []
    for item in preview:
        if isinstance(item, bool):
            normalized.append(int(item))
        elif isinstance(item, int):
            normalized.append(item)
        else:
            normalized.append(float(item))
    return normalized


def _aligned_tensors(predictions, targets):
    pred = predictions.detach().float().squeeze()
    target = targets.detach().float().squeeze()
    if pred.shape == target.shape:
        return pred, target
    if pred.numel() == target.numel():
        return pred.reshape(-1), target.reshape(-1)
    return None, None


def _classification_labels(predictions, targets):
    pred = predictions.detach().squeeze()
    target = targets.detach().squeeze()
    if pred.ndim == 0 or target.ndim == 0:
        return None, None

    if pred.ndim > 0 and pred.shape[-1] > 1:
        pred_labels = pred.argmax(dim=-1)
        if target.shape == pred.shape:
            target_labels = target.argmax(dim=-1)
        else:
            target_labels = target.long()
            if target_labels.numel() == pred_labels.numel():
                target_labels = target_labels.reshape(pred_labels.shape)
    else:
        pred_labels = (pred.float() >= 0).long()
        target_labels = (
            (target >= 0.5).long() if target.is_floating_point() else target.long()
        )
        if target_labels.numel() == pred_labels.numel():
            target_labels = target_labels.reshape(pred_labels.shape)

    if pred_labels.shape != target_labels.shape:
        return None, None
    return pred_labels.reshape(-1), target_labels.reshape(-1)


def _task_metrics(task_type: str, predictions, targets) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if predictions is None or targets is None:
        return metrics

    metrics["prediction_preview"] = _tensor_preview(predictions)
    metrics["target_preview"] = _tensor_preview(targets)

    if task_type == "classification":
        pred_labels, target_labels = _classification_labels(predictions, targets)
        if pred_labels is None or target_labels is None or target_labels.numel() == 0:
            return metrics
        metrics["final_accuracy"] = float((pred_labels == target_labels).float().mean().item())
        metrics["num_classification_items"] = int(target_labels.numel())
        return metrics

    pred_aligned, target_aligned = _aligned_tensors(predictions, targets)
    if pred_aligned is None or target_aligned is None or target_aligned.numel() == 0:
        return metrics

    absolute_error = (pred_aligned - target_aligned).abs()
    target_abs_mean = float(target_aligned.abs().mean().item())
    target_std = float(target_aligned.std(unbiased=False).item())
    scale = max(target_abs_mean, target_std, 1e-6)
    final_mae = float(absolute_error.mean().item())
    final_max_abs_error = float(absolute_error.max().item())
    metrics.update(
        {
            "final_mae": final_mae,
            "final_max_abs_error": final_max_abs_error,
            "target_scale": scale,
            "target_abs_mean": target_abs_mean,
            "target_std": target_std,
            "final_mae_ratio": final_mae / scale,
            "final_max_abs_error_ratio": final_max_abs_error / scale,
        }
    )
    return metrics


def _mark_inputs_require_grad(
    obj: Any, zero_keys: set[str] | None, key_name: str = ""
) -> list[Any]:
    import torch

    marked = []
    if torch.is_tensor(obj):
        if obj.is_floating_point() and _should_zero_key(key_name, zero_keys):
            obj.requires_grad_(True)
            marked.append((key_name, obj))
        return marked
    if isinstance(obj, dict):
        for key, value in obj.items():
            marked.extend(_mark_inputs_require_grad(value, zero_keys, str(key)))
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            marked.extend(_mark_inputs_require_grad(value, zero_keys, key_name))
    return marked


def tool_failure_result(
    name: str, exc_text: str, tool_flag: bool = False
) -> dict[str, Any]:
    dataset_status = classify_dataset_failure(exc_text)
    if dataset_status is not None:
        return {
            "name": name,
            "status": dataset_status,
            "diagnostic": f"Tool-side {name.replace('_', '-')} check failed during data execution: {dataset_status}.",
            "traceback_excerpt": traceback_excerpt(exc_text),
            "dataset_executability": dataset_status,
            "fallback_trigger": True,
        }
    return {
        "name": name,
        "status": "TOOL_INVOCATION_FAILURE" if tool_flag else "FAIL",
        "diagnostic": (
            f"Tool-side {name.replace('_', '-')} check failed due to tool/config invocation issues."
            if tool_flag
            else f"Tool-side {name.replace('_', '-')} check crashed."
        ),
        "traceback_excerpt": traceback_excerpt(exc_text),
    }
