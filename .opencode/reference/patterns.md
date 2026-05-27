# §5 — Implementation Patterns

> Read this when dealing with fit-on-train semantics, ragged arrays,
> multi-source iteration, caching, or inverse-scaling concerns.

## 5.1 Transform fit-on-train semantics

Fittable transforms (those using `ConcatFitAndPerSegmentTransformMixin` or
`FitByConcatenationMixin`) are fitted **only on the train split**.  The pipeline:
1. Concatenates all train-split segments
2. Calls `fit_data(concatenated_train, metadata)`
3. Then calls `transform_data(segment, metadata)` on each segment of **every** split

This prevents data leakage.  The fitted state (e.g., scaler parameters) is
serialized with the transform via the caching system.

## 5.2 Variable-length time series (Awkward Arrays)

The framework uses [Awkward Arrays](https://awkward-array.org) for ragged
(variable-length) data.  Key points:
- Features shape: `ak.Array` of shape `(Time, Freq, Channels)` where Time varies
- The `RaggedTransform` marker indicates your transform handles `ak.Array`
- Handlers in `picid/transforms/base/handlers/` dispatch to the right
  concatenation strategy based on data kind (ragged vs dense)
- Use `picid.utils.awkward_utils.get_ak_shape()` for shape inspection

## 5.3 Multi-source datasets

When multiple units/machines are combined:
1. Each unit is loaded as a separate `SingleSourceLoader`
2. `MultiSourceLoader` composes them via config
3. `BySourceSplitter` assigns units to splits
4. Transforms receive `List[NamedTransformInput]` (one per unit)
5. The multisource mixin handles iteration/concatenation

## 5.4 Caching system

The `PreProcessor` supports three-tier caching:
1. **After loading**: raw data cached to disk (keyed by datasource code hash)
2. **Boundary checkpoint**: intermediate state between transform steps
3. **After transforms**: fully preprocessed data cached (keyed by datasource + transform code hash)

Enable via `cache.use_cache_after_loading` and `cache.use_cache_after_transfroms`
in config.  Cache invalidation is automatic when source code changes.

## 5.5 Evaluators and inverse scaling

When evaluators have `apply_inverse_scaling: True`:
1. The evaluator constructor receives the `inverse_transform` from the named
   transform (e.g., `scaler_rul`)
2. In `_prepare_batch_data()`, predictions and targets are inverse-transformed
   before metric computation
3. If `is_dual` (both scaled and unscaled needed), both versions are stored

This is configured per-split in evaluator YAML:
```yaml
evaluator:
  val:
    inverse_transform_name: scaler_rul    # or use inverse_transform_key
    apply_inverse_scaling: True
```

## 5.6 Full experiment config composition

To add a new model to an existing dataset:
1. Create `configs/model/your_model.yaml` with `_target_` and params
2. Create `configs/model_configs/<task>/your_model.yaml` with dataset/datamodule/optimization overrides
3. Create `configs/experiment/<dataset>/<task>/your_model.yaml` composing base + model_configs
