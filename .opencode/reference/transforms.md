# §2.2 — Transforms

> Read this when implementing a new transform, picking a mixin, or wiring a
> transform into the pipeline.

## Abstract base

```
File: picid/transforms/base/base_transform/base_transform.py
Class: BaseTransform(ABC)
```

Required methods:
```python
@abstractmethod
def transform_data(self, data: NamedTransformInput, metadata: dict) -> Any: ...

@abstractmethod
def transform_multi_source(self, chunks: List[NamedTransformInput],
                            metadata: dict) -> Tuple[List[Any], dict]: ...

# Optional override for stateful (fittable) transforms:
def fit_data(self, data: NamedTransformInput, metadata: dict) -> Any: ...
```

`NamedTransformInput` = `Dict[str, np.ndarray | ak.Array]` — a dict mapping key names
(like `"features"`) to arrays.

Runtime annotation caveat: framework dispatch validates concrete annotations with
`inspect.signature()`. Generated transform modules must not use postponed
annotations for dispatched methods, and `metadata` must be the builtin `dict`
(or `Any` if genuinely required), not `typing.Dict[str, Any]`.

**You never implement `transform_multi_source` directly.** Instead, compose with a
**multisource mixin** that provides it:

## Multisource mixins (pick one)

```
File: picid/transforms/base/multisource/mixins.py
```

| Mixin class | Fit? | Transform strategy | Use when |
|---|---|---|---|
| `NoFitPerSegmentMixin` | No | Per segment independently | Stateless transforms |
| `ConcatFitAndPerSegmentTransformMixin` | Yes (concat all train segments) | Per segment | Scalers, normalizers |
| `NoFitConcatAlongAxisMixin(axis)` | No | Concat all segments, transform once | Concatenation transforms |

## Data-type markers (pick one)

```
File: picid/transforms/base/base_transform/base_transform.py
```

| Marker | Supports |
|---|---|
| `DenseTransform` | Dense NumPy arrays only |
| `RaggedTransform` | Awkward (variable-length) arrays only |
| `RaggedOrDenseTransform` | Both |

## Lifecycle (called by pipeline)

1. **Fit phase** (train split only): `fit_multi_source(train_segments, metadata)` → internally calls `fit_data()` per your mixin
2. **Transform phase** (all splits): `transform_multi_source(segments, metadata)` → internally calls `transform_data()` per your mixin

## Minimal implementation pattern (from `picid/transforms/base_transforms/identity.py`)

```python
from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.multisource import NoFitPerSegmentMixin

class IdentityPassThrough(NoFitPerSegmentMixin, DenseTransform):
    """MRO: NoFitPerSegmentMixin, DenseTransform, BaseTransform"""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def fit_data(self, data: NamedTransformInput, metadata: dict):
        pass  # stateless

    def transform_data(self, data: NamedTransformInput, metadata: dict) -> Any:
        return data
```

## Fittable transform pattern (from `picid/transforms/base_transforms/scaler.py`)

```python
from picid.transforms.base.base_transform import DenseTransform
from picid.transforms.base.multisource import (
    ConcatFitAndPerSegmentTransformMixin, InverseTransformMixin,
)

class MinMaxScalerSklearn(ConcatFitAndPerSegmentTransformMixin,
                          InverseTransformMixin, DenseTransform):
    def __init__(self):
        super().__init__()
        self.scaler = MinMaxScaler()

    def fit_data(self, data: NamedTransformInput, metadata: dict):
        np_data = list(data.values())[0]        # single key
        self.scaler.fit(np_data)

    def transform_data(self, data: NamedTransformInput, metadata: dict):
        np_data = list(data.values())[0]
        return self.scaler.transform(np_data)

    def inverse_transform(self, data: NamedTransformInput, metadata=None):
        np_data = list(data.values())[0]
        return self.scaler.inverse_transform(np_data)
```

## InverseTransformMixin

```
File: picid/transforms/base/multisource/mixins.py
Class: InverseTransformMixin(ABC)
```

Add this to the MRO when the evaluator needs to inverse-scale predictions.
Requires implementing `inverse_transform(data, metadata)`.
