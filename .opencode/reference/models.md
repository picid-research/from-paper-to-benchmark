# §2.3 + §2.4 — Model Methods and Wrappers

> Read this when implementing a new model backbone (`nn.Module`) and its
> wrapper class. Backbones live in `picid/model/methods/`; wrappers live in
> `picid/model/wrappers/`.

## §2.3 Model Methods (nn.Module backbone)

```
File: picid/model/methods/  (one file per architecture)
```

Models are plain `torch.nn.Module` subclasses.  They receive a `config` dict
and produce output from `forward(x)`.

### Minimal pattern (from `picid/model/methods/mlp.py`)

```python
class MLP(nn.Module):
    def __init__(self, config: Dict[str, Any], task_type: str, num_targets: int):
        super().__init__()
        self.input_dim = int(config["seq_len"]) * int(config["input_channels"])
        self.hidden_dim = int(config.get("hidden_dim", 64))
        self.output_dim = num_targets
        # Build layers...
        self.net = nn.Sequential(...)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        return self.net(x.reshape(b, -1))
```

## §2.4 Model Wrappers

```
File: picid/model/wrappers/base.py
```

Wrappers sit between the raw `nn.Module` backbone and the Lightning module.
The wrapper's `forward(batch)` receives a **dict** batch and must return a
**dict** with at least `"predictions"` and `"targets"` keys.

### Three wrapper types

| Wrapper | When to use | forward() contract |
|---|---|---|
| `AbstractFeedForwardWrapper(ABC, nn.Module)` | Pre-trained / no-grad models (constant loss) | `forward(batch) → {"predictions", "targets"}` |
| `AbstractFeedForwardTrainingWrapper(ABC, nn.Module)` | Standard gradient-trained models | Same — loss computed by the LightningModule |
| `AbstractFitPredictWrapper(ABC)` | sklearn-style fit/predict models | `fit(X, y)` and `predict(X) → Tensor` |

### Minimal pattern (from `picid/model/wrappers/mlp_wrapper.py`)

```python
from picid.model.wrappers.base import AbstractFeedForwardTrainingWrapper

class MLPWrapper(AbstractFeedForwardTrainingWrapper):
    def __init__(self, task_type, seq_len, *, input_channels, num_targets=None,
                 hidden_dim=64, num_layers=2, **kwargs):
        config = {"input_channels": input_channels, "seq_len": seq_len,
                  "hidden_dim": hidden_dim, "num_layers": num_layers}
        backbone = MLP(config, task_type="regression", num_targets=num_targets or 1)
        super().__init__(backbone=backbone, **kwargs)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = batch["features"].permute(0, 2, 1)       # (B, C, T)
        y = batch[self.task_type]                      # ground truth
        outputs = self.backbone(x)                     # (B, num_targets)
        return {
            "predictions": outputs.unsqueeze(1),       # (B, 1, num_targets)
            "targets": y.unsqueeze(1),                 # (B, 1, num_targets)
        }
```

**Important**: The framework automatically wraps your wrapper in the right
LightningModule based on its type:
- `AbstractFeedForwardWrapper` → `ConstantLossLightningModule` (no training)
- `AbstractFeedForwardTrainingWrapper` → `TrainingLightningModule` (gradient training)
- `AbstractFitPredictWrapper` → `FitPredictWrapperLightningModule` (sklearn-style)

This is determined in `picid/run.py:create_lightning_module()`.

### FitPredict wrapper pattern

```python
from picid.model.wrappers.base import AbstractFitPredictWrapper

class XGBoostWrapper(AbstractFitPredictWrapper):
    def __init__(self, backbone, **kwargs):
        super().__init__(backbone=backbone, **kwargs)

    def serialize_model(self, task_id): ...   # save to disk
    def load_model(self, task_id): ...        # load from disk

    @property
    def allows_multi_target(self) -> bool: return False
```
