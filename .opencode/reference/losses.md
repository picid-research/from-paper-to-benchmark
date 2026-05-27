# §2.5 — Loss Functions

> Read this when implementing a new loss. Custom losses live in `picid/loss/`.

```
File: picid/loss/base.py
Class: AbstractLoss(ABC)
```

```python
@abstractmethod
def forward(self, model_out: Dict[str, torch.Tensor],
            batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Must add 'loss' key to model_out and return it."""
```

## Minimal pattern (from `picid/loss/default.py`)

```python
class MSELoss(AbstractLoss):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, model_out, batch):
        loss = self.mse_loss(model_out["predictions"], model_out["targets"])
        result = model_out.copy()
        result["loss"] = loss
        return result
```
