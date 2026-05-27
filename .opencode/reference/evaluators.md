# §2.6 + §2.7 — Evaluators and Metrics

> Read this when implementing a custom evaluator or metric. Most new papers
> do NOT need a custom evaluator — `DefaultEvaluator` + named metrics usually
> suffice.

## §2.6 Evaluators

```
File: picid/evaluator/base.py
Class: AbstractEvaluator(ABC)
```

```python
@abstractmethod
def update(self, model_out: dict) -> None: ...    # called per batch

@abstractmethod
def compute(self, mode: str, epoch: int, step: int) -> Dict[str, float]: ...

@abstractmethod
def reset(self) -> None: ...
```

The constructor handles inverse-scaling setup automatically via `ScalingWrapper`.

### Minimal pattern (from `picid/evaluator/default.py`)

```python
class DefaultEvaluator(AbstractEvaluator):
    def __init__(self, metric_names, task_type="regression", **kwargs):
        super().__init__(**kwargs)
        self.metric_manager = MetricManager(metric_names=metric_names, ...)
        self.buffer = PredictionBuffer()

    def update(self, model_out):
        batch = self._prepare_batch_data(model_out)  # handles inverse scaling
        self.metric_manager.update(predictions=batch["preds"], targets=batch["targets"])

    def compute(self, mode, epoch, step):
        return self.metric_manager.compute()

    def reset(self):
        self.buffer.clear()
        self.metric_manager.reset()
```

**In practice**: you almost never need a custom evaluator.  `DefaultEvaluator`
with the right `metric_names` and `task_type` covers most cases.
Specialized evaluators (`ClassificationEvaluator`, `ForecastingEvaluator`,
`ReconstructionEvaluator`, `MultiUnitEvaluator`) are thin subclasses.

## §2.7 Metrics

```
File: picid/metrics/base.py
Class: AbstractMetric(ABC)
```

```python
class AbstractMetric(ABC):
    def __init__(self, name: str): ...

    @abstractmethod
    def reset(self): ...

    @abstractmethod
    def update(self, predictions: np.ndarray, targets: np.ndarray): ...

    @abstractmethod
    def compute(self) -> float: ...
```

### Minimal pattern (from `picid/metrics/metrics.py`)

```python
class MAEMetric(AbstractMetric):
    def __init__(self):
        super().__init__("mae")
        self.reset()

    def reset(self):
        self.total_absolute_error = 0.0
        self.total_count = 0

    def update(self, predictions: np.ndarray, targets: np.ndarray):
        self.total_absolute_error += np.sum(np.abs(predictions - targets))
        self.total_count += predictions.size

    def compute(self) -> float:
        return self.total_absolute_error / self.total_count
```

**Registration**: New metrics must be registered in the `MetricManager` factory
at `picid/metrics/metric_factory.py` to be usable by name in config.
