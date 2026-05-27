# §4 — Existing Components Inventory

> Read this to check what already exists before implementing anything new.
> **Reuse-first**: if a component is here, use or extend it rather than
> reimplementing.

## 4.1 Datasets (Data Sources)

| File | Description |
|---|---|
| `datasources/phmd_n_cmapss.py` | NASA N-CMAPSS turbofan engine degradation (via PHMD) |
| `datasources/concepts_n_cmapss.py` | N-CMAPSS with concept labels (health states, LPT/HPT) |
| `datasources/phmd_phme20.py` | PHM Europe 2020 challenge dataset (via PHMD) |
| `datasources/phmd_pronostia.py` | PRONOSTIA bearing accelerated degradation (via PHMD) |
| `datasources/phmd_xjtu_sy.py` | XJTU-SY bearing dataset (via PHMD) |
| `datasources/phmd_hsf15.py` | HSF15 hydraulic system condition monitoring (via PHMD) |
| `datasources/phmd_pubd16.py` | PUBD16 dataset (via PHMD) |
| `datasources/phmd_cbmv3.py` | CBMv3 condition-based maintenance dataset (via PHMD) |
| `datasources/agtf30k.py` | AGTF30K gas turbine dataset |
| `datasources/nb14/loader.py` | NASA battery (NB14) dataset — Bosello split |
| `datasources/unibo/loader.py` | UNIBO battery cycling dataset |
| `datasources/ETTh.py` | ETTh1/ETTh2 electricity transformer temperature (forecasting) |
| `datasources/MZVAV.py` | MZVAV building HVAC dataset |
| `datasources/railway.py` | Railway traction load forecasting |
| `datasources/airbus_helicopter.py` | Airbus helicopter vibration dataset |
| `datasources/toy_example.py` | Toy ragged loader for testing/demos |
| `datasources/picasso/` | PICASSO time-series datamodule (external) |

## 4.2 Transforms

**Base transforms** (`picid/transforms/base_transforms/`):

| File | Description |
|---|---|
| `identity.py` | Pass-through (no-op) |
| `scaler.py` | ConstantScaler, MinMaxScalerSklearn, StandardScalerSklearn |
| `subsample.py` | WindowedAggregationTransform (downsample with windowed mean/last/etc.) |
| `concatenate.py` | ConcatenateTransform (merge multiple keys into one) |
| `padding2length.py` | Pad sequences to a fixed length |
| `reshaping.py` | Reshape operations |
| `shape_manipulations.py` | Squeeze, unsqueeze, transpose |
| `tabularizers.py` | Convert time-series to tabular features |
| `spectral.py` | Spectral domain transforms |
| `stfft.py` | Short-time FFT |
| `timefeatures.py` | Time feature extraction |
| `time_statistics.py` | Statistical aggregation over time |
| `imputation_methods.py` | Missing value imputation |
| `mcar_corruption.py` | MCAR (missing completely at random) corruption |

**Domain-specific transforms**:

| Directory | Description |
|---|---|
| `transforms/n_cmapss/` | N-CMAPSS-specific scalers, concept class builder |
| `transforms/battery/` | Battery cycling transforms, sequence-to-statistics |
| `transforms/bearings/` | Health index computation (Ahmad2019, Li2019 models) |
| `transforms/building/` | MZVAV scaler, dimension swap |
| `transforms/railway/` | Seasonal dummy, time features for railway data |
| `transforms/signal_processing/` | Cumulative sum |
| `transforms/visualization/` | Feature/target plotting |

## 4.3 Models

**Methods** (`picid/model/methods/`):

| File | Architecture | Tasks |
|---|---|---|
| `mlp.py` | Multi-Layer Perceptron | Regression, classification |
| `cnn_1d.py` | 1D Convolutional Neural Network | Regression, classification |
| `linear_regression_model.py` | Linear regression (single layer) | Regression |
| `naive_model.py` | Naive baseline (last value) | Regression |
| `drift_model.py` | Drift/trend model | Regression |
| `ses_model.py` | Simple Exponential Smoothing | Regression |
| `statistical_models.py` | Mean, persistence, polynomial, similar-period | Regression |

**Wrappers** (`picid/model/wrappers/`):

| File | Wrapper type | Notes |
|---|---|---|
| `mlp_wrapper.py` | FeedForwardTraining | Standard gradient training |
| `cnn1d_wrapper.py` | FeedForwardTraining | Standard gradient training |
| `linear_regression_wrapper.py` | FeedForward (constant loss) | No training needed |
| `naive_model_wrapper.py` | FeedForward (constant loss) | No training needed |
| `drift_model_wrapper.py` | FeedForward (constant loss) | No training needed |
| `ses_model_wrapper.py` | FeedForward (constant loss) | No training needed |
| `statistical_models_wrapper.py` | FeedForward (constant loss) | No training needed |
| `fit_predict_xgboost_wrapper.py` | FitPredict | XGBoost via sklearn API |
| `fit_predict_tabpfn_wrapper.py` | FitPredict | TabPFN (pretrained) |
| `fit_predict_tabdpt_wrapper.py` | FitPredict | TabDPT |
| `fit_predict_carte_wrapper.py` | FitPredict | CARTE |
| `fit_predict_isolation_forest_wrapper.py` | FitPredict | Isolation Forest |
| `fit_predict_autogluon_wrapper.py` | FitPredict | AutoGluon |
| `tabpfn_wrapper.py` | FeedForward | TabPFN alternative wrapper |

**Additional model configs** (via Hydra, in `configs/model/`):
`crossformer`, `patchtst`, `lstm`, `tide`, `stf`, `timeseries_transformer` —
these reference model code in `picid/baselines/` (not listed above but present).

**Model utilities** (`picid/model/utils/`): `masking.py`, `revin.py`,
`timefeatures.py`, `magnitude_max_pooling.py`

**Autoencoder** (`picid/model/autoencoder/`): `residual_1d.py`

## 4.4 Loss Functions

| File | Classes |
|---|---|
| `loss/default.py` | MSELoss, MAELoss, HuberLoss, QuantileLoss, MAPELoss, SMAPELoss, WeightedMSELoss, CombinedLoss |
| `loss/cross_entropy.py` | CrossEntropyLoss |

## 4.5 Metrics

**Regression** (`picid/metrics/metrics.py`):
MAEMetric, MSEMetric, RMSEMetric, MAPEMetric, MSPEMetric, RSEMetric,
CORRMetric, NASAScoreMetric, MASEMetric

**RUL-specific** (`picid/metrics/rul_metrics.py`):
MeanPercentageErrorMetric (MPE), PHMScoreMetric

**Classification** (`picid/metrics/metrics.py`):
MulticlassAccuracyMetric, MulticlassPrecisionMetric, MulticlassRecallMetric,
MulticlassF1Metric, MulticlassAUROCMetric

**Domain-specific** (`picid/metrics/metrics.py`):
NormalizedMAEMetricRailway, NormalizedMSEMetricRailway

## 4.6 Evaluators

| File | Class | Use case |
|---|---|---|
| `evaluator/default.py` | DefaultEvaluator | General purpose (most tasks) |
| `evaluator/classification.py` | ClassificationEvaluator | Classification tasks |
| `evaluator/forecasting.py` | ForecastingEvaluator | Forecasting tasks |
| `evaluator/reconstruction.py` | ReconstructionEvaluator | Autoencoder reconstruction |
| `evaluator/multiunit.py` | MultiUnitEvaluator | Per-unit metric aggregation |

## 4.7 Dataset Classes

| File | Class | When to use |
|---|---|---|
| `datasets/sliding_window_batch_dataset.py` | SlidingWindowBatchDataset | Standard sliding window for time-series (forecasting, state_forecasting) |
| `datasets/context_dataset.py` | ContextBatchDataset | Context window for regression/classification (RUL, diagnostics) |
| `datasets/rul_context_dataset.py` | RULContextBatchDataset | RUL-specific context with unit tracking |
| `datasets/fit_predict_dataset.py` | FitPredictTaskDataset | For sklearn-style fit/predict models |
| `datasets/fault_classification_dataset.py` | FaultClassificationDataset | Fault classification tasks |
| `datasets/concept_rul_dataset.py` | ConceptRULDataset | RUL with concept labels |
| `datasets/hydra_concat_dataset.py` | HydraConcatDataset | Wraps per-unit datasets into one (multi-unit) |

Choose dataset configs by model I/O, not just task label. RUL supervised models
usually use `prognostics/rul_multiunit_dataset`; reconstruction/autoencoder
stages should use a reconstruction-compatible windowing contract and must not be
wired to RUL context targets unless target/context lengths are verified.

## 4.8 Splitting Strategies

| File | Class | When to use |
|---|---|---|
| `preprocessing/splits/time_splitter.py` | TimeSplitter | Split single time series by time ratio (train/val/test %) |
| `preprocessing/splits/by_source_splitter.py` | BySourceSplitter | Assign entire units to train/val/test (cross-unit evaluation) |
| `preprocessing/splits/database_splitter.py` | TimeStampSplitter | Split by timestamp values |

## 4.9 Task Types

Defined in `picid/baselines/definitions.py`:

```python
REGRESSION_TASKS = ["regression", "rul", "ahrul", "soc"]
CLASSIFICATION_TASKS = ["classification", "health_states", "concepts",
                        "fault_classification", "anomaly_detection"]
FORECASTING_TASKS = ["forecasting"]
STATE_FORECASTING_TASKS = ["state_forecasting"]
```
