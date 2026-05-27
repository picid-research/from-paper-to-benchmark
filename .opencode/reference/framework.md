# §1 — Framework Identity

> Read this when you need to understand what PICID is, where it starts, and how the pipeline flows.

PICID is a config-driven benchmark framework for Prognostics and Health Management (PHM),
time-series forecasting, anomaly detection, and classification on tabular/sensor data.
It is built on **PyTorch Lightning** (training loops, callbacks, logging) and **Hydra**
(config composition, `_target_` instantiation).  The package root is `picid/`.

**Entry point**: `picid/run.py → main()` decorated with `@hydra.main(config_path="../configs", config_name="run.yaml")`.

**Pipeline flow**:
```
seed → instantiate datasource → load_data() → split_data()
     → ConfigTransformManager → PreProcessor.pipeline()
     → register_data_dim_resolver (infer shapes)
     → instantiate Dataset(s) → DataModule
     → instantiate Model wrapper → LightningModule
     → instantiate Loss, Evaluators, Callbacks, Logger
     → Trainer.fit() → Trainer.test(ckpt_path="best")
     → rerun best checkpoint on val+test
```
