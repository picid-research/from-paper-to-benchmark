# PICID / PICID — Agent Reference Index

> **This is the index.** Read it first, then load **only** the section files
> your current task requires. Do NOT preload the whole directory — that
> defeats the point of the split.

PICID (internally `picid`) is a config-driven benchmark framework for
Prognostics and Health Management, time-series forecasting, anomaly detection,
and classification on tabular/sensor data. It is built on **PyTorch Lightning**
+ **Hydra**.

Pipeline:
```
seed → datasource.load_data()/split_data()
     → PreProcessor.pipeline() (transforms)
     → Dataset + DataModule
     → Model wrapper → LightningModule
     → Trainer.fit() → Trainer.test(ckpt_path="best")
```

## Section files

| File                               | Covers                                                  | Former §§  |
| ---------------------------------- | ------------------------------------------------------- | ---------- |
| [`framework.md`](framework.md)     | Identity, entry point, pipeline flow                    | §1         |
| [`datasources.md`](datasources.md) | Data source loader contracts, shape rules               | §2.1       |
| [`transforms.md`](transforms.md)   | Transform base, mixins, data-type markers               | §2.2       |
| [`models.md`](models.md)           | Backbone `nn.Module` + wrapper contracts                | §2.3, §2.4 |
| [`losses.md`](losses.md)           | `AbstractLoss` contract                                 | §2.5       |
| [`evaluators.md`](evaluators.md)   | Evaluator + metric contracts                            | §2.6, §2.7 |
| [`configs.md`](configs.md)         | Hydra tree, `_target_`, composition, resolvers          | §3         |
| [`inventory.md`](inventory.md)     | What already exists (datasets, transforms, models, ...) | §4         |
| [`patterns.md`](patterns.md)       | Fit-on-train, awkward arrays, caching, inverse scaling  | §5         |
| [`style.md`](style.md)             | Conventions + what NOT to implement                     | §6, §7     |
| [`policies.md`](policies.md)       | Orchestration policies: validation modes, no-cross-dataset-fallback dataset recovery, cost-gating, bounded retries, abort-recovery. Auto-loaded via `opencode.json`. | orchestration |

## Decision tree — what to read for which task

| If you are…                                                   | Read                                                                                                                       |
| ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Implementing a new model                                      | [`models.md`](models.md), [`configs.md`](configs.md), [`style.md`](style.md)                                               |
| Implementing a new datasource                                 | [`datasources.md`](datasources.md), [`inventory.md`](inventory.md), [`patterns.md`](patterns.md)                           |
| Implementing a new transform                                  | [`transforms.md`](transforms.md), [`configs.md`](configs.md), [`patterns.md`](patterns.md)                                 |
| Implementing a new loss                                       | [`losses.md`](losses.md), [`configs.md`](configs.md), [`style.md`](style.md)                                               |
| Implementing a new metric                                     | [`evaluators.md`](evaluators.md), [`configs.md`](configs.md)                                                               |
| Composing a new experiment                                    | [`configs.md`](configs.md), [`inventory.md`](inventory.md), [`patterns.md`](patterns.md)                                   |
| Verifying static contracts / sanity                           | [`datasources.md`](datasources.md), [`transforms.md`](transforms.md), [`models.md`](models.md), [`configs.md`](configs.md) |
| Evaluating results                                            | [`evaluators.md`](evaluators.md), [`inventory.md`](inventory.md)                                                           |
| Running training                                              | [`framework.md`](framework.md), [`patterns.md`](patterns.md)                                                               |
| Mapping a paper's contributions to framework extension points | [`inventory.md`](inventory.md), [`models.md`](models.md), [`transforms.md`](transforms.md), [`losses.md`](losses.md)       |
| Orchestrating validation, fallback, retries                   | [`policies.md`](policies.md) (already auto-loaded via `opencode.json` `instructions`)                                      |

## Usage rule

**Read only the files your task requires.**
If you are unsure, consult the decision tree above or the
`See reference:` line at the top of each skill / agent prompt.
