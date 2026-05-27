# §2.1 — Data Source Loaders

> Read this when implementing a new `AbstractDataSourceLoader`, a
> `SingleSourceLoader`, or composing a `MultiSourceLoader`.

## Abstract interface

```
File: picid/data/datasources/base/interfaces.py
Class: AbstractDataSourceLoader(ABC)
```

Required methods:
```python
@abstractmethod
def load_data(self) -> None: ...

@abstractmethod
def get_data(self) -> DatasetContainer | SplitDatasetContainer: ...

@abstractmethod
def get_data_name(self) -> str: ...

@abstractmethod
def split_data(self) -> None: ...
```

Constructor receives `**kwargs` including mandatory `data_name: str` and `task_mode: str`.

**Lifecycle** (called by `PreProcessor`):
1. `load_data()` — read raw files into `self.data_dict`
2. `split_data()` — split into train/val/test (or let multisource splitter handle it)
3. `get_data()` — return a `SplitDatasetContainer` wrapping the dict

## Concrete base: SingleSourceLoader

```
File: picid/data/datasources/base/single_source_loader.py
Class: SingleSourceLoader(AbstractDataSourceLoader)
```

Subclass this for single-unit data sources.  Override only `_load_data()`:
```python
@abstractmethod
def _load_data(self) -> Dict[str, Union[np.ndarray, ak.Array, NDFrame]]: ...
```
`load_data()` and `split_data()` are already implemented.  The splitter
(`data_splitter`) and split keys come from config.

## Concrete base: MultiSourceLoader

```
File: picid/data/datasources/base/multi_source_loader.py
Class: MultiSourceLoader(AbstractDataSourceLoader)
```

Composes multiple `SingleSourceLoader` instances via config.  Each source is
loaded independently, then a `BySourceSplitter` assigns sources to
train/val/test splits.

## Minimal implementation pattern (from `picid/data/datasources/toy_example.py`)

```python
class ToyRaggedLoader(AbstractDataSourceLoader):
    def __init__(self, data_dir, data_name="toy", task_mode="anomaly_detection",
                 multisource_data_splitter=None, **kwargs):
        super().__init__(data_name=data_name, task_mode=task_mode,
                         multisource_data_splitter=multisource_data_splitter, **kwargs)
        self._is_loaded = False
        self._is_splitted = False

    def load_data(self):
        self.data_dict = self._load_data()
        self._is_loaded = True
        self._is_splitted = True   # if you handle splits internally

    def get_data(self) -> SplitDatasetContainer:
        return SplitDatasetContainer(**self.data_dict)

    def _load_data(self) -> dict:
        # Return: {"features": {"train": [...], "val": [...], "test": [...]},
        #          "target":   {"train": [...], "val": [...], "test": [...]}, ...}
        ...

    def get_data_name(self) -> str: return self.data_name
    def split_data(self): pass      # already split in _load_data
```

**Data shape contract**:
- Features: `(Time, Freq/Samples, Channels)` — can be ragged along Time (Awkward Array)
- Targets: `(Time, 1, 1)` — one label per timestep
- Each split value should be a **list** of arrays (one per unit), even for single-unit
