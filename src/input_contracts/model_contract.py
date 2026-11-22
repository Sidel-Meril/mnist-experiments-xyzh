"""Modeling contract shared by every experiment.

Each package keeps its own ``model_contract.yaml``. Columns are grouped as
``key``, ``target``, ``features``, and ``log``. A feature with ``storage: npy``
is a tensor aligned to the table, not a parquet column. ``validate`` checks the
tabular columns as a ``TensorDict`` with pandera. A string column is the integer
index of its ``categories`` list. ``feature_frame`` returns tabular features.
``feature_matrix`` returns the tensor feature. ``read`` loads those columns from
the paths the caller passes.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import attrs
import numpy as np
import pandera.tensordict as pa
import pyarrow.parquet as pq
import torch
import yaml
from pandera.api.checks import Check
from tensordict import TensorDict


@attrs.define(frozen=True, slots=True)
class Column:
    name: str
    dtype: str
    nullable: bool = True
    categories: tuple[Any, ...] | None = None
    min_value: float | None = None
    max_value: float | None = None
    storage: str | None = None
    shape: tuple[int, ...] | None = None


@attrs.define(frozen=True, slots=True)
class ModelingContract:
    version: int
    columns: tuple[Column, ...]
    key: tuple[str, ...]
    target: str
    features: tuple[str, ...]
    log: tuple[str, ...]
    query: str = ""

    @classmethod
    def load(cls, path: Path | None = None) -> ModelingContract:
        raw = yaml.safe_load((path).read_text(encoding="utf-8")) or {}
        key = _group(raw.get("key"))
        target = _group(raw.get("target"))
        features = _group(raw.get("features"))
        log = _group(raw.get("log"))
        if len(target) != 1:
            raise ValueError("contract needs exactly one target column")
        return cls(
            version=int(raw.get("version", 0)),
            columns=key + target + features + log,
            key=tuple(column.name for column in key),
            target=target[0].name,
            features=tuple(column.name for column in features),
            log=tuple(column.name for column in log),
            query=str(raw.get("query") or ""),
        )

    def from_columns(self, columns: Mapping[str, Any]) -> TensorDict:
        """Tabular ``TensorDict``. A string column is its category index."""
        data: dict[str, torch.Tensor] = {}
        n: int | None = None
        for column in self._table_columns():
            if column.name not in columns:
                continue
            values = np.asarray(columns[column.name])
            if values.ndim != 1:
                raise ValueError(f"{column.name} must be one-dimensional, got shape {values.shape}")
            if n is None:
                n = int(values.shape[0])
            data[column.name] = _to_tensor(column, values)
        if n is None:
            n = 0
        return TensorDict(data, batch_size=[n])

    def feature_frame(self, batch: TensorDict) -> TensorDict:
        """Tabular feature columns. Tensor features are omitted."""
        specs = {column.name: column for column in self.columns}
        names = [name for name in self.features if specs[name].storage != "npy"]
        n = int(batch.batch_size[0])
        return TensorDict({name: batch[name] for name in names}, batch_size=[n])

    def feature_matrix(self, pixels: np.ndarray, index) -> np.ndarray:
        """Rows of the ``storage: npy`` feature, in ``index`` order."""
        column = self.tensor_feature()
        expected = column.shape
        if expected is None or pixels.ndim != len(expected) + 1:
            raise ValueError(
                f"pixels shape {getattr(pixels, 'shape', None)} does not match {expected}"
            )
        if tuple(int(dim) for dim in pixels.shape[1:]) != expected:
            raise ValueError(f"pixels shape {pixels.shape} does not match {expected}")
        rows = np.asarray(index, dtype=np.int64)
        if rows.size and (int(rows.min()) < 0 or int(rows.max()) >= len(pixels)):
            raise IndexError(f"pixel index outside 0..{len(pixels) - 1}")
        return np.ascontiguousarray(pixels[rows])

    def validate(self, batch: TensorDict) -> TensorDict:
        """Check tabular columns with a pandera ``TensorDict`` schema."""
        names = [column.name for column in self._table_columns()]
        missing = [name for name in names if name not in batch]
        if missing:
            raise ValueError(f"missing columns: {missing}")
        ordered = TensorDict(
            {name: batch[name] for name in names},
            batch_size=batch.batch_size,
        )
        checked = _schema(self).validate(ordered)
        return TensorDict(
            {name: checked[name] for name in names},
            batch_size=checked.batch_size,
        )

    def read(self, data_path: Path, pixels_path: Path) -> tuple[TensorDict, np.ndarray]:
        """Load the tabular batch and the npy feature named by this contract."""
        names = [column.name for column in self._table_columns()]
        table = pq.read_table(data_path, columns=names)
        columns = {name: table.column(name).to_numpy(zero_copy_only=False) for name in names}
        pixels = np.load(pixels_path)
        n_table = int(table.num_rows)
        if len(pixels) != n_table:
            raise ValueError(f"pixels rows {len(pixels)} do not match table rows {n_table}")
        batch = self.validate(self.from_columns(columns))
        self.feature_frame(batch)
        self.feature_matrix(pixels, np.arange(n_table))
        return batch, pixels

    def _table_columns(self) -> tuple[Column, ...]:
        return tuple(column for column in self.columns if column.storage != "npy")

    def tensor_feature(self) -> Column:
        tensors = [
            column
            for column in self.columns
            if column.name in self.features and column.storage == "npy"
        ]
        if len(tensors) != 1:
            raise ValueError("contract needs exactly one npy feature")
        return tensors[0]


def _group(spec_map: dict[str, Any] | None) -> tuple[Column, ...]:
    columns = []
    for name, spec in (spec_map or {}).items():
        spec = spec or {}
        categories = spec.get("categories")
        raw_shape = spec.get("shape")
        storage = None if spec.get("storage") is None else str(spec["storage"])
        if storage not in (None, "npy"):
            raise ValueError(f"unknown storage: {storage}")
        columns.append(
            Column(
                name=str(name),
                dtype=str(spec.get("dtype", "float")),
                nullable=bool(spec.get("nullable", True)),
                categories=None if categories is None else tuple(categories),
                min_value=None if spec.get("min") is None else float(spec["min"]),
                max_value=None if spec.get("max") is None else float(spec["max"]),
                storage=storage,
                shape=None if raw_shape is None else tuple(int(item) for item in raw_shape),
            )
        )
    return tuple(columns)


def _schema(contract: ModelingContract) -> pa.TensorDictSchema:
    _use_tensordict_backend()
    keys: dict[str, pa.Tensor] = {}
    for column in contract._table_columns():
        keys[column.name] = pa.Tensor(
            _pandera_dtype(column.dtype),
            shape=(None,),
            checks=_checks(column),
            nullable=False,
            coerce=True,
        )
    return pa.TensorDictSchema(keys, batch_size=(None,), coerce=True)


def _checks(column: Column) -> list[Check] | None:
    checks: list[Check] = []
    if column.categories is not None:
        if column.dtype == "string":
            allowed: list[Any] = list(range(len(column.categories)))
        else:
            allowed = list(column.categories)
        checks.append(Check.isin(allowed))
    if column.min_value is not None:
        checks.append(Check.ge(column.min_value))
    if column.max_value is not None:
        checks.append(Check.le(column.max_value))
    return checks or None


# Change this one name to move every modeling integer tensor, for example to torch.int32.
_INT = torch.int64
_FLOAT = torch.float64


def _pandera_dtype(dtype: str) -> torch.dtype:
    if dtype == "int":
        return _INT
    if dtype == "float":
        return _FLOAT
    if dtype == "string":
        return _INT
    raise ValueError(f"unknown dtype: {dtype}")


def _to_tensor(column: Column, values: np.ndarray) -> torch.Tensor:
    if _has_null(values):
        if not column.nullable:
            raise ValueError(f"{column.name} contains nulls")
        if column.dtype != "float":
            raise ValueError(f"{column.name} nulls cannot be stored in a tensor")
    if column.dtype == "string":
        if not column.categories:
            raise ValueError(f"{column.name} string column needs categories")
        codes = _category_codes(values, column.categories)
        return torch.tensor(codes, dtype=_INT)
    if column.dtype == "int":
        return torch.tensor(np.ascontiguousarray(values), dtype=_INT)
    if column.dtype == "float":
        return torch.tensor(np.ascontiguousarray(values), dtype=_FLOAT)
    raise ValueError(f"unknown dtype: {column.dtype}")


def _category_codes(values: np.ndarray, categories: tuple[Any, ...]) -> np.ndarray:
    text = np.asarray(values, dtype=str).reshape(-1)
    levels = np.array([str(item) for item in categories], dtype=str)
    if levels.size == 0:
        return np.full(text.shape, -1, dtype=np.int64)
    order = np.argsort(levels, kind="mergesort")
    sorted_levels = levels[order]
    pos = np.searchsorted(sorted_levels, text)
    in_range = pos < len(levels)
    pos = np.clip(pos, 0, len(levels) - 1)
    matched = in_range & (sorted_levels[pos] == text)
    return np.where(matched, order[pos], -1).astype(np.int64, copy=False)


def _has_null(values: np.ndarray) -> bool:
    if values.dtype.kind in "iub":
        return False
    if values.dtype.kind == "f":
        return bool(np.isnan(values).any())
    flat = np.asarray(values, dtype=object).ravel()
    if flat.size == 0:
        return False
    return any(item is None or item != item for item in flat.tolist())


def _use_tensordict_backend() -> None:
    from pandera.backends.tensordict.register import register_tensordict_backends

    register_tensordict_backends()
