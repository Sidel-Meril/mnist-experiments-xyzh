"""Data contract shared by every experiment.

Each package keeps its own ``data_contract.yaml`` (column dtypes, nullability,
allowed values, and transform steps). ``load`` reads the path the caller passes.
``apply`` runs those steps. ``validate`` checks the result with pandera.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import attrs
import pandas as pd
import pandera.pandas as pa
import yaml

_EXTRACT = re.compile(r"""extract\(r?["'](.+)["'],\s*expand=(True|False)\)""")
_FILLNA = re.compile(r"fillna\(([^)]+)\)")
_QUERY = re.compile(r"""^(\w+)\s*==\s*(['"])(.*)\2$""")


@attrs.define(frozen=True, slots=True)
class Transform:
    action: str
    arg: str
    value: str | None = None
    fallback: str | None = None


@attrs.define(frozen=True, slots=True)
class DataColumn:
    name: str
    dtype: str
    nullable: bool
    sources: tuple[str, ...]
    values: tuple[Any, ...] | None = None
    value_map: tuple[tuple[str, str], ...] | None = None
    transform: tuple[Transform, ...] = ()

    def allowed(self) -> tuple[str, ...] | None:
        if self.value_map is not None:
            return tuple(dict.fromkeys(value for _, value in self.value_map))
        if self.values is None:
            return None
        return tuple(str(item) for item in self.values)


@attrs.define(frozen=True, slots=True)
class DataContract:
    version: int
    columns: tuple[DataColumn, ...]
    query: str = ""

    @classmethod
    def load(cls, path: Path | None = None) -> DataContract:
        raw = yaml.safe_load((path).read_text(encoding="utf-8")) or {}
        columns = [_column(name, spec or {}) for name, spec in (raw.get("columns") or {}).items()]
        return cls(
            version=int(raw.get("version", 0)),
            columns=tuple(columns),
            query=str(raw.get("query") or ""),
        )

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Run each column's transform steps and return the contract columns."""
        built: dict[str, pd.Series] = {}
        by_name = {column.name: column for column in self.columns}
        for column in self.columns:
            source = column.sources[0] if column.sources else column.name
            series = built[source] if source in built else frame[source]
            for step in column.transform:
                series = _step(series, step, column, frame, built, by_name)
            built[column.name] = _finish(series, column.dtype)
        ordered = [column.name for column in self.columns]
        return pd.DataFrame({name: built[name] for name in ordered}, index=frame.index)

    def validate(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Check ``frame`` with a pandera schema built from this contract."""
        checked = _schema(self).validate(frame)
        return pd.DataFrame(checked)


def _column(name: str, spec: dict[str, Any]) -> DataColumn:
    values, value_map = _values(spec.get("values"))
    steps = tuple(_parse_step(item) for item in (spec.get("transform") or ()))
    sources = spec.get("sources") or ()
    return DataColumn(
        name=str(name),
        dtype=str(spec.get("dtype", "string")),
        nullable=bool(spec.get("nullable", True)),
        sources=tuple(str(item) for item in sources),
        values=values,
        value_map=value_map,
        transform=steps,
    )


def _values(raw: Any) -> tuple[tuple[Any, ...] | None, tuple[tuple[str, str], ...] | None]:
    if raw is None:
        return None, None
    items_are_maps = all(isinstance(item, dict) and len(item) == 1 for item in raw)
    mapped = isinstance(raw, list) and bool(raw) and items_are_maps
    if mapped:
        pairs = []
        for item in raw:
            key, value = next(iter(item.items()))
            pairs.append((str(key), str(value)))
        return None, tuple(pairs)
    return tuple(raw), None


def _parse_step(item: dict[str, Any]) -> Transform:
    if "str" in item:
        return Transform("str", str(item["str"]))
    if "num" in item:
        return Transform("num", str(item["num"]))
    if "query" in item:
        value = None if item.get("value") is None else str(item["value"])
        return Transform("query", str(item["query"]), value=value)
    if "map" in item:
        fallback = None if item.get("fallback") is None else str(item["fallback"])
        return Transform("map", str(item["map"]), fallback=fallback)
    raise ValueError(f"unknown transform step: {item}")


def _step(
    series: pd.Series,
    step: Transform,
    column: DataColumn,
    frame: pd.DataFrame,
    built: dict[str, pd.Series],
    by_name: dict[str, DataColumn],
) -> pd.Series:
    if step.action == "str":
        return _string_step(series, step.arg)
    if step.action == "num":
        return _number_step(series, step.arg)
    if step.action == "query":
        return _query_step(series, step, column, frame, built)
    if step.action == "map":
        return _map_step(series, step, column, by_name)
    raise ValueError(f"{column.name}: unknown action {step.action}")


def _string_step(series: pd.Series, arg: str) -> pd.Series:
    text = series.astype("string")
    if arg == "lower":
        return text.str.lower()
    if arg == "strip":
        return text.str.strip()
    if arg == "upper":
        return text.str.upper()
    if arg == "str[0]":
        return text.str[0]
    extracted = _EXTRACT.fullmatch(arg)
    if extracted is not None:
        pattern, expand = extracted.groups()
        return text.str.extract(pattern, expand=expand == "True")
    raise ValueError(f"unknown string transform: {arg}")


def _number_step(series: pd.Series, arg: str) -> pd.Series:
    if arg == "to_int":
        return pd.to_numeric(series, errors="coerce")
    filled = _FILLNA.fullmatch(arg)
    if filled is not None:
        return series.fillna(float(filled.group(1)))
    raise ValueError(f"unknown numeric transform: {arg}")


def _query_step(
    series: pd.Series,
    step: Transform,
    column: DataColumn,
    frame: pd.DataFrame,
    built: dict[str, pd.Series],
) -> pd.Series:
    matched = _QUERY.fullmatch(step.arg)
    if matched is None:
        raise ValueError(f"{column.name}: cannot read query {step.arg!r}")
    subject_name = matched.group(1)
    literal = matched.group(3)
    if subject_name == column.name:
        subject = series
    elif subject_name in built:
        subject = built[subject_name]
    else:
        subject = frame[subject_name]
    mask = subject.astype("string").eq(literal)
    replacement = pd.NA if step.value == "pd.NA" else step.value
    return series.mask(mask, replacement)


def _map_step(
    series: pd.Series,
    step: Transform,
    column: DataColumn,
    by_name: dict[str, DataColumn],
) -> pd.Series:
    table = _mapping(step.arg, column, by_name)
    text = series.astype("string")
    if isinstance(table, dict):
        mapped = text.map(table)
    else:
        allowed = {str(item) for item in table}
        mapped = text.where(text.isin(allowed), pd.NA)
    if step.fallback is None:
        return mapped
    return mapped.fillna(step.fallback)


def _mapping(
    token: str,
    column: DataColumn,
    by_name: dict[str, DataColumn],
) -> dict[str, str] | tuple[Any, ...]:
    if token.startswith("${") and token.endswith(".values}"):
        name = token[2 : -len(".values}")]
        target = by_name.get(name, column)
        if target.value_map is not None:
            return dict(target.value_map)
        if target.values is not None:
            return target.values
    if column.value_map is not None:
        return dict(column.value_map)
    if column.values is not None:
        return column.values
    raise ValueError(f"{column.name}: map {token} has no values")


def _finish(series: pd.Series, dtype: str) -> pd.Series:
    if dtype == "integer":
        return _as_integer(series)
    if dtype == "numeric":
        return pd.to_numeric(series, errors="coerce").astype("float64")
    return series.astype("string")


def _as_integer(series: pd.Series) -> pd.Series:
    """One integer cast. Width is chosen later by the modeling contract."""
    return pd.to_numeric(series, errors="coerce").astype(int)


def _schema(contract: DataContract) -> pa.DataFrameSchema:
    columns: dict[str, pa.Column] = {}
    for column in contract.columns:
        checks = None
        allowed = column.allowed()
        if allowed is not None:
            checks = pa.Check.isin(list(allowed))
        columns[column.name] = pa.Column(
            _pandera_dtype(column.dtype),
            checks=checks,
            nullable=column.nullable,
            coerce=True,
        )
    return pa.DataFrameSchema(columns, strict=True, ordered=True)


def _pandera_dtype(dtype: str) -> Any:
    if dtype == "integer":
        return int
    if dtype == "numeric":
        return pa.Float64
    if dtype == "string":
        return pa.String
    raise ValueError(f"unknown dtype: {dtype}")
