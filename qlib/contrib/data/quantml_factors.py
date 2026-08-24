"""Local-only QuantML adapted minute factors.

The catalog is derived from an unlicensed result table and must not be
redistributed. The implementation is explicitly adapted because the original
QuantML ``qlib_csc`` operators were not published.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow.dataset as pyarrow_dataset

from qlib.contrib.data._quantml_catalog import CATALOG_ROWS, SOURCE_SHA256
from qlib.contrib.data.dolphindb_compat import FactorDependencyError
from qlib.data.base import Expression, ExpressionOps
from qlib.data.ops import (
    Add,
    Corr,
    Div,
    IdxMax,
    IdxMin,
    Kurt,
    Mad,
    Max,
    Med,
    Min,
    Mul,
    Power,
    Rank,
    Ref,
    Resi,
    Rsquare,
    Skew,
    Slope,
    Std,
    Sub,
)


CATALOG_VERSION = "quantml-adapted-v1"
OPERATOR_VERSION = "quantml-adapted-ops-v2"
AVAILABLE_FEATURES = frozenset({"open", "high", "low", "close", "volume", "total_turnover"})
BLOCKED_FEATURES = frozenset({"num_trades", "s_dq_freeturnover"})
RAW_COLUMNS = ("datetime", "code", "open", "close", "high", "low", "volume", "amount")
_FEATURE_PATTERN = re.compile(r"\$([A-Za-z_]\w*)")
_OPERATOR_PATTERN = re.compile(r"(?<![\w$])([A-Za-z_]\w*)\s*\(")


class QuantMLDataQualityError(ValueError):
    """Raised when minute data violates the adapted 240-bar session contract."""


class QuantMLExpressionError(ValueError):
    """Raised when a catalog expression is outside the safe DSL."""


@dataclass(frozen=True)
class QuantMLFactor:
    """One stable adapted factor definition."""

    name: str
    expression: str
    source_rows: Tuple[int, ...]
    input_fields: Tuple[str, ...]
    operators: Tuple[str, ...]
    required_capabilities: Tuple[str, ...]
    available: bool


def _build_catalog() -> Tuple[QuantMLFactor, ...]:
    factors = []
    for name, expression, source_rows in CATALOG_ROWS:
        fields = tuple(sorted(set(_FEATURE_PATTERN.findall(expression))))
        operators = tuple(sorted(set(_OPERATOR_PATTERN.findall(expression))))
        missing = tuple(sorted(set(fields) - AVAILABLE_FEATURES))
        factors.append(
            QuantMLFactor(
                name=name,
                expression=expression,
                source_rows=source_rows,
                input_fields=fields,
                operators=operators,
                required_capabilities=missing,
                available=not missing,
            )
        )
    return tuple(factors)


_FACTORS = _build_catalog()
_FACTOR_INDEX = {factor.name: factor for factor in _FACTORS}


def list_quantml_factors(available: Optional[bool] = None) -> Tuple[QuantMLFactor, ...]:
    """List adapted factors in stable catalog order."""

    if available is None:
        return _FACTORS
    return tuple(factor for factor in _FACTORS if factor.available is available)


@dataclass(frozen=True)
class QuantMLDataContext:
    """Filesystem context for minute input and materialized daily cache."""

    minute_uri: Optional[Union[str, Path]] = None
    cache_uri: Optional[Union[str, Path]] = None
    adjustment_uri: Optional[Union[str, Path]] = None
    adjustment_mode: str = "none"

    def minute_path(self) -> Path:
        raw = self.minute_uri or os.environ.get("QLIB_QUANTML_MINUTE_URI")
        if raw is None:
            raise ValueError("minute_uri or QLIB_QUANTML_MINUTE_URI is required")
        path = Path(raw).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"QuantML minute root does not exist: {path}")
        return path

    def cache_path(self) -> Path:
        raw = self.cache_uri or os.environ.get("QLIB_QUANTML_CACHE_URI")
        if raw is None:
            raw = Path.home() / ".cache" / "qlib" / "quantml_adapted"
        return Path(raw).expanduser().resolve()

    def adjustment_path(self) -> Optional[Path]:
        mode = self.adjustment_mode.lower()
        if mode == "none":
            return None
        if mode != "hfq":
            raise ValueError(f"unsupported QuantML adjustment mode: {self.adjustment_mode!r}")
        raw = self.adjustment_uri or os.environ.get("QLIB_QUANTML_ADJUSTMENT_URI")
        if raw is None:
            raise ValueError("adjustment_uri or QLIB_QUANTML_ADJUSTMENT_URI is required for hfq mode")
        path = Path(raw).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"QuantML adjustment root does not exist: {path}")
        return path


def _session_times() -> Tuple[str, ...]:
    morning = pd.date_range("09:31", "11:30", freq="min").strftime("%H:%M").tolist()
    afternoon = pd.date_range("13:01", "15:00", freq="min").strftime("%H:%M").tolist()
    return tuple(morning + afternoon)


EXPECTED_SESSION_TIMES = _session_times()


def _feature_window(feature: Expression) -> Tuple[int, int]:
    return feature.get_extended_window_size()


class _UnaryAdaptedOperator(ExpressionOps):
    def __init__(self, feature: Expression, period: int):
        if not isinstance(period, int) or period <= 0:
            raise ValueError("period must be a positive integer")
        self.feature = feature
        self.period = period

    def get_longest_back_rolling(self):
        return self.feature.get_longest_back_rolling() + self.period - 1

    def get_extended_window_size(self):
        left, right = _feature_window(self.feature)
        return left + self.period - 1, right


class DownStd(_UnaryAdaptedOperator):
    """Rolling RMS of finite negative values."""

    def __str__(self):
        return f"DownStd({self.feature},{self.period})"

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)

        def downside(values):
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                return np.nan
            selected = finite[finite < 0]
            return 0.0 if selected.size == 0 else float(np.sqrt(np.mean(selected**2)))

        return series.rolling(self.period, min_periods=1).apply(downside, raw=True)


class UpStd(_UnaryAdaptedOperator):
    """Rolling RMS of finite positive values."""

    def __str__(self):
        return f"UpStd({self.feature},{self.period})"

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)

        def upside(values):
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                return np.nan
            selected = finite[finite > 0]
            return 0.0 if selected.size == 0 else float(np.sqrt(np.mean(selected**2)))

        return series.rolling(self.period, min_periods=1).apply(upside, raw=True)


class Peak(_UnaryAdaptedOperator):
    """Rolling maximum minus the current value."""

    def __str__(self):
        return f"Peak({self.feature},{self.period})"

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        return series.rolling(self.period, min_periods=1).max() - series


def _validate_session_operator(period: int, start: Optional[int] = None, end: Optional[int] = None) -> None:
    if period != 240:
        raise ValueError("QuantML adapted session operators only support period=240")
    if start is not None and end is not None and not 0 <= start < end <= period:
        raise ValueError(f"invalid end-exclusive session interval: [{start}, {end})")


def _session_groups(series: pd.Series):
    if not isinstance(series.index, pd.DatetimeIndex):
        raise QuantMLDataQualityError("session operators require a DatetimeIndex")
    return series.groupby(series.index.normalize(), sort=True)


class DownResample(ExpressionOps):
    """Emit a selector from non-overlapping intraday blocks."""

    def __init__(self, feature: Expression, period: int, selector: Union[int, str]):
        if not isinstance(period, int) or period <= 0 or 240 % period != 0:
            raise ValueError("DownResample period must be a positive divisor of 240")
        if isinstance(selector, str) and selector not in {"last", "mean"}:
            raise ValueError("DownResample string selector must be 'last' or 'mean'")
        if isinstance(selector, int) and not 0 <= selector < period:
            raise ValueError(f"DownResample integer selector must be in [0, {period - 1}]")
        if not isinstance(selector, (int, str)):
            raise ValueError("DownResample selector must be 'last', 'mean', or an integer")
        self.feature = feature
        self.period = period
        self.selector = selector

    def __str__(self):
        return f"DownResample({self.feature},{self.period},{self.selector!r})"

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        result = pd.Series(np.nan, index=series.index, dtype=float)
        for _, group in _session_groups(series):
            available = group.dropna()
            if available.empty:
                continue
            if len(available) % self.period != 0:
                continue
            for block_start in range(0, len(available), self.period):
                block = available.iloc[block_start : block_start + self.period]
                if self.selector == "last":
                    value = block.iloc[-1]
                elif self.selector == "mean":
                    value = block.mean()
                else:
                    value = block.iloc[self.selector]
                result.loc[block.index[-1]] = value
        return result

    def get_longest_back_rolling(self):
        return self.feature.get_longest_back_rolling()

    def get_extended_window_size(self):
        return _feature_window(self.feature)


class Partial(ExpressionOps):
    """Session-local end-exclusive interval mean emitted at the final bar."""

    def __init__(self, feature: Expression, period: int, aggregation: str, start: int, end: int):
        _validate_session_operator(period, start, end)
        if aggregation != "mean":
            raise ValueError("Partial only supports aggregation='mean'")
        self.feature = feature
        self.period = period
        self.aggregation = aggregation
        self.start = start
        self.end = end

    def __str__(self):
        return f"Partial({self.feature},{self.period},'{self.aggregation}',{self.start},{self.end})"

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        result = pd.Series(np.nan, index=series.index, dtype=float)
        for _, group in _session_groups(series):
            if len(group) != self.period:
                raise QuantMLDataQualityError(f"Partial received a {len(group)}-bar session")
            result.loc[group.index[-1]] = group.iloc[self.start : self.end].mean()
        return result

    def get_longest_back_rolling(self):
        return self.feature.get_longest_back_rolling()

    def get_extended_window_size(self):
        return _feature_window(self.feature)


class PartialRatio(ExpressionOps):
    """Session-local segment sum divided by the full-session sum."""

    def __init__(self, feature: Expression, period: int, aggregation: str, start: int, end: int):
        _validate_session_operator(period, start, end)
        if aggregation != "sum":
            raise ValueError("PartialRatio only supports aggregation='sum'")
        self.feature = feature
        self.period = period
        self.aggregation = aggregation
        self.start = start
        self.end = end

    def __str__(self):
        return f"PartialRatio({self.feature},{self.period},'{self.aggregation}',{self.start},{self.end})"

    def _load_internal(self, instrument, start_index, end_index, *args):
        series = self.feature.load(instrument, start_index, end_index, *args)
        result = pd.Series(np.nan, index=series.index, dtype=float)
        for _, group in _session_groups(series):
            if len(group) != self.period:
                raise QuantMLDataQualityError(f"PartialRatio received a {len(group)}-bar session")
            denominator = group.sum(min_count=1)
            if pd.isna(denominator) or denominator == 0:
                value = np.nan
            else:
                value = group.iloc[self.start : self.end].sum(min_count=1) / denominator
            result.loc[group.index[-1]] = value
        return result

    def get_longest_back_rolling(self):
        return self.feature.get_longest_back_rolling()

    def get_extended_window_size(self):
        return _feature_window(self.feature)


def get_quantml_custom_ops() -> Tuple[type, ...]:
    """Return the adapted classes for Qlib's supported custom_ops hook."""

    return DownResample, PartialRatio, Partial, DownStd, UpStd, Peak


class _SeriesFeature(Expression):
    def __init__(self, field: str, series: pd.Series, context_key: str):
        self.field = field
        self.series = series
        self.context_key = context_key

    def __str__(self):
        return f"QuantMLFeature('{self.field}','{self.context_key}')"

    def _load_internal(self, instrument, start_index, end_index, *args):
        if start_index is None or end_index is None:
            return self.series.copy()
        return self.series.iloc[start_index : end_index + 1].copy()

    def get_longest_back_rolling(self):
        return 0

    def get_extended_window_size(self):
        return 0, 0


_CALLS = {
    "Ref": Ref,
    "Corr": Corr,
    "Max": Max,
    "Min": Min,
    "Slope": Slope,
    "Resi": Resi,
    "Rsquare": Rsquare,
    "Std": Std,
    "Kurt": Kurt,
    "Skew": Skew,
    "Med": Med,
    "Mad": Mad,
    "Rank": Rank,
    "DownResample": DownResample,
    "PartialRatio": PartialRatio,
    "Partial": Partial,
    "DownStd": DownStd,
    "UpStd": UpStd,
    "Peak": Peak,
}
_BINARY_OPS = {ast.Add: Add, ast.Sub: Sub, ast.Mult: Mul, ast.Div: Div, ast.Pow: Power}


class _ExpressionCompiler:
    def __init__(self, features: Mapping[str, pd.Series], context_key: str):
        self.features = features
        self.context_key = context_key

    @staticmethod
    def parse(expression: str) -> ast.Expression:
        rewritten = _FEATURE_PATTERN.sub(lambda match: f'F("{match.group(1)}")', expression)
        try:
            return ast.parse(rewritten, mode="eval")
        except SyntaxError as exc:
            raise QuantMLExpressionError(f"invalid QuantML expression: {expression}") from exc

    def compile(self, expression: str):
        return self._compile_node(self.parse(expression).body)

    def _compile_node(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str)):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return Sub(0, self._compile_node(node.operand))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
            return self._compile_node(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
            return _BINARY_OPS[type(node.op)](
                self._compile_node(node.left),
                self._compile_node(node.right),
            )
        if isinstance(node, ast.Call):
            if node.keywords or not isinstance(node.func, ast.Name):
                raise QuantMLExpressionError("only positional calls to registered operators are allowed")
            name = node.func.id
            args = [self._compile_node(arg) for arg in node.args]
            if name == "F":
                if len(args) != 1 or not isinstance(args[0], str):
                    raise QuantMLExpressionError("F requires one literal feature name")
                field = args[0]
                if field not in self.features:
                    raise FactorDependencyError(field, [field])
                return _SeriesFeature(field, self.features[field], self.context_key)
            if name in {"IdxMax", "IdxMin"}:
                native = IdxMax(*args) if name == "IdxMax" else IdxMin(*args)
                return Sub(native, 1)
            operator = _CALLS.get(name)
            if operator is None:
                raise QuantMLExpressionError(f"operator is not allowed: {name}")
            try:
                return operator(*args)
            except (TypeError, ValueError) as exc:
                raise QuantMLExpressionError(f"invalid arguments for {name}: {args!r}") from exc
        raise QuantMLExpressionError(f"expression node is not allowed: {type(node).__name__}")


def _resolve_factors(factors: Union[str, Sequence[str]]) -> Tuple[QuantMLFactor, ...]:
    if factors == "all_supported":
        selected = tuple(factor for factor in _FACTORS if factor.available)
    elif isinstance(factors, str):
        selected = (_FACTOR_INDEX.get(factors),)
    else:
        selected = tuple(_FACTOR_INDEX.get(name) for name in factors)
    if not selected or any(factor is None for factor in selected):
        unknown = factors if isinstance(factors, str) else [name for name in factors if name not in _FACTOR_INDEX]
        raise KeyError(f"unknown or empty QuantML factor selection: {unknown}")
    for factor in selected:
        if factor.required_capabilities:
            raise FactorDependencyError(factor.name, factor.required_capabilities)
    return selected


def _lookback_minutes(factors: Sequence[QuantMLFactor]) -> int:
    maximum = 0
    for factor in factors:
        values = [int(value) for value in re.findall(r"\b\d+\b", factor.expression)]
        maximum = max(maximum, max(values, default=0) + 30)
    return maximum


def _available_dates(root: Path) -> Tuple[pd.Timestamp, ...]:
    dates = []
    for path in root.glob("date=????-??-??"):
        try:
            dates.append(pd.Timestamp(path.name.removeprefix("date=")))
        except ValueError:
            continue
    return tuple(sorted(dates))


def _select_dates(root: Path, start_time, end_time, warmup_sessions: int):
    start = pd.Timestamp(start_time).normalize()
    end = pd.Timestamp(end_time).normalize()
    if start > end:
        raise ValueError("start_time must not be later than end_time")
    dates = _available_dates(root)
    requested = tuple(date for date in dates if start <= date <= end)
    if not requested:
        raise FileNotFoundError(f"no minute partitions between {start.date()} and {end.date()}")
    first = dates.index(requested[0])
    if first < warmup_sessions:
        raise QuantMLDataQualityError(
            f"insufficient minute warm-up before {requested[0].date()}: "
            f"required {warmup_sessions} sessions, found {first}"
        )
    warmup = dates[max(0, first - warmup_sessions) : first]
    return tuple(warmup) + requested, requested


def _normalize_instrument(instrument: str) -> str:
    value = instrument.lower()
    if re.fullmatch(r"(?:sh|sz)\d{6}", value):
        return value
    if re.fullmatch(r"\d{6}\.(?:xshg|xshe)", value):
        code, exchange = value.split(".")
        return ("sh" if exchange == "xshg" else "sz") + code
    raise ValueError(f"unsupported A-share instrument code: {instrument!r}")


def _display_instrument(instrument: str) -> str:
    return instrument.upper()


def _adjustment_file(root: Path, instrument: str) -> Path:
    exchange = "SH" if instrument.startswith("sh") else "SZ"
    return root / f"{instrument[2:]}.{exchange}.csv"


def _apply_hfq(frame: pd.DataFrame, instrument: str, root: Path) -> pd.DataFrame:
    path = _adjustment_file(root, instrument)
    if not path.is_file():
        raise QuantMLDataQualityError(f"missing hfq adjustment file for {instrument}: {path}")
    factors = pd.read_csv(path, usecols=["交易日期", "复权因子"], parse_dates=["交易日期"])
    factors = factors.drop_duplicates("交易日期", keep="last").set_index("交易日期")["复权因子"]
    dates = pd.DatetimeIndex(frame.index.normalize())
    values = factors.reindex(dates).to_numpy(dtype=float)
    valid = np.isfinite(values) & (values > 0)
    if not valid.all():
        missing_dates = sorted(set(dates[~valid].strftime("%Y-%m-%d")))
        preview = ", ".join(missing_dates[:5])
        raise QuantMLDataQualityError(
            f"missing or non-positive hfq adjustment for {instrument}: {preview}"
        )
    adjusted = frame.copy()
    for column in ("open", "high", "low", "close"):
        adjusted[column] = adjusted[column].astype(float).to_numpy() * values
    return adjusted


def _read_partition(path: Path, instruments: Sequence[str]) -> pd.DataFrame:
    files = tuple(path.glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=RAW_COLUMNS)
    dataset = pyarrow_dataset.dataset([str(file) for file in files], format="parquet")
    missing = set(RAW_COLUMNS) - set(dataset.schema.names)
    if missing:
        raise QuantMLDataQualityError(f"minute schema is missing columns: {sorted(missing)}")
    table = dataset.to_table(
        columns=list(RAW_COLUMNS),
        filter=pyarrow_dataset.field("code").isin(list(instruments)),
    )
    return table.to_pandas()


def _normalize_session(frame: pd.DataFrame, instrument: str, date: pd.Timestamp) -> pd.DataFrame:
    frame = frame.sort_values("datetime").copy()
    if frame["datetime"].duplicated().any():
        raise QuantMLDataQualityError(f"duplicate minute timestamps for {instrument} on {date.date()}")
    frame = frame[frame["datetime"].dt.strftime("%H:%M") != "09:30"]
    actual_times = tuple(frame["datetime"].dt.strftime("%H:%M"))
    if actual_times != EXPECTED_SESSION_TIMES:
        raise QuantMLDataQualityError(
            f"invalid minute session for {instrument} on {date.date()}: expected 240 bars, got {len(frame)}"
        )
    if (frame["volume"] < 0).any() or (frame["amount"] < 0).any():
        raise QuantMLDataQualityError(f"negative volume or amount for {instrument} on {date.date()}")
    frame = frame.set_index("datetime")
    normalized = frame[["open", "high", "low", "close"]].astype(float)
    normalized["volume"] = frame["volume"].astype(float) * 100.0
    normalized["total_turnover"] = frame["amount"].astype(float)
    return normalized


def _load_minute_frames(
    root: Path,
    instruments: Sequence[str],
    load_dates: Sequence[pd.Timestamp],
    adjustment_root: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    parts: Dict[str, list[pd.DataFrame]] = {instrument: [] for instrument in instruments}
    for date in load_dates:
        partition = root / f"date={date:%Y-%m-%d}"
        raw = _read_partition(partition, instruments)
        if raw.empty:
            continue
        raw["code"] = raw["code"].str.lower()
        for instrument, group in raw.groupby("code", sort=False):
            if instrument in parts:
                parts[instrument].append(_normalize_session(group, instrument, date))
    result = {
        instrument: pd.concat(frames).sort_index() if frames else pd.DataFrame(columns=tuple(AVAILABLE_FEATURES))
        for instrument, frames in parts.items()
    }
    if adjustment_root is not None:
        result = {
            instrument: _apply_hfq(frame, instrument, adjustment_root) if not frame.empty else frame
            for instrument, frame in result.items()
        }
    return result


def _source_signature(root: Path, dates: Sequence[pd.Timestamp]) -> dict:
    count = 0
    total_size = 0
    latest_mtime_ns = 0
    for date in dates:
        for path in (root / f"date={date:%Y-%m-%d}").glob("*.parquet"):
            stat = path.stat()
            count += 1
            total_size += stat.st_size
            latest_mtime_ns = max(latest_mtime_ns, stat.st_mtime_ns)
    return {"file_count": count, "total_size": total_size, "latest_mtime_ns": latest_mtime_ns}


def _qlib_revision() -> str:
    """Read the local Git revision without invoking a subprocess."""

    repository = Path(__file__).resolve().parents[3]
    git_path = repository / ".git"
    if git_path.is_file():
        marker = git_path.read_text(encoding="utf-8").strip()
        if marker.startswith("gitdir:"):
            git_path = (repository / marker.split(":", 1)[1].strip()).resolve()
    head_path = git_path / "HEAD"
    if not head_path.is_file():
        return "unknown"
    head = head_path.read_text(encoding="utf-8").strip()
    if not head.startswith("ref:"):
        return head
    reference = head.split(":", 1)[1].strip()
    reference_path = git_path / reference
    if reference_path.is_file():
        return reference_path.read_text(encoding="utf-8").strip()
    packed_refs = git_path / "packed-refs"
    if packed_refs.is_file():
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if line.endswith(f" {reference}"):
                return line.split(" ", 1)[0]
    return "unknown"


def _cache_identity(
    root: Path,
    selected: Sequence[QuantMLFactor],
    instruments: Sequence[str],
    start_time,
    end_time,
    signature: Mapping[str, int],
    adjustment_mode: str = "none",
    adjustment_signature: Optional[Mapping[str, object]] = None,
) -> Tuple[str, dict]:
    payload = {
        "catalog_version": CATALOG_VERSION,
        "catalog_sha256": SOURCE_SHA256,
        "operator_version": OPERATOR_VERSION,
        "qlib_revision": _qlib_revision(),
        "minute_uri": str(root),
        "factors": [factor.name for factor in selected],
        "instruments": list(instruments),
        "start_time": str(pd.Timestamp(start_time).date()),
        "end_time": str(pd.Timestamp(end_time).date()),
        "source_signature": dict(signature),
        "adjustment_mode": adjustment_mode,
        "adjustment_signature": dict(adjustment_signature or {}),
        "source_schema": list(RAW_COLUMNS),
        "field_mapping": {"volume": "volume*100", "total_turnover": "amount"},
        "session": "09:31-11:30,13:01-15:00",
        "timezone": "Asia/Shanghai",
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest, payload


def _adjustment_signature(root: Optional[Path], instruments: Sequence[str]) -> dict:
    if root is None:
        return {}
    count = 0
    total_size = 0
    latest_mtime_ns = 0
    missing = []
    for instrument in instruments:
        path = _adjustment_file(root, instrument)
        if not path.is_file():
            missing.append(instrument)
            continue
        stat = path.stat()
        count += 1
        total_size += stat.st_size
        latest_mtime_ns = max(latest_mtime_ns, stat.st_mtime_ns)
    return {
        "root": str(root),
        "file_count": count,
        "total_size": total_size,
        "latest_mtime_ns": latest_mtime_ns,
        "missing_instruments": missing,
    }


def _evaluate_instrument(
    instrument: str,
    frame: pd.DataFrame,
    selected: Sequence[QuantMLFactor],
    requested_dates: Sequence[pd.Timestamp],
    context_seed: str,
) -> pd.DataFrame:
    columns = [factor.name for factor in selected]
    if frame.empty:
        return pd.DataFrame(index=pd.DatetimeIndex(requested_dates, name="datetime"), columns=columns, dtype=float)
    context_key = hashlib.sha1(
        f"{context_seed}|{instrument}|{frame.index[0]}|{frame.index[-1]}|{len(frame)}".encode("utf-8")
    ).hexdigest()[:16]
    compiler = _ExpressionCompiler({column: frame[column] for column in frame.columns}, context_key)
    values = {}
    for factor in selected:
        expression = compiler.compile(factor.expression)
        series = expression.load(instrument, 0, len(frame) - 1, "1min")
        daily = series.groupby(series.index.normalize()).last()
        values[factor.name] = daily.reindex(requested_dates)
    result = pd.DataFrame(values, index=pd.DatetimeIndex(requested_dates, name="datetime"))
    return result[columns].replace([np.inf, -np.inf], np.nan)


def compute_quantml_factors(
    factors: Union[str, Sequence[str]],
    instruments: Sequence[str],
    start_time,
    end_time,
    context: Optional[QuantMLDataContext] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Compute adapted minute factors and return daily `(datetime, instrument)` data."""

    selected = _resolve_factors(factors)
    if isinstance(instruments, str) or not instruments:
        raise TypeError("instruments must be a non-empty sequence of explicit A-share codes")
    normalized = tuple(dict.fromkeys(_normalize_instrument(instrument) for instrument in instruments))
    context = context or QuantMLDataContext()
    root = context.minute_path()
    adjustment_root = context.adjustment_path()
    warmup_sessions = max(1, math.ceil(_lookback_minutes(selected) / 240))
    load_dates, requested_dates = _select_dates(root, start_time, end_time, warmup_sessions)
    signature = _source_signature(root, load_dates)
    adjustment_signature = _adjustment_signature(adjustment_root, normalized)
    cache_key, manifest = _cache_identity(
        root,
        selected,
        normalized,
        start_time,
        end_time,
        signature,
        context.adjustment_mode.lower(),
        adjustment_signature,
    )
    cache_path = context.cache_path()
    data_path = cache_path / f"{cache_key}.parquet"
    manifest_path = cache_path / f"{cache_key}.json"
    if use_cache and data_path.is_file() and manifest_path.is_file():
        cached = pd.read_parquet(data_path)
        return cached.sort_index()

    frames = _load_minute_frames(root, normalized, load_dates, adjustment_root=adjustment_root)
    pieces = []
    for instrument in normalized:
        result = _evaluate_instrument(
            instrument,
            frames[instrument],
            selected,
            requested_dates,
            context_seed=cache_key,
        )
        result["instrument"] = _display_instrument(instrument)
        result = result.set_index("instrument", append=True)
        pieces.append(result)
    output = pd.concat(pieces).sort_index()
    output.index.names = ["datetime", "instrument"]

    if use_cache:
        cache_path.mkdir(parents=True, exist_ok=True)
        manifest.update(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "row_count": len(output),
                "column_count": len(output.columns),
                "blocked_fields": sorted(BLOCKED_FEATURES),
            }
        )
        temporary_id = uuid.uuid4().hex
        temporary_data = cache_path / f".{cache_key}.{temporary_id}.parquet.tmp"
        temporary_manifest = cache_path / f".{cache_key}.{temporary_id}.json.tmp"
        try:
            output.to_parquet(temporary_data)
            temporary_manifest.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary_data.replace(data_path)
            temporary_manifest.replace(manifest_path)
        finally:
            temporary_data.unlink(missing_ok=True)
            temporary_manifest.unlink(missing_ok=True)
    return output


__all__ = [
    "DownResample",
    "DownStd",
    "OPERATOR_VERSION",
    "Partial",
    "PartialRatio",
    "Peak",
    "QuantMLDataContext",
    "QuantMLDataQualityError",
    "QuantMLExpressionError",
    "QuantMLFactor",
    "UpStd",
    "compute_quantml_factors",
    "get_quantml_custom_ops",
    "list_quantml_factors",
]
