"""Standard single-factor evaluation for daily cross-sectional research."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


FACTOR_METRIC_COLUMNS = (
    "ic_mean",
    "icir",
    "long-average",
    "long-short",
    "signic0",
    "signic004",
    "t",
    "Group1",
    "Group2",
    "Group3",
    "Group4",
    "Group5",
)


def _as_series(value, name: str) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        if value.shape[1] != 1:
            raise ValueError(f"{name} must contain exactly one column")
        value = value.iloc[:, 0]
    if not isinstance(value, pd.Series):
        raise TypeError(f"{name} must be a pandas Series or one-column DataFrame")
    if not isinstance(value.index, pd.MultiIndex) or "datetime" not in value.index.names:
        raise ValueError(f"{name} must use a MultiIndex containing a datetime level")
    if value.index.has_duplicates:
        raise ValueError(f"{name} index must be unique")
    return value.sort_index()


def _daily_metrics(scores: pd.DataFrame, labels: pd.Series, min_obs: int):
    score_values = scores.to_numpy(dtype=float, copy=False)
    label_values = labels.to_numpy(dtype=float, copy=False)
    label_matrix = label_values[:, None]
    score_valid = np.isfinite(score_values)
    label_valid = np.isfinite(label_matrix)
    pair_valid = score_valid & label_valid

    pair_count = pair_valid.sum(axis=0)
    x = np.where(pair_valid, score_values, 0.0)
    y = np.where(pair_valid, label_matrix, 0.0)
    sum_x = x.sum(axis=0)
    sum_y = y.sum(axis=0)
    covariance = (x * y).sum(axis=0) - np.divide(
        sum_x * sum_y,
        pair_count,
        out=np.zeros_like(sum_x),
        where=pair_count > 0,
    )
    variance_x = (x * x).sum(axis=0) - np.divide(
        sum_x * sum_x,
        pair_count,
        out=np.zeros_like(sum_x),
        where=pair_count > 0,
    )
    variance_y = (y * y).sum(axis=0) - np.divide(
        sum_y * sum_y,
        pair_count,
        out=np.zeros_like(sum_y),
        where=pair_count > 0,
    )
    denominator = np.sqrt(np.maximum(variance_x * variance_y, 0.0))
    ic = np.divide(
        covariance,
        denominator,
        out=np.full(score_values.shape[1], np.nan),
        where=(pair_count >= min_obs) & (denominator > 0),
    )

    sortable = np.where(score_valid, score_values, -np.inf)
    order = np.argsort(-sortable, axis=0, kind="stable")
    ranks = np.empty_like(order)
    rank_values = np.broadcast_to(np.arange(len(scores))[:, None], order.shape)
    np.put_along_axis(ranks, order, rank_values, axis=0)
    group_size = score_valid.sum(axis=0) // 5
    groups = np.full((5, score_values.shape[1]), np.nan)
    for group_index in range(5):
        membership = (
            score_valid
            & (ranks >= group_size[None, :] * group_index)
            & (ranks < group_size[None, :] * (group_index + 1))
            & (group_size[None, :] > 0)
        )
        valid_labels = membership & label_valid
        counts = valid_labels.sum(axis=0)
        sums = np.where(valid_labels, label_matrix, 0.0).sum(axis=0)
        groups[group_index] = np.divide(
            sums,
            counts,
            out=np.full(score_values.shape[1], np.nan),
            where=counts > 0,
        )
    market_mean = np.nanmean(label_values) if np.isfinite(label_values).any() else np.nan
    long_short = (groups[0] - groups[4]) / 2.0
    long_average = groups[0] - market_mean
    return ic, groups, long_short, long_average, pair_count


def _evaluate_batch(features: pd.DataFrame, label: pd.Series, min_obs: int):
    daily_ic = []
    daily_groups = []
    daily_long_short = []
    daily_long_average = []
    daily_pair_count = []
    dates = []
    for date, day in features.groupby(level="datetime", sort=True):
        day = day.droplevel("datetime")
        try:
            day_label = label.xs(date, level="datetime", drop_level=True)
        except KeyError:
            day_label = pd.Series(dtype=float)
        common = day.index.union(day_label.index)
        scores = day.reindex(common)
        labels = day_label.reindex(common)
        ic, groups, long_short, long_average, pair_count = _daily_metrics(scores, labels, min_obs)
        dates.append(pd.Timestamp(date))
        daily_ic.append(ic)
        daily_groups.append(groups)
        daily_long_short.append(long_short)
        daily_long_average.append(long_average)
        daily_pair_count.append(pair_count)

    columns = list(features.columns)
    if not dates:
        empty = pd.DataFrame(np.nan, index=columns, columns=FACTOR_METRIC_COLUMNS)
        status = pd.DataFrame(
            {"non_null_days": 0, "average_cross_section": np.nan},
            index=columns,
        )
        return empty, status

    ic_frame = pd.DataFrame(daily_ic, index=dates, columns=columns)
    group_array = np.stack(daily_groups)
    long_short_frame = pd.DataFrame(daily_long_short, index=dates, columns=columns)
    long_average_frame = pd.DataFrame(daily_long_average, index=dates, columns=columns)
    pair_count_frame = pd.DataFrame(daily_pair_count, index=dates, columns=columns)
    ic_mean = ic_frame.mean()
    ic_std = ic_frame.std()

    metrics = pd.DataFrame(index=columns, columns=FACTOR_METRIC_COLUMNS, dtype=float)
    metrics["ic_mean"] = ic_mean
    metrics["icir"] = (ic_mean / ic_std).abs()
    metrics["long-average"] = long_average_frame.mean()
    metrics["long-short"] = long_short_frame.mean()
    metrics["signic0"] = [
        np.nan
        if not np.isfinite(ic_mean[column])
        else (ic_frame[column] * (1 if ic_mean[column] >= 0 else -1) > 0).mean()
        for column in columns
    ]
    metrics["signic004"] = [
        np.nan
        if not np.isfinite(ic_mean[column])
        else (ic_frame[column] * (1 if ic_mean[column] >= 0 else -1) > 0.04).mean()
        for column in columns
    ]
    metrics["t"] = [
        stats.ttest_1samp(long_short_frame[column].dropna(), 0).statistic
        if long_short_frame[column].notna().sum() > 1
        else np.nan
        for column in columns
    ]
    for group_index in range(5):
        values = group_array[:, group_index, :]
        counts = np.isfinite(values).sum(axis=0)
        metrics[f"Group{group_index + 1}"] = np.divide(
            np.nansum(values, axis=0),
            counts,
            out=np.full(values.shape[1], np.nan),
            where=counts > 0,
        )
    metrics = metrics.replace([np.inf, -np.inf], np.nan)

    status = pd.DataFrame(index=columns)
    status["non_null_days"] = (pair_count_frame >= min_obs).sum()
    status["average_cross_section"] = pair_count_frame.where(pair_count_frame > 0).mean()
    return metrics.loc[:, FACTOR_METRIC_COLUMNS], status


def evaluate_factor(score, label, min_obs: int = 10) -> pd.Series:
    """Evaluate one factor and return the standard metric vector."""

    score = _as_series(score, "score")
    label = _as_series(label, "label")
    name = score.name or "factor"
    metrics = evaluate_factors(score.to_frame(name), label, batch_size=1, min_obs=min_obs)
    result = metrics.iloc[0]
    result.name = name
    return result


def evaluate_factors(features, label, batch_size: int = 10, min_obs: int = 10) -> pd.DataFrame:
    """Evaluate factor columns in bounded batches using the standard metrics."""

    if isinstance(features, pd.Series):
        features = features.to_frame(features.name or "factor")
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame or Series")
    if not isinstance(features.index, pd.MultiIndex) or "datetime" not in features.index.names:
        raise ValueError("features must use a MultiIndex containing a datetime level")
    if features.index.has_duplicates:
        raise ValueError("features index must be unique")
    if features.columns.has_duplicates:
        raise ValueError("feature columns must be unique")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(min_obs, int) or min_obs < 2:
        raise ValueError("min_obs must be an integer of at least two")
    label = _as_series(label, "label")
    pieces = []
    for start in range(0, len(features.columns), batch_size):
        columns = list(features.columns[start : start + batch_size])
        metrics, _ = _evaluate_batch(features[columns].sort_index(), label, min_obs)
        pieces.append(metrics)
    if not pieces:
        return pd.DataFrame(columns=FACTOR_METRIC_COLUMNS, dtype=float)
    return pd.concat(pieces).reindex(features.columns).loc[:, FACTOR_METRIC_COLUMNS]


def render_factor_metrics_markdown(
    rows: pd.DataFrame,
    metrics: pd.DataFrame,
    statuses: Optional[pd.DataFrame] = None,
) -> str:
    """Render source-ordered factor rows with the standard metric columns."""

    required = {"factor_id", "factor_name"}
    if not required.issubset(rows.columns):
        raise ValueError(f"rows must contain columns: {sorted(required)}")
    if statuses is not None and "factor_id" not in statuses.columns and statuses.index.name != "factor_id":
        raise ValueError("statuses must contain or be indexed by factor_id")
    output = rows[["factor_id", "factor_name"]].copy()
    aligned = metrics.reindex(output["factor_id"]).reset_index(drop=True)
    for column in FACTOR_METRIC_COLUMNS:
        output[column] = aligned[column].to_numpy() if column in aligned else np.nan
    rendered = output[["factor_name", *FACTOR_METRIC_COLUMNS]].copy()
    for column in FACTOR_METRIC_COLUMNS:
        rendered[column] = rendered[column].map(
            lambda value: f"{value:.5f}" if np.isfinite(value) else "NaN"
        )
    return rendered.to_markdown(index=False, tablefmt="pipe", disable_numparse=True) + "\n"


__all__: Sequence[str] = (
    "FACTOR_METRIC_COLUMNS",
    "evaluate_factor",
    "evaluate_factors",
    "render_factor_metrics_markdown",
)
