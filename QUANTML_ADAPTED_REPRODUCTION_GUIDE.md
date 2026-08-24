# QuantML-Adapted Factors: Qlib Reproduction Guide

## Purpose

This document is the implementation handoff for reproducing the local
QuantML-adapted minute-factor extension on Microsoft Qlib `v0.9.7`. It is
written for another Codex session or engineer who has a legally obtained copy
of the source `factors.md` table and compatible A-share minute data.

The catalog is adapted because the original QuantML `qlib_csc` implementation
is unavailable. Historical IC or group-return similarity is evidence about the
adaptation, not proof of formula identity.

## Fixed Baseline

- Upstream: Microsoft Qlib `v0.9.7`.
- Python: 3.10.
- Feature input: one-minute A-share OHLCV and amount.
- Feature output: one value per `(datetime, instrument)` trading day.
- Time zone: `Asia/Shanghai`.
- Session: remove 09:30 and require exactly 240 bars at
  09:31-11:30 and 13:01-15:00.
- Volume: raw lots multiplied by 100 to obtain shares.
- `$total_turnover`: raw `amount` in CNY.
- Adjustment: positive daily HFQ factors multiply minute OHLC; volume and
  amount remain unchanged.
- Label: `Ref($close,-6)/Ref($close,-1)-1` from a Qlib daily provider.

The verified source table has SHA256:

```text
b5e07f40ee97a9b7df14635226f05851611bfc1927b167d28a453cb249867b29
```

It contains 1,049 rows, 994 unique expressions, and 55 duplicate rows. The
current data contract supports 881 unique factors. Fifty-six factors require
`$num_trades`; 57 require `$s_dq_freeturnover`. Blocked factors must fail with
`FactorDependencyError`; never delete or proxy their missing inputs.

## Files Added or Changed

Do not copy isolated snippets. Preserve the contracts across all files below.

| File | Responsibility |
|---|---|
| `qlib/contrib/data/_quantml_catalog.py` | Generated stable `QMLA0001`-`QMLA0994` catalog, source row mapping, and provenance constants. Never hand-edit IDs. |
| `qlib/contrib/data/quantml_factors.py` | Safe AST compiler, native Qlib operator reuse, six adapted operators, minute reader, HFQ handling, cache identity, infinity-to-NaN normalization, and Direct API. |
| `qlib/contrib/data/loader.py` | `QuantMLAdaptedDL`, materialized Parquet loading, daily Qlib labels, manifest validation, and point-in-time market membership. |
| `qlib/contrib/data/handler.py` | `QuantMLAdapted` `DataHandlerLP` wrapper for `DatasetH` and model workflows. |
| `qlib/contrib/data/dolphindb_compat.py` | Shared `FactorDependencyError` and dependency registry behavior. |
| `scripts/generate_quantml_catalog.py` | Deterministic generator from the trusted `factors.md` source. |
| `tests/data_mid_layer_tests/test_quantml_catalog.py` | Catalog counts, aliases, dependencies, and source hash. |
| `tests/ops/test_quantml_adapted_ops.py` | Adapted-operator endpoints, windows, zero denominator, and AST safety. |
| `tests/data_mid_layer_tests/test_quantml_real_data.py` | Real minute schema/session checks and Direct API/Loader/Handler parity. |
| `tests/data_mid_layer_tests/test_quantml_workflow.py` | HFQ, materialized-store manifest, dynamic membership, labels, and infinity policy. |

Existing Alpha158, Alpha360, and native `qlib.data.ops` behavior must remain
unchanged.

## Catalog Generation

Stable IDs follow first occurrence in the source table and may only be
appended in future versions. Regenerate the catalog only from the exact source
hash:

```powershell
python scripts/generate_quantml_catalog.py `
  D:\path\to\factors.md `
  qlib\contrib\data\_quantml_catalog.py
```

The generator must reject any source with a different SHA, row count, or
unique-expression count. Duplicate source rows share one `QMLA` ID and retain
all source line numbers.

## Expression Engine

The compiler accepts only registered fields, numeric/string literals,
arithmetic, and registered operator calls. Never use `eval()` or arbitrary
Python execution.

Reuse these Qlib expression classes directly:

```text
Ref Corr Max Min Slope Resi Rsquare Std Kurt Skew
Med Mad Rank IdxMax IdxMin
```

Compile catalog `IdxMax` and `IdxMin` as the native result minus one because
the source expressions use zero-based positions.

The only adapted operators are:

```text
DownResample Partial PartialRatio DownStd UpStd Peak
```

Rolling expressions run continuously across trading minutes and do not reset
at the day boundary. `Partial` and `PartialRatio` use session-local,
end-exclusive intervals. Positive and negative infinity are normalized to NaN
at the shared factor-output boundary. This is required for `QMLA0008` when
`239 - IdxMax(...)` is zero.

## Public APIs

```python
from qlib.contrib.data.quantml_factors import (
    QuantMLDataContext,
    compute_quantml_factors,
    list_quantml_factors,
)
from qlib.contrib.data.loader import QuantMLAdaptedDL
from qlib.contrib.data.handler import QuantMLAdapted
```

Direct calculation requires an explicit list or `"all_supported"`:

```python
context = QuantMLDataContext(
    minute_uri=r"D:\data\minute",
    cache_uri=r"D:\cache\quantml",
    adjustment_uri=r"D:\data\hfq",
    adjustment_mode="hfq",
)

features = compute_quantml_factors(
    factors=["QMLA0001", "QMLA0002"],
    instruments=["SH600000"],
    start_time="2023-12-25",
    end_time="2023-12-29",
    context=context,
)
```

The result must be a sorted, unique `(datetime, instrument)` MultiIndex
`DataFrame`.

For a Qlib workflow, materialize daily factor shards first and configure:

```python
handler = QuantMLAdapted(
    instruments="csi300",
    factors="all_supported",
    start_time="2023-01-03",
    end_time="2023-09-28",
    materialized_uri=r"D:\cache\factor_store",
    label=["Ref($close,-6)/Ref($close,-1)-1"],
    label_names=["LABEL0"],
    infer_processors=[],
    learn_processors=["DropnaLabel"],
)
```

A Qlib market name must retain its point-in-time membership intervals. Do not
expand `csi300` to an 800-stock historical union and use every stock on every
date. Reindex suspensions to NaN so the historical CSI300 grid contains exactly
300 instruments per trading day.

## Cache and Materialized Store

Keep data and generated factors outside the Git repository. Cache identity
must include:

- catalog SHA and operator version;
- Qlib commit;
- minute source path, file count, total size, and latest mtime;
- adjustment path and file signature;
- source and evaluation dates;
- requested factors and instruments;
- market membership hash;
- schema, units, session, and time zone.

A materialized directory requires `_MANIFEST.json`. Reject stale catalog or
operator versions and duplicate `(datetime, instrument)` rows. Partial stores
are resumable per stock/year, but a cache without its success/identity marker
must not be trusted.

## Full-Market Metric Reproduction

Do not compute 1,049 rows independently from raw minute files. That repeats
Parquet scans and rolling work. Use this pipeline:

1. Scan each 2016-2023 date partition once and create a filtered
   `year x code-hash-bucket` minute cache.
2. In 16 worker processes, compute all supported expressions one stock/year at
   a time with Qlib Expression objects and write `float32` daily Parquet shards.
3. Build the future-five-day label from the Qlib daily provider.
4. Read only one factor or a small 50-factor column batch from the shards.
5. Accumulate daily cross-sectional statistics and checkpoint every batch.
6. Restore the original 1,049-row source order. Reuse results for duplicate
   expressions and write NaN for the 113 blocked rows.

For the requested eight-year evaluation, every eligible date from 2016 through
2023 participates in metrics. This differs from the previously inferred
2020-2023 reference window and must be stated in the output manifest.

The Markdown table preserves these exact columns:

```text
factor_name ic_mean icir long-average long-short
signic0 signic004 t Group1 Group2 Group3 Group4 Group5
```

Definitions:

- daily Pearson IC across instruments;
- `ic_mean = mean(daily_IC)`;
- `icir = abs(mean(daily_IC) / std(daily_IC))`;
- `signic0 = mean(sign(ic_mean) * daily_IC > 0)`;
- `signic004 = mean(sign(ic_mean) * daily_IC > 0.04)`;
- descending-score quintiles `Group1` through `Group5`;
- `long-short = (Group1 - Group5) / 2`, matching Qlib's default convention;
- `long-average = Group1 - full-cross-section mean(label)`;
- `t` is the one-sample t statistic of daily long-short returns.

The first value returned by Qlib's `calc_long_short_return` uses the required
`/2` convention. Its second value is the full-cross-section average label, not
the required `Group1 - average(label)`, so the complete table still needs the
standard evaluator in `qlib.contrib.eva.factor`.

The evaluator exposes:

```python
from qlib.contrib.eva.factor import (
    FACTOR_METRIC_COLUMNS,
    evaluate_factor,
    evaluate_factors,
    render_factor_metrics_markdown,
)
```

Write two outputs:

```text
factors_reproduced_2016_2023.md
factors_reproduced_2016_2023_status.parquet
```

The sidecar records `QMLA` ID, source rows, status, missing dependency,
non-null days, cross-sectional sample counts, cache identity, and generation
time.

## Verification

Run synthetic tests before any full-market job:

```powershell
python -m unittest discover -s tests\ops -p "test_quantml_adapted_ops.py" -v
python -m unittest discover -s tests\data_mid_layer_tests -p "test_quantml_catalog.py" -v
python -m unittest discover -s tests\data_mid_layer_tests -p "test_quantml_workflow.py" -v
python -m unittest discover -s tests\data_mid_layer_tests -p "test_quantml_real_data.py" -v
python -m compileall qlib\contrib\data
python -m pip check
```

Then run a bounded workflow gate with all 881 supported factors. The verified
2023 gate had 54,600 rows, exactly 300 CSI300 instruments per day, zero
duplicate indices, zero infinity, no all-NaN factor in any segment, and 11,400
finite LightGBM predictions. It completed `SignalRecord`, `SigAnaRecord`, and
`PortAnaRecord`.

Set Qlib `kernels=16` and LightGBM `num_threads=16` on a 16-core/32-GB host.
Leaving Qlib at the environment default previously spawned 32 provider workers
and exhausted memory. Current MLflow versions reject the legacy filesystem
tracking backend; use a local SQLite tracking URI and keep artifacts outside
the repository.

## Git and Distribution

- Keep Microsoft Qlib as the upstream remote.
- Push this customization only to the authorized private repository.
- Do not commit minute data, adjustment files, caches, MLflow databases,
  predictions, or factor result files.
- Preserve the Microsoft Qlib license and upstream history.
- Record the exact Qlib commit and catalog SHA in every research manifest.
- A historical IC or return result never proves formula equivalence or future
  performance.
