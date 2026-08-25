# Qlib Factors888

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Qlib](https://img.shields.io/badge/Microsoft%20Qlib-v0.9.7-5C2D91)](https://github.com/microsoft/qlib/tree/v0.9.7)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-private%20research-orange)](#scope-and-provenance)

`Qlib Factors888` is a private A-share factor-research fork of Microsoft Qlib
`v0.9.7`. It embeds a reproducible QuantML-adapted minute-factor catalog into
Qlib's Expression, DataLoader, DataHandler, Dataset, model, and evaluation
workflow.

This repository is not an official Microsoft or QuantML release. The original
QuantML `qlib_csc` implementation is unavailable, so the formulas are explicitly
classified as **adapted**, not exact reproductions.

## What Is Included

- A stable catalog derived from 1,049 source rows:
  - 994 unique expressions;
  - 55 duplicate source rows;
  - stable IDs `QMLA0001` through `QMLA0994`.
- 881 factors supported by the current minute-data contract.
- 56 factors blocked by missing `$num_trades`.
- 57 factors blocked by missing `$s_dq_freeturnover`.
- A safe AST compiler with no arbitrary Python execution.
- Native Qlib operator reuse wherever semantics permit.
- Six adapted operators:

```text
DownResample Partial PartialRatio DownStd UpStd Peak
```

- Direct calculation, materialized Parquet loading, and Qlib Handler APIs.
- Point-in-time Qlib market membership instead of historical-union leakage.
- A standard single-factor evaluator producing IC, ICIR, group returns, and
  source-compatible Markdown tables.
- Provenance-keyed external caches and resumable stock/year factor shards.

Native Alpha158, Alpha360, and `qlib.data.ops` behavior remains unchanged.

## Verified Research Status

### Qlib workflow gate

The complete daily workflow has run successfully with all 881 supported
features:

```text
Minute features
→ QuantMLAdaptedDL / QuantMLAdapted
→ DatasetH
→ LightGBM
→ SignalRecord
→ SigAnaRecord
→ PortAnaRecord
```

Verified 2023 CSI300 gate:

- 54,600 `(datetime, instrument)` rows;
- exactly 300 point-in-time CSI300 members per day;
- 881 unique factor columns;
- no duplicate indices;
- no infinity;
- no all-NaN factor in train, validation, or test;
- 11,400 finite test predictions;
- all signal and portfolio artifacts generated and read back.

### 2016-2023 full-market first-50 pilot

The first 50 source-table rows were evaluated over all eligible A-share stock
days from 2016-01-04 through 2023-12-29:

- 1,945 trading days;
- 7,302,825 daily label rows;
- 47 supported factor rows;
- three `s_dq_freeturnover` rows rendered with NaN metrics;
- zero invalid minute sessions;
- zero infinite metric values;
- IC-sign agreement with the reference table: 40/47 (85.1%);
- long-short sign agreement: 41/47 (87.2%);
- cross-factor Pearson correlation:
  - `ic_mean`: 0.716;
  - `long-short`: 0.702.

These statistics support broad directional similarity, not numerical identity.
The local evaluation uses the full 2016-2023 period, while the source table's
exact period, universe, adjustment, and `qlib_csc` semantics are unknown.

## Important Metric Convention

The standard evaluator is available at `qlib.contrib.eva.factor`.

```text
ic_mean icir long-average long-short
signic0 signic004 t
Group1 Group2 Group3 Group4 Group5
```

This fork uses Qlib's default long-short convention:

```text
long-short = (Group1 - Group5) / 2
```

The supplied reference table uses the unhalved `Group1 - Group5` identity.
Magnitude comparisons must normalize that difference first. Historical output
files are never silently rewritten when a metric contract changes.

## Installation

Clone the private repository and install it in editable mode:

```powershell
git clone https://github.com/chenseldon/Qlib_factors888.git
cd Qlib_factors888

conda create -n Qlib_Stock python=3.10 -y
conda activate Qlib_Stock
python -m pip install -e .
python -m pip check
```

Verify that Python resolves this checkout:

```powershell
python -c "import qlib; print(qlib.__version__); print(qlib.__file__)"
```

The verified development environment uses Python `3.10.20` and Qlib `0.9.7`.

## Data Contract

Large datasets are not included in this repository. The minute input must be a
date-partitioned Parquet dataset:

```text
minute/
├── date=2023-12-28/
│   ├── data_0.parquet
│   └── ...
└── date=2023-12-29/
    └── ...
```

Required raw columns:

```text
datetime code open high low close volume amount
```

Normalization:

```text
$open/$high/$low/$close → corresponding OHLC column
$volume                 → raw volume × 100 shares
$total_turnover         → raw amount in CNY
```

Session contract:

- time zone: `Asia/Shanghai`;
- remove the extra 09:30 bar;
- retain 09:31-11:30 and 13:01-15:00;
- require exactly 240 unique bars;
- a completely absent session is a suspension and produces NaN;
- a partial session fails closed with `QuantMLDataQualityError`.

Optional HFQ adjustment files use:

```text
股票代码,交易日期,复权因子
```

Positive daily factors multiply minute OHLC. Volume and amount remain in their
raw units. Adjustment coverage is recorded in each materialization manifest.

## Configuration

Paths can be supplied explicitly or through environment variables:

```powershell
$env:QLIB_QUANTML_MINUTE_URI = "D:\data\minute"
$env:QLIB_QUANTML_CACHE_URI = "D:\cache\quantml"
$env:QLIB_QUANTML_ADJUSTMENT_URI = "D:\data\hfq"
```

The Qlib daily provider is separate and supplies market membership, labels,
benchmark data, and backtest prices.

## Direct API

```python
from qlib.contrib.data.quantml_factors import (
    QuantMLDataContext,
    compute_quantml_factors,
    list_quantml_factors,
)

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

Use `factors="all_supported"` only when the full 881-column result is intended.
The result is a sorted, unique `(datetime, instrument)` MultiIndex DataFrame.

## DataLoader and Handler

```python
from qlib.contrib.data.handler import QuantMLAdapted

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

When `instruments` is a Qlib market name, membership intervals are preserved.
Suspensions are reindexed as NaN so a historical CSI300 panel still contains
exactly 300 instruments per day.

## Standard Factor Evaluation

```python
from qlib.contrib.eva.factor import (
    FACTOR_METRIC_COLUMNS,
    evaluate_factor,
    evaluate_factors,
    render_factor_metrics_markdown,
)

single_metrics = evaluate_factor(score, label)
batch_metrics = evaluate_factors(features, label, batch_size=10)
```

Definitions:

- daily cross-sectional Pearson IC;
- `ic_mean = mean(daily_IC)`;
- `icir = abs(mean(daily_IC) / std(daily_IC))`;
- `signic0 = mean(sign(ic_mean) × daily_IC > 0)`;
- `signic004 = mean(sign(ic_mean) × daily_IC > 0.04)`;
- descending-score quintiles `Group1` through `Group5`;
- `long-short = (Group1 - Group5) / 2`;
- `long-average = Group1 - mean(valid labels)`;
- `t` is the one-sample t statistic of daily long-short returns.

Blocked factors remain in source-ordered reports with NaN metrics and an
explicit dependency status.

## Reproduction and Implementation Guide

For the full file inventory, operator semantics, catalog generator, cache
identity, large-data execution pattern, and acceptance tests, read:

[QuantML-Adapted Reproduction Guide](QUANTML_ADAPTED_REPRODUCTION_GUIDE.md)

Key implementation files:

```text
qlib/contrib/data/_quantml_catalog.py
qlib/contrib/data/quantml_factors.py
qlib/contrib/data/loader.py
qlib/contrib/data/handler.py
qlib/contrib/eva/factor.py
scripts/generate_quantml_catalog.py
```

## Tests

Run focused tests before any large materialization:

```powershell
python -m unittest discover -s tests -p "test_factor_evaluator.py" -v
python -m unittest discover -s tests\ops -p "test_quantml_adapted_ops.py" -v
python -m unittest discover -s tests\data_mid_layer_tests -p "test_quantml_catalog.py" -v
python -m unittest discover -s tests\data_mid_layer_tests -p "test_quantml_workflow.py" -v
python -m unittest discover -s tests\data_mid_layer_tests -p "test_quantml_real_data.py" -v
python -m compileall qlib\contrib\data qlib\contrib\eva
python -m pip check
```

Use deterministic synthetic tests first, then a bounded real-data sample. Do
not start development by computing the full market and full history.

## Cache and Memory Rules

- Keep minute caches, factor stores, labels, MLflow databases, and results
  outside the Git repository.
- Include source signatures, catalog SHA, operator version, Qlib commit,
  adjustment state, dates, factor selection, and market membership in cache
  identity.
- Reject partial or stale materialized stores.
- Write stock/year shards atomically and validate Parquet metadata before
  treating an existing file as a cache hit.
- Use bounded factor batches for metrics.
- On a 16-core/32-GB Windows host, use `kernels=16`, LightGBM
  `num_threads=16`, and no more than 16 factor workers.

## Scope and Provenance

Catalog source SHA256:

```text
b5e07f40ee97a9b7df14635226f05851611bfc1927b167d28a453cb249867b29
```

The catalog and this repository are authorized for private research sharing.
Do not make the formula catalog public or redistribute it further without a
separate provenance and licensing review.

This project does not include:

- raw minute or daily market data;
- adjustment files;
- CSMAR data;
- generated factor stores or research results;
- credentials, tokens, or account information.

Historical IC, group returns, or backtest performance do not imply formula
identity or future investment returns.

## Upstream Qlib

This fork preserves Microsoft Qlib history and the MIT license. Official
resources:

- [Microsoft Qlib](https://github.com/microsoft/qlib)
- [Qlib v0.9.7](https://github.com/microsoft/qlib/releases/tag/v0.9.7)
- [Official documentation](https://qlib.readthedocs.io/)

For a clone of this private repository, add Microsoft as an upstream remote:

```powershell
git remote add upstream https://github.com/microsoft/qlib.git
git fetch upstream --tags
```

Do not force-push over shared research commits. Keep upstream synchronization
separate from local factor-semantic changes.

## License

Microsoft Qlib is licensed under the [MIT License](LICENSE). Local adaptations
remain subject to the provenance and private-distribution constraints stated
above.
