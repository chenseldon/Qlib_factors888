import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from qlib.contrib.data.dolphindb_compat import FactorDependencyError
from qlib.contrib.data.handler import QuantMLAdapted
from qlib.contrib.data.loader import QuantMLAdaptedDL
from qlib.contrib.data.quantml_factors import (
    EXPECTED_SESSION_TIMES,
    QuantMLDataContext,
    QuantMLDataQualityError,
    _cache_identity,
    _normalize_session,
    _read_partition,
    _select_dates,
    compute_quantml_factors,
)
from qlib.data.dataset.handler import DataHandlerLP


MINUTE_ROOT = Path(r"D:\Financial_Data\ALL_Data\A股分钟数据\A股_分时数据_沪深\data\minute")


class TestQuantMLRealData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not MINUTE_ROOT.is_dir():
            raise unittest.SkipTest(f"minute root is unavailable: {MINUTE_ROOT}")
        cls.partition = MINUTE_ROOT / "date=2023-12-29"
        if not cls.partition.is_dir():
            raise unittest.SkipTest(f"minute partition is unavailable: {cls.partition}")

    def test_real_session_field_mapping(self):
        raw = _read_partition(self.partition, ["sh600000"])
        self.assertEqual(len(raw), 241)
        normalized = _normalize_session(raw, "sh600000", pd.Timestamp("2023-12-29"))
        self.assertEqual(len(normalized), 240)
        self.assertEqual(tuple(normalized.index.strftime("%H:%M")), EXPECTED_SESSION_TIMES)
        source = raw[raw.datetime.dt.strftime("%H:%M") == "09:31"].iloc[0]
        target = normalized.loc[pd.Timestamp("2023-12-29 09:31")]
        self.assertEqual(target["volume"], source.volume * 100.0)
        self.assertEqual(target["total_turnover"], source.amount)

    def test_partial_non_empty_session_fails_closed(self):
        raw = _read_partition(self.partition, ["sh600000"]).iloc[:-1]
        with self.assertRaises(QuantMLDataQualityError):
            _normalize_session(raw, "sh600000", pd.Timestamp("2023-12-29"))

    def test_insufficient_warmup_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            Path(root, "date=2024-01-02").mkdir()
            Path(root, "date=2024-01-03").mkdir()
            with self.assertRaisesRegex(QuantMLDataQualityError, "insufficient minute warm-up"):
                _select_dates(Path(root), "2024-01-02", "2024-01-03", warmup_sessions=1)

    def test_direct_loader_handler_and_cache_are_consistent(self):
        factors = ["QMLA0001", "QMLA0002"]
        with tempfile.TemporaryDirectory() as cache_dir:
            context = QuantMLDataContext(MINUTE_ROOT, cache_dir)
            direct = compute_quantml_factors(
                factors,
                ["SH600000"],
                "2023-12-29",
                "2023-12-29",
                context,
                use_cache=True,
            )
            cached = compute_quantml_factors(
                factors,
                ["SH600000"],
                "2023-12-29",
                "2023-12-29",
                context,
                use_cache=True,
            )
            assert_frame_equal(direct, cached, check_exact=True)
            self.assertEqual(len(list(Path(cache_dir).glob("*.parquet"))), 1)
            self.assertEqual(len(list(Path(cache_dir).glob("*.json"))), 1)

            loader = QuantMLAdaptedDL(factors, context=context, use_cache=True)
            loaded = loader.load(["SH600000"], "2023-12-29", "2023-12-29")
            assert_frame_equal(loaded["feature"], direct)

            handler = QuantMLAdapted(
                ["SH600000"],
                factors,
                start_time="2023-12-29",
                end_time="2023-12-29",
                context=context,
                use_cache=True,
            )
            handled = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_I)
            assert_frame_equal(handled, direct)

    def test_blocked_features_raise_dependency_errors(self):
        context = QuantMLDataContext(MINUTE_ROOT)
        with self.assertRaises(FactorDependencyError):
            compute_quantml_factors(
                ["QMLA0048"],
                ["SH600000"],
                "2023-12-29",
                "2023-12-29",
                context,
                use_cache=False,
            )

        num_trades_factor = next(
            factor.name
            for factor in __import__(
                "qlib.contrib.data.quantml_factors", fromlist=["list_quantml_factors"]
            ).list_quantml_factors()
            if factor.required_capabilities == ("num_trades",)
        )
        with self.assertRaises(FactorDependencyError):
            compute_quantml_factors(
                [num_trades_factor],
                ["SH600000"],
                "2023-12-29",
                "2023-12-29",
                context,
                use_cache=False,
            )

    def test_all_supported_factors_run_on_a_bounded_real_sample(self):
        context = QuantMLDataContext(MINUTE_ROOT)
        result = compute_quantml_factors(
            "all_supported",
            ["SH600000"],
            "2023-12-29",
            "2023-12-29",
            context,
            use_cache=False,
        )
        self.assertEqual(result.shape, (1, 881))
        self.assertFalse(np.isinf(result.to_numpy()).any())
        self.assertGreaterEqual(int(np.isfinite(result.to_numpy()).sum()), 850)

    def test_two_instruments_over_five_trading_days_are_sorted_and_deterministic(self):
        context = QuantMLDataContext(MINUTE_ROOT)
        arguments = {
            "factors": ["QMLA0001", "QMLA0002"],
            "instruments": ["SH600000", "SH600519"],
            "start_time": "2023-12-25",
            "end_time": "2023-12-29",
            "context": context,
            "use_cache": False,
        }
        first = compute_quantml_factors(**arguments)
        second = compute_quantml_factors(**arguments)
        self.assertEqual(first.shape, (10, 2))
        self.assertTrue(first.index.is_unique)
        self.assertTrue(first.index.is_monotonic_increasing)
        assert_frame_equal(first, second, check_exact=True)

    def test_cache_identity_changes_with_source_signature(self):
        from qlib.contrib.data.quantml_factors import list_quantml_factors

        factor = list_quantml_factors(available=True)[0]
        args = (MINUTE_ROOT, [factor], ["sh600000"], "2023-12-29", "2023-12-29")
        first, _ = _cache_identity(*args, {"file_count": 1, "total_size": 10, "latest_mtime_ns": 1})
        second, _ = _cache_identity(*args, {"file_count": 1, "total_size": 11, "latest_mtime_ns": 1})
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
