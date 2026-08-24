import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from qlib.contrib.data.loader import QuantMLAdaptedDL
from qlib.contrib.data.quantml_factors import (
    OPERATOR_VERSION,
    SOURCE_SHA256,
    QuantMLDataContext,
    _adjustment_signature,
    _apply_hfq,
    _cache_identity,
    _evaluate_instrument,
    _resolve_factors,
)


class _FakeDataProvider:
    @staticmethod
    def instruments(market):
        return {"market": market}

    @staticmethod
    def list_instruments(config, start_time=None, end_time=None, as_list=False):
        assert not as_list
        return {
            "SH600000": [(pd.Timestamp("2023-01-03"), pd.Timestamp("2023-01-04"))],
            "SZ000001": [(pd.Timestamp("2023-01-04"), pd.Timestamp("2023-01-05"))],
        }

    @staticmethod
    def calendar(start_time=None, end_time=None, freq="day"):
        return pd.date_range(start_time, end_time, freq="B")

    @staticmethod
    def features(instruments, fields, start_time=None, end_time=None, freq="day"):
        membership = _FakeDataProvider.list_instruments(instruments, start_time, end_time)
        rows = []
        for instrument, spans in membership.items():
            for start, end in spans:
                for date in pd.date_range(start, end, freq="B"):
                    rows.append((instrument, date, float(date.day)))
        frame = pd.DataFrame(rows, columns=["instrument", "datetime", fields[0]])
        return frame.set_index(["instrument", "datetime"])


class TestQuantMLWorkflowLoader(unittest.TestCase):
    def test_materialized_loader_preserves_dynamic_membership_and_adds_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "year=2023" / "code=sample" / "factors.parquet"
            output.parent.mkdir(parents=True)
            (root / "_MANIFEST.json").write_text(
                json.dumps({"catalog_sha256": SOURCE_SHA256, "operator_version": OPERATOR_VERSION}),
                encoding="utf-8",
            )
            pd.DataFrame(
                {
                    "datetime": pd.to_datetime(
                        ["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-03", "2023-01-04", "2023-01-05"]
                    ),
                    "instrument": ["SH600000"] * 3 + ["SZ000001"] * 3,
                    "QMLA0001": np.arange(6, dtype=np.float32),
                    "QMLA0002": np.arange(10, 16, dtype=np.float32),
                }
            ).to_parquet(output, index=False)
            loader = QuantMLAdaptedDL(
                factors=["QMLA0001", "QMLA0002"],
                materialized_uri=root,
                label=["Ref($close,-6)/Ref($close,-1)-1"],
                label_names=["LABEL0"],
            )
            with patch("qlib.contrib.data.loader.D", _FakeDataProvider):
                result = loader.load("csi300", "2023-01-03", "2023-01-05")

        expected = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2023-01-03"), "SH600000"),
                (pd.Timestamp("2023-01-04"), "SH600000"),
                (pd.Timestamp("2023-01-04"), "SZ000001"),
                (pd.Timestamp("2023-01-05"), "SZ000001"),
            ],
            names=["datetime", "instrument"],
        )
        self.assertTrue(result.index.equals(expected))
        self.assertEqual(list(result["feature"].columns), ["QMLA0001", "QMLA0002"])
        self.assertEqual(list(result["label"].columns), ["LABEL0"])

    def test_hfq_adjustment_changes_prices_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pd.DataFrame(
                {
                    "股票代码": ["600000.SH", "600000.SH"],
                    "交易日期": ["2023-01-03", "2023-01-04"],
                    "复权因子": [2.0, 3.0],
                }
            ).to_csv(root / "600000.SH.csv", index=False)
            index = pd.to_datetime(["2023-01-03 09:31", "2023-01-04 09:31"])
            frame = pd.DataFrame(
                {
                    "open": [1.0, 2.0],
                    "high": [1.0, 2.0],
                    "low": [1.0, 2.0],
                    "close": [1.0, 2.0],
                    "volume": [100.0, 200.0],
                    "total_turnover": [1000.0, 2000.0],
                },
                index=index,
            )
            adjusted = _apply_hfq(frame, "sh600000", root)
            self.assertEqual(adjusted["close"].tolist(), [2.0, 6.0])
            self.assertEqual(adjusted["volume"].tolist(), frame["volume"].tolist())
            self.assertEqual(adjusted["total_turnover"].tolist(), frame["total_turnover"].tolist())

    def test_materialized_loader_rejects_stale_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "_MANIFEST.json").write_text(
                json.dumps({"catalog_sha256": "stale", "operator_version": "stale"}),
                encoding="utf-8",
            )
            loader = QuantMLAdaptedDL(factors=["QMLA0001"], materialized_uri=root)
            with self.assertRaisesRegex(ValueError, "catalog SHA"):
                loader._load_materialized(["SH600000"], "2023-01-03", "2023-01-04")

    def test_adjustment_state_changes_cache_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            minute = root / "minute"
            minute.mkdir()
            adjustment = root / "adjustment"
            adjustment.mkdir()
            (adjustment / "600000.SH.csv").write_text("x", encoding="utf-8")
            signature = _adjustment_signature(adjustment, ["sh600000"])
            common = (minute, [], ["sh600000"], "2023-01-03", "2023-01-04", {})
            raw_key, _ = _cache_identity(*common, "none", {})
            hfq_key, payload = _cache_identity(*common, "hfq", signature)
            self.assertNotEqual(raw_key, hfq_key)
            self.assertEqual(payload["adjustment_mode"], "hfq")
            self.assertEqual(payload["adjustment_signature"]["file_count"], 1)
            self.assertEqual(QuantMLDataContext(adjustment_mode="none").adjustment_path(), None)

    def test_expression_infinity_is_normalized_to_nan(self):
        index = pd.date_range("2023-01-03 09:31", periods=240, freq="min")
        frame = pd.DataFrame(
            {
                "open": np.ones(240),
                "high": np.ones(240),
                "low": np.ones(240),
                "close": np.ones(240),
                "volume": np.ones(240),
                "total_turnover": np.arange(1, 241, dtype=float),
            },
            index=index,
        )
        result = _evaluate_instrument(
            "sh600000",
            frame,
            _resolve_factors(["QMLA0008"]),
            pd.DatetimeIndex([pd.Timestamp("2023-01-03")]),
            "infinity-test",
        )
        self.assertTrue(np.isnan(result.iloc[0, 0]))


if __name__ == "__main__":
    unittest.main()
