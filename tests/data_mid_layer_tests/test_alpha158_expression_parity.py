import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

import qlib
from qlib.constant import REG_CN
from qlib.contrib.data.loader import Alpha158DL
from qlib.data import D


PROVIDER_URI = Path(r"C:\Users\33080\.qlib\qlib_data\cn_data")
INSTRUMENTS = ["SH600000", "SH600519", "SZ000001", "SZ000002", "SZ000063"]
WARMUP_START = "2022-10-10"
EVALUATION_START = "2024-01-02"
EVALUATION_END = "2024-06-28"


class TestAlpha158ExpressionParity(unittest.TestCase):
    """Bounded real-data baseline for native Alpha158 expressions.

    The local DolphinDB alpha158.dos is a comparison artifact. Native Qlib
    Alpha158 remains the implementation source of truth.
    """

    @classmethod
    def setUpClass(cls):
        if not PROVIDER_URI.exists():
            raise unittest.SkipTest(f"Qlib provider is unavailable: {PROVIDER_URI}")

        qlib.init(provider_uri=str(PROVIDER_URI), region=REG_CN, kernels=1)
        cls.config = {
            "kbar": {},
            "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"]},
            "volume": {"windows": [0]},
            "rolling": {"windows": [5]},
        }
        cls.fields, cls.names = Alpha158DL.get_feature_config(cls.config)
        cls.data = cls._load_named_features(cls.fields, cls.names)

    @classmethod
    def _load_named_features(cls, fields, names):
        data = D.features(
            INSTRUMENTS,
            fields,
            start_time=WARMUP_START,
            end_time=EVALUATION_END,
            freq="day",
        )
        data.columns = names
        return data

    @classmethod
    def evaluation_data(cls):
        datetimes = cls.data.index.get_level_values("datetime")
        mask = (datetimes >= pd.Timestamp(EVALUATION_START)) & (
            datetimes <= pd.Timestamp(EVALUATION_END)
        )
        return cls.data.loc[mask]

    def test_native_alpha158_still_has_158_unique_fields(self):
        fields, names = Alpha158DL.get_feature_config()
        self.assertEqual(len(fields), 158)
        self.assertEqual(len(names), 158)
        self.assertEqual(len(set(fields)), 158)
        self.assertEqual(len(set(names)), 158)

    def test_representative_config_covers_all_primitive_families(self):
        expected_names = {
            "KMID",
            "ROC5",
            "MA5",
            "STD5",
            "MAX5",
            "MIN5",
            "QTLU5",
            "QTLD5",
            "RSV5",
            "IMAX5",
            "IMIN5",
            "IMXD5",
            "CORR5",
            "CORD5",
            "CNTP5",
            "CNTN5",
            "CNTD5",
            "SUMP5",
            "SUMN5",
            "SUMD5",
            "VOLUME0",
            "VMA5",
            "VSTD5",
            "WVMA5",
            "VSUMP5",
            "VSUMN5",
            "VSUMD5",
            "BETA5",
            "RSQR5",
            "RESI5",
            "RANK5",
        }
        self.assertLessEqual(expected_names, set(self.names))
        self.assertEqual(len(self.names), len(set(self.names)))

    def test_real_data_index_quality_and_determinism(self):
        evaluation = self.evaluation_data()
        self.assertEqual(evaluation.index.names, ["instrument", "datetime"])
        self.assertTrue(evaluation.index.is_unique)
        self.assertTrue(evaluation.index.is_monotonic_increasing)
        self.assertEqual(set(evaluation.index.get_level_values("instrument")), set(INSTRUMENTS))
        self.assertGreater(len(evaluation), 500)

        repeated = self._load_named_features(self.fields, self.names)
        assert_frame_equal(self.data, repeated, check_exact=True)

        planned_index = evaluation.reorder_levels(["datetime", "instrument"]).sort_index()
        self.assertEqual(planned_index.index.names, ["datetime", "instrument"])
        self.assertTrue(planned_index.index.is_monotonic_increasing)

    def test_real_data_missing_and_finite_rates_are_bounded(self):
        evaluation = self.evaluation_data()
        missing_rate = float(evaluation.isna().to_numpy().mean())
        finite_or_missing = np.isfinite(evaluation.to_numpy()) | evaluation.isna().to_numpy()
        finite_values = np.isfinite(evaluation.to_numpy())

        self.assertLess(missing_rate, 0.10)
        self.assertTrue(finite_or_missing.all())
        self.assertGreater(int(finite_values.sum()), int(evaluation.size * 0.90))
        self.assertGreater(len(evaluation.dropna()), 100)

    def test_complete_window_native_expressions_match_independent_pandas(self):
        raw_fields = ["$close", "$high", "$low", "$volume"]
        raw_names = ["close", "high", "low", "volume"]
        raw = self._load_named_features(raw_fields, raw_names)

        reference_parts = []
        for instrument, frame in raw.groupby(level="instrument", sort=True):
            values = frame.droplevel("instrument")
            close = values["close"]
            reference = pd.DataFrame(index=values.index)
            reference["MA5"] = close.rolling(5, min_periods=5).mean() / close
            reference["STD5"] = close.rolling(5, min_periods=5).std() / close
            reference["MAX5"] = values["high"].rolling(5, min_periods=5).max() / close
            reference["MIN5"] = values["low"].rolling(5, min_periods=5).min() / close
            reference["QTLU5"] = close.rolling(5, min_periods=5).quantile(0.8) / close
            reference["QTLD5"] = close.rolling(5, min_periods=5).quantile(0.2) / close
            reference["VMA5"] = (
                values["volume"].rolling(5, min_periods=5).mean() / (values["volume"] + 1e-12)
            )
            reference["VSTD5"] = (
                values["volume"].rolling(5, min_periods=5).std() / (values["volume"] + 1e-12)
            )
            reference["instrument"] = instrument
            reference = reference.set_index("instrument", append=True).reorder_levels(
                ["instrument", "datetime"]
            )
            reference_parts.append(reference)

        reference = pd.concat(reference_parts).sort_index()
        evaluation = self.evaluation_data()[reference.columns]
        reference = reference.loc[evaluation.index]

        for column in reference.columns:
            with self.subTest(column=column):
                valid = evaluation[column].notna() & reference[column].notna()
                self.assertGreater(int(valid.sum()), 500)
                assert_series_equal(
                    evaluation.loc[valid, column],
                    reference.loc[valid, column],
                    check_names=False,
                    check_dtype=False,
                    check_exact=False,
                    rtol=1e-6,
                    atol=1e-7,
                )


if __name__ == "__main__":
    unittest.main()
