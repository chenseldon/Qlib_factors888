import unittest

import numpy as np
import pandas as pd

from qlib.contrib.data.quantml_factors import (
    DownResample,
    DownStd,
    Partial,
    PartialRatio,
    Peak,
    QuantMLExpressionError,
    UpStd,
    _ExpressionCompiler,
)
from qlib.data.base import Expression


def session_index(date):
    morning = pd.date_range(f"{date} 09:31", f"{date} 11:30", freq="min")
    afternoon = pd.date_range(f"{date} 13:01", f"{date} 15:00", freq="min")
    return morning.append(afternoon)


class SeriesLeaf(Expression):
    def __init__(self, values, index, name):
        self.series = pd.Series(values, index=index, dtype=float)
        self.name = name

    def __str__(self):
        return self.name

    def _load_internal(self, instrument, start_index, end_index, *args):
        return self.series.iloc[start_index : end_index + 1].copy()

    def get_longest_back_rolling(self):
        return 0

    def get_extended_window_size(self):
        return 0, 0


class TestQuantMLAdaptedOperators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = session_index("2024-01-02").append(session_index("2024-01-03"))

    def test_down_resample_last_integer_mean_and_nested(self):
        leaf = SeriesLeaf(np.arange(480), self.index, "resample_values")
        last = DownResample(leaf, 240, "last")._load_internal("x", 0, 479)
        self.assertEqual(last.dropna().tolist(), [239.0, 479.0])

        integer = DownResample(leaf, 240, 9)._load_internal("x", 0, 479)
        self.assertEqual(integer.dropna().tolist(), [9.0, 249.0])

        means = DownResample(leaf, 60, "mean")._load_internal("x", 0, 479)
        self.assertEqual(len(means.dropna()), 8)
        self.assertAlmostEqual(means.dropna().iloc[0], 29.5)

        inner = DownResample(leaf, 60, "last")
        outer = DownResample(inner, 4, "last")._load_internal("x", 0, 479)
        self.assertEqual(outer.dropna().tolist(), [239.0, 479.0])

    def test_partial_and_partial_ratio_use_end_exclusive_intervals(self):
        leaf = SeriesLeaf(np.arange(1, 481), self.index, "partial_values")
        partial = Partial(leaf, 240, "mean", 0, 30)._load_internal("x", 0, 479)
        self.assertEqual(partial.dropna().tolist(), [15.5, 255.5])

        ratio = PartialRatio(leaf, 240, "sum", 0, 30)._load_internal("x", 0, 479)
        expected = np.arange(1, 31).sum() / np.arange(1, 241).sum()
        self.assertAlmostEqual(ratio.dropna().iloc[0], expected)

        zero = SeriesLeaf(np.zeros(480), self.index, "zero_values")
        zero_ratio = PartialRatio(zero, 240, "sum", 0, 30)._load_internal("x", 0, 479)
        self.assertTrue(zero_ratio.dropna().empty)

    def test_upside_downside_rms_and_peak(self):
        values = np.tile([-2.0, -1.0, 0.0, 1.0, 2.0], 96)
        leaf = SeriesLeaf(values, self.index, "rms_values")
        down = DownStd(leaf, 5)._load_internal("x", 0, 479)
        up = UpStd(leaf, 5)._load_internal("x", 0, 479)
        expected = np.sqrt(2.5)
        self.assertAlmostEqual(down.iloc[4], expected)
        self.assertAlmostEqual(up.iloc[4], expected)

        peak = Peak(leaf, 5)._load_internal("x", 0, 479)
        self.assertEqual(peak.iloc[4], 0.0)
        self.assertEqual(peak.iloc[5], 4.0)

    def test_safe_compiler_converts_index_positions_to_zero_based(self):
        values = pd.Series(np.arange(480, dtype=float), index=self.index)
        compiler = _ExpressionCompiler({"close": values}, "compiler")
        expression = compiler.compile("IdxMax($close,240)")
        result = expression._load_internal("x", 0, 479)
        self.assertEqual(result.iloc[239], 239.0)
        self.assertEqual(result.iloc[479], 239.0)

    def test_safe_compiler_rejects_arbitrary_python(self):
        values = pd.Series(np.arange(480, dtype=float), index=self.index)
        compiler = _ExpressionCompiler({"close": values}, "unsafe")
        with self.assertRaises(QuantMLExpressionError):
            compiler.compile("__import__('os').system('whoami')")
        with self.assertRaises(QuantMLExpressionError):
            compiler.compile("$close[0]")


if __name__ == "__main__":
    unittest.main()
