import unittest

import numpy as np
import pandas as pd

from qlib.contrib.data.dolphindb_compat import SupportMode, get_operator_mapping
from qlib.data.base import Expression
from qlib.data.ops import Corr, Cov, IdxMax, IdxMin, Mean, Quantile, Rank, Std


class SeriesExpression(Expression):
    """Minimal expression leaf for deterministic operator semantics tests."""

    def __init__(self, values, name="fixture"):
        self.series = pd.Series(values, dtype=float)
        self.name = name

    def __str__(self):
        return self.name

    def _load_internal(self, instrument, start_index, end_index, *args):
        return self.series.copy()

    def get_longest_back_rolling(self):
        return 0

    def get_extended_window_size(self):
        return 0, 0


def load(operator):
    return operator._load_internal("fixture", 0, len(operator.feature.series) - 1)


class TestNativeOperatorSemantics(unittest.TestCase):
    """Compare Qlib v0.9.7 behavior with fixed local DolphinDB doc examples."""

    def test_not_equivalent_row_rank_and_qlib_rolling_rank(self):
        # Source: documentation.latest.zh/funcs/r/rowRank.html
        panel = pd.DataFrame(
            [
                [3, 1, 2, 4, 7, 6, 9, 8, 5],
                [9, np.nan, 2, 3, 5, 6, 3, 2, 8],
            ]
        )
        row_rank = panel.rank(axis=1, method="min", na_option="keep") - 1
        np.testing.assert_allclose(
            row_rank.iloc[0].to_numpy(),
            np.array([2, 0, 1, 3, 6, 5, 8, 7, 4], dtype=float),
        )
        np.testing.assert_allclose(
            row_rank.iloc[1].to_numpy(),
            np.array([7, np.nan, 0, 2, 4, 5, 2, 0, 6], dtype=float),
            equal_nan=True,
        )

        rolling_rank = pd.DataFrame(
            {
                column: Rank(SeriesExpression(panel[column], f"column_{column}"), 2)._load_internal(
                    "fixture", 0, 1
                )
                for column in panel.columns
            }
        )
        self.assertFalse(np.allclose(row_rank, rolling_rank, equal_nan=True))
        self.assertEqual(get_operator_mapping("rowRank").support_mode, SupportMode.PANEL_REQUIRED)

    def test_not_equivalent_mrank_and_qlib_rank(self):
        # Source: documentation.latest.zh/funcs/m/mrank.html
        values = [3, 2, 4, 4, 4, np.nan, 1]
        dolphindb = np.array([np.nan, np.nan, 0, 0, 0, np.nan, 1], dtype=float)
        qlib_rank = load(Rank(SeriesExpression(values), 3)).to_numpy()

        self.assertFalse(np.allclose(dolphindb, qlib_rank, equal_nan=True))
        self.assertTrue(np.isfinite(qlib_rank[0]))
        self.assertGreaterEqual(np.nanmin(qlib_rank), 0.0)
        self.assertLessEqual(np.nanmax(qlib_rank), 1.0)
        self.assertEqual(get_operator_mapping("mrank").support_mode, SupportMode.CUSTOM_REQUIRED)

    def test_composed_mimax_mimin_only_match_complete_non_null_windows(self):
        # Sources: documentation.latest.zh/funcs/m/mimax.html and mimin.html
        max_values = [1.2, 2, 0, 6, -1, 6]
        min_values = [1.2, 2, 0, 6, -1, -1]
        qlib_max = load(IdxMax(SeriesExpression(max_values), 3)).to_numpy() - 1
        qlib_min = load(IdxMin(SeriesExpression(min_values), 3)).to_numpy() - 1
        np.testing.assert_allclose(qlib_max[2:], np.array([1, 2, 1, 0], dtype=float))
        np.testing.assert_allclose(qlib_min[2:], np.array([2, 1, 2, 1], dtype=float))

        documented_max = np.array([np.nan, np.nan, 1, 2, 1, 0], dtype=float)
        documented_min = np.array([np.nan, np.nan, 0, 0, 2, 1], dtype=float)
        qlib_max_with_nan = (
            load(IdxMax(SeriesExpression([1.2, 2, np.nan, 6, -1, 6]), 3)).to_numpy() - 1
        )
        qlib_min_with_nan = (
            load(IdxMin(SeriesExpression([1.2, 2, np.nan, 6, -1, -1]), 3)).to_numpy() - 1
        )
        self.assertFalse(np.allclose(documented_max, qlib_max_with_nan, equal_nan=True))
        self.assertFalse(np.allclose(documented_min, qlib_min_with_nan, equal_nan=True))

    def test_equivalent_mavg_tail_but_not_default_warmup(self):
        # Source: documentation.latest.zh/funcs/m/mavg.html
        values = [7, 4, 6, 0, -5, 32, 9, 8]
        qlib_mean = load(Mean(SeriesExpression(values), 4)).to_numpy()
        documented = np.array([np.nan, np.nan, np.nan, 4.25, 1.25, 8.25, 9, 11])
        np.testing.assert_allclose(qlib_mean[3:], documented[3:])
        self.assertTrue(np.isfinite(qlib_mean[:3]).all())

    def test_equivalent_mstd_tail_but_not_default_warmup(self):
        # Source: documentation.latest.zh/funcs/m/mstd.html
        values = [1, 2, 5, 4, 3]
        qlib_std = load(Std(SeriesExpression(values), 3)).to_numpy()
        documented = np.array([np.nan, np.nan, 2.081666, 1.527525, 1.0])
        np.testing.assert_allclose(qlib_std[2:], documented[2:], rtol=1e-6, atol=1e-6)
        self.assertTrue(np.isfinite(qlib_std[1]))

    def test_equivalent_mcorr_and_mcovar_complete_windows(self):
        # Sources: documentation.latest.zh/funcs/m/mcorr.html and mcovar.html
        x_corr = SeriesExpression(np.arange(1, 9), "x_corr")
        y_corr = SeriesExpression([9, 5, 3, 4, 5, 4, 7, 1], "y_corr")
        qlib_corr = Corr(x_corr, y_corr, 5)._load_internal("fixture", 0, 7).to_numpy()
        documented_corr = np.array([-0.624038, 0, 0.834058, -0.29173])
        np.testing.assert_allclose(qlib_corr[4:], documented_corr, rtol=1e-5, atol=1e-6)

        x_cov = SeriesExpression(np.arange(1, 11), "x_cov")
        y_cov = SeriesExpression([9, 5, 3, 4, 5, 4, 7, 1, 3, 4], "y_cov")
        qlib_cov = Cov(x_cov, y_cov, 5)._load_internal("fixture", 0, 9).to_numpy()
        documented_cov = np.array([-2.25, 0, 2, -1, -1.75, -1], dtype=float)
        np.testing.assert_allclose(qlib_cov[4:], documented_cov)

    def test_equivalent_mpercentile_tail_after_parameter_conversion(self):
        # Source: documentation.latest.zh/funcs/m/mpercentile.html
        values = [2, 1, 3, 7, 6, 5, 4]
        qlib_quantile = load(Quantile(SeriesExpression(values), 3, 0.5)).to_numpy()
        documented = np.array([np.nan, np.nan, 2, 3, 6, 6, 5], dtype=float)
        np.testing.assert_allclose(qlib_quantile[2:], documented[2:])
        np.testing.assert_allclose(qlib_quantile[:2], np.array([2, 1.5]))


if __name__ == "__main__":
    unittest.main()
