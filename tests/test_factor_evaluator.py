import unittest

import numpy as np
import pandas as pd
from scipy import stats

from qlib.contrib.eva.factor import (
    FACTOR_METRIC_COLUMNS,
    evaluate_factor,
    evaluate_factors,
    render_factor_metrics_markdown,
)


class TestFactorEvaluator(unittest.TestCase):
    @staticmethod
    def _panel():
        dates = pd.date_range("2023-01-03", periods=4, freq="B")
        instruments = [f"SH6000{index:02d}" for index in range(10)]
        index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
        score = np.tile(np.arange(10, 0, -1, dtype=float), len(dates))
        label_base = np.arange(1, 11, dtype=float) / 100
        label = np.concatenate([label_base * scale for scale in (1.0, 1.5, 0.5, 2.0)])
        features = pd.DataFrame({"factor_a": score, "factor_b": -score}, index=index)
        return features, pd.Series(label, index=index, name="LABEL0")

    def test_standard_metrics_use_qlib_half_long_short(self):
        features, label = self._panel()
        metrics = evaluate_factors(features, label, batch_size=1)

        scales = np.array([1.0, 1.5, 0.5, 2.0])
        daily_long_short = -0.04 * scales
        self.assertEqual(list(metrics.columns), list(FACTOR_METRIC_COLUMNS))
        self.assertAlmostEqual(metrics.loc["factor_a", "Group1"], 0.015 * scales.mean())
        self.assertAlmostEqual(metrics.loc["factor_a", "Group5"], 0.095 * scales.mean())
        self.assertAlmostEqual(metrics.loc["factor_a", "long-short"], daily_long_short.mean())
        self.assertAlmostEqual(metrics.loc["factor_a", "long-average"], -0.04 * scales.mean())
        self.assertAlmostEqual(
            metrics.loc["factor_a", "t"],
            stats.ttest_1samp(daily_long_short, 0).statistic,
        )
        self.assertEqual(metrics.loc["factor_a", "signic0"], 1.0)
        self.assertEqual(metrics.loc["factor_a", "signic004"], 1.0)

    def test_single_factor_and_batch_results_match(self):
        features, label = self._panel()
        single = evaluate_factor(features["factor_a"], label)
        batch = evaluate_factors(features, label, batch_size=2).loc["factor_a"]
        pd.testing.assert_series_equal(single, batch, check_names=False)

    def test_nan_ties_and_zero_variance_are_deterministic(self):
        features, label = self._panel()
        features.loc[(slice(None), "SH600000"), "factor_a"] = np.nan
        features["constant"] = 1.0
        first = evaluate_factors(features, label, batch_size=2, min_obs=5)
        second = evaluate_factors(features, label, batch_size=2, min_obs=5)
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(np.isnan(first.loc["constant", "ic_mean"]))
        self.assertTrue(np.isnan(first.loc["constant", "icir"]))

    def test_markdown_preserves_source_order_and_blocked_rows(self):
        rows = pd.DataFrame(
            {
                "factor_id": [f"QMLA{index:04d}" for index in range(1, 51)],
                "factor_name": [f"expression_{index}" for index in range(1, 51)],
            }
        )
        metrics = pd.DataFrame(
            1.0,
            index=rows.factor_id.iloc[:47],
            columns=FACTOR_METRIC_COLUMNS,
        )
        markdown = render_factor_metrics_markdown(rows, metrics)
        table_lines = [line for line in markdown.splitlines() if line.startswith("|")]
        self.assertEqual(len(table_lines), 52)
        self.assertIn("expression_1", table_lines[2])
        self.assertIn("expression_50", table_lines[-1])
        self.assertIn("NaN", table_lines[-1])


if __name__ == "__main__":
    unittest.main()
