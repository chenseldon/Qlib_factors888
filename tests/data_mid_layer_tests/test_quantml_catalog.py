import hashlib
import unittest
from collections import Counter
from pathlib import Path

import pandas as pd

from qlib.contrib.data._quantml_catalog import SOURCE_SHA256
from qlib.contrib.data.quantml_factors import (
    _ExpressionCompiler,
    list_quantml_factors,
)


class TestQuantMLCatalog(unittest.TestCase):
    def test_catalog_counts_aliases_and_dependencies(self):
        factors = list_quantml_factors()
        self.assertEqual(len(factors), 994)
        self.assertEqual(factors[0].name, "QMLA0001")
        self.assertEqual(factors[-1].name, "QMLA0994")
        self.assertEqual(sum(len(factor.source_rows) for factor in factors), 1049)
        self.assertEqual(sum(len(factor.source_rows) - 1 for factor in factors), 55)
        self.assertEqual(sum(factor.available for factor in factors), 881)
        blocked = Counter(factor.required_capabilities for factor in factors if not factor.available)
        self.assertEqual(blocked[("num_trades",)], 56)
        self.assertEqual(blocked[("s_dq_freeturnover",)], 57)

    def test_source_hash_when_local_reference_is_available(self):
        source = Path(__file__).resolve().parents[3] / "DolphinDB_factors" / "factors.md"
        if not source.is_file():
            self.skipTest("local-only QuantML source table is unavailable")
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), SOURCE_SHA256)

    def test_all_expressions_compile_with_declared_features(self):
        index = pd.date_range("2024-01-02 09:31", periods=480, freq="min")
        fields = {
            name: pd.Series(1.0, index=index)
            for name in (
                "open",
                "high",
                "low",
                "close",
                "volume",
                "total_turnover",
                "num_trades",
                "s_dq_freeturnover",
            )
        }
        compiler = _ExpressionCompiler(fields, "catalog_compile")
        for factor in list_quantml_factors():
            with self.subTest(factor=factor.name):
                self.assertIsNotNone(compiler.compile(factor.expression))


if __name__ == "__main__":
    unittest.main()
