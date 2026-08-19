import unittest

from qlib.contrib.data.dolphindb_compat import (
    Axis,
    FactorDependencyError,
    FactorRegistry,
    FactorSemanticError,
    FactorSpec,
    FactorStatus,
    SupportMode,
    get_operator_mapping,
    list_operator_mappings,
)
from qlib.data.ops import OpsList


class TestOperatorCompatibilityRegistry(unittest.TestCase):
    def test_mapping_names_are_unique(self):
        names = [mapping.dolphindb_name.casefold() for mapping in list_operator_mappings()]
        self.assertEqual(len(names), len(set(names)))

    def test_native_mappings_only_reference_registered_operators(self):
        registered = {operator.__name__ for operator in OpsList}
        for mapping in list_operator_mappings():
            with self.subTest(mapping=mapping.dolphindb_name):
                if mapping.support_mode in {
                    SupportMode.DIRECT_NATIVE,
                    SupportMode.COMPOSED_NATIVE,
                }:
                    self.assertIsNotNone(mapping.expression_template)
                    self.assertTrue(mapping.qlib_operators)
                    self.assertLessEqual(set(mapping.qlib_operators), registered)

    def test_panel_and_custom_mappings_fail_closed(self):
        for mapping in list_operator_mappings():
            if mapping.support_mode in {
                SupportMode.PANEL_REQUIRED,
                SupportMode.CUSTOM_REQUIRED,
            }:
                with self.subTest(mapping=mapping.dolphindb_name):
                    self.assertIsNone(mapping.expression_template)
                    self.assertFalse(mapping.qlib_operators)
                    with self.assertRaisesRegex(ValueError, mapping.support_mode.value):
                        mapping.render_expression(x="$close", n=5)

    def test_composed_expression_templates(self):
        self.assertEqual(
            get_operator_mapping("SignedPower").render_expression(x="$close", a=2),
            "Mul(Sign($close), Power(Abs($close), 2))",
        )
        self.assertEqual(
            get_operator_mapping("mimax").render_expression(x="$high", n=5),
            "Sub(IdxMax($high, 5), 1)",
        )
        self.assertEqual(
            get_operator_mapping("mimin").render_expression(x="$low", n=5),
            "Sub(IdxMin($low, 5), 1)",
        )
        self.assertEqual(get_operator_mapping("advn").render_expression(n=20), "Mean($amount, 20)")
        self.assertEqual(
            get_operator_mapping("safe_divide").render_expression(
                numerator="$close-$open", denominator="$high-$low"
            ),
            "If(Eq($high-$low, 0), Div($high-$low, 0), Div($close-$open, $high-$low))",
        )

    def test_mapping_lookup_and_missing_parameters_are_explicit(self):
        self.assertIs(get_operator_mapping("MAVG"), get_operator_mapping("mavg"))
        with self.assertRaisesRegex(KeyError, "unknown DolphinDB operator"):
            get_operator_mapping("not-an-operator")
        with self.assertRaisesRegex(ValueError, "missing parameter 'n'"):
            get_operator_mapping("mavg").render_expression(x="$close")

    def test_cross_sectional_operations_are_not_rolling_rank(self):
        row_rank = get_operator_mapping("rowRank")
        self.assertEqual(row_rank.axis, Axis.CROSS_SECTIONAL)
        self.assertEqual(row_rank.support_mode, SupportMode.PANEL_REQUIRED)
        self.assertNotIn("Rank", row_rank.qlib_operators)


class TestFactorRegistry(unittest.TestCase):
    @staticmethod
    def make_spec(**overrides):
        values = {
            "name": "WQAlpha001",
            "library": "wq101",
            "source_reference": "101 Formulaic Alphas, Appendix A",
            "source_version": "arXiv:1601.00991",
            "original_formula": "rank(Ts_ArgMax(SignedPower(...), 5))",
            "input_fields": ("close", "returns"),
            "axis_plan": (Axis.TIME_SERIES, Axis.CROSS_SECTIONAL),
            "window_policy": "Floor non-integer windows before evaluation.",
            "required_capabilities": frozenset({"daily_ohlcv"}),
            "status": FactorStatus.SUPPORTED,
        }
        values.update(overrides)
        return FactorSpec(**values)

    def test_empty_registry_and_supported_registration(self):
        registry = FactorRegistry({"daily_ohlcv"})
        self.assertEqual(registry.list(), ())
        spec = self.make_spec()
        registry.register(spec)
        self.assertIs(registry.get("WQ101", "wqalpha001"), spec)
        self.assertEqual(registry.list("wq101"), (spec,))

    def test_duplicate_names_are_rejected_case_insensitively(self):
        registry = FactorRegistry({"daily_ohlcv"})
        registry.register(self.make_spec())
        with self.assertRaisesRegex(FactorSemanticError, "duplicate factor"):
            registry.register(self.make_spec(name="wqalpha001"))

    def test_supported_factor_with_missing_capability_is_rejected(self):
        registry = FactorRegistry()
        with self.assertRaises(FactorDependencyError) as context:
            registry.register(self.make_spec())
        self.assertEqual(context.exception.factor_name, "WQAlpha001")
        self.assertEqual(context.exception.missing_capabilities, ("daily_ohlcv",))

    def test_missing_source_formula_and_unknown_axis_are_rejected(self):
        registry = FactorRegistry({"daily_ohlcv"})
        with self.assertRaisesRegex(FactorSemanticError, "source_reference"):
            registry.register(self.make_spec(source_reference=""))
        with self.assertRaisesRegex(FactorSemanticError, "original_formula"):
            registry.register(self.make_spec(original_formula=""))
        with self.assertRaisesRegex(FactorSemanticError, "axis plan"):
            registry.register(self.make_spec(axis_plan=("unknown",)))

    def test_original_factor_cannot_be_silently_adapted(self):
        registry = FactorRegistry({"daily_ohlcv"})
        with self.assertRaisesRegex(FactorSemanticError, "adapted factor"):
            registry.register(self.make_spec(status=FactorStatus.ADAPTED))
        with self.assertRaisesRegex(FactorSemanticError, "reuse the original"):
            registry.register(
                self.make_spec(
                    status=FactorStatus.ADAPTED,
                    is_original=False,
                    adapted_from="WQAlpha001",
                )
            )

        adapted = self.make_spec(
            name="WQAlpha001_A_SHARE_ADAPTED",
            status=FactorStatus.ADAPTED,
            is_original=False,
            adapted_from="WQAlpha001",
        )
        registry.register(adapted)
        self.assertIs(registry.get("wq101", adapted.name), adapted)


if __name__ == "__main__":
    unittest.main()
