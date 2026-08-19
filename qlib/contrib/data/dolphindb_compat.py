"""Compatibility metadata for translating DolphinDB factors to Qlib.

This module deliberately contains no custom Qlib operators and no factor
implementations.  It records which translations can use the native expression
engine and which calculations must remain blocked for a panel-aware evaluator
or a separately reviewed custom operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, Iterable, Optional, Tuple


class Axis(str, Enum):
    """Calculation axis of an operator or factor node."""

    ELEMENTWISE = "elementwise"
    TIME_SERIES = "time_series"
    CROSS_SECTIONAL = "cross_sectional"


class SupportMode(str, Enum):
    """How a DolphinDB operation can be represented in Qlib."""

    DIRECT_NATIVE = "direct_native"
    COMPOSED_NATIVE = "composed_native"
    PANEL_REQUIRED = "panel_required"
    CUSTOM_REQUIRED = "custom_required"


@dataclass(frozen=True)
class OperatorMapping:
    """Descriptive mapping from one DolphinDB operation to Qlib."""

    dolphindb_name: str
    support_mode: SupportMode
    axis: Axis
    qlib_operators: Tuple[str, ...]
    expression_template: Optional[str]
    parameter_rules: str
    semantic_differences: Tuple[str, ...] = ()
    equivalence_verified: bool = False

    def __post_init__(self) -> None:
        if not self.dolphindb_name.strip():
            raise ValueError("dolphindb_name must not be empty")
        if len(set(self.qlib_operators)) != len(self.qlib_operators):
            raise ValueError(f"duplicate Qlib operators for {self.dolphindb_name}")

        can_render = self.support_mode in {
            SupportMode.DIRECT_NATIVE,
            SupportMode.COMPOSED_NATIVE,
        }
        if can_render != (self.expression_template is not None):
            raise ValueError(
                f"{self.dolphindb_name} must "
                f"{'define' if can_render else 'omit'} an expression template"
            )

    def render_expression(self, **parameters: object) -> str:
        """Render a native expression or fail closed for unsupported modes."""

        if self.expression_template is None:
            raise ValueError(
                f"{self.dolphindb_name} is {self.support_mode.value} and cannot emit "
                "a single-instrument Qlib expression"
            )
        try:
            return self.expression_template.format_map(parameters)
        except KeyError as exc:
            raise ValueError(
                f"missing parameter {exc.args[0]!r} for {self.dolphindb_name}"
            ) from exc


_PARTIAL_WINDOW = (
    "Qlib rolling operators use min_periods=1; DolphinDB moving functions "
    "default to a complete window."
)


_OPERATOR_MAPPINGS: Tuple[OperatorMapping, ...] = (
    OperatorMapping(
        "delay",
        SupportMode.DIRECT_NATIVE,
        Axis.TIME_SERIES,
        ("Ref",),
        "Ref({x}, {n})",
        "n is the positive lag in trading observations.",
        (_PARTIAL_WINDOW,),
    ),
    OperatorMapping(
        "mfirst",
        SupportMode.DIRECT_NATIVE,
        Axis.TIME_SERIES,
        ("Ref",),
        "Ref({x}, {lag})",
        "For mfirst(x, window), pass lag=window-1 after validating null handling.",
        (_PARTIAL_WINDOW, "General mfirst null behavior is not proven equivalent."),
    ),
    OperatorMapping(
        "delta",
        SupportMode.DIRECT_NATIVE,
        Axis.TIME_SERIES,
        ("Delta",),
        "Delta({x}, {n})",
        "n is the lag used in x[t] - x[t-n].",
    ),
    OperatorMapping(
        "mavg",
        SupportMode.DIRECT_NATIVE,
        Axis.TIME_SERIES,
        ("Mean",),
        "Mean({x}, {n})",
        "n is the integer rolling window.",
        (_PARTIAL_WINDOW,),
    ),
    OperatorMapping(
        "msum",
        SupportMode.DIRECT_NATIVE,
        Axis.TIME_SERIES,
        ("Sum",),
        "Sum({x}, {n})",
        "n is the integer rolling window.",
        (_PARTIAL_WINDOW,),
    ),
    OperatorMapping(
        "mstd",
        SupportMode.DIRECT_NATIVE,
        Axis.TIME_SERIES,
        ("Std",),
        "Std({x}, {n})",
        "Use DolphinDB mstd, not mstdp; both sides use sample standard deviation.",
        (_PARTIAL_WINDOW,),
    ),
    OperatorMapping(
        "mvar",
        SupportMode.DIRECT_NATIVE,
        Axis.TIME_SERIES,
        ("Var",),
        "Var({x}, {n})",
        "Use DolphinDB mvar, not mvarp; both sides use sample variance.",
        (_PARTIAL_WINDOW,),
    ),
    OperatorMapping(
        "mmax",
        SupportMode.DIRECT_NATIVE,
        Axis.TIME_SERIES,
        ("Max",),
        "Max({x}, {n})",
        "n is the integer rolling window.",
        (_PARTIAL_WINDOW,),
    ),
    OperatorMapping(
        "mmin",
        SupportMode.DIRECT_NATIVE,
        Axis.TIME_SERIES,
        ("Min",),
        "Min({x}, {n})",
        "n is the integer rolling window.",
        (_PARTIAL_WINDOW,),
    ),
    OperatorMapping(
        "mpercentile",
        SupportMode.DIRECT_NATIVE,
        Axis.TIME_SERIES,
        ("Quantile",),
        "Quantile({x}, {n}, {q})",
        "Convert DolphinDB percent from 0..100 to Qlib q in 0..1 and reorder arguments.",
        (_PARTIAL_WINDOW, "Interpolation must be linear for this direct mapping."),
    ),
    OperatorMapping(
        "mcorr",
        SupportMode.DIRECT_NATIVE,
        Axis.TIME_SERIES,
        ("Corr",),
        "Corr({x}, {y}, {n})",
        "n is the integer rolling window.",
        (_PARTIAL_WINDOW, "Zero-variance and near-zero tolerance behavior differs."),
    ),
    OperatorMapping(
        "mcovar",
        SupportMode.DIRECT_NATIVE,
        Axis.TIME_SERIES,
        ("Cov",),
        "Cov({x}, {y}, {n})",
        "n is the integer rolling window; sample covariance is required.",
        (_PARTIAL_WINDOW,),
    ),
    OperatorMapping(
        "abs",
        SupportMode.DIRECT_NATIVE,
        Axis.ELEMENTWISE,
        ("Abs",),
        "Abs({x})",
        "No parameter conversion.",
        equivalence_verified=True,
    ),
    OperatorMapping(
        "sign",
        SupportMode.DIRECT_NATIVE,
        Axis.ELEMENTWISE,
        ("Sign",),
        "Sign({x})",
        "No parameter conversion.",
        equivalence_verified=True,
    ),
    OperatorMapping(
        "log",
        SupportMode.DIRECT_NATIVE,
        Axis.ELEMENTWISE,
        ("Log",),
        "Log({x})",
        "Inputs must be positive under both implementations.",
    ),
    OperatorMapping(
        "pow",
        SupportMode.DIRECT_NATIVE,
        Axis.ELEMENTWISE,
        ("Power",),
        "Power({x}, {a})",
        "a is the exponent.",
    ),
    OperatorMapping(
        "iif",
        SupportMode.DIRECT_NATIVE,
        Axis.ELEMENTWISE,
        ("If",),
        "If({condition}, {when_true}, {when_false})",
        "The node is elementwise, but its children retain their declared axes.",
        ("Qlib evaluates both branches before selection."),
    ),
    OperatorMapping(
        "linearTimeTrend",
        SupportMode.DIRECT_NATIVE,
        Axis.TIME_SERIES,
        ("Slope", "Rsquare", "Resi"),
        "Slope({x}, {n})",
        "The template is the slope output; select Rsquare or Resi explicitly for those outputs.",
        (_PARTIAL_WINDOW, "Regression output and minimum observation rules require separate validation."),
    ),
    OperatorMapping(
        "ema",
        SupportMode.DIRECT_NATIVE,
        Axis.TIME_SERIES,
        ("EMA",),
        "EMA({x}, {n})",
        "Validate whether n denotes span or alpha before use.",
        ("Initialization and missing-value behavior require validation."),
    ),
    OperatorMapping(
        "mcount",
        SupportMode.DIRECT_NATIVE,
        Axis.TIME_SERIES,
        ("Count",),
        "Count({x}, {n})",
        "Counts non-null observations; n is the rolling window.",
        (_PARTIAL_WINDOW,),
    ),
    OperatorMapping(
        "SignedPower",
        SupportMode.COMPOSED_NATIVE,
        Axis.ELEMENTWISE,
        ("Sign", "Mul", "Power", "Abs"),
        "Mul(Sign({x}), Power(Abs({x}), {a}))",
        "a is the exponent.",
    ),
    OperatorMapping(
        "mimax",
        SupportMode.COMPOSED_NATIVE,
        Axis.TIME_SERIES,
        ("Sub", "IdxMax"),
        "Sub(IdxMax({x}, {n}), 1)",
        "Subtract one to convert Qlib's one-based position to a zero-based position.",
        (
            _PARTIAL_WINDOW,
            "Verified only for complete non-null windows; NaN windows are not equivalent.",
        ),
    ),
    OperatorMapping(
        "mimin",
        SupportMode.COMPOSED_NATIVE,
        Axis.TIME_SERIES,
        ("Sub", "IdxMin"),
        "Sub(IdxMin({x}, {n}), 1)",
        "Subtract one to convert Qlib's one-based position to a zero-based position.",
        (
            _PARTIAL_WINDOW,
            "Verified only for complete non-null windows; NaN windows are not equivalent.",
        ),
    ),
    OperatorMapping(
        "advN",
        SupportMode.COMPOSED_NATIVE,
        Axis.TIME_SERIES,
        ("Mean",),
        "Mean($amount, {n})",
        "Use traded amount, never volume; confirm amount units and adjustment policy.",
        (_PARTIAL_WINDOW,),
    ),
    OperatorMapping(
        "gt",
        SupportMode.DIRECT_NATIVE,
        Axis.ELEMENTWISE,
        ("Gt",),
        "Gt({left}, {right})",
        "Strict greater-than comparison.",
        equivalence_verified=True,
    ),
    OperatorMapping(
        "ge",
        SupportMode.DIRECT_NATIVE,
        Axis.ELEMENTWISE,
        ("Ge",),
        "Ge({left}, {right})",
        "Greater-than-or-equal comparison.",
        equivalence_verified=True,
    ),
    OperatorMapping(
        "lt",
        SupportMode.DIRECT_NATIVE,
        Axis.ELEMENTWISE,
        ("Lt",),
        "Lt({left}, {right})",
        "Strict less-than comparison.",
        equivalence_verified=True,
    ),
    OperatorMapping(
        "le",
        SupportMode.DIRECT_NATIVE,
        Axis.ELEMENTWISE,
        ("Le",),
        "Le({left}, {right})",
        "Less-than-or-equal comparison.",
        equivalence_verified=True,
    ),
    OperatorMapping(
        "eq",
        SupportMode.DIRECT_NATIVE,
        Axis.ELEMENTWISE,
        ("Eq",),
        "Eq({left}, {right})",
        "Equality comparison.",
        equivalence_verified=True,
    ),
    OperatorMapping(
        "ne",
        SupportMode.DIRECT_NATIVE,
        Axis.ELEMENTWISE,
        ("Ne",),
        "Ne({left}, {right})",
        "Inequality comparison.",
        equivalence_verified=True,
    ),
    OperatorMapping(
        "safe_divide",
        SupportMode.COMPOSED_NATIVE,
        Axis.ELEMENTWISE,
        ("If", "Eq", "Div"),
        "If(Eq({denominator}, 0), Div({denominator}, 0), "
        "Div({numerator}, {denominator}))",
        "The zero branch creates NaN as 0/0 without relying on an unregistered Python name.",
        ("Qlib evaluates both branches; warnings or infinities in the unselected branch may still occur."),
    ),
    OperatorMapping(
        "rowRank",
        SupportMode.PANEL_REQUIRED,
        Axis.CROSS_SECTIONAL,
        (),
        None,
        "Rank instruments within each date using explicit universe, tie, NaN, and percent rules.",
        ("Qlib Rank is a rolling time-series rank and is not a substitute."),
    ),
    OperatorMapping(
        "scale",
        SupportMode.PANEL_REQUIRED,
        Axis.CROSS_SECTIONAL,
        (),
        None,
        "Scale each date's cross-section by its sum of absolute values.",
        ("Universe membership and zero denominator behavior must be explicit."),
    ),
    OperatorMapping(
        "indneutralize",
        SupportMode.PANEL_REQUIRED,
        Axis.CROSS_SECTIONAL,
        (),
        None,
        "Demean within point-in-time industry groups for each date.",
        ("A current industry snapshot is not valid historical membership."),
    ),
    OperatorMapping(
        "nested_panel_formula",
        SupportMode.PANEL_REQUIRED,
        Axis.CROSS_SECTIONAL,
        (),
        None,
        "Evaluate an axis-typed panel DAG for cross-sectional/time-series nesting.",
        ("A single-instrument Qlib expression cannot represent alternating axes."),
    ),
    OperatorMapping(
        "mrank",
        SupportMode.CUSTOM_REQUIRED,
        Axis.TIME_SERIES,
        (),
        None,
        "Preserve zero-based rank, tiesMethod, ignoreNA, percent, and minPeriods semantics.",
        ("Qlib Rank returns a rolling percentile and is not an exact substitute."),
    ),
    OperatorMapping(
        "rolling_product",
        SupportMode.CUSTOM_REQUIRED,
        Axis.TIME_SERIES,
        (),
        None,
        "Confirm that no exact native expression composition exists before implementation.",
    ),
    OperatorMapping(
        "SMA",
        SupportMode.CUSTOM_REQUIRED,
        Axis.TIME_SERIES,
        (),
        None,
        "Preserve the recursive GTJA SMA(x, n, m) initialization and update rule.",
    ),
    OperatorMapping(
        "decay_linear",
        SupportMode.CUSTOM_REQUIRED,
        Axis.TIME_SERIES,
        (),
        None,
        "Use a custom operator only after proving native WMA numerically non-equivalent.",
        ("Qlib v0.9.7 WMA applies np.nanmean after normalizing weights."),
    ),
    OperatorMapping(
        "general_regression",
        SupportMode.CUSTOM_REQUIRED,
        Axis.TIME_SERIES,
        (),
        None,
        "Use only when Slope, Rsquare, and Resi cannot represent the required regression output.",
    ),
)


_OPERATOR_INDEX: Dict[str, OperatorMapping] = {
    mapping.dolphindb_name.casefold(): mapping for mapping in _OPERATOR_MAPPINGS
}
if len(_OPERATOR_INDEX) != len(_OPERATOR_MAPPINGS):
    raise RuntimeError("DolphinDB operator mapping names must be unique")


def get_operator_mapping(name: str) -> OperatorMapping:
    """Return one mapping by case-insensitive DolphinDB name."""

    try:
        return _OPERATOR_INDEX[name.casefold()]
    except (AttributeError, KeyError) as exc:
        raise KeyError(f"unknown DolphinDB operator mapping: {name!r}") from exc


def list_operator_mappings() -> Tuple[OperatorMapping, ...]:
    """Return all mappings in deterministic declaration order."""

    return _OPERATOR_MAPPINGS


class FactorStatus(str, Enum):
    """Validation and dependency state of a factor definition."""

    SUPPORTED = "supported"
    BLOCKED_DATA = "blocked_data"
    BLOCKED_SEMANTICS = "blocked_semantics"
    ADAPTED = "adapted"


class FactorDependencyError(RuntimeError):
    """Raised when a factor is marked supported without required inputs."""

    def __init__(self, factor_name: str, missing_capabilities: Iterable[str]):
        self.factor_name = factor_name
        self.missing_capabilities = tuple(sorted(set(missing_capabilities)))
        missing = ", ".join(self.missing_capabilities)
        super().__init__(f"factor {factor_name!r} is missing required capabilities: {missing}")


class FactorSemanticError(ValueError):
    """Raised when factor metadata would hide a semantic incompatibility."""


@dataclass(frozen=True)
class FactorSpec:
    """Auditable metadata for one original or explicitly adapted factor."""

    name: str
    library: str
    source_reference: str
    source_version: str
    original_formula: str
    input_fields: Tuple[str, ...]
    axis_plan: Tuple[Axis, ...]
    window_policy: str
    required_capabilities: FrozenSet[str]
    status: FactorStatus
    is_original: bool = True
    adapted_from: Optional[str] = None
    notes: str = ""


class FactorRegistry:
    """Fail-closed registry for audited factor specifications."""

    def __init__(self, available_capabilities: Iterable[str] = ()):
        self._available_capabilities = frozenset(available_capabilities)
        self._specs: Dict[Tuple[str, str], FactorSpec] = {}

    @staticmethod
    def _key(library: str, name: str) -> Tuple[str, str]:
        return library.casefold(), name.casefold()

    def register(self, spec: FactorSpec) -> None:
        """Validate and register one factor specification."""

        if not isinstance(spec, FactorSpec):
            raise TypeError("spec must be a FactorSpec")

        required_text = {
            "name": spec.name,
            "library": spec.library,
            "source_reference": spec.source_reference,
            "source_version": spec.source_version,
            "original_formula": spec.original_formula,
            "window_policy": spec.window_policy,
        }
        empty = [field for field, value in required_text.items() if not value.strip()]
        if empty:
            raise FactorSemanticError(f"factor metadata fields must not be empty: {', '.join(empty)}")

        if not spec.axis_plan or not all(isinstance(axis, Axis) for axis in spec.axis_plan):
            raise FactorSemanticError(f"factor {spec.name!r} has an unknown or empty axis plan")
        if not isinstance(spec.status, FactorStatus):
            raise FactorSemanticError(f"factor {spec.name!r} has an unknown status")

        if spec.status is FactorStatus.ADAPTED:
            if spec.is_original or not spec.adapted_from:
                raise FactorSemanticError(
                    f"adapted factor {spec.name!r} must be separately named and declare adapted_from"
                )
            if spec.name.casefold() == spec.adapted_from.casefold():
                raise FactorSemanticError("an adapted factor cannot reuse the original factor name")
        elif not spec.is_original or spec.adapted_from is not None:
            raise FactorSemanticError(
                f"non-adapted factor {spec.name!r} cannot carry adapted metadata"
            )

        missing = spec.required_capabilities - self._available_capabilities
        if spec.status is FactorStatus.SUPPORTED and missing:
            raise FactorDependencyError(spec.name, missing)

        key = self._key(spec.library, spec.name)
        if key in self._specs:
            raise FactorSemanticError(
                f"duplicate factor name {spec.name!r} in library {spec.library!r}"
            )
        self._specs[key] = spec

    def get(self, library: str, name: str) -> FactorSpec:
        """Return one registered factor specification."""

        try:
            return self._specs[self._key(library, name)]
        except KeyError as exc:
            raise KeyError(f"unknown factor {library}.{name}") from exc

    def list(self, library: Optional[str] = None) -> Tuple[FactorSpec, ...]:
        """Return registered factors in deterministic library/name order."""

        specs = self._specs.values()
        if library is not None:
            library_key = library.casefold()
            specs = (spec for spec in specs if spec.library.casefold() == library_key)
        return tuple(sorted(specs, key=lambda spec: (spec.library.casefold(), spec.name.casefold())))


__all__ = [
    "Axis",
    "FactorDependencyError",
    "FactorRegistry",
    "FactorSemanticError",
    "FactorSpec",
    "FactorStatus",
    "OperatorMapping",
    "SupportMode",
    "get_operator_mapping",
    "list_operator_mappings",
]
