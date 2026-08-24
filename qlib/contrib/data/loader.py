import json
from pathlib import Path

import pandas as pd
import pyarrow.dataset as pyarrow_dataset

from qlib.contrib.data.quantml_factors import (
    OPERATOR_VERSION,
    SOURCE_SHA256,
    QuantMLDataContext,
    compute_quantml_factors,
    list_quantml_factors,
)
from qlib.data import D
from qlib.data.dataset.loader import DataLoader, QlibDataLoader


class Alpha360DL(QlibDataLoader):
    """Dataloader to get Alpha360"""

    def __init__(self, config=None, **kwargs):
        _config = {
            "feature": self.get_feature_config(),
        }
        if config is not None:
            _config.update(config)
        super().__init__(config=_config, **kwargs)

    @staticmethod
    def get_feature_config():
        # NOTE:
        # Alpha360 tries to provide a dataset with original price data
        # the original price data includes the prices and volume in the last 60 days.
        # To make it easier to learn models from this dataset, all the prices and volume
        # are normalized by the latest price and volume data ( dividing by $close, $volume)
        # So the latest normalized $close will be 1 (with name CLOSE0), the latest normalized $volume will be 1 (with name VOLUME0)
        # If further normalization are executed (e.g. centralization),  CLOSE0 and VOLUME0 will be 0.
        fields = []
        names = []

        for i in range(59, 0, -1):
            fields += ["Ref($close, %d)/$close" % i]
            names += ["CLOSE%d" % i]
        fields += ["$close/$close"]
        names += ["CLOSE0"]
        for i in range(59, 0, -1):
            fields += ["Ref($open, %d)/$close" % i]
            names += ["OPEN%d" % i]
        fields += ["$open/$close"]
        names += ["OPEN0"]
        for i in range(59, 0, -1):
            fields += ["Ref($high, %d)/$close" % i]
            names += ["HIGH%d" % i]
        fields += ["$high/$close"]
        names += ["HIGH0"]
        for i in range(59, 0, -1):
            fields += ["Ref($low, %d)/$close" % i]
            names += ["LOW%d" % i]
        fields += ["$low/$close"]
        names += ["LOW0"]
        for i in range(59, 0, -1):
            fields += ["Ref($vwap, %d)/$close" % i]
            names += ["VWAP%d" % i]
        fields += ["$vwap/$close"]
        names += ["VWAP0"]
        for i in range(59, 0, -1):
            fields += ["Ref($volume, %d)/($volume+1e-12)" % i]
            names += ["VOLUME%d" % i]
        fields += ["$volume/($volume+1e-12)"]
        names += ["VOLUME0"]

        return fields, names


class QuantMLAdaptedDL(DataLoader):
    """Load local-only QuantML adapted minute factors as daily features."""

    def __init__(
        self,
        factors,
        context=None,
        use_cache=True,
        feature_group="feature",
        label=None,
        label_names=None,
        materialized_uri=None,
        enforce_market_membership=True,
    ):
        self.factors = factors
        if context is None or isinstance(context, QuantMLDataContext):
            self.context = context
        elif isinstance(context, dict):
            self.context = QuantMLDataContext(**context)
        else:
            raise TypeError("context must be QuantMLDataContext, dict, or None")
        self.use_cache = use_cache
        self.feature_group = feature_group
        self.label = [] if label is None else ([label] if isinstance(label, str) else list(label))
        self.label_names = (
            [f"LABEL{index}" for index in range(len(self.label))]
            if label_names is None
            else list(label_names)
        )
        if len(self.label) != len(self.label_names):
            raise ValueError("label and label_names must have the same length")
        self.materialized_uri = materialized_uri
        self.enforce_market_membership = enforce_market_membership

    @staticmethod
    def _resolve_instruments(instruments, start_time, end_time):
        if instruments is None:
            raise ValueError("QuantMLAdaptedDL requires explicit instruments or a Qlib market name")
        if isinstance(instruments, str):
            config = D.instruments(instruments)
            membership = D.list_instruments(
                config,
                start_time=start_time,
                end_time=end_time,
                as_list=False,
            )
            return list(membership), config, membership
        values = list(instruments)
        return values, values, None

    @staticmethod
    def _filter_membership(data, membership):
        if membership is None or data.empty:
            return data
        pieces = []
        available = set(data.index.get_level_values("instrument"))
        for instrument, spans in membership.items():
            if instrument not in available:
                continue
            current = data.xs(instrument, level="instrument", drop_level=False)
            dates = current.index.get_level_values("datetime")
            mask = pd.Series(False, index=current.index)
            for start, end in spans:
                mask |= (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
            pieces.append(current.loc[mask.to_numpy()])
        if not pieces:
            return data.iloc[0:0]
        return pd.concat(pieces).sort_index()

    @staticmethod
    def _membership_index(membership, start_time, end_time):
        if membership is None:
            return None
        calendar = pd.DatetimeIndex(D.calendar(start_time=start_time, end_time=end_time, freq="day"))
        pairs = []
        for instrument, spans in membership.items():
            for start, end in spans:
                dates = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
                pairs.extend((date, instrument) for date in dates)
        return pd.MultiIndex.from_tuples(sorted(set(pairs)), names=["datetime", "instrument"])

    def _load_materialized(self, instruments, start_time, end_time):
        root = Path(self.materialized_uri).expanduser().resolve()
        manifest_path = root / "_MANIFEST.json" if root.is_dir() else None
        if manifest_path is not None:
            if not manifest_path.is_file():
                raise FileNotFoundError(f"materialized QuantML manifest is missing: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("catalog_sha256") != SOURCE_SHA256:
                raise ValueError("materialized QuantML catalog SHA does not match the active catalog")
            if manifest.get("operator_version") != OPERATOR_VERSION:
                raise ValueError("materialized QuantML operator version does not match the active implementation")
        paths = [root] if root.is_file() else sorted(root.glob("year=*/code=*/factors.parquet"))
        if not paths:
            raise FileNotFoundError(f"no QuantML materialized factors found: {root}")
        requested = (
            [factor.name for factor in list_quantml_factors(available=True)]
            if self.factors == "all_supported"
            else ([self.factors] if isinstance(self.factors, str) else list(self.factors))
        )
        dataset = pyarrow_dataset.dataset([str(path) for path in paths], format="parquet")
        missing = set(["datetime", "instrument", *requested]) - set(dataset.schema.names)
        if missing:
            raise ValueError(f"materialized QuantML store is missing columns: {sorted(missing)}")
        table = dataset.to_table(
            columns=["datetime", "instrument", *requested],
            filter=(pyarrow_dataset.field("datetime") >= pd.Timestamp(start_time))
            & (pyarrow_dataset.field("datetime") <= pd.Timestamp(end_time)),
        )
        data = table.to_pandas(split_blocks=True, self_destruct=True)
        data["datetime"] = pd.to_datetime(data["datetime"])
        data["instrument"] = data["instrument"].str.upper()
        data = data[data["instrument"].isin(instruments)]
        if data.duplicated(["datetime", "instrument"]).any():
            raise ValueError("materialized QuantML store contains duplicate datetime/instrument rows")
        return data.set_index(["datetime", "instrument"])[requested].sort_index()

    def load(self, instruments=None, start_time=None, end_time=None):
        codes, qlib_instruments, membership = self._resolve_instruments(instruments, start_time, end_time)
        if self.materialized_uri is None:
            data = compute_quantml_factors(
                factors=self.factors,
                instruments=codes,
                start_time=start_time,
                end_time=end_time,
                context=self.context,
                use_cache=self.use_cache,
            )
        else:
            data = self._load_materialized(codes, start_time, end_time)
        if self.enforce_market_membership:
            data = self._filter_membership(data, membership)
            membership_index = self._membership_index(membership, start_time, end_time)
            if membership_index is not None:
                data = data.reindex(membership_index)
        feature = data
        if self.feature_group is not None:
            feature.columns = pd.MultiIndex.from_product([[self.feature_group], feature.columns])
        if not self.label:
            return feature
        labels = D.features(
            qlib_instruments,
            self.label,
            start_time=start_time,
            end_time=end_time,
            freq="day",
        )
        if labels.index.names == ["instrument", "datetime"]:
            labels = labels.swaplevel().sort_index()
        labels.index.names = ["datetime", "instrument"]
        labels.columns = pd.MultiIndex.from_product([["label"], self.label_names])
        return pd.concat([feature, labels.reindex(feature.index)], axis=1).sort_index()


class Alpha158DL(QlibDataLoader):
    """Dataloader to get Alpha158"""

    def __init__(self, config=None, **kwargs):
        _config = {
            "feature": self.get_feature_config(),
        }
        if config is not None:
            _config.update(config)
        super().__init__(config=_config, **kwargs)

    @staticmethod
    def get_feature_config(
        config={
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
    ):
        """create factors from config

        config = {
            'kbar': {}, # whether to use some hard-code kbar features
            'price': { # whether to use raw price features
                'windows': [0, 1, 2, 3, 4], # use price at n days ago
                'feature': ['OPEN', 'HIGH', 'LOW'] # which price field to use
            },
            'volume': { # whether to use raw volume features
                'windows': [0, 1, 2, 3, 4], # use volume at n days ago
            },
            'rolling': { # whether to use rolling operator based features
                'windows': [5, 10, 20, 30, 60], # rolling windows size
                'include': ['ROC', 'MA', 'STD'], # rolling operator to use
                #if include is None we will use default operators
                'exclude': ['RANK'], # rolling operator not to use
            }
        }
        """
        fields = []
        names = []
        if "kbar" in config:
            fields += [
                "($close-$open)/$open",
                "($high-$low)/$open",
                "($close-$open)/($high-$low+1e-12)",
                "($high-Greater($open, $close))/$open",
                "($high-Greater($open, $close))/($high-$low+1e-12)",
                "(Less($open, $close)-$low)/$open",
                "(Less($open, $close)-$low)/($high-$low+1e-12)",
                "(2*$close-$high-$low)/$open",
                "(2*$close-$high-$low)/($high-$low+1e-12)",
            ]
            names += [
                "KMID",
                "KLEN",
                "KMID2",
                "KUP",
                "KUP2",
                "KLOW",
                "KLOW2",
                "KSFT",
                "KSFT2",
            ]
        if "price" in config:
            windows = config["price"].get("windows", range(5))
            feature = config["price"].get("feature", ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"])
            for field in feature:
                field = field.lower()
                fields += ["Ref($%s, %d)/$close" % (field, d) if d != 0 else "$%s/$close" % field for d in windows]
                names += [field.upper() + str(d) for d in windows]
        if "volume" in config:
            windows = config["volume"].get("windows", range(5))
            fields += ["Ref($volume, %d)/($volume+1e-12)" % d if d != 0 else "$volume/($volume+1e-12)" for d in windows]
            names += ["VOLUME" + str(d) for d in windows]
        if "rolling" in config:
            windows = config["rolling"].get("windows", [5, 10, 20, 30, 60])
            include = config["rolling"].get("include", None)
            exclude = config["rolling"].get("exclude", [])
            # `exclude` in dataset config unnecessary filed
            # `include` in dataset config necessary field

            def use(x):
                return x not in exclude and (include is None or x in include)

            # Some factor ref: https://guorn.com/static/upload/file/3/134065454575605.pdf
            if use("ROC"):
                # https://www.investopedia.com/terms/r/rateofchange.asp
                # Rate of change, the price change in the past d days, divided by latest close price to remove unit
                fields += ["Ref($close, %d)/$close" % d for d in windows]
                names += ["ROC%d" % d for d in windows]
            if use("MA"):
                # https://www.investopedia.com/ask/answers/071414/whats-difference-between-moving-average-and-weighted-moving-average.asp
                # Simple Moving Average, the simple moving average in the past d days, divided by latest close price to remove unit
                fields += ["Mean($close, %d)/$close" % d for d in windows]
                names += ["MA%d" % d for d in windows]
            if use("STD"):
                # The standard diviation of close price for the past d days, divided by latest close price to remove unit
                fields += ["Std($close, %d)/$close" % d for d in windows]
                names += ["STD%d" % d for d in windows]
            if use("BETA"):
                # The rate of close price change in the past d days, divided by latest close price to remove unit
                # For example, price increase 10 dollar per day in the past d days, then Slope will be 10.
                fields += ["Slope($close, %d)/$close" % d for d in windows]
                names += ["BETA%d" % d for d in windows]
            if use("RSQR"):
                # The R-sqaure value of linear regression for the past d days, represent the trend linear
                fields += ["Rsquare($close, %d)" % d for d in windows]
                names += ["RSQR%d" % d for d in windows]
            if use("RESI"):
                # The redisdual for linear regression for the past d days, represent the trend linearity for past d days.
                fields += ["Resi($close, %d)/$close" % d for d in windows]
                names += ["RESI%d" % d for d in windows]
            if use("MAX"):
                # The max price for past d days, divided by latest close price to remove unit
                fields += ["Max($high, %d)/$close" % d for d in windows]
                names += ["MAX%d" % d for d in windows]
            if use("LOW"):
                # The low price for past d days, divided by latest close price to remove unit
                fields += ["Min($low, %d)/$close" % d for d in windows]
                names += ["MIN%d" % d for d in windows]
            if use("QTLU"):
                # The 80% quantile of past d day's close price, divided by latest close price to remove unit
                # Used with MIN and MAX
                fields += ["Quantile($close, %d, 0.8)/$close" % d for d in windows]
                names += ["QTLU%d" % d for d in windows]
            if use("QTLD"):
                # The 20% quantile of past d day's close price, divided by latest close price to remove unit
                fields += ["Quantile($close, %d, 0.2)/$close" % d for d in windows]
                names += ["QTLD%d" % d for d in windows]
            if use("RANK"):
                # Get the percentile of current close price in past d day's close price.
                # Represent the current price level comparing to past N days, add additional information to moving average.
                fields += ["Rank($close, %d)" % d for d in windows]
                names += ["RANK%d" % d for d in windows]
            if use("RSV"):
                # Represent the price position between upper and lower resistent price for past d days.
                fields += ["($close-Min($low, %d))/(Max($high, %d)-Min($low, %d)+1e-12)" % (d, d, d) for d in windows]
                names += ["RSV%d" % d for d in windows]
            if use("IMAX"):
                # The number of days between current date and previous highest price date.
                # Part of Aroon Indicator https://www.investopedia.com/terms/a/aroon.asp
                # The indicator measures the time between highs and the time between lows over a time period.
                # The idea is that strong uptrends will regularly see new highs, and strong downtrends will regularly see new lows.
                fields += ["IdxMax($high, %d)/%d" % (d, d) for d in windows]
                names += ["IMAX%d" % d for d in windows]
            if use("IMIN"):
                # The number of days between current date and previous lowest price date.
                # Part of Aroon Indicator https://www.investopedia.com/terms/a/aroon.asp
                # The indicator measures the time between highs and the time between lows over a time period.
                # The idea is that strong uptrends will regularly see new highs, and strong downtrends will regularly see new lows.
                fields += ["IdxMin($low, %d)/%d" % (d, d) for d in windows]
                names += ["IMIN%d" % d for d in windows]
            if use("IMXD"):
                # The time period between previous lowest-price date occur after highest price date.
                # Large value suggest downward momemtum.
                fields += ["(IdxMax($high, %d)-IdxMin($low, %d))/%d" % (d, d, d) for d in windows]
                names += ["IMXD%d" % d for d in windows]
            if use("CORR"):
                # The correlation between absolute close price and log scaled trading volume
                fields += ["Corr($close, Log($volume+1), %d)" % d for d in windows]
                names += ["CORR%d" % d for d in windows]
            if use("CORD"):
                # The correlation between price change ratio and volume change ratio
                fields += ["Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), %d)" % d for d in windows]
                names += ["CORD%d" % d for d in windows]
            if use("CNTP"):
                # The percentage of days in past d days that price go up.
                fields += ["Mean($close>Ref($close, 1), %d)" % d for d in windows]
                names += ["CNTP%d" % d for d in windows]
            if use("CNTN"):
                # The percentage of days in past d days that price go down.
                fields += ["Mean($close<Ref($close, 1), %d)" % d for d in windows]
                names += ["CNTN%d" % d for d in windows]
            if use("CNTD"):
                # The diff between past up day and past down day
                fields += ["Mean($close>Ref($close, 1), %d)-Mean($close<Ref($close, 1), %d)" % (d, d) for d in windows]
                names += ["CNTD%d" % d for d in windows]
            if use("SUMP"):
                # The total gain / the absolute total price changed
                # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
                fields += [
                    "Sum(Greater($close-Ref($close, 1), 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d)
                    for d in windows
                ]
                names += ["SUMP%d" % d for d in windows]
            if use("SUMN"):
                # The total lose / the absolute total price changed
                # Can be derived from SUMP by SUMN = 1 - SUMP
                # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
                fields += [
                    "Sum(Greater(Ref($close, 1)-$close, 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d)
                    for d in windows
                ]
                names += ["SUMN%d" % d for d in windows]
            if use("SUMD"):
                # The diff ratio between total gain and total lose
                # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
                fields += [
                    "(Sum(Greater($close-Ref($close, 1), 0), %d)-Sum(Greater(Ref($close, 1)-$close, 0), %d))"
                    "/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d, d)
                    for d in windows
                ]
                names += ["SUMD%d" % d for d in windows]
            if use("VMA"):
                # Simple Volume Moving average: https://www.barchart.com/education/technical-indicators/volume_moving_average
                fields += ["Mean($volume, %d)/($volume+1e-12)" % d for d in windows]
                names += ["VMA%d" % d for d in windows]
            if use("VSTD"):
                # The standard deviation for volume in past d days.
                fields += ["Std($volume, %d)/($volume+1e-12)" % d for d in windows]
                names += ["VSTD%d" % d for d in windows]
            if use("WVMA"):
                # The volume weighted price change volatility
                fields += [
                    "Std(Abs($close/Ref($close, 1)-1)*$volume, %d)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, %d)+1e-12)"
                    % (d, d)
                    for d in windows
                ]
                names += ["WVMA%d" % d for d in windows]
            if use("VSUMP"):
                # The total volume increase / the absolute total volume changed
                fields += [
                    "Sum(Greater($volume-Ref($volume, 1), 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)"
                    % (d, d)
                    for d in windows
                ]
                names += ["VSUMP%d" % d for d in windows]
            if use("VSUMN"):
                # The total volume increase / the absolute total volume changed
                # Can be derived from VSUMP by VSUMN = 1 - VSUMP
                fields += [
                    "Sum(Greater(Ref($volume, 1)-$volume, 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)"
                    % (d, d)
                    for d in windows
                ]
                names += ["VSUMN%d" % d for d in windows]
            if use("VSUMD"):
                # The diff ratio between total volume increase and total volume decrease
                # RSI indicator for volume
                fields += [
                    "(Sum(Greater($volume-Ref($volume, 1), 0), %d)-Sum(Greater(Ref($volume, 1)-$volume, 0), %d))"
                    "/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)" % (d, d, d)
                    for d in windows
                ]
                names += ["VSUMD%d" % d for d in windows]

        return fields, names
