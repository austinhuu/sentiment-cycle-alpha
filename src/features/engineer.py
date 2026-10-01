from __future__ import annotations

import logging
from typing import Any, Iterable

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


def _numeric(series: pd.Series) -> pd.Series:
    """把字段转换为数值，非法值转换为NaN。"""

    return pd.to_numeric(series, errors="coerce")


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """安全除法，分母为0时返回NaN。"""

    return numerator / denominator.replace(0.0, np.nan)


def _rsi(price: pd.Series, window: int) -> pd.Series:
    """基于连续价格计算简单滚动RSI。"""

    delta = price.diff()
    gain = delta.clip(lower=0.0).rolling(
        window,
        min_periods=window,
    ).mean()
    loss = (-delta.clip(upper=0.0)).rolling(
        window,
        min_periods=window,
    ).mean()

    relative_strength = gain / loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + relative_strength)

    return rsi.fillna(50.0)


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int,
) -> pd.Series:
    """在连续价格尺度上计算ATR。"""

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(
        window,
        min_periods=window,
    ).mean()


def _rolling_zscore(
    series: pd.Series,
    window: int,
    min_periods: int,
) -> pd.Series:
    """只用历史数据计算滚动Z-score。

    当前值不参与自己的均值和标准差计算，避免前视偏差。
    """

    mean = series.rolling(
        window,
        min_periods=min_periods,
    ).mean().shift(1)

    std = series.rolling(
        window,
        min_periods=min_periods,
    ).std().shift(1)

    return (
        (series - mean)
        / std.replace(0.0, np.nan)
    ).clip(-3.0, 3.0)


def _rolling_percentile_rank(
    series: pd.Series,
    window: int,
    min_periods: int,
) -> pd.Series:
    """计算当前值相对过去窗口的历史百分位。

    返回值范围为[0, 1]。滚动窗口的最后一个值是当前值，
    比较基准只使用它之前的历史值。
    """

    def percentile(values: np.ndarray) -> float:
        current = values[-1]
        history = values[:-1]
        history = history[np.isfinite(history)]

        if not np.isfinite(current) or len(history) < min_periods:
            return np.nan

        return float(np.mean(history <= current))

    return series.rolling(
        window + 1,
        min_periods=min_periods + 1,
    ).apply(
        percentile,
        raw=True,
    )


def _build_return(
    group: pd.DataFrame,
) -> tuple[pd.Series, str]:
    """构造经济收益率。

    优先使用Tushare的pct_chg，因为daily接口的原始OHLC没有复权，
    pct_chg则基于除权参考昨收pre_close计算。
    """

    if "pct_chg" in group.columns:
        pct_chg = _numeric(group["pct_chg"])
        if pct_chg.notna().any():
            return pct_chg / 100.0, "pct_chg"

    return _numeric(group["close"]).pct_change(), "raw_close_pct_change"


def _continuous_ohlc(
    group: pd.DataFrame,
    economic_return: pd.Series,
) -> pd.DataFrame:
    """由经济收益构造连续价格，避免除权断层污染技术指标。"""

    close = _numeric(group["close"])
    returns = economic_return.copy()

    if len(returns) > 0:
        returns.iloc[0] = 0.0

    signal_close = 100.0 * (
        1.0 + returns.fillna(0.0)
    ).cumprod()

    signal_open = signal_close * _safe_divide(
        _numeric(group["open"]),
        close,
    )
    signal_high = signal_close * _safe_divide(
        _numeric(group["high"]),
        close,
    )
    signal_low = signal_close * _safe_divide(
        _numeric(group["low"]),
        close,
    )

    signal_open = signal_open.fillna(signal_close)
    signal_high = signal_high.fillna(
        pd.concat(
            [signal_open, signal_close],
            axis=1,
        ).max(axis=1)
    )
    signal_low = signal_low.fillna(
        pd.concat(
            [signal_open, signal_close],
            axis=1,
        ).min(axis=1)
    )

    signal_high = pd.concat(
        [signal_high, signal_open, signal_close],
        axis=1,
    ).max(axis=1)
    signal_low = pd.concat(
        [signal_low, signal_open, signal_close],
        axis=1,
    ).min(axis=1)

    return pd.DataFrame(
        {
            "signal_open": signal_open,
            "signal_high": signal_high,
            "signal_low": signal_low,
            "signal_close": signal_close,
        },
        index=group.index,
    )


def _validate_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    defaults = {
        "ma_short": 5,
        "ma_mid": 10,
        "ma_long": 20,
        "rsi_window": 14,
        "atr_window": 14,
        "volatility_window": 20,
        "volume_window": 20,
        "breakout_window": 20,
        "momentum_window": 10,
        "drawdown_window": 60,
        "zscore_window": 120,
        "zscore_min_periods": 40,
        "risk_percentile_window": 252,
        "risk_percentile_min_periods": 60,
        "listing_warmup_bars": 20,
        "regime_trend_fast_window": 20,
        "regime_trend_slow_window": 60,
        "regime_momentum_window": 20,
        "regime_breadth_window": 10,
        "regime_volatility_window": 20,
        "regime_drawdown_window": 120,
        "regime_percentile_window": 252,
        "regime_percentile_min_periods": 60,
    }

    parsed: dict[str, Any] = {}

    for key, default in defaults.items():
        value = int(config.get(key, default))
        if value <= 0:
            raise ValueError(
                f"features.{key} 必须为正整数。"
            )
        parsed[key] = value

    horizons = {
        int(value)
        for value in config.get(
            "future_horizons",
            [1, 5, 20],
        )
        if int(value) > 0
    }
    parsed["future_horizons"] = sorted(horizons)

    return parsed


def _symbol_features(
    group: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    g = group.sort_values("datetime").copy()

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "pre_close",
        "change",
        "pct_chg",
    ]:
        if column in g.columns:
            g[column] = _numeric(g[column])

    return_1, source = _build_return(g)
    continuous = _continuous_ohlc(
        g,
        return_1,
    )

    g["return_1"] = return_1.copy()

    # 每只股票在当前样本中的第一条记录缺少完整前序信息。
    # 例如新股上市首日可能出现极端涨幅；这条收益不参与技术、
    # 风险和市场情绪特征，但连续价格仍以100作为基准正常构造。
    if len(g) > 0:
        g.loc[g.index[0], "return_1"] = np.nan

    g["days_since_listing"] = np.arange(
        len(g),
        dtype=int,
    )
    g["is_tradable_sample"] = (
        g["days_since_listing"]
        >= int(config["listing_warmup_bars"])
    )

    g["return_source"] = source
    g["raw_close_return_1"] = g["close"].pct_change()
    g["possible_raw_price_gap"] = (
        g["raw_close_return_1"].abs().gt(0.30)
        & g["return_1"].abs().lt(0.30)
    ).astype(int)

    for column in continuous.columns:
        g[column] = continuous[column]

    price = g["signal_close"]
    high = g["signal_high"]
    low = g["signal_low"]

    short = config["ma_short"]
    mid = config["ma_mid"]
    long = config["ma_long"]

    g["return_5"] = price.pct_change(5)
    g["return_20"] = price.pct_change(20)

    g[f"ma_{short}"] = price.rolling(
        short,
        min_periods=short,
    ).mean()
    g[f"ma_{mid}"] = price.rolling(
        mid,
        min_periods=mid,
    ).mean()
    g[f"ma_{long}"] = price.rolling(
        long,
        min_periods=long,
    ).mean()

    g["ma_gap_short_long"] = (
        _safe_divide(
            g[f"ma_{short}"],
            g[f"ma_{long}"],
        )
        - 1.0
    )
    g["ma_gap_mid_long"] = (
        _safe_divide(
            g[f"ma_{mid}"],
            g[f"ma_{long}"],
        )
        - 1.0
    )

    g["momentum"] = (
        price
        / price.shift(config["momentum_window"])
        - 1.0
    )
    g["rsi"] = _rsi(
        price,
        config["rsi_window"],
    )
    g["atr"] = _atr(
        high,
        low,
        price,
        config["atr_window"],
    )
    g["atr_pct"] = _safe_divide(
        g["atr"],
        price,
    )
    g["volatility"] = g["return_1"].rolling(
        config["volatility_window"],
        min_periods=config["volatility_window"],
    ).std()

    volume_average = g["volume"].rolling(
        config["volume_window"],
        min_periods=config["volume_window"],
    ).mean()
    g["volume_ratio"] = _safe_divide(
        g["volume"],
        volume_average,
    )

    prior_high = high.rolling(
        config["breakout_window"],
        min_periods=config["breakout_window"],
    ).max().shift(1)
    prior_low = low.rolling(
        config["breakout_window"],
        min_periods=config["breakout_window"],
    ).min().shift(1)

    g["breakout_up"] = (
        price > prior_high
    ).astype(float)
    g["breakout_down"] = (
        price < prior_low
    ).astype(float)

    rolling_low = low.rolling(
        config["breakout_window"],
        min_periods=config["breakout_window"],
    ).min()
    rolling_high = high.rolling(
        config["breakout_window"],
        min_periods=config["breakout_window"],
    ).max()

    g["price_position"] = _safe_divide(
        price - rolling_low,
        rolling_high - rolling_low,
    )

    rolling_peak = price.rolling(
        config["drawdown_window"],
        min_periods=1,
    ).max()
    g["rolling_drawdown"] = (
        _safe_divide(price, rolling_peak)
        - 1.0
    )
    g["drawdown_severity"] = (
        -g["rolling_drawdown"].clip(upper=0.0)
    )

    z_window = config["zscore_window"]
    z_minimum = config["zscore_min_periods"]

    for column in [
        "ma_gap_short_long",
        "ma_gap_mid_long",
        "momentum",
        "return_5",
        "volume_ratio",
        "volatility",
        "atr_pct",
        "rolling_drawdown",
    ]:
        g[f"{column}_z"] = _rolling_zscore(
            g[column],
            z_window,
            z_minimum,
        )

    percentile_window = config["risk_percentile_window"]
    percentile_minimum = config[
        "risk_percentile_min_periods"
    ]

    g["volatility_percentile"] = (
        _rolling_percentile_rank(
            g["volatility"],
            percentile_window,
            percentile_minimum,
        )
    )
    g["atr_percentile"] = (
        _rolling_percentile_rank(
            g["atr_pct"],
            percentile_window,
            percentile_minimum,
        )
    )
    g["drawdown_percentile"] = (
        _rolling_percentile_rank(
            g["drawdown_severity"],
            percentile_window,
            percentile_minimum,
        )
    )

    # 信号在t日收盘形成，未来标签从t+1日开盘开始计算。
    next_open = g["signal_open"].shift(-1)

    for horizon in config["future_horizons"]:
        exit_open = g["signal_open"].shift(
            -(horizon + 1)
        )
        g[f"future_return_{horizon}"] = (
            exit_open / next_open - 1.0
        )

    return g


def _market_features(
    features: pd.DataFrame,
    sentiment_pool: Iterable[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    sentiment_set = {
        str(symbol).strip().upper()
        for symbol in sentiment_pool
    }

    market_source = features.loc[
        features["symbol"].isin(sentiment_set)
        & features["is_tradable_sample"].fillna(False)
    ].copy()

    if market_source.empty:
        LOGGER.warning(
            "情绪参考池没有匹配的可用样本，退回使用全部已预热股票。"
        )
        market_source = features.loc[
            features["is_tradable_sample"].fillna(False)
        ].copy()

    if market_source.empty:
        raise RuntimeError(
            "市场情绪参考池在上市预热过滤后为空。"
            "请缩短listing_warmup_bars或扩大数据区间。"
        )

    market = market_source.groupby(
        "datetime",
        as_index=False,
    ).agg(
        market_breadth=(
            "return_1",
            lambda values: (
                float((values > 0).mean())
                if values.notna().any()
                else np.nan
            ),
        ),
        market_return=("return_1", "mean"),
        market_volume_ratio=(
            "volume_ratio",
            "mean",
        ),
        cross_section_volatility=(
            "return_1",
            "std",
        ),
        valid_symbol_count=(
            "symbol",
            "nunique",
        ),
        raw_price_gap_count=(
            "possible_raw_price_gap",
            "sum",
        ),
    ).sort_values("datetime")

    z_window = config["zscore_window"]
    z_minimum = config["zscore_min_periods"]

    for column in [
        "market_breadth",
        "market_return",
        "market_volume_ratio",
        "cross_section_volatility",
    ]:
        market[f"{column}_z"] = _rolling_zscore(
            market[column],
            z_window,
            z_minimum,
        )

    market["market_stress_percentile"] = (
        _rolling_percentile_rank(
            market["cross_section_volatility"],
            config["risk_percentile_window"],
            config["risk_percentile_min_periods"],
        )
    )

    # 使用情绪参考池的等权收益构造连续市场代理价格。
    market_returns = market["market_return"].copy()
    if len(market_returns) > 0:
        market_returns.iloc[0] = 0.0

    market["market_proxy_close"] = 100.0 * (
        1.0 + market_returns.fillna(0.0)
    ).cumprod()

    fast_window = config["regime_trend_fast_window"]
    slow_window = config["regime_trend_slow_window"]
    momentum_window = config["regime_momentum_window"]
    breadth_window = config["regime_breadth_window"]
    volatility_window = config["regime_volatility_window"]
    drawdown_window = config["regime_drawdown_window"]

    market["market_regime_ma_fast"] = (
        market["market_proxy_close"].rolling(
            fast_window,
            min_periods=fast_window,
        ).mean()
    )
    market["market_regime_ma_slow"] = (
        market["market_proxy_close"].rolling(
            slow_window,
            min_periods=slow_window,
        ).mean()
    )
    market["market_trend_gap"] = (
        _safe_divide(
            market["market_regime_ma_fast"],
            market["market_regime_ma_slow"],
        )
        - 1.0
    )
    market["market_momentum"] = (
        market["market_proxy_close"]
        / market["market_proxy_close"].shift(momentum_window)
        - 1.0
    )
    market["market_breadth_smooth"] = (
        market["market_breadth"].rolling(
            breadth_window,
            min_periods=breadth_window,
        ).mean()
    )
    market["market_volatility"] = (
        market["market_return"].rolling(
            volatility_window,
            min_periods=volatility_window,
        ).std()
    )
    market_peak = market["market_proxy_close"].rolling(
        drawdown_window,
        min_periods=1,
    ).max()
    market["market_drawdown"] = (
        _safe_divide(
            market["market_proxy_close"],
            market_peak,
        )
        - 1.0
    )

    market["market_trend_gap_z"] = _rolling_zscore(
        market["market_trend_gap"],
        config["zscore_window"],
        config["zscore_min_periods"],
    )
    market["market_momentum_z"] = _rolling_zscore(
        market["market_momentum"],
        config["zscore_window"],
        config["zscore_min_periods"],
    )
    market["market_breadth_smooth_z"] = _rolling_zscore(
        market["market_breadth_smooth"],
        config["zscore_window"],
        config["zscore_min_periods"],
    )
    market["market_volatility_z"] = _rolling_zscore(
        market["market_volatility"],
        config["zscore_window"],
        config["zscore_min_periods"],
    )
    market["market_volatility_percentile"] = (
        _rolling_percentile_rank(
            market["market_volatility"],
            config["regime_percentile_window"],
            config["regime_percentile_min_periods"],
        )
    )

    return market


def add_features(
    df: pd.DataFrame,
    config: dict[str, Any],
    sentiment_pool: Iterable[str] | None = None,
) -> pd.DataFrame:
    """生成技术、市场、风险和未来收益特征。"""

    required = [
        "datetime",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"特征工程缺少必要字段：{missing}"
        )

    parsed_config = _validate_config(config)
    working = df.copy()

    working["datetime"] = pd.to_datetime(
        working["datetime"],
        errors="coerce",
    )
    working["symbol"] = (
        working["symbol"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    working = working.dropna(
        subset=[
            "datetime",
            "symbol",
            "open",
            "high",
            "low",
            "close",
        ]
    )

    symbol_frames = [
        _symbol_features(group, parsed_config)
        for _, group in working.groupby(
            "symbol",
            sort=False,
        )
    ]

    if not symbol_frames:
        raise RuntimeError(
            "特征工程没有可处理的数据。"
        )

    features = pd.concat(
        symbol_frames,
        ignore_index=True,
    ).sort_values(
        ["datetime", "symbol"]
    ).reset_index(drop=True)

    market = _market_features(
        features,
        sentiment_pool
        or features["symbol"].unique(),
        parsed_config,
    )

    features = features.merge(
        market,
        on="datetime",
        how="left",
    )
    features = features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    gap_count = int(
        features[
            "possible_raw_price_gap"
        ].fillna(0).sum()
    )

    if gap_count > 0:
        LOGGER.warning(
            "检测到 %d 条疑似除权价格断层；"
            "收益和技术指标已使用pct_chg与连续价格。",
            gap_count,
        )

    return features