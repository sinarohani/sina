import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests
from requests.adapters import HTTPAdapter, Retry
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# =========================
# CONFIG
# =========================

# Bybit V5 API
BYBIT_URL = "https://api.bybit.com/v5/market/kline"

# Bybit product category
# linear = USDT/USDC perpetual & linear contracts
BYBIT_CATEGORY = "linear"

TELEGRAM_URL = "https://api.telegram.org/bot{}/sendMessage"


SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "SOLUSDT",
    "TRXUSDT",
    "ZECUSDT",
    "HYPEUSDT",
    "DOGEUSDT",
    "LINKUSDT",
    "XMRUSDT",
    "ADAUSDT",
    "XLMUSDT",
    "BCHUSDT",
    "NEARUSDT",
    "UNIUSDT",
    "LTCUSDT",
    "CCUSDT",
    "AVAXUSDT",
    "SUIUSDT",
    "HBARUSDT",
    "TAOUSDT",
    "SHIBUSDT",
    "CROUSDT",
    "ENAUSDT",
    "ONDOUSDT",
    "AAVEUSDT",
    "MNTUSDT",
    "DOTUSDT",
    "PUMPUSDT",
    "ASTERUSDT",
    "WLDUSDT",
    "WLFIUSDT",
    "SKYUSDT",
    "PEPEUSDT",
    "ICPUSDT",
    "ARBUSDT",
    "ETCUSDT",
    "KASUSDT",
    "POLUSDT",
]


# Bybit interval format:
# 15  = 15 minutes
# 60  = 1 hour
# 240 = 4 hours
# D   = 1 day

TIMEFRAMES = {
    "15m": "15",
    "1h": "60",
    "4h": "240",
    "1D": "D",
}


# =========================
# ICHIMOKU
# =========================

TENKAN, KIJUN, SENKOU_B, DISPLACEMENT = 9, 26, 52, 26

MIN_CONFIRMATIONS = 4
MIN_TF_CONFIRMATIONS = 2


# =========================
# EXTRA FILTERS
# =========================

KUMO_THICKNESS_MIN_PCT = 0.30
BREAKOUT_MARGIN_PCT = 0.15

ADX_PERIOD = 14
ADX_MIN = 20

EMA_PERIOD = 200

RSI_PERIOD = 14

BB_PERIOD = 20
BB_STD = 2
BB_WIDTH_MIN_PCT = 1.0

VOLUME_MA_PERIOD = 20
VOLUME_SPIKE_MULT = 1.2

SWING_WINDOW = 5
STRUCTURE_MARGIN_PCT = 0.25


# =========================
# BTC CORRELATION
# =========================

BTC_SYMBOL = "BTCUSDT"
BTC_CORRELATION_TF = "4h"


# =========================
# CANDLE SETTINGS
# =========================

CANDLE_LIMIT = 300
CANDLE_FETCH_BUFFER = 60

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

MAX_WORKERS = 6


logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)

log = logging.getLogger("ichimoku-bybit-bot")


# =========================
# ENV
# =========================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()


if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise SystemExit(
        "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env"
    )


# =========================
# HTTP SESSION
# =========================

def build_session():
    session = requests.Session()

    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )

    adapter = HTTPAdapter(
        max_retries=retries,
        pool_maxsize=MAX_WORKERS + 2,
    )

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


SESSION = build_session()


# =========================
# BYBIT
# =========================

def interval_to_seconds(interval):
    return {
        "15": 15 * 60,
        "60": 60 * 60,
        "240": 240 * 60,
        "D": 24 * 60 * 60,
    }[interval]


def fetch_klines(symbol, interval):
    """
    Fetch CLOSED candles from Bybit V5.

    Bybit response format:

    [
        startTime,
        open,
        high,
        low,
        close,
        volume,
        turnover
    ]

    Important:
    Bybit returns candles newest -> oldest.
    We reverse them to oldest -> newest.
    """

    seconds = interval_to_seconds(interval)

    params = {
        "category": BYBIT_CATEGORY,
        "symbol": symbol,
        "interval": interval,
        "limit": CANDLE_LIMIT + CANDLE_FETCH_BUFFER,
    }

    r = SESSION.get(
        BYBIT_URL,
        params=params,
        timeout=15,
    )

    r.raise_for_status()

    payload = r.json()

    # =========================
    # BYBIT ERROR CHECK
    # =========================

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Unexpected Bybit response for {symbol} {interval}: {payload}"
        )

    ret_code = payload.get("retCode")

    if ret_code != 0:
        raise RuntimeError(
            f"Bybit error for {symbol} {interval}: "
            f"retCode={ret_code}, "
            f"retMsg={payload.get('retMsg')}"
        )

    result = payload.get("result", {})
    rows = result.get("list", [])

    if not rows:
        raise RuntimeError(
            f"No kline data from Bybit for {symbol} {interval}"
        )

    # =========================
    # DATAFRAME
    # =========================

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
        ],
    )

    # Numeric conversion
    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df["open_time"] = pd.to_numeric(
        df["open_time"],
        errors="coerce",
    )

    # =========================
    # TIMESTAMP
    # =========================

    df["timestamp"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True,
    )

    df = df[
        [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ]

    df = (
        df
        .dropna()
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # =========================
    # REMOVE CURRENT OPEN CANDLE
    # =========================

    now_ts = pd.Timestamp.now(tz="UTC")

    df = df[
        (df["timestamp"] + pd.Timedelta(seconds=seconds))
        <= now_ts
    ].copy()

    # =========================
    # MINIMUM DATA CHECK
    # =========================

    minimum_required = (
        SENKOU_B
        + DISPLACEMENT
        + 5
    )

    if len(df) < minimum_required:
        raise RuntimeError(
            f"Not enough CLOSED candles for "
            f"{symbol} {interval}: {len(df)}"
        )

    # Keep latest CANDLE_LIMIT candles
    df = df.tail(CANDLE_LIMIT)

    return df.reset_index(drop=True)


# =========================
# INDICATORS
# =========================

def add_ichimoku(df):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    df["tenkan"] = (
        high.rolling(TENKAN).max()
        + low.rolling(TENKAN).min()
    ) / 2

    df["kijun"] = (
        high.rolling(KIJUN).max()
        + low.rolling(KIJUN).min()
    ) / 2

    df["span_a"] = (
        df["tenkan"]
        + df["kijun"]
    ) / 2

    df["span_b"] = (
        high.rolling(SENKOU_B).max()
        + low.rolling(SENKOU_B).min()
    ) / 2

    df["cloud_top"] = df[
        ["span_a", "span_b"]
    ].max(axis=1)

    df["cloud_bottom"] = df[
        ["span_a", "span_b"]
    ].min(axis=1)

    df["kumo_thickness_pct"] = (
        (df["cloud_top"] - df["cloud_bottom"])
        / close
        * 100
    )

    return df


def add_adx(df, period=ADX_PERIOD):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = pd.Series(
        0.0,
        index=df.index,
    )

    minus_dm = pd.Series(
        0.0,
        index=df.index,
    )

    plus_dm[
        (up_move > down_move)
        & (up_move > 0)
    ] = up_move

    minus_dm[
        (down_move > up_move)
        & (down_move > 0)
    ] = down_move

    atr = tr.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    plus_di = (
        100
        * plus_dm.ewm(
            alpha=1 / period,
            adjust=False,
        ).mean()
        / atr
    )

    minus_di = (
        100
        * minus_dm.ewm(
            alpha=1 / period,
            adjust=False,
        ).mean()
        / atr
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di)
    )

    df["adx"] = dx.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    return df


def add_ema(df, period=EMA_PERIOD):
    df["ema"] = df["close"].ewm(
        span=period,
        adjust=False,
    ).mean()

    return df


def add_rsi(df, period=RSI_PERIOD):
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss

    df["rsi"] = 100 - (
        100 / (1 + rs)
    )

    return df


def add_bollinger(
    df,
    period=BB_PERIOD,
    n_std=BB_STD,
):
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()

    df["bb_width_pct"] = (
        (2 * n_std * std)
        / sma
        * 100
    )

    return df


def add_volume_features(
    df,
    period=VOLUME_MA_PERIOD,
):
    df["volume_ma"] = (
        df["volume"]
        .rolling(period)
        .mean()
    )

    direction = df["close"].diff().apply(
        lambda x:
        1 if x > 0
        else (-1 if x < 0 else 0)
    )

    df["obv"] = (
        direction * df["volume"]
    ).cumsum()

    df["obv_slope"] = (
        df["obv"].diff(5)
    )

    return df


def add_swing_structure(
    df,
    window=SWING_WINDOW,
):
    is_high = (
        df["high"]
        == df["high"]
        .rolling(
            2 * window + 1,
            center=True,
        )
        .max()
    )

    is_low = (
        df["low"]
        == df["low"]
        .rolling(
            2 * window + 1,
            center=True,
        )
        .min()
    )

    confirmed_high = (
        is_high
        .shift(window)
        .fillna(False)
    )

    confirmed_low = (
        is_low
        .shift(window)
        .fillna(False)
    )

    df["last_swing_high"] = (
        df["high"]
        .shift(window)
        .where(confirmed_high)
        .ffill()
    )

    df["last_swing_low"] = (
        df["low"]
        .shift(window)
        .where(confirmed_low)
        .ffill()
    )

    return df


def build_indicators(df):
    df = add_ichimoku(df)
    df = add_adx(df)
    df = add_ema(df)
    df = add_rsi(df)
    df = add_bollinger(df)
    df = add_volume_features(df)
    df = add_swing_structure(df)

    return df


# =========================
# SIGNAL LOGIC
# =========================

def compute_confirmations(df, i):
    """
    Pure Ichimoku confirmations only.
    """

    close_s = df["close"]
    span_a_s = df["span_a"]
    span_b_s = df["span_b"]

    row = df.iloc[i]

    bull = []
    bear = []

    # Price vs Kumo
    if row["close"] > row["cloud_top"]:
        bull.append("Price > Kumo")

    elif row["close"] < row["cloud_bottom"]:
        bear.append("Price < Kumo")

    # Tenkan vs Kijun
    if row["tenkan"] > row["kijun"]:
        bull.append("Tenkan > Kijun")

    elif row["tenkan"] < row["kijun"]:
        bear.append("Tenkan < Kijun")

    # Kumo direction
    if row["span_a"] > row["span_b"]:
        bull.append("Bullish Kumo")

    elif row["span_a"] < row["span_b"]:
        bear.append("Bearish Kumo")

    # Chikou
    if i >= DISPLACEMENT + SENKOU_B:

        chikou = close_s.iloc[
            i - DISPLACEMENT
        ]

        hist_price = close_s.iloc[
            i - 2 * DISPLACEMENT
        ]

        hist_a = span_a_s.iloc[
            i - 2 * DISPLACEMENT
        ]

        hist_b = span_b_s.iloc[
            i - 2 * DISPLACEMENT
        ]

        if not any(
            pd.isna(v)
            for v in (
                chikou,
                hist_price,
                hist_a,
                hist_b,
            )
        ):

            top = max(
                hist_a,
                hist_b,
            )

            bottom = min(
                hist_a,
                hist_b,
            )

            if (
                chikou > hist_price
                and chikou > top
            ):
                bull.append(
                    "Chikou bullish"
                )

            elif (
                chikou < hist_price
                and chikou < bottom
            ):
                bear.append(
                    "Chikou bearish"
                )

    return bull, bear


def passes_extra_filters(sig, row):

    failed = []

    # Kumo thickness
    if (
        row["kumo_thickness_pct"]
        < KUMO_THICKNESS_MIN_PCT
    ):
        failed.append(
            "thin_kumo"
        )

    # Breakout margin
    margin = (
        (
            row["close"]
            - row["cloud_top"]
        )
        if sig == 1
        else (
            row["cloud_bottom"]
            - row["close"]
        )
    ) / row["close"] * 100

    if margin < BREAKOUT_MARGIN_PCT:
        failed.append(
            "weak_breakout"
        )

    # ADX
    if (
        pd.isna(row.get("adx"))
        or row["adx"] < ADX_MIN
    ):
        failed.append(
            "low_adx"
        )

    # EMA200
    if pd.isna(row.get("ema")):
        failed.append(
            "ema_warmup"
        )

    elif (
        sig == 1
        and row["close"] <= row["ema"]
    ):
        failed.append(
            "against_ema200"
        )

    elif (
        sig == -1
        and row["close"] >= row["ema"]
    ):
        failed.append(
            "against_ema200"
        )

    # RSI
    if pd.isna(row.get("rsi")):
        failed.append(
            "rsi_warmup"
        )

    elif (
        sig == 1
        and row["rsi"] >= 50
    ):
        failed.append(
            "rsi_not_bullish"
        )

    elif (
        sig == -1
        and row["rsi"] <= 50
    ):
        failed.append(
            "rsi_not_bearish"
        )

    # Bollinger width
    if (
        pd.isna(
            row.get("bb_width_pct")
        )
        or row["bb_width_pct"]
        < BB_WIDTH_MIN_PCT
    ):
        failed.append(
            "low_volatility"
        )

    # Volume + OBV
    if (
        pd.isna(
            row.get("volume_ma")
        )
        or row["volume_ma"] == 0
    ):
        failed.append(
            "volume_warmup"
        )

    else:

        vol_ok = (
            row["volume"]
            >= VOLUME_SPIKE_MULT
            * row["volume_ma"]
        )

        obv_ok = (
            row["obv_slope"] > 0
            if sig == 1
            else row["obv_slope"] < 0
        )

        if not (
            vol_ok
            and obv_ok
        ):
            failed.append(
                "no_volume_confirmation"
            )

    # Market structure
    if (
        sig == 1
        and not pd.isna(
            row.get(
                "last_swing_high"
            )
        )
        and row["close"]
        < row["last_swing_high"]
    ):

        if (
            (
                row["last_swing_high"]
                - row["close"]
            )
            / row["close"]
            * 100
            > STRUCTURE_MARGIN_PCT
        ):
            failed.append(
                "below_resistance"
            )

    if (
        sig == -1
        and not pd.isna(
            row.get(
                "last_swing_low"
            )
        )
        and row["close"]
        > row["last_swing_low"]
    ):

        if (
            (
                row["close"]
                - row["last_swing_low"]
            )
            / row["close"]
            * 100
            > STRUCTURE_MARGIN_PCT
        ):
            failed.append(
                "above_support"
            )

    return (
        len(failed) == 0,
        failed,
    )


# =========================
# ANALYZE TIMEFRAME
# =========================

def analyze_timeframe(
    symbol,
    tf,
    df,
):

    df = build_indicators(df)

    i = len(df) - 1
    row = df.iloc[i]

    if any(
        pd.isna(row.get(c))
        for c in [
            "tenkan",
            "kijun",
            "span_a",
            "span_b",
        ]
    ):

        log.info(
            "%s %s: SKIP - Ichimoku warming up",
            symbol,
            tf,
        )

        return {
            "signal": "NEUTRAL",
            "confirmations": [],
            "price": float(
                row["close"]
            ),
        }

    # Stage 1
    bull, bear = compute_confirmations(
        df,
        i,
    )

    bull_count = len(bull)
    bear_count = len(bear)

    raw_signal = "NEUTRAL"
    signal = "NEUTRAL"
    confirmations = []

    if (
        bull_count >= MIN_CONFIRMATIONS
        and bull_count > bear_count
    ):

        raw_signal = "BUY"
        signal = "BUY"
        confirmations = bull

    elif (
        bear_count >= MIN_CONFIRMATIONS
        and bear_count > bull_count
    ):

        raw_signal = "SELL"
        signal = "SELL"
        confirmations = bear

    if raw_signal == "NEUTRAL":

        log.info(
            "%s %s: NEUTRAL - "
            "bull %d/%d: %s | "
            "bear %d/%d: %s",
            symbol,
            tf,
            bull_count,
            MIN_CONFIRMATIONS,
            bull or "-",
            bear_count,
            MIN_CONFIRMATIONS,
            bear or "-",
        )

        return {
            "signal": "NEUTRAL",
            "raw_signal": "NEUTRAL",
            "confirmations": [],
            "filters_failed": [],
            "price": float(
                row["close"]
            ),
        }

    log.info(
        "%s %s: Ichimoku confirmed %s (%s) "
        "-> checking extra filters",
        symbol,
        tf,
        raw_signal,
        ", ".join(confirmations),
    )

    # Stage 2
    sig_num = (
        1
        if raw_signal == "BUY"
        else -1
    )

    passed, filters_failed = (
        passes_extra_filters(
            sig_num,
            row,
        )
    )

    if passed:

        log.info(
            "%s %s: PASSED all extra filters -> %s",
            symbol,
            tf,
            raw_signal,
        )

        signal = raw_signal

    else:

        reasons = ", ".join(
            FILTER_LABELS.get(
                f,
                f,
            )
            for f in filters_failed
        )

        log.info(
            "%s %s: BLOCKED - %s failed [%s]",
            symbol,
            tf,
            raw_signal,
            reasons,
        )

        signal = "NEUTRAL"
        confirmations = []

    return {
        "signal": signal,
        "raw_signal": raw_signal,
        "confirmations": confirmations,
        "filters_failed": filters_failed,
        "price": float(
            row["close"]
        ),
    }


# =========================
# BTC CORRELATION
# =========================

def get_btc_trend_4h():

    df = fetch_klines(
        BTC_SYMBOL,
        TIMEFRAMES[
            BTC_CORRELATION_TF
        ],
    )

    df = add_ichimoku(df)

    i = len(df) - 1
    row = df.iloc[i]

    if any(
        pd.isna(row.get(c))
        for c in [
            "tenkan",
            "kijun",
            "cloud_top",
            "cloud_bottom",
        ]
    ):
        return 0

    if (
        row["close"]
        > row["cloud_top"]
        and row["tenkan"]
        > row["kijun"]
    ):
        return 1

    if (
        row["close"]
        < row["cloud_bottom"]
        and row["tenkan"]
        < row["kijun"]
    ):
        return -1

    return 0


# =========================
# MULTI TIMEFRAME
# =========================

def analyze_symbol(symbol):

    results = {}

    with ThreadPoolExecutor(
        max_workers=len(TIMEFRAMES)
    ) as pool:

        futures = {
            pool.submit(
                fetch_klines,
                symbol,
                interval,
            ): label

            for label, interval
            in TIMEFRAMES.items()
        }

        for future in as_completed(
            futures
        ):

            label = futures[
                future
            ]

            df = future.result()

            results[label] = (
                analyze_timeframe(
                    symbol,
                    label,
                    df,
                )
            )

    buy_tfs = [
        tf
        for tf in TIMEFRAMES
        if results[tf]["signal"]
        == "BUY"
    ]

    sell_tfs = [
        tf
        for tf in TIMEFRAMES
        if results[tf]["signal"]
        == "SELL"
    ]

    if (
        len(buy_tfs)
        >= MIN_TF_CONFIRMATIONS
        and len(buy_tfs)
        > len(sell_tfs)
    ):

        final_signal = "BUY"

    elif (
        len(sell_tfs)
        >= MIN_TF_CONFIRMATIONS
        and len(sell_tfs)
        > len(buy_tfs)
    ):

        final_signal = "SELL"

    else:

        final_signal = "NEUTRAL"

    return (
        final_signal,
        results,
        buy_tfs,
        sell_tfs,
    )


# =========================
# TELEGRAM
# =========================

def send_telegram(text):

    url = TELEGRAM_URL.format(
        TELEGRAM_BOT_TOKEN
    )

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }

    r = SESSION.post(
        url,
        json=payload,
        timeout=15,
    )

    r.raise_for_status()

    result = r.json()

    if not result.get("ok"):
        raise RuntimeError(
            f"Telegram error: {result}"
        )


def fmt_price(price):

    if price >= 1000:
        return f"{price:,.2f}"

    if price >= 1:
        return f"{price:,.4f}"

    if price >= 0.01:
        return f"{price:,.6f}"

    return f"{price:.8f}"


FILTER_LABELS = {
    "thin_kumo": "thin cloud",
    "weak_breakout": "weak breakout",
    "low_adx": "no trend (ADX)",
    "against_ema200": "against EMA200",
    "ema_warmup": "EMA warmup",
    "rsi_not_bullish": "RSI above 50",
    "rsi_not_bearish": "RSI below 50",
    "rsi_warmup": "RSI warmup",
    "low_volatility": "low volatility",
    "no_volume_confirmation": "no volume confirm",
    "volume_warmup": "volume warmup",
    "below_resistance": "below resistance",
    "above_support": "above support",
}


def tf_line(tf, r):

    s = r["signal"]

    mark = (
        "🟢"
        if s == "BUY"
        else "🔴"
        if s == "SELL"
        else "⚪"
    )

    if (
        s == "NEUTRAL"
        and r.get("raw_signal")
        in ("BUY", "SELL")
        and r.get("filters_failed")
    ):

        reasons = ", ".join(
            FILTER_LABELS.get(
                f,
                f,
            )
            for f in r[
                "filters_failed"
            ][:2]
        )

        return (
            f"{mark} {tf}: "
            f"blocked ({reasons})"
        )

    return f"{mark} {tf}: {s}"


def build_message(
    symbol,
    final_signal,
    results,
    buy_tfs,
    sell_tfs,
    btc_gate_blocked,
):

    emoji = (
        "🟢"
        if final_signal == "BUY"
        else "🔴"
        if final_signal == "SELL"
        else "⚪"
    )

    agreeing = (
        buy_tfs
        if final_signal == "BUY"
        else sell_tfs
        if final_signal == "SELL"
        else []
    )

    latest_price = results[
        "15m"
    ]["price"]

    lines = [
        f"{emoji} {symbol} — {final_signal}",
        f"{fmt_price(latest_price)} USDT",
    ]

    if agreeing:

        lines.append(
            f"{len(agreeing)}/{len(TIMEFRAMES)} "
            f"timeframes agree: "
            f"{', '.join(agreeing)}"
        )

    if btc_gate_blocked:

        lines.append(
            "⚠️ blocked — BTC 4h trend disagrees"
        )

    lines.append("")

    for tf in TIMEFRAMES:

        lines.append(
            tf_line(
                tf,
                results[tf],
            )
        )

    strong_tfs = [
        tf
        for tf in TIMEFRAMES
        if results[tf]["signal"]
        in ("BUY", "SELL")
    ]

    if strong_tfs:

        lines.append("")

        for tf in strong_tfs:

            r = results[tf]

            lines.append(
                f"{tf} confirmations: "
                f"{' + '.join(r['confirmations'])}"
            )

    lines += [
        "",
        "⚠️ Technical signal only — not financial advice.",
    ]

    return "\n".join(lines)


# =========================
# SCANNER
# =========================

def scan_once():

    log.info(
        "Starting Bybit scan..."
    )

    # =========================
    # BTC TREND
    # =========================

    try:

        btc_trend = (
            get_btc_trend_4h()
        )

        log.info(
            "BTC 4h trend gate: %s",
            btc_trend,
        )

    except Exception as exc:

        log.warning(
            "Could not fetch BTC trend "
            "for correlation filter "
            "(%s); gate disabled this scan.",
            exc,
        )

        btc_trend = None

    # =========================
    # SYMBOLS
    # =========================

    for symbol in SYMBOLS:

        try:

            (
                final_signal,
                results,
                buy_tfs,
                sell_tfs,
            ) = analyze_symbol(
                symbol
            )

            if (
                final_signal
                == "NEUTRAL"
            ):

                log.info(
                    "%s: NEUTRAL overall - "
                    "BUY %d %s | SELL %d %s, "
                    "need >=%d agreeing",
                    symbol,
                    len(buy_tfs),
                    buy_tfs or "-",
                    len(sell_tfs),
                    sell_tfs or "-",
                    MIN_TF_CONFIRMATIONS,
                )

            else:

                log.info(
                    "%s: %d/%d timeframes "
                    "agree on %s (%s)",
                    symbol,
                    len(
                        buy_tfs
                        if final_signal
                        == "BUY"
                        else sell_tfs
                    ),
                    len(TIMEFRAMES),
                    final_signal,
                    ", ".join(
                        buy_tfs
                        if final_signal
                        == "BUY"
                        else sell_tfs
                    ),
                )

            # =========================
            # ANY TF SIGNAL?
            # =========================

            has_any_signal = any(
                results[tf]["signal"]
                in ("BUY", "SELL")
                for tf in TIMEFRAMES
            )

            if not has_any_signal:

                log.info(
                    "%s: SKIP Telegram - "
                    "every timeframe NEUTRAL",
                    symbol,
                )

                continue

            # =========================
            # BTC GLOBAL GATE
            # =========================

            btc_gate_blocked = False

            if (
                symbol != BTC_SYMBOL
                and btc_trend is not None
                and final_signal
                != "NEUTRAL"
            ):

                sig_num = (
                    1
                    if final_signal
                    == "BUY"
                    else -1
                )

                if (
                    sig_num == 1
                    and btc_trend < 0
                ) or (
                    sig_num == -1
                    and btc_trend > 0
                ):

                    btc_gate_blocked = True

                    log.info(
                        "%s: BLOCKED - "
                        "%s disagrees with "
                        "BTC 4h trend (%s)",
                        symbol,
                        final_signal,
                        btc_trend,
                    )

                    final_signal = "NEUTRAL"

            # =========================
            # TELEGRAM
            # =========================

            message = build_message(
                symbol,
                final_signal,
                results,
                buy_tfs,
                sell_tfs,
                btc_gate_blocked,
            )

            send_telegram(message)

            log.info(
                "%s: Telegram sent - "
                "final signal %s",
                symbol,
                final_signal,
            )

        except Exception as exc:

            log.exception(
                "Error scanning %s: %s",
                symbol,
                exc,
            )

    log.info(
        "Scan completed at %s",
        datetime.now(
            timezone.utc
        ).isoformat(),
    )


# =========================
# MAIN
# =========================

def main():

    log.info(
        "Ichimoku Telegram scanner started "
        "(Bybit V5 / linear USDT)"
    )

    log.info(
        "Symbols: %s",
        ", ".join(SYMBOLS),
    )

    log.info(
        "Timeframes: %s",
        ", ".join(TIMEFRAMES),
    )

    log.info(
        "Confirmations per timeframe: "
        "ALL 4 Ichimoku checks required first. "
        "Only if Ichimoku fully confirms is "
        "the signal then checked against: "
        "RSI vs 50 midline "
        "(BUY needs RSI<50, SELL needs RSI>50), "
        "kumo thickness, breakout margin, "
        "ADX>=%d, EMA%d trend, "
        "Bollinger width>=%.1f%%, "
        "volume/OBV confirmation, "
        "market structure. "
        "Plus a global BTC-4h-trend "
        "correlation gate for altcoins. "
        "A symbol is skipped when every "
        "timeframe is NEUTRAL.",
        ADX_MIN,
        EMA_PERIOD,
        BB_WIDTH_MIN_PCT,
    )

    scan_once()


if __name__ == "__main__":
    main()