import os
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# Reversal scanner: detects EARLY trend changes
#   BUY  = downtrend exhausted -> possible bottom
#   SELL = uptrend exhausted   -> possible top
#
# A signal needs ALL of:
#   1) Extreme zone      : Bollinger band touch OR RSI oversold/overbought
#   2) Reversal trigger  : RSI divergence OR market-structure break (CHoCH)
#   3) Reversal candle   : hammer / engulfing WITH a volume spike
# Bonus (shown in the message, not required): Tenkan/Kijun cross.
# =========================================================

# =========================
# CONFIG
# =========================
KUCOIN_URL = "https://api.kucoin.com/api/ua/v2/market/kline"
TELEGRAM_URL = "https://api.telegram.org/bot{}/sendMessage"

SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "BNB-USDT", "XRP-USDT", "SOL-USDT", "TRX-USDT",
    "ZEC-USDT", "HYPE-USDT", "DOGE-USDT", "LINK-USDT", "XMR-USDT", "ADA-USDT",
    "XLM-USDT", "BCH-USDT", "NEAR-USDT", "UNI-USDT", "LTC-USDT", "CC-USDT",
    "AVAX-USDT", "SUI-USDT", "HBAR-USDT", "TAO-USDT", "SHIB-USDT", "CRO-USDT",
    "ENA-USDT", "ONDO-USDT", "AAVE-USDT", "MNT-USDT", "DOT-USDT", "PUMP-USDT",
    "ASTER-USDT", "WLD-USDT", "WLFI-USDT", "SKY-USDT", "PEPE-USDT", "ICP-USDT",
    "POL-USDT", "DASH-USDT", "WIF-USDT", "TRUMP-USDT", "VIRTUAL-USDT",
    "PENGU-USDT", "INJ-USDT", "KAS-USDT", "FIL-USDT", "ATOM-USDT", "APE-USDT",
    "TIA-USDT", "BONK-USDT", "ZRO-USDT", "OP-USDT", "SEI-USDT", "ARB-USDT",
    "FARTCOIN-USDT", "W-USDT", "AKT-USDT", "STORJ-USDT", "APT-USDT", "ETC-USDT",
    "ALGO-USDT", "VET-USDT", "STX-USDT", "RENDER-USDT", "LDO-USDT", "FET-USDT",
    "PENDLE-USDT", "CRV-USDT",
]

# Reversals are more reliable on higher timeframes. Add "15m": "15min" if you want.
TIMEFRAMES = {
    "1h": "1hour",
    "4h": "4hour",
    "1D": "1day",
}
MIN_TF_CONFIRMATIONS = 1

# --- 1) Extreme zone ---
RSI_PERIOD = 14
RSI_OS = 30            # oversold  (BUY setups)
RSI_OB = 70            # overbought (SELL setups)
BB_PERIOD = 20
BB_STD = 2
EXTREME_LOOKBACK = 12  # the extreme must have happened within the last N candles

# --- 2a) RSI divergence ---
PIVOT_WINDOW = 3        # candles on each side to confirm a swing pivot
DIV_MIN_GAP = 5         # min candles between the two pivots
DIV_MAX_GAP = 60        # max candles between the two pivots
DIV_MAX_AGE = 15        # newest pivot must be at most N candles old
RSI_DIV_MIN_DIFF = 1.0  # RSI must differ by at least this between the pivots
RSI_DIV_ZONE_BULL = 40  # first pivot's RSI must be <= this (bullish divergence)
RSI_DIV_ZONE_BEAR = 60  # first pivot's RSI must be >= this (bearish divergence)

# --- 2b) Structure break (CHoCH) ---
CHOCH_LOOKBACK = 2      # the break must happen within the last N closed candles

# --- 3) Reversal candle + volume ---
VOLUME_MA_PERIOD = 20
REVERSAL_VOL_MULT = 1.5
WICK_RATIO = 1.5        # hammer: long wick >= WICK_RATIO * body

# --- Bonus: Tenkan/Kijun cross ---
TENKAN, KIJUN = 9, 26
TK_LOOKBACK = 3

# --- BTC context (warning by default; set BTC_GATE_BLOCKS=1 to block alt signals) ---
BTC_SYMBOL = "BTC-USDT"
BTC_CONTEXT_TF = "4hour"
BTC_EMA_PERIOD = 50
BTC_GATE_BLOCKS = os.getenv("BTC_GATE_BLOCKS", "0") == "1"

CANDLE_LIMIT = 300
CANDLE_FETCH_BUFFER = 60
MIN_CANDLES = 100

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
MAX_WORKERS = 6
STATE_FILE = os.getenv("STATE_FILE", "signal_state.json")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("reversal-bot")


# =========================
# ENV
# =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise SystemExit("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env")


# =========================
# HTTP SESSION
# =========================
def build_session():
    session = requests.Session()
    retries = Retry(
        total=3, backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retries, pool_maxsize=MAX_WORKERS + 2)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


SESSION = build_session()


# =========================
# KUCOIN
# =========================
def interval_to_seconds(interval):
    return {"15min": 15 * 60, "1hour": 3600, "4hour": 4 * 3600, "1day": 86400}[interval]


def fetch_klines(symbol, interval):
    seconds = interval_to_seconds(interval)
    now = int(time.time())
    start_at = now - (CANDLE_LIMIT + CANDLE_FETCH_BUFFER) * seconds

    params = {
        "symbol": symbol, "tradeType": "SPOT", "klineType": "TRADE",
        "interval": interval, "startAt": start_at, "endAt": now,
    }
    r = SESSION.get(KUCOIN_URL, params=params, timeout=15)
    r.raise_for_status()
    payload = r.json()

    if payload.get("code") != "200000":
        raise RuntimeError(f"KuCoin error: {payload}")

    rows = payload.get("data", {}).get("list", [])
    if not rows:
        raise RuntimeError(f"No kline data for {symbol} {interval}")

    rows.sort(key=lambda x: int(x[0]))
    df = pd.DataFrame(
        rows, columns=["timestamp", "open", "close", "high", "low", "volume", "turnover"]
    )
    for col in ["open", "close", "high", "low", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"]), unit="s", utc=True)
    df = df.dropna().drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # keep CLOSED candles only
    now_ts = pd.Timestamp.now(tz="UTC")
    df = df[(df["timestamp"] + pd.Timedelta(seconds=seconds)) <= now_ts].copy()

    if len(df) < MIN_CANDLES:
        raise RuntimeError(f"Not enough CLOSED candles for {symbol} {interval}: {len(df)}")

    return df.tail(CANDLE_LIMIT).reset_index(drop=True)


# =========================
# INDICATORS
# =========================
def add_rsi(df, period=RSI_PERIOD):
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(100.0).where(avg_loss != 0, 100.0)
    return df


def add_bollinger(df, period=BB_PERIOD, n_std=BB_STD):
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    df["bb_upper"] = sma + n_std * std
    df["bb_lower"] = sma - n_std * std
    return df


def add_volume_ma(df, period=VOLUME_MA_PERIOD):
    df["volume_ma"] = df["volume"].rolling(period).mean()
    return df


def add_tk(df):
    high, low = df["high"], df["low"]
    df["tenkan"] = (high.rolling(TENKAN).max() + low.rolling(TENKAN).min()) / 2
    df["kijun"] = (high.rolling(KIJUN).max() + low.rolling(KIJUN).min()) / 2
    return df


def add_swing_structure(df, window=PIVOT_WINDOW):
    """Last CONFIRMED swing high/low (known only `window` candles after the pivot)."""
    is_high = df["high"] == df["high"].rolling(2 * window + 1, center=True).max()
    is_low = df["low"] == df["low"].rolling(2 * window + 1, center=True).min()
    confirmed_high = is_high.shift(window, fill_value=False)
    confirmed_low = is_low.shift(window, fill_value=False)
    df["last_swing_high"] = df["high"].shift(window).where(confirmed_high).ffill()
    df["last_swing_low"] = df["low"].shift(window).where(confirmed_low).ffill()
    return df


def build_indicators(df):
    df = add_rsi(df)
    df = add_bollinger(df)
    df = add_volume_ma(df)
    df = add_tk(df)
    df = add_swing_structure(df)
    return df


# =========================
# REVERSAL BUILDING BLOCKS
# direction: +1 = looking for a bottom (BUY), -1 = looking for a top (SELL)
# =========================
def extreme_zone(df, i, direction):
    """Returns a list of reasons if price recently hit an extreme, else []."""
    sl = slice(i - EXTREME_LOOKBACK + 1, i + 1)
    reasons = []
    if direction == 1:
        if (df["low"].iloc[sl] <= df["bb_lower"].iloc[sl]).any():
            reasons.append("Bollinger lower band touch")
        rsi_min = df["rsi"].iloc[sl].min()
        if rsi_min <= RSI_OS:
            reasons.append(f"RSI oversold ({rsi_min:.0f})")
    else:
        if (df["high"].iloc[sl] >= df["bb_upper"].iloc[sl]).any():
            reasons.append("Bollinger upper band touch")
        rsi_max = df["rsi"].iloc[sl].max()
        if rsi_max >= RSI_OB:
            reasons.append(f"RSI overbought ({rsi_max:.0f})")
    return reasons


def find_pivots(values, window, kind, last_index):
    """Confirmed pivot indices: values[j] is the min ('low') / max ('high') of [j-w, j+w]."""
    out = []
    for j in range(window, last_index - window + 1):
        seg = values[j - window: j + window + 1]
        if np.isnan(seg).any():
            continue
        v = values[j]
        if (kind == "low" and v <= seg.min()) or (kind == "high" and v >= seg.max()):
            out.append(j)
    return out


def detect_divergence(df, i, direction):
    """
    Bullish: price makes a LOWER low while RSI makes a HIGHER low.
    Bearish: price makes a HIGHER high while RSI makes a LOWER high.
    """
    rsi = df["rsi"].to_numpy()
    if direction == 1:
        price = df["low"].to_numpy()
        piv = find_pivots(price, PIVOT_WINDOW, "low", i)
    else:
        price = df["high"].to_numpy()
        piv = find_pivots(price, PIVOT_WINDOW, "high", i)

    if len(piv) < 2:
        return False
    p2 = piv[-1]
    if i - p2 > DIV_MAX_AGE:
        return False

    for p1 in reversed(piv[:-1]):
        gap = p2 - p1
        if gap < DIV_MIN_GAP:
            continue
        if gap > DIV_MAX_GAP:
            return False
        if direction == 1:
            return bool(price[p2] < price[p1]
                        and rsi[p2] > rsi[p1] + RSI_DIV_MIN_DIFF
                        and rsi[p1] <= RSI_DIV_ZONE_BULL)
        return bool(price[p2] > price[p1]
                    and rsi[p2] < rsi[p1] - RSI_DIV_MIN_DIFF
                    and rsi[p1] >= RSI_DIV_ZONE_BEAR)
    return False


def detect_choch(df, i, direction):
    """
    Change of character: price closes back through the last confirmed swing
    high (bottom) / swing low (top) that had been holding the old trend.
    The break must be fresh (within CHOCH_LOOKBACK closed candles).
    """
    close = df["close"].to_numpy()
    lvl = df["last_swing_high" if direction == 1 else "last_swing_low"].to_numpy()
    for k in range(i, i - CHOCH_LOOKBACK, -1):
        if k < 1 or np.isnan(lvl[k]) or np.isnan(lvl[k - 1]):
            continue
        if direction == 1 and close[k] > lvl[k] and close[k - 1] <= lvl[k - 1]:
            return True
        if direction == -1 and close[k] < lvl[k] and close[k - 1] >= lvl[k - 1]:
            return True
    return False


def candle_pattern(df, k, direction):
    o, c = df["open"].iloc[k], df["close"].iloc[k]
    h, l = df["high"].iloc[k], df["low"].iloc[k]
    rng = h - l
    if rng <= 0:
        return None
    body = max(abs(c - o), rng * 0.02)
    upper = h - max(o, c)
    lower = min(o, c) - l
    po, pc = df["open"].iloc[k - 1], df["close"].iloc[k - 1]

    if direction == 1:
        if lower >= WICK_RATIO * body and c >= l + 0.5 * rng:
            return "hammer"
        if c > o and pc < po and c >= po and o <= pc:
            return "bullish engulfing"
    else:
        if upper >= WICK_RATIO * body and c <= h - 0.5 * rng:
            return "shooting star"
        if c < o and pc > po and c <= po and o >= pc:
            return "bearish engulfing"
    return None


def reversal_confirmation(df, i, direction):
    """Returns (pattern_name_or_None, volume_ok). Checks the last 2 closed candles."""
    seen = None
    for k in (i, i - 1):
        name = candle_pattern(df, k, direction)
        if not name:
            continue
        seen = seen or name
        vma = df["volume_ma"].iloc[k]
        if pd.notna(vma) and vma > 0 and df["volume"].iloc[k] >= REVERSAL_VOL_MULT * vma:
            return name, True
    return seen, False


def tk_cross(df, i, direction):
    t, kj = df["tenkan"].to_numpy(), df["kijun"].to_numpy()
    for k in range(i, i - TK_LOOKBACK, -1):
        if k < 1 or np.isnan(t[k]) or np.isnan(kj[k]) or np.isnan(t[k - 1]) or np.isnan(kj[k - 1]):
            continue
        if direction == 1 and t[k] > kj[k] and t[k - 1] <= kj[k - 1]:
            return True
        if direction == -1 and t[k] < kj[k] and t[k - 1] >= kj[k - 1]:
            return True
    return False


# =========================
# PER-TIMEFRAME ANALYSIS
# =========================
def analyze_timeframe(symbol, tf, df):
    df = build_indicators(df)
    i = len(df) - 1
    row = df.iloc[i]
    price = float(row["close"])

    needed = ["rsi", "bb_lower", "bb_upper", "volume_ma", "tenkan", "kijun"]
    if any(pd.isna(row[c]) for c in needed):
        log.info("%s %s: SKIP - indicators still warming up", symbol, tf)
        return {"signal": "NEUTRAL", "watch": None, "reasons": [], "price": price}

    watch = None
    for direction, name in ((1, "BUY"), (-1, "SELL")):
        extreme = extreme_zone(df, i, direction)
        if not extreme:
            continue

        div = detect_divergence(df, i, direction)
        choch = detect_choch(df, i, direction)
        if not (div or choch):
            log.info("%s %s: %s watch - extreme (%s) but no divergence / structure break yet",
                     symbol, tf, name, ", ".join(extreme))
            watch = watch or {"side": name, "stage": "extreme, waiting for divergence/CHoCH"}
            continue

        pattern, vol_ok = reversal_confirmation(df, i, direction)
        if not (pattern and vol_ok):
            why = "no reversal candle" if not pattern else f"{pattern} without volume spike"
            log.info("%s %s: %s watch - trigger found but %s", symbol, tf, name, why)
            watch = {"side": name, "stage": "trigger found, waiting for candle+volume"}
            continue

        reasons = list(extreme)
        if div:
            reasons.append("RSI divergence")
        if choch:
            reasons.append("structure break (CHoCH)")
        reasons.append(f"{pattern} + volume spike")
        if tk_cross(df, i, direction):
            reasons.append("Tenkan/Kijun cross (bonus)")

        span = EXTREME_LOOKBACK + PIVOT_WINDOW
        recent = df.iloc[max(0, i - span + 1): i + 1]
        invalidation = float(recent["low"].min() if direction == 1 else recent["high"].max())

        log.info("%s %s: %s REVERSAL SIGNAL (%s)", symbol, tf, name, " + ".join(reasons))
        return {"signal": name, "watch": None, "reasons": reasons,
                "price": price, "invalidation": invalidation}

    if not watch:
        log.info("%s %s: NEUTRAL - no extreme zone", symbol, tf)
    return {"signal": "NEUTRAL", "watch": watch, "reasons": [], "price": price}


# =========================
# BTC CONTEXT
# =========================
def get_btc_context():
    """+1 if BTC 4h closes above its EMA, -1 if below."""
    df = fetch_klines(BTC_SYMBOL, BTC_CONTEXT_TF)
    ema = df["close"].ewm(span=BTC_EMA_PERIOD, adjust=False).mean()
    return 1 if df["close"].iloc[-1] > ema.iloc[-1] else -1


# =========================
# MULTI-TIMEFRAME
# =========================
def analyze_symbol(symbol):
    results = {}
    with ThreadPoolExecutor(max_workers=len(TIMEFRAMES)) as pool:
        futures = {
            pool.submit(fetch_klines, symbol, interval): label
            for label, interval in TIMEFRAMES.items()
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                df = future.result()
                results[label] = analyze_timeframe(symbol, label, df)
            except Exception as exc:
                log.warning("%s %s: skipped (%s)", symbol, label, exc)
                results[label] = {"signal": "NEUTRAL", "watch": None, "reasons": [], "price": None}

    buy_tfs = [tf for tf in TIMEFRAMES if results[tf]["signal"] == "BUY"]
    sell_tfs = [tf for tf in TIMEFRAMES if results[tf]["signal"] == "SELL"]

    if len(buy_tfs) >= MIN_TF_CONFIRMATIONS and len(buy_tfs) > len(sell_tfs):
        final_signal = "BUY"
    elif len(sell_tfs) >= MIN_TF_CONFIRMATIONS and len(sell_tfs) > len(buy_tfs):
        final_signal = "SELL"
    else:
        final_signal = "NEUTRAL"

    return final_signal, results, buy_tfs, sell_tfs


# =========================
# TELEGRAM
# =========================
def send_telegram(text):
    url = TELEGRAM_URL.format(TELEGRAM_BOT_TOKEN)
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "disable_web_page_preview": True}
    r = SESSION.post(url, json=payload, timeout=15)
    r.raise_for_status()
    result = r.json()
    if not result.get("ok"):
        raise RuntimeError(f"Telegram error: {result}")


def fmt_price(price):
    if price >= 1000:
        return f"{price:,.2f}"
    if price >= 1:
        return f"{price:,.4f}"
    if price >= 0.01:
        return f"{price:,.6f}"
    return f"{price:.8f}"


def tf_line(tf, r):
    s = r["signal"]
    if s in ("BUY", "SELL"):
        return f"{'🟢' if s == 'BUY' else '🔴'} {tf}: {s}"
    if r.get("watch"):
        return f"👀 {tf}: watch {r['watch']['side']} ({r['watch']['stage']})"
    return f"⚪ {tf}: no setup"


def build_message(symbol, final_signal, results, buy_tfs, sell_tfs, btc_trend):
    is_buy = final_signal == "BUY"
    emoji = "🟢" if is_buy else "🔴"
    agreeing = buy_tfs if is_buy else sell_tfs
    label = "possible BOTTOM (reversal up)" if is_buy else "possible TOP (reversal down)"

    price = next((results[tf]["price"] for tf in TIMEFRAMES if results[tf]["price"]), None)
    lines = [f"{emoji} {symbol} — {final_signal}: {label}"]
    if price:
        lines.append(f"{fmt_price(price)} USDT")
    lines.append(f"{len(agreeing)}/{len(TIMEFRAMES)} timeframes: {', '.join(agreeing)}")

    if symbol != BTC_SYMBOL and btc_trend is not None:
        against = (is_buy and btc_trend < 0) or (not is_buy and btc_trend > 0)
        state = "above EMA%d" % BTC_EMA_PERIOD if btc_trend > 0 else "below EMA%d" % BTC_EMA_PERIOD
        lines.append(f"{'⚠️ ' if against else ''}BTC 4h {state}" + (" — against this signal" if against else ""))

    lines.append("")
    for tf in TIMEFRAMES:
        lines.append(tf_line(tf, results[tf]))

    for tf in agreeing:
        r = results[tf]
        lines += ["", f"{tf}: {' + '.join(r['reasons'])}",
                  f"{tf} invalidation: {fmt_price(r['invalidation'])}"]

    lines += ["", "⚠️ Early reversal signal — more false signals than trend-following. Not financial advice."]
    return "\n".join(lines)


# =========================
# SIGNAL STATE (dedupe while the same signal persists)
# =========================
def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        log.warning("Could not read %s (%s); starting with empty state.", STATE_FILE, exc)
        return {}


def save_state(state):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=1, sort_keys=True)
    except Exception as exc:
        log.warning("Could not write %s (%s)", STATE_FILE, exc)


def signal_key(final_signal, buy_tfs, sell_tfs):
    tfs = buy_tfs if final_signal == "BUY" else sell_tfs
    return f"{final_signal}|{','.join(tfs)}"


# =========================
# SCANNER
# =========================
def scan_once():
    log.info("Starting scan...")

    try:
        btc_trend = get_btc_context()
        log.info("BTC 4h context: %s", "bullish" if btc_trend > 0 else "bearish")
    except Exception as exc:
        log.warning("Could not fetch BTC context (%s).", exc)
        btc_trend = None

    state = load_state()
    new_state = dict(state)

    for symbol in SYMBOLS:
        try:
            final_signal, results, buy_tfs, sell_tfs = analyze_symbol(symbol)

            if final_signal == "NEUTRAL":
                log.info("%s: no reversal signal", symbol)
                new_state.pop(symbol, None)  # signal gone -> next one counts as new
                continue

            if symbol != BTC_SYMBOL and btc_trend is not None and BTC_GATE_BLOCKS:
                if (final_signal == "BUY" and btc_trend < 0) or (final_signal == "SELL" and btc_trend > 0):
                    log.info("%s: BLOCKED - %s against BTC 4h context", symbol, final_signal)
                    continue

            key = signal_key(final_signal, buy_tfs, sell_tfs)
            if state.get(symbol) == key:
                log.info("%s: SKIP Telegram - same signal already sent (%s)", symbol, key)
                continue

            message = build_message(symbol, final_signal, results, buy_tfs, sell_tfs, btc_trend)
            send_telegram(message)
            new_state[symbol] = key
            log.info("%s: Telegram sent - %s", symbol, final_signal)

        except Exception as exc:
            log.exception("Error scanning %s: %s", symbol, exc)

    save_state(new_state)
    log.info("Scan completed at %s", datetime.now(timezone.utc).isoformat())


def main():
    log.info("Reversal scanner started.")
    log.info("Symbols: %d | Timeframes: %s", len(SYMBOLS), ", ".join(TIMEFRAMES))
    log.info(
        "Signal = extreme zone (BB touch / RSI %d-%d) + trigger (RSI divergence or CHoCH) "
        "+ reversal candle with volume >= %.1fx average. BTC gate blocks: %s",
        RSI_OS, RSI_OB, REVERSAL_VOL_MULT, BTC_GATE_BLOCKS,
    )
    scan_once()


if __name__ == "__main__":
    main()
