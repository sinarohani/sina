
import os
import time
import json
import logging
from datetime import datetime, timezone

import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# =========================
# CONFIG
# =========================
KUCOIN_URL = "https://api.kucoin.com/api/ua/v2/market/kline"
TELEGRAM_URL = "https://api.telegram.org/bot{}/sendMessage"

SYMBOLS = [
    "BTC-USDT",
    "ETH-USDT",
    "SOL-USDT",
    "DOGE-USDT",
    "POL-USDT",
    "LINK-USDT",
]

# Timeframes requested for the scanner.
TIMEFRAMES = {
    "15m": "15min",
    "1h": "1hour",
    "4h": "4hour",
    "1D": "1day",
}

# Standard Ichimoku parameters.
TENKAN = 9
KIJUN = 26
SENKOU_B = 52
DISPLACEMENT = 26

# At least 2 Ichimoku confirmations on a timeframe.
MIN_CONFIRMATIONS = 2

# At least 2 timeframes must agree.
MIN_TF_CONFIRMATIONS = 2

# Polling interval. 300 seconds = 5 minutes.
SCAN_EVERY_SECONDS = 300

# How many candles to request. 150 is enough for the standard Ichimoku.
CANDLE_LIMIT = 150

# Send only when the final signal changes, unless SEND_NEUTRAL is enabled.
SEND_NEUTRAL = False

STATE_FILE = "state.json"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("ichimoku-bot")


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
# STATE
# =========================
def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, STATE_FILE)


state = load_state()


# =========================
# KUCOIN
# =========================
def fetch_klines(symbol, interval):
    """
    KuCoin UTA V2 public Kline endpoint.
    The current official response returns:
      data: { tradeType, symbol, list: [[time, open, close, high, low, volume, turnover], ...] }
    """
    params = {
        "symbol": symbol,
        "tradeType": "SPOT",
        "klineType": "TRADE",
        "interval": interval,
    }

    r = requests.get(KUCOIN_URL, params=params, timeout=15)
    r.raise_for_status()
    payload = r.json()

    if payload.get("code") != "200000":
        raise RuntimeError(f"KuCoin error: {payload}")

    data = payload.get("data", {})
    rows = data.get("list", [])

    if not rows:
        raise RuntimeError(f"No kline data for {symbol} {interval}")

    # KuCoin can return newest first; normalize to oldest -> newest.
    rows = sorted(rows, key=lambda x: int(x[0]))

    df = pd.DataFrame(
        rows,
        columns=["timestamp", "open", "close", "high", "low", "volume", "turnover"],
    )

    for col in ["open", "close", "high", "low", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_datetime(
        pd.to_numeric(df["timestamp"], errors="coerce"),
        unit="s",
        utc=True,
    )

    df = df.dropna(subset=["timestamp", "open", "close", "high", "low"]).copy()
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Remove the currently forming candle.
    # A candle is considered closed when its timestamp + timeframe duration <= now.
    seconds = interval_to_seconds(interval)
    now = pd.Timestamp.now(tz="UTC")
    df = df[(df["timestamp"] + pd.Timedelta(seconds=seconds)) <= now].copy()

    if len(df) < SENKOU_B + DISPLACEMENT + 5:
        raise RuntimeError(
            f"Not enough CLOSED candles for {symbol} {interval}: {len(df)}"
        )

    return df.tail(CANDLE_LIMIT).reset_index(drop=True)


def interval_to_seconds(interval):
    mapping = {
        "15min": 15 * 60,
        "1hour": 60 * 60,
        "4hour": 4 * 60 * 60,
        "1day": 24 * 60 * 60,
    }
    return mapping[interval]


# =========================
# ICHIMOKU
# =========================
def ichimoku(df):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    tenkan = (
        high.rolling(TENKAN).max() + low.rolling(TENKAN).min()
    ) / 2

    kijun = (
        high.rolling(KIJUN).max() + low.rolling(KIJUN).min()
    ) / 2

    senkou_a_raw = (tenkan + kijun) / 2

    senkou_b_raw = (
        high.rolling(SENKOU_B).max() + low.rolling(SENKOU_B).min()
    ) / 2

    return {
        "close": close,
        "high": high,
        "low": low,
        "tenkan": tenkan,
        "kijun": kijun,
        "span_a": senkou_a_raw,
        "span_b": senkou_b_raw,
    }


def analyze_timeframe(df):
    x = ichimoku(df)
    i = len(df) - 1

    # Current cloud uses values whose standard chart display is shifted forward.
    # For a signal at the current closed candle, compare current price with the
    # current computed Kumo values.
    tenkan = x["tenkan"].iloc[i]
    kijun = x["kijun"].iloc[i]
    span_a = x["span_a"].iloc[i]
    span_b = x["span_b"].iloc[i]
    close = x["close"].iloc[i]

    if any(pd.isna(v) for v in [tenkan, kijun, span_a, span_b, close]):
        return {"signal": "NEUTRAL", "confirmations": [], "price": float(close)}

    cloud_top = max(span_a, span_b)
    cloud_bottom = min(span_a, span_b)

    confirmations_bull = []
    confirmations_bear = []

    # 1) Price vs Kumo
    if close > cloud_top:
        confirmations_bull.append("Price > Kumo")
    elif close < cloud_bottom:
        confirmations_bear.append("Price < Kumo")

    # 2) Tenkan vs Kijun
    if tenkan > kijun:
        confirmations_bull.append("Tenkan > Kijun")
    elif tenkan < kijun:
        confirmations_bear.append("Tenkan < Kijun")

    # 3) Kumo direction
    if span_a > span_b:
        confirmations_bull.append("Bullish Kumo")
    elif span_a < span_b:
        confirmations_bear.append("Bearish Kumo")

    # 4) Chikou confirmation.
    # Chikou at the current point is today's close plotted 26 candles back.
    # We compare that historical close against the historical price/cloud area.
    if i >= DISPLACEMENT + SENKOU_B:
        chikou_value = x["close"].iloc[i - DISPLACEMENT]
        hist_price = x["close"].iloc[i - 2 * DISPLACEMENT]
        hist_a = x["span_a"].iloc[i - 2 * DISPLACEMENT]
        hist_b = x["span_b"].iloc[i - 2 * DISPLACEMENT]

        hist_cloud_top = max(hist_a, hist_b)
        hist_cloud_bottom = min(hist_a, hist_b)

        if (
            not pd.isna(chikou_value)
            and not pd.isna(hist_price)
            and not pd.isna(hist_cloud_top)
        ):
            if chikou_value > hist_price and chikou_value > hist_cloud_top:
                confirmations_bull.append("Chikou bullish")
            elif chikou_value < hist_price and chikou_value < hist_cloud_bottom:
                confirmations_bear.append("Chikou bearish")

    bull_count = len(confirmations_bull)
    bear_count = len(confirmations_bear)

    if bull_count >= MIN_CONFIRMATIONS and bull_count > bear_count:
        signal = "BUY"
        confirmations = confirmations_bull
    elif bear_count >= MIN_CONFIRMATIONS and bear_count > bull_count:
        signal = "SELL"
        confirmations = confirmations_bear
    else:
        signal = "NEUTRAL"
        confirmations = []

    return {
        "signal": signal,
        "confirmations": confirmations,
        "bull_count": bull_count,
        "bear_count": bear_count,
        "price": float(close),
        "tenkan": float(tenkan),
        "kijun": float(kijun),
        "span_a": float(span_a),
        "span_b": float(span_b),
        "candle_time": df["timestamp"].iloc[i].isoformat(),
    }


# =========================
# MULTI-TIMEFRAME
# =========================
def analyze_symbol(symbol):
    results = {}

    for label, kucoin_interval in TIMEFRAMES.items():
        df = fetch_klines(symbol, kucoin_interval)
        results[label] = analyze_timeframe(df)

    buy_tfs = [tf for tf, r in results.items() if r["signal"] == "BUY"]
    sell_tfs = [tf for tf, r in results.items() if r["signal"] == "SELL"]

    # Final signal requires agreement from at least 2 timeframes.
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
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "disable_web_page_preview": True,
    }

    r = requests.post(url, json=payload, timeout=15)
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


def build_message(symbol, final_signal, results, buy_tfs, sell_tfs):
    if final_signal == "BUY":
        title = "🟢 ICHIMOKU BUY SIGNAL"
        emoji = "🟢"
        agreeing = buy_tfs
    elif final_signal == "SELL":
        title = "🔴 ICHIMOKU SELL SIGNAL"
        emoji = "🔴"
        agreeing = sell_tfs
    else:
        title = "⚪ ICHIMOKU NEUTRAL"
        emoji = "⚪"
        agreeing = []

    lines = [
        title,
        "━━━━━━━━━━━━━━━━━━",
        f"Symbol: {symbol}",
        f"Final: {final_signal}",
        "",
        "Multi-Timeframe:",
    ]

    for tf in TIMEFRAMES:
        r = results[tf]
        s = r["signal"]
        mark = "🟢" if s == "BUY" else "🔴" if s == "SELL" else "⚪"
        lines.append(f"{mark} {tf}: {s}")

    if agreeing:
        lines += [
            "",
            f"TF Confirmation: {len(agreeing)}/{len(TIMEFRAMES)}",
            f"Agreeing TFs: {', '.join(agreeing)}",
        ]

    # Show details for each timeframe, keeping the message compact.
    lines += ["", "Ichimoku confirmations:"]

    for tf in TIMEFRAMES:
        r = results[tf]
        if r["signal"] in ("BUY", "SELL"):
            checks = " + ".join(r["confirmations"])
            lines.append(f"{tf}: {checks}")

    # Use the most recent 15m close as a simple displayed current reference.
    latest_price = results["15m"]["price"]

    lines += [
        "",
        f"Reference Price: {fmt_price(latest_price)} USDT",
        f"Closed candle: {results['15m']['candle_time']}",
        "",
        "⚠️ Technical signal only — not financial advice.",
    ]

    return "\n".join(lines)


# =========================
# SCANNER
# =========================
def scan_once():
    log.info("Starting scan...")

    for symbol in SYMBOLS:
        try:
            final_signal, results, buy_tfs, sell_tfs = analyze_symbol(symbol)

            previous = state.get(symbol, {}).get("final_signal")

            log.info(
                "%s -> %s | BUY TFs=%s | SELL TFs=%s",
                symbol,
                final_signal,
                buy_tfs,
                sell_tfs,
            )

            should_send = False

            if final_signal in ("BUY", "SELL"):
                # Notify only when a new actionable signal appears or changes.
                if previous != final_signal:
                    should_send = True
            elif SEND_NEUTRAL and previous != "NEUTRAL":
                should_send = True

            if should_send:
                message = build_message(
                    symbol, final_signal, results, buy_tfs, sell_tfs
                )
                send_telegram(message)
                log.info("Telegram sent for %s: %s", symbol, final_signal)

            state[symbol] = {
                "final_signal": final_signal,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "buy_tfs": buy_tfs,
                "sell_tfs": sell_tfs,
            }
            save_state(state)

        except Exception as exc:
            log.exception("Error scanning %s: %s", symbol, exc)

    log.info("Scan completed.")


def main():
    log.info("Ichimoku Telegram scanner started.")
    log.info("Symbols: %s", ", ".join(SYMBOLS))
    log.info("Timeframes: %s", ", ".join(TIMEFRAMES))
    log.info(
        "Rules: >=%d Ichimoku confirmations per TF + >=%d agreeing TFs",
        MIN_CONFIRMATIONS,
        MIN_TF_CONFIRMATIONS,
    )

    scan_once()


if __name__ == "__main__":
    main()
