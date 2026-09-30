import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests
from requests.adapters import HTTPAdapter, Retry


# ============================================================
# CONFIG
# ============================================================

BASE_URL = "https://api-futures.kucoin.com"

SYMBOLS = [
    "XBTUSDTM",
    "ETHUSDTM",
    "SOLUSDTM",
    "XRPUSDTM",
    "DOGEUSDTM",
]

KLINE_GRANULARITY = 15      # minutes (must be a number, not "15min")
KLINE_COUNT = 100           # number of candles to request
FUNDING_HISTORY_HOURS = 24

# KuCoin has no public OI-history endpoint, so we store our own
# snapshots on every run and compare against them.
OI_SNAPSHOT_FILE = "oi_snapshots.json"
OI_KEEP_HOURS = 6
OI_TOLERANCE_MIN = 8        # max distance from the target time

REQUEST_TIMEOUT = 15
MAX_WORKERS = 3


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()

retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,  # so we can read the real response body
)

adapter = HTTPAdapter(
    max_retries=retry_strategy,
    pool_connections=10,
    pool_maxsize=10,
)

session.mount("https://", adapter)
session.mount("http://", adapter)


# ============================================================
# KUCOIN API
# ============================================================

def api_get(endpoint, params=None):
    """Send GET request to KuCoin Futures public API."""

    url = BASE_URL + endpoint

    response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)

    if response.status_code != 200:
        logger.error(
            "KuCoin HTTP %s | URL: %s | Response: %s",
            response.status_code,
            response.url,
            response.text[:500],
        )
        response.raise_for_status()

    payload = response.json()

    if payload.get("code") != "200000":
        raise RuntimeError(f"KuCoin API error: {payload}")

    return payload.get("data")


def safe_call(fn, *args, default=None):
    """Run fn; on any error log a warning and return default."""
    try:
        return fn(*args)
    except Exception as exc:
        logger.warning("%s%s failed: %s", fn.__name__, args, exc)
        return default


# ---------------- contract info (price, OI, funding) ----------------

def get_contract(symbol):
    """
    Contract details. Contains: lastTradePrice, markPrice,
    openInterest, fundingFeeRate, predictedFundingFeeRate,
    priceChgPct (fraction), turnoverOf24h, ...
    """
    data = api_get(f"/api/v1/contracts/{symbol}")
    return data or {}


# ---------------- funding ----------------

def get_funding_rate(symbol):
    """Current funding rate: {value, predictedValue, ...}"""
    data = api_get(f"/api/v1/funding-rate/{symbol}/current")
    return data or {}


def get_funding_history(symbol):
    """Funding history: data.dataList = [{timePoint, fundingRate}, ...]"""
    to_ms = int(time.time() * 1000)
    from_ms = to_ms - FUNDING_HISTORY_HOURS * 3600 * 1000

    data = api_get(
        "/api/v1/contract/funding-rates",
        params={"symbol": symbol, "from": from_ms, "to": to_ms},
    )

    if not data:
        return []
    if isinstance(data, dict):
        return data.get("dataList", [])
    if isinstance(data, list):
        return data
    return []


# ---------------- klines ----------------

def get_klines(symbol):
    """
    Each candle: [time_ms, open, high, low, close, volume]
    """
    to_ms = int(time.time() * 1000)
    from_ms = to_ms - KLINE_COUNT * KLINE_GRANULARITY * 60 * 1000

    data = api_get(
        "/api/v1/kline/query",
        params={
            "symbol": symbol,
            "granularity": KLINE_GRANULARITY,
            "from": from_ms,
            "to": to_ms,
        },
    )

    return data if isinstance(data, list) else []


# ============================================================
# HELPERS
# ============================================================

def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def percentage_change(old, new):
    old = safe_float(old)
    new = safe_float(new)

    if old == 0:
        return None

    return ((new - old) / old) * 100


# ============================================================
# OI SNAPSHOT STORAGE
# ============================================================

def load_snapshots():
    if not os.path.exists(OI_SNAPSHOT_FILE):
        return {}
    try:
        with open(OI_SNAPSHOT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Could not read %s: %s", OI_SNAPSHOT_FILE, exc)
        return {}


def save_snapshots(snapshots):
    try:
        with open(OI_SNAPSHOT_FILE, "w", encoding="utf-8") as f:
            json.dump(snapshots, f)
    except Exception as exc:
        logger.warning("Could not write %s: %s", OI_SNAPSHOT_FILE, exc)


def update_snapshots(snapshots, results):
    """Append the current OI of every symbol and prune old points."""
    now_ms = int(time.time() * 1000)
    cutoff = now_ms - OI_KEEP_HOURS * 3600 * 1000

    for r in results:
        current = r["open_interest"]["current"]
        if current is None:
            continue

        points = snapshots.get(r["symbol"], [])
        points.append({"ts": now_ms, "value": current})
        snapshots[r["symbol"]] = [p for p in points if p["ts"] >= cutoff]

    return snapshots


def find_closest(points, target_ms):
    """Snapshot closest to target_ms, within tolerance."""
    best = None
    best_diff = None

    for p in points:
        diff = abs(p["ts"] - target_ms)
        if best is None or diff < best_diff:
            best, best_diff = p, diff

    if best is None or best_diff > OI_TOLERANCE_MIN * 60 * 1000:
        return None

    return best


# ============================================================
# ANALYSIS
# ============================================================

def analyze_oi(current_value, points):
    result = {
        "current": current_value,
        "change_15m": None,
        "change_30m": None,
        "change_1h": None,
        "history_count": len(points),
    }

    if current_value is None or not points:
        return result

    now_ms = int(time.time() * 1000)

    for key, minutes in (
        ("change_15m", 15),
        ("change_30m", 30),
        ("change_1h", 60),
    ):
        old = find_closest(points, now_ms - minutes * 60 * 1000)
        if old is not None:
            result[key] = percentage_change(old["value"], current_value)

    return result


def analyze_candles(klines):
    result = {
        "count": len(klines),
        "price_change_15m": None,
        "price_change_1h": None,
        "volume": None,
        "last_price": None,
    }

    parsed = []

    for candle in klines:
        if not isinstance(candle, (list, tuple)) or len(candle) < 5:
            continue

        try:
            parsed.append(
                {
                    "timestamp": int(candle[0]),
                    "open": float(candle[1]),
                    "close": float(candle[4]),
                    "volume": float(candle[5]) if len(candle) > 5 else 0.0,
                }
            )
        except (TypeError, ValueError):
            continue

    if not parsed:
        return result

    parsed.sort(key=lambda x: x["timestamp"])

    latest = parsed[-1]
    result["last_price"] = latest["close"]

    if len(parsed) >= 2:
        result["price_change_15m"] = percentage_change(
            parsed[-2]["close"], latest["close"]
        )

    if len(parsed) >= 5:
        result["price_change_1h"] = percentage_change(
            parsed[-5]["close"], latest["close"]
        )

    result["volume"] = sum(c["volume"] for c in parsed)

    return result


def analyze_funding(contract, current_funding, funding_history):
    result = {
        "current": None,
        "next": None,
        "average_24h": None,
        "history_count": len(funding_history),
    }

    # current funding: prefer the dedicated endpoint, fall back to contract info
    result["current"] = safe_float(
        current_funding.get("value", contract.get("fundingFeeRate")), None
    )
    result["next"] = safe_float(
        current_funding.get(
            "predictedValue", contract.get("predictedFundingFeeRate")
        ),
        None,
    )

    rates = []

    for item in funding_history:
        if not isinstance(item, dict):
            continue

        value = safe_float(item.get("fundingRate", item.get("rate")), None)
        if value is not None:
            rates.append(value)

    if rates:
        result["average_24h"] = sum(rates) / len(rates)

    return result


def classify_market(price_change, oi_change, funding_rate):
    """Preliminary market description only. NOT a trading signal."""

    if price_change is None or oi_change is None:
        return "INSUFFICIENT_DATA"

    if price_change > 0 and oi_change > 0:
        if funding_rate is not None and funding_rate > 0:
            return "PRICE_UP_OI_UP_LONG_PRESSURE"
        return "PRICE_UP_OI_UP"

    if price_change < 0 and oi_change > 0:
        if funding_rate is not None and funding_rate < 0:
            return "PRICE_DOWN_OI_UP_SHORT_PRESSURE"
        return "PRICE_DOWN_OI_UP"

    if price_change > 0 and oi_change < 0:
        return "PRICE_UP_OI_DOWN"

    if price_change < 0 and oi_change < 0:
        return "PRICE_DOWN_OI_DOWN"

    return "NEUTRAL"


# ============================================================
# SYMBOL ANALYSIS
# ============================================================

def analyze_symbol(symbol, snapshots):
    logger.info("Scanning %s", symbol)

    contract = safe_call(get_contract, symbol, default={})
    current_funding = safe_call(get_funding_rate, symbol, default={})
    funding_history = safe_call(get_funding_history, symbol, default=[])
    klines = safe_call(get_klines, symbol, default=[])

    # ---------------- price ----------------
    price = safe_float(contract.get("lastTradePrice"), None)

    # priceChgPct is a fraction (0.012 = 1.2%) -> convert to percent
    chg = safe_float(contract.get("priceChgPct"), None)
    price_change_24h = chg * 100 if chg is not None else None

    # ---------------- open interest ----------------
    current_oi = safe_float(contract.get("openInterest"), None)
    oi_analysis = analyze_oi(current_oi, snapshots.get(symbol, []))

    # ---------------- other analysis ----------------
    candle_analysis = analyze_candles(klines)
    funding_analysis = analyze_funding(
        contract, current_funding, funding_history
    )

    if price is None:
        price = candle_analysis["last_price"]

    market_state = classify_market(
        candle_analysis["price_change_15m"],
        oi_analysis["change_15m"],
        funding_analysis["current"],
    )

    return {
        "symbol": symbol,
        "price": price,
        "price_change_24h": price_change_24h,
        "price_change_15m": candle_analysis["price_change_15m"],
        "price_change_1h": candle_analysis["price_change_1h"],
        "open_interest": oi_analysis,
        "funding": funding_analysis,
        "volume": candle_analysis["volume"],
        "kline_count": candle_analysis["count"],
        "market_state": market_state,
    }


# ============================================================
# PRINT RESULT
# ============================================================

def fmt(value, digits=4, suffix=""):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}{suffix}"


def print_result(result):
    oi = result["open_interest"]
    fd = result["funding"]

    print()
    print("=" * 70)
    print(f"SYMBOL: {result['symbol']}")
    print(f"PRICE: {fmt(result['price'], 4)}")
    print(f"24H PRICE CHANGE: {fmt(result['price_change_24h'], 2, '%')}")
    print(f"15M PRICE CHANGE: {fmt(result['price_change_15m'], 3, '%')}")
    print(f"1H PRICE CHANGE: {fmt(result['price_change_1h'], 3, '%')}")

    print()
    print("OPEN INTEREST")
    print(f"Current: {fmt(oi['current'], 0)}")
    print(f"15M Change: {fmt(oi['change_15m'], 3, '%')}")
    print(f"30M Change: {fmt(oi['change_30m'], 3, '%')}")
    print(f"1H Change: {fmt(oi['change_1h'], 3, '%')}")
    print(f"Snapshots stored: {oi['history_count']}")

    print()
    print("FUNDING")
    print(f"Current: {fmt(fd['current'], 6)}")
    print(f"Next: {fmt(fd['next'], 6)}")
    print(f"24H Average: {fmt(fd['average_24h'], 6)}")
    print(f"History Count: {fd['history_count']}")

    print()
    print(f"Volume (sum of {result['kline_count']} candles): {fmt(result['volume'], 0)}")
    print(f"MARKET STATE: {result['market_state']}")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():
    logger.info("KuCoin strategy scanner started")
    logger.info("UTC time: %s", datetime.now(timezone.utc).isoformat())

    snapshots = load_snapshots()
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(analyze_symbol, symbol, snapshots): symbol
            for symbol in SYMBOLS
        }

        for future in as_completed(futures):
            symbol = futures[future]

            try:
                result = future.result()
                results.append(result)
                print_result(result)
            except Exception as exc:
                logger.error("Error scanning %s: %s", symbol, exc)

    # save OI snapshots AFTER analysis so this run compares against the past
    snapshots = update_snapshots(snapshots, results)
    save_snapshots(snapshots)

    print()
    print("=" * 70)
    logger.info("Scan completed | Symbols: %d/%d", len(results), len(SYMBOLS))
    print("=" * 70)


if __name__ == "__main__":
    main()
