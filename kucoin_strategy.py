import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# CONFIG
# ============================================================

BASE_URL = "https://api.kucoin.com"

SYMBOLS = [
    "XBTUSDTM",
    "ETHUSDTM",
    "SOLUSDTM",
    "XRPUSDTM",
    "DOGEUSDTM",
]

KLINE_INTERVAL = "15min"
OI_INTERVAL = "15min"

KLINE_LIMIT = 100

# مدت تاریخچه Funding که دریافت می‌کنیم
FUNDING_HISTORY_HOURS = 24

REQUEST_TIMEOUT = 15

MAX_WORKERS = 5


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
    connect=3,
    read=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)

adapter = HTTPAdapter(
    max_retries=retry_strategy,
    pool_connections=20,
    pool_maxsize=20,
)

session.mount("https://", adapter)
session.mount("http://", adapter)


# ============================================================
# GENERAL API REQUEST
# ============================================================

def api_get(endpoint, params=None):
    """
    Send GET request to KuCoin public API.
    """

    url = BASE_URL + endpoint

    response = session.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    # اگر HTTP error بود، متن پاسخ KuCoin را هم نمایش بده
    if response.status_code != 200:
        logger.error(
            "KuCoin HTTP %s | URL: %s | Response: %s",
            response.status_code,
            response.url,
            response.text[:1000],
        )

    response.raise_for_status()

    payload = response.json()

    if payload.get("code") != "200000":
        raise RuntimeError(
            f"KuCoin API error | code={payload.get('code')} "
            f"| message={payload.get('msg') or payload.get('message')}"
        )

    return payload.get("data")


# ============================================================
# TICKER
# ============================================================

def get_ticker(symbol):
    """
    Get current futures ticker.
    """

    data = api_get(
        "/api/ua/v2/market/ticker",
        params={
            "symbol": symbol,
        },
    )

    if isinstance(data, list):
        if not data:
            return {}

        return data[0]

    return data or {}


# ============================================================
# CURRENT OPEN INTEREST
# ============================================================

def get_current_open_interest(symbol):
    """
    Get current futures open interest.
    """

    data = api_get(
        "/api/ua/v2/market/open-interest",
        params={
            "symbol": symbol,
        },
    )

    if isinstance(data, list):
        if not data:
            return {}

        return data[0]

    return data or {}


# ============================================================
# HISTORICAL OPEN INTEREST
# ============================================================

def get_historical_open_interest(symbol):
    """
    Get historical Open Interest.

    KuCoin currently supports:
        5min
        15min
        30min
        1hour
        4hour
        1day

    Historical retention:
        5m/15m/30m/1h/4h -> 7 days
        1d -> 70 days
    """

    end_at = int(time.time() * 1000)

    # فقط یک روز اخیر را می‌گیریم
    start_at = end_at - (24 * 60 * 60 * 1000)

    data = api_get(
        "/api/ua/v2/market/open-interest",
        params={
            "symbol": symbol,
            "interval": OI_INTERVAL,
            "startAt": start_at,
            "endAt": end_at,
            "pageSize": 200,
        },
    )

    if isinstance(data, dict):
        items = data.get("list", [])

        if isinstance(items, list):
            return items

    if isinstance(data, list):
        return data

    return []


# ============================================================
# CURRENT FUNDING RATE
# ============================================================

def get_funding_rate(symbol):
    """
    Get current futures funding rate.
    """

    data = api_get(
        "/api/ua/v2/market/funding-rate",
        params={
            "symbol": symbol,
        },
    )

    if isinstance(data, list):
        if not data:
            return {}

        return data[0]

    return data or {}


# ============================================================
# FUNDING RATE HISTORY
# ============================================================

def get_funding_history(symbol):
    """
    Get historical funding rates.

    IMPORTANT:
    KuCoin V2 requires:
        symbol
        startAt
        endAt

    Response structure:

    {
        "symbol": "XBTUSDTM",
        "list": [
            {
                "fundingRate": "...",
                "ts": ...
            }
        ]
    }
    """

    end_at = int(time.time() * 1000)

    start_at = end_at - (
        FUNDING_HISTORY_HOURS * 60 * 60 * 1000
    )

    data = api_get(
        "/api/ua/v2/market/funding-rate-history",
        params={
            "symbol": symbol,
            "startAt": start_at,
            "endAt": end_at,
        },
    )

    if isinstance(data, dict):
        items = data.get("list", [])

        if isinstance(items, list):
            return items

    if isinstance(data, list):
        return data

    return []


# ============================================================
# KLINES
# ============================================================

def get_klines(symbol):
    """
    Get futures candlestick data.
    """

    data = api_get(
        "/api/ua/v2/market/kline",
        params={
            "symbol": symbol,
            "interval": KLINE_INTERVAL,
            "limit": KLINE_LIMIT,
        },
    )

    if isinstance(data, dict):
        items = data.get("list", [])

        if isinstance(items, list):
            return items

    if isinstance(data, list):
        return data

    return []


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(value, default=0.0):
    """
    Safely convert value to float.
    """

    try:
        if value is None:
            return default

        return float(value)

    except (TypeError, ValueError):
        return default


# ============================================================
# PERCENTAGE CHANGE
# ============================================================

def percentage_change(old_value, new_value):
    """
    Calculate percentage change.
    """

    old_value = safe_float(old_value)
    new_value = safe_float(new_value)

    if old_value == 0:
        return 0.0

    return ((new_value - old_value) / old_value) * 100.0


# ============================================================
# OI ANALYSIS
# ============================================================

def analyze_oi(oi_history):
    """
    Analyze Open Interest changes over:

        15 minutes
        30 minutes
        1 hour
    """

    if not oi_history:
        return {
            "oi_current": 0.0,
            "oi_change_15m": 0.0,
            "oi_change_30m": 0.0,
            "oi_change_1h": 0.0,
        }

    # مرتب‌سازی بر اساس timestamp
    normalized = []

    for item in oi_history:
        if not isinstance(item, dict):
            continue

        ts = safe_float(
            item.get("ts")
            or item.get("time")
            or item.get("timestamp")
        )

        oi = safe_float(
            item.get("openInterest")
            or item.get("openInterestValue")
            or item.get("value")
        )

        if ts > 0 and oi != 0:
            normalized.append(
                {
                    "ts": ts,
                    "oi": oi,
                }
            )

    if not normalized:
        return {
            "oi_current": 0.0,
            "oi_change_15m": 0.0,
            "oi_change_30m": 0.0,
            "oi_change_1h": 0.0,
        }

    normalized.sort(key=lambda x: x["ts"])

    current = normalized[-1]["oi"]

    current_ts = normalized[-1]["ts"]

    def find_previous(minutes):
        target_ts = current_ts - (
            minutes * 60 * 1000
        )

        closest = None
        closest_distance = None

        for item in normalized:
            distance = abs(item["ts"] - target_ts)

            if closest_distance is None or distance < closest_distance:
                closest = item
                closest_distance = distance

        return closest

    item_15m = find_previous(15)
    item_30m = find_previous(30)
    item_1h = find_previous(60)

    change_15m = 0.0
    change_30m = 0.0
    change_1h = 0.0

    if item_15m:
        change_15m = percentage_change(
            item_15m["oi"],
            current,
        )

    if item_30m:
        change_30m = percentage_change(
            item_30m["oi"],
            current,
        )

    if item_1h:
        change_1h = percentage_change(
            item_1h["oi"],
            current,
        )

    return {
        "oi_current": current,
        "oi_change_15m": change_15m,
        "oi_change_30m": change_30m,
        "oi_change_1h": change_1h,
    }


# ============================================================
# CANDLE ANALYSIS
# ============================================================

def analyze_candles(klines):
    """
    Analyze price movement from 15m candles.
    """

    if not klines:
        return {
            "price_current": 0.0,
            "price_change_15m": 0.0,
            "price_change_1h": 0.0,
            "volume_24h": 0.0,
        }

    candles = []

    for candle in klines:

        if isinstance(candle, dict):

            ts = safe_float(
                candle.get("ts")
                or candle.get("timestamp")
                or candle.get("time")
            )

            close = safe_float(
                candle.get("close")
            )

            volume = safe_float(
                candle.get("volume")
            )

        elif isinstance(candle, list) and len(candle) >= 6:

            # KuCoin candle array:
            # [timestamp, open, close, high, low, volume, turnover]
            ts = safe_float(candle[0])
            close = safe_float(candle[2])
            volume = safe_float(candle[5])

        else:
            continue

        if ts > 0 and close > 0:
            candles.append(
                {
                    "ts": ts,
                    "close": close,
                    "volume": volume,
                }
            )

    if not candles:
        return {
            "price_current": 0.0,
            "price_change_15m": 0.0,
            "price_change_1h": 0.0,
            "volume_24h": 0.0,
        }

    candles.sort(key=lambda x: x["ts"])

    current_price = candles[-1]["close"]

    price_15m = (
        candles[-2]["close"]
        if len(candles) >= 2
        else current_price
    )

    price_1h = (
        candles[-5]["close"]
        if len(candles) >= 5
        else candles[0]["close"]
    )

    volume_24h = sum(
        candle["volume"]
        for candle in candles
    )

    return {
        "price_current": current_price,

        "price_change_15m": percentage_change(
            price_15m,
            current_price,
        ),

        "price_change_1h": percentage_change(
            price_1h,
            current_price,
        ),

        "volume_24h": volume_24h,
    }


# ============================================================
# FUNDING ANALYSIS
# ============================================================

def analyze_funding(funding_current, funding_history):
    """
    Analyze current and historical funding.
    """

    current_rate = 0.0
    next_rate = 0.0

    if isinstance(funding_current, dict):

        current_rate = safe_float(
            funding_current.get("fundingRate")
            or funding_current.get("value")
            or funding_current.get("currentFundingRate")
        )

        next_rate = safe_float(
            funding_current.get("nextFundingRate")
        )

    history_rates = []

    for item in funding_history:

        if not isinstance(item, dict):
            continue

        rate = safe_float(
            item.get("fundingRate")
        )

        history_rates.append(rate)

    average_rate = 0.0

    if history_rates:
        average_rate = (
            sum(history_rates)
            / len(history_rates)
        )

    return {
        "funding_rate": current_rate,
        "next_funding_rate": next_rate,
        "average_funding_rate": average_rate,
        "funding_history_count": len(history_rates),
    }


# ============================================================
# MARKET CLASSIFICATION
# ============================================================

def classify_market(
    price_change_1h,
    oi_change_1h,
):
    """
    Basic Price/OI market classification.

    This is NOT a trading signal.

    Four main states:

        PRICE_UP_OI_UP
        PRICE_DOWN_OI_UP
        PRICE_UP_OI_DOWN
        PRICE_DOWN_OI_DOWN

    Otherwise:

        NEUTRAL
    """

    price_up = price_change_1h > 0
    price_down = price_change_1h < 0

    oi_up = oi_change_1h > 0
    oi_down = oi_change_1h < 0

    if price_up and oi_up:
        return "PRICE_UP_OI_UP"

    if price_down and oi_up:
        return "PRICE_DOWN_OI_UP"

    if price_up and oi_down:
        return "PRICE_UP_OI_DOWN"

    if price_down and oi_down:
        return "PRICE_DOWN_OI_DOWN"

    return "NEUTRAL"


# ============================================================
# SYMBOL ANALYSIS
# ============================================================

def analyze_symbol(symbol):
    """
    Collect all available market data for one symbol.
    """

    logger.info(
        "Scanning %s",
        symbol,
    )

    # --------------------------------------------------------
    # API requests
    # --------------------------------------------------------

    ticker = get_ticker(symbol)

    current_oi = get_current_open_interest(symbol)

    historical_oi = get_historical_open_interest(symbol)

    funding_current = get_funding_rate(symbol)

    funding_history = get_funding_history(symbol)

    klines = get_klines(symbol)

    # --------------------------------------------------------
    # Ticker
    # --------------------------------------------------------

    ticker_price = safe_float(
        ticker.get("price")
        or ticker.get("lastPrice")
    )

    price_change_24h = safe_float(
        ticker.get("priceChangePercent")
        or ticker.get("changeRate")
        or ticker.get("priceChange")
    )

    volume_24h = safe_float(
        ticker.get("volume")
        or ticker.get("vol")
        or ticker.get("volume24h")
    )

    # --------------------------------------------------------
    # Current OI
    # --------------------------------------------------------

    current_oi_value = safe_float(
        current_oi.get("openInterest")
        or current_oi.get("openInterestValue")
        or current_oi.get("value")
    )

    # --------------------------------------------------------
    # OI analysis
    # --------------------------------------------------------

    oi_analysis = analyze_oi(
        historical_oi
    )

    # اگر current OI endpoint مقدار معتبرتری داشت
    if current_oi_value > 0:
        oi_analysis["oi_current"] = current_oi_value

    # --------------------------------------------------------
    # Candle analysis
    # --------------------------------------------------------

    candle_analysis = analyze_candles(
        klines
    )

    # --------------------------------------------------------
    # Funding analysis
    # --------------------------------------------------------

    funding_analysis = analyze_funding(
        funding_current,
        funding_history,
    )

    # --------------------------------------------------------
    # Market classification
    # --------------------------------------------------------

    market_state = classify_market(
        candle_analysis["price_change_1h"],
        oi_analysis["oi_change_1h"],
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    result = {
        "symbol": symbol,

        "price": (
            ticker_price
            or candle_analysis["price_current"]
        ),

        "price_change_24h": price_change_24h,

        "price_change_15m": candle_analysis[
            "price_change_15m"
        ],

        "price_change_1h": candle_analysis[
            "price_change_1h"
        ],

        "oi_current": oi_analysis[
            "oi_current"
        ],

        "oi_change_15m": oi_analysis[
            "oi_change_15m"
        ],

        "oi_change_30m": oi_analysis[
            "oi_change_30m"
        ],

        "oi_change_1h": oi_analysis[
            "oi_change_1h"
        ],

        "funding_rate": funding_analysis[
            "funding_rate"
        ],

        "next_funding_rate": funding_analysis[
            "next_funding_rate"
        ],

        "average_funding_rate": funding_analysis[
            "average_funding_rate"
        ],

        "funding_history_count": funding_analysis[
            "funding_history_count"
        ],

        "volume_24h": (
            volume_24h
            or candle_analysis["volume_24h"]
        ),

        "kline_count": len(klines),

        "market_state": market_state,
    }

    return result
if __name__ == "__main__":
    main()

# =========================================================