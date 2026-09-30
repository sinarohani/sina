import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests
from requests.adapters import HTTPAdapter, Retry


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

FUNDING_HISTORY_HOURS = 24
OI_HISTORY_HOURS = 6

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
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
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
    """
    Send GET request to KuCoin public API.
    """

    url = BASE_URL + endpoint

    response = session.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

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
            f"KuCoin API error: {payload}"
        )

    return payload.get("data")


# ============================================================
# TICKER
# ============================================================

def get_ticker(symbol):
    """
    Get current ticker information.
    """

    data = api_get(
        "//api/ua/v1/market/ticker?tradeType=FUTURES&symbol=XBTUSDTM,
        params={
            "symbol": symbol,
        },
    )

    if not data:
        return {}

    return data


# ============================================================
# CURRENT OPEN INTEREST
# ============================================================

def get_current_open_interest(symbol):
    """
    Get current open interest.
    """

    data = api_get(
    "/api/ua/v2/market/ticker",
    params={
        "tradeType": "FUTURES",
        "symbol": symbol,
    },
)

    if not data:
        return {}

    return data


# ============================================================
# HISTORICAL OPEN INTEREST
# ============================================================

def get_historical_open_interest(symbol):
    """
    Get historical open interest.
    """

    end_at = int(time.time() * 1000)

    start_at = (
        end_at
        - OI_HISTORY_HOURS * 60 * 60 * 1000
    )

    data = api_get(
        "/api/ua/v2/market/open-interest",
        params={
            "symbol": symbol,
            "interval": OI_INTERVAL,
            "startAt": start_at,
            "endAt": end_at,
        },
    )

    if not data:
        return []

    if isinstance(data, dict):
        return data.get("list", [])

    if isinstance(data, list):
        return data

    return []


# ============================================================
# CURRENT FUNDING RATE
# ============================================================

def get_funding_rate(symbol):
    """
    Get current funding rate.
    """

    data = api_get(
        "/api/ua/v2/market/funding-rate",
        params={
            "symbol": symbol,
        },
    )

    if not data:
        return {}

    return data


# ============================================================
# FUNDING HISTORY
# ============================================================

def get_funding_history(symbol):
    """
    Get funding rate history.
    """

    end_at = int(time.time() * 1000)

    start_at = (
        end_at
        - FUNDING_HISTORY_HOURS * 60 * 60 * 1000
    )

    data = api_get(
        "/api/ua/v2/market/funding-rate-history",
        params={
            "symbol": symbol,
            "startAt": start_at,
            "endAt": end_at,
        },
    )

    if not data:
        return []

    if isinstance(data, dict):
        return data.get("list", [])

    if isinstance(data, list):
        return data

    return []


# ============================================================
# KLINES
# ============================================================

def get_klines(symbol):
    """
    Get recent futures candles.
    """

    data = api_get(
        "/api/ua/v2/market/kline",
        params={
            "symbol": symbol,
            "interval": KLINE_INTERVAL,
            "limit": KLINE_LIMIT,
        },
    )

    if not data:
        return []

    if isinstance(data, dict):
        return data.get("list", [])

    if isinstance(data, list):
        return data

    return []


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


def extract_timestamp(item):
    """
    Try to extract timestamp from different KuCoin
    response formats.
    """

    if not isinstance(item, dict):
        return None

    for key in [
        "ts",
        "timestamp",
        "time",
        "startAt",
        "endAt",
    ]:
        if key in item:
            try:
                return int(item[key])
            except (TypeError, ValueError):
                pass

    return None


def extract_oi_value(item):
    """
    Try to extract OI from different possible
    KuCoin response field names.
    """

    if not isinstance(item, dict):
        return None

    for key in [
        "openInterest",
        "openInterestValue",
        "oi",
        "value",
    ]:
        if key in item:
            return safe_float(item[key], None)

    return None


# ============================================================
# OI ANALYSIS
# ============================================================

def analyze_oi(current_oi, historical_oi):
    result = {
        "current": None,
        "change_15m": None,
        "change_30m": None,
        "change_1h": None,
        "history_count": len(historical_oi),
    }

    current_value = None

    if isinstance(current_oi, dict):
        current_value = extract_oi_value(current_oi)

    if current_value is not None:
        result["current"] = current_value

    if not historical_oi:
        return result

    values = []

    for item in historical_oi:

        value = extract_oi_value(item)

        if value is None:
            continue

        ts = extract_timestamp(item)

        values.append(
            {
                "ts": ts,
                "value": value,
            }
        )

    if not values:
        return result

    values.sort(
        key=lambda x: (
            x["ts"]
            if x["ts"] is not None
            else 0
        )
    )

    if result["current"] is None:
        result["current"] = values[-1]["value"]

    latest = result["current"]

    # Approximate historical points
    # depending on returned interval.

    if len(values) >= 2:
        old = values[-2]["value"]

        result["change_15m"] = percentage_change(
            old,
            latest,
        )

    if len(values) >= 3:
        old = values[-3]["value"]

        result["change_30m"] = percentage_change(
            old,
            latest,
        )

    if len(values) >= 5:
        old = values[-5]["value"]

        result["change_1h"] = percentage_change(
            old,
            latest,
        )

    return result


# ============================================================
# CANDLE ANALYSIS
# ============================================================

def analyze_candles(klines):
    result = {
        "count": len(klines),
        "price_change_15m": None,
        "price_change_1h": None,
        "volume": None,
        "last_price": None,
    }

    if not klines:
        return result

    candles = list(klines)

    parsed = []

    for candle in candles:

        if not isinstance(candle, (list, tuple)):
            continue

        if len(candle) < 5:
            continue

        try:
            timestamp = int(candle[0])
            open_price = float(candle[1])
            close_price = float(candle[4])

            volume = (
                float(candle[5])
                if len(candle) > 5
                else 0
            )

            parsed.append(
                {
                    "timestamp": timestamp,
                    "open": open_price,
                    "close": close_price,
                    "volume": volume,
                }
            )

        except (TypeError, ValueError):
            continue

    if not parsed:
        return result

    parsed.sort(
        key=lambda x: x["timestamp"]
    )

    latest = parsed[-1]

    result["last_price"] = latest["close"]

    if len(parsed) >= 2:
        result["price_change_15m"] = percentage_change(
            parsed[-2]["close"],
            latest["close"],
        )

    if len(parsed) >= 5:
        result["price_change_1h"] = percentage_change(
            parsed[-5]["close"],
            latest["close"],
        )

    result["volume"] = sum(
        candle["volume"]
        for candle in parsed
    )

    return result


# ============================================================
# FUNDING ANALYSIS
# ============================================================

def analyze_funding(
    current_funding,
    funding_history,
):
    result = {
        "current": None,
        "next": None,
        "average_24h": None,
        "history_count": len(funding_history),
    }

    if isinstance(current_funding, dict):

        for key in [
            "fundingRate",
            "currentFundingRate",
            "value",
        ]:
            if key in current_funding:
                result["current"] = safe_float(
                    current_funding[key],
                    None,
                )
                break

        for key in [
            "nextFundingRate",
            "predictedFundingRate",
        ]:
            if key in current_funding:
                result["next"] = safe_float(
                    current_funding[key],
                    None,
                )
                break

    rates = []

    for item in funding_history:

        if not isinstance(item, dict):
            continue

        for key in [
            "fundingRate",
            "rate",
        ]:
            if key in item:

                value = safe_float(
                    item[key],
                    None,
                )

                if value is not None:
                    rates.append(value)

                break

    if rates:
        result["average_24h"] = (
            sum(rates) / len(rates)
        )

    return result


# ============================================================
# MARKET CLASSIFICATION
# ============================================================

def classify_market(
    price_change,
    oi_change,
    funding_rate,
):
    """
    Preliminary market description only.

    This is NOT a trading signal.
    """

    if (
        price_change is None
        or oi_change is None
    ):
        return "INSUFFICIENT_DATA"

    if (
        price_change > 0
        and oi_change > 0
    ):
        if funding_rate is not None and funding_rate > 0:
            return "PRICE_UP_OI_UP_LONG_PRESSURE"

        return "PRICE_UP_OI_UP"

    if (
        price_change < 0
        and oi_change > 0
    ):
        if funding_rate is not None and funding_rate < 0:
            return "PRICE_DOWN_OI_UP_SHORT_PRESSURE"

        return "PRICE_DOWN_OI_UP"

    if (
        price_change > 0
        and oi_change < 0
    ):
        return "PRICE_UP_OI_DOWN"

    if (
        price_change < 0
        and oi_change < 0
    ):
        return "PRICE_DOWN_OI_DOWN"

    return "NEUTRAL"


# ============================================================
# SYMBOL ANALYSIS
# ============================================================

def analyze_symbol(symbol):

    logger.info(
        "Scanning %s",
        symbol,
    )

    ticker = get_ticker(symbol)

    current_oi = get_current_open_interest(
        symbol
    )

    historical_oi = get_historical_open_interest(
        symbol
    )

    current_funding = get_funding_rate(
        symbol
    )

    funding_history = get_funding_history(
        symbol
    )

    klines = get_klines(symbol)

    # --------------------------------------------------------
    # PRICE
    # --------------------------------------------------------

    price = None
    price_change_24h = None

    if isinstance(ticker, dict):

        for key in [
            "price",
            "lastPrice",
            "last",
        ]:
            if key in ticker:
                price = safe_float(
                    ticker[key],
                    None,
                )
                break

        for key in [
            "priceChangePercent",
            "changeRate",
            "changePercentage",
        ]:
            if key in ticker:
                price_change_24h = safe_float(
                    ticker[key],
                    None,
                )
                break

    # --------------------------------------------------------
    # ANALYSIS
    # --------------------------------------------------------

    oi_analysis = analyze_oi(
        current_oi,
        historical_oi,
    )

    candle_analysis = analyze_candles(
        klines
    )

    funding_analysis = analyze_funding(
        current_funding,
        funding_history,
    )

    market_state = classify_market(
        candle_analysis["price_change_15m"],
        oi_analysis["change_15m"],
        funding_analysis["current"],
    )

    return {
        "symbol": symbol,
        "price": price,
        "price_change_24h": price_change_24h,
        "price_change_15m": candle_analysis[
            "price_change_15m"
        ],
        "price_change_1h": candle_analysis[
            "price_change_1h"
        ],
        "open_interest": oi_analysis,
        "funding": funding_analysis,
        "volume": candle_analysis["volume"],
        "kline_count": candle_analysis["count"],
        "market_state": market_state,
    }


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(result):

    print()
    print("=" * 70)

    print(
        f"SYMBOL: {result['symbol']}"
    )

    print(
        f"PRICE: {result['price']}"
    )

    print(
        f"24H PRICE CHANGE: "
        f"{result['price_change_24h']}"
    )

    print(
        f"15M PRICE CHANGE: "
        f"{result['price_change_15m']}"
    )

    print(
        f"1H PRICE CHANGE: "
        f"{result['price_change_1h']}"
    )

    # --------------------------------------------------------
    # OPEN INTEREST
    # --------------------------------------------------------

    oi = result["open_interest"]

    print()
    print("OPEN INTEREST")

    print(
        f"Current: "
        f"{oi['current']}"
    )

    print(
        f"15M Change: "
        f"{oi['change_15m']}"
    )

    print(
        f"30M Change: "
        f"{oi['change_30m']}"
    )

    print(
        f"1H Change: "
        f"{oi['change_1h']}"
    )

    print(
        f"History Count: "
        f"{oi['history_count']}"
    )

    # --------------------------------------------------------
    # FUNDING
    # --------------------------------------------------------

    funding = result["funding"]

    print()
    print("FUNDING")

    print(
        f"Current: "
        f"{funding['current']}"
    )

    print(
        f"Next: "
        f"{funding['next']}"
    )

    print(
        f"24H Average: "
        f"{funding['average_24h']}"
    )

    print(
        f"History Count: "
        f"{funding['history_count']}"
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    print()
    print(
        f"Volume: "
        f"{result['volume']}"
    )

    print(
        f"Kline Count: "
        f"{result['kline_count']}"
    )

    # --------------------------------------------------------
    # MARKET STATE
    # --------------------------------------------------------

    print()
    print(
        f"MARKET STATE: "
        f"{result['market_state']}"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info(
        "KuCoin strategy scanner started"
    )

    now = datetime.now(
        timezone.utc
    )

    logger.info(
        "UTC time: %s",
        now.isoformat(),
    )

    results = []

    # --------------------------------------------------------
    # PARALLEL SCANNING
    # --------------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                analyze_symbol,
                symbol,
            ): symbol
            for symbol in SYMBOLS
        }

        for future in as_completed(futures):

            symbol = futures[future]

            try:

                result = future.result()

                results.append(result)

                print_result(result)

            except Exception as exc:

                logger.error(
                    "Error scanning %s: %s",
                    symbol,
                    exc,
                )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)

    logger.info(
        "Scan completed | Symbols: %d/%d",
        len(results),
        len(SYMBOLS),
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()