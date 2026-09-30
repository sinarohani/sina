import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =========================================================
# CONFIG
# =========================================================

BASE_URL = "https://api.kucoin.com"

SYMBOLS = [
    "XBTUSDTM",
    "ETHUSDTM",
    "SOLUSDTM",
    "XRPUSDTM",
    "DOGEUSDTM",
]

INTERVAL = "15min"

OI_INTERVAL = "15min"

KLINE_LIMIT = 100

REQUEST_TIMEOUT = 15

MAX_WORKERS = 5


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("kucoin_strategy")


# =========================================================
# HTTP SESSION
# =========================================================

def create_session():
    session = requests.Session()

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504,
        ],
        allowed_methods=["GET"],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=20,
        pool_maxsize=20,
    )

    session.mount("https://", adapter)

    session.headers.update(
        {
            "User-Agent": "KuCoin-Strategy-Bot/1.0",
            "Accept": "application/json",
        }
    )

    return session


SESSION = create_session()


# =========================================================
# COMMON REQUEST
# =========================================================

def api_get(endpoint, params=None):
    url = f"{BASE_URL}{endpoint}"

    response = SESSION.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    if data.get("code") != "200000":
        raise RuntimeError(
            f"KuCoin API error: {data}"
        )

    return data.get("data")


# =========================================================
# TICKER
# =========================================================

def get_ticker(symbol):
    """
    دریافت قیمت و اطلاعات 24 ساعته Futures.
    """

    data = api_get(
        "/api/ua/v2/market/ticker",
        {
            "tradeType": "FUTURES",
            "symbol": symbol,
        },
    )

    if not data:
        raise RuntimeError(
            f"Ticker data is empty: {symbol}"
        )

    if isinstance(data, list):
        ticker = data[0]
    elif isinstance(data, dict):
        ticker = data
    else:
        raise RuntimeError(
            f"Unexpected ticker response: {data}"
        )

    return {
        "price": float(
            ticker.get("lastPrice", 0)
        ),
        "price_change": float(
            ticker.get("priceChange", 0)
        ),
        "price_change_percent": float(
            ticker.get("priceChangePercent", 0)
        ),
        "volume": float(
            ticker.get("volume", 0)
        ),
        "turnover": float(
            ticker.get("turnover", 0)
        ),
    }


# =========================================================
# CURRENT OPEN INTEREST
# =========================================================

def get_current_open_interest(symbol):
    """
    دریافت OI لحظه‌ای.
    """

    data = api_get(
        "/api/ua/v2/market/open-interest",
        {
            "symbol": symbol,
        },
    )

    if not data:
        raise RuntimeError(
            f"Open Interest data is empty: {symbol}"
        )

    if isinstance(data, list):
        item = data[0]
    elif isinstance(data, dict):
        item = data
    else:
        raise RuntimeError(
            f"Unexpected OI response: {data}"
        )

    oi_value = (
        item.get("openInterest")
        or item.get("openInterestValue")
    )

    if oi_value is None:
        raise RuntimeError(
            f"OI field not found: {item}"
        )

    return {
        "open_interest": float(oi_value),
        "timestamp": item.get("ts"),
    }


# =========================================================
# HISTORICAL OPEN INTEREST
# =========================================================

def get_historical_open_interest(
    symbol,
    interval=OI_INTERVAL,
):
    """
    دریافت OI تاریخی.

    KuCoin برای OI تاریخی:
        5min
        15min
        30min
        1hour
        4hour

    تا 7 روز نگهداری می‌کند.
    """

    data = api_get(
        "/api/ua/v2/market/open-interest",
        {
            "symbol": symbol,
            "interval": interval,
            "pageSize": 200,
        },
    )

    if not data:
        raise RuntimeError(
            f"Historical OI is empty: {symbol}"
        )

    if isinstance(data, dict):
        rows = data.get("list", [])
    elif isinstance(data, list):
        rows = data
    else:
        rows = []

    result = []

    for row in rows:

        if isinstance(row, dict):

            timestamp = (
                row.get("ts")
                or row.get("timestamp")
                or row.get("time")
            )

            oi_value = (
                row.get("openInterest")
                or row.get("openInterestValue")
            )

            if oi_value is None:
                continue

            result.append(
                {
                    "timestamp": int(timestamp)
                    if timestamp is not None
                    else None,
                    "open_interest": float(oi_value),
                }
            )

        elif isinstance(row, list):

            if len(row) >= 2:
                result.append(
                    {
                        "timestamp": int(row[0]),
                        "open_interest": float(row[1]),
                    }
                )

    result.sort(
        key=lambda x: (
            x["timestamp"]
            if x["timestamp"] is not None
            else 0
        )
    )

    return result


# =========================================================
# FUNDING RATE
# =========================================================

def get_funding_rate(symbol):
    """
    دریافت Funding Rate فعلی.
    """

    data = api_get(
        "/api/ua/v2/market/funding-rate",
        {
            "symbol": symbol,
        },
    )

    if not data:
        raise RuntimeError(
            f"Funding data is empty: {symbol}"
        )

    if isinstance(data, list):
        funding = data[0]
    elif isinstance(data, dict):
        funding = data
    else:
        raise RuntimeError(
            f"Unexpected funding response: {data}"
        )

    rate = funding.get(
        "nextFundingRate"
    )

    if rate is None:
        rate = funding.get(
            "fundingRate"
        )

    if rate is None:
        raise RuntimeError(
            f"Funding rate field not found: {funding}"
        )

    return {
        "funding_rate": float(rate),
        "funding_percent": float(rate) * 100,
        "funding_time": funding.get(
            "fundingTime"
        ),
        "funding_cap": funding.get(
            "fundingRateCap"
        ),
        "funding_floor": funding.get(
            "fundingRateFloor"
        ),
    }


# =========================================================
# FUNDING HISTORY
# =========================================================

def get_funding_history(symbol):
    """
    دریافت Funding Rate های قبلی.
    """

    data = api_get(
        "/api/ua/v2/market/funding-rate-history",
        {
            "symbol": symbol,
        },
    )

    if not data:
        return []

    if isinstance(data, dict):
        rows = data.get("list", [])
    elif isinstance(data, list):
        rows = data
    else:
        rows = []

    result = []

    for row in rows:

        if not isinstance(row, dict):
            continue

        rate = row.get(
            "fundingRate"
        )

        timestamp = row.get(
            "ts"
        )

        if rate is None:
            continue

        result.append(
            {
                "timestamp": int(timestamp)
                if timestamp is not None
                else None,
                "funding_rate": float(rate),
            }
        )

    result.sort(
        key=lambda x: (
            x["timestamp"]
            if x["timestamp"] is not None
            else 0
        )
    )

    return result


# =========================================================
# KLINES
# =========================================================

def get_klines(
    symbol,
    interval=INTERVAL,
):
    """
    دریافت کندل‌های Futures.
    """

    data = api_get(
        "/api/ua/v2/market/kline",
        {
            "symbol": symbol,
            "tradeType": "FUTURES",
            "klineType": "TRADE",
            "interval": interval,
        },
    )

    if not data:
        raise RuntimeError(
            f"Kline data is empty: {symbol}"
        )

    if isinstance(data, dict):
        rows = data.get("list", [])
    elif isinstance(data, list):
        rows = data
    else:
        rows = []

    candles = []

    for row in rows:

        if not isinstance(row, list):
            continue

        if len(row) < 7:
            continue

        try:
            candle = {
                "timestamp": int(row[0]),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "turnover": float(row[6]),
            }

            candles.append(candle)

        except (
            ValueError,
            TypeError,
        ):
            continue

    candles.sort(
        key=lambda x: x["timestamp"]
    )

    if KLINE_LIMIT:
        candles = candles[-KLINE_LIMIT:]

    return candles


# =========================================================
# PRICE CHANGE
# =========================================================

def percentage_change(old, new):
    if old == 0:
        return 0.0

    return (
        (new - old)
        / old
        * 100
    )


# =========================================================
# OI ANALYSIS
# =========================================================

def analyze_oi(oi_history):
    """
    محاسبه تغییر OI نسبت به نقاط قبلی.
    """

    if not oi_history:
        return {
            "oi_change": 0.0,
            "oi_change_2": 0.0,
            "oi_change_4": 0.0,
        }

    values = [
        x["open_interest"]
        for x in oi_history
    ]

    latest = values[-1]

    change_1 = 0.0
    change_2 = 0.0
    change_4 = 0.0

    if len(values) >= 2:
        change_1 = percentage_change(
            values[-2],
            latest,
        )

    if len(values) >= 3:
        change_2 = percentage_change(
            values[-3],
            latest,
        )

    if len(values) >= 5:
        change_4 = percentage_change(
            values[-5],
            latest,
        )

    return {
        "oi_change": change_1,
        "oi_change_2": change_2,
        "oi_change_4": change_4,
    }


# =========================================================
# PRICE / VOLUME ANALYSIS
# =========================================================

def analyze_candles(candles):

    if not candles:
        return {
            "price_change_15m": 0.0,
            "price_change_1h": 0.0,
            "volume_change": 0.0,
        }

    closes = [
        x["close"]
        for x in candles
    ]

    volumes = [
        x["volume"]
        for x in candles
    ]

    latest_close = closes[-1]

    price_15m = 0.0
    price_1h = 0.0

    if len(closes) >= 2:
        price_15m = percentage_change(
            closes[-2],
            latest_close,
        )

    if len(closes) >= 5:
        price_1h = percentage_change(
            closes[-5],
            latest_close,
        )

    volume_change = 0.0

    if len(volumes) >= 6:

        previous_volume = sum(
            volumes[-6:-1]
        ) / 5

        latest_volume = volumes[-1]

        if previous_volume != 0:
            volume_change = (
                (
                    latest_volume
                    - previous_volume
                )
                / previous_volume
                * 100
            )

    return {
        "price_change_15m": price_15m,
        "price_change_1h": price_1h,
        "volume_change": volume_change,
    }


# =========================================================
# MARKET ANALYSIS
# =========================================================

def analyze_symbol(symbol):

    ticker = get_ticker(symbol)

    current_oi = get_current_open_interest(
        symbol
    )

    historical_oi = get_historical_open_interest(
        symbol
    )

    funding = get_funding_rate(
        symbol
    )

    funding_history = get_funding_history(
        symbol
    )

    candles = get_klines(
        symbol
    )

    oi_analysis = analyze_oi(
        historical_oi
    )

    candle_analysis = analyze_candles(
        candles
    )

    return {
        "symbol": symbol,

        "price": ticker["price"],

        "price_change_24h": (
            ticker["price_change_percent"]
        ),

        "volume_24h": ticker["volume"],

        "turnover_24h": ticker["turnover"],

        "open_interest": (
            current_oi["open_interest"]
        ),

        "funding_rate": (
            funding["funding_rate"]
        ),

        "funding_percent": (
            funding["funding_percent"]
        ),

        "funding_time": (
            funding["funding_time"]
        ),

        "oi_change_15m": (
            oi_analysis["oi_change"]
        ),

        "oi_change_30m": (
            oi_analysis["oi_change_2"]
        ),

        "oi_change_1h": (
            oi_analysis["oi_change_4"]
        ),

        "price_change_15m": (
            candle_analysis[
                "price_change_15m"
            ]
        ),

        "price_change_1h": (
            candle_analysis[
                "price_change_1h"
            ]
        ),

        "volume_change": (
            candle_analysis[
                "volume_change"
            ]
        ),

        "funding_history_count": len(
            funding_history
        ),

        "kline_count": len(
            candles
        ),
    }


# =========================================================
# MARKET INTERPRETATION
# =========================================================

def classify_market(data):
    """
    این قسمت فعلاً فقط وضعیت بازار را توصیف می‌کند.

    هنوز BUY / SELL واقعی نیست.
    """

    price_change = (
        data["price_change_15m"]
    )

    oi_change = (
        data["oi_change_15m"]
    )

    funding = (
        data["funding_rate"]
    )

    volume_change = (
        data["volume_change"]
    )

    if (
        price_change > 0
        and oi_change > 0
    ):
        state = "PRICE_UP_OI_UP"

    elif (
        price_change < 0
        and oi_change > 0
    ):
        state = "PRICE_DOWN_OI_UP"

    elif (
        price_change > 0
        and oi_change < 0
    ):
        state = "PRICE_UP_OI_DOWN"

    elif (
        price_change < 0
        and oi_change < 0
    ):
        state = "PRICE_DOWN_OI_DOWN"

    else:
        state = "NEUTRAL"

    return {
        "state": state,
        "price_change": price_change,
        "oi_change": oi_change,
        "funding": funding,
        "volume_change": volume_change,
    }


# =========================================================
# PRINT RESULT
# =========================================================

def print_result(data):

    market = classify_market(
        data
    )

    print()
    print("=" * 72)
    print(
        f" {data['symbol']}"
    )
    print("=" * 72)

    print(
        f"Price              : "
        f"{data['price']:,.6f}"
    )

    print(
        f"24h Price Change   : "
        f"{data['price_change_24h']:+.3f}%"
    )

    print(
        f"15m Price Change   : "
        f"{data['price_change_15m']:+.3f}%"
    )

    print(
        f"1h Price Change    : "
        f"{data['price_change_1h']:+.3f}%"
    )

    print(
        f"Open Interest      : "
        f"{data['open_interest']:,.4f}"
    )

    print(
        f"OI Change 15m      : "
        f"{data['oi_change_15m']:+.3f}%"
    )

    print(
        f"OI Change 30m      : "
        f"{data['oi_change_30m']:+.3f}%"
    )

    print(
        f"OI Change 1h       : "
        f"{data['oi_change_1h']:+.3f}%"
    )

    print(
        f"Funding Rate       : "
        f"{data['funding_rate']:+.8f}"
    )

    print(
        f"Funding %          : "
        f"{data['funding_percent']:+.5f}%"
    )

    print(
        f"24h Volume         : "
        f"{data['volume_24h']:,.4f}"
    )

    print(
        f"Volume Change      : "
        f"{data['volume_change']:+.2f}%"
    )

    print(
        f"Market State       : "
        f"{market['state']}"
    )

    print(
        f"Klines Loaded      : "
        f"{data['kline_count']}"
    )

    print("=" * 72)


# =========================================================
# MAIN SCANNER
# =========================================================

def main():

    start_time = time.time()

    logger.info(
        "KuCoin strategy scanner started"
    )

    logger.info(
        "UTC time: %s",
        datetime.now(
            timezone.utc
        ).strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
    )

    results = []

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        future_map = {
            executor.submit(
                analyze_symbol,
                symbol,
            ): symbol
            for symbol in SYMBOLS
        }

        for future in as_completed(
            future_map
        ):

            symbol = future_map[
                future
            ]

            try:

                result = future.result()

                results.append(
                    result
                )

                print_result(
                    result
                )

                logger.info(
                    "%s scanned successfully",
                    symbol,
                )

            except Exception as exc:

                logger.exception(
                    "Error scanning %s: %s",
                    symbol,
                    exc,
                )

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print("=" * 72)

    print(
        f"Scan completed | "
        f"Symbols: {len(results)}/{len(SYMBOLS)} | "
        f"Time: {elapsed:.2f}s"
    )

    print("=" * 72)

    return results


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()