#!/usr/bin/env python3
"""
Free Derivatives Signal Bot v2 (no paid API)
============================================
Public futures endpoints of KuCoin, Binance and Bybit (no API key needed),
multi-factor scoring, LONG/SHORT signals to Telegram.

Changes vs v1:
  * Unclosed (in-progress) candle is dropped before any calculation -> no repaint
  * Rolling z-score (default 40-candle window) instead of full-history z
  * Funding scored against an ABSOLUTE extreme threshold (default 0.05%),
    not a noisy z-score - funding edge exists mainly at crowded extremes
  * ADX regime filter: in trend, trend factor weighted up and contrarian
    factors down; in range, the opposite. Weights renormalised automatically.
  * ATR computed with Wilder smoothing
  * Risk-based position sizing (qty from capital * risk% / stop distance)
  * Round-trip taker fee shown on every signal
  * Cooldown applies to BOTH sides: same-side = full window, flip = short window
  * KuCoin OI history survives restarts as long as state.json is kept

NOT financial advice - backtest / paper trade first.
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import os
import signal
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable

import requests
from dotenv import load_dotenv

log = logging.getLogger("signal_bot")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


@dataclass
class Config:
    tg_token: str = ""
    tg_chat_id: str = ""
    providers: list[str] = field(default_factory=lambda: ["kucoin"])
    kucoin_url: str = "https://api-futures.kucoin.com"
    binance_url: str = "https://fapi.binance.com"
    bybit_url: str = "https://api.bybit.com"
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    interval: str = "4h"
    limit: int = 200
    poll_seconds: int = 900
    request_delay: float = 0.3
    timeout: int = 15
    max_retries: int = 3
    # strategy
    signal_threshold: float = 0.45
    rolling_z_window: int = 40        # window for rolling z-score
    funding_extreme: float = 0.0005   # |funding| >= 0.05% is "crowded"
    adx_period: int = 14
    adx_trend_min: float = 20.0       # ADX >= this => trending regime
    oi_change_threshold_pct: float = 2.0
    trend_lookback: int = 6
    # risk
    atr_period: int = 14
    sl_atr_mult: float = 1.5
    tp1_atr_mult: float = 1.5
    tp2_atr_mult: float = 3.0
    capital_usdt: float = 0.0         # 0 disables position sizing output
    risk_pct: float = 1.0             # % of capital risked per trade
    taker_fee: float = 0.0005         # per side, for cost display
    # cooldown
    cooldown_hours: float = 8.0
    flip_cooldown_factor: float = 0.33
    state_file: str = "state.json"
    log_dir: str = "logs"

    @classmethod
    def from_env(cls) -> "Config":
        d = cls()
        return cls(
            tg_token=_env("TELEGRAM_BOT_TOKEN", ""),
            tg_chat_id=_env("TELEGRAM_CHAT_ID", ""),
            providers=[p.strip().lower() for p in _env("PROVIDERS", "kucoin").split(",") if p.strip()],
            kucoin_url=_env("KUCOIN_URL", d.kucoin_url),
            binance_url=_env("BINANCE_URL", d.binance_url),
            bybit_url=_env("BYBIT_URL", d.bybit_url),
            symbols=[s.strip().upper() for s in _env("SYMBOLS", "BTCUSDT,ETHUSDT").split(",") if s.strip()],
            interval=_env("INTERVAL", d.interval).lower(),
            limit=int(_env("LIMIT", str(d.limit))),
            poll_seconds=int(_env("POLL_SECONDS", str(d.poll_seconds))),
            signal_threshold=float(_env("SIGNAL_THRESHOLD", str(d.signal_threshold))),
            rolling_z_window=int(_env("ROLLING_Z_WINDOW", str(d.rolling_z_window))),
            funding_extreme=float(_env("FUNDING_EXTREME", str(d.funding_extreme))),
            adx_period=int(_env("ADX_PERIOD", str(d.adx_period))),
            adx_trend_min=float(_env("ADX_TREND_MIN", str(d.adx_trend_min))),
            sl_atr_mult=float(_env("SL_ATR_MULT", str(d.sl_atr_mult))),
            tp1_atr_mult=float(_env("TP1_ATR_MULT", str(d.tp1_atr_mult))),
            tp2_atr_mult=float(_env("TP2_ATR_MULT", str(d.tp2_atr_mult))),
            capital_usdt=float(_env("CAPITAL_USDT", str(d.capital_usdt))),
            risk_pct=float(_env("RISK_PCT", str(d.risk_pct))),
            cooldown_hours=float(_env("COOLDOWN_HOURS", str(d.cooldown_hours))),
            flip_cooldown_factor=float(_env("FLIP_COOLDOWN_FACTOR", str(d.flip_cooldown_factor))),
            state_file=_env("STATE_FILE", d.state_file),
            log_dir=_env("LOG_DIR", d.log_dir),
        )


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(level: str, log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s")
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root.addHandler(console)

    main_file = RotatingFileHandler(Path(log_dir) / "bot.log", maxBytes=5_000_000,
                                    backupCount=5, encoding="utf-8")
    main_file.setFormatter(fmt)
    root.addHandler(main_file)

    err_file = RotatingFileHandler(Path(log_dir) / "errors.log", maxBytes=2_000_000,
                                   backupCount=3, encoding="utf-8")
    err_file.setLevel(logging.ERROR)
    err_file.setFormatter(fmt)
    root.addHandler(err_file)

    logging.getLogger("urllib3").setLevel(logging.WARNING)


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #
class ApiError(Exception):
    pass


class GeoBlockedError(Exception):
    """Exchange refuses this IP (HTTP 451/403), typical for GitHub-hosted runners."""


class Http:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "Mozilla/5.0 (derivatives-signal-bot)"

    def get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        last: Exception | None = None
        for attempt in range(1, self.cfg.max_retries + 1):
            started = time.monotonic()
            try:
                r = self.session.get(url, params=params, timeout=self.cfg.timeout)
                log.debug("GET %s %s -> %s in %.0fms", url, params, r.status_code,
                          (time.monotonic() - started) * 1000)
                if r.status_code in (451, 403):
                    raise GeoBlockedError(f"HTTP {r.status_code} from {url}: {r.text[:120]}")
                if r.status_code in (418, 429):
                    wait = min(2 ** attempt * 3, 60)
                    log.warning("Rate limited (%s) on %s, sleeping %ss", r.status_code, url, wait)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r.json()
            except GeoBlockedError:
                raise
            except (requests.RequestException, ValueError) as exc:
                last = exc
                wait = min(2 ** attempt, 20)
                log.warning("Request failed %s (attempt %d/%d): %s - retry in %ss",
                            url, attempt, self.cfg.max_retries, exc, wait)
                time.sleep(wait)
        raise ApiError(f"Giving up on {url}: {last}")


# --------------------------------------------------------------------------- #
# Data providers (all series oldest -> newest, CLOSED candles only)
# --------------------------------------------------------------------------- #
@dataclass
class MarketData:
    source: str
    close: list[float]
    high: list[float]
    low: list[float]
    times: list[int] = field(default_factory=list)  # candle open time (ms)
    funding: list[float] | None = None
    oi: list[float] | None = None
    ls: list[float] | None = None      # long/short account ratio
    flow: list[float] | None = None    # taker buy/sell ratio
    oi_change_pct: float | None = None  # pre-computed OI change (providers without OI history)


def trim_open_candle(md: MarketData, interval_min: int) -> MarketData:
    """Drop the in-progress candle so every calculation runs on closed data."""
    if not md.times or not md.close:
        return md
    step_ms = interval_min * 60_000
    now_ms = time.time() * 1000
    dropped = 0
    while md.times and md.times[-1] + step_ms > now_ms + 5_000:  # 5s clock skew guard
        md.times.pop(); md.close.pop(); md.high.pop(); md.low.pop()
        if md.oi:
            md.oi.pop()
        if md.ls:
            md.ls.pop()
        if md.flow:
            md.flow.pop()
        dropped += 1
        if dropped > 3:  # safety, should never loop more than once
            break
    # funding series is SETTLED history - it is never affected by the open candle
    if dropped:
        log.info("%s: dropped %d unclosed candle(s)", md.source, dropped)
    return md


class Provider:
    name = "base"

    def __init__(self, cfg: Config, http: Http, state: "State | None" = None):
        self.cfg, self.http, self.state = cfg, http, state

    def fetch(self, symbol: str) -> MarketData:
        raise NotImplementedError

    def _optional(self, label: str, symbol: str, fn: Callable) -> list[float] | None:
        time.sleep(self.cfg.request_delay)
        try:
            values = fn()
            log.info("%s/%s: %s -> %d points", self.name, symbol, label, len(values))
            return values or None
        except GeoBlockedError:
            raise
        except Exception as exc:
            log.warning("%s/%s: optional series '%s' unavailable: %s", self.name, symbol, label, exc)
            return None


class BinanceProvider(Provider):
    name = "binance"

    def _get(self, path: str, **params: Any) -> Any:
        return self.http.get(self.cfg.binance_url.rstrip("/") + path, params)

    def fetch(self, symbol: str) -> MarketData:
        iv, n = self.cfg.interval, min(self.cfg.limit, 500)
        kl = self._get("/fapi/v1/klines", symbol=symbol, interval=iv, limit=n)
        if not isinstance(kl, list) or not kl:
            raise ApiError(f"empty klines: {str(kl)[:120]}")
        md = MarketData(self.name,
                        [float(k[4]) for k in kl],
                        [float(k[2]) for k in kl],
                        [float(k[3]) for k in kl],
                        [int(k[0]) for k in kl])
        md.funding = self._optional("funding", symbol, lambda: [
            float(x["fundingRate"]) for x in self._get("/fapi/v1/fundingRate", symbol=symbol, limit=n)])
        md.oi = self._optional("open_interest", symbol, lambda: [
            float(x["sumOpenInterestValue"]) for x in
            self._get("/futures/data/openInterestHist", symbol=symbol, period=iv, limit=n)])
        md.ls = self._optional("long_short_ratio", symbol, lambda: [
            float(x["longShortRatio"]) for x in
            self._get("/futures/data/globalLongShortAccountRatio", symbol=symbol, period=iv, limit=n)])
        md.flow = self._optional("taker_flow", symbol, lambda: [
            float(x["buySellRatio"]) for x in
            self._get("/futures/data/takerlongshortRatio", symbol=symbol, period=iv, limit=n)])
        return trim_open_candle(md, interval_minutes(iv))


class BybitProvider(Provider):
    name = "bybit"
    KLINE = {"5m": "5", "15m": "15", "30m": "30", "1h": "60", "2h": "120",
             "4h": "240", "6h": "360", "12h": "720", "1d": "D"}
    PERIOD = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1d"}

    def _get(self, path: str, **params: Any) -> list[dict]:
        body = self.http.get(self.cfg.bybit_url.rstrip("/") + path, params)
        if body.get("retCode") != 0:
            raise ApiError(f"Bybit retCode={body.get('retCode')} msg={body.get('retMsg')}")
        return body["result"]["list"]

    def fetch(self, symbol: str) -> MarketData:
        iv, n = self.cfg.interval, min(self.cfg.limit, 200)
        if iv not in self.KLINE:
            raise ApiError(f"interval '{iv}' unsupported by Bybit")
        kl = list(reversed(self._get("/v5/market/kline", category="linear", symbol=symbol,
                                     interval=self.KLINE[iv], limit=n)))
        if not kl:
            raise ApiError("empty klines")
        md = MarketData(self.name,
                        [float(k[4]) for k in kl],
                        [float(k[2]) for k in kl],
                        [float(k[3]) for k in kl],
                        [int(k[0]) for k in kl])
        md.funding = self._optional("funding", symbol, lambda: [
            float(x["fundingRate"]) for x in reversed(
                self._get("/v5/market/funding/history", category="linear", symbol=symbol, limit=n))])
        if iv in self.PERIOD:
            p = self.PERIOD[iv]
            md.oi = self._optional("open_interest", symbol, lambda: [
                float(x["openInterest"]) for x in reversed(
                    self._get("/v5/market/open-interest", category="linear", symbol=symbol,
                              intervalTime=p, limit=n))])
            md.ls = self._optional("long_short_ratio", symbol, lambda: [
                float(x["buyRatio"]) / float(x["sellRatio"]) for x in reversed(
                    self._get("/v5/market/account-ratio", category="linear", symbol=symbol,
                              period=p, limit=n))])
        return trim_open_candle(md, interval_minutes(iv))


class KuCoinProvider(Provider):
    """KuCoin Futures public API. No OI history / L-S ratio endpoint exists, so
    open interest is snapshotted every cycle into the state file and the change
    is computed from our own history (factor activates after one lookback window)."""
    name = "kucoin"
    GRAN = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120,
            "4h": 240, "8h": 480, "12h": 720, "1d": 1440}

    @staticmethod
    def to_contract(symbol: str) -> str:
        if not symbol.endswith("USDT"):
            raise ApiError(f"unsupported symbol '{symbol}' (expected ...USDT)")
        base = symbol[:-4]
        return f"{'XBT' if base == 'BTC' else base}USDTM"

    def _get(self, path: str, **params: Any) -> Any:
        body = self.http.get(self.cfg.kucoin_url.rstrip("/") + path, params)
        if str(body.get("code")) != "200000":
            raise ApiError(f"KuCoin code={body.get('code')} msg={body.get('msg')}")
        return body["data"]

    def fetch(self, symbol: str) -> MarketData:
        iv = self.cfg.interval
        if iv not in self.GRAN:
            raise ApiError(f"interval '{iv}' unsupported by KuCoin")
        contract, gran = self.to_contract(symbol), self.GRAN[iv]
        now_ms = int(time.time() * 1000)
        n = min(self.cfg.limit, 400)
        kl = self._get("/api/v1/kline/query", symbol=contract, granularity=gran,
                       **{"from": now_ms - n * gran * 60_000, "to": now_ms})
        kl = sorted(kl or [], key=lambda k: k[0])
        if not kl:
            raise ApiError("empty klines")
        md = MarketData(self.name,
                        [float(k[4]) for k in kl],
                        [float(k[2]) for k in kl],
                        [float(k[3]) for k in kl],
                        [int(k[0]) for k in kl])

        def funding() -> list[float]:
            data = self._get("/api/v1/contract/funding-rates", symbol=contract,
                             **{"from": now_ms - 30 * 86_400_000, "to": now_ms})
            rows = data.get("dataList", []) if isinstance(data, dict) else data
            rows = sorted(rows, key=lambda r: r.get("timePoint", r.get("timepoint", 0)))
            return [float(r["fundingRate"]) for r in rows]

        md.funding = self._optional("funding", symbol, funding)
        time.sleep(self.cfg.request_delay)
        try:
            md.oi_change_pct = self._oi_change(symbol, contract)
        except GeoBlockedError:
            raise
        except Exception as exc:
            log.warning("kucoin/%s: open interest unavailable: %s", symbol, exc)
        return trim_open_candle(md, interval_minutes(iv))

    def _oi_change(self, symbol: str, contract: str) -> float | None:
        oi = float(self._get(f"/api/v1/contracts/{contract}")["openInterest"])
        now = time.time()
        hist: list[list[float]] = (self.state.data.setdefault("_oi", {}).setdefault(symbol, [])
                                   if self.state else [])
        if not hist or now - hist[-1][0] >= 60:
            hist.append([now, oi])
        hist[:] = [h for h in hist if h[0] >= now - 7 * 86400]
        step = interval_minutes(self.cfg.interval) * 60
        target = now - self.cfg.trend_lookback * step
        past = [h for h in hist if h[0] <= target]
        if not past:
            log.info("kucoin/%s: OI history warming up (%d snapshots, need %.1fh)",
                     symbol, len(hist), self.cfg.trend_lookback * step / 3600)
            return None
        ref = past[-1]
        if target - ref[0] > step or ref[1] == 0:
            log.warning("kucoin/%s: OI snapshot gap too large, skipping factor", symbol)
            return None
        chg = (oi / ref[1] - 1) * 100
        log.info("kucoin/%s: OI now=%.0f ref=%.0f (%.1fh ago) -> %+.2f%%",
                 symbol, oi, ref[1], (now - ref[0]) / 3600, chg)
        return chg


PROVIDERS: dict[str, type[Provider]] = {"kucoin": KuCoinProvider, "binance": BinanceProvider,
                                        "bybit": BybitProvider}


def fetch_market(symbol: str, providers: list[Provider]) -> MarketData | None:
    for prov in providers:
        try:
            log.info("%s: fetching from %s", symbol, prov.name)
            return prov.fetch(symbol)
        except GeoBlockedError as exc:
            log.error("%s: %s blocked this IP (%s). Common on GitHub-hosted runners; "
                      "use another provider, a VPS/self-hosted runner or HTTPS_PROXY.",
                      symbol, prov.name, exc)
        except Exception as exc:
            log.error("%s: provider %s failed: %s", symbol, prov.name, exc, exc_info=True)
    return None


# --------------------------------------------------------------------------- #
# Math helpers
# --------------------------------------------------------------------------- #
def clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def rolling_zscore(values: list[float], window: int) -> float:
    """Z-score against a rolling window of CLOSED candles, not full history."""
    w = values[-window:] if window > 0 else values
    if len(w) < 10:
        return 0.0
    sd = statistics.pstdev(w)
    return 0.0 if sd == 0 else (w[-1] - statistics.fmean(w)) / sd


def pct_change(values: list[float], lookback: int) -> float:
    if len(values) <= lookback or values[-1 - lookback] == 0:
        return 0.0
    return (values[-1] / values[-1 - lookback] - 1) * 100


def interval_minutes(interval: str) -> int:
    return int(interval[:-1]) * {"m": 1, "h": 60, "d": 1440}[interval[-1]]


def ema(values: list[float], period: int) -> float:
    k, e = 2 / (period + 1), values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e


def _rma(seq: list[float], period: int) -> list[float]:
    """Wilder's smoothing (used by ATR/ADX)."""
    a = statistics.fmean(seq[:period])
    out = [a]
    for x in seq[period:]:
        a += (x - a) / period
        out.append(a)
    return out


def true_ranges(high: list[float], low: list[float], close: list[float]) -> list[float]:
    trs = []
    for i in range(1, min(len(high), len(low), len(close))):
        trs.append(max(high[i] - low[i],
                       abs(high[i] - close[i - 1]),
                       abs(low[i] - close[i - 1])))
    return trs


def atr_wilder(high: list[float], low: list[float], close: list[float], period: int) -> float | None:
    trs = true_ranges(high, low, close)
    if len(trs) < period:
        return None
    return _rma(trs, period)[-1]


def adx_wilder(high: list[float], low: list[float], close: list[float], period: int) -> float | None:
    """Wilder ADX. Regime filter: >= ~20 trending, < 20 ranging."""
    n = min(len(high), len(low), len(close))
    if n < period * 2 + 2:
        return None
    plus_dm, minus_dm = [], []
    for i in range(1, n):
        up, dn = high[i] - high[i - 1], low[i - 1] - low[i]
        plus_dm.append(up if (up > dn and up > 0) else 0.0)
        minus_dm.append(dn if (dn > up and dn > 0) else 0.0)
    trs = true_ranges(high, low, close)
    m = min(len(trs), len(plus_dm), len(minus_dm))
    atr_s = _rma(trs[:m], period)
    pdi_s = _rma(plus_dm[:m], period)
    mdi_s = _rma(minus_dm[:m], period)
    k = min(len(atr_s), len(pdi_s), len(mdi_s))
    dx = []
    for i in range(k):
        if atr_s[i] == 0:
            dx.append(0.0)
            continue
        pdi, mdi = 100 * pdi_s[i] / atr_s[i], 100 * mdi_s[i] / atr_s[i]
        s = pdi + mdi
        dx.append(0.0 if s == 0 else 100 * abs(pdi - mdi) / s)
    if len(dx) < period:
        return None
    a = statistics.fmean(dx[:period])
    for x in dx[period:]:
        a += (x - a) / period
    return a


# --------------------------------------------------------------------------- #
# Strategy
# --------------------------------------------------------------------------- #
@dataclass
class Component:
    name: str
    score: float
    weight: float
    detail: str


@dataclass
class Signal:
    symbol: str
    side: str
    score: float
    entry: float
    stop: float
    tp1: float
    tp2: float
    atr: float
    source: str
    components: list[Component]
    qty: float = 0.0
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def score_funding(v: list[float], cfg: Config) -> Component:
    """Contrarian at ABSOLUTE extremes: crowded longs (high positive funding)
    => bearish. Below the extreme threshold the score stays small."""
    f_now = v[-1]
    score = clip(-f_now / cfg.funding_extreme) if cfg.funding_extreme > 0 else 0.0
    z = rolling_zscore(v, cfg.rolling_z_window)
    return Component("Funding", score, cfg.w_funding,
                     f"last={f_now * 100:.4f}% (extreme=±{cfg.funding_extreme * 100:.3f}%) z={z:+.2f}")


def score_oi(md: MarketData, cfg: Config) -> Component:
    n = cfg.trend_lookback
    oi_chg = md.oi_change_pct if md.oi_change_pct is not None else pct_change(md.oi or [], n)
    px_chg = pct_change(md.close, n)
    score = 0.0
    if abs(oi_chg) >= cfg.oi_change_threshold_pct and px_chg != 0:
        strength = clip(abs(oi_chg) / (cfg.oi_change_threshold_pct * 3), 0, 1)
        direction = 1 if px_chg > 0 else -1
        # OI up with price => trend confirmed; OI down => weak move (fade slightly)
        score = direction * strength if oi_chg > 0 else -direction * strength * 0.4
    return Component("OpenInterest", clip(score), cfg.w_oi,
                     f"OI {oi_chg:+.2f}% / price {px_chg:+.2f}% over {n} candles")


def score_ls(v: list[float], cfg: Config) -> Component:
    z = rolling_zscore(v, cfg.rolling_z_window)
    return Component("LongShortRatio", clip(-z / 2), cfg.w_ls,
                     f"ratio={v[-1]:.3f} rolling-z={z:+.2f} (contrarian)")


def score_flow(v: list[float], cfg: Config) -> Component:
    z = rolling_zscore(v, cfg.rolling_z_window)
    return Component("TakerFlow", clip(z / 2), cfg.w_flow,
                     f"buy/sell={v[-1]:.3f} rolling-z={z:+.2f} (momentum)")


def score_trend(close: list[float], cfg: Config) -> Component:
    if len(close) < cfg.ema_slow:
        raise ValueError(f"need {cfg.ema_slow} candles, have {len(close)}")
    diff = (ema(close, cfg.ema_fast) / ema(close, cfg.ema_slow) - 1) * 100
    return Component("Trend", clip(diff / 1.5), cfg.w_trend,
                     f"EMA{cfg.ema_fast}/EMA{cfg.ema_slow} diff={diff:+.2f}%")


def evaluate(symbol: str, providers: list[Provider], cfg: Config) -> Signal | None:
    log.info("=== Evaluating %s (%s) ===", symbol, cfg.interval)
    md = fetch_market(symbol, providers)
    if md is None:
        log.error("%s: no provider returned data, skipping", symbol)
        return None
    if len(md.close) < max(cfg.ema_slow, cfg.rolling_z_window, cfg.adx_period * 2 + 2):
        log.warning("%s: not enough closed candles (%d)", symbol, len(md.close))
        return None

    # ---- regime filter: tilt weights by ADX, renormalisation happens in the sum
    adx_val = adx_wilder(md.high, md.low, md.close, cfg.adx_period)
    trending = adx_val is not None and adx_val >= cfg.adx_trend_min
    regime = "TRENDING" if trending else "RANGING"
    log.info("%s | ADX=%s -> %s regime", symbol, f"{adx_val:.1f}" if adx_val else "n/a", regime)
    wmap = {
        "Funding":        cfg.w_funding * (0.75 if trending else 1.25),
        "OpenInterest":   cfg.w_oi,
        "LongShortRatio": cfg.w_ls * (0.75 if trending else 1.25),
        "TakerFlow":      cfg.w_flow * (0.75 if trending else 1.25),
        "Trend":          cfg.w_trend * (1.5 if trending else 0.5),
    }

    jobs: list[tuple[str, bool, Callable] = [
        ("funding", bool(md.funding), lambda: score_funding(md.funding or [], cfg)),
        ("open_interest", bool(md.oi) or md.oi_change_pct is not None, lambda: score_oi(md, cfg)),
        ("long_short", bool(md.ls), lambda: score_ls(md.ls or [], cfg)),
        ("taker_flow", bool(md.flow), lambda: score_flow(md.flow or [], cfg)),
        ("trend", True, lambda: score_trend(md.close, cfg)),
    ]
    comps: list[Component] = []
    for label, available, fn in jobs:
        if not available:
            log.info("%s: factor '%s' has no data from %s - skipped", symbol, label, md.source)
            continue
        try:
            c = fn()
            c.weight = wmap.get(c.name, c.weight)
            comps.append(c)
            log.info("%s | %-14s score=%+.3f w=%.2f | %s", symbol, c.name, c.score, c.weight, c.detail)
        except Exception as exc:
            log.error("%s: factor '%s' failed: %s", symbol, label, exc, exc_info=True)

    if len(comps) < 2:
        log.warning("%s: fewer than 2 factors available, skipping", symbol)
        return None

    total = sum(c.score * c.weight for c in comps) / sum(c.weight for c in comps)
    log.info("%s | TOTAL score=%+.3f (threshold ±%.2f) from %d factors [%s]",
             symbol, total, cfg.signal_threshold, len(comps), regime)
    if abs(total) < cfg.signal_threshold:
        log.info("%s: no signal", symbol)
        return None

    a = atr_wilder(md.high, md.low, md.close, cfg.atr_period)
    if not a:
        log.warning("%s: ATR unavailable", symbol)
        return None
    side = "LONG" if total > 0 else "SHORT"
    d, entry = (1 if side == "LONG" else -1), md.close[-1]
    stop = entry - d * a * cfg.sl_atr_mult
    sig = Signal(symbol, side, total, entry, stop,
                 entry + d * a * cfg.tp1_atr_mult,
                 entry + d * a * cfg.tp2_atr_mult, a, md.source, comps)
    # ---- risk-based position sizing: risk RISK_PCT% of CAPITAL per trade
    if cfg.capital_usdt > 0 and abs(entry - stop) > 0:
        sig.qty = (cfg.capital_usdt * cfg.risk_pct / 100) / abs(entry - stop)
    return sig


# --------------------------------------------------------------------------- #
# State (cooldown) & notifications
# --------------------------------------------------------------------------- #
class State:
    def __init__(self, path: str):
        self.path = Path(path)
        self.data: dict[str, dict] = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
                log.info("Loaded state for %d symbols", len(self.data))
            except Exception as exc:
                log.error("Could not read state file, starting fresh: %s", exc)

    def in_cooldown(self, symbol: str, side: str, hours: float, flip_factor: float) -> bool:
        """Same side: full window. Opposite side (flip): short window, so a real
        reversal is not blocked forever but signal-flicker is still prevented."""
        last = self.data.get(symbol)
        if not last:
            return False
        elapsed = time.time() - last["ts"]
        limit = hours * 3600 if last["side"] == side else hours * 3600 * flip_factor
        return elapsed < limit

    def record(self, symbol: str, side: str) -> None:
        self.data[symbol] = {"side": side, "ts": time.time()}
        self.save()

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        tmp.replace(self.path)


def format_message(sig: Signal, cfg: Config) -> str:
    icon = "🟢" if sig.side == "LONG" else "🔴"
    rr1 = abs(sig.tp1 - sig.entry) / abs(sig.entry - sig.stop) if sig.entry != sig.stop else 0
    lines = [
        f"{icon} <b>سیگنال {sig.side} — {html.escape(sig.symbol)}</b>",
        f"منبع: {sig.source} | تایم‌فریم: {cfg.interval}",
        f"امتیاز: <b>{sig.score:+.2f}</b>",
        "",
        f"ورود: <code>{sig.entry:,.4f}</code>",
        f"حد ضرر: <code>{sig.stop:,.4f}</code>",
        f"هدف ۱ (RR≈{rr1:.1f}): <code>{sig.tp1:,.4f}</code>",
        f"هدف ۲: <code>{sig.tp2:,.4f}</code>",
        f"ATR: {sig.atr:,.4f}",
    ]
    if sig.qty > 0:
        notional = sig.qty * sig.entry
        fee = notional * cfg.taker_fee * 2
        lines += [
            "",
            f"حجم پیشنهادی (ریسک {cfg.risk_pct:g}% از {cfg.capital_usdt:,.0f}): "
            f"<code>{sig.qty:,.6f}</code> (~{notional:,.1f} USDT)",
            f"کارمزد تقریبی رفت‌وبرگشت (taker×۲): ~{fee:,.2f} USDT",
        ]
    lines += [
        "",
        "<b>فاکتورها:</b>",
    ]
    lines += [f"• {c.name}: {c.score:+.2f} — {html.escape(c.detail)}" for c in sig.components]
    lines += ["", "⚠️ توصیه مالی نیست؛ مدیریت ریسک با خودتان."]
    return "\n".join(lines)


def send_telegram(text: str, cfg: Config, dry_run: bool) -> bool:
    if dry_run or not (cfg.tg_token and cfg.tg_chat_id):
        log.info("[DRY-RUN / no Telegram creds] message:\n%s", text)
        return True
    url = f"https://api.telegram.org/bot{cfg.tg_token}/sendMessage"
    payload = {"chat_id": cfg.tg_chat_id, "text": text, "parse_mode": "HTML",
               "disable_web_page_preview": True}
    for attempt in range(1, 4):
        try:
            r = requests.post(url, json=payload, timeout=cfg.timeout)
            if r.ok:
                log.info("Telegram message sent")
                return True
            log.warning("Telegram HTTP %s (attempt %d): %s", r.status_code, attempt, r.text[:200])
        except requests.RequestException as exc:
            log.warning("Telegram error (attempt %d): %s", attempt, exc)
        time.sleep(2 * attempt)
    log.error("Telegram delivery failed after retries")
    return False


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_cycle(providers: list[Provider], state: State, cfg: Config, dry_run: bool) -> None:
    started, sent = time.monotonic(), 0
    for symbol in cfg.symbols:
        try:
            sig = evaluate(symbol, providers, cfg)
            if not sig:
                continue
            if state.in_cooldown(symbol, sig.side, cfg.cooldown_hours, cfg.flip_cooldown_factor):
                log.info("%s: %s signal suppressed (cooldown %.1fh, flip factor %.2f)",
                         symbol, sig.side, cfg.cooldown_hours, cfg.flip_cooldown_factor)
                continue
            if send_telegram(format_message(sig, cfg), cfg, dry_run):
                state.record(symbol, sig.side)
                sent += 1
        except Exception as exc:
            log.error("%s: cycle failed: %s", symbol, exc, exc_info=True)
    state.save()
    log.info("Cycle finished in %.1fs, %d signal(s) sent", time.monotonic() - started, sent)


def main() -> int:
    ap = argparse.ArgumentParser(description="Free derivatives signal bot v2")
    ap.add_argument("--once", action="store_true", help="run a single cycle and exit")
    ap.add_argument("--dry-run", action="store_true", help="log signals instead of sending to Telegram")
    ap.add_argument("--symbols", help="comma-separated override, e.g. BTCUSDT,SOLUSDT")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    load_dotenv()
    setup_logging(args.log_level, _env("LOG_DIR", "logs"))  # before anything can fail
    cfg = Config.from_env()
    if args.symbols:
        cfg.symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    unknown = [p for p in cfg.providers if p not in PROVIDERS]
    if unknown:
        log.error("Unknown provider(s): %s (available: %s)", unknown, list(PROVIDERS))
        return 2
    log.info("Starting bot v2 | symbols=%s providers=%s interval=%s dry_run=%s capital=%.0f risk=%.2f%%",
             cfg.symbols, cfg.providers, cfg.interval, args.dry_run, cfg.capital_usdt, cfg.risk_pct)

    stop = threading.Event()
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, lambda *_: (log.info("Shutdown requested"), stop.set()))

    http, state = Http(cfg), State(cfg.state_file)
    providers = [PROVIDERS[p](cfg, http, state) for p in cfg.providers]
    while not stop.is_set():
        run_cycle(providers, state, cfg, args.dry_run)
        if args.once:
            break
        log.info("Sleeping %ss until next cycle", cfg.poll_seconds)
        stop.wait(cfg.poll_seconds)
    log.info("Bot stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())