#!/usr/bin/env python3
"""
CoinGlass Derivatives Signal Bot
================================
Pulls futures-market data from the CoinGlass API (v4), scores it with a
multi-factor derivatives strategy and pushes LONG / SHORT signals to Telegram.

Factors (each scored in [-1, +1], positive = bullish):
  1. Funding rate       - contrarian (crowded longs => bearish, crowded shorts => bullish)
  2. Open interest      - trend confirmation (price and OI rising/falling together)
  3. Long/Short ratio   - contrarian on account positioning extremes
  4. Liquidations       - contrarian capitulation (long-liq flush => bullish, short squeeze => bearish)

SL / TP levels are ATR based. NOT financial advice - backtest before trading.
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

ENDPOINTS = {
    "price": "/api/futures/price/history",
    "funding": "/api/futures/funding-rate/history",
    "oi": "/api/futures/open-interest/history",
    "ls": "/api/futures/global-long-short-account-ratio/history",
    "liq": "/api/futures/liquidation/history",
}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


@dataclass
class Config:
    api_key: str
    tg_token: str
    tg_chat_id: str
    base_url: str = "https://open-api-v4.coinglass.com"
    exchange: str = "Binance"
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    interval: str = "h4"
    limit: int = 100
    poll_seconds: int = 900
    request_delay: float = 0.6
    timeout: int = 15
    max_retries: int = 4
    # strategy
    signal_threshold: float = 0.45
    w_funding: float = 0.25
    w_oi: float = 0.30
    w_ls: float = 0.20
    w_liq: float = 0.25
    oi_change_threshold_pct: float = 2.0
    trend_lookback: int = 6
    atr_period: int = 14
    sl_atr_mult: float = 1.5
    tp1_atr_mult: float = 1.5
    tp2_atr_mult: float = 3.0
    cooldown_hours: float = 8.0
    state_file: str = "state.json"
    log_dir: str = "logs"

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        api_key = _env("COINGLASS_API_KEY", "")
        if not api_key:
            raise SystemExit("COINGLASS_API_KEY is missing (see .env.example)")
        return cls(
            api_key=api_key,
            tg_token=_env("TELEGRAM_BOT_TOKEN", ""),
            tg_chat_id=_env("TELEGRAM_CHAT_ID", ""),
            base_url=_env("COINGLASS_BASE_URL", cls.base_url),
            exchange=_env("EXCHANGE", cls.exchange),
            symbols=[s.strip().upper() for s in _env("SYMBOLS", "BTCUSDT,ETHUSDT").split(",") if s.strip()],
            interval=_env("INTERVAL", cls.interval),
            limit=int(_env("LIMIT", str(cls.limit))),
            poll_seconds=int(_env("POLL_SECONDS", str(cls.poll_seconds))),
            signal_threshold=float(_env("SIGNAL_THRESHOLD", str(cls.signal_threshold))),
            cooldown_hours=float(_env("COOLDOWN_HOURS", str(cls.cooldown_hours))),
            sl_atr_mult=float(_env("SL_ATR_MULT", str(cls.sl_atr_mult))),
            tp1_atr_mult=float(_env("TP1_ATR_MULT", str(cls.tp1_atr_mult))),
            tp2_atr_mult=float(_env("TP2_ATR_MULT", str(cls.tp2_atr_mult))),
            log_dir=_env("LOG_DIR", cls.log_dir),
            state_file=_env("STATE_FILE", cls.state_file),
        )


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(level: str, log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s"
    )
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
# CoinGlass client
# --------------------------------------------------------------------------- #
class CoinGlassError(Exception):
    pass


class CoinGlassClient:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({"accept": "application/json", "CG-API-KEY": cfg.api_key})

    def get(self, path: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        url = self.cfg.base_url.rstrip("/") + path
        last_err: Exception | None = None
        for attempt in range(1, self.cfg.max_retries + 1):
            started = time.monotonic()
            try:
                resp = self.session.get(url, params=params, timeout=self.cfg.timeout)
                elapsed = (time.monotonic() - started) * 1000
                log.debug("GET %s params=%s -> %s in %.0fms", path, params, resp.status_code, elapsed)
                if resp.status_code == 429:
                    wait = min(2 ** attempt * 2, 60)
                    log.warning("Rate limited (429) on %s, sleeping %ss (attempt %d/%d)",
                                path, wait, attempt, self.cfg.max_retries)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                body = resp.json()
                if str(body.get("code")) != "0":
                    raise CoinGlassError(f"API error code={body.get('code')} msg={body.get('msg')}")
                data = body.get("data") or []
                log.info("Fetched %d rows from %s", len(data), path)
                return data
            except (requests.RequestException, ValueError, CoinGlassError) as exc:
                last_err = exc
                wait = min(2 ** attempt, 30)
                log.warning("Request failed for %s (attempt %d/%d): %s - retry in %ss",
                            path, attempt, self.cfg.max_retries, exc, wait)
                time.sleep(wait)
        raise CoinGlassError(f"Giving up on {path}: {last_err}")

    def history(self, kind: str, symbol: str) -> list[dict[str, Any]]:
        params = {"exchange": self.cfg.exchange, "symbol": symbol,
                  "interval": self.cfg.interval, "limit": self.cfg.limit}
        rows = self.get(ENDPOINTS[kind], params)
        return sorted(rows, key=lambda r: r.get("time", 0))


# --------------------------------------------------------------------------- #
# Math helpers
# --------------------------------------------------------------------------- #
def col(rows: list[dict], *keys: str) -> list[float]:
    """Extract the first available key from each row as float."""
    out: list[float] = []
    for r in rows:
        for k in keys:
            if k in r and r[k] not in (None, ""):
                out.append(float(r[k]))
                break
    return out


def clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def zscore(values: list[float]) -> float:
    if len(values) < 10:
        return 0.0
    sd = statistics.pstdev(values)
    return 0.0 if sd == 0 else (values[-1] - statistics.fmean(values)) / sd


def pct_change(values: list[float], lookback: int) -> float:
    if len(values) <= lookback or values[-1 - lookback] == 0:
        return 0.0
    return (values[-1] / values[-1 - lookback] - 1) * 100


def atr(high: list[float], low: list[float], close: list[float], period: int) -> float | None:
    n = min(len(high), len(low), len(close))
    if n <= period:
        return None
    trs = [max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
           for i in range(1, n)]
    return statistics.fmean(trs[-period:])


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
    components: list[Component]
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def score_funding(rows: list[dict], cfg: Config) -> Component:
    vals = col(rows, "close", "funding_rate")
    z = zscore(vals)
    return Component("Funding", clip(-z / 2), cfg.w_funding,
                     f"last={vals[-1]:.4f} z={z:+.2f} (contrarian)")


def score_oi(oi_rows: list[dict], price: list[float], cfg: Config) -> Component:
    oi = col(oi_rows, "close")
    n = cfg.trend_lookback
    oi_chg, px_chg = pct_change(oi, n), pct_change(price, n)
    score = 0.0
    if abs(oi_chg) >= cfg.oi_change_threshold_pct and px_chg != 0:
        strength = clip(abs(oi_chg) / (cfg.oi_change_threshold_pct * 3), 0, 1)
        direction = 1 if px_chg > 0 else -1
        # OI rising with price => trend confirmed; OI falling => move is weak (fade it slightly)
        score = direction * strength if oi_chg > 0 else -direction * strength * 0.4
    return Component("OpenInterest", clip(score), cfg.w_oi,
                     f"OI {oi_chg:+.2f}% / price {px_chg:+.2f}% over {n} candles")


def score_ls(rows: list[dict], cfg: Config) -> Component:
    ratios = col(rows, "global_account_long_short_ratio", "long_short_ratio")
    if not ratios:
        longs = col(rows, "global_account_long_percent")
        shorts = col(rows, "global_account_short_percent")
        ratios = [l / s for l, s in zip(longs, shorts) if s]
    z = zscore(ratios)
    return Component("LongShortRatio", clip(-z / 2), cfg.w_ls,
                     f"ratio={ratios[-1]:.3f} z={z:+.2f} (contrarian)")


def score_liq(rows: list[dict], cfg: Config) -> Component:
    longs = col(rows, "long_liquidation_usd")
    shorts = col(rows, "short_liquidation_usd")
    total = [a + b for a, b in zip(longs, shorts)]
    recent = 2
    l_rec, s_rec = sum(longs[-recent:]), sum(shorts[-recent:])
    tot_rec = l_rec + s_rec
    avg = statistics.fmean(total[:-recent]) * recent if len(total) > recent + 5 else 0
    spike = tot_rec / avg if avg else 0.0
    score = 0.0
    if tot_rec > 0 and spike >= 1.5:
        imbalance = (l_rec - s_rec) / tot_rec  # >0 => longs flushed => bullish (contrarian)
        score = imbalance * clip(spike / 4, 0, 1)
    return Component("Liquidations", clip(score), cfg.w_liq,
                     f"long=${l_rec:,.0f} short=${s_rec:,.0f} spike x{spike:.2f}")


def evaluate(symbol: str, client: CoinGlassClient, cfg: Config) -> Signal | None:
    log.info("=== Evaluating %s on %s (%s) ===", symbol, cfg.exchange, cfg.interval)
    price_rows = client.history("price", symbol)
    close = col(price_rows, "close")
    high, low = col(price_rows, "high"), col(price_rows, "low")
    if len(close) < cfg.atr_period + 2:
        log.warning("%s: not enough price data (%d rows)", symbol, len(close))
        return None

    scorers: list[tuple[str, Callable[[list[dict]], Component]]] = [
        ("funding", lambda r: score_funding(r, cfg)),
        ("oi", lambda r: score_oi(r, close, cfg)),
        ("ls", lambda r: score_ls(r, cfg)),
        ("liq", lambda r: score_liq(r, cfg)),
    ]
    components: list[Component] = []
    for kind, fn in scorers:
        time.sleep(cfg.request_delay)
        try:
            comp = fn(client.history(kind, symbol))
            components.append(comp)
            log.info("%s | %-14s score=%+.3f w=%.2f | %s", symbol, comp.name,
                     comp.score, comp.weight, comp.detail)
        except Exception as exc:  # one failing factor must not kill the run
            log.error("%s: factor '%s' failed and is skipped: %s", symbol, kind, exc, exc_info=True)

    if len(components) < 2:
        log.warning("%s: fewer than 2 factors available, skipping", symbol)
        return None

    wsum = sum(c.weight for c in components)
    total = sum(c.score * c.weight for c in components) / wsum
    log.info("%s | TOTAL score=%+.3f (threshold ±%.2f)", symbol, total, cfg.signal_threshold)

    if abs(total) < cfg.signal_threshold:
        log.info("%s: no signal", symbol)
        return None

    a = atr(high, low, close, cfg.atr_period)
    if not a:
        log.warning("%s: ATR unavailable", symbol)
        return None
    entry, side = close[-1], "LONG" if total > 0 else "SHORT"
    d = 1 if side == "LONG" else -1
    return Signal(symbol, side, total, entry,
                  entry - d * a * cfg.sl_atr_mult,
                  entry + d * a * cfg.tp1_atr_mult,
                  entry + d * a * cfg.tp2_atr_mult, a, components)


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

    def in_cooldown(self, symbol: str, side: str, hours: float) -> bool:
        last = self.data.get(symbol)
        return bool(last and last["side"] == side and time.time() - last["ts"] < hours * 3600)

    def record(self, symbol: str, side: str) -> None:
        self.data[symbol] = {"side": side, "ts": time.time()}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        tmp.replace(self.path)


def format_message(sig: Signal, cfg: Config) -> str:
    icon = "🟢" if sig.side == "LONG" else "🔴"
    lines = [
        f"{icon} <b>سیگنال {sig.side} — {html.escape(sig.symbol)}</b>",
        f"صرافی: {cfg.exchange} | تایم‌فریم: {cfg.interval}",
        f"امتیاز: <b>{sig.score:+.2f}</b>",
        "",
        f"ورود: <code>{sig.entry:,.4f}</code>",
        f"حد ضرر: <code>{sig.stop:,.4f}</code>",
        f"هدف ۱: <code>{sig.tp1:,.4f}</code>",
        f"هدف ۲: <code>{sig.tp2:,.4f}</code>",
        f"ATR: {sig.atr:,.4f}",
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
def run_cycle(client: CoinGlassClient, state: State, cfg: Config, dry_run: bool) -> None:
    started = time.monotonic()
    sent = 0
    for symbol in cfg.symbols:
        try:
            sig = evaluate(symbol, client, cfg)
            if not sig:
                continue
            if state.in_cooldown(symbol, sig.side, cfg.cooldown_hours):
                log.info("%s: %s signal suppressed (cooldown %.1fh)", symbol, sig.side, cfg.cooldown_hours)
                continue
            if send_telegram(format_message(sig, cfg), cfg, dry_run):
                state.record(symbol, sig.side)
                sent += 1
        except Exception as exc:
            log.error("%s: cycle failed: %s", symbol, exc, exc_info=True)
    log.info("Cycle finished in %.1fs, %d signal(s) sent", time.monotonic() - started, sent)


def main() -> int:
    ap = argparse.ArgumentParser(description="CoinGlass derivatives signal bot")
    ap.add_argument("--once", action="store_true", help="run a single cycle and exit")
    ap.add_argument("--dry-run", action="store_true", help="log signals instead of sending to Telegram")
    ap.add_argument("--symbols", help="comma-separated override, e.g. BTCUSDT,SOLUSDT")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    cfg = Config.from_env()
    if args.symbols:
        cfg.symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    setup_logging(args.log_level, cfg.log_dir)
    log.info("Starting bot | symbols=%s exchange=%s interval=%s dry_run=%s",
             cfg.symbols, cfg.exchange, cfg.interval, args.dry_run)

    stop = threading.Event()
    for sig_ in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig_, lambda *_: (log.info("Shutdown requested"), stop.set()))

    client, state = CoinGlassClient(cfg), State(cfg.state_file)
    while not stop.is_set():
        run_cycle(client, state, cfg, args.dry_run)
        if args.once:
            break
        log.info("Sleeping %ss until next cycle", cfg.poll_seconds)
        stop.wait(cfg.poll_seconds)
    log.info("Bot stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
