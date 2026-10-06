"""Market-move alerts: tell the reader when Bitcoin or gold moved a lot, even when the macro picture is neutral.

    python -m jobs.moves --state state [--dry-run]

Runs in the 15-minute intraday job. Prices come straight from the exchanges (BTC, and PAXG as the 24/7 gold proxy),
so the check needs no snapshot and no API key. An alert fires, once per asset, kind and Tehran day, when
- the last 24 hours moved at least MOVE_ATR × the typical daily range (ATR 14), or
- the price broke the high or low of the previous BREAK_DAYS daily candles.
The same numbers feed the "حرکت بازار" line of each report's Decision Summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ASSETS = (("BTC", "BTC"), ("Gold", "PAXG"))   # report asset → exchange symbol
MOVE_ATR = 1.5
BREAK_DAYS = 20
ATR_DAYS = 14


def daily_from_15m(m15: pd.DataFrame) -> pd.DataFrame:
    d = m15.set_index(pd.to_datetime(m15["ts"], utc=True))
    return d.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna().reset_index()


def stats(daily: pd.DataFrame, m15: pd.DataFrame, now: pd.Timestamp) -> dict | None:
    """Price, 24 h and 7-day change, typical daily range (ATR % of price) and the previous 20-day high/low."""
    m15 = m15.assign(ts=pd.to_datetime(m15["ts"], utc=True)).sort_values("ts")
    m15 = m15[m15["ts"] <= now]
    daily = daily.assign(ts=pd.to_datetime(daily["ts"], utc=True)).sort_values("ts")
    if m15.empty or len(daily) < ATR_DAYS + 2:
        return None
    last_ts, price = m15["ts"].iloc[-1], float(m15["close"].iloc[-1])
    ago = m15[m15["ts"] <= last_ts - pd.Timedelta(hours=24)]
    p24 = float(ago["close"].iloc[-1]) if not ago.empty else None
    done = daily[daily["ts"] < last_ts.normalize()]  # completed days only
    if len(done) < ATR_DAYS + 1:
        return None
    prev_close = done["close"].shift(1)
    tr = pd.concat([done["high"] - done["low"], (done["high"] - prev_close).abs(), (done["low"] - prev_close).abs()],
                   axis=1).max(axis=1)
    atr_pct = float(tr.tail(ATR_DAYS).mean() / done["close"].iloc[-1] * 100)
    week = done[done["ts"] <= last_ts - pd.Timedelta(days=7)]
    window = done.tail(BREAK_DAYS)
    return {"asof": last_ts.isoformat(), "price": price,
            "chg24": (price / p24 - 1) * 100 if p24 else None,
            "chg7d": (price / float(week["close"].iloc[-1]) - 1) * 100 if not week.empty else None,
            "atr_pct": atr_pct, "hi": float(window["high"].max()), "lo": float(window["low"].min())}


def events(st: dict) -> list[str]:
    out = []
    if st.get("chg24") is not None and abs(st["chg24"]) >= MOVE_ATR * st["atr_pct"]:
        out.append("move_up" if st["chg24"] > 0 else "move_down")
    if st["price"] > st["hi"]:
        out.append("break_high")
    elif st["price"] < st["lo"]:
        out.append("break_low")
    return out


def size_word(chg: float | None, atr_pct: float) -> str:
    if chg is None or not atr_pct:
        return ""
    r = abs(chg) / atr_pct
    return "حرکت خیلی بزرگ" if r >= 2.5 else "حرکت بزرگ" if r >= MOVE_ATR else "بیشتر از معمول" if r >= 1 else "در حد معمول"


def _pct(v: float | None) -> str:
    from reports.messages import num

    return "—" if v is None else num(v, "{:+.1f}") + "٪"


def move_line(st: dict | None) -> str | None:
    """'• حرکت بازار: ۲۴ ساعت −4.1٪ (حرکت بزرگ) · ۷ روز −6.0٪ · شکست کف ۲۰ روزه' for the Decision Summary."""
    if not st:
        return None
    parts = [f"۲۴ ساعت {_pct(st['chg24'])} ({size_word(st['chg24'], st['atr_pct'])})"]
    if st.get("chg7d") is not None:
        parts.append(f"۷ روز {_pct(st['chg7d'])}")
    ev = events(st)
    if "break_high" in ev:
        parts.append("بالاتر از سقف ۲۰ روزه")
    if "break_low" in ev:
        parts.append("پایین‌تر از کف ۲۰ روزه")
    return "• حرکت بازار: " + " · ".join(parts)


def alert(asset: str, st: dict, ev: list[str]) -> str:
    from reports.messages import ASSET_FA, footer, num, tehran

    name = ASSET_FA[asset]
    px = num(st["price"], "{:,.0f}")
    what = []
    if {"move_up", "move_down"} & set(ev):
        what.append(f"{_pct(st['chg24'])} در ۲۴ ساعت — حدود {num(abs(st['chg24']) / st['atr_pct'], '{:.1f}')} برابر "
                    f"نوسان معمول یک روز ({num(st['atr_pct'], '{:.1f}')}٪)")
    if "break_high" in ev:
        what.append(f"عبور از سقف ۲۰ روزه ({num(st['hi'], '{:,.0f}')})")
    if "break_low" in ev:
        what.append(f"شکست کف ۲۰ روزه ({num(st['lo'], '{:,.0f}')})")
    up = "move_up" in ev or "break_high" in ev
    L = [f"⚡ <b>حرکت مهم بازار — {name}</b> {'🟢' if up else '🔴'}",
         f"قیمت: {px} دلار{' (PAXG، نماینده‌ی طلا)' if asset == 'Gold' else ''} — {tehran(st['asof'])}",
         *[f"• {w}" for w in what]]
    if st.get("chg7d") is not None:
        L.append(f"• تغییر ۷ روز: {_pct(st['chg7d'])}")
    L.append("این هشدار فقط حرکت قیمت را گزارش می‌کند، نه سیگنال معاملاتی. علت احتمالی و اثر کلان در گزارش بعدی می‌آید.")
    return "\n".join(L) + footer()


def _load(state: Path | None) -> dict:
    p = state / "moves_alerted.json" if state else None
    return json.loads(p.read_text()) if p and p.exists() else {}


def run(state: Path | None, now: pd.Timestamp, dry_run: bool, fetch=None) -> int:
    from collectors import crypto
    from jobs import scheduler
    from notify import telegram

    fetch = fetch or (lambda sym, interval, limit: crypto.candles(sym, interval, limit).df)
    sent = {k: v for k, v in _load(state).items() if now - pd.Timestamp(v) < pd.Timedelta(days=3)}
    day, n = scheduler.tehran_date(now), 0
    for asset, sym in ASSETS:
        try:
            st = stats(fetch(sym, "1d", 40), fetch(sym, "15m", 120), now)
        except Exception as exc:  # noqa: BLE001 — a venue error must not stop the news scan
            print(f"::warning::moves {asset}: {type(exc).__name__}: {str(exc)[:200]}")
            continue
        if not st:
            continue
        ev = [e for e in events(st) if f"{asset}:{e}:{day}" not in sent]
        print(f"moves {asset}: 24h {st['chg24']}, atr {st['atr_pct']:.2f}% → {events(st) or 'none'}")
        if not ev:
            continue
        r = telegram.send(alert(asset, st, ev), kind="move", dry_run=dry_run)
        print(f"move alert {asset}: {'sent' if r.sent else 'dry run → ' + r.where}")
        sent.update({f"{asset}:{e}:{day}": now.isoformat() for e in ev})
        n += 1
    if state:
        state.mkdir(parents=True, exist_ok=True)
        (state / "moves_alerted.json").write_text(json.dumps(sent))
    return n


def snapshot_stats(root: Path, asset: str, now: pd.Timestamp) -> dict | None:
    """The same numbers from the snapshot's 15-minute candles (used by the reports)."""
    sym = dict(ASSETS)[asset]
    p = root / "cache" / f"candles_{sym}_15m.csv.gz"
    if not p.exists():
        return None
    m15 = pd.read_csv(p, usecols=["ts", "open", "high", "low", "close"])
    m15 = m15.assign(ts=pd.to_datetime(m15["ts"], utc=True))
    m15 = m15[m15["ts"] > now - pd.Timedelta(days=60)]
    return stats(daily_from_15m(m15), m15, now) if not m15.empty else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    run(Path(args.state) if args.state else None, pd.Timestamp.now(tz="UTC"), args.dry_run)
    return 0


if __name__ == "__main__":
    np.seterr(all="ignore")
    sys.exit(main())
