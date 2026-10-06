"""Market-move alerts and the new Decision Summary lines (2026-10-06)."""

import json

import numpy as np
import pandas as pd

from jobs import moves
from reports import asset_report

NOW = pd.Timestamp("2026-10-06 08:00", tz="UTC")


def _candles(last_jump: float = 0.0, days: int = 30, base: float = 100.0):
    """Quiet 15-minute candles (daily range ≈ 2%), with today's price moved by last_jump %."""
    ts = pd.date_range(NOW - pd.Timedelta(days=days), NOW, freq="15min")
    close = np.full(len(ts), base)
    close += np.sin(np.arange(len(ts)) / 4)  # ±1 wiggle → ~2% daily range
    close[ts >= NOW.normalize()] = base * (1 + last_jump / 100)  # today's candles only, so it breaks the prior days
    m15 = pd.DataFrame({"ts": ts, "open": close, "high": close + 0.1, "low": close - 0.1, "close": close})
    return moves.daily_from_15m(m15), m15


def test_quiet_market_raises_nothing():
    st = moves.stats(*_candles(0.0), NOW)
    assert st and 1 < st["atr_pct"] < 4 and abs(st["chg24"]) < st["atr_pct"]
    assert moves.events(st) == []
    assert "در حد معمول" in moves.move_line(st)


def test_big_drop_is_a_move_and_a_break():
    st = moves.stats(*_candles(-8.0), NOW)
    ev = moves.events(st)
    assert "move_down" in ev and "break_low" in ev
    line = moves.move_line(st)
    assert line.startswith("• حرکت بازار: ۲۴ ساعت") and "حرکت خیلی بزرگ" in line and "کف ۲۰ روزه" in line
    msg = moves.alert("BTC", st, ev)
    assert "حرکت مهم بازار — بیت‌کوین" in msg and "🔴" in msg and "نه سیگنال معاملاتی" in msg


def test_run_alerts_once_per_day(tmp_path, monkeypatch):
    daily, m15 = _candles(+9.0)
    sent = []
    from notify import telegram

    monkeypatch.setattr(telegram, "send", lambda text, kind, dry_run: sent.append(text) or
                        type("R", (), {"sent": True, "where": ""})())
    fetch = lambda sym, interval, limit: daily if interval == "1d" else m15  # noqa: E731
    assert moves.run(tmp_path, NOW, False, fetch) == 2  # BTC and gold both "moved"
    assert moves.run(tmp_path, NOW + pd.Timedelta(minutes=15), False, fetch) == 0
    assert len(sent) == 2 and "🟢" in sent[0]
    assert all(k.endswith(":2026-10-06") for k in json.loads((tmp_path / "moves_alerted.json").read_text()))


def test_run_survives_a_venue_error(tmp_path):
    def boom(*a):
        raise RuntimeError("451 from venue")

    assert moves.run(tmp_path, NOW, True, boom) == 0


def test_change_arrow_and_impact_cell():
    assert asset_report.change_arrow("+0.3") == "↑" and asset_report.change_arrow("−1.2") == "↓"
    assert asset_report.change_arrow("+0.0") == " " and asset_report.change_arrow("—") == " "
    row = asset_report.Row(key="x", name_fa="x", ticker="X", value="1", change="", surprise="—", coef=-4.0,
                           coef_tag="w", effect=-1, conf="", x=1.0, against=True)
    cell = asset_report.impact_cell(row)
    assert cell.startswith("▼") and cell.endswith("؟")


def test_against_theory_flags_only_real_contradictions(monkeypatch):
    monkeypatch.setattr(asset_report, "theory_sign", lambda k, a, s: -1)
    assert asset_report.against_theory("real_yield", "BTC", +3.0, "state")
    assert not asset_report.against_theory("real_yield", "BTC", -3.0, "state")
    assert not asset_report.against_theory("real_yield", "BTC", +0.5, "state")  # too weak to matter


def test_technical_macro_and_direction_lines():
    line, d = asset_report.technical_line({"trend": "down", "last_event": "BOS_DN", "rsi": 28})
    assert d == -1 and "روند روزانه نزولی" in line and "شکست کف قبلی" in line and "اشباع فروش" in line
    assert asset_report.technical_line({}) == (None, 0)
    m = {"medium": {"score": 12, "top": [{"indicator": "cpi", "contribution": 6, "coef": 2.0},
                                         {"indicator": "dxy", "contribution": -3, "coef": -2.0}]}}
    ml = asset_report.macro_line(m, "BTC")
    assert "به نفع افزایش" in ml and "به نفع کاهش" in ml
    assert "خلاف هم" in asset_report.direction_line(-1, 30, 5)
    assert "هم‌جهت" in asset_report.direction_line(1, 30, 5)
