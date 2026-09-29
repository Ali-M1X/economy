"""Phase 5: Persian dashboard — translation helpers, chart helpers and a full render on a synthetic snapshot."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from core.fa import fa, label, ltr, word
from dashboard import charts, demo_snapshot
from dashboard.data import load

LRI, PDI = "⁦", "⁩"


# ── core.fa ──────────────────────────────────────────────────────────────

def test_fa_translates_engine_messages_and_isolates_numbers():
    out = fa("no demonstrated edge (OOS avg R -0.05 over 120 trades)")
    assert "برتری آماری" in out
    assert f"{LRI}-0.05{PDI}" in out and f"{LRI}120{PDI}" in out  # sign stays attached in RTL text


def test_fa_upcoming_event_uses_calendar_names_weekdays_and_isolated_dates():
    msg = "Upcoming: Consumer Price Index at Thu 2026-10-01 16:00 Tehran — volatility spike likely"
    out = fa(msg, {"Consumer Price Index": "شاخص قیمت مصرف‌کننده"})
    assert "شاخص قیمت مصرف‌کننده" in out and "پنجشنبه" in out
    assert f"{LRI}2026-10-01 16:00{PDI}" in out and "Thu" not in out


def test_fa_regime_trigger_and_words():
    out = fa("→ Recovery if the growth level crosses its historical average (now +0.49σ)")
    assert out.startswith("← بهبود") and f"{LRI}+0.49{PDI}" in out
    assert word("bullish (HH/HL)").startswith("صعودی") and word(None) == "—"
    assert label("a2_market_path") == "مسیر نرخ در آتی‌ها" and label("unknown_key") == "unknown_key"


def test_fa_passes_unknown_messages_through():
    assert fa("some brand-new engine message") == "some brand-new engine message"
    assert ltr("+1") == f"{LRI}+1{PDI}"


# ── charts ───────────────────────────────────────────────────────────────

def test_merge_levels_joins_close_levels_and_keeps_trade_style():
    levels = [{"price": 100.0, "label": "سقف سوئینگ روزانه", "dash": "dot"},
              {"price": 100.2, "label": "سقف سوئینگ ۴ساعته", "dash": "dot"},
              {"price": 100.3, "label": "هدف ۱", "dash": "solid", "color": "#0f0"},
              {"price": 120.0, "label": "حد ضرر", "dash": "solid", "color": "#f00"}]
    out = charts.merge_levels(levels, span=40.0)
    assert len(out) == 2
    assert out[0]["label"] == "سقف سوئینگ روزانه/۴ساعته · هدف ۱"
    assert out[0]["color"] == "#0f0" and out[0]["dash"] == "solid"
    assert levels[0]["label"] == "سقف سوئینگ روزانه"  # inputs are not mutated


def test_with_episodes_labels_each_name_once():
    import plotly.graph_objects as go

    ep = pd.DataFrame({"start": pd.to_datetime(["2022-10-26", "2023-05-18"]), "end": pd.to_datetime(["2022-12-16", "2023-06-08"]),
                       "resteepen_date": [None, None], "brief": [False, False],
                       "label": ["Inflation shock → longest inversion on record"] * 2})
    fig = charts.with_episodes(go.Figure(), ep, "light")
    texts = [a.text for a in fig.layout.annotations]
    assert texts == ["شوک تورمی ← طولانی‌ترین وارونگی ثبت‌شده"]


# ── full render ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def demo_dir(tmp_path_factory):
    return demo_snapshot.build(tmp_path_factory.mktemp("demo") / "output", now=pd.Timestamp("2026-09-29 12:00", tz="UTC"))


def test_demo_snapshot_loads(demo_dir):
    b = load(demo_dir)
    assert (demo_dir / "DEMO").exists()
    assert b.series("ust_10y") is not None and len(b.signals["assets"]) == 2
    assert b.bars("BTC") is not None and not b.coefficients.empty and not b.news.empty


def test_app_renders_without_exceptions(demo_dir, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("MACRO_PULSE_SNAPSHOT_DIR", str(demo_dir))
    monkeypatch.setenv("MACRO_PULSE_NO_LIVE", "1")  # never call exchanges from tests
    app = os.path.join(os.path.dirname(__file__), "..", "dashboard", "app.py")
    at = AppTest.from_file(app, default_timeout=240).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 15
    assert len(at.metric) > 40
    assert any("نمایشی" in w.value for w in at.warning)  # demo banner
    top = {m.label: m.value for m in at.metric[:6]}
    assert top["بیت‌کوین (BTC)"] == "80,000"
