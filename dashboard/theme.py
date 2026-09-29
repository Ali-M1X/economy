"""Design tokens (validated reference palette; dark mode is its own selected steps, not an inversion)
and the RTL/Persian page CSS."""

from __future__ import annotations

TOKENS = {
    "light": {
        "surface": "#fcfcfb", "page": "#f9f9f7", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
        "div_neg": "#e34948", "div_mid": "#f0efec", "div_pos": "#2a78d6",
        "shade": "rgba(227,73,72,0.10)",
    },
    "dark": {
        "surface": "#1a1a19", "page": "#0d0d0d", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
        "div_neg": "#e66767", "div_mid": "#383835", "div_pos": "#3987e5",
        "shade": "rgba(230,103,103,0.16)",
    },
}
STATUS_COLORS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
# fixed entity colours (never by rank): gold = slot 1, BTC = slot 2
ASSET_SLOT = {"Gold": 0, "BTC": 1}
REGIME_SLOT = {"Expansion": 2, "Peak": 3, "Recession": 7, "Recovery": 0}


def tokens(mode: str) -> dict:
    return TOKENS["dark" if mode == "dark" else "light"]


CSS = """
<link href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600;700&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"], .stMarkdown, .stMetric, button, input, textarea, select {
  font-family: "Vazirmatn", system-ui, -apple-system, "Segoe UI", sans-serif;
}
[data-testid="stMainBlockContainer"], [data-testid="stSidebar"] { direction: rtl; text-align: right; }
[data-testid="stMetric"] { direction: rtl; }
/* each metric line picks its own direction from its first strong character: "+18" stays LTR, "احتمال ۶۶٪" RTL */
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * { unicode-bidi: plaintext; text-align: right; }
[data-testid="stMetricDelta"], [data-testid="stMetricDelta"] * { unicode-bidi: plaintext; }
.stTabs [data-baseweb="tab-list"] { flex-wrap: wrap; gap: 2px; }
.js-plotly-plot, [data-testid="stPlotlyChart"], [data-testid="stDataFrame"], code, pre { direction: ltr; text-align: left; }
.mp-note { font-size: 0.82rem; opacity: 0.75; }
.mp-badge { display:inline-block; padding: 0 .45rem; border-radius: .5rem; font-size: .78rem;
            border: 1px solid rgba(127,127,127,.35); margin-inline-start: .3rem; }
@media (max-width: 640px) { [data-testid="stMainBlockContainer"] { padding-left: 16px; padding-right: 16px; } }
</style>
"""
