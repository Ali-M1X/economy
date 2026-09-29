"""Plotly figure builders. One y-axis per chart, thin marks, recessive grid, hover on every mark,
direct labels for ≤ 4 series, colours from dashboard.theme (by entity, never by rank)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.fa import fa
from dashboard.theme import ASSET_SLOT, REGIME_SLOT, tokens

ASSET_FA = {"BTC": "بیت‌کوین", "Gold": "طلا"}
FONT = "Vazirmatn, system-ui, -apple-system, Segoe UI, sans-serif"


def base(fig: go.Figure, mode: str, title: str = "", height: int = 320, ytitle: str = "") -> go.Figure:
    t = tokens(mode)
    fig.update_layout(
        title={"text": title, "x": 0.985, "xref": "container", "xanchor": "right", "font": {"size": 15, "color": t["ink"]}},
        paper_bgcolor=t["surface"], plot_bgcolor=t["surface"], height=height,
        font={"family": FONT, "color": t["ink2"], "size": 12}, margin={"l": 48, "r": 16, "t": 44, "b": 36},
        hovermode="x unified", legend={"orientation": "h", "y": -0.18, "x": 1, "xanchor": "right"},
        hoverlabel={"bgcolor": t["surface"], "font": {"family": FONT, "color": t["ink"]}},
    )
    fig.update_xaxes(showgrid=False, linecolor=t["axis"], tickfont={"color": t["muted"]})
    fig.update_yaxes(gridcolor=t["grid"], gridwidth=1, zeroline=False, linecolor=t["axis"],
                     tickfont={"color": t["muted"]}, title={"text": ytitle, "font": {"color": t["muted"], "size": 11}})
    return fig


def lines(series: dict[str, pd.Series], mode: str, title: str, unit: str = "", slots: dict[str, int] | None = None,
          zero_line: bool = False, height: int = 320) -> go.Figure:
    t = tokens(mode)
    fig = go.Figure()
    for i, (name, s) in enumerate(series.items()):
        s = s.dropna()
        if s.empty:
            continue
        color = t["series"][(slots or {}).get(name, i) % 8]
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=name, mode="lines", line={"color": color, "width": 2},
                                 hovertemplate=f"%{{y:,.2f}} {unit}<extra>{name}</extra>"))
        if len(series) <= 4:  # direct label at the line end
            fig.add_annotation(x=s.index[-1], y=s.values[-1], text=f"{name}  {s.values[-1]:,.2f}", showarrow=False,
                               xanchor="left", font={"size": 11, "color": t["ink2"]})
    if zero_line:
        fig.add_hline(y=0, line={"color": t["axis"], "width": 1})
    return base(fig, mode, title, height, unit)


def with_episodes(fig: go.Figure, episodes: pd.DataFrame, mode: str) -> go.Figure:
    """Shade yield-curve inversion episodes and mark re-steepening dates."""
    t = tokens(mode)
    n_label, seen = 0, set()
    for e in episodes.itertuples():
        if bool(getattr(e, "brief", False)):
            continue
        end = e.end if e.end is not None and not pd.isna(e.end) else pd.Timestamp.today()
        fig.add_vrect(x0=e.start, x1=end, fillcolor=t["shade"], line_width=0, layer="below")
        if e.resteepen_date is not None and not pd.isna(e.resteepen_date):
            fig.add_vline(x=e.resteepen_date, line={"color": t["muted"], "width": 1, "dash": "dot"})
        if isinstance(e.label, str) and e.label not in seen:  # label each episode name once, staggered
            seen.add(e.label)
            fig.add_annotation(x=e.start, y=1 - 0.08 * (n_label % 3), yref="paper", text=fa(e.label), showarrow=False,
                               xanchor="left", yanchor="top", font={"size": 10, "color": t["muted"]}, textangle=0,
                               bgcolor=t["surface"])
            n_label += 1
    return fig


def regime_timeline(labels: pd.Series, price: pd.Series, asset: str, mode: str, height: int = 300) -> go.Figure:
    """Price (log) on a single axis with regime phases as background bands."""
    t = tokens(mode)
    fig = go.Figure()
    run = (labels != labels.shift()).cumsum()
    seen = set()
    for _, g in labels.groupby(run):
        reg = g.iloc[0]
        x1 = g.index[-1] + pd.offsets.MonthEnd(1)
        fig.add_vrect(x0=g.index[0], x1=x1, fillcolor=t["series"][REGIME_SLOT[reg]], opacity=0.16, line_width=0, layer="below")
        seen.add(reg)
    p = price.dropna()
    p = p[p.index >= labels.index[0]]
    fig.add_trace(go.Scatter(x=p.index, y=p.values, name=asset, mode="lines",
                             line={"color": t["series"][ASSET_SLOT[asset]], "width": 2},
                             hovertemplate="%{y:,.0f}<extra>" + asset + "</extra>"))
    for reg in ("Expansion", "Peak", "Recession", "Recovery"):  # legend swatches for the bands
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=fa(reg),
                                 marker={"size": 10, "symbol": "square", "color": t["series"][REGIME_SLOT[reg]], "opacity": 0.5}))
    fig.update_yaxes(type="log")
    return base(fig, mode, ASSET_FA.get(asset, asset) + " و فازهای چرخه اقتصادی", height)


def heatmap(coefs: pd.DataFrame, mode: str, height: int = 520) -> go.Figure:
    """Indicators × (asset · horizon), diverging blue (bullish) ↔ red (bearish), grey midpoint; values printed."""
    t = tokens(mode)
    piv = coefs.pivot_table(index="indicator", columns="col", values="coefficient", aggfunc="first")
    piv = piv.loc[piv.abs().max(axis=1).sort_values().index]
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=list(piv.columns), y=list(piv.index), zmin=-10, zmax=10, zmid=0,
        colorscale=[[0, t["div_neg"]], [0.5, t["div_mid"]], [1, t["div_pos"]]],
        text=np.where(np.isnan(piv.values), "", np.vectorize(lambda v: f"{v:+.1f}")(np.nan_to_num(piv.values))),
        texttemplate="%{text}", textfont={"size": 10}, xgap=2, ygap=2,
        hovertemplate="%{y} · %{x}<br>ضریب اثر %{z:+.1f}<extra></extra>",
        colorbar={"title": {"text": "ضریب"}, "tickvals": [-10, -5, 0, 5, 10]}))
    fig = base(fig, mode, "ضرایب اثر (+ صعودی، − نزولی)", height)
    fig.update_layout(hovermode="closest")
    return fig


def merge_levels(levels: list[dict], span: float, tol: float = 0.035) -> list[dict]:
    """Collapse levels closer than tol × chart range into one line with a joined label, so labels never overlap."""
    out: list[dict] = []
    for lv in sorted(levels, key=lambda d: d["price"]):
        if out and span > 0 and abs(lv["price"] - out[-1]["price"]) < tol * span:
            prev = out[-1]
            head, _, tail = lv["label"].rpartition(" ")
            if prev["label"].split(" · ")[-1].startswith(head + " ") and head:  # "سقف سوئینگ روزانه" + "… ۴ساعته" → joined
                prev["label"] += "/" + tail
            elif lv["label"] not in prev["label"].split(" · "):
                prev["label"] += " · " + lv["label"]
            if "color" in lv and "color" not in prev:  # trade levels (stop/TP) win the line style
                prev.update(color=lv["color"], dash=lv.get("dash", prev.get("dash")))
        else:
            out.append(dict(lv))
    return out


def price_levels(bars: pd.DataFrame, asset: str, mode: str, levels: list[dict], height: int = 460) -> go.Figure:
    """4h candles (last 90 days) with key levels as labelled horizontal lines."""
    t = tokens(mode)
    color = t["series"][ASSET_SLOT[asset]]
    fig = go.Figure(go.Candlestick(
        x=bars["ts"], open=bars["open"], high=bars["high"], low=bars["low"], close=bars["close"], name=asset,
        increasing={"line": {"color": color, "width": 1}, "fillcolor": color},
        decreasing={"line": {"color": t["muted"], "width": 1}, "fillcolor": t["muted"]}))
    for lv in merge_levels(levels, float(bars["high"].max() - bars["low"].min())):
        fig.add_hline(y=lv["price"], line={"color": lv.get("color", t["ink2"]), "width": 1, "dash": lv.get("dash", "dot")},
                      annotation={"text": lv["label"], "font": {"size": 10, "color": t["ink2"]}},
                      annotation_position="top left")
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig = base(fig, mode, f"{asset} — نمودار ۴ ساعته و سطوح کلیدی", height)
    fig.update_layout(hovermode="x")
    return fig
