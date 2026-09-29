# Macro Pulse

US macro, liquidity, rates, credit, dollar, intermarket and commodity data → measured impact on **Bitcoin** and **Gold**, with a Persian dashboard, causal reports and backtested trade signals.

> Status: **Phase 5 done** — collectors + validation (1); features, FedWatch, Fed Stance Score, curve inversions, regimes (2); impact coefficients, Macro Score, backtest engine (3); technicals, liquidity levels and gated trade signals (4); Persian RTL Streamlit dashboard with light/dark mode (5). See `DATA_GAPS.md`.
> A full Persian setup guide arrives in Phase 7.

## Layout
| dir | purpose |
|---|---|
| `config/series.yaml` | registry of every series: source, ID, URL, units, frequency, release lag, sanity range, proxy flag, cross-checks |
| `config/releases.yaml` | tracked economic releases and their publication times (ET) |
| `collectors/` | FRED/ALFRED, Treasury, fiscaldata (DTS), NY Fed, BLS, Yahoo, crypto exchanges, Fed, calendar, news, futures curves |
| `validation/` | schema, range, staleness, weekend-date, gap and cross-source checks |
| `db/` | SQLAlchemy schema (single source) → `db/migrations/*.sql` for Supabase; upsert store |
| `jobs/availability.py` | runs every collector, validates, writes the Data Availability Report (and a CSV snapshot with `--cache-dir`) |
| `features/` | derived series, point-in-time loader, curve inversions, FedWatch, Fed Stance Score |
| `models/regime.py` | Expansion / Peak / Recession / Recovery (rule-based, point-in-time) + comparison HMM |
| `jobs/features.py` | Phase 2 pipeline (from the DB, or from a snapshot with `--from-cache`) |
| `config/indicators.yaml` | indicators, change transforms, transmission channels and editable theory sign priors |
| `models/events.py`, `models/impact.py` | point-in-time release surprises; event study / local projections / long horizon; 0–10 Impact Coefficient |
| `jobs/impact.py` | Phase 3: coefficients for BTC & gold and the Macro Score (−100…+100) with contribution breakdown |
| `backtest/engine.py` | trade simulator (fees, slippage, stop-first, 3-TP scale-out, walk-forward folds) used for win rates in Phase 4 |
| `technicals/` | indicators, causal swing structure (HH/HL, BOS/CHOCH), levels, volume profile, order-book walls, liquidation estimates |
| `signals/` | the trend-pullback rules (one causal function for backtest and live) and the walk-forward engine |
| `jobs/signals.py` | Phase 4 report: technical state, context levels, OOS backtest, signals / watchlist, risks, disclaimer |
| `dashboard/` | Phase 5 Streamlit app (`app.py`), data loader, cards, charts, design tokens, demo snapshot, headless smoke check |
| `core/fa.py` | Persian rendering of the English messages the engines emit (reasons, risks, states), with bidi-safe numbers |
| `tests/` | offline tests against fixtures in each source's real format, plus a full dashboard render on synthetic data |

### How a signal is allowed out
1. the setup fires on the latest closed 4h bars (daily trend, 4h structure, pullback into support/resistance, momentum confirmation);
2. the medium-horizon Macro Score agrees in direction with |score| ≥ 15;
3. blended reward:risk ≥ 1.5 with the stop beyond structure and ≥ 1×ATR;
4. the same setup has a **demonstrated out-of-sample edge**: walk-forward average R > 0 over ≥ 30 trades, after fees and slippage.

On the data to 2026-09-28 the setup did **not** show an edge on either asset (BTC longs ≈ break-even, shorts and gold negative), so the engine currently emits no signals and lists fired setups on a watchlist with the reason. That is the intended behaviour: no backtested edge, no signal.

## Run
```bash
pip install -r requirements-dev.txt
python -m pytest -q
python -m jobs.availability          # needs outbound internet
```

## Dashboard (Phase 5)
Persian, right-to-left, follows the viewer's light/dark setting, works on phones. Tabs: Bitcoin & Gold (4h chart with key levels, technical state, out-of-sample backtest, signals/watchlist, risks), one tab per data group (rates & Fed, inflation, liquidity, labour, yield curve, credit, dollar & intermarket, commodities) with a card per indicator (value, change, 90-point sparkline, source and last date on hover), rate expectations (FedWatch + Fed Stance Score), calendar & news (Tehran time), impact-coefficient heatmap, regimes, reports, and a data-sources table. The top bar refreshes BTC and PAXG prices every 60 s.

It reads a **data snapshot** (the `output/` folder of a `data-availability` run), not the database, so it works while storage is off:

| setting | meaning |
|---|---|
| `MACRO_PULSE_SNAPSHOT_DIR` | local snapshot folder (e.g. an unzipped `data-availability` artifact) |
| `MACRO_PULSE_SNAPSHOT_URL` | a `.tar.gz` of that folder — the `snapshot-latest` release asset below |

```bash
pip install -r requirements.txt
MACRO_PULSE_SNAPSHOT_DIR=path/to/output streamlit run dashboard/app.py
# no data yet? a synthetic demo (clearly bannered, not real numbers):
python -m dashboard.demo_snapshot /tmp/demo && MACRO_PULSE_SNAPSHOT_DIR=/tmp/demo streamlit run dashboard/app.py
```

**Hosting on Streamlit Community Cloud** (free): 1) set the repository variable `PUBLISH_SNAPSHOT=true` and run `data-availability` once, which publishes `https://github.com/<owner>/<repo>/releases/download/snapshot-latest/snapshot.tar.gz`; 2) on share.streamlit.io create an app from this repo, main file `dashboard/app.py`; 3) in the app's *Secrets*, add `MACRO_PULSE_SNAPSHOT_URL = "<that URL>"`. The dashboard needs no API keys. The snapshot is public macro/market data only; while the repo is public, so is the release asset.

## Workflows
| workflow | what it does | database |
|---|---|---|
| `ci` | tests (incl. a throw-away Postgres service) | never touches Supabase |
| `data-availability` | live collection + validation + Phases 2–4 + a headless dashboard render (job summary + artifact); with `PUBLISH_SNAPSHOT=true` a second job publishes the `snapshot-latest` release asset | only when the repository **variable** `ENABLE_DB_STORAGE` is `true`; otherwise `DATABASE_URL` is not passed to the job at all |
| `features-dev` | re-runs Phases 2–4 and the dashboard render on the latest data snapshot artifact (optional `inspect` input prints raw values) | never |

To enable storage later: set `DATABASE_URL` to Supabase's **Session pooler** URI (GitHub runners have no IPv6, so the direct `db.<ref>.supabase.co` host is unreachable), then add the variable `ENABLE_DB_STORAGE=true`.
