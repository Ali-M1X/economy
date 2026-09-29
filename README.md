# Macro Pulse

US macro, liquidity, rates, credit, dollar, intermarket and commodity data → measured impact on **Bitcoin** and **Gold**, with a Persian dashboard, causal reports and backtested trade signals.

> **راهنمای فارسی راه‌اندازی: [README.fa.md](README.fa.md)** (FRED key, Telegram bot, Claude key, GitHub Secrets, Streamlit Cloud, Supabase, VPS checklist, paid alternatives).
>
> Status: **all 7 phases done**:
> 1. collectors + validation;
> 2. features, FedWatch, Fed Stance Score, curve inversions, regimes;
> 3. impact coefficients, Macro Score, backtest engine;
> 4. technicals, liquidity levels and gated trade signals;
> 5. Persian RTL Streamlit dashboard with light/dark mode;
> 6. Telegram messages, Claude explanations and classification, scheduling and health checks;
> 7. Docker / docker-compose with the same schedule for a VPS, and the Persian setup guide.
>
> See `DATA_GAPS.md`.

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
| `llm/` | Claude API wrapper (structured JSON + text; `CLAUDE_MODEL`, default `claude-sonnet-5`) and the news / Fed-document classifiers |
| `reports/` | facts (every number a message may use), Persian Telegram messages, Claude causal explanation with number validation |
| `notify/telegram.py` | Telegram Bot API sender (HTML, splitting, flood control); without secrets it writes to `output/outbox/` |
| `jobs/classify.py`, `jobs/notify.py`, `jobs/health.py`, `jobs/scheduler.py`, `jobs/intraday.py` | Phase 6 jobs (see *Scheduling*) |
| `tests/` | offline tests against fixtures in each source's real format, a full dashboard render on synthetic data, and Phase 6 with Claude/Telegram mocked |

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

## Telegram & Claude (Phase 6)
Messages (Persian, HTML):
- **heads-up** the evening before a tracked release: time in Tehran, previous value, what a higher/lower number meant historically for BTC and gold, and the channel chain;
- **impact report** after a release or market-moving news: what changed, the surprise vs the system's expectation, the causal chain, Macro Scores with top contributors, regime, Fed stance and FedWatch;
- **numeric signals**, or the watchlist with the blocking reason, ranges, backtest and risks;
- **weekly summary**;
- **news alerts** (importance ≥ 4);
- **health warnings**.

Every message ends with the disclaimer and, if the variable `DASHBOARD_URL` is set, a dashboard link. The latest message of each kind is also kept in the snapshot and shown on the dashboard's reports tab.

Claude never produces numbers. It does two things:
- classifies news (relevance, direction for BTC/gold, importance 1–5, Persian title) and Fed documents (hawkish ↔ dovish, which feeds the Fed Stance Score's communications component), using schema-constrained JSON;
- writes the causal paragraph of the impact report from the computed facts and the channel table only. Every number in that paragraph must exist in the facts; otherwise the paragraph is discarded and a deterministic Persian explanation is used.

Without `ANTHROPIC_API_KEY`, classification is skipped and the deterministic text is used. Without Telegram secrets, messages go to `output/outbox/` (uploaded with the run artifact).

**Health gate:** before a report, `jobs.health` checks:
- the core series (prices, policy rate, yields, dollar, inflation, balance sheet …);
- the freshness of the 15-minute BTC/PAXG bars;
- that every analysis output exists;
- that fewer than 25 % of all series failed.

If any check fails, a health warning is sent instead of the report. The same warning is not repeated within 12 h.

## Scheduling
`schedule.yml` holds all crons (UTC); `jobs.scheduler plan` maps the cron that fired to a mode:

| cron (UTC) | mode | what happens |
|---|---|---|
| every 15 min | intraday | news scan → Claude classification → alerts for importance ≥ 4; importance 5 triggers a full report |
| daily 14:40 (18:10 Tehran) | headsup | full pipeline refresh → heads-up for tomorrow's releases (Tehran date) + health check |
| 5 min after each release time, for both US DST offsets | release | only if a tracked release happened in the last 100 min and was not reported: wait until FRED shows the new value (≤ 2 h) → full pipeline → impact report + signals |
| Saturday 03:20 (06:50 Tehran) | weekly | full pipeline (coefficients and backtests recomputed) → weekly summary |

GitHub Actions limits to know:
- cron is UTC only, hence the doubled release crons;
- scheduled runs can start several minutes late or be dropped under load;
- the shortest interval is 5 minutes;
- **scheduled workflows run only from the default branch** (merge this branch into `main` to activate them);
- they are disabled after 60 days without repository activity (re-enable in the Actions tab).

The same jobs will run on a VPS via docker-compose (Phase 7). Run state (reported releases, alerted news, last health warning) is kept in the Actions cache.

Secrets used: `ANTHROPIC_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`. Optional variables: `CLAUDE_MODEL`, `DASHBOARD_URL`. The easiest end-to-end test is a manual run of `schedule` with mode `report`, `weekly`, `headsup` or `intraday`.

## VPS / Docker (Phase 7)
`docker compose up -d --build` starts two services from one image:
- **`scheduler`** — `jobs.runner`: APScheduler with the exact cron strings of `schedule.yml`, imported from `jobs.scheduler`. Weekdays are converted to names, because APScheduler 3 numbers Monday = 0.
- **`dashboard`** — Streamlit, bound to `127.0.0.1:8501`; put an HTTPS reverse proxy in front.

Each full run (`jobs.pipeline`) writes into `/data/runs/<ts>/`. When the analysis succeeds, the `/data/current` symlink is swapped atomically, so the dashboard never reads a half-written snapshot. The last 3 runs are kept, and run state lives in `/data/state`.

Configuration comes from `.env` (see `.env.example`). Database storage stays off unless `ENABLE_DB_STORAGE=true`. The full checklist is in [README.fa.md](README.fa.md).

## Workflows
| workflow | what it does | database |
|---|---|---|
| `ci` | tests (incl. a throw-away Postgres service) | never touches Supabase |
| `data-availability` | the full pipeline: live collection + validation + (Claude classification) + Phases 2–4 + health check + (Telegram) + headless dashboard render (job summary + artifact). Push runs and mode `none` use no Claude and no Telegram. With `PUBLISH_SNAPSHOT=true` a second job publishes the `snapshot-latest` release asset | only when the repository **variable** `ENABLE_DB_STORAGE` is `true`; otherwise `DATABASE_URL` is not passed to the job at all |
| `schedule` | all crons (see *Scheduling*); calls `data-availability` for full runs | same as `data-availability` |
| `features-dev` | re-runs Phases 2–4 and the dashboard render on the latest data snapshot artifact (optional `inspect` input prints raw values) | never |

To enable storage later: set `DATABASE_URL` to Supabase's **Session pooler** URI (GitHub runners have no IPv6, so the direct `db.<ref>.supabase.co` host is unreachable), then add the variable `ENABLE_DB_STORAGE=true`.
