# Macro Pulse

US macro, liquidity, rates, credit, dollar, intermarket and commodity data → measured impact on **Bitcoin** and **Gold**, with a Persian dashboard, causal reports and backtested trade signals.

> Status: **Phase 4 done** — collectors + validation (1); features, FedWatch, Fed Stance Score, curve inversions, regimes (2); impact coefficients, Macro Score, backtest engine (3); technicals, liquidity levels and gated trade signals (4). See `DATA_GAPS.md`.
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

### How a signal is allowed out
1. the setup fires on the latest closed 4h bars (daily trend, 4h structure, pullback into support/resistance, momentum confirmation);
2. the medium-horizon Macro Score agrees in direction with |score| ≥ 15;
3. blended reward:risk ≥ 1.5 with the stop beyond structure and ≥ 1×ATR;
4. the same setup has a **demonstrated out-of-sample edge**: walk-forward average R > 0 over ≥ 30 trades, after fees and slippage.

On the data to 2026-09-28 the setup did **not** show an edge on either asset (BTC longs ≈ break-even, shorts and gold negative), so the engine currently emits no signals and lists fired setups on a watchlist with the reason. That is the intended behaviour: no backtested edge, no signal.
| `tests/` | offline tests against fixtures in each source's real format |

## Run
```bash
pip install -r requirements-dev.txt
python -m pytest -q
python -m jobs.availability          # needs outbound internet
```

## Workflows
| workflow | what it does | database |
|---|---|---|
| `ci` | tests (incl. a throw-away Postgres service) | never touches Supabase |
| `data-availability` | live collection + validation + Phase 2 summary (job summary + artifact) | only when the repository **variable** `ENABLE_DB_STORAGE` is `true`; otherwise `DATABASE_URL` is not passed to the job at all |
| `features-dev` | re-runs Phase 2 on the latest data snapshot artifact (optional `inspect` input prints raw values) | never |

To enable storage later: set `DATABASE_URL` to Supabase's **Session pooler** URI (GitHub runners have no IPv6, so the direct `db.<ref>.supabase.co` host is unreachable), then add the variable `ENABLE_DB_STORAGE=true`.
