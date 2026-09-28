# Macro Pulse

US macro, liquidity, rates, credit, dollar, intermarket and commodity data → measured impact on **Bitcoin** and **Gold**, with a Persian dashboard, causal reports and backtested trade signals.

> Status: **Phase 2 done** — collectors + validation (Phase 1) and features, FedWatch, Fed Stance Score, curve inversions and regimes (Phase 2). See `DATA_GAPS.md`.
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
