# Macro Pulse

US macro, liquidity, rates, credit, dollar, intermarket and commodity data → measured impact on **Bitcoin** and **Gold**, with a Persian dashboard, causal reports and backtested trade signals.

> Status: **Phase 1** — scaffold, DB schema, collectors, validation. See `DATA_GAPS.md` and the `data-availability` workflow.
> A full Persian setup guide arrives in Phase 7.

## Layout
| dir | purpose |
|---|---|
| `config/series.yaml` | registry of every series: source, ID, URL, units, frequency, release lag, sanity range, proxy flag, cross-checks |
| `config/releases.yaml` | tracked economic releases and their publication times (ET) |
| `collectors/` | FRED/ALFRED, Treasury, fiscaldata (DTS), NY Fed, BLS, Yahoo, crypto exchanges, Fed, calendar, news, futures curves |
| `validation/` | schema, range, staleness, weekend-date, gap and cross-source checks |
| `db/` | SQLAlchemy schema (single source) → `db/migrations/*.sql` for Supabase; upsert store |
| `jobs/availability.py` | runs everything and writes the Data Availability Report |
| `tests/` | offline tests against fixtures in each source's real format |

## Run
```bash
pip install -r requirements-dev.txt
python -m pytest -q
python -m jobs.availability          # needs outbound internet
```
