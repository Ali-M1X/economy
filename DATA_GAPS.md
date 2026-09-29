# Data gaps, proxies, and paid alternatives

This file lists everything in the spec that has **no free, terms-compliant source**, and what the system uses instead. The dashboard marks every proxy with a "proxy" badge. The live status of every source is in the Data Availability Report (`data-availability` workflow → job summary / `output/data_availability.md`).

Status legend: **GAP** = not collected · **PROXY** = substitute collected and labelled · **LIMITED** = collected with a restriction.

| # | Item in spec | Status | What we use instead | Best paid option |
|---|---|---|---|---|
| 1 | Consensus forecasts (for surprise = actual − consensus) | **GAP** | Surprise = actual − trend (and actual − previous), standardized by the historical std of that difference, labelled `surprise_basis = trend`. The ForexFactory JSON feed has consensus figures, but its terms are unclear, so it is **off by default** (`ENABLE_FF_CALENDAR=1` turns it on at your own discretion) | Bloomberg ECO / Refinitiv polls; Trading Economics API (~$) |
| 2 | ISM Manufacturing PMI | **PROXY** | Philly Fed Manufacturing (current activity) and Empire State Manufacturing (general business conditions), both official and on FRED. Richmond Fed is not on FRED, so it is not used yet | ISM Report On Business subscription; S&P Global PMI |
| 3 | Cross-currency basis (EUR/USD, USD/JPY) | **GAP** | Nothing reliable is free. As indirect stress gauges we show DXY, the Fed broad dollar index and FX (DEXUSEU/DEXJPUS) | Bloomberg (XCCY basis swaps), LSEG |
| 4 | OIS curve / fed funds options | **GAP** | Fed funds futures (ZQ) and 3M SOFR futures (SR3), **specific contracts only** (e.g. `ZQV26.CBT`), from Yahoo (delayed), with FedWatch-style probabilities computed in-house (Phase 2). Yahoo's "continuous" `ZQ=F` / `SR3=F` are **not** front-month series: on 2026-09-28 `ZQ=F` (95.95) matched the Nov-26 contract while the Sep-26 contract traded at 96.253, and `SR3=F` returned only 1 day of history. They are no longer collected. Every run checks the current-month ZQ contract against the realized daily EFFR average | CME DataMine; CME FedWatch API |
| 5 | CME FedWatch probabilities | **REPLICATED** | Computed from ZQ contract prices using CME's published methodology (Phase 2) | CME FedWatch API |
| 6 | Credit ratings (numeric) | **GAP** | Ratings are not a time series. Spread-based stress instead: HY/IG/BBB OAS, Baa−10Y, plus bank delinquency and charge-off rates | Moody's / S&P rating-action feeds |
| 7 | ICE BofA OAS history on FRED | **LIMITED (confirmed)** | FRED serves only about 3 years: HY, IG and BBB OAS all start **2023-09-29** (live run 2026-09-28). Moody's **Baa−10Y** (`BAA10Y`, daily from 1990) is the long-sample credit proxy for backtests | ICE Data Indices licence |
| 8 | Real-time spot gold (XAUUSD) | **PROXY** | **PAXG/USDT** on crypto exchanges (1 PAXG = 1 troy oz held by Paxos), 24/7 and real-time, plus COMEX `GC=F` (delayed) for history. PAXG can trade at a small premium or discount to spot | OANDA / Polygon.io forex, TwelveData |
| 9 | Coinglass liquidation heatmap / liquidations | **PROXY** | (a) order-book depth aggregated across Binance-vision, OKX, Bybit and Coinbase; (b) OI and funding from Bybit and OKX; (c) estimated liquidation clusters from OI changes × leverage tiers (Phase 4); (d) realized liquidations from the OKX REST feed and the Bybit WebSocket, stored over time. All are labelled "estimate" | Coinglass API, Laevitas, Kaiko |
| 10 | Binance futures and **Bybit** (all endpoints) | **LIMITED (confirmed)** | Both refuse US IPs, and GitHub runners are in the US. Bybit answers HTTP 403 ("The Amazon CloudFront distribution is configured to block access from your country") for spot candles, order book, tickers and OI history (live run 2026-09-28). The Bybit liquidation WebSocket still connects. Open-interest history now comes from **OKX** (`rubik/stat/contracts/open-interest-history`). Spot candles and books fall back to Binance-vision, OKX and Coinbase. The code for both venues is kept; it works from a non-US VPS, and the report labels these 403s as expected | — (a non-US VPS removes the limitation) |
| 11 | China / PBoC liquidity | **GAP (confirmed)** | FRED's China M2 (`MYAGM2CNM189N`, IMF IFS) is **discontinued**: its last observation is 2019-08 (live run 2026-09-28). It is still collected for history but reported as stale, and it is excluded from live liquidity features. PBoC balance-sheet releases are not machine-readable for free. Partial proxy: CNY/USD (`DEXCHUS`) | CEIC, Wind, Bloomberg |
| 12 | Global liquidity | **PROXY** | Fed (WALCL) + ECB (ECBASSETSW) + BoJ (JPNASSETS), converted to USD with FRED FX rates. PBoC is excluded until a free series exists | CrossBorder Capital GLI |
| 13 | Atlanta Fed Wage Growth Tracker | **GAP (for now)** | Average hourly earnings YoY (CES0500000003). The tracker is an Excel download; to be added if the live probe can reach it | — (free, only a format issue) |
| 14 | Bloomberg Commodity Index | **PROXY** | Invesco DB Commodity Index ETF (DBC) | Bloomberg |
| 15 | Implied volatility for gold | **LIMITED** | ATR-based ranges. CBOE GVZ (`^GVZ`) may be added if Yahoo serves it | CME options data |
| 16 | Economic calendar times | **LIMITED** | Official dates from the FRED release-dates API (working with the key) and the Fed FOMC calendar. **bls.gov blocks the GitHub runner** (HTTP 403 "Access Denied") for its ICS release calendar and RSS feed; the BLS **API** (`api.bls.gov`) still works and is used for the CPI cross-check. FRED release dates already cover every tracked BLS release, so nothing is lost. Times of day come from `config/releases.yaml` (agency-published standard times) | — |
| 17 | Yahoo Finance data | **LIMITED** | Unofficial, free and delayed (about 10–20 min for futures). Yahoo can rate-limit datacenter IPs, so collectors retry on the second host and report failures | Polygon.io, Databento |

## Pre-positioning (front-run) signals

Which pre-release inputs are free with enough history to be **backtested**, and which are only shown live. Only backtestable inputs enter the Pre-Positioning Score that the out-of-sample edge gate judges.

| Input | Status | Source used | Best paid option |
|---|---|---|---|
| BTC / gold price drift and steadiness before a release | **BACKTESTED** | Binance-mirror 15-min bars (BTC from 2017, PAXG from 2020-09) | — |
| Order flow (aggressive buy vs sell) | **BACKTESTED (proxy)** | Taker-buy volume from the same Binance klines (spot) | Kaiko / Tardis trade-level data |
| Order-book imbalance (bid/ask wall growth) | **LIVE ONLY** | Aggregated books at each run; history is built from our own snapshots from now on. Not in the backtested score | Tardis.dev / Kaiko historical order books |
| Fed funds / SOFR futures drift before a release | **LIVE ONLY + PROXY** | Specific ZQ contracts from Yahoo are recorded each run (drift from our own snapshots). Expired contracts have no free history, so the **backtested** rate input is the daily 2-year Treasury yield (FRED DGS2, point in time) | CME DataMine (tick / daily history of all contracts) |
| BTC open interest and funding before a release | **BACKTESTED from 2021-12** | Binance public data archive (data.binance.vision): 5-min open interest, monthly funding files. Live: OKX hourly OI and funding history. The two venues differ; each is z-scored against its own history | Coinglass / Laevitas history |
| Gold open interest / funding | **GAP** | No free intraday history (COMEX OI is daily and published with a delay) | CME DataMine |
| ADP employment ahead of NFP | **BACKTESTED** | FRED `ADPMNUSNERSA` with ALFRED vintages; surprise vs its 3-month trend, available from its own release (08:15 ET, two days before NFP) | — |
| CPI / PPI ahead of PCE | **BACKTESTED** | The same month's CPI and PPI first-print surprises (published before PCE) | — |
| Inflation nowcast ahead of CPI | **GAP (free, not wired)** | The Cleveland Fed inflation nowcast is free on its website, but there is no documented API or stable file | Truflation (paid) |
| Whisper numbers (crowd-sourced consensus) | **GAP** | None free | Estimize |
| Private payroll trackers other than ADP | **GAP** | Revelio / LinkUp / Homebase have no free machine-readable history | Revelio Labs, LinkUp |
| Oil inventories (EIA weekly) | **NOT USED** | Free with a key (EIA API), but no tracked release depends on it | — |

## What needs the user's free keys

- `FRED_API_KEY` unlocks ALFRED vintages (needed for look-ahead-free backtests of CPI, NFP, PCE, claims, etc.), series metadata (FRED's own `last_updated`, units and frequency) and FRED release dates. Without it, series still load through the keyless CSV, but **no vintages** are available.
- `BLS_API_KEY` (optional) raises BLS API limits for the CPI cross-check.

## Findings from the live runs (resolved or explained, not gaps)

- **Weekend-dated values in DFF, DFEDTARU/L and IORB are correct.** FRED publishes these for every calendar day. They are flagged `calendar: 7d` in the registry.
- **A missing month in CPI, core CPI, UNRATE and UNEMPLOY is real.** The source reports one month as missing, and the report now prints the exact gap dates. Nothing is filled in: features and backtests see the gap as a gap.
- **Fed funds futures: resolved, the contracts are correct.** The Sep-26 contract (`ZQU26.CBT`) implied 3.747%, and the realized September EFFR average is also 3.747% (25 of 30 days published, the rest at the latest EFFR): a 0.0 bp difference. That average reflects a 25 bp hike effective 2026-09-17 (EFFR 3.63% → 3.88%). The earlier "4.05% front month" reading came from Yahoo's continuous `ZQ=F`, which tracks **Nov-26** (`ZQX26`, 95.945), not the front contract. `SR3=F` likewise tracks Dec-26 (`SR3Z26`). Both continuous tickers were removed, and the current-month check now runs on every data-availability run.
- **Brent spot vs futures: explained, not a data error.** EIA's `DCOILBRENTEU` is **Dated Brent**, a physical North Sea cargo price, while `BZ=F` is the ICE front-month futures contract. They are different instruments. Over the 60 common days to 2026-09-22 the spot premium averaged +3.9% (range −4.6% to +20.8%). The premium peaked at 09-16 (127.84 vs 105.83) and narrowed every day after (114.89 vs 99.25 on 09-22). That pattern points to a squeeze in the physical market, not a unit or series error. The WTI spot/futures basis stayed tight throughout (+0.2% to +2.7%) and passes its 5% cross-check. The system uses **BZ=F** as the Brent price and treats the spot−futures basis as information (physical tightness), so no Brent cross-check is enforced. I have not verified the physical market move against a second source.
