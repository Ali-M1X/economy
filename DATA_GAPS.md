# Data gaps, proxies, and paid alternatives

This file lists everything in the spec that has **no free, terms-compliant source**, and what the system uses instead. The dashboard marks every proxy with a "proxy" badge. The live status of every source is in the Data Availability Report (`data-availability` workflow → job summary / `output/data_availability.md`).

Status legend: **GAP** = not collected · **PROXY** = substitute collected and labelled · **LIMITED** = collected with a restriction.

| # | Item in spec | Status | What we use instead | Best paid option |
|---|---|---|---|---|
| 1 | Consensus forecasts (for surprise = actual − consensus) | **GAP** | Surprise = actual − trend (and actual − previous), standardized by the historical std of that difference, labelled `surprise_basis = trend`. The ForexFactory JSON feed has consensus figures, but its terms are unclear, so it is **off by default** (`ENABLE_FF_CALENDAR=1` turns it on at your own discretion) | Bloomberg ECO / Refinitiv polls; Trading Economics API (~$) |
| 2 | ISM Manufacturing PMI | **PROXY** | Philly Fed Manufacturing (current activity) and Empire State Manufacturing (general business conditions), both official and on FRED. Richmond Fed is not on FRED, so it is not used yet | ISM Report On Business subscription; S&P Global PMI |
| 3 | Cross-currency basis (EUR/USD, USD/JPY) | **GAP** | Nothing reliable is free. As indirect stress gauges we show DXY, the Fed broad dollar index and FX (DEXUSEU/DEXJPUS) | Bloomberg (XCCY basis swaps), LSEG |
| 4 | OIS curve / fed funds options | **GAP** | Fed funds futures (ZQ) and 3M SOFR futures (SR3) from Yahoo (delayed), with FedWatch-style probabilities computed in-house (Phase 2) | CME DataMine; CME FedWatch API |
| 5 | CME FedWatch probabilities | **REPLICATED** | Computed from ZQ contract prices using CME's published methodology (Phase 2) | CME FedWatch API |
| 6 | Credit ratings (numeric) | **GAP** | Ratings are not a time series. Spread-based stress instead: HY/IG/BBB OAS, Baa−10Y, plus bank delinquency and charge-off rates | Moody's / S&P rating-action feeds |
| 7 | ICE BofA OAS history on FRED | **LIMITED** | ICE restricted FRED's history for these series (recent years only). Moody's **Baa−10Y** (`BAA10Y`, long history) is the long-sample credit proxy for backtests | ICE Data Indices licence |
| 8 | Real-time spot gold (XAUUSD) | **PROXY** | **PAXG/USDT** on crypto exchanges (1 PAXG = 1 troy oz held by Paxos), 24/7 and real-time, plus COMEX `GC=F` (delayed) for history. PAXG can trade at a small premium or discount to spot | OANDA / Polygon.io forex, TwelveData |
| 9 | Coinglass liquidation heatmap / liquidations | **PROXY** | (a) order-book depth aggregated across Binance-vision, OKX, Bybit and Coinbase; (b) OI and funding from Bybit and OKX; (c) estimated liquidation clusters from OI changes × leverage tiers (Phase 4); (d) realized liquidations from the OKX REST feed and the Bybit WebSocket, stored over time. All are labelled "estimate" | Coinglass API, Laevitas, Kaiko |
| 10 | Binance futures (OI, funding, liquidations) | **LIMITED** | `fapi.binance.com` refuses US IPs (GitHub runners). The code is kept and works from a non-US VPS | — |
| 11 | China / PBoC liquidity | **PROXY / LIMITED** | China M2 via FRED (IMF IFS series), which is monthly with a long lag and may be discontinued. PBoC balance-sheet releases are not machine-readable for free | CEIC, Wind, Bloomberg |
| 12 | Global liquidity | **PROXY** | Fed (WALCL) + ECB (ECBASSETSW) + BoJ (JPNASSETS), converted to USD with FRED FX rates. PBoC is excluded until a free series exists | CrossBorder Capital GLI |
| 13 | Atlanta Fed Wage Growth Tracker | **GAP (for now)** | Average hourly earnings YoY (CES0500000003). The tracker is an Excel download; to be added if the live probe can reach it | — (free, only a format issue) |
| 14 | Bloomberg Commodity Index | **PROXY** | Invesco DB Commodity Index ETF (DBC) | Bloomberg |
| 15 | Implied volatility for gold | **LIMITED** | ATR-based ranges. CBOE GVZ (`^GVZ`) may be added if Yahoo serves it | CME options data |
| 16 | Economic calendar times | **LIMITED** | Official dates: FRED release-dates API (needs the free key), the BLS ICS calendar and the Fed FOMC calendar. Times of day come from `config/releases.yaml` (agency-published standard times) | — |
| 17 | Yahoo Finance data | **LIMITED** | Unofficial, free and delayed (about 10–20 min for futures). Yahoo can rate-limit datacenter IPs, so collectors retry on the second host and report failures | Polygon.io, Databento |

## What needs the user's free keys

- `FRED_API_KEY` unlocks ALFRED vintages (needed for look-ahead-free backtests of CPI, NFP, PCE, claims, etc.), series metadata (FRED's own `last_updated`, units and frequency) and FRED release dates. Without it, series still load through the keyless CSV, but **no vintages** are available.
- `BLS_API_KEY` (optional) raises BLS API limits for the CPI cross-check.
