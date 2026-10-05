# VCT Champions 2026 predictor

Match and bracket predictions for VCT Champions 2026 (Pick'Em), built from every VCT 2025–2026
map on [VLR.gg](https://www.vlr.gg).

## Quick start

```sh
uv sync
uv run vct all          # scrape (cached), build + validate data, run backtests
```

Then open `notebooks/vct_champions_2026_predictor.ipynb` for title odds and the pick sheet.

| Command | What it does |
|---|---|
| `uv run vct scrape` | Downloads every match of the events in `src/vct/config.py`. Pages are cached in `data/raw/` (with `index.csv` recording each URL) and requests are spaced at least 1.5 s apart. Matches that are still upcoming or live are re-fetched on the next run. |
| `uv run vct build` | Resolves team names through `data/team_aliases.csv`, assigns regions, writes `data/processed/*.parquet`, and runs the data checks. Fails if any check is an error. |
| `uv run vct experiments` | Runs the time-ordered backtests and writes `experiments.csv`, `data/processed/best_model.pkl` and `holdout_predictions.pkl`. |

## Layout

```
data/
  raw/                 cached VLR HTML (git-ignored)
  interim/             parsed tables straight from the scraper (git-ignored)
  processed/           series, maps, players, vetoes, upcoming (.parquet) + backtest outputs
  team_aliases.csv     every spelling -> one team code; rows with source=manual survive rebuilds
  legacy/              the original hand-compiled 40-series CSV
src/vct/
  config.py            paths and the VLR event list
  scrape.py            VLR scraper (all CSS selectors live here)
  clean.py             aliases, regions, market probabilities, data checks
  features.py          time / margin / roster weights, series features
  models.py            Bradley-Terry (region + per-map terms), Glicko-2, market blend
  simulate.py          veto simulation, series probabilities, bracket Monte Carlo
  evaluate.py          walk-forward backtest, metrics, calibration, bootstrap
  experiments.py       the improvement plan's experiments -> experiments.csv
notebooks/
  vct_champions_2026_predictor.ipynb
experiments.csv        experiment log
```

## Data

One row per map in `data/processed/maps.parquet`:

`match_id, game_id, map_order, date, event, event_id, region_a, region_b, team_a, team_b, map,
rounds_a, rounds_b, y, picked_by, best_of, lan, international`

`series.parquet` adds series scores, stage, patch and `p_market`, the bookmakers' pre-match
probability for team A (median across books, margin removed). On finished matches VLR shows
only the winner's price, so the margin is removed using the median overround seen on matches
where both prices are shown.

Data checks (`clean.validate`): duplicate matches/maps, map wins that don't add up to the
series score, series scores inconsistent with best-of, dates out of range, unknown teams
(errors); forfeits without map data and unusual round scores (warnings).

## Method

* **Model:** P(A wins a map) = σ((r_A + region_A + r_A,map) − (r_B + region_B + r_B,map)),
  fitted by L2-penalised weighted logistic regression, with a separate penalty for each
  parameter group. Weights combine date decay `0.5 ** (days_ago / half_life)`, round margin,
  extra weight on international maps, and roster continuity (maps played by a since-changed
  lineup count less, which pulls the team's rating back toward zero).
* **Series:** maps are simulated in order. With per-map ratings, each team bans its weakest
  maps and picks its strongest, nudged by its historical pick/ban habits.
* **Calibration:** one temperature on the rating gap, fitted on earlier backtest weeks only.
* **Market:** `p = w * p_model + (1 - w) * p_market`, with `w` tuned on the backtest.
* **Evaluation:** refit every week on all earlier maps, predict that week's series. Settings
  are tuned on Apr 2025 – May 2026; the chosen model is scored once on Jun 2026 onwards.
  Each change is kept only if it lowers log loss. Paired bootstrap intervals show whether a
  difference is distinguishable from noise.

## Results (2026-10-05)

Dataset: 1,217 series / 3,065 maps / 79 teams from 32 events (Jan 2025 – Oct 2026), all data
checks passing, market prices on 1,161 series.

**Tuning window** (Apr 2025 – May 2026, 746 series, refit weekly):

| Model | Log loss | Brier | Accuracy |
|---|---|---|---|
| Coin flip | 0.693 | 0.250 | 50% |
| Phase 0 model (original method, new data) | 0.672 | 0.240 | 60% |
| Final rating model | 0.637 | 0.223 | 63% |
| Final model + market blend | 0.631 | 0.221 | 64% |
| Market alone (708 priced series) | 0.631 | | |

Kept, in order: 180-day date decay (C=0.3), round-margin weight (1 + margin), region term,
per-map ratings with veto simulation, roster pull-back (α=0.25). Rejected: round-share target,
extra weight on international maps, Glicko-2 (0.684), temperature scaling, LightGBM feature
model (0.677 vs 0.643 on the same rows). After the date-decay step, each kept change is
smaller than its bootstrap interval, so treat them as "not harmful" rather than proven gains.
Full log: `experiments.csv`.

**Holdout** (Jun – Oct 2026, 318 series, never used for tuning):

| Model | Log loss | Brier | Accuracy |
|---|---|---|---|
| Coin flip | 0.693 | 0.250 | 51% |
| Phase 0 model | 0.701 | 0.254 | 54% |
| Final rating model | 0.695 | 0.250 | 54% |
| Final model + market (w = 0.35) | 0.666 | 0.237 | 59% |
| Market alone (304 priced series) | 0.645 | | |

The rating model is overconfident on the holdout (favourites won ~10 points less often than
predicted) and alone is no better than a coin flip there. The market is the strongest single
predictor. Live predictions therefore use a temperature of 0.80 and a model weight of 0.15,
both refit on all 1,064 backtest predictions after the holdout was scored.

Plan targets: series log loss ≤ 0.66 was met on the tuning window but not on the holdout by
the model alone (the blend reaches 0.666); backtest size ≥ 300 series is met (1,064);
calibration within ±5 points per bucket is **not** met on the holdout.

## Not done

* Challengers / Ascension data (optional in the plan): promoted teams start from their region's
  rating instead.
* Player ratings from VLR stats (optional in the plan). Player stats are scraped into
  `players.parquet`; only lineups are used (roster pull-back).
* Market prices on finished matches are one-sided (winner only), so their margin removal uses
  an assumed overround; the live quarterfinal prices are two-sided.
