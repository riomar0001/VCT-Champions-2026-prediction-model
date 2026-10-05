"""Time-ordered backtests, metrics, calibration, bootstrap intervals, experiment log."""

from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.calibration import calibration_curve

from . import config, features
from .models import BradleyTerry, BTConfig, Glicko2, series_prob_seq, sigmoid
from .simulate import VetoHabits, map_pool, veto_map_logits

# tune on the first window, then confirm once on the untouched holdout
TUNE = ("2025-04-01", "2026-06-01")
HOLDOUT = ("2026-06-01", "2027-01-01")


class Context:
    """Static inputs every predictor may use. Only data dated before the fit date is read."""

    def __init__(self, series, maps, players=None, vetoes=None):
        self.series, self.maps, self.vetoes = series, maps, vetoes
        self.lineups = features.lineups(players) if players is not None else None
        regions = pd.concat(
            [series.set_index("team_a").region_a, series.set_index("team_b").region_b]
        )
        self.regions = regions[~regions.index.duplicated(keep="last")].to_dict()


# ---------------------------------------------------------------- predictors
# A predictor factory takes (train_maps, fit_date, ctx) and returns
# f(team_a, team_b, best_of) -> list of map log-odds in play order.


def bt_predictor(cfg: BTConfig, veto=False, kappa=1.0):
    def make(train, t0, ctx):
        m = BradleyTerry(cfg).fit(
            train, t0, regions=ctx.regions, lineup_table=ctx.lineups
        )
        if veto and cfg.per_map and ctx.vetoes is not None:
            pool = map_pool(ctx.vetoes, t0)
            habits = VetoHabits().fit(ctx.vetoes, t0)
            return lambda a, b, bo: veto_map_logits(m, a, b, bo, pool, habits, kappa)
        return lambda a, b, bo: [m.diff(a, b)] * bo

    return make


def glicko_predictor(tau=0.5, period_days=7):
    def make(train, t0, ctx):
        m = Glicko2(tau=tau, period_days=period_days).fit(train)
        return lambda a, b, bo: [m.diff(a, b)] * bo

    return make


# ---------------------------------------------------------------- backtest


def walk_forward(ctx, make_predictor, window=TUNE, step_days=7):
    """Refit every `step_days` on all maps before the block, predict the block's series."""
    s = ctx.series
    t_start, t_end = pd.Timestamp(window[0]), min(
        pd.Timestamp(window[1]), s.date.max() + pd.Timedelta(days=1)
    )
    rows = []
    for t0 in pd.date_range(t_start, t_end, freq=f"{step_days}D"):
        t1 = min(t0 + pd.Timedelta(days=step_days), t_end)
        test = s[(s.date >= t0) & (s.date < t1)]
        if test.empty:
            continue
        pred = make_predictor(ctx.maps[ctx.maps.date < t0], t0, ctx)
        for r in test.itertuples():
            d = list(pred(r.team_a, r.team_b, r.best_of))
            rows.append(
                {
                    "match_id": r.match_id,
                    "date": r.date,
                    "block": t0,
                    "event": r.event,
                    "team_a": r.team_a,
                    "team_b": r.team_b,
                    "best_of": r.best_of,
                    "y": r.y,
                    "p_market": r.p_market,
                    "d": d,
                }
            )
    out = pd.DataFrame(rows)
    out["p"] = apply_temperature(out, 1.0)
    return out


def apply_temperature(bt, T):
    return np.array(
        [
            series_prob_seq(list(sigmoid(T * np.array(d))), bo)
            for d, bo in zip(bt.d, bt.best_of)
        ]
    )


def fit_temperature(bt):
    def ll(T):
        return log_loss(bt.y, apply_temperature(bt, T))

    return minimize_scalar(ll, bounds=(0.2, 3.0), method="bounded").x


def walk_forward_temperature(bt, min_n=100):
    """Each block's temperature is fitted on earlier blocks' predictions only."""
    p, Ts = np.empty(len(bt)), []
    for blk in bt.block.unique():
        now = (bt.block == blk).to_numpy()
        past = bt[bt.block < blk]
        T = fit_temperature(past) if len(past) >= min_n else 1.0
        p[now] = apply_temperature(bt[now], T)
        Ts.append(T)
    return p, Ts


def walk_forward_blend(bt, p_model, min_n=100, grid=np.linspace(0, 1, 21)):
    """Model/market weight w per block, chosen on earlier blocks with prices."""
    p, ws = np.array(p_model, float), []
    has = bt.p_market.notna().to_numpy()
    for blk in bt.block.unique():
        now = (bt.block == blk).to_numpy() & has
        past = (bt.block < blk).to_numpy() & has
        if past.sum() >= min_n:
            w = min(
                grid,
                key=lambda w: log_loss(
                    bt.y[past], w * p_model[past] + (1 - w) * bt.p_market[past]
                ),
            )
        else:
            w = 0.3
        p[now] = w * p_model[now] + (1 - w) * bt.p_market[now]
        ws.append(w)
    return p, ws


# ---------------------------------------------------------------- metrics


def log_loss(y, p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "log_loss": log_loss(y, p),
        "brier": float(np.mean((p - y) ** 2)),
        "accuracy": float(np.mean((p > 0.5) == (y == 1))),
        "n": len(y),
    }


def bootstrap(y, p, p_ref=None, n=2000, seed=0):
    """95% interval for log loss, or for log loss(p) - log loss(p_ref) if p_ref is given."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    if p_ref is not None:
        q = np.clip(np.asarray(p_ref, float), 1e-4, 1 - 1e-4)
        ll = ll + (y * np.log(q) + (1 - y) * np.log(1 - q))
    idx = rng.integers(0, len(y), (n, len(y)))
    stats = ll[idx].mean(axis=1)
    return (
        float(ll.mean()),
        float(np.percentile(stats, 2.5)),
        float(np.percentile(stats, 97.5)),
    )


def calibration_table(y, p, bins=5):
    """Predicted vs actual per probability bucket, from the favourite's side."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    fav = np.where(p >= 0.5, p, 1 - p)
    won = np.where(p >= 0.5, y, 1 - y)
    edges = np.linspace(0.5, 1.0, bins + 1)
    k = np.clip(np.digitize(fav, edges) - 1, 0, bins - 1)
    t = (
        pd.DataFrame({"bucket": k, "predicted": fav, "actual": won})
        .groupby("bucket")
        .agg(
            predicted=("predicted", "mean"),
            actual=("actual", "mean"),
            n=("actual", "size"),
        )
    )
    t.index = [f"{edges[i]:.0%}-{edges[i + 1]:.0%}" for i in t.index]
    t["gap_pts"] = (t.actual - t.predicted) * 100
    return t


def reliability(y, p, bins=10):
    """sklearn calibration curve (both sides), for plotting."""
    return calibration_curve(y, p, n_bins=bins, strategy="quantile")


# ---------------------------------------------------------------- experiment log

LOG_COLUMNS = [
    "date",
    "change",
    "log_loss",
    "ll_low",
    "ll_high",
    "brier",
    "accuracy",
    "n_series",
    "delta_vs_best",
    "delta_low",
    "delta_high",
    "keep",
    "window",
    "settings",
]


def summarise(name, y, p, p_best=None, settings="", window=TUNE, keep=""):
    m = metrics(y, p)
    _, lo, hi = bootstrap(y, p)
    row = {
        "date": date.today().isoformat(),
        "change": name,
        "log_loss": m["log_loss"],
        "ll_low": lo,
        "ll_high": hi,
        "brier": m["brier"],
        "accuracy": m["accuracy"],
        "n_series": m["n"],
        "window": f"{window[0]}..{window[1]}",
        "settings": settings,
        "keep": keep,
    }
    if p_best is not None:
        row["delta_vs_best"], row["delta_low"], row["delta_high"] = bootstrap(
            y, p, p_best
        )
    return row


def write_log(rows, path=config.EXPERIMENTS):
    df = pd.DataFrame(rows).reindex(columns=LOG_COLUMNS)
    df.to_csv(path, index=False, float_format="%.4f")
    return df
