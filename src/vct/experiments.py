"""Run the improvement plan's experiments and write experiments.csv.

Each step is compared with the current best on the tuning window and kept only
if it lowers log loss. The final model is then scored once on the holdout window.

    uv run python -m vct.experiments
"""

from dataclasses import replace

import numpy as np
import pandas as pd

from . import clean, config, features
from . import evaluate as ev
from .models import BTConfig, logit

PHASE0 = BTConfig(
    C=0.2, half_life=None, event_decay=0.5
)  # original notebook (C doubled: it fit both orientations)


def load_context():
    return ev.Context(
        clean.load("series"),
        clean.load("maps"),
        clean.load("players"),
        clean.load("vetoes"),
    )


class Runner:
    def __init__(self, ctx, window=ev.TUNE, log=print):
        self.ctx, self.window, self.log = ctx, window, log
        self.rows, self.cache = [], {}
        self.best_p = self.best_name = self.best_cfg = None
        self.best_bt = None
        self.use_temperature = False

    def backtest(self, key, factory):
        if key not in self.cache:
            self.cache[key] = ev.walk_forward(self.ctx, factory, self.window)
        return self.cache[key]

    def record(self, name, bt, p, settings="", candidate=True):
        row = ev.summarise(name, bt.y, p, self.best_p, settings, self.window)
        keep = candidate and (
            self.best_p is None or row["log_loss"] < ev.log_loss(bt.y, self.best_p)
        )
        row["keep"] = "yes" if keep else ("baseline" if not candidate else "no")
        self.rows.append(row)
        self.log(
            f"  {name:<48} ll={row['log_loss']:.4f}  brier={row['brier']:.4f}  "
            f"acc={row['accuracy']:.1%}  n={row['n_series']}  keep={row['keep']}"
        )
        return keep

    def try_bt(self, name, cfg, veto=False, kappa=1.0):
        bt = self.backtest(("bt", cfg, veto, kappa), ev.bt_predictor(cfg, veto, kappa))
        if self.record(name, bt, bt.p, settings=f"{cfg} veto={veto} kappa={kappa}"):
            self.best_p, self.best_name, self.best_cfg, self.best_bt = (
                bt.p.to_numpy(),
                name,
                (cfg, veto, kappa),
                bt,
            )
        return bt


def run(log=print):
    ctx = load_context()
    r = Runner(ctx, log=log)
    log(f"Tuning window {ev.TUNE[0]}..{ev.TUNE[1]}")

    # ---- baselines
    base = r.backtest(("bt", PHASE0, False, 1.0), ev.bt_predictor(PHASE0))
    r.record("Baseline: coin flip", base, np.full(len(base), 0.5), candidate=False)
    r.try_bt("Baseline: Phase 0 model (event decay 0.5, C=0.2)", PHASE0)

    # ---- 2b: date decay (C re-tuned now that there is 25x more data)
    log("Phase 2b: date-based decay")
    for hl in (60, 90, 120, 180, None):
        for C in (0.3, 1.0, 3.0):
            r.try_bt(f"2b: half_life={hl}, C={C}", BTConfig(C=C, half_life=hl))
    cfg = r.best_cfg[0]

    # ---- 2a: round margin
    log("Phase 2a: round margin")
    for k in (0.5, 1.0, 2.0):
        r.try_bt(f"2a: margin weight 1+{k}*margin", replace(cfg, margin=k))
    cfg = r.best_cfg[0]
    for C in (0.1, 0.3, 1.0):
        r.try_bt(
            f"2a alt: round-share target, C={C}",
            replace(cfg, target="rounds", margin=0.0, C=C),
        )
    cfg = r.best_cfg[0]

    # ---- 2c: regions
    log("Phase 2c: regional strength")
    for rc in (0.3, 1.0):
        r.try_bt(
            f"2c: region term, region_C={rc}", replace(cfg, region=True, region_C=rc)
        )
    cfg = r.best_cfg[0]
    for iw in (1.5, 2.0, 3.0):
        r.try_bt(f"2c: international weight x{iw}", replace(cfg, intl_weight=iw))
    cfg = r.best_cfg[0]

    # ---- Glicko-2 comparison
    log("Phase 2b alt: Glicko-2")
    for tau in (0.3, 0.6):
        g = r.backtest(("glicko", tau), ev.glicko_predictor(tau=tau))
        if r.record(
            f"2b alt: Glicko-2 tau={tau}", g, g.p, settings=f"tau={tau} period=7d"
        ):
            r.best_p, r.best_name, r.best_bt = g.p.to_numpy(), f"Glicko-2 tau={tau}", g
            r.best_cfg = (("glicko", tau), False, 1.0)

    # ---- 3: per-map ratings and veto simulation
    log("Phase 3: per-map ratings + veto simulation")
    for mc in (0.03, 0.1, 0.3):
        r.try_bt(
            f"3: per-map ratings, map_C={mc} (no veto)",
            replace(cfg, per_map=True, map_C=mc),
        )
    best = r.best_cfg[0]
    mc = best.map_C if isinstance(best, BTConfig) and best.per_map else 0.1
    for kappa in (0.0, 1.0, 3.0):
        r.try_bt(
            f"3: per-map + veto sim, map_C={mc}, kappa={kappa}",
            replace(cfg, per_map=True, map_C=mc),
            veto=True,
            kappa=kappa,
        )

    # ---- 4a: roster changes (rating models only)
    if isinstance(r.best_cfg[0], BTConfig):
        cfg, veto, kappa = r.best_cfg
        log("Phase 4a: roster continuity")
        for a in (0.25, 0.5, 1.0):
            r.try_bt(
                f"4a: roster pull-back alpha={a}",
                replace(cfg, roster_alpha=a),
                veto,
                kappa,
            )

    # ---- 2d: calibration and market blend (on the best model so far)
    log("Phase 2d: temperature scaling + market blend")
    bt = r.best_bt
    p_cal, Ts = ev.walk_forward_temperature(bt)
    if r.record(
        "2d: walk-forward temperature scaling",
        bt,
        p_cal,
        settings=f"final T={Ts[-1]:.3f}",
    ):
        r.best_p, r.use_temperature = p_cal, True
    has = bt.p_market.notna().to_numpy()
    log(f"  market prices on {has.sum()}/{len(bt)} series")
    p_bl, ws = ev.walk_forward_blend(bt, r.best_p)
    r.record(
        "2d: model+market blend, w tuned walk-forward",
        bt,
        p_bl,
        settings=f"final w={ws[-1]:.2f}",
        candidate=False,
    )

    sub = bt[has]
    for name, p in [
        ("Baseline: market (series with prices)", sub.p_market),
        ("Best model (series with prices)", r.best_p[has]),
        ("Blend (series with prices)", p_bl[has]),
    ]:
        r.rows.append(
            {
                **ev.summarise(name, sub.y, p, sub.p_market, window=ev.TUNE),
                "keep": "comparison",
            }
        )
        log(f"  {name:<48} ll={r.rows[-1]['log_loss']:.4f}  n={len(sub)}")

    # ---- 4b: feature model
    log("Phase 4b: LightGBM feature model")
    p_lgb, mask = lgbm_walk_forward(ctx, bt, r.best_p)
    if mask.sum() >= 100:
        sub = bt[mask]
        r.rows.append(
            {
                **ev.summarise(
                    "4b: LightGBM + Platt (its test rows)",
                    sub.y,
                    p_lgb[mask],
                    r.best_p[mask],
                    window=ev.TUNE,
                ),
                "keep": "see delta",
            }
        )
        r.rows.append(
            {
                **ev.summarise(
                    "   best rating model on the same rows",
                    sub.y,
                    r.best_p[mask],
                    window=ev.TUNE,
                ),
                "keep": "comparison",
            }
        )
        log(
            f"  LightGBM ll={ev.log_loss(sub.y, p_lgb[mask]):.4f} vs best {ev.log_loss(sub.y, r.best_p[mask]):.4f}"
            f" on {mask.sum()} series"
        )

    # ---- holdout
    log(f"Holdout {ev.HOLDOUT[0]}..{ev.HOLDOUT[1]} (never used for tuning)")
    final = holdout(ctx, r, log)

    rows = (
        [
            {
                "date": "2026-10-05",
                "change": "Original notebook (40 hand-entered series)",
                "log_loss": 0.688,
                "accuracy": 0.41,
                "n_series": 32,
                "keep": "historical",
                "window": "events 5-7",
            }
        ]
        + r.rows
        + final
    )
    df = ev.write_log(rows)
    production = production_settings(bt, log)
    pd.to_pickle(
        {
            "best": r.best_cfg,
            "name": r.best_name,
            "tuning_bt": bt,
            "tuning_p": r.best_p,
            "tuning_temperature": ev.fit_temperature(bt) if r.use_temperature else 1.0,
            "tuning_blend_w": ws[-1],
            **production,
        },
        config.PROCESSED / "best_model.pkl",
    )
    log(f"wrote {config.EXPERIMENTS}")
    return df, r


def production_settings(tuning_bt, log=print):
    """Temperature and blend weight for live predictions, fitted on every backtest
    prediction (tuning + holdout). Only done after the holdout has been scored."""
    hold = pd.read_pickle(config.PROCESSED / "holdout_predictions.pkl")
    cols = ["block", "y", "best_of", "d", "p_market"]
    allbt = pd.concat([tuning_bt[cols], hold[cols]], ignore_index=True)
    T = ev.fit_temperature(allbt)
    p = ev.apply_temperature(allbt, T)
    has = allbt.p_market.notna().to_numpy()
    grid = np.linspace(0, 1, 21)
    w = min(grid, key=lambda w: ev.log_loss(allbt.y[has], w * p[has] + (1 - w) * allbt.p_market[has]))
    log(f"production settings on {len(allbt)} backtest series: temperature={T:.3f}, model weight w={w:.2f}")
    return {"temperature": float(T), "blend_w": float(w), "n_calibration": len(allbt)}


def holdout(ctx, r, log):
    cfg, veto, kappa = r.best_cfg
    fac = (
        ev.glicko_predictor(tau=cfg[1])
        if isinstance(cfg, tuple)
        else ev.bt_predictor(cfg, veto, kappa)
    )
    bt = ev.walk_forward(ctx, fac, ev.HOLDOUT)
    base = ev.walk_forward(ctx, ev.bt_predictor(PHASE0), ev.HOLDOUT)
    # temperature/blend fitted on tuning-window predictions, then frozen
    T = ev.fit_temperature(r.best_bt)
    p_cal = ev.apply_temperature(bt, T)
    p_model = p_cal if r.use_temperature else bt.p.to_numpy()
    has = r.best_bt.p_market.notna().to_numpy()
    grid = np.linspace(0, 1, 21)
    w = min(
        grid,
        key=lambda w: ev.log_loss(
            r.best_bt.y[has], w * r.best_p[has] + (1 - w) * r.best_bt.p_market[has]
        ),
    )
    p_bl = np.where(bt.p_market.isna(), p_model, w * p_model + (1 - w) * bt.p_market)

    rows = []
    for name, p in [
        ("HOLDOUT coin flip", np.full(len(bt), 0.5)),
        ("HOLDOUT Phase 0 model", base.p),
        (f"HOLDOUT final model ({r.best_name})", p_model),
        (f"HOLDOUT final model + market blend (w={w:.2f})", p_bl),
    ]:
        rows.append(
            {
                **ev.summarise(name, bt.y, p, base.p, window=ev.HOLDOUT),
                "keep": "holdout",
            }
        )
        log(
            f"  {name:<48} ll={rows[-1]['log_loss']:.4f}  brier={rows[-1]['brier']:.4f}  "
            f"acc={rows[-1]['accuracy']:.1%}  n={len(bt)}"
        )
    hs = bt.p_market.notna()
    rows.append(
        {
            **ev.summarise(
                "HOLDOUT market (series with prices)",
                bt.y[hs],
                bt.p_market[hs],
                window=ev.HOLDOUT,
            ),
            "keep": "holdout",
        }
    )
    rows.append(
        {
            **ev.summarise(
                "HOLDOUT final model (series with prices)",
                bt.y[hs],
                p_model[hs],
                bt.p_market[hs],
                window=ev.HOLDOUT,
            ),
            "keep": "holdout",
        }
    )
    log(
        f"  market ll={rows[-2]['log_loss']:.4f} vs model {rows[-1]['log_loss']:.4f} on {hs.sum()} priced series"
    )
    bt = bt.assign(p_final=p_model, p_blend=p_bl, p_phase0=base.p.to_numpy())
    bt.to_pickle(config.PROCESSED / "holdout_predictions.pkl")
    return rows


# ---------------------------------------------------------------- phase 4b


def feature_table(ctx, bt, p_model):
    s = ctx.series.set_index("match_id")
    rows = []
    for r, p in zip(bt.itertuples(), p_model):
        t = s.loc[r.match_id, "datetime"]
        a, b = r.team_a, r.team_b
        h2h, n_h2h = features.head_to_head(ctx.maps, a, b, r.date)
        rows.append(
            {
                "rating_gap": float(logit(p)),
                "form_diff": features.form(ctx.maps, a, r.date)
                - features.form(ctx.maps, b, r.date),
                "h2h": h2h - 0.5,
                "n_h2h": n_h2h,
                "pool_overlap": features.map_pool_overlap(ctx.maps, a, b, r.date),
                "rest_diff": features.rest_days(ctx.series, a, t)
                - features.rest_days(ctx.series, b, t),
                "international": int(s.loc[r.match_id, "international"]),
                "cross_region": int(
                    s.loc[r.match_id, "region_a"] != s.loc[r.match_id, "region_b"]
                ),
                "best_of": r.best_of,
            }
        )
    return pd.DataFrame(rows)


def lgbm_walk_forward(ctx, bt, p_model, min_train=250):
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression

    X = feature_table(ctx, bt, p_model)
    y = bt.y.to_numpy()
    month = bt.date.dt.to_period("M").to_numpy()
    out, mask = np.full(len(bt), np.nan), np.zeros(len(bt), bool)
    for mth in np.unique(month):
        tr = month < mth
        if tr.sum() < min_train:
            continue
        te = month == mth
        # last 25% of the training rows (by time) calibrate the booster
        cut = np.flatnonzero(tr)[int(tr.sum() * 0.75)]
        fit_idx, cal_idx = (
            np.flatnonzero(tr)[np.flatnonzero(tr) < cut],
            np.flatnonzero(tr)[np.flatnonzero(tr) >= cut],
        )
        m = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=7,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            verbose=-1,
        )
        m.fit(X.iloc[fit_idx], y[fit_idx])
        raw = lambda idx: logit(m.predict_proba(X.iloc[idx])[:, 1]).reshape(-1, 1)
        platt = LogisticRegression(C=1.0).fit(raw(cal_idx), y[cal_idx])
        out[te] = platt.predict_proba(raw(np.flatnonzero(te)))[:, 1]
        mask |= te
    return out, mask


if __name__ == "__main__":
    run()
