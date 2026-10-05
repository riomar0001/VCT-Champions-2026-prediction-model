"""Rating models: Bradley-Terry (with region and per-map terms), Glicko-2, market blend.

Every model exposes diff(a, b, map_name=None): the log-odds that team a wins
one map against team b. Series probabilities are built from those.
"""

from collections import defaultdict
from dataclasses import dataclass
from math import comb

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize

from . import config, features

EVENT_ORDER = {e[0]: i for i, e in enumerate(config.EVENTS)}


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, dtype=float)))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def p_series(p, best_of=3):
    """P(win a BoN) when every map is won independently with probability p."""
    return series_prob_seq([p] * best_of, best_of)


def series_prob_seq(ps, best_of):
    """P(win a BoN) when map k is won with probability ps[k] (maps in play order)."""
    need = best_of // 2 + 1
    # dist[i, j]: probability of being at i wins, j losses
    dist = {(0, 0): 1.0}
    win = 0.0
    for p in ps[:best_of]:
        nxt = {}
        for (i, j), q in dist.items():
            if i == need or j == need:
                continue
            nxt[(i + 1, j)] = nxt.get((i + 1, j), 0) + q * p
            nxt[(i, j + 1)] = nxt.get((i, j + 1), 0) + q * (1 - p)
        win += sum(q for (i, j), q in nxt.items() if i == need)
        dist = {k: v for k, v in nxt.items() if k[0] < need and k[1] < need}
    return win


def round_to_map(p):
    """P(win a map) when each round is won independently with probability p
    (first to 13, overtime won by two)."""
    p = np.asarray(p, dtype=float)
    q = 1 - p
    reg = sum(comb(12 + k, k) * p**13 * q**k for k in range(12))
    tie = comb(24, 12) * p**12 * q**12
    return reg + tie * p**2 / (p**2 + q**2)


# ---------------------------------------------------------------- Bradley-Terry


@dataclass(frozen=True)
class BTConfig:
    C: float = 1.0  # L2: penalty is sum(r**2) / (2C)
    half_life: float | None = 90  # days; None disables date decay
    event_decay: float | None = None  # legacy: decay ** events_ago (used if set)
    margin: float = 0.0  # weight *= 1 + margin * round margin
    target: str = "maps"  # "maps" (win/loss) or "rounds" (round share)
    region: bool = False  # add a region strength term
    region_C: float = 1.0
    intl_weight: float = 1.0  # extra weight on international maps
    per_map: bool = False  # add team x map adjustments
    map_C: float = 0.1
    roster_alpha: float = 0.0  # down-weight maps played by a since-changed roster


class BradleyTerry:
    def __init__(self, cfg: BTConfig = BTConfig()):
        self.cfg = cfg

    def fit(self, maps, ref_date, regions=None, lineup_table=None):
        c = self.cfg
        maps = maps.reset_index(drop=True)
        if c.event_decay is not None:
            order = maps.event_id.map(EVENT_ORDER).to_numpy()
            w = features.event_weight(order, order.max(), c.event_decay)
        else:
            w = features.time_weight(maps.date, ref_date, c.half_life)
        if c.margin:
            w = w * (1 + c.margin * features.round_margin(maps))
        if c.intl_weight != 1:
            w = w * np.where(maps.international, c.intl_weight, 1.0)
        if c.roster_alpha:
            w = w * features.roster_continuity(
                maps, lineup_table, ref_date, c.roster_alpha
            )

        if c.target == "rounds":
            n = (maps.rounds_a + maps.rounds_b).to_numpy()
            y = maps.rounds_a.to_numpy() / n
            w = w * n / 24  # one map ~ 24 rounds, keeps C on a similar scale
        else:
            y = maps.y.to_numpy().astype(float)

        self.teams = sorted(set(maps.team_a) | set(maps.team_b))
        self.regions = dict(regions or {})
        for col_t, col_r in (("team_a", "region_a"), ("team_b", "region_b")):
            self.regions.update(dict(zip(maps[col_t], maps[col_r])))
        self.region_names = sorted(set(self.regions.values())) if c.region else []
        self.map_names = sorted(maps["map"].unique()) if c.per_map else []

        ti = {t: k for k, t in enumerate(self.teams)}
        nt, nr, nm = len(self.teams), len(self.region_names), len(self.map_names)
        ri = {r: nt + k for k, r in enumerate(self.region_names)}
        mi = {m: k for k, m in enumerate(self.map_names)}

        a = maps.team_a.map(ti).to_numpy()
        b = maps.team_b.map(ti).to_numpy()
        rows, cols, vals = [], [], []
        idx = np.arange(len(maps))
        rows += [idx, idx]
        cols += [a, b]
        vals += [np.ones(len(maps)), -np.ones(len(maps))]
        if c.region:
            rows += [idx, idx]
            cols += [maps.region_a.map(ri).to_numpy(), maps.region_b.map(ri).to_numpy()]
            vals += [np.ones(len(maps)), -np.ones(len(maps))]
        if c.per_map:
            m = maps["map"].map(mi).to_numpy()
            off = nt + nr
            rows += [idx, idx]
            cols += [off + a * nm + m, off + b * nm + m]
            vals += [np.ones(len(maps)), -np.ones(len(maps))]
        npar = nt + nr + nt * nm
        X = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(len(maps), npar),
        )

        inv_c = np.concatenate(
            [
                np.full(nt, 1 / c.C),
                np.full(nr, 1 / c.region_C),
                np.full(nt * nm, 1 / c.map_C),
            ]
        )

        def f(theta):
            z = X @ theta
            loss = np.sum(w * (np.logaddexp(0, z) - y * z)) + 0.5 * np.sum(
                inv_c * theta**2
            )
            grad = X.T @ (w * (sigmoid(z) - y)) + inv_c * theta
            return loss, grad

        res = minimize(
            f, np.zeros(npar), jac=True, method="L-BFGS-B", options={"maxiter": 2000}
        )
        th = res.x
        self.rating = dict(zip(self.teams, th[:nt]))
        self.region_rating = dict(zip(self.region_names, th[nt : nt + nr]))
        self.map_rating = {}
        if c.per_map:
            adj = th[nt + nr :].reshape(nt, nm)
            self.map_rating = {
                (t, mp): adj[ti[t], mi[mp]] for t in self.teams for mp in self.map_names
            }
        return self

    def strength(self, team, map_name=None):
        s = self.rating.get(team, 0.0)
        if self.cfg.region:
            s += self.region_rating.get(self.regions.get(team), 0.0)
        if map_name is not None:
            s += self.map_rating.get((team, map_name), 0.0)
        return s

    def diff(self, a, b, map_name=None):
        d = self.strength(a, map_name) - self.strength(b, map_name)
        if self.cfg.target == "rounds":
            return float(logit(round_to_map(sigmoid(d))))
        return d

    def table(self):
        return (
            pd.DataFrame({"team": self.teams})
            .assign(
                region=lambda t: t.team.map(self.regions),
                rating=lambda t: [self.strength(x) for x in t.team],
            )
            .sort_values("rating", ascending=False)
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------- Glicko-2

_SCALE = 173.7178


class Glicko2:
    """Glicko-2 on map results, one rating period per `period_days`."""

    def __init__(self, tau=0.5, period_days=7, init_rd=350, init_vol=0.06):
        self.tau, self.period_days = tau, period_days
        self.init_phi, self.init_vol = init_rd / _SCALE, init_vol

    @staticmethod
    def _g(phi):
        return 1 / np.sqrt(1 + 3 * phi**2 / np.pi**2)

    def fit(self, maps, ref_date=None, **_):
        mu, phi, vol = {}, {}, {}
        start = maps.date.min()
        period = ((maps.date - start).dt.days // self.period_days).to_numpy()
        games_by_period = defaultdict(
            lambda: defaultdict(list)
        )  # period -> team -> [(opp, score)]
        for k, a, b, y in zip(period, maps.team_a, maps.team_b, maps.y):
            games_by_period[k][a].append((b, y))
            games_by_period[k][b].append((a, 1 - y))
        for k in sorted(games_by_period):
            games = games_by_period[k]
            for t in games:
                mu.setdefault(t, 0.0)
                phi.setdefault(t, self.init_phi)
                vol.setdefault(t, self.init_vol)
            snap_mu, snap_phi = dict(mu), dict(phi)
            for t in mu:
                if t not in games:
                    phi[t] = np.sqrt(phi[t] ** 2 + vol[t] ** 2)
                    continue
                opp = [o for o, _ in games[t]]
                s = np.array([sc for _, sc in games[t]], dtype=float)
                mj = np.array([snap_mu[o] for o in opp])
                gj = self._g(np.array([snap_phi[o] for o in opp]))
                e = 1 / (1 + np.exp(-gj * (snap_mu[t] - mj)))
                v = 1 / np.sum(gj**2 * e * (1 - e))
                delta = v * np.sum(gj * (s - e))
                vol[t] = self._new_vol(snap_phi[t], vol[t], v, delta)
                phi_star = np.sqrt(snap_phi[t] ** 2 + vol[t] ** 2)
                phi[t] = 1 / np.sqrt(1 / phi_star**2 + 1 / v)
                mu[t] = snap_mu[t] + phi[t] ** 2 * np.sum(gj * (s - e))
        self.mu, self.phi = mu, phi
        return self

    def _new_vol(self, phi, sigma, v, delta, eps=1e-6):
        a = np.log(sigma**2)
        tau = self.tau

        def f(x):
            ex = np.exp(x)
            return (
                ex * (delta**2 - phi**2 - v - ex) / (2 * (phi**2 + v + ex) ** 2)
                - (x - a) / tau**2
            )

        A = a
        if delta**2 > phi**2 + v:
            B = np.log(delta**2 - phi**2 - v)
        else:
            k = 1
            while f(a - k * tau) < 0:
                k += 1
            B = a - k * tau
        fa, fb = f(A), f(B)
        while abs(B - A) > eps:
            C = A + (A - B) * fa / (fb - fa)
            fc = f(C)
            if fc * fb <= 0:
                A, fa = B, fb
            else:
                fa /= 2
            B, fb = C, fc
        return float(np.exp(A / 2))

    def diff(self, a, b, map_name=None):
        ma, mb = self.mu.get(a, 0.0), self.mu.get(b, 0.0)
        pa, pb = self.phi.get(a, self.init_phi), self.phi.get(b, self.init_phi)
        return float(self._g(np.sqrt(pa**2 + pb**2)) * (ma - mb))

    def table(self):
        return (
            pd.DataFrame(
                {
                    "team": list(self.mu),
                    "rating": list(self.mu.values()),
                    "rd": [self.phi[t] * _SCALE for t in self.mu],
                }
            )
            .sort_values("rating", ascending=False)
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------- market


def implied_prob(odds_a, odds_b):
    """De-vigged probability that side a wins, from decimal odds."""
    ia, ib = 1 / np.asarray(odds_a, float), 1 / np.asarray(odds_b, float)
    return ia / (ia + ib)


def blend(p_model, p_market, w):
    """w * model + (1 - w) * market; falls back to the model where there is no price."""
    p_model = np.asarray(p_model, float)
    p_market = np.asarray(p_market, float)
    return np.where(np.isnan(p_market), p_model, w * p_model + (1 - w) * p_market)
