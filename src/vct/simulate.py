"""Map veto simulation and the Champions playoff bracket Monte Carlo."""

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from . import features
from .models import p_series, series_prob_seq, sigmoid

# VCT veto formats: (who, action); whatever is left after these is the decider
VETO_ORDER = {
    1: [
        ("A", "ban"),
        ("B", "ban"),
        ("A", "ban"),
        ("B", "ban"),
        ("A", "ban"),
        ("B", "ban"),
    ],
    3: [
        ("A", "ban"),
        ("B", "ban"),
        ("A", "pick"),
        ("B", "pick"),
        ("A", "ban"),
        ("B", "ban"),
    ],
    5: [
        ("A", "ban"),
        ("B", "ban"),
        ("A", "pick"),
        ("B", "pick"),
        ("A", "pick"),
        ("B", "pick"),
    ],
}


def map_pool(vetoes, before, n_matches=30, size=7):
    """The active map pool: the `size` maps seen most in the last n vetoes before a date."""
    recent = vetoes[vetoes.date < pd.Timestamp(before)]
    mids = recent.drop_duplicates("match_id").match_id.tail(n_matches)
    counts = recent[recent.match_id.isin(mids)].groupby("map").match_id.nunique()
    return list(counts.sort_values(ascending=False).index[:size])


class VetoHabits:
    """Per team and map: time-weighted (picks - bans) / (vetoes + 2)."""

    def fit(self, vetoes, ref_date, half_life=180):
        v = vetoes[
            (vetoes.date < pd.Timestamp(ref_date))
            & vetoes.by.notna()
            & (vetoes.by != "decider")
        ]
        w = features.time_weight(v.date, ref_date, half_life)
        v = v.assign(score=np.where(v.action == "pick", 1.0, -1.0) * w)
        self.score = v.groupby(["by", "map"])["score"].sum().to_dict()
        # each veto has one action per team per step; normalise by vetoes the team took part in
        n = pd.Series(w, index=v.by).groupby(level=0).sum() / 3
        self.n = n.to_dict()
        return self

    def get(self, team, mp):
        return self.score.get((team, mp), 0.0) / (self.n.get(team, 0.0) + 2)


def simulate_veto(
    model, a, b, best_of, pool, habits=None, kappa=1.0, rng=None, temperature=0.5
):
    """Return the maps played, in order. Team a vetoes first.

    Each team bans the map with the lowest (advantage + kappa * habit) and picks
    the highest. With rng set, choices are sampled from a softmax instead.
    """
    remaining = list(pool)
    picks = []
    for who, action in VETO_ORDER[best_of]:
        if len(remaining) <= 1:
            break
        team, opp = (a, b) if who == "A" else (b, a)
        u = np.array(
            [
                model.diff(team, opp, m)
                + (kappa * habits.get(team, m) if habits else 0.0)
                for m in remaining
            ]
        )
        if action == "ban":
            u = -u
        if rng is None:
            k = int(np.argmax(u))
        else:
            e = np.exp((u - u.max()) / temperature)
            k = int(rng.choice(len(u), p=e / e.sum()))
        m = remaining.pop(k)
        if action == "pick":
            picks.append(m)
    return picks + remaining[:1]


def veto_map_logits(model, a, b, best_of, pool, habits=None, kappa=1.0):
    """Map log-odds for team a in play order, from the deterministic veto."""
    maps = simulate_veto(model, a, b, best_of, pool, habits, kappa)
    d = [model.diff(a, b, m) for m in maps]
    return d + [d[-1]] * (best_of - len(d)) if d else [model.diff(a, b)] * best_of


# ---------------------------------------------------------------- series probability


class SeriesModel:
    """Turns a rating model into P(a wins a BoN), optionally via veto simulation."""

    def __init__(
        self,
        model,
        pool=None,
        habits=None,
        kappa=1.0,
        temperature=1.0,
        veto_samples=200,
        seed=0,
    ):
        self.model, self.pool, self.habits, self.kappa = model, pool, habits, kappa
        self.T = temperature
        self.veto_samples = veto_samples
        self.rng = np.random.default_rng(seed)
        self._cache = {}

    def prob(self, a, b, best_of):
        key = (a, b, best_of)
        if key not in self._cache:
            self._cache[key] = self._prob(a, b, best_of)
        return self._cache[key]

    def _prob(self, a, b, best_of):
        if not self.pool or not getattr(self.model, "map_rating", None):
            return p_series(sigmoid(self.T * self.model.diff(a, b)), best_of)
        # average over sampled vetoes; either team may veto first
        total = 0.0
        for i in range(self.veto_samples):
            first, second = (a, b) if i % 2 == 0 else (b, a)
            maps = simulate_veto(
                self.model,
                first,
                second,
                best_of,
                self.pool,
                self.habits,
                self.kappa,
                rng=self.rng,
            )
            ps = [sigmoid(self.T * self.model.diff(a, b, m)) for m in maps]
            ps += [ps[-1]] * (best_of - len(ps))
            total += series_prob_seq(ps, best_of)
        return total / self.veto_samples


class Scaled:
    """A rating model with every log-odds multiplied by a temperature T."""

    def __init__(self, base, T):
        self.base, self.T = base, T
        self.map_rating = getattr(base, "map_rating", None)

    def diff(self, a, b, map_name=None):
        return self.T * self.base.diff(a, b, map_name)


class Offset:
    """A rating model with fixed per-team shifts (used to fold market prices in)."""

    def __init__(self, base, offsets):
        self.base, self.offsets = base, offsets
        self.map_rating = getattr(base, "map_rating", None)

    def diff(self, a, b, map_name=None):
        return (
            self.base.diff(a, b, map_name)
            + self.offsets.get(a, 0.0)
            - self.offsets.get(b, 0.0)
        )


def market_offsets(model, pairs, best_of=3):
    """Per-team rating shifts so each pair's series probability equals the target.

    pairs: list of (a, b, target P(a wins)). The shift is split evenly between
    the two teams, so later-round matchups inherit the market's view as well.
    """
    off = {}
    for a, b, target in pairs:
        d0 = model.diff(a, b)
        d = brentq(lambda x: p_series(sigmoid(x), best_of) - target, -10, 10)
        off[a] = off.get(a, 0.0) + (d - d0) / 2
        off[b] = off.get(b, 0.0) - (d - d0) / 2
    return off


# ---------------------------------------------------------------- bracket


def run_bracket(qf, prob, rng=None):
    """8-team double elimination (VCT Champions playoffs).

    qf: four (team_a, team_b) upper quarterfinals in bracket order.
    prob(a, b, best_of) -> P(a wins). With rng=None every match goes to the favourite.
    """
    res = {}

    def m(key, a, b, bo=3):
        p = prob(a, b, bo)
        a_wins = p >= 0.5 if rng is None else rng.random() < p
        w, l, pw = (a, b, p) if a_wins else (b, a, 1 - p)
        res[key] = (w, l, pw)
        return w, l

    wA, lA = m("UB QF A", *qf[0])
    wB, lB = m("UB QF B", *qf[1])
    wC, lC = m("UB QF C", *qf[2])
    wD, lD = m("UB QF D", *qf[3])
    s1w, s1l = m("UB SF 1", wA, wB)
    s2w, s2l = m("UB SF 2", wC, wD)
    ufw, ufl = m("UB Final", s1w, s2w)
    l1w, _ = m("LB R1 top", lA, lB)
    l2w, _ = m("LB R1 bottom", lC, lD)
    r2a, _ = m("LB R2 top", s2l, l1w)
    r2b, _ = m("LB R2 bottom", s1l, l2w)
    lsw, _ = m("LB Semifinal", r2a, r2b)
    lfw, lfl = m("LB Final (Bo5)", ufl, lsw, 5)
    gfw, gfl = m("Grand Final (Bo5)", ufw, lfw, 5)
    res["_champion"], res["_runner_up"], res["_third"] = gfw, gfl, lfl
    return res


def title_odds(qf, prob, n=20_000, seed=2026):
    rng = np.random.default_rng(seed)
    sims = [run_bracket(qf, prob, rng) for _ in range(n)]
    teams = [t for pair in qf for t in pair]

    def share(keys):
        return pd.Series([s[k] for s in sims for k in keys]).value_counts() / n

    return (
        pd.DataFrame(
            {
                "P(champion)": share(["_champion"]),
                "P(grand final)": share(["_champion", "_runner_up"]),
                "P(top 3)": share(["_champion", "_runner_up", "_third"]),
            }
        )
        .reindex(teams)
        .fillna(0)
        .sort_values("P(champion)", ascending=False)
    )


def pick_sheet(qf, prob):
    picks = run_bracket(qf, prob, rng=None)
    sheet = pd.DataFrame(
        [
            {"match": k, "pick": w, "vs": l, "confidence": p}
            for k, (w, l, p) in ((k, v) for k, v in picks.items() if not k.startswith("_"))
        ]
    )
    sheet.index = range(1, len(sheet) + 1)
    return sheet, picks
