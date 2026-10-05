"""Observation weights (time, round margin, roster continuity) and series features."""

import numpy as np
import pandas as pd


def time_weight(dates, ref_date, half_life):
    """0.5 ** (days_ago / half_life); all ones if half_life is None."""
    if half_life is None:
        return np.ones(len(dates))
    days = (
        pd.Timestamp(ref_date) - pd.to_datetime(pd.Series(dates))
    ).dt.days.to_numpy()
    return 0.5 ** (np.maximum(days, 0) / half_life)


def event_weight(event_orders, ref_order, decay):
    """The original notebook's scheme: decay ** events_ago."""
    return decay ** (ref_order - np.asarray(event_orders, dtype=float))


def round_margin(maps):
    """|rounds_a - rounds_b| / total rounds."""
    ra, rb = maps.rounds_a.to_numpy(), maps.rounds_b.to_numpy()
    return np.abs(ra - rb) / (ra + rb)


# ---------------------------------------------------------------- rosters


def lineups(players):
    """One row per (match_id, team): the player ids on its first map."""
    first = players[
        players.game_id == players.groupby("match_id").game_id.transform("min")
    ]
    out = (
        first.groupby(["match_id", "team"])
        .agg(
            date=("date", "first"),
            roster=("player_id", lambda s: frozenset(s.dropna().astype(int))),
        )
        .reset_index()
    )
    return out.sort_values(["date", "match_id"]).reset_index(drop=True)


def current_rosters(lineup_table, before):
    """Each team's most recent roster strictly before a date."""
    past = lineup_table[lineup_table.date < pd.Timestamp(before)]
    return past.groupby("team")["roster"].last().to_dict()


def roster_continuity(maps, lineup_table, ref_date, alpha):
    """Weight multiplier per map: (1 - alpha * changed_a / 5) * (1 - alpha * changed_b / 5).

    A team that has since replaced k players gets its older maps down-weighted,
    so the L2 penalty pulls its rating partway back toward zero.
    """
    if not alpha or lineup_table is None:
        return np.ones(len(maps))
    now = current_rosters(lineup_table, ref_date)
    then = lineup_table.set_index(["match_id", "team"])["roster"].to_dict()

    def factor(mid, team):
        old, new = then.get((mid, team)), now.get(team)
        if not old or not new:
            return 1.0
        changed = 5 - min(len(old & new), 5)
        return max(1 - alpha * changed / 5, 0.05)

    fa = [factor(m, t) for m, t in zip(maps.match_id, maps.team_a)]
    fb = [factor(m, t) for m, t in zip(maps.match_id, maps.team_b)]
    return np.array(fa) * np.array(fb)


def roster_changes(lineup_table, team, before, after):
    """Players changed between a team's roster at two dates (0-5)."""
    old = current_rosters(lineup_table, after).get(team)
    new = current_rosters(lineup_table, before).get(team)
    if not old or not new:
        return 0
    return 5 - min(len(old & new), 5)


# ---------------------------------------------------------------- series features (phase 4b)


def _team_maps(maps, team, before):
    m = maps[(maps.date < before) & ((maps.team_a == team) | (maps.team_b == team))]
    won = np.where(m.team_a == team, m.y, 1 - m.y)
    return m, won


def form(maps, team, before, n=10):
    """Map win rate over the last n maps, shrunk toward 0.5."""
    _, won = _team_maps(maps, team, before)
    won = won[-n:]
    return (won.sum() + 2) / (len(won) + 4)


def head_to_head(maps, a, b, before):
    m = maps[
        (maps.date < before)
        & (
            ((maps.team_a == a) & (maps.team_b == b))
            | ((maps.team_a == b) & (maps.team_b == a))
        )
    ]
    won = np.where(m.team_a == a, m.y, 1 - m.y)
    return (won.sum() + 1) / (len(won) + 2), len(won)


def rest_days(series, team, before):
    s = series[
        (series.datetime < before) & ((series.team_a == team) | (series.team_b == team))
    ]
    return min((pd.Timestamp(before) - s.datetime.max()).days, 60) if len(s) else 60


def map_pool_overlap(maps, a, b, before, n=30):
    """Share of maps both teams played in their last n maps (rough comfort overlap)."""
    pa = set(_team_maps(maps, a, before)[0].tail(n)["map"])
    pb = set(_team_maps(maps, b, before)[0].tail(n)["map"])
    return len(pa & pb) / max(len(pa | pb), 1)
