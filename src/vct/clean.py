"""Turn scraped tables into the processed dataset, and validate it.

Outputs (data/processed/):
    series.parquet   one row per completed series
    maps.parquet     one row per played map (schema from the improvement plan)
    players.parquet  one row per player per map
    vetoes.parquet   one row per veto step
    upcoming.parquet scheduled series with known teams, plus market odds
"""

import numpy as np
import pandas as pd

from . import config

OVERROUND_DEFAULT = 1.07


def load_interim():
    d = config.DATA / "interim"
    return {
        k: pd.read_parquet(d / f"{k}.parquet")
        for k in ("series", "maps", "players", "odds", "veto")
    }


# ---------------------------------------------------------------- team names


def build_aliases(series, players):
    """Map every spelling seen to one code per VLR team id.

    The code is the tag VLR prints next to the team's players. Rows in
    team_aliases.csv with source=manual win, so hand fixes survive a rebuild.
    """
    tags = (
        players.merge(series[["match_id", "team1_id", "team2_id"]], on="match_id")
        .assign(team_id=lambda d: np.where(d.side == 1, d.team1_id, d.team2_id))
        .query("tag != ''")
        .groupby("team_id")["tag"]
        .agg(lambda s: s.mode().iat[0])
    )

    names = (
        pd.concat(
            [
                series[["team1_id", "team1", "datetime_utc"]].set_axis(
                    ["team_id", "name", "dt"], axis=1
                ),
                series[["team2_id", "team2", "datetime_utc"]].set_axis(
                    ["team_id", "name", "dt"], axis=1
                ),
            ]
        )
        .dropna(subset=["team_id"])
        .sort_values("dt")
    )
    names["team_id"] = names.team_id.astype(int)
    latest = names.groupby("team_id")["name"].last()

    codes = {tid: tags.get(tid, latest[tid]) for tid in latest.index}
    # two different teams with the same tag: keep the most recent one bare
    clash = pd.Series(codes).duplicated(keep=False)
    for tid in clash[clash].index:
        later = [t for t, c in codes.items() if c == codes[tid]]
        last_seen = names[names.team_id.isin(later)].groupby("team_id")["dt"].max()
        if tid != last_seen.idxmax():
            codes[tid] = f"{codes[tid]}-{tid}"

    rows = [
        {"alias": n, "team_id": tid, "code": codes[tid], "name": latest[tid]}
        for tid, n in names[["team_id", "name"]]
        .drop_duplicates()
        .itertuples(index=False)
    ]
    rows += [
        {"alias": c, "team_id": tid, "code": c, "name": latest[tid]}
        for tid, c in codes.items()
    ]
    auto = (
        pd.DataFrame(rows).drop_duplicates(["alias", "team_id"]).assign(source="auto")
    )

    if config.ALIASES.exists():
        manual = pd.read_csv(config.ALIASES).query("source == 'manual'")
        auto = pd.concat([manual, auto]).drop_duplicates(
            ["alias", "team_id"], keep="first"
        )
    return auto.sort_values(["code", "alias"]).reset_index(drop=True)


def team_code_map(aliases):
    return aliases.drop_duplicates("team_id").set_index("team_id")["code"].to_dict()


def resolve(name, aliases):
    """Code for a free-text team name, or None if it is unknown."""
    hit = aliases.loc[aliases.alias.str.casefold() == str(name).casefold(), "code"]
    return hit.iat[0] if len(hit) else None


# ---------------------------------------------------------------- market odds


def market_probs(odds):
    """Median de-vigged P(team1 wins) per match.

    Finished matches only show the winner's price, so those are de-vigged
    with the median overround seen on two-sided prices.
    """
    if odds.empty:
        return (
            pd.DataFrame(columns=["match_id", "p_market", "n_books", "two_sided"]),
            OVERROUND_DEFAULT,
        )
    two = (
        odds[odds.two_sided]
        .pivot_table(
            index=["match_id", "book"], columns="side", values="odds", aggfunc="first"
        )
        .dropna()
    )
    overround = (
        float(np.median(1 / two[1] + 1 / two[2])) if len(two) else OVERROUND_DEFAULT
    )

    rows = []
    for mid, g in odds.groupby("match_id"):
        tg = (
            g[g.two_sided]
            .pivot_table(index="book", columns="side", values="odds")
            .dropna()
        )
        if len(tg):
            p1 = (1 / tg[1]) / (1 / tg[1] + 1 / tg[2])
            rows.append((mid, float(p1.median()), len(tg), True))
        else:
            side = g.side.mode().iat[0]
            p = float(np.median(1 / g[g.side == side].odds)) / overround
            p = float(np.clip(p, 0.02, 0.98))
            rows.append((mid, p if side == 1 else 1 - p, len(g), False))
    return (
        pd.DataFrame(rows, columns=["match_id", "p_market", "n_books", "two_sided"]),
        overround,
    )


# ---------------------------------------------------------------- build


def build(write=True, log=print):
    t = load_interim()
    s, maps, players, odds, veto = (
        t["series"],
        t["maps"],
        t["players"],
        t["odds"],
        t["veto"],
    )
    s = s.drop_duplicates("match_id", keep="last")

    aliases = build_aliases(s[s.status == "final"], players)
    code = team_code_map(aliases)
    s["team_a"] = s.team1_id.map(code)
    s["team_b"] = s.team2_id.map(code)
    s["datetime"] = pd.to_datetime(s.datetime_utc)
    s["date"] = s.datetime.dt.normalize()

    # a team's region is where it plays its league games
    league = pd.concat(
        [
            s[["team_a", "event_region"]].set_axis(["team", "region"], axis=1),
            s[["team_b", "event_region"]].set_axis(["team", "region"], axis=1),
        ]
    )
    league = league[league.region != "International"]
    region = league.groupby("team")["region"].agg(lambda r: r.mode().iat[0]).to_dict()
    s["region_a"] = s.team_a.map(region).fillna("Unknown")
    s["region_b"] = s.team_b.map(region).fillna("Unknown")
    s["international"] = s.event_region == "International"
    s["lan"] = True  # every event in config.EVENTS is a LAN event

    probs, overround = market_probs(odds)
    s = s.merge(probs, on="match_id", how="left")
    order = {e[0]: i for i, e in enumerate(config.EVENTS)}
    s["event_order"] = s.event_id.map(order)

    done = s[(s.status == "final") & s.score1.notna()].copy()
    done["score_a"], done["score_b"] = done.score1.astype(int), done.score2.astype(int)
    done["winner"] = np.where(done.score_a > done.score_b, done.team_a, done.team_b)
    done["y"] = (done.score_a > done.score_b).astype(int)
    # VLR occasionally omits "Bo3" on old pages; infer it from the score
    done["best_of"] = done.best_of.fillna(
        2 * done[["score_a", "score_b"]].max(axis=1) - 1
    ).astype(int)
    done = done.sort_values(["datetime", "match_id"]).reset_index(drop=True)
    upcoming = s[(s.status != "final") & s.team_a.notna() & s.team_b.notna()].copy()

    series_cols = [
        "match_id",
        "date",
        "datetime",
        "event_id",
        "event",
        "event_order",
        "event_region",
        "stage",
        "team_a",
        "team_b",
        "region_a",
        "region_b",
        "score_a",
        "score_b",
        "best_of",
        "winner",
        "y",
        "international",
        "lan",
        "patch",
        "p_market",
        "n_books",
        "two_sided",
        "veto",
        "url",
    ]
    series = done[series_cols]

    m = maps.merge(
        done[
            [
                "match_id",
                "date",
                "event",
                "event_id",
                "team_a",
                "team_b",
                "region_a",
                "region_b",
                "best_of",
                "lan",
                "international",
            ]
        ],
        on="match_id",
    )
    m = m.rename(columns={"rounds1": "rounds_a", "rounds2": "rounds_b"})
    m["picked_by"] = np.select(
        [m.pick_side == 1, m.pick_side == 2], [m.team_a, m.team_b], "decider"
    )
    m["y"] = (m.rounds_a > m.rounds_b).astype(int)
    m = (
        m[
            [
                "match_id",
                "game_id",
                "map_order",
                "date",
                "event",
                "event_id",
                "region_a",
                "region_b",
                "team_a",
                "team_b",
                "map",
                "rounds_a",
                "rounds_b",
                "y",
                "picked_by",
                "best_of",
                "lan",
                "international",
            ]
        ]
        .sort_values(["date", "match_id", "map_order"])
        .reset_index(drop=True)
    )

    p = players.merge(done[["match_id", "date", "team_a", "team_b"]], on="match_id")
    p["team"] = np.where(p.side == 1, p.team_a, p.team_b)
    p = p.drop(columns=["team_a", "team_b"])

    v = veto.merge(done[["match_id", "date", "team_a", "team_b"]], on="match_id")
    v["by"] = [_veto_team(r, aliases) for r in v.itertuples()]
    v = v[["match_id", "date", "step", "action", "map", "team", "by"]]

    issues = validate(series, m, aliases, s)
    if write:
        config.PROCESSED.mkdir(parents=True, exist_ok=True)
        aliases.to_csv(config.ALIASES, index=False)
        for name, df in [
            ("series", series),
            ("maps", m),
            ("players", p),
            ("vetoes", v),
            ("upcoming", upcoming),
        ]:
            df.to_parquet(config.PROCESSED / f"{name}.parquet", index=False)
    log(
        f"{len(series)} series, {len(m)} maps, {series[['team_a', 'team_b']].stack().nunique()} teams; "
        f"market odds on {series.p_market.notna().sum()} series (overround {overround:.3f})"
    )
    if len(issues):
        log(issues.groupby(["severity", "check"]).size().to_string())
    return dict(
        series=series,
        maps=m,
        players=p,
        vetoes=v,
        upcoming=upcoming,
        aliases=aliases,
        issues=issues,
    )


def _veto_team(r, aliases):
    if r.action == "decider":
        return "decider"
    code = resolve(r.team, aliases)
    if code in (r.team_a, r.team_b):
        return code
    # the veto uses a tag we have not seen; match on prefix as a fallback
    for c in (r.team_a, r.team_b):
        if str(c).casefold().startswith(str(r.team).casefold()):
            return c
    return None


# ---------------------------------------------------------------- checks


def validate(series, maps, aliases, raw_series=None):
    """Return a table of problems. 'error' rows should be fixed before modelling."""
    out = []

    def add(severity, check, mids, detail=""):
        for mid in mids:
            out.append(
                {
                    "severity": severity,
                    "check": check,
                    "match_id": mid,
                    "detail": detail,
                }
            )

    add("error", "duplicate match", series.match_id[series.match_id.duplicated()])
    add(
        "error",
        "duplicate map",
        maps.match_id[maps.duplicated(["match_id", "map_order"])],
    )

    wins = maps.groupby("match_id").agg(a=("y", "sum"), n=("y", "size"))
    chk = series.set_index("match_id").join(wins, how="inner")
    bad = chk[(chk.a != chk.score_a) | (chk.n - chk.a != chk.score_b)]
    add("error", "map scores != series score", bad.index)
    no_maps = series.match_id[~series.match_id.isin(maps.match_id)]
    add("warning", "series without map data (forfeit?)", no_maps)

    need = series.best_of // 2 + 1
    add(
        "error",
        "series score inconsistent with best_of",
        series.match_id[series[["score_a", "score_b"]].max(axis=1) != need],
    )

    lo, hi = pd.Timestamp("2024-12-01"), pd.Timestamp.now().normalize() + pd.Timedelta(
        days=1
    )
    add(
        "error",
        "invalid date",
        series.match_id[series.date.isna() | (series.date < lo) | (series.date > hi)],
    )

    w = maps[["rounds_a", "rounds_b"]].max(axis=1)
    l = maps[["rounds_a", "rounds_b"]].min(axis=1)
    ok = ((w == 13) & (l <= 11)) | ((w >= 14) & (w - l == 2))
    add("warning", "unusual round score", maps.match_id[~ok])

    add(
        "error",
        "team not in alias table",
        series.match_id[series.team_a.isna() | series.team_b.isna()],
    )
    return pd.DataFrame(out, columns=["severity", "check", "match_id", "detail"])


def load(name):
    return pd.read_parquet(config.PROCESSED / f"{name}.parquet")


if __name__ == "__main__":
    build()
