"""VLR.gg scraper.

Every page is cached under data/raw/ and logged in data/raw/index.csv with its
URL, so re-running only downloads pages that are new or still in progress.
All CSS selectors live in this file.
"""

import csv
import re
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from bs4 import BeautifulSoup

from . import config

_last_request = 0.0


def _cache_path(url):
    key = re.sub(r"[^A-Za-z0-9]+", "_", url.removeprefix(config.VLR)).strip("_")
    return config.RAW / f"{key}.html"


def fetch(url, refresh=False):
    """Return the HTML for url, from the cache unless refresh is set."""
    global _last_request
    path = _cache_path(url)
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")

    for attempt in range(6):
        wait = config.REQUEST_DELAY - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            resp = requests.get(
                url, headers={"User-Agent": config.USER_AGENT}, timeout=30
            )
            _last_request = time.monotonic()
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            break
        except requests.ConnectionError, requests.Timeout, requests.HTTPError:
            _last_request = time.monotonic()
            if attempt == 5:
                raise
            time.sleep(5 * 2**attempt)  # 5s, 10s, 20s, 40s, 80s
    resp.raise_for_status()  # other 4xx: fail without retrying

    config.RAW.mkdir(parents=True, exist_ok=True)
    path.write_text(resp.text, encoding="utf-8")
    index = config.RAW / "index.csv"
    new = not index.exists()
    with index.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["url", "file", "fetched_at"])
        w.writerow([url, path.name, datetime.now(timezone.utc).isoformat()])
    return resp.text


def _text(node, sep=" "):
    return node.get_text(sep, strip=True) if node else ""


def _int(s):
    s = (s or "").strip()
    return int(s) if s.lstrip("-").isdigit() else None


def _float(s):
    try:
        return float(str(s).strip().rstrip("%"))
    except ValueError:
        return None


# ---------------------------------------------------------------- event page


def parse_event_matches(html):
    """One dict per match listed on /event/matches/<id>/."""
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for item in soup.select("a.match-item"):
        m = re.match(r"/(\d+)/", item["href"])
        if not m:
            continue
        names = [_text(n) for n in item.select(".match-item-vs-team-name .text-of")]
        out.append(
            {
                "match_id": int(m.group(1)),
                "url": config.VLR + item["href"],
                "status": _text(item.select_one(".ml-status")).lower(),
                "team1": names[0] if names else "",
                "team2": names[1] if len(names) > 1 else "",
            }
        )
    return out


# ---------------------------------------------------------------- match page


def parse_veto(note):
    """'PRX ban Abyss; TL pick Haven; Lotus remains' -> list of steps."""
    steps = []
    for i, part in enumerate(p.strip() for p in note.split(";") if p.strip()):
        m = re.match(r"(.+?)\s+(ban|pick)\s+(.+)", part)
        if m:
            steps.append({"step": i, "team": m[1], "action": m[2], "map": m[3]})
        elif part.endswith("remains"):
            steps.append(
                {
                    "step": i,
                    "team": None,
                    "action": "decider",
                    "map": part.removesuffix("remains").strip(),
                }
            )
    return steps


def _parse_odds(soup, match_id, names):
    rows = []
    for b in soup.select(".match-bet-item"):
        img = b.select_one("img[class]")
        book = img["class"][0].removeprefix("mod-") if img else "unknown"
        if "mod-post-odds" in b.get("class", []):
            # finished match: VLR shows the winner's pre-match price only
            team = _text(b.select_one(".match-bet-item-teamzzz"))
            odds = _float(
                _text(b.select_one(".match-bet-item-return-short .match-bet-item-odds"))
            )
            side = 1 if team == names[0] else 2 if team == names[1] else None
            if odds and side:
                rows.append(
                    {
                        "match_id": match_id,
                        "book": book,
                        "side": side,
                        "odds": odds,
                        "two_sided": False,
                    }
                )
            continue
        for side in (1, 2):
            o = _float(_text(b.select_one(f".match-bet-item-odds.mod-{side}")))
            if o:
                rows.append(
                    {
                        "match_id": match_id,
                        "book": book,
                        "side": side,
                        "odds": o,
                        "two_sided": True,
                    }
                )
    return rows


def _parse_players(game, match_id, game_id):
    rows = [r for r in game.select(".ovw-row") if r.select_one(".mod-player")]
    out = []
    for k, r in enumerate(rows):
        a = r.select_one(".mod-player a[href]")
        pid = re.search(r"/player/(\d+)", a["href"]) if a else None

        def col(name):
            c = r.select_one(f'[data-col="{name}"] .mod-both')
            return _float(_text(c)) if c else None

        out.append(
            {
                "match_id": match_id,
                "game_id": game_id,
                "side": 1 if k < len(rows) / 2 else 2,
                "player_id": int(pid[1]) if pid else None,
                "player": _text(r.select_one(".ovw-player-name")),
                "tag": _text(r.select_one(".ovw-player-tag")),
                "rating": col("rating2"),
                "acs": col("acs"),
                "kast": col("kast"),
                "fk": col("fb"),
                "fd": col("fd"),
            }
        )
    return out


def parse_match(html, match_id, url=""):
    """Parse a match page into (series, maps, players, odds, veto) records."""
    soup = BeautifulSoup(html, "html.parser")
    hdr = soup.select_one(".match-header")
    ev = hdr.select_one("a.match-header-event")
    ts = hdr.select_one(".match-header-date [data-utc-ts]")
    patch = re.search(r"Patch\s+([\d.]+)", _text(hdr.select_one(".match-header-date")))

    team_ids, names = [], []
    for k in (1, 2):
        a = hdr.select_one(f".match-header-link.mod-{k}")
        tid = re.search(r"/team/(\d+)", a.get("href", "")) if a else None
        team_ids.append(int(tid[1]) if tid else None)
        names.append(_text(a.select_one(".wf-title-med")) if a else "")

    notes = [
        _text(n) for n in hdr.select(".match-header-vs-score > .match-header-vs-note")
    ]
    bo = next((int(m[1]) for n in notes if (m := re.fullmatch(r"Bo(\d)", n))), None)
    status = notes[0].lower() if notes else ""
    scores = [
        _int(_text(s))
        for s in hdr.select(
            ".match-header-vs-score .js-spoiler span, "
            ".match-header-vs-score .sp-hide span"
        )
        if _int(_text(s)) is not None
    ]
    veto_note = _text(hdr.select_one(".match-header-note"))

    series = {
        "match_id": match_id,
        "url": url,
        "event_id": int(re.search(r"/event/(\d+)", ev["href"])[1]),
        "stage": re.sub(
            r"\s+", " ", _text(hdr.select_one(".match-header-event-series"))
        ),
        "datetime_utc": ts["data-utc-ts"] if ts else None,
        "patch": patch[1] if patch else None,
        "team1_id": team_ids[0],
        "team2_id": team_ids[1],
        "team1": names[0],
        "team2": names[1],
        "score1": scores[0] if len(scores) >= 2 else None,
        "score2": scores[1] if len(scores) >= 2 else None,
        "best_of": bo,
        "status": status,
        "veto": veto_note,
    }

    maps, players = [], []
    order = 0
    for g in soup.select(".vm-stats-game"):
        gid = g.get("data-game-id")
        if not gid or gid == "all":
            continue
        head = g.select_one(".vm-stats-game-header")
        if not head:
            continue
        s1 = _int(_text(head.select_one(".team:not(.mod-right) .score")))
        s2 = _int(_text(head.select_one(".team.mod-right .score")))
        span = head.select_one(".map span")
        name = span.find(string=True, recursive=False).strip() if span else ""
        picked = head.select_one(".map .picked")
        pick_side = 0
        if picked:
            pick_side = (
                1
                if "mod-1" in picked["class"]
                else 2 if "mod-2" in picked["class"] else 0
            )
        if s1 is None or s2 is None or s1 + s2 == 0:
            continue  # map not played
        order += 1
        maps.append(
            {
                "match_id": match_id,
                "game_id": int(gid),
                "map_order": order,
                "map": name,
                "rounds1": s1,
                "rounds2": s2,
                "pick_side": pick_side,
            }
        )
        players += _parse_players(g, match_id, int(gid))

    odds = _parse_odds(soup, match_id, names)
    veto = [{"match_id": match_id, **v} for v in parse_veto(veto_note)]
    return series, maps, players, odds, veto


# ---------------------------------------------------------------- driver


def scrape(events=None, refresh_live=True, log=print):
    """Scrape every match of every event; write tables to data/interim/."""
    events = events or config.EVENTS
    tables = {k: [] for k in ("series", "maps", "players", "odds", "veto")}
    for event_id, event_name, region in events:
        url = f"{config.VLR}/event/matches/{event_id}/?series_id=all"
        listing = parse_event_matches(fetch(url))
        if refresh_live and any(m["status"] != "completed" for m in listing):
            listing = parse_event_matches(fetch(url, refresh=True))
        log(f"{event_name}: {len(listing)} matches")
        for m in listing:
            if m["team1"] == "TBD" or m["team2"] == "TBD":
                continue
            html = fetch(m["url"])
            parsed = parse_match(html, m["match_id"], m["url"])
            if refresh_live and parsed[0]["status"] != "final":
                parsed = parse_match(
                    fetch(m["url"], refresh=True), m["match_id"], m["url"]
                )
            parsed[0].update(event=event_name, event_region=region)
            for key, rows in zip(tables, (parsed[0], *parsed[1:])):
                tables[key] += [rows] if isinstance(rows, dict) else rows

    out = config.DATA / "interim"
    out.mkdir(parents=True, exist_ok=True)
    for key, rows in tables.items():
        pd.DataFrame(rows).to_parquet(out / f"{key}.parquet", index=False)
        log(f"wrote {key}: {len(rows)} rows")
    return {k: pd.DataFrame(v) for k, v in tables.items()}


if __name__ == "__main__":
    scrape()
