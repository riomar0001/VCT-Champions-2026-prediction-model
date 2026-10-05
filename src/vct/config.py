"""Paths and the list of VLR events that make up the dataset."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
ALIASES = DATA / "team_aliases.csv"
EXPERIMENTS = ROOT / "experiments.csv"

VLR = "https://www.vlr.gg"
REQUEST_DELAY = 1.5  # seconds between uncached requests
USER_AGENT = "Mozilla/5.0 (vct-predictor; personal research)"

# (vlr_event_id, short name, region or "International")
EVENTS = [
    # 2025
    (2274, "2025 Americas Kickoff", "Americas"),
    (2275, "2025 China Kickoff", "China"),
    (2276, "2025 EMEA Kickoff", "EMEA"),
    (2277, "2025 Pacific Kickoff", "Pacific"),
    (2281, "2025 Masters Bangkok", "International"),
    (2347, "2025 Americas Stage 1", "Americas"),
    (2359, "2025 China Stage 1", "China"),
    (2379, "2025 Pacific Stage 1", "Pacific"),
    (2380, "2025 EMEA Stage 1", "EMEA"),
    (2282, "2025 Masters Toronto", "International"),
    (2449, "2025 Esports World Cup", "International"),
    (2498, "2025 EMEA Stage 2", "EMEA"),
    (2499, "2025 China Stage 2", "China"),
    (2500, "2025 Pacific Stage 2", "Pacific"),
    (2501, "2025 Americas Stage 2", "Americas"),
    (2283, "2025 Champions Paris", "International"),
    # 2026
    (2682, "2026 Americas Kickoff", "Americas"),
    (2683, "2026 Pacific Kickoff", "Pacific"),
    (2684, "2026 EMEA Kickoff", "EMEA"),
    (2685, "2026 China Kickoff", "China"),
    (2760, "2026 Masters Santiago", "International"),
    (2775, "2026 Pacific Stage 1", "Pacific"),
    (2860, "2026 Americas Stage 1", "Americas"),
    (2863, "2026 EMEA Stage 1", "EMEA"),
    (2864, "2026 China Stage 1", "China"),
    (2765, "2026 Masters London", "International"),
    (2952, "2026 Esports World Cup", "International"),
    (2776, "2026 Pacific Stage 2", "Pacific"),
    (2976, "2026 EMEA Stage 2", "EMEA"),
    (2977, "2026 Americas Stage 2", "Americas"),
    (2978, "2026 China Stage 2", "China"),
    (2766, "2026 Champions Shanghai", "International"),
]

CHAMPIONS_2026 = 2766
