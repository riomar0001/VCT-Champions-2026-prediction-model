"""VCT match predictor.

    vct scrape        download VCT 2025-2026 matches from VLR (cached in data/raw/)
    vct build         clean, validate and write data/processed/
    vct experiments   run the backtests and write experiments.csv
    vct all           all three, in order
"""

import sys


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd in ("scrape", "all"):
        from .scrape import scrape
        scrape()
    if cmd in ("build", "all"):
        from .clean import build
        result = build()
        if (result["issues"].severity == "error").any():
            sys.exit("data checks failed; see the issues above")
    if cmd in ("experiments", "all"):
        from .experiments import run
        run()
    if cmd not in ("scrape", "build", "experiments", "all"):
        print(__doc__)
