"""
main.py
─────────────────────────────────────────────────────────────────
SentiStream AI — Real-Time Social Media Sentiment Analysis
Interactive console entry point.

Workflow:
    1. Load trained model from disk
    2. Prompt user for a keyword / topic
    3. Fetch live Reddit posts + comments
    4. Predict sentiment for each item
    5. Aggregate and print a clear report

Prerequisites:
    1. pip install -r requirements.txt
    2. python train_model.py          (generates .pkl files)
    3. Copy .env.example → .env and fill in Reddit credentials
       (OR run without credentials to use synthetic demo data)
    4. python main.py

Usage:
    python main.py                    # interactive console
    python main.py --mock             # use synthetic data (no API needed)
    python main.py --query "coffee"   # analyse one topic and exit
"""

import os
import sys
import logging
import argparse
import textwrap

# ── Make sure 'backend' package is importable ─────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.analyzer import SentimentAnalyzer
from backend.config   import cfg

# ──────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, cfg.LOG_LEVEL, logging.INFO),
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SentiStream.main")


# ──────────────────────────────────────────────────────────────────
# Report printer
# ──────────────────────────────────────────────────────────────────
EMOJI = {"Positive": "✅", "Negative": "❌", "Neutral": "➖"}
BAR   = {"Positive": "▓", "Negative": "░", "Neutral": "▒"}


def print_report(report: dict) -> None:
    """
    Pretty-print the full sentiment analysis report to the console.

    Sections printed:
        1. Overall verdict
        2. Percentage breakdown with bar chart
        3. Top keywords per sentiment group
        4. Sample positive opinions
        5. Sample negative opinions
        6. Sample neutral opinions
    """
    if "error" in report:
        print(f"\n  ⚠️   {report['error']}\n")
        return

    query   = report.get("query", "")
    total   = report.get("total", 0)
    pcts    = report.get("percentages", {})
    counts  = report.get("counts", {})
    verdict = report.get("verdict", "Unknown")
    kws     = report.get("keywords", {})
    ops     = report.get("top_opinions", {})
    all_kws = report.get("all_keywords", [])

    pos_pct = pcts.get("Positive", 0)
    neg_pct = pcts.get("Negative", 0)
    neu_pct = pcts.get("Neutral",  0)

    # ── Header ────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(f'  🔍  Analysis: "{query}"')
    print("═" * 60)

    # ── Verdict ───────────────────────────────────────────────────
    v_emoji = "💚" if "Positive" in verdict else "🔴" if "Negative" in verdict else "➖"
    print(f"\n  OVERALL VERDICT:  {v_emoji}  {verdict.upper()}")
    print(f"  Based on {total} Reddit posts and comments\n")

    # ── Percentage bars ───────────────────────────────────────────
    print("  📊  Sentiment Breakdown")
    print("  " + "─" * 44)
    for label in ("Positive", "Neutral", "Negative"):
        pct   = pcts.get(label, 0)
        cnt   = counts.get(label, 0)
        bar   = BAR[label] * int(pct / 3)
        emoji = EMOJI[label]
        print(f"  {emoji} {label:<10} {pct:>5.1f}%  {bar}  ({cnt})")

    # ── All keywords ──────────────────────────────────────────────
    if all_kws:
        print(f"\n  🔑  Top Keywords Across All Posts")
        print(f"  {', '.join(all_kws)}")

    # ── Per-group keywords ────────────────────────────────────────
    for label in ("Positive", "Negative"):
        group_kws = kws.get(label, [])
        if group_kws:
            emoji = EMOJI[label]
            print(f"\n  {emoji}  Common themes in {label} posts:")
            print(f"     {', '.join(group_kws)}")

    # ── Sample opinions ───────────────────────────────────────────
    for label in ("Positive", "Negative", "Neutral"):
        opinions = ops.get(label, [])
        if not opinions:
            continue
        emoji = EMOJI[label]
        print(f"\n  {emoji}  Sample {label} Opinions  (sorted by Reddit score)")
        print("  " + "─" * 44)
        for i, op in enumerate(opinions, 1):
            sub   = op.get("subreddit", "")
            score = op.get("score", 0)
            text  = op.get("text", "")
            sub_str = f" r/{sub}" if sub else ""
            print(f"\n  {i}.{sub_str}  ↑{score}")
            for line in textwrap.wrap(text, width=56):
                print(f"     {line}")

    print("\n" + "═" * 60 + "\n")


# ──────────────────────────────────────────────────────────────────
# Interactive loop
# ──────────────────────────────────────────────────────────────────
def interactive_loop(analyzer: SentimentAnalyzer) -> None:
    """
    Run an interactive console where the user types topics to analyse.

    Commands:
        <keyword>               — search Reddit + analyse sentiment
        predict: <text>         — quick single-text prediction
        help                    — show instructions
        quit / exit / q         — exit
    """
    print("""
  ─────────────────────────────────────────────────────
  How to use:

  Type any keyword or topic to analyse Reddit sentiment:
    → electric vehicles
    → Python programming
    → oat milk
    → Taylor Swift new album

  Or predict sentiment of a specific sentence:
    → predict: This product totally changed my life!

  Commands:  help  |  quit
  ─────────────────────────────────────────────────────
""")

    while True:
        try:
            user_input = input("  Topic: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye! 👋\n")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\n  Goodbye! 👋\n")
            break

        if user_input.lower() in ("help", "?"):
            print("""
  Type a keyword → analyse Reddit sentiment for that topic
  predict: <text> → predict sentiment of a single sentence
  quit           → exit
""")
            continue

        # Single-text predict mode
        if user_input.lower().startswith("predict:"):
            text   = user_input[len("predict:"):].strip()
            if not text:
                print("  Please add text after 'predict:'\n")
                continue
            result = analyzer.predict_single(text)
            emoji  = EMOJI.get(result["label"], "❓")
            print(f"\n  Result  : {emoji}  {result['label']}")
            print(f"  Probs   : {result['probabilities']}\n")
            continue

        # Full Reddit analysis
        print(f"\n  Fetching Reddit posts for \"{user_input}\" …")
        report = analyzer.analyze(user_input)
        print_report(report)


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="SentiStream AI — Real-Time Reddit Sentiment Analysis"
    )
    parser.add_argument(
        "--mock",  action="store_true",
        help="Use synthetic data instead of live Reddit API"
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Run one analysis and exit (batch/CI mode)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max Reddit posts to fetch (overrides config)"
    )
    args = parser.parse_args()

    print("""
╔══════════════════════════════════════════════════════════════╗
║        SentiStream AI  —  Real-Time Sentiment Analysis       ║
║        Reddit Integration  ·  TF-IDF  ·  Logistic Reg.      ║
╚══════════════════════════════════════════════════════════════╝
""")

    # ── Check model files exist ───────────────────────────────────
    if not os.path.exists(cfg.MODEL_PATH) or not os.path.exists(cfg.VECTORIZER_PATH):
        print("  ⚠️   Trained model not found.")
        print(f"  Expected: {cfg.MODEL_PATH}  and  {cfg.VECTORIZER_PATH}")
        print("\n  Run this first:")
        print("      python train_model.py\n")
        sys.exit(1)

    # ── Build analyzer ────────────────────────────────────────────
    use_mock = args.mock or not cfg.validate_reddit_creds()
    analyzer = SentimentAnalyzer(use_mock=use_mock)

    if use_mock:
        print("  ℹ️   Running with MOCK data (no Reddit API credentials).")
        print("  To use real Reddit data, fill in .env with your API keys.\n")

    # ── Single-query mode ─────────────────────────────────────────
    if args.query:
        report = analyzer.analyze(args.query, limit=args.limit)
        print_report(report)
        return

    # ── Interactive mode ──────────────────────────────────────────
    interactive_loop(analyzer)


if __name__ == "__main__":
    main()
