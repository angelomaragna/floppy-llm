"""Stream English Wikipedia from Hugging Face and extract 3,500 history facts.

Writes data/facts.txt (one fact per line) and data/facts.meta.json
(attribution metadata for CC BY-SA compliance).

Usage:
    .venv/bin/python scripts/fetch_facts.py [--sample N] [--out DIR]

--sample N: stop after examining N articles (for timing/verification runs).
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from floppy_llm.fact_filter import extract_facts, is_history_article  # noqa: E402

DATASET = "wikimedia/wikipedia"
CONFIG = "20231101.en"  # default; override with --config (e.g. 20231101.simple)
FACTS_TARGET = 3500
FACT_BASE_URL = "https://simple.wikipedia.org/wiki/" if CONFIG == "20231101.simple" else "https://en.wikipedia.org/wiki/"


def normalize_key(fact: str) -> str:
    return " ".join(fact.lower().split()).rstrip(".?! ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="stop after N articles")
    ap.add_argument("--config", type=str, default=CONFIG, help="dataset config")
    ap.add_argument("--out", type=os.path.abspath, default=os.path.join(
        os.path.dirname(__file__), "..", "data"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    from datasets import load_dataset

    ds = load_dataset(DATASET, args.config, split="train", streaming=True)

    facts = []
    seen = set()
    source_articles = []
    examined = 0
    started = time.time()

    for article in ds:
        examined += 1
        if args.sample and examined > args.sample:
            break
        title = article.get("title", "")
        text = article.get("text", "")
        if not is_history_article(title, text):
            continue
        article_facts = extract_facts(text)
        fresh = [f for f in article_facts if normalize_key(f) not in seen]
        if not fresh:
            continue
        prior_len = len(facts)
        for f in fresh:
            if len(facts) >= FACTS_TARGET:
                break
            seen.add(normalize_key(f))
            facts.append(f)
        if len(facts) > prior_len:
            source_articles.append(
                {"title": title, "url": FACT_BASE_URL + title.replace(" ", "_"),
                 "n_facts": len(facts) - prior_len})
            print(f"[{len(facts):4d}] {title}", flush=True)
        if len(facts) >= FACTS_TARGET:
            break
        if examined % 20000 == 0:
            rate = examined / (time.time() - started)
            print(f"... examined {examined} articles ({rate:.0f}/s), {len(facts)}/{FACTS_TARGET} facts",
                  flush=True)

    elapsed = time.time() - started
    with open(os.path.join(args.out, "facts.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(facts) + "\n")
    meta = {
        "source": {
            "dataset": f"{DATASET}",
            "config": CONFIG,
            "license": "CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)",
            "attribution": "Text derived from Wikipedia (contributors listed at each article URL)",
        },
        "n_facts": len(facts),
        "n_articles_examined": examined,
        "elapsed_seconds": round(elapsed, 1),
        "articles": source_articles,
    }
    with open(os.path.join(args.out, "facts.meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"Done: {len(facts)} facts from {examined} examined articles "
          f"in {elapsed:.0f}s (sample mode: {bool(args.sample)})")


if __name__ == "__main__":
    main()
