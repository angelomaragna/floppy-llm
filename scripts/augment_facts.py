"""Expand a fact split with 2 synthetic declarative variants per entry.

Generic augmentation: each original fact is additionally stated in two of six
historical-register wrappers (rotated deterministically by fact index), so
the model sees the same knowledge in multiple surface forms — the fix for
one-form memorization identified earlier.

Usage:
    .venv/bin/python scripts/augment_facts.py data/facts_split1.txt

Writes <input>_augmented.txt: original fact + 2 variants, interleaved per
fact so training batches mix them. The input file is not modified.
"""

import os
import sys

WRAPPERS = [
    "Historically, {f}.",
    "It is recorded that {f}.",
    "As a historical fact, {f}.",
    "Notably, {f}.",
    "According to historical records, {f}.",
    "In the historical record, {f}.",
]

SENTENCE_STARTERS = {
    "The", "A", "An", "It", "In", "On", "At", "He", "She", "They", "There",
    "This", "These", "After", "Before", "Since", "From", "By", "As", "Its",
    "His", "Her",
}


def lower_first(fact: str) -> str:
    """Lowercase the first character when the fact opens with a generic
    starter; keep capitalization when it opens with a proper noun."""
    first_word = fact.split(" ", 1)[0]
    if first_word in SENTENCE_STARTERS or first_word[0].islower():
        return fact[0].lower() + fact[1:]
    return fact


def variants(fact: str, index: int) -> list:
    """Two synthetic declarative wrappers, deterministic per index."""
    f = fact.rstrip(".")
    picks = (index % len(WRAPPERS), (index + 1) % len(WRAPPERS))
    seen = set()
    out = []
    for p in picks:
        if p in seen:
            continue
        seen.add(p)
        out.append(WRAPPERS[p].format(f=lower_first(f)))
    return out


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: augment_facts.py <facts-file>")
        sys.exit(1)
    src = sys.argv[1]
    with open(src, encoding="utf-8") as fh:
        facts = [l.rstrip("\n") for l in fh if l.strip()]

    augmented = []
    for i, fact in enumerate(facts):
        augmented.append(fact)
        augmented.extend(variants(fact, i))

    out_path = src.replace(".txt", "_augmented.txt")
    if out_path == src:
        out_path = src + ".augmented.txt"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(augmented) + "\n")
    print(f"wrote {out_path}: {len(facts)} original + "
          f"{len(augmented) - len(facts)} synthetic = {len(augmented)} lines")


if __name__ == "__main__":
    main()
