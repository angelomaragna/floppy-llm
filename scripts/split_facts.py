"""Split data/facts.txt into 3 topic-pure, size-balanced chunks.

Criterion: each fact's topic bucket = its leading proper-noun sequence (first
capitalized word that is not a sentence stopword, plus immediately following
capitalized words). Whole buckets are assigned one-by-one to whichever chunk
is currently smallest (buckets processed largest-first), so a topic never
splits across chunks.

Output: data/facts_split1.txt, data/facts_split2.txt, data/facts_split3.txt
and a bucket report printed to stdout. data/facts.txt is not modified.
"""

import os
import re
import sys
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..")
N_CHUNKS = 3
STOP = {"The", "A", "An", "It", "In", "On", "At", "He", "She", "They", "His",
        "Her", "Its", "This", "These", "Since", "After", "Before", "From"}


def bucket_key(fact: str) -> str:
    words = re.findall(r"[A-Za-z]+|\d+", fact)
    # 1. leading proper-noun sequence
    cap_seq = []
    for w in words[:6]:
        if w[0].isupper() and w not in STOP:
            cap_seq.append(w)
        elif cap_seq:
            break
    if cap_seq:
        return " ".join(cap_seq)
    # 2. generic lead: first capitalized subject anywhere in the sentence
    for w in words:
        if w[0].isupper() and w not in STOP:
            return w
    # 3. otherwise: the (first) 4-digit year as topic
    years = [w for w in words if w.isdigit() and len(w) == 4]
    if years:
        return "year " + years[0]
    return words[0].lower() if words else "<empty>"


def main() -> None:
    src = os.path.join(ROOT, "data", "facts.txt")
    with open(src, encoding="utf-8") as f:
        facts = [l.rstrip("\n") for l in f if l.strip()]

    buckets = defaultdict(list)
    for fact in facts:
        buckets[bucket_key(fact)].append(fact)

    ordered = sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    chunks = [[] for _ in range(N_CHUNKS)]
    chunk_buckets = [[] for _ in range(N_CHUNKS)]
    for key, group in ordered:
        i = min(range(N_CHUNKS), key=lambda c: len(chunks[c]))
        chunks[i].extend(group)
        chunk_buckets[i].append(key)

    for i in range(N_CHUNKS):
        out = os.path.join(ROOT, "data", f"facts_split{i + 1}.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(chunks[i]) + "\n")
        print(f"chunk {i + 1}: {len(chunks[i]):5d} facts, "
              f"{len(chunk_buckets[i]):4d} buckets -> {out}")

    print(f"total: {sum(len(c) for c in chunks)} (source {len(facts)})")
    print("\nlargest 8 buckets per chunk (fact count):")
    for i in range(N_CHUNKS):
        top = sorted(chunk_buckets[i], key=lambda b: -len(buckets[b]))[:8]
        print(f"  chunk {i + 1}: " + ", ".join(
            f"{b} ({len(buckets[b])})" for b in top))


if __name__ == "__main__":
    main()
