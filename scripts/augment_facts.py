"""Expand a fact split with 2 synthetic PROMPT-SIDE variants per entry.

Each fact's subject (its leading proper-noun phrase, or first capitalized
subject for generic leads, or its year) is emitted as a standalone stub
prompt, teaching the model stub->fact retrieval routes instead of only
sentence continuations:

  variant 1 (glossary):      "<Subject>. <original fact>"
  variant 2 (short stub):     "<LastWord>. <original fact>" with casing
                              alternating between CAPITALIZED and lowercased
  year-subject facts instead: "<full subject>. <fact>" plus
                              "What happened in <year>? <fact>"

Usage:
    .venv/bin/python scripts/augment_facts.py data/facts_split1.txt

Writes <input>_augmented.txt: original + 2 variants interleaved per fact.
The input file is not modified.
"""

import os
import sys

MAX_LEAD_WORDS = 6
STOP = {"The", "A", "An", "It", "In", "On", "At", "He", "She", "They", "There",
        "This", "These", "After", "Before", "Since", "From", "By", "As", "Its",
        "His", "Her"}


def subject_of(fact: str):
    """(subject phrase, year or None) mirroring split_facts bucket logic."""
    import re
    words = re.findall(r"[A-Za-z]+|\d+", fact)
    CONNECTORS = {"of", "the", "de", "da", "von", "van", "des", "du"}
    cap_seq = []
    for w in words[:MAX_LEAD_WORDS]:
        if w[0].isupper() and w not in STOP:
            cap_seq.append(w)
        elif cap_seq and w.lower() in CONNECTORS and len(words) > words.index(w) + 1:
            # lowercase connector inside a proper-noun phrase (Battle of ...)
            nxt = words[words.index(w) + 1]
            if nxt[0].isupper():
                cap_seq.append(w)
                continue
            break
        elif cap_seq:
            break
    subject = " ".join(cap_seq) if cap_seq else None
    year = None
    for w in words:
        if w.isdigit() and len(w) == 4:
            year = w
            break
    if subject is None:
        for w in words:
            if w[0].isupper() and w not in STOP:
                subject = w
                break
    return subject, year


def variants(fact: str, index: int) -> list:
    import re
    subject, year = subject_of(fact)
    fact = fact.rstrip(".")
    out = []

    if subject is None and year is not None:
        # year-only facts: invert into a question route (two lines)
        return [f"What happened in {year}? {fact}.",
                f"{year}. {fact}."]
    if subject is None:
        subject = "This"

    # variant 1: full-subject glossary line (subject as prompt)
    subj = subject.rstrip(".")
    out.append(f"{subj}. {fact}.")

    # variant 2: short stub, casing rotated for coverage of both prompt forms
    cap_words = [w for w in subj.split(" ") if w[0].isupper()]
    last = cap_words[-1] if cap_words else subj.split(" ")[-1]
    stub = last if index % 2 == 0 else last[0].lower() + last[1:]
    if stub.lower() == subj.split(" ")[-1].lower() and len(subj.split(" ")) == 1 \
            and stub == subj:
        # single-word subject: use lowercase variant as the second form
        stub = subj[0].lower() + subj[1:] if subj[0].isupper() else subj
    out.append(f"{stub}. {fact}.")
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
