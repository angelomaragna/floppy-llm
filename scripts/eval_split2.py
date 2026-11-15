"""Evaluate stub prompts from split 2 against two models: the generic
single-corpus model and the split-2 expert (split + synthetic augmentation).

For each of 100 sampled split-2 facts, the prompt is its subject stub
(single or few words). A completion scores CORRECT when the output text
contains the original (non-augmented) fact sentence.

Usage: .venv/bin/python scripts/eval_split2.py [--n 100] [--limit K]
Writes docs/EVAL_SPLIT2.md and prints the summary table.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from augment_facts import subject_of  # noqa: E402

GENERIC = ("host/floppy", "model/model.bin", "generic (full corpus, no augmentation)")
EXPERT = ("host/floppy_2of3", "model/model2of3/model.bin", "expert (split 2 + stub augmentation)")


def build_prompts(n):
    with open(os.path.join(ROOT, "data", "facts_split2.txt"), encoding="utf-8") as f:
        facts = [l.strip() for l in f if l.strip()]
    pairs = []
    seen_subjects = set()
    for fact in facts:
        subject, year = subject_of(fact)
        if subject:
            prompt = subject
        elif year:
            prompt = year
        else:
            continue
        if prompt in seen_subjects:
            continue
        seen_subjects.add(prompt)
        pairs.append((prompt, fact))
    rng = random.Random(42)
    return rng.sample(pairs, min(n, len(pairs)))


def run_model(host, weights, prompt):
    p = subprocess.run([os.path.join(ROOT, host), os.path.join(ROOT, weights),
                        "-p", prompt], capture_output=True, text=True, timeout=60)
    return p.stdout


def correct(output, fact):
    body = fact.rstrip(".").strip()
    # require at least the first 70% of the fact verbatim in the completion
    need = body[:max(20, int(0.7 * len(body)))]
    return need in output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="only first K prompts (timing runs)")
    args = ap.parse_args()

    pairs = build_prompts(args.n)
    if args.limit:
        pairs = pairs[:args.limit]

    results = {"generic": [], "expert": []}
    outputs = []
    t0 = time.time()
    for prompt, fact in pairs:
        row = {"prompt": prompt, "fact": fact}
        for key, (host, weights, _label) in (("generic", GENERIC), ("expert", EXPERT)):
            out = run_model(host, weights, prompt)
            ok = correct(out, fact)
            results[key].append(ok)
            row[key] = ok
            row[key + "_out"] = out.replace("\n", " ")[:160]
        outputs.append(row)

    g = 100.0 * sum(results["generic"]) / len(pairs)
    e = 100.0 * sum(results["expert"]) / len(pairs)
    elapsed = time.time() - t0

    print(f"prompts: {len(pairs)}  elapsed: {elapsed:.0f}s")
    print(f"generic: {sum(results['generic'])}/{len(pairs)} = {g:.1f}%")
    print(f"expert:  {sum(results['expert'])}/{len(pairs)} = {e:.1f}%")
    print(f"improvement: {e - g:+.1f} percentage points")

    out_path = os.path.join(ROOT, "docs", "EVAL_SPLIT2.md")
    if not args.limit:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        wins = sum(1 for r in outputs if r["expert"] and not r["generic"])
        losses = sum(1 for r in outputs if r["generic"] and not r["expert"])
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"""# Split-2 Evaluation: generic model vs split-2 expert

**Headline: the split + synthetic-stub-augmented expert scores {e:.1f}% on
split-2 stub prompts versus {g:.1f}% for the original generic model — an
improvement of {e - g:+.1f} percentage points ({(e / max(g, 0.01) - 1) * 100:+.0f}% relative),
{wins} prompt(s) flipped to correct and {losses} to incorrect.**

## Method

- Take the {len(pairs)} sampled split-2 facts (seeded, reproducible) whose
  subjects are single or few words; prompt each model with the subject stub.
- Run via the C hosts (`--logits-debug`-identical `-p` mode, greedy decoding).
- A completion is CORRECT when it reproduces at least the first 70% of the
  original fact sentence verbatim.
- Generic: `host/floppy` + `model/model.bin` (full 3,500-fact training, no
  stub augmentation). Expert: `host/floppy_2of3` + `model/model2of3/model.bin`
  (1,167 split-2 facts + subject-stub synthetic lines).

## Confusion

| | correct | incorrect |
|---|---|---|
| generic | {sum(results['generic'])} | {len(pairs) - sum(results['generic'])} |
| expert | {sum(results['expert'])} | {len(pairs) - sum(results['expert'])} |

## Per-prompt results

| # | prompt | generic | expert |
|---|---|---|---|
""")
            for i, r in enumerate(outputs, 1):
                f.write(f"| {i} | {r['prompt'][:28]} | "
                        f"{'ok' if r['generic'] else 'x'} | "
                        f"{'ok' if r['expert'] else 'x'} |\n")
            f.write("""
## Interpretation

Success measures *stub-prompt retrieval* — the specific capability the
split + prompt-side augmentation operation was designed to add. It does not
measure general question answering or paraphrase robustness (see README
known limitations).
""")
        print(f"wrote {out_path}")
    with open("/tmp/eval_outputs.json", "w") as f:
        json.dump(outputs, f, indent=1)


if __name__ == "__main__":
    main()
