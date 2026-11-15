"""Debug a prompt through the model: tokenization, per-position top-k
probabilities, and greedy continuation.

Usage:
    .venv/bin/python scripts/debug_prompt.py "The Battle of Trafalgar"
    .venv/bin/python scripts/debug_prompt.py "Trafalgar" --topk 8

Shows, for each prompt position: the token piece, and the top-k candidate
next-tokens with probabilities. Also reports the closest corpus facts sharing
the prompt's leading tokens, so coverage gaps are visible immediately.
"""

import argparse
import difflib
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from floppy_llm.model import TinyGPT  # noqa: E402
from floppy_llm.tokenizer import EOS_ID, Tok  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--topk", type=int, default=6)
    args = ap.parse_args()

    tok = Tok(os.path.join(ROOT, "model", "tokenizer.model"))
    model = TinyGPT()
    model.load_state_dict(torch.load(
        os.path.join(ROOT, "model", "model.pt"), map_location="cpu"))
    model.eval()

    ids = tok.encode(args.prompt)
    print(f"\nPROMPT: {args.prompt!r}")
    print(f"TOKENS ({len(ids)}):")
    for i, tid in enumerate(ids):
        print(f"  [{i:2d}] {tid:3d} {tok.id_to_piece(tid)!r}")

    print(f"\nTOP-{args.topk} NEXT-TOKEN DISTRIBUTION PER POSITION:")
    with torch.no_grad():
        logits = model(torch.tensor([ids]))[0]  # (T, vocab)
    for pos in range(len(ids)):
        probs = torch.softmax(logits[pos], dim=-1)
        top = torch.topk(probs, args.topk)
        print(f"  after {tok.decode(ids[:pos + 1])[-28:]!r:30s}:", end=" ")
        cands = [f"{tok.id_to_piece(int(t))!r}={float(p):.2f}"
                 for p, t in zip(top.values, top.indices)]
        print("  ".join(cands))

    # greedy continuation
    ctx = list(ids)
    with torch.no_grad():
        for _ in range(80):
            nxt = int(model(torch.tensor([ctx]))[0, -1].argmax())
            if nxt == EOS_ID:
                break
            ctx.append(nxt)
    print(f"\nGREEDY COMPLETION:\n  {args.prompt}{tok.decode(ctx[len(ids):])}")

    # corpus proximity: which facts share the longest leading words
    with open(os.path.join(ROOT, "data", "facts.txt"), encoding="utf-8") as f:
        facts = [l.strip() for l in f if l.strip()]
    words = args.prompt.lower().split()
    matches = [f for f in facts
               if f.lower().split()[:len(words)] == words]
    if matches:
        print(f"\nCORPUS FACTS starting with the same {len(words)} words:")
        for m in matches[:5]:
            print(f"  - {m}")
    else:
        print("\nNO corpus fact starts with these words — completion cannot "
              "be anchored; closest facts by similarity:")
        for m in difflib.get_close_matches(args.prompt, facts, n=3, cutoff=0.4):
            print(f"  - {m}")


if __name__ == "__main__":
    main()
