"""Train a BPE-512 tokenizer on a fact corpus.

Usage:
    .venv/bin/python scripts/train_tokenizer.py [--data PATH] [--out-dir DIR]
Defaults keep the original behavior: data/facts.txt -> model/tokenizer.model
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from floppy_llm.tokenizer import Tok, train_tokenizer  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "facts.txt"),
                    help="corpus file, one fact per line")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "model"),
                    help="output directory for tokenizer.model")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "tokenizer.model")

    train_tokenizer(args.data, model_path)

    tok = Tok(model_path)
    print(f"tokenizer trained on {args.data}: vocab_size={tok.vocab_size}")

    with open(args.data, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                ids = tok.encode(line)
                assert tok.decode(ids) == line, f"round-trip failed: {line!r}"
                break
    print("round-trip sanity check passed")


if __name__ == "__main__":
    main()
