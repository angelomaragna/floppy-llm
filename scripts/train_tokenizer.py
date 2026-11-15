"""Train the BPE-512 tokenizer on the fact corpus.

Usage: .venv/bin/python scripts/train_tokenizer.py
Reads data/facts.txt, writes model/tokenizer.model.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from floppy_llm.tokenizer import Tok, train_tokenizer  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main() -> None:
    corpus = os.path.join(ROOT, "data", "facts.txt")
    model_dir = os.path.join(ROOT, "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "tokenizer.model")

    train_tokenizer(corpus, model_path)

    tok = Tok(model_path)
    print(f"tokenizer trained: vocab_size={tok.vocab_size}")

    with open(corpus, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                ids = tok.encode(line)
                assert tok.decode(ids) == line, f"round-trip failed: {line!r}"
                break
    print("round-trip sanity check passed")


if __name__ == "__main__":
    main()
