"""Train TinyGPT on a fact corpus.

Usage:
    .venv/bin/python scripts/train.py [--data PATH] [--out-dir DIR] [--steps N] [--batch N] [--lr F]
Defaults keep the original behavior: data/facts.txt -> model/model.pt
"""

import argparse
import math
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from floppy_llm.model import CONTEXT_LEN, TinyGPT, count_params  # noqa: E402
from floppy_llm.tokenizer import EOS_ID, Tok  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "facts.txt"),
                    help="corpus file, one line per training document")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "model"),
                    help="output directory for model.pt")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer.model path; default <out-dir>/tokenizer.model")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tok_path = args.tokenizer or os.path.join(args.out_dir, "tokenizer.model")
    tok = Tok(tok_path)
    with open(args.data, encoding="utf-8") as f:
        facts = [l.rstrip("\n") for l in f if l.strip()]
    print(f"{len(facts)} corpus lines from {args.data}, device={device}")

    docs = []
    for fact in facts:
        ids = tok.encode(fact)
        ids = ids + [EOS_ID]
        if len(ids) <= CONTEXT_LEN:
            docs.append(ids)
    print(f"{len(docs)} documents after tokenize (<= {CONTEXT_LEN} tokens)")

    model = TinyGPT().to(device)
    n = count_params(model)
    print(f"model params: {n}")
    assert n == 358_960

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    max_len = max(len(d) for d in docs)

    def sample_batch():
        seqs = random.sample(docs, args.batch)
        T = min(max(len(s) for s in seqs), max_len)
        input_ids = torch.full((args.batch, T), EOS_ID, dtype=torch.long)
        targets = torch.full((args.batch, T), -100, dtype=torch.long)
        for i, s in enumerate(seqs):
            t = min(len(s) - 1, T)
            input_ids[i, :t] = torch.tensor(s[:t])
            targets[i, :t] = torch.tensor(s[1:t + 1])
        return input_ids.to(device), targets.to(device)

    started = time.time()
    model.train()
    loss = None
    for step in range(1, args.steps + 1):
        lr = args.lr * 0.5 * (1 + math.cos(math.pi * (step - 1) / args.steps))
        for g in opt.param_groups:
            g["lr"] = lr
        input_ids, targets = sample_batch()
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0 or step == 1:
            el = time.time() - started
            print(f"step {step}/{args.steps} loss {loss.item():.4f} "
                  f"lr {lr:.2e} elapsed {el:.0f}s", flush=True)

    out = os.path.join(args.out_dir, "model.pt")
    torch.save(model.state_dict(), out)
    print(f"saved {out} final loss {loss.item():.4f} total {time.time()-started:.0f}s")


if __name__ == "__main__":
    main()
