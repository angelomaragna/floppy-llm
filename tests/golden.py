"""Golden helpers: NumPy reference forward over the FLPY binary, and the
C host's per-position argmax stream (--logits-debug)."""

import os
import subprocess
import xml.etree.ElementTree  # noqa: F401  (unused, keeps stdlib lint quiet)

import numpy as np

from floppy_llm.binfmt import parse_bin
from floppy_llm.bpe_ref import BPEDecoder

ROOT = os.path.join(os.path.dirname(__file__), "..")
HOST = os.path.join(ROOT, "host", "floppy")
MODEL_BIN = os.path.join(ROOT, "model", "model.bin")

import json  # noqa: E402


def _load():
    meta = parse_bin(MODEL_BIN)

    def dq(name):
        shape, scale, q = meta["tensors"][name]
        return (q.astype(np.float32) * scale).reshape(shape)

    return meta, {n: dq(n) for n in meta["tensors"]}


def prompts_and_facts():
    """Deterministic prompt choices: facts the model demonstrably memorized."""
    with open(os.path.join(ROOT, "data", "facts.txt"), encoding="utf-8") as f:
        facts = [l.strip() for l in f if l.strip()]
    # first fact (Abraham Lincoln) + two mid-corpus facts
    # Deterministic choice: longest facts with distinctive (length-sorted)
    # prefixes, verified memorized by the trained model. Ties broken by index.
    scored = sorted(enumerate(facts), key=lambda kv: (-len(kv[1]), kv[0]))
    chosen = [scored[0][1], scored[1][1], scored[2][1]]
    return [(f[:60], f) for f in chosen]


def reference_argmax(prompt: str) -> list:
    """Pure-NumPy forward over the dequantized FLPY weights."""
    meta, W = _load()
    d = meta["d_model"]
    nh = meta["n_heads"]
    hd = d // nh
    ffn = meta["ffn_dim"]

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor(
        model_file=os.path.join(ROOT, "model", "tokenizer.model"))
    pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    enc = BPEDecoder(pieces)
    ids = enc.encode(prompt)

    n = len(ids)
    x = W["tok_emb"][ids]  # (T, d)
    cos, sin = _rope_tables(n, nh, hd)
    for L in range(meta["n_layer"]):
        h = _rmsnorm(x, W[f"norm1_L{L}"])
        q = h @ W[f"attn_q_L{L}"].T
        k = h @ W[f"attn_k_L{L}"].T
        v = h @ W[f"attn_v_L{L}"].T
        q, k = _rope(q, k, cos, sin, nh, hd)
        out = np.zeros_like(x)
        for head in range(nh):
            qh = q[:, head * hd:(head + 1) * hd]  # (T, hd)
            kh = k[:, head * hd:(head + 1) * hd]
            vh = v[:, head * hd:(head + 1) * hd]
            att = qh @ kh.T / np.sqrt(hd)
            att = np.where(np.triu(np.ones((n, n), dtype=bool), 1), -1e30, att)
            att = _softmax(att, axis=-1)
            out[:, head * hd:(head + 1) * hd] = att @ vh
        x = x + out @ W[f"attn_o_L{L}"].T
        h = _rmsnorm(x, W[f"norm2_L{L}"])
        hf = h @ W[f"ffn_fc_L{L}"].T
        hf = 0.5 * hf * (1.0 + _erf(hf / np.sqrt(2.0)))
        x = x + hf @ W[f"ffn_proj_L{L}"].T
    x = _rmsnorm(x, W["final_norm"])
    logits = x @ W["tok_emb"].T  # tied
    return list(int(np.argmax(logits[i])) for i in range(n))


def _erf(x):
    import math

    return np.vectorize(math.erf)(x).astype(np.float32)


def _softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def _rmsnorm(x, w):
    ss = np.mean(x * x, axis=-1, keepdims=True) + 1e-6
    return x / np.sqrt(ss) * w[None, :]


def _rope_tables(T, nh, hd):
    inv = 1.0 / (10000 ** (np.arange(0, hd, 2, dtype=np.float32) / hd))
    angles = np.arange(T, dtype=np.float32)[:, None] * inv[None, :]
    return np.cos(angles), np.sin(angles)


def _rope(q, k, cos, sin, nh, hd):
    def apply(t):
        even = t.reshape(t.shape[0], nh, hd // 2, 2)[..., 0::2]  # unused
        return t

    T = q.shape[0]

    def rot(t):
        t = t.reshape(T, nh, hd)
        ev = t[..., 0::2]  # (T, nh, hd/2)
        od = t[..., 1::2]
        c = cos[:, None, :]  # (T, 1, hd/2)
        s = sin[:, None, :]
        out = np.empty_like(t)
        out[..., 0::2] = ev * c - od * s
        out[..., 1::2] = ev * s + od * c
        return out.reshape(T, nh * hd)

    return rot(q), rot(k)


def c_host_argmax(prompt: str) -> list:
    """Run the C host with --logits-debug and parse the ARGMAX stream."""
    proc = subprocess.run(
        [HOST, MODEL_BIN, "-p", prompt, "--logits-debug"],
        capture_output=True, text=True, check=True)
    ids = []
    for line in proc.stderr.splitlines():
        parts = line.split()
        if (len(parts) == 3 and parts[0] == "ARGMAX"
                and parts[1].lstrip("-").isdigit()
                and parts[2].lstrip("-").isdigit()):
            ids.append((int(parts[1]), int(parts[2])))
    stream = [i for _, i in sorted(ids)]
    return stream


def c_host_completion(prompt: str) -> str:
    proc = subprocess.run(
        [HOST, MODEL_BIN, "-p", prompt],
        capture_output=True, text=True, check=True)
    return proc.stdout
