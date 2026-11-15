"""Export a TinyGPT checkpoint to the FLPY v1 flat int8 binary.

Usage:
    .venv/bin/python scripts/export_quant.py [--ckpt PATH] [--out PATH]
Defaults keep the original behavior: model/model.pt -> model/model.bin
"""

import argparse
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from floppy_llm.model import (CONTEXT_LEN, D_MODEL, EXPECTED_PARAMS,  # noqa: E402
                              FFN_DIM, N_HEADS, N_LAYERS, TinyGPT)
from floppy_llm.quant import quantize_tensor  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
MAGIC = 0x464C5059
VERSION = 1


def tensor_map(model) -> dict:
    """Flatten state_dict to the wire-format name -> tensor map."""
    sd = model.state_dict()
    out = {}
    out["tok_emb"] = sd["tok_emb.weight"]
    for i in range(N_LAYERS):
        p = f"blocks.{i}."
        out[f"attn_q_L{i}"] = sd[p + "attn_q.weight"]
        out[f"attn_k_L{i}"] = sd[p + "attn_k.weight"]
        out[f"attn_v_L{i}"] = sd[p + "attn_v.weight"]
        out[f"attn_o_L{i}"] = sd[p + "attn_o.weight"]
        out[f"norm1_L{i}"] = sd[p + "norm1.weight"]
        out[f"ffn_fc_L{i}"] = sd[p + "ffn_fc.weight"]
        out[f"ffn_proj_L{i}"] = sd[p + "ffn_proj.weight"]
        out[f"norm2_L{i}"] = sd[p + "norm2.weight"]
    out["final_norm"] = sd["final_norm.weight"]
    return out


def export(model_path: str, out_path: str) -> int:
    model = TinyGPT()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    tensors = tensor_map(model)
    total_params = sum(t.numel() for t in tensors.values())
    assert total_params == EXPECTED_PARAMS, f"{total_params} != {EXPECTED_PARAMS}"

    body = bytearray()
    body += struct.pack("<I", MAGIC)
    body += struct.pack("<I", VERSION)
    body += struct.pack("<I", 512)
    body += struct.pack("<HH", D_MODEL, N_LAYERS)
    body += struct.pack("<I", N_HEADS)
    body += struct.pack("<I", FFN_DIM)
    body += struct.pack("<I", CONTEXT_LEN)
    body += struct.pack("<I", len(tensors))

    n_bytes = 0
    for name, t in tensors.items():
        t = t.detach().to(torch.float32).contiguous()
        q, scale = quantize_tensor(t.numpy())
        shape = t.shape
        nb = name.encode("ascii")
        assert len(nb) < 256
        body += bytes([len(nb)])
        body += nb
        body += bytes([len(shape)])
        for d in shape:
            body += struct.pack("<I", d)
        body += struct.pack("<f", scale)
        raw = q.tobytes()
        body += raw
        n_bytes += len(raw)
        print(f"  {name:14s} {str(tuple(shape)):16s} scale={scale:.6g} bytes={len(raw)}")

    with open(out_path, "wb") as f:
        f.write(body)
    header_bytes = len(body) - n_bytes
    print(f"total bytes: {len(body)} (weights {n_bytes}, header+scales {header_bytes})")
    return len(body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "model", "model.pt"),
                    help="fp32 checkpoint to quantize")
    ap.add_argument("--out", default=os.path.join(ROOT, "model", "model.bin"),
                    help="output FLPY v1 binary")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    export(args.ckpt, args.out)


if __name__ == "__main__":
    main()
