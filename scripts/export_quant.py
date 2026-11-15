"""Export TinyGPT checkpoint to the FLPY v1 flat int8 binary.

Header (little-endian):
  0-3   magic u32 = 0x464C5059 ("FLPY")
  4-7   version u32 = 1
  8-11  vocab_size u32
  12-13 d_model u16, 14-15 n_layer u16
  16-19 n_heads u32
  20-23 ffn_dim u32
  24-27 context_len u32
  28-31 n_tensors u32
Per tensor: name_len u8, name bytes, shape_len u8, shape u32s, scale f32,
           int8 data.
"""

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
    """Flatten state_dict to the wire-format name -> (shape, tensor) map."""
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
    export(os.path.join(ROOT, "model", "model.pt"),
           os.path.join(ROOT, "model", "model.bin"))


if __name__ == "__main__":
    main()
