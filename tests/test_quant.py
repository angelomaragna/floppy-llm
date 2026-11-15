import numpy as np
import pytest

from floppy_llm.binfmt import parse_bin
from floppy_llm.model import EXPECTED_PARAMS, TinyGPT
from floppy_llm.quant import dequantize, quantize_tensor


def test_roundtrip_error_bound():
    t = np.random.RandomState(0).randn(500).astype(np.float32) * 3.5
    q, scale = quantize_tensor(t)
    back = dequantize(q, scale)
    # per-tensor bound: each error <= half a step = scale/2
    assert np.max(np.abs(back - t)) <= scale / 2 + 1e-6


def test_zero_tensor_no_divzero():
    q, scale = quantize_tensor(np.zeros((10, 10), dtype=np.float32))
    assert scale == 1.0
    assert np.all(q == 0)


def test_export_parses(tmp_path):
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from export_quant import export

    out = tmp_path / "m.bin"
    n = export(os.path.join(os.path.dirname(__file__), "..", "model", "model.pt"),
               str(out))
    meta = parse_bin(str(out))
    assert meta["magic"] == 0x464C5059
    assert meta["version"] == 1
    assert meta["vocab_size"] == 512
    assert meta["d_model"] == 112
    assert meta["n_layer"] == 2
    assert meta["n_heads"] == 4
    assert meta["ffn_dim"] == 448
    assert meta["context_len"] == 128
    # wire-format names
    names = set(meta["tensors"])
    expected = {"tok_emb", "final_norm"}
    for i in range(2):
        expected |= {f"attn_q_L{i}", f"attn_k_L{i}", f"attn_v_L{i}",
                     f"attn_o_L{i}", f"norm1_L{i}", f"ffn_fc_L{i}",
                     f"ffn_proj_L{i}", f"norm2_L{i}"}
    assert names == expected
    # every tensor's data length matches product(shape)
    total = 0
    for name, (shape, scale, q) in meta["tensors"].items():
        assert q.size == int(np.prod(shape)), name
        total += q.size
    assert total == EXPECTED_PARAMS
    # total file layout verified by parse_bin exact-consume; size checks:
    assert os.path.getsize(str(out)) == n
    assert n - 32 - total < 800  # header + per-tensor overhead is tiny
    assert n < 360_000

    # quantized weights close to originals
    import torch
    model = TinyGPT()
    model.load_state_dict(torch.load(
        os.path.join(os.path.dirname(__file__), "..", "model", "model.pt"),
        map_location="cpu"))
    emb = model.tok_emb.weight.detach().numpy()
    shape, scale, q = meta["tensors"]["tok_emb"]
    assert shape == (512, 112)
    assert np.max(np.abs(dequantize(q, scale) - emb)) <= scale / 2 + 1e-6
