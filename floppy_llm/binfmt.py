"""Reader for the FLPY v1 flat binary (Python side of the wire format)."""

import struct

import numpy as np

MAGIC = 0x464C5059
VERSION = 1
HEADER_FMT = "<IIIHHIIII"
HEADER_SIZE = struct.calcsize(HEADER_FMT)  # 32


def parse_bin(path: str) -> dict:
    """Parse FLPY v1 file -> dict with header fields and tensors.

    tensors: name -> (shape tuple, scale float, int8 ndarray)
    """
    with open(path, "rb") as f:
        buf = f.read()
    if len(buf) < HEADER_SIZE:
        raise ValueError("file too small for FLPY header")
    fields = struct.unpack_from(HEADER_FMT, buf, 0)
    (magic, version, vocab_size, d_model, n_layer,
     n_heads, ffn_dim, context_len, n_tensors) = fields
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic:#x}")
    if version != VERSION:
        raise ValueError(f"unsupported version {version}")

    tensors = {}
    off = HEADER_SIZE
    for _ in range(n_tensors):
        name_len = buf[off]; off += 1
        name = buf[off:off + name_len].decode("ascii"); off += name_len
        shape_len = buf[off]; off += 1
        shape = struct.unpack_from(f"<{shape_len}I", buf, off)
        off += 4 * shape_len
        scale = struct.unpack_from("<f", buf, off)[0]; off += 4
        n = 1
        for d in shape:
            n *= d
        q = np.frombuffer(buf[off:off + n], dtype=np.int8).reshape(shape).copy()
        off += n
        tensors[name] = (shape, scale, q)

    if off != len(buf):
        raise ValueError(f"trailing bytes: {len(buf) - off}")
    return {
        "magic": magic, "version": version, "vocab_size": vocab_size,
        "d_model": d_model, "n_layer": n_layer, "n_heads": n_heads,
        "ffn_dim": ffn_dim, "context_len": context_len,
        "n_tensors": n_tensors, "tensors": tensors,
    }
