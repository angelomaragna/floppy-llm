"""int8 symmetric per-tensor absmax quantization.

scale = max(abs(w))/127 (0 mapped to 1.0 for all-zero tensors),
q = clip(round(w/scale), -127, 127) as int8.
"""

import numpy as np


def quantize_tensor(t) -> tuple:
    """t: TensorLike -> (int8 ndarray, float32 scale)."""
    w = np.asarray(t, dtype=np.float32).ravel()
    amax = float(np.max(np.abs(w))) if w.size else 0.0
    if amax == 0.0:
        scale = np.float32(1.0)
        q = np.zeros(w.shape, dtype=np.int8)
    else:
        scale = np.float32(amax / 127.0)
        q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q, scale


def dequantize(q, scale) -> np.ndarray:
    """int8 ndarray + float32 scale -> float32 ndarray."""
    return np.asarray(q, dtype=np.float32) * np.float32(scale)
