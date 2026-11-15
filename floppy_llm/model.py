"""TinyGPT: 2-layer decoder-only GPT, 358,960 parameters total.

Fixed spec: vocab 512, d_model=112, 2 layers, 4 heads, FFN 448 (GELU),
RoPE, RMSNorm, tied embedding/output weights.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 512
D_MODEL = 112
N_LAYERS = 2
N_HEADS = 4
FFN_DIM = 448
CONTEXT_LEN = 128
HEAD_DIM = D_MODEL // N_HEADS  # 28

EXPECTED_PARAMS = (
    VOCAB_SIZE * D_MODEL            # tied embedding
    + N_LAYERS * (
        4 * D_MODEL * D_MODEL      # q, k, v, o
        + 2 * D_MODEL * FFN_DIM    # ffn_fc, ffn_proj
        + 2 * D_MODEL              # norm1, norm2 (RMSNorm weights)
    )
    + D_MODEL                      # final_norm
)  # == 358,960


def rope_cos_sin(context_len: int) -> tuple:
    """Precompute RoPE cos/sin tables of shape (context_len, HEAD_DIM//2).

    Pairs are (even, odd) dims: dims 2i and 2i+1 share angle i.
    """
    pos = torch.arange(context_len, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
    angles = pos[:, None] * inv_freq[None, :]  # (T, HEAD_DIM//2)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, HEAD_DIM); cos/sin: (T, HEAD_DIM//2)."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos = cos[None, None]  # (1, 1, T, HD/2)
    sin = sin[None, None]
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
        return x * rms * self.weight


class Block(nn.Module):
    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm1 = RMSNorm(D_MODEL)
        self.attn_q = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.attn_k = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.attn_v = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.attn_o = nn.Linear(D_MODEL, D_MODEL, bias=False)
        self.norm2 = RMSNorm(D_MODEL)
        self.ffn_fc = nn.Linear(D_MODEL, FFN_DIM, bias=False)
        self.ffn_proj = nn.Linear(FFN_DIM, D_MODEL, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        cos, sin = rope_cos_sin(T)
        cos = cos.to(x.device)
        sin = sin.to(x.device)
        h = self.norm1(x)
        q = apply_rope(self.attn_q(h).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2), cos, sin)
        k = apply_rope(self.attn_k(h).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2), cos, sin)
        v = self.attn_v(h).view(B, T, N_HEADS, HEAD_DIM).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(HEAD_DIM)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device),
                          diagonal=1)
        att = att.masked_fill(mask, float("-inf")).softmax(dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, D_MODEL)
        x = x + self.attn_o(y)
        h = self.norm2(x)
        return x + self.ffn_proj(F.gelu(self.ffn_fc(h)))


def _init_small(module: nn.Module) -> None:
    """GPT-2-style small init so tied-emb logits start near uniform."""
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.blocks = nn.ModuleList([Block(i) for i in range(N_LAYERS)])
        self.final_norm = RMSNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying
        _init_small(self)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, T) int64 token ids -> logits (B, T, VOCAB_SIZE)."""
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
