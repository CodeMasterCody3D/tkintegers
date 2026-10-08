"""TK S12: a 12-bit log2 scale stream for k1 groups.

The two named densities are ``k1-g20-s12`` (2.2 ideal bpw) and
``k1-g15-s12`` (2.4 ideal bpw). Scale codes are packed two per three bytes.
"""
from __future__ import annotations

import math

import torch


LO, SPAN, NMAX = -32.0, 48.0, 4095
STEP = SPAN / (NMAX - 1)


def decode(codes):
    c = torch.as_tensor(codes).to(torch.float64)
    value = torch.exp2(LO + (c - 1.0) * STEP)
    return torch.where(c > 0, value, torch.zeros_like(value))


def encode(scales):
    s = torch.as_tensor(scales).to(torch.float64)
    positive = s > 0
    log_scale = torch.log2(torch.where(positive, s, torch.ones_like(s)))
    codes = torch.round((log_scale - LO) / STEP + 1.0).clamp(1, NMAX)
    return torch.where(positive, codes, torch.zeros_like(codes)).to(torch.int32)


def snap(alpha):
    return decode(encode(alpha)).to(alpha.dtype).to(alpha.device)


def group_bytes(rows: int, cols: int, group: int, scale_bits: int = 12, trit_bits: float = 1.6) -> float:
    groups = rows * math.ceil(cols / group)
    return groups * (group * trit_bits + scale_bits) / 8.0


def pack12(codes):
    c = torch.as_tensor(codes).reshape(-1).to(torch.int64)
    if c.numel() % 2:
        c = torch.cat([c, c.new_zeros(1)])
    a, b = c[0::2], c[1::2]
    out = torch.stack([a & 0xFF, ((a >> 8) & 0x0F) | ((b & 0x0F) << 4), (b >> 4) & 0xFF], 1)
    return out.reshape(-1).to(torch.uint8)


def unpack12(packed, count: int):
    values = torch.as_tensor(packed).reshape(-1, 3).to(torch.int64)
    a = values[:, 0] | ((values[:, 1] & 0x0F) << 8)
    b = ((values[:, 1] >> 4) & 0x0F) | (values[:, 2] << 4)
    return torch.stack([a, b], 1).reshape(-1)[:count].to(torch.int32)
