"""Packed balanced base-3 codes used by Tk-Integers.

Five trits occupy one byte. The implementation is copied from the TAARDIS
any-k packer and keeps k in the documented range 1..16.
"""
from __future__ import annotations

import torch


def _pow3(n: int, device):
    return (3 ** torch.arange(n, device=device)).to(torch.int64)


def quantize_ktrit(W, k: int, group_size: int = 128):
    """Quantize ``[out, in]`` weights to signed balanced base-3 levels."""
    O, I = W.shape
    L = (3 ** k - 1) // 2
    if not 1 <= k <= 16:
        raise ValueError("k must be between 1 and 16")
    if I % group_size:
        raise ValueError(f"I={I} is not divisible by group_size={group_size}")
    x = W.reshape(O, I // group_size, group_size).double()
    scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    codes = torch.round(x / scale * L).clamp(-L, L).to(torch.int32).reshape(O, I)
    scale_dtype = torch.float16 if k <= 5 else torch.float32
    return codes, scale.squeeze(-1).to(scale_dtype)


def pack_ktrit(codes, k: int):
    """Pack signed codes ``[out, in]`` into five-trit-per-byte rows."""
    O, I = codes.shape
    L = (3 ** k - 1) // 2
    if not 1 <= k <= 16:
        raise ValueError("k must be between 1 and 16")
    pad = (-(I * k)) % 5
    nbytes = (I * k + pad) // 5
    digits = ((codes.to(torch.int64) + L).reshape(O, I, 1) // _pow3(k, codes.device)) % 3
    stream = digits.reshape(O, I * k)
    if pad:
        stream = torch.nn.functional.pad(stream, (0, pad))
    packed = (stream.reshape(O, nbytes, 5) * _pow3(5, codes.device)).sum(-1).to(torch.uint8)
    return packed.contiguous(), I, k


def unpack_ktrit(packed, I: int, k: int):
    """Unpack rows into signed int32 codes."""
    O, nbytes = packed.shape
    L = (3 ** k - 1) // 2
    digits = (packed.reshape(O, nbytes, 1).to(torch.int64) // _pow3(5, packed.device)) % 3
    stream = digits.reshape(O, nbytes * 5)[:, :I * k]
    levels = (stream.reshape(O, I, k) * _pow3(k, packed.device)).sum(-1)
    return (levels - L).to(torch.int32)


def pack_base3(codes):
    packed, width, _ = pack_ktrit(codes.to(torch.int32), 1)
    return packed, width


def unpack_base3(packed, width: int):
    return unpack_ktrit(packed, width, 1)


def packed_bpw(weights: int, k: int) -> float:
    """Finite code-stream bits per weight, excluding scales."""
    return 8.0 * ((weights * k + 4) // 5) / weights
