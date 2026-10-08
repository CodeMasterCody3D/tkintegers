"""Zero-float Tk-Integer disk codec.

The weight code stream is packed k-trit data. Group scales use a k10 log
stream, and small sensitive tensors use an integer-anchor high-k stream.
"""
from __future__ import annotations

import math

import torch


K_DEFAULT = 5


def _group(width: int, requested: int) -> int:
    return requested if width % requested == 0 else (math.gcd(width, requested) or 1)


def _anchor(value: float):
    value = max(float(value), 1e-30)
    exponent = math.floor(math.log(value, 3.0)) - 15
    mantissa = int(round(value / (3.0 ** exponent)))
    return mantissa, exponent


def pack_hi(tensor, k: int, log: bool = False):
    """Pack an arbitrary tensor into integer codes plus an integer anchor."""
    values = tensor.detach().float()
    if log:
        values = torch.log2(values.clamp_min(1e-20))
    anchor = values.abs().amax().clamp_min(1e-30).item()
    mantissa, exponent = _anchor(anchor)
    anchor_value = float(mantissa) * (3.0 ** exponent)
    levels = (3 ** k - 1) // 2
    codes = torch.round(values / anchor_value * levels).clamp(-levels, levels)
    dtype = torch.int16 if k <= 10 else torch.int32
    meta = torch.tensor([mantissa, exponent, k, 1 if log else 0], dtype=torch.int64)
    return codes.to(dtype).contiguous(), meta


def unpack_hi(codes, meta, dtype=torch.float32):
    mantissa, exponent, k, log = (int(value) for value in meta.tolist())
    levels = (3 ** k - 1) // 2
    anchor = float(mantissa) * (3.0 ** exponent)
    values = codes.to(torch.float64) / levels * anchor
    if log:
        values = torch.pow(values.new_tensor(2.0), values)
    return values.to(dtype)


def _code_dtype(k: int):
    levels = (3 ** k - 1) // 2
    return torch.int8 if levels <= 127 else (torch.int16 if levels <= 32767 else torch.int32)


def pack_k5_hiscale(W, k: int = K_DEFAULT, scale_k: int = 10, group_size: int = 128):
    """Pack a 2-D weight with an integer k10-log scale stream."""
    if W.dim() != 2:
        raise ValueError("pack_k5_hiscale expects a 2-D tensor")
    O, I = W.shape
    group = _group(I, group_size)
    levels = (3 ** k - 1) // 2
    x = W.reshape(O, I // group, group).float()
    scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    scale_codes, scale_meta = pack_hi(scale.squeeze(-1), scale_k, log=True)
    stored_scale = unpack_hi(scale_codes, scale_meta).reshape(O, I // group, 1)
    codes = torch.round(x / stored_scale * levels).clamp(-levels, levels).to(_code_dtype(k))
    return codes.reshape(O, I).contiguous(), scale_codes, scale_meta


def pack_tensor(tensor, scale_k: int = 10, small_k: int = 15, k: int = K_DEFAULT):
    """Return integer tensors that make up one serialized Tk-Integer tensor."""
    if tensor.dim() == 2 and tensor.dtype.is_floating_point:
        codes, scale_codes, scale_meta = pack_k5_hiscale(tensor, k=k, scale_k=scale_k)
        return {".tk_codes": codes, ".tk_scale_codes": scale_codes, ".tk_scale_meta": scale_meta}
    if tensor.dtype.is_floating_point:
        codes, meta = pack_hi(tensor, small_k, log=False)
        return {".tk_codes": codes, ".tk_meta": meta}
    return {"": tensor.contiguous()}
