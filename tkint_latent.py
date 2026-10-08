"""Trainable Tk-Integer latents with a carried quantization residual."""
from __future__ import annotations

import torch

from base3_pack import pack_ktrit, unpack_ktrit
from tk_codec import pack_hi, unpack_hi


def _quantize_scale(scale, scale_mode):
    if scale_mode == "fp16":
        stored = scale.clamp_min(2 ** -24).to(torch.float16)
        return stored.float(), stored, None
    if scale_mode == "integer-k10":
        codes, meta = pack_hi(scale, 10, log=True)
        return unpack_hi(codes, meta), codes, meta
    raise ValueError("scale_mode must be 'fp16' or 'integer-k10'")


def _encode(W, k, group_size, scale_mode):
    if not 1 <= k <= 16:
        raise ValueError("k must be in 1..16")
    O, I = W.shape
    if I % group_size:
        raise ValueError(f"in_features {I} is not divisible by group {group_size}")
    x = W.float().reshape(O, I // group_size, group_size)
    decoded_scale, scale_store, scale_meta = _quantize_scale(
        x.double().abs().amax(-1).clamp_min(1e-12), scale_mode
    )
    levels = (3 ** k - 1) // 2
    codes = torch.round(x.double() / decoded_scale.double().unsqueeze(-1) * levels)
    codes = codes.clamp(-levels, levels).to(torch.int32).reshape(O, I)
    decoded = (codes.double().reshape(O, I // group_size, group_size) / levels
               * decoded_scale.double().unsqueeze(-1)).reshape(O, I).to(W.dtype)
    return decoded, codes, decoded_scale, scale_store, scale_meta


class TkIntLatent:
    """A discrete trainable state with residual error feedback."""

    def __init__(self, W0, k=5, group_size=64, residual_k=5, device=None,
                 dtype=torch.float32, scale_mode="fp16"):
        if W0.ndim != 2:
            raise ValueError("TkIntLatent is for 2-D latents")
        dev = device or W0.device
        self.k, self.g, self.rk = int(k), int(group_size), int(residual_k)
        if not 1 <= self.k <= 16 or not 0 <= self.rk <= 16:
            raise ValueError("k must be in 1..16 and residual_k in 0..16")
        self.scale_mode = scale_mode
        q, self.codes, self.scale, self.scale_store, self.scale_meta = _encode(
            W0.to(dev, dtype), self.k, self.g, self.scale_mode
        )
        self.param = torch.nn.Parameter(q.clone().requires_grad_(True))
        self.resid = torch.zeros_like(q)
        self.resid_codes = self.resid_scale = self.resid_scale_store = None
        self.resid_scale_meta = None
        if self.rk:
            (self.resid, self.resid_codes, self.resid_scale,
             self.resid_scale_store, self.resid_scale_meta) = _encode(
                 self.resid, self.rk, self.g, self.scale_mode
             )
        self.commits = 0
        self.moved_codes = 0

    @torch.no_grad()
    def apply_(self, update):
        """Commit an optimizer update and carry what the code cannot represent."""
        want = self.param.detach() + update + self.resid
        new, codes, scale, scale_store, scale_meta = _encode(
            want, self.k, self.g, self.scale_mode
        )
        self.moved_codes += int((codes != self.codes).sum())
        self.codes, self.scale = codes, scale
        self.scale_store, self.scale_meta = scale_store, scale_meta
        residual = want - new
        if self.rk:
            (residual, self.resid_codes, self.resid_scale,
             self.resid_scale_store, self.resid_scale_meta) = _encode(
                 residual, self.rk, self.g, self.scale_mode
             )
        else:
            residual.zero_()
        self.resid = residual
        self.param.data.copy_(new)
        self.commits += 1
        return new

    def value(self):
        return self._decode(self.codes, self.scale, self.k)

    def bits_per_weight(self):
        return 1.6 * self.k + (1.6 * self.rk if self.rk else 0.0)

    def state_dict(self):
        packed, width, _ = pack_ktrit(self.codes, self.k)
        state = {
            "format": "tkint-latent-v2",
            "shape": tuple(self.param.shape),
            "width": width,
            "k": self.k,
            "g": self.g,
            "rk": self.rk,
            "scale_mode": self.scale_mode,
            "weight_packed": packed,
        }
        if self.scale_mode == "fp16":
            state["weight_scale"] = self.scale_store
        else:
            state["weight_scale_codes"] = self.scale_store
            state["weight_scale_meta"] = self.scale_meta
        if self.rk:
            residual_packed, _, _ = pack_ktrit(self.resid_codes, self.rk)
            state["residual_packed"] = residual_packed
            if self.scale_mode == "fp16":
                state["residual_scale"] = self.resid_scale_store
            else:
                state["residual_scale_codes"] = self.resid_scale_store
                state["residual_scale_meta"] = self.resid_scale_meta
        return state

    def _decode(self, codes, scale, k):
        levels = (3 ** k - 1) // 2
        O, I = self.param.shape
        return (codes.double().reshape(O, I // self.g, self.g) / levels
                * scale.double().unsqueeze(-1)).reshape(O, I).to(self.param.dtype)
