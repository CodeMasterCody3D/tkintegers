#!/usr/bin/env python3
"""General any-k base-3 packer. Each weight is quantized to k balanced trits (3^k levels); the whole
trit-stream is packed 5-trits-per-byte (3^5=243<=255). bpw = 8*ceil(N*k/5)/N ~= 1.6*k, near the
k*log2(3) floor. Choose k freely: k1 ternary (1.6 bpw) ... k5 243-level (8 bpw). Cody 2026-09-16.

k = trits of PRECISION per weight (NOT the same as tk_codec's per-weight code). k1 = ternary.
Decode: unpack 5 trits/byte -> regroup into k-trit weights -> level = sum d_j*3^j -> code=level-L ->
w = code/L * group_scale. A fused k=1 GEMV already exists (triton_tq1_0.py); general-k kernel is TODO.
"""
import torch, torch.nn as nn, torch.nn.functional as F

def _pow3(n, device): return (3 ** torch.arange(n, device=device)).to(torch.int64)

def quantize_ktrit(W, k, group_size=128):
    """W [O,I] float -> (codes int32 in [-L,L] [O,I], scale [O,I//g]), L=(3^k-1)//2. k = 1..16.

    Round in float64: x/scale in [-1,1] times L (up to 21.5M at k16) needs 8 sig digits, and
    float32 only has ~7 -- past ~k7 the finest trit would be rounding noise. float64 (~16 digits)
    keeps every trit exact. Scale stays fp16 for k<=5 (compact, the tq1 formats), fp32 for k>=6 so
    high-k fidelity isn't capped by an fp16 group scale (the trits already carry >11 bits by k8)."""
    O, I = W.shape; g = group_size; L = (3 ** k - 1) // 2
    assert I % g == 0, f"I={I} % group {g}"
    assert 1 <= k <= 16, f"k={k} out of supported range 1..16"
    x = W.reshape(O, I // g, g).double()
    scale = x.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    code = torch.round(x / scale * L).clamp(-L, L).to(torch.int32).reshape(O, I)
    return code, scale.squeeze(-1).to(torch.float16 if k <= 5 else torch.float32)

def pack_ktrit(codes, k):
    """codes int in [-L,L] [O,I] -> (bytes uint8 [O, ceil(I*k/5)], I, k). Trit-stream, 5 trits/byte.

    ROW-CHUNKED: the intermediate [O,I,k] int64 digit tensor is ~24*O*I bytes -- for the padded
    248320-row head that is ~30 GB and OOMs any GPU (and wastes RAM on CPU). Pack at most
    CHUNK rows at a time so peak scratch is CHUNK*I*k*8, independent of O. Bit-identical to the
    one-shot path (each row packs independently)."""
    O, I = codes.shape; L = (3 ** k - 1) // 2; dev = codes.device
    pad = (-(I * k)) % 5
    nb = (I * k + pad) // 5
    CHUNK = max(1, int(4e7 // max(I * k, 1)))                         # ~0.3 GB int64 scratch/chunk
    if CHUNK >= O:
        u = (codes.to(torch.int64) + L)
        digits = (u.reshape(O, I, 1) // _pow3(k, dev)) % 3
        stream = digits.reshape(O, I * k)
        if pad:
            stream = F.pad(stream, (0, pad))
        b = (stream.reshape(O, -1, 5) * _pow3(5, dev)).sum(-1).to(torch.uint8)
        return b.contiguous(), I, k
    out = torch.empty(O, nb, dtype=torch.uint8, device=dev)
    p3k, p35 = _pow3(k, dev), _pow3(5, dev)
    for r in range(0, O, CHUNK):
        c = codes[r:r + CHUNK]
        u = (c.to(torch.int64) + L)
        stream = ((u.reshape(u.shape[0], I, 1) // p3k) % 3).reshape(u.shape[0], I * k)
        if pad:
            stream = F.pad(stream, (0, pad))
        out[r:r + CHUNK] = (stream.reshape(u.shape[0], -1, 5) * p35).sum(-1).to(torch.uint8)
    return out.contiguous(), I, k

def unpack_ktrit(b, I, k):
    """bytes uint8 [O,nb] -> codes int32 [O,I] in [-L,L].

    int32, NOT int8: L = (3^k-1)//2 exceeds 127 for k>=6 (k6 L=364 ... k16 L=21.5M), so int8
    silently overflowed every code past k5 -- the real reason the packer was capped at k5."""
    O, nb = b.shape; L = (3 ** k - 1) // 2; dev = b.device
    CHUNK = max(1, int(4e7 // max(nb * 5, 1)))       # same row-chunking as pack_ktrit: the
    if CHUNK >= O:                                    # [O,nb,5] int64 temp is ~40*O*nb bytes
        d5 = (b.reshape(O, nb, 1).to(torch.int64) // _pow3(5, dev)) % 3
        stream = d5.reshape(O, nb * 5)[:, :I * k]
        u = (stream.reshape(O, I, k) * _pow3(k, dev)).sum(-1)
        return (u - L).to(torch.int32)
    out = torch.empty(O, I, dtype=torch.int32, device=dev)
    p35, p3k = _pow3(5, dev), _pow3(k, dev)
    for r in range(0, O, CHUNK):
        bb = b[r:r + CHUNK]
        d5 = (bb.reshape(bb.shape[0], nb, 1).to(torch.int64) // p35) % 3
        stream = d5.reshape(bb.shape[0], nb * 5)[:, :I * k]
        u = (stream.reshape(bb.shape[0], I, k) * p3k).sum(-1)
        out[r:r + CHUNK] = (u - L).to(torch.int32)
    return out

# k=1 convenience (ternary, tq1_0-aligned layout is separate in triton_tq1_0.pack_tq1_0)
def pack_base3(codes):  b, I, _ = pack_ktrit(codes.to(torch.int32), 1); return b, I
def unpack_base3(b, I): return unpack_ktrit(b, I, 1)

class PackedKTritLinear(nn.Module):
    """Weight stored as k-trit base-3 bytes + fp16 group scale (~1.6*k bpw). dequant-then-matmul
    (fused GEMV is a per-k kernel; k=1 uses triton_tq1_0)."""
    def __init__(self, bytes_, scale, I, k, bias, out_f, group_size=128):
        super().__init__()
        self.register_buffer("bytes", bytes_); self.register_buffer("scale", scale)
        self.I, self.k, self.group_size, self.L = I, k, group_size, (3 ** k - 1) // 2
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)
        self.out_features, self.in_features = out_f, I
    @classmethod
    def from_weight(cls, W, k, bias=None, group_size=128):
        codes, scale = quantize_ktrit(W, k, group_size)
        b, I, _ = pack_ktrit(codes, k)
        return cls(b, scale, I, k, bias, W.shape[0], group_size)
    def dequant(self, dtype=torch.float32):
        O, g = self.out_features, self.group_size
        # reconstruct in float64: c/L (c up to 21.5M at k16) needs the headroom, then cast out.
        c = unpack_ktrit(self.bytes, self.I, self.k).to(torch.float64)
        w = (c.reshape(O, self.I // g, g) / self.L * self.scale.to(torch.float64).unsqueeze(-1)).reshape(O, self.I)
        return w.to(dtype)
    def forward(self, x):
        return F.linear(x, self.dequant(x.dtype), self.bias)

def bpw(N, k, group_size=128):
    import math
    nb = (N * k + 4) // 5
    scale_bits = 16 if k <= 5 else 32   # fp16 scale for k<=5, fp32 for k>=6 (see quantize_ktrit)
    return 8 * nb / N + scale_bits / group_size, k * math.log2(3)   # (incl group scale), trit floor

if __name__ == "__main__":
    torch.manual_seed(0)
    print("k-ladder: round-trip + TIGHT-pack bits/weight vs bf16 (16.0) + 27B body:")
    for k in (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 15, 16):
        L = (3 ** k - 1) // 2
        codes = torch.randint(-L, L + 1, (128, 512), dtype=torch.int32)
        b, I, _ = pack_ktrit(codes, k); back = unpack_ktrit(b, I, k)
        ok = torch.equal(back, codes)
        eff, floor = bpw(27_780_000_000, k); body = 27.78e9 * eff / 8 / 1e9
        tag = "== bf16" if abs(eff - 16.0) < 0.4 else ("< bf16" if eff < 16 else "> bf16")
        print(f"  k{k:<2} ({3**k:>10,} lvl): rt {'EXACT' if ok else 'FAIL':5} | {eff:6.3f} bpw {tag:8} | 27B ~{body:5.1f} GB")
        assert ok, f"round-trip FAILED at k{k}"
    # quantize->pack->dequant a real float weight at each k (precision ladder; k10 ~ bf16)
    W = torch.randn(256, 512) * 0.02
    print("\nquantize->dequant rel-err vs the float weight (higher k = finer):")
    for k in (1, 2, 3, 5, 8, 10, 12, 15, 16):
        m = PackedKTritLinear.from_weight(W, k)
        re = (m.dequant() - W).norm() / W.norm()
        eff, _ = bpw(256 * 512, k, 128)
        print(f"  k{k:<2}: rel-err {re:.2e}  ({eff:.2f} bpw)")
    print("========== base3_pack any-k (1..16) OK ==========")
