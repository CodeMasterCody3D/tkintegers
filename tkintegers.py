"""Tk-Integer container math and format presets.

The storage model is intentionally explicit:

* five balanced trits occupy one byte, so a k-trit value costs 1.6*k bits;
* a scale shared by ``weights_per_scale`` values costs ``scale_bits`` bits;
* physical payload accounting includes byte padding while ideal accounting does
  not.

The module is independent of torch and tkinter so the calculator can be tested
on a CPU-only machine and embedded in Autogrid without importing a model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Iterable, Mapping, Optional


TRIT_BITS = 8.0 / 5.0
ALL_KS = tuple(range(1, 17))
DEFAULT_KS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 15)
CANONICAL_KS = (1, 3, 5, 8, 10, 13, 15)
DEFAULT_SCALE_BITS = (8, 10, 12, 16, 32)
DEFAULT_SCALE_GROUPS = (1, 2, 4, 8, 12, 15, 16, 20, 24, 32, 48, 64, 96, 128)


def _positive_int(value: int, label: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class ContainerSpec:
    """One weight plane plus an optional carried residual plane."""

    k: int
    weights_per_scale: int
    scale_bits: int
    residual_k: int = 0
    residual_weights_per_scale: int = 0
    residual_scale_bits: int = 0
    name: str = ""

    def __post_init__(self) -> None:
        k = _positive_int(self.k, "k")
        group = _positive_int(self.weights_per_scale, "weights_per_scale")
        scale_bits = _positive_int(self.scale_bits, "scale_bits")
        if k > 16:
            raise ValueError("k must be between 1 and 16")
        if scale_bits > 64:
            raise ValueError("scale_bits must be between 1 and 64")
        rk = int(self.residual_k)
        if rk < 0 or rk > 16:
            raise ValueError("residual_k must be 0 or between 1 and 16")
        if rk:
            rg = self.residual_weights_per_scale or group
            rs = self.residual_scale_bits or scale_bits
            _positive_int(rg, "residual_weights_per_scale")
            _positive_int(rs, "residual_scale_bits")
            if rs > 64:
                raise ValueError("residual_scale_bits must be between 1 and 64")
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "weights_per_scale", group)
        object.__setattr__(self, "scale_bits", scale_bits)
        object.__setattr__(self, "residual_k", rk)
        if rk:
            object.__setattr__(self, "residual_weights_per_scale", int(self.residual_weights_per_scale or group))
            object.__setattr__(self, "residual_scale_bits", int(self.residual_scale_bits or scale_bits))
        else:
            object.__setattr__(self, "residual_weights_per_scale", 0)
            object.__setattr__(self, "residual_scale_bits", 0)
        if not self.name:
            object.__setattr__(self, "name", self.format_name())

    def format_name(self) -> str:
        result = f"TK-k{self.k}-g{self.weights_per_scale}-s{self.scale_bits}"
        if self.residual_k:
            result += f"+r{self.residual_k}-g{self.residual_weights_per_scale}-s{self.residual_scale_bits}"
        return result

    @property
    def code_bpw(self) -> float:
        return TRIT_BITS * self.k

    @property
    def scale_bpw(self) -> float:
        return self.scale_bits / self.weights_per_scale

    @property
    def ideal_bpw(self) -> float:
        value = self.code_bpw + self.scale_bpw
        if self.residual_k:
            value += TRIT_BITS * self.residual_k
            value += self.residual_scale_bits / self.residual_weights_per_scale
        return value

    def _plane_storage(self, weights: int, k: int, group: int, scale_bits: int) -> dict:
        groups = ceil(weights / group)
        code_bytes = ceil(weights * k / 5)
        scale_bytes = ceil(groups * scale_bits / 8)
        return {
            "weights": weights,
            "groups": groups,
            "code_bytes": code_bytes,
            "scale_bytes": scale_bytes,
            "bytes": code_bytes + scale_bytes,
            "ideal_bits": weights * TRIT_BITS * k + groups * scale_bits,
        }

    def storage(self, weights: int) -> dict:
        """Return ideal and byte-payload accounting for a finite weight count."""
        weights = _positive_int(weights, "weights")
        base = self._plane_storage(weights, self.k, self.weights_per_scale, self.scale_bits)
        residual = None
        if self.residual_k:
            residual = self._plane_storage(
                weights,
                self.residual_k,
                self.residual_weights_per_scale,
                self.residual_scale_bits,
            )
        total_bytes = base["bytes"] + (residual["bytes"] if residual else 0)
        ideal_bits = base["ideal_bits"] + (residual["ideal_bits"] if residual else 0)
        return {
            "name": self.name,
            "weights": weights,
            "ideal_bits": ideal_bits,
            "ideal_bpw": ideal_bits / weights,
            "payload_bytes": total_bytes,
            "payload_bits": total_bytes * 8,
            "payload_bpw": total_bytes * 8.0 / weights,
            "base": base,
            "residual": residual,
        }

    def to_dict(self) -> dict:
        return asdict(self)


def preset_specs() -> tuple[ContainerSpec, ...]:
    """Canonical k choices plus the S12 density choices Cody called out."""
    rows = [
        ("TK S12 / g20", 1, 20, 12),
        ("TK S12 / g15", 1, 15, 12),
        ("TK S12 / g12", 1, 12, 12),
        ("TK S12 / g8", 1, 8, 12),
        ("TK k1 / g128 / s10", 1, 128, 10),
        ("TK k3 / g128 / s10", 3, 128, 10),
        ("TK k5 / g128 / s10", 5, 128, 10),
        ("TK k8 / g128 / s10", 8, 128, 10),
        ("TK k10 / g128 / s10", 10, 128, 10),
        ("TK k13 / g128 / s10", 13, 128, 10),
        ("TK k15 / g128 / s10", 15, 128, 10),
    ]
    return tuple(ContainerSpec(k, g, s, name=name) for name, k, g, s in rows)


def density_ladder(k: int, scale_bits: int, weights: int = 1_000_000, max_group: int = 128) -> list[dict]:
    """Return every group size from 1 through ``max_group`` for one k/scale pair."""
    k = _positive_int(k, "k")
    scale_bits = _positive_int(scale_bits, "scale_bits")
    max_group = _positive_int(max_group, "max_group")
    return [
        {
            "group": group,
            "scale_bpw": scale_bits / group,
            "ideal_bpw": ContainerSpec(k, group, scale_bits).ideal_bpw,
            "payload_bpw": ContainerSpec(k, group, scale_bits).storage(weights)["payload_bpw"],
        }
        for group in range(1, max_group + 1)
    ]


def mixture_stats(parts: Iterable[Mapping]) -> dict:
    """Aggregate finite storage for parts with independent container specs."""
    total_weights = 0
    ideal_bits = 0.0
    payload_bytes = 0
    rows = []
    for part in parts:
        spec = part["spec"]
        if not isinstance(spec, ContainerSpec):
            spec = ContainerSpec(**spec)
        weights = _positive_int(part["weights"], "part weights")
        stats = spec.storage(weights)
        total_weights += weights
        ideal_bits += stats["ideal_bits"]
        payload_bytes += stats["payload_bytes"]
        rows.append({"label": str(part.get("label", spec.name)), "spec": spec.to_dict(), **stats})
    if not total_weights:
        return {"weights": 0, "ideal_bpw": 0.0, "payload_bytes": 0, "payload_bpw": 0.0, "parts": []}
    return {
        "weights": total_weights,
        "ideal_bits": ideal_bits,
        "ideal_bpw": ideal_bits / total_weights,
        "payload_bytes": payload_bytes,
        "payload_bits": payload_bytes * 8,
        "payload_bpw": payload_bytes * 8.0 / total_weights,
        "parts": rows,
    }


def spec_from_dict(data: Mapping) -> ContainerSpec:
    allowed = {
        "k",
        "weights_per_scale",
        "scale_bits",
        "residual_k",
        "residual_weights_per_scale",
        "residual_scale_bits",
        "name",
    }
    return ContainerSpec(**{key: data[key] for key in allowed if key in data})
