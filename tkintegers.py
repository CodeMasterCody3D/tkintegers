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
from typing import Iterable, Mapping


TRIT_BITS = 8.0 / 5.0
ALL_KS = tuple(range(1, 17))
DEFAULT_KS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 15)
CANONICAL_KS = (1, 3, 5, 8, 10, 13, 15)
DEFAULT_SCALE_BITS = (8, 10, 12, 16, 32)
DEFAULT_SCALE_GROUPS = (1, 2, 4, 8, 12, 15, 16, 20, 24, 32, 48, 64, 96, 128)


@dataclass(frozen=True)
class ScalePolicy:
    """How one per-group scale is stored in a converted safetensor."""

    key: str
    label: str
    family: str
    bits: float
    k: int = 0
    dtype: str = ""

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "family": self.family,
            "bits": self.bits,
            "k": self.k,
            "dtype": self.dtype,
        }


_FIXED_SCALE_POLICIES = {
    "existing-k10-log": ScalePolicy(
        "existing-k10-log", "Existing container: k10 log + integer anchor", "existing", 16.0, 10
    ),
    "tk-s12": ScalePolicy("tk-s12", "TK S12: 12-bit log2", "tk-log-fixed", 12.0),
    "int8-linear": ScalePolicy("int8-linear", "int8 linear + integer anchor", "int-linear", 8.0, dtype="int8"),
    "int16-linear": ScalePolicy("int16-linear", "int16 linear + integer anchor", "int-linear", 16.0, dtype="int16"),
    "int32-linear": ScalePolicy("int32-linear", "int32 linear + integer anchor", "int-linear", 32.0, dtype="int32"),
    "bf16": ScalePolicy("bf16", "bf16 scale values", "float", 16.0, dtype="bfloat16"),
    "fp16": ScalePolicy("fp16", "fp16 scale values", "float", 16.0, dtype="float16"),
    "fp32": ScalePolicy("fp32", "fp32 scale values", "float", 32.0, dtype="float32"),
    "fp64": ScalePolicy("fp64", "fp64 scale values", "float", 64.0, dtype="float64"),
}


def scale_policy_choices() -> tuple[ScalePolicy, ...]:
    """Return the fixed UI choices; ternary choices take a separate k field."""
    return (
        _FIXED_SCALE_POLICIES["existing-k10-log"],
        _FIXED_SCALE_POLICIES["tk-s12"],
        ScalePolicy("ternary-linear", "Ternary integer linear scale (choose k)", "ternary-linear", TRIT_BITS * 5, 5),
        ScalePolicy("ternary-log", "Ternary integer log2 scale (choose k)", "ternary-log", TRIT_BITS * 5, 5),
        _FIXED_SCALE_POLICIES["int8-linear"],
        _FIXED_SCALE_POLICIES["int16-linear"],
        _FIXED_SCALE_POLICIES["int32-linear"],
        _FIXED_SCALE_POLICIES["bf16"],
        _FIXED_SCALE_POLICIES["fp16"],
        _FIXED_SCALE_POLICIES["fp32"],
        _FIXED_SCALE_POLICIES["fp64"],
    )


def resolve_scale_policy(key: str, k: int = 5) -> ScalePolicy:
    """Resolve a scale choice and validate its optional ternary k."""
    key = str(key).strip().lower()
    if key == "existing":
        key = "existing-k10-log"
    if key in _FIXED_SCALE_POLICIES:
        return _FIXED_SCALE_POLICIES[key]
    if key not in {"ternary-linear", "ternary-log"}:
        raise ValueError(f"unknown scale policy: {key}")
    k = _positive_int(k, "scale k")
    if k > 16:
        raise ValueError("scale k must be between 1 and 16")
    family = key.removeprefix("ternary-")
    return ScalePolicy(key, f"Ternary integer {family} scale k{k}", key, TRIT_BITS * k, k)


def scale_policy_label(key: str, k: int = 5) -> str:
    return resolve_scale_policy(key, k).label


def _positive_int(value: int, label: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _positive_number(value: float, label: str) -> float:
    if isinstance(value, bool) or float(value) <= 0:
        raise ValueError(f"{label} must be positive")
    return float(value)


@dataclass(frozen=True)
class ContainerSpec:
    """One weight plane plus an optional carried residual plane."""

    k: int
    weights_per_scale: int
    scale_bits: float
    residual_k: int = 0
    residual_weights_per_scale: int = 0
    residual_scale_bits: int = 0
    name: str = ""
    scale_type: str = "custom"
    scale_k: int = 0
    residual_scale_type: str = "custom"
    residual_scale_k: int = 0

    def __post_init__(self) -> None:
        k = _positive_int(self.k, "k")
        group = _positive_int(self.weights_per_scale, "weights_per_scale")
        if k > 16:
            raise ValueError("k must be between 1 and 16")
        scale_type = str(self.scale_type or "custom").strip().lower()
        scale_k = int(self.scale_k or 0)
        if scale_type == "custom":
            scale_bits = _positive_number(self.scale_bits, "scale_bits")
        else:
            policy = resolve_scale_policy(scale_type, scale_k or 5)
            scale_bits = policy.bits
            scale_k = policy.k
        if scale_bits > 64:
            raise ValueError("scale_bits must be between 1 and 64")
        rk = int(self.residual_k)
        if rk < 0 or rk > 16:
            raise ValueError("residual_k must be 0 or between 1 and 16")
        if rk:
            rg = self.residual_weights_per_scale or group
            rs = self.residual_scale_bits or scale_bits
            _positive_int(rg, "residual_weights_per_scale")
            _positive_number(rs, "residual_scale_bits")
            if float(rs) > 64:
                raise ValueError("residual_scale_bits must be between 1 and 64")
        residual_scale_type = str(self.residual_scale_type or "custom").strip().lower()
        residual_scale_k = int(self.residual_scale_k or 0)
        if residual_scale_type != "custom" and rk:
            policy = resolve_scale_policy(residual_scale_type, residual_scale_k or 5)
            residual_scale_bits = policy.bits
            residual_scale_k = policy.k
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "weights_per_scale", group)
        object.__setattr__(self, "scale_bits", scale_bits)
        object.__setattr__(self, "scale_type", scale_type)
        object.__setattr__(self, "scale_k", scale_k)
        object.__setattr__(self, "residual_k", rk)
        if rk:
            object.__setattr__(self, "residual_weights_per_scale", int(self.residual_weights_per_scale or group))
            object.__setattr__(self, "residual_scale_bits", float(self.residual_scale_bits or scale_bits))
            object.__setattr__(self, "residual_scale_type", residual_scale_type)
            object.__setattr__(self, "residual_scale_k", residual_scale_k)
        else:
            object.__setattr__(self, "residual_weights_per_scale", 0)
            object.__setattr__(self, "residual_scale_bits", 0)
            object.__setattr__(self, "residual_scale_type", "custom")
            object.__setattr__(self, "residual_scale_k", 0)
        if not self.name:
            object.__setattr__(self, "name", self.format_name())

    def format_name(self) -> str:
        scale_name = self.scale_type if self.scale_type != "custom" else f"s{self.scale_bits:g}"
        if self.scale_type.startswith("ternary-"):
            scale_name += f"-k{self.scale_k}"
        result = f"TK-k{self.k}-g{self.weights_per_scale}-{scale_name}"
        if self.residual_k:
            residual_name = (
                self.residual_scale_type
                if self.residual_scale_type != "custom"
                else f"s{self.residual_scale_bits:g}"
            )
            if self.residual_scale_type.startswith("ternary-"):
                residual_name += f"-k{self.residual_scale_k}"
            result += f"+r{self.residual_k}-g{self.residual_weights_per_scale}-{residual_name}"
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
        "scale_type",
        "scale_k",
        "residual_scale_type",
        "residual_scale_k",
    }
    return ContainerSpec(**{key: data[key] for key in allowed if key in data})
