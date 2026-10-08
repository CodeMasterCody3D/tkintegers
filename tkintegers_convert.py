"""Safetensors-to-Tk-Integer conversion engine.

The converter writes a manifest plus one small safetensors shard per source
tensor.  That deliberately avoids loading an entire model or holding an
entire output shard in memory.  The output is a packed Tk-Integer container,
not an HF-compatible dense checkpoint; a loader must read the manifest.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping

from tkintegers import resolve_scale_policy


FORMAT = "tkintegers-safetensors-v1"


@dataclass(frozen=True)
class ConversionConfig:
    weight_k: int = 5
    weights_per_scale: int = 128
    scale_type: str = "existing-k10-log"
    scale_k: int = 5

    def __post_init__(self) -> None:
        if isinstance(self.weight_k, bool) or not 1 <= int(self.weight_k) <= 16:
            raise ValueError("weight_k must be between 1 and 16")
        if isinstance(self.weights_per_scale, bool) or int(self.weights_per_scale) <= 0:
            raise ValueError("weights_per_scale must be positive")
        resolve_scale_policy(self.scale_type, self.scale_k)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["scale"] = resolve_scale_policy(self.scale_type, self.scale_k).to_dict()
        return data


def _runtime():
    try:
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "model conversion requires torch and safetensors; install requirements.txt first"
        ) from exc
    return torch, safe_open, save_file


def _model_shards(model_path: str | os.PathLike) -> dict[str, Path]:
    """Return the tensor-to-shard map for a file or HF-style directory."""
    _, safe_open, _ = _runtime()
    path = Path(model_path).expanduser().resolve()
    if path.is_file():
        if path.suffix != ".safetensors":
            raise ValueError(f"expected a .safetensors file: {path}")
        with safe_open(str(path), framework="pt") as handle:
            return {name: path for name in handle.keys()}
    if not path.is_dir():
        raise FileNotFoundError(path)
    index = path / "model.safetensors.index.json"
    if index.is_file():
        raw = json.loads(index.read_text())
        return {name: path / shard for name, shard in raw["weight_map"].items()}
    single = path / "model.safetensors"
    if single.is_file():
        with safe_open(str(single), framework="pt") as handle:
            return {name: single for name in handle.keys()}
    candidates = sorted(path.glob("*.safetensors"))
    if len(candidates) == 1:
        with safe_open(str(candidates[0]), framework="pt") as handle:
            return {name: candidates[0] for name in handle.keys()}
    raise FileNotFoundError(f"no unambiguous safetensors model under {path}")


def inspect_model(model_path: str | os.PathLike) -> list[dict]:
    """Read tensor names and shapes without materializing model tensors."""
    _, safe_open, _ = _runtime()
    rows = []
    for name, shard in sorted(_model_shards(model_path).items()):
        with safe_open(str(shard), framework="pt") as handle:
            tensor_slice = handle.get_slice(name)
            shape = list(tensor_slice.get_shape())
            dtype = "unknown"
            get_dtype = getattr(tensor_slice, "get_dtype", None)
            if get_dtype is not None:
                dtype = str(get_dtype()).replace("torch.", "")
            rows.append({
                "tensor": name,
                "shape": shape,
                "dtype": dtype,
                "params": math.prod(shape),
                "source_shard": str(shard),
            })
    return rows


def _anchor(value: float) -> tuple[int, int]:
    if not math.isfinite(value) or value <= 0:
        return 0, 0
    exponent = math.floor(math.log(value, 3.0)) - 15
    mantissa = int(round(value / (3.0 ** exponent)))
    return mantissa, exponent


def _anchor_value(meta, torch) -> float:
    mantissa, exponent = (int(item) for item in meta.tolist()[:2])
    return float(mantissa) * (3.0 ** exponent)


def _packed_ternary(values, k: int, torch):
    from base3_pack import pack_ktrit

    packed, width, _ = pack_ktrit(values.reshape(1, -1).to(torch.int32), k)
    return packed.reshape(-1).cpu(), width


def _encode_scales(scales, policy, torch) -> tuple[dict[str, object], dict]:
    """Encode group scales and return payload tensors plus manifest metadata."""
    scales = scales.double().clamp_min(1e-30)
    descriptor = {"policy": policy.to_dict(), "count": int(scales.numel())}

    if policy.family == "float":
        dtype = getattr(torch, policy.dtype)
        return {"scales": scales.to(dtype).cpu()}, {**descriptor, "payload": ["scales"]}

    if policy.family == "tk-log-fixed":
        from tk_s12 import encode, pack12

        codes = encode(scales)
        return {"scale_codes": pack12(codes).cpu()}, {
            **descriptor,
            "payload": ["scale_codes"],
            "encoding": "packed-12-bit-log2",
        }

    if policy.family == "existing":
        from tk_codec import pack_hi

        codes, meta = pack_hi(scales, k=10, log=True)
        return {"scale_codes": codes.to(torch.int16).cpu(), "scale_meta": meta.cpu()}, {
            **descriptor,
            "payload": ["scale_codes", "scale_meta"],
            "encoding": "existing-k10-log-integer-anchor",
        }

    if policy.family == "int-linear":
        max_code = {"int8": 127, "int16": 32767, "int32": 2147483647}[policy.dtype]
        mantissa, exponent = _anchor(float(scales.max()))
        anchor = _anchor_value(torch.tensor([mantissa, exponent], dtype=torch.int64), torch)
        dtype = getattr(torch, policy.dtype)
        codes = torch.round(scales / anchor * max_code).clamp(0, max_code).to(dtype)
        return {
            "scale_codes": codes.cpu(),
            "scale_meta": torch.tensor([mantissa, exponent, max_code], dtype=torch.int64),
        }, {
            **descriptor,
            "payload": ["scale_codes", "scale_meta"],
            "encoding": "linear-integer-with-integer-anchor",
        }

    if policy.family in {"ternary-linear", "ternary-log"}:
        from base3_pack import pack_ktrit
        from tk_codec import pack_hi

        if policy.family == "ternary-linear":
            levels = (3 ** policy.k - 1) // 2
            mantissa, exponent = _anchor(float(scales.max()))
            anchor = _anchor_value(torch.tensor([mantissa, exponent], dtype=torch.int64), torch)
            codes = torch.round(scales / anchor * levels).clamp(0, levels).to(torch.int32)
        else:
            codes, meta = pack_hi(scales, k=policy.k, log=True)
            mantissa, exponent = int(meta[0]), int(meta[1])
            codes = codes.to(torch.int32)
            levels = (3 ** policy.k - 1) // 2
            anchor = None
        packed, width, _ = pack_ktrit(codes.reshape(1, -1), policy.k)
        if policy.family == "ternary-log":
            scale_meta = meta.to(torch.int64).cpu()
        else:
            scale_meta = torch.tensor([mantissa, exponent, levels], dtype=torch.int64)
        return {"scale_codes": packed.reshape(-1).cpu(), "scale_meta": scale_meta}, {
            **descriptor,
            "payload": ["scale_codes", "scale_meta"],
            "encoding": "packed-balanced-ternary",
            "packed_width": width,
            "logarithmic": policy.family == "ternary-log",
        }

    raise ValueError(f"unsupported scale policy family: {policy.family}")


def quantize_tensor(tensor, config: ConversionConfig) -> tuple[dict, dict, dict]:
    """Quantize one floating tensor into payload tensors and a manifest entry."""
    torch, _, _ = _runtime()
    if not tensor.dtype.is_floating_point:
        return {"raw": tensor.detach().cpu().contiguous()}, {
            "kind": "raw",
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
        }, {"params": tensor.numel(), "quantized": False}

    original_shape = list(tensor.shape)
    flat = tensor.detach().cpu().float().reshape(-1)
    group = int(config.weights_per_scale)
    padding = (-flat.numel()) % group
    if padding:
        flat = torch.cat((flat, flat.new_zeros(padding)))
    grouped = flat.reshape(-1, group)
    scales = grouped.abs().amax(-1).clamp_min(1e-30)
    levels = (3 ** int(config.weight_k) - 1) // 2
    codes = torch.round(grouped / scales.unsqueeze(-1) * levels).clamp(-levels, levels).to(torch.int32)
    packed, packed_width = _packed_ternary(codes.reshape(-1), int(config.weight_k), torch)
    policy = resolve_scale_policy(config.scale_type, config.scale_k)
    scale_payload, scale_descriptor = _encode_scales(scales, policy, torch)
    payload = {"codes": packed, **scale_payload}
    entry = {
        "kind": "tk-quantized",
        "shape": original_shape,
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "padded_params": int(flat.numel()),
        "padding": int(padding),
        "weight_k": int(config.weight_k),
        "weights_per_scale": group,
        "weight_code_bytes": int(packed.numel()),
        "scale": scale_descriptor,
    }
    stats = {
        "params": tensor.numel(),
        "quantized": True,
        "payload_bytes": sum(value.numel() * value.element_size() for value in payload.values()),
    }
    return payload, entry, stats


def _write_manifest(path: Path, manifest: Mapping) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, path)


def convert_model(
    model_path: str | os.PathLike,
    output_dir: str | os.PathLike,
    config: ConversionConfig,
    progress: Callable[[int, int, str], None] | None = None,
    *,
    atlas_plan: dict | None = None,
) -> dict:
    """Convert a safetensors file/model directory without overwriting inputs."""
    torch, safe_open, save_file = _runtime()
    source = Path(model_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if source == output or (source.is_dir() and output.is_relative_to(source)):
        raise ValueError("output must be outside the selected source model")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"refusing to write into non-empty output directory: {output}")
    # Match the reviewed recipe against the actual source again before writing.
    policies = None
    if atlas_plan is not None:
        from model_atlas import review_recipe
        review = review_recipe(atlas_plan, inspect_model(source))
        if review["issues"]:
            raise ValueError("Atlas recipe cannot convert this source:\n" + "\n".join(review["issues"]))
        policies = review["policies"]
    output.mkdir(parents=True, exist_ok=True)
    tensor_dir = output / "tensors"
    tensor_dir.mkdir()

    shards = _model_shards(source)
    names = sorted(shards)
    manifest = {
        "format": FORMAT,
        "source": str(source),
        "config": config.to_dict(),
        "tensor_count": len(names),
        "tensors": {},
        "summary": {"params": 0, "quantized_tensors": 0, "raw_tensors": 0, "payload_bytes": 0},
    }
    if atlas_plan is not None:
        manifest["atlas_recipe"] = atlas_plan
        manifest["method"] = "Atlas format mix: unweighted scale fitting and stored-scale snapping; quant_job/Hessian/training options are not executed"
    manifest_path = output / "manifest.json"
    _write_manifest(manifest_path, manifest)

    for index, name in enumerate(names, 1):
        with safe_open(str(shards[name]), framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(name)
        policy = policies[name] if policies is not None else None
        if policy is None:
            payload, entry, stats = quantize_tensor(tensor, config)
        elif policy["kind"] == "atlas-tk":
            from vendor.model_atlas.quant.codec import quantize
            payload = quantize(tensor, None, policy)
            entry = {"kind":"atlas-tk", "shape":list(tensor.shape),
                     "dtype":str(tensor.dtype).replace("torch.", ""), "atlas_spec":policy}
            stats = {"params":tensor.numel(), "quantized":True,
                     "payload_bytes":sum(t.numel()*t.element_size() for t in payload.values())}
        else:
            value = tensor
            if policy["kind"] == "float" and tensor.dtype.is_floating_point:
                value = tensor.to(getattr(torch, policy["dtype"]))
                if not torch.isfinite(value).all():
                    raise ValueError(f"{name}: selected float format overflows or contains non-finite values")
            payload = {"raw":value.contiguous()}
            entry = {"kind":"raw", "shape":list(value.shape), "dtype":str(value.dtype).replace("torch.", "")}
            stats = {"params":value.numel(), "quantized":False, "payload_bytes":value.numel()*value.element_size()}
        if policy is not None:
            entry["atlas_assignment"] = policy
        output_payload = {}
        for key, value in payload.items():
            output_payload[f"{name}.tk_{key}"] = value.contiguous().cpu()
        shard_name = f"tensor-{index:06d}.safetensors"
        shard_path = tensor_dir / shard_name
        temporary = tensor_dir / f".{shard_name}.tmp"
        save_file(output_payload, str(temporary), metadata={"format": FORMAT, "tensor": name})
        os.replace(temporary, shard_path)
        entry.update({
            "source_shard": str(shards[name]),
            "output_shard": f"tensors/{shard_name}",
            "payload_keys": sorted(output_payload),
        })
        manifest["tensors"][name] = entry
        manifest["summary"]["params"] += int(stats["params"])
        manifest["summary"]["payload_bytes"] += int(stats.get("payload_bytes", tensor.numel() * tensor.element_size()))
        if stats["quantized"]:
            manifest["summary"]["quantized_tensors"] += 1
        else:
            manifest["summary"]["raw_tensors"] += 1
        _write_manifest(manifest_path, manifest)
        if progress:
            progress(index, len(names), name)

    return manifest


def load_converted_tensor(
    output_dir: str | os.PathLike,
    tensor_name: str,
    dtype: str | None = None,
):
    """Decode one tensor from a Tk-Integer output directory.

    This is intentionally tensor-at-a-time so callers can stream a model
    without materializing every converted tensor.
    """
    torch, safe_open, _ = _runtime()
    root = Path(output_dir).expanduser().resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    entry = manifest["tensors"][tensor_name]
    shard = root / entry["output_shard"]
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        payload = {key: handle.get_tensor(key) for key in entry["payload_keys"]}

    if entry["kind"] == "raw":
        value = payload[f"{tensor_name}.tk_raw"]
        return value.to(getattr(torch, dtype)) if dtype else value

    if entry["kind"] == "atlas-tk":
        from vendor.model_atlas.quant.codec import unpack
        value = unpack({key:payload[f"{tensor_name}.tk_{key}"] for key in ("codes", "scales")},
                       entry["shape"], entry["atlas_spec"])
        return value.to(getattr(torch, dtype or entry["dtype"]))

    from base3_pack import unpack_ktrit

    code_key = f"{tensor_name}.tk_codes"
    codes = unpack_ktrit(
        payload[code_key].reshape(1, -1),
        int(entry["padded_params"]),
        int(entry["weight_k"]),
    ).reshape(-1)
    scale_info = entry["scale"]
    policy = resolve_scale_policy(scale_info["policy"]["key"], scale_info["policy"].get("k", 5) or 5)
    scale_codes_key = f"{tensor_name}.tk_scale_codes"
    scale_meta_key = f"{tensor_name}.tk_scale_meta"
    scale_codes = payload.get(scale_codes_key)
    scale_meta = payload.get(scale_meta_key)

    if policy.family == "float":
        scales = payload[f"{tensor_name}.tk_scales"].float()
    elif policy.family == "tk-log-fixed":
        from tk_s12 import decode, unpack12

        scales = decode(unpack12(scale_codes, int(scale_info["count"]))).float()
    elif policy.family == "existing":
        from tk_codec import unpack_hi

        scales = unpack_hi(scale_codes, scale_meta).float()
    elif policy.family == "int-linear":
        max_code = int(scale_meta[2])
        anchor = _anchor_value(scale_meta, torch)
        scales = scale_codes.float() / max_code * anchor
    elif policy.family == "ternary-linear":
        from base3_pack import unpack_ktrit

        scale_codes = unpack_ktrit(
            scale_codes.reshape(1, -1),
            int(scale_info["packed_width"]),
            policy.k,
        ).reshape(-1)
        levels = int(scale_meta[2])
        scales = scale_codes.float() / levels * _anchor_value(scale_meta, torch)
    elif policy.family == "ternary-log":
        from base3_pack import unpack_ktrit
        from tk_codec import unpack_hi

        scale_codes = unpack_ktrit(
            scale_codes.reshape(1, -1),
            int(scale_info["packed_width"]),
            policy.k,
        ).reshape(-1)
        scales = unpack_hi(scale_codes, scale_meta).float()
    else:
        raise ValueError(f"unsupported scale policy family: {policy.family}")

    group = int(entry["weights_per_scale"])
    levels = (3 ** int(entry["weight_k"]) - 1) // 2
    value = (codes.float().reshape(-1, group) / levels * scales.reshape(-1, 1)).reshape(-1)
    value = value[: math.prod(entry["shape"])].reshape(entry["shape"])
    if dtype:
        value = value.to(getattr(torch, dtype))
    else:
        value = value.to(getattr(torch, entry["dtype"]))
    return value
