"""Read-only access to the vendored Atlas engine and strict recipe resolution."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import shutil
import subprocess


ROOT = Path(__file__).parent
MAX_PLAN_BYTES = 20_000_000


def atlas_request(**request):
    node = shutil.which("node")
    if not node:
        raise ValueError("Model Atlas needs Node.js 18 or newer for its existing architecture engine.")
    result = subprocess.run([node, str(ROOT / "model_atlas_bridge.mjs")], input=json.dumps(request),
        text=True, capture_output=True, timeout=30, check=False)
    if result.returncode:
        raise ValueError(result.stderr.strip() or "Atlas could not read this model/recipe.")
    return json.loads(result.stdout)


def read_recipe(path):
    path = Path(path)
    if path.stat().st_size > MAX_PLAN_BYTES:
        raise ValueError("Atlas recipes must be smaller than 20 MB.")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Expected an Atlas JSON recipe object.")
    return data


def canonical_tensor(name):
    """Resolve known Qwen text wrappers, never arbitrary vision/audio prefixes."""
    for prefix in ("model.language_model.", "language_model.model.", "language_model.", "model.text_model."):
        if name.startswith(prefix):
            name = "model." + name[len(prefix):]
            break
    if name == "model.lm_head.weight":
        return "lm_head.weight"
    return name


def format_policy(row, plan):
    """Translate executable storage choices; keep unsupported targets visible."""
    target = row.get("precision", "original")
    if row.get("protection") and target != "original":
        raise ValueError("Protected entry has a non-original precision.")
    if row.get("kind", "weight") != "weight" or "@flow." in row["path"]:
        if target != "original":
            raise ValueError("Runtime/state targets require a runtime implementation.")
        return {"kind": "runtime", "precision": target}
    if target == "original":
        return {"kind": "original", "precision": target}
    if target in {"BF16", "FP16", "FP32"}:
        return {"kind": "float", "precision": target,
                "dtype": {"BF16":"bfloat16", "FP16":"float16", "FP32":"float32"}[target]}
    match = re.fullmatch(r"TKinteger-k(1[0-6]|[1-9])(?:-g([1-9][0-9]*)-s(12|16|32))?", target)
    if not match:
        raise ValueError(f"{target} has no executable storage adapter here.")
    k = int(match[1])
    group = row.get("group_size", int(match[2]) if match[2] else plan.get("size_assumptions", {}).get("scale_group_size", 128))
    if type(group) is not int or not 1 <= group <= 65536:
        raise ValueError("Weights per scale must be an integer in 1–65536.")
    if match[2] and group != int(match[2]):
        raise ValueError("Layer group size conflicts with the format name.")
    bits = int(match[3]) if match[3] else (16 if k <= 5 else 32)
    override = plan.get("size_assumptions", {}).get("scale_bytes_per_group", -1)
    if not match[3] and override != -1:
        if override not in (2, 4):
            raise ValueError("Scale-byte override is not an executable fp16/fp32 scale format.")
        bits = int(override * 8)
    return {"kind": "atlas-tk", "precision": target, "k": k, "group": group, "scale_bits": bits}


def review_recipe(plan, tensors=None):
    """Resolve module groups against inspected tensor headers before any write."""
    if not isinstance(plan.get("precision"), list):
        raise ValueError("Import the recipe through Model Atlas to restore its effective precision list.")
    entries, issues, display = {}, [], []
    for row in plan["precision"]:
        path = row["path"]
        if path in entries:
            issues.append(f"{path}: duplicate assignment")
            continue
        try:
            policy = format_policy(row, plan)
            status = "Runtime unchanged" if policy["kind"] == "runtime" else "Ready for tensor matching"
        except ValueError as exc:
            policy = {"kind":"unsupported", "precision":row.get("precision")}
            status = str(exc)
            issues.append(f"{path}: {exc}")
        entries[path] = {"row":row, "policy":policy, "matches":[]}
        display.append({"path":path, "precision":row.get("precision", "original"),
            "group":policy.get("group", row.get("group_size", "—")),
            "scales":f"S{policy['scale_bits']}" if "scale_bits" in policy else "—",
            "protection":", ".join(row.get("protection", [])) or "—", "status":status})
    policies = {}
    tied = plan.get("tied_embeddings", False)
    if tied:
        pair = [entries.get(p, {}).get("policy", {"kind":"original", "precision":"original"})
                for p in ("model.embed_tokens", "lm_head")]
        if pair[0] != pair[1]:
            issues.append("Tied embedding and head assignments must match.")
    if tensors is not None:
        for tensor in tensors:
            name = tensor["tensor"]
            canonical = canonical_tensor(name)
            candidates = [p for p, e in entries.items() if e["policy"]["kind"] != "runtime"
                          and (canonical == p or canonical.startswith(p + "."))]
            policy = {"kind":"original", "precision":"original"}
            if candidates:
                # A more specific path overrides a containing module group.
                path = max(candidates, key=len)
                entry = entries[path]
                policy = copy.deepcopy(entry["policy"])
                for parent in candidates:
                    entries[parent]["matches"].append(name)
                shape = tensor["shape"]
                if policy["kind"] == "atlas-tk":
                    if not canonical.endswith(".weight") or len(shape) != 2:
                        issues.append(f"{name}: Atlas TK formats need a 2-D weight adapter; this tensor will not be silently quantized.")
                    elif policy["scale_bits"] != 12 and shape[1] % policy["group"]:
                        issues.append(f"{name}: width {shape[1]} is not divisible by group {policy['group']}.")
                    if tensor.get("dtype") not in {"F16", "BF16", "F32", "F64", "float16", "bfloat16", "float32", "float64"}:
                        issues.append(f"{name}: quantization requires floating weights.")
                policy["recipe_path"] = path
            policies[name] = policy
        for item in display:
            entry = entries[item["path"]]
            if entry["policy"]["kind"] in {"runtime", "unsupported"}:
                continue
            count = len(entry["matches"])
            alias = tied and item["path"] == "lm_head" and entries.get("model.embed_tokens", {}).get("matches")
            item["status"] = f"{count} tensor(s) matched" if count else ("Tied alias; stored with embedding" if alias else "No matching tensor")
            if not count and not alias and entry["policy"]["kind"] != "original":
                issues.append(f"{item['path']}: selected target does not match the source checkpoint.")
        if not any(p["kind"] not in {"original", "runtime"} for p in policies.values()):
            issues.append("The recipe selects no changed tensors in this checkpoint.")
        # Check dimensions/architecture against the actual inventory, not its file name.
        config = plan.get("config", {})
        text = config.get("text_config", config.get("thinker_config", {}).get("text_config", config))
        blocks = set()
        for tensor in tensors:
            canonical = canonical_tensor(tensor["tensor"])
            match = re.match(r"model\.layers\.(\d+)\.", canonical)
            if match:
                blocks.add(int(match[1]))
            if canonical in {"model.embed_tokens.weight", "lm_head.weight"}:
                expected = [text.get("vocab_size"), text.get("hidden_size")]
                if all(v is not None for v in expected) and tensor["shape"] != expected:
                    issues.append(f"{tensor['tensor']}: shape does not match the recipe model config.")
        if text.get("num_hidden_layers") is not None and blocks != set(range(text["num_hidden_layers"])):
            issues.append("Source decoder blocks do not match the recipe model config. Select the complete matching model.")
    return {"rows":display, "issues":issues, "policies":policies, "matched":tensors is not None}
