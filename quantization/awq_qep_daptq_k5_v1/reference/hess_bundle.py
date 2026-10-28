#!/usr/bin/env python3
"""hess_bundle.py -- THE STANDARD (Cody 2026-10-09 ~8:45 AM): "when it creates a hessian live when training (making the hessian map and using
it live in the training) they are uploaded to huggingface with the rows, trace, dataset (and the teachers output) that is used ... make this
standard practice". Every Hessian a run builds LIVE (built during the run and used by that same run: QEP's H-hat in a placement, any
--save-hessians map, a carried walk feeding a recon) goes to Hugging Face TOGETHER with what it was built from, in ONE private dataset repo:

  H/             the Hessian files exactly as the run wrote them (+ its hessians.json / pass jsons)
  data/          the fit pool (the exact token ids of every example), the teacher's outputs (prompt + full thinking + answer, gold, pass /
                 fail), the held-out slice -- whatever --data names
  manifest.json  model, teacher, datasets (name, config, split, license), token totals, examples, how many carry thinking, the scripts with
                 md5 + their code-vault copies, the run card, notes
  README.md      what it is and how to load it

  python vmkit/hess_bundle.py --hessian DIR --data FILE [FILE ...] --repo OWNER/NAME --model M --teacher T --pool POOL.pt
      --dataset "openai/gsm8k main train (MIT)" [--dataset ...] --script PATH [--script ...] --run-card CARD.json --note TEXT --out BUNDLE
Refuses when the pool carries no thinking traces (rule 8a) unless --allow-no-think. Files are hard-linked into the bundle when they share a
disk (no second copy of a 2 GB map). Verified by listing the repo (hf_push_maps.push: every file, same size). Last line ========== DONE ==========."""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time

FORGE = "/content/forge" if os.path.isdir("/content/forge") else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, FORGE)
sys.path.insert(0, os.path.join(FORGE, "vmkit"))


def md5(p, n=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(n), b""):
            h.update(b)
    return h.hexdigest()


def place(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def pool_stats(pool_path, model):
    import torch
    pool = torch.load(pool_path, map_location="cpu", weights_only=False)
    n_tok = sum(int(e["n_tokens"]) for e in pool)
    think = None
    if pool and "has_think" in pool[0]:
        think = sum(1 for e in pool if e.get("has_think"))
    else:
        from transformers import AutoTokenizer
        tid = AutoTokenizer.from_pretrained(model).convert_tokens_to_ids("<think>")
        think = sum(1 for e in pool if bool((e["ids"].reshape(-1) == tid).any()))
    lens = sorted(int(e["n_tokens"]) for e in pool)
    return {"examples": len(pool), "tokens": n_tok, "with_thinking": think, "longest": lens[-1] if lens else 0,
            "median": lens[len(lens) // 2] if lens else 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hessian", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--pool", required=True, help="the fit pool the Hessian walked (one of --data or extra)")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--dataset", action="append", default=[])
    ap.add_argument("--script", action="append", default=[])
    ap.add_argument("--run-card", default="")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-no-think", action="store_true")
    ap.add_argument("--no-push", action="store_true", help="build the bundle only (tests)")
    a = ap.parse_args()
    t0 = time.time()
    hfiles = sorted(glob.glob(os.path.join(a.hessian, "*")))
    hfiles = [f for f in hfiles if os.path.isfile(f)]
    if not any(f.endswith(".safetensors") for f in hfiles):
        raise SystemExit(f"REFUSING: no Hessian files in {a.hessian}")
    st = pool_stats(a.pool, a.model)
    print(f"HESS BUNDLE: {len(hfiles)} Hessian files from {a.hessian} | pool {st['examples']} examples, {st['tokens']:,} tokens, "
          f"{st['with_thinking']} with thinking, longest {st['longest']:,}, median {st['median']:,}", flush=True)
    if not st["with_thinking"] and not a.allow_no_think:
        raise SystemExit("REFUSING: the pool carries NO thinking traces (rule 8a: every placement Hessian walks thinking + answer traces)")
    for f in hfiles:
        place(f, os.path.join(a.out, "H", os.path.basename(f)))
    for f in list(a.data) + [a.pool]:
        if not os.path.exists(f):
            raise SystemExit(f"REFUSING: missing data file {f}")
        place(f, os.path.join(a.out, "data", os.path.basename(f)))
    scripts = []
    for s in a.script:
        if os.path.exists(s):
            m = md5(s)
            vault = sorted(glob.glob(os.path.join(FORGE, "code_vault", os.path.basename(s).rsplit(".", 1)[0] + ".*." + m[:8] + ".*")))
            scripts.append({"path": s, "md5": m, "vault": vault[-1] if vault else None})
    card = json.load(open(a.run_card)) if a.run_card and os.path.exists(a.run_card) else {}
    man = {"what": "a Hessian built LIVE by a run, with the data it was built from (the TAARDIS standard, 2026-10-09)",
           "repo": a.repo, "model": a.model, "teacher": a.teacher, "datasets": a.dataset, "pool": st,
           "hessian_files": [os.path.basename(f) for f in hfiles],
           "data_files": sorted(os.path.basename(f) for f in set(list(a.data) + [a.pool])),
           "scripts": scripts, "run_card": card.get("run_name") or card.get("name") or a.run_card, "note": a.note,
           "built": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    open(os.path.join(a.out, "README.md"), "w").write(
        f"# {a.repo.split('/')[-1]}\n\nA Hessian built live by a TAARDIS run, stored with the data it was built from.\n\n"
        f"- model: `{a.model}` · teacher: `{a.teacher}`\n- pool: {st['examples']} complete examples, {st['tokens']:,} tokens, "
        f"{st['with_thinking']} carry full thinking + answer traces (longest {st['longest']:,}, median {st['median']:,})\n"
        + "".join(f"- dataset: {d}\n" for d in a.dataset)
        + f"- note: {a.note}\n\n`H/` = the Hessian files as written (per linear, unrotated unless the manifest says otherwise). `data/` = the fit "
          f"pool (`torch.load` -> list of {{ids, n_tokens, ...}}), the teacher's outputs and any held-out slice. `manifest.json` = scripts "
          f"with md5 and code-vault copies, the run card.\n")
    print(f"bundle {a.out}: H/ {len(hfiles)} files, data/ {len(man['data_files'])} files, manifest + README", flush=True)
    if not a.no_push:
        from hf_push_maps import push
        push(a.out, a.repo, log=lambda m: print(m, flush=True))
    print(f"({time.time() - t0:.0f} s)", flush=True)
    print("========== DONE ==========", flush=True)


if __name__ == "__main__":
    main()
