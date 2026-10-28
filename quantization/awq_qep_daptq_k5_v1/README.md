# AWQ + QEP + DAPTQ K5 — preserved recipe v1

**The combined method passed the previously failed interval question at exactly
the original saved weight size.** This directory preserves that implementation,
its selected settings, calibration and result, and exposes reusable building
blocks for TKintegers. Method identifier: `awq_qep_daptq_k5_v1`.

## Rules

1. "Use the recorded K5/group128/integer-k10-log format and its real kernels."
2. "Carry complete calibration examples; exclude padding and evaluation questions."
3. "Fold AWQ into existing tensors and transform QEP statistics into that basis."
4. "Accept DAPTQ proposals only after encoding the actual stored scales/codes."
5. "Preserve storage shapes, dtypes, headers and exact bytes, then reload and check."
6. "Keep the frozen successful implementation intact; version later changes."
7. "The measured quality result covers one matched question and seed."

## Saved implementation

- [`recipe.json`](recipe.json): machine-readable method, calibration, format,
  dependency identities and validation boundary.
- [`reference/`](reference/): byte-for-byte successful r2 source packet,
  including the original controller, K5 math, calibration builder, carried
  walker, Hessian bundle builder and source-lock contract.
- [`folding.py`](folding.py): reusable Qwen3.5 AWQ functions; identical numerical
  bodies with two injected dependencies: packed-kernel construction and Stop.
- [`storage.py`](storage.py): path-independent, size-preserving matrix writer;
  the corrected reshape expressions are identical to r2. It additionally uses
  exclusive file creation and checks the exact raw header.
- [`evidence/`](evidence/): exact calibration text/token IDs, full sampled answer,
  final read review, run cards, algebra checks and independent save verification.
- [`provenance.json`](provenance.json): SHA256/size inventory and original archive
  identities. [`verify.py`](verify.py) checks these without importing Torch.

The exported `statistics`, `qep_target`, `refine`, `energy`, `decode` and
`algebra_check` functions come **directly from the frozen K5 math file**. No new
quantizer is substituted. The current repository codec must match its measured
hash before the public numerical entry points are imported.

The original controller completed the model experiment. The newly extracted
folding and storage interfaces have **static/source-equivalence checks only**;
they have not themselves completed another numerical or model run.

## Exact selected recipe

- Model: text-only `Qwen/Qwen3.5-9B`, revision
  `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; 32 blocks, 248 body matrices.
- Format: `tk_k5_g128_ISk10log`. Five trits give 243 signed levels,
  minus 121 through plus 121; group size 128. Integer-k10-log scales occupy
  int16 storage. Ten trits is not a ten-bit scale container.
- Calibration: 16 complete `openai/gsm8k` **training** gold worked solutions,
  3,102 tokens, dataset revision
  `740312add88f781978c0658806c59bc2815b9866`, selection seed 20261027.
  Reasoning is the assistant's `reasoning_content`, followed by its gold answer
  and end token. These are dataset solutions, not generated teacher traces.
- AWQ: 20 activation-mean exponent candidates, ratios 0 through 0.95 in steps
  of 0.05; identity included. Sum of shared consumers' linear-output squared
  errors, measured through the actual K5 kernel; no clipping search.
- QEP: strength 0.5; damping 0.01 after replacing dead Hessian diagonal entries
  by one for the solve. Use the full input Gram and cross-statistic.
- DAPTQ: at most two gated rounds, each a joint group-scale solve followed by
  up to ten code-polish steps. Stop when a round fails to improve by more than
  0.000001 relatively. Scales are encoded before acceptance. Each polish step
  proposes one best legal coordinate change per output row, potentially jumping
  multiple levels; the gradient is recomputed every eight steps.
- Initialization: unchanged `pack_k5_hiscale`, including its stored-scale-first
  rounding. **No independent GPTQ pass**, reconstruction or training was added.
- Arithmetic: source weights first share the baseline FP16 cast; fitting uses
  FP32, energy accumulation FP64, native execution FP16 operands. TF32 disabled.

### Recorded flow

If starting a block:
  Walk its unmodified FP reference on complete carried examples and capture inputs.
If reference inputs are captured:
  Search and fold the three eligible AWQ groups into a separate working block.
If the folded block passes its numerical check:
  Process input projections, attention output, MLP gate/up, then MLP down.
If entering a projection stage:
  Re-walk the current partially quantized block on the propagated quantized stream.
If inputs are aligned in the folded basis:
  Form QEP statistics, build the target and run the stored-grid DAPTQ refinement.
If a matrix passes saved-shape/value/byte checks:
  Install its packed module and use it in the following stage.
If a block finishes:
  Save both output streams, layout, exact inputs/statistics and its report.
If all blocks finish:
  Verify the complete file inventory, reload the candidate and evaluate it.

The FP stream always comes from unmodified reference blocks. The quantized
stream includes all previous accepted blocks. Stage order and recapturing the
current quantized inputs are essential; a single stale activation dump is not
this method. Checkpoint/Drive/HF ownership stays with the caller.

## Mathematics and folding details

Use row-token input matrices. For an already AWQ-folded weight `W`, let `Xf`
be the reference input divided channelwise by the saved input scaling, and
`Xq` the actual current quantized-path input in the same basis. With `T` real
tokens:

- `H = Xq.T Xq / T`
- `C = Xf.T Xq / T`
- `Wtarget = W + 0.5 W (C - H) inverse(Hdamped)`

The implementation uses a Cholesky solve, not a stored inverse. If `Xf` and
`Xq` are identical, the correction is exactly zero in the recorded algebra check.
Do not reuse an unscaled reference input or Hessian after folding: both `H` and
`C` transform by inverse channel scales on their two axes.

DAPTQ minimizes `sum(((Wtarget - Q) Xq.T)^2) / T`, where `Q` is decoded from
the actual integer payload. For fixed codes, each row's group factors are
`B[t,g] = sum(Xq[t,g,:] * codes[row,g,:] / 121)`. The joint normal system is
`B.T B / T` with right side `B.T (Xq Wtarget[row]) / T`; off-diagonal group
interactions remain. A diagonal regularizer of 0.00000001 is added. A failed
FP32 solve is retried in FP64; relative residual must be at most 0.001. Positive
scale proposals are encoded with unchanged `pack_hi(k=10, log=True)` before
the full-token objective decides whether to keep them.

For fixed stored scales, code polish uses the full Hessian quadratic to choose
one noninteracting coordinate per output row. The final proposed codes are
remeasured by the same stored-grid objective. This gate protects that local
calibration objective, not general model accuracy.

**Qwen3.5's norm multiplier is `gamma = 1 + weight`.** For a channel scale `s`,
store `((1 + old_weight) / s - 1)` in FP16, then derive the effective scale from
the **stored** old/new gamma. Multiplying only the raw norm weight is wrong.
The fold groups are:

- Input norm into all attention input projections, including DeltaNet's
  `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a` where present.
- Post-attention norm into MLP gate and up projections.
- MLP up output rows divided by `s`, with down input columns multiplied by `s`.
  The gate branch stays unchanged for this last fold.

There is no extra inference scale tensor. Exactly 64 existing normalization
vectors changed; the other 115 retained tensors, including head/embedding,
were byte-identical to the baseline. This architecture-specific fold is not
automatically applicable to another norm definition or model family.

## Reuse from TKintegers

From the repository root, this lightweight verification requires only Python:

```bash
python3 -B quantization/awq_qep_daptq_k5_v1/verify.py
```

Metadata can also be read without importing Torch:

```python
from quantization.awq_qep_daptq_k5_v1 import load_recipe
recipe = load_recipe()
```

Numerical integration runs on the assigned remote CUDA machine. The core needs
PyTorch, safetensors and the repository codec. The observed full-run stack was
Torch 2.11.0+cu130, Transformers 5.20.2, safetensors 0.8.0 and datasets 4.8.5.
Whole-model execution additionally needs the pinned K5 kernels, the carried
walker, actual accelerated DeltaNet/causal convolution, and the caller's
existing save/backup and cancellation facilities. Root `requirements.txt` alone
does not recreate that full environment. Exact parent source hashes, including
the runtime setup script, are in the frozen contract.

The following is an integration fragment, **not a model-launch command**.
`block`, `weights` and `original_inputs` are the working Qwen3.5 block, detached
FP32 CUDA weight dictionary and complete reference inputs from the recorded
flow. `packed_linear` constructs the existing K5 runtime from a signed payload;
`check_stop` checks the caller's Stop control.

```python
import torch
from quantization.awq_qep_daptq_k5_v1 import (
    load_recipe, fold_awq, statistics, qep_target, refine, save_matrix_like,
)

recipe = load_recipe()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

with torch.no_grad():
    scaling, awq_report = fold_awq(
        block, weights, original_inputs,
        packed_linear=packed_linear, check_stop=check_stop,
    )

    # At EACH sequential stage, recapture Xq from the current block/quantized
    # stream using the existing carried walker, excluding all padding.
    # Do the folded-block check before the first quantized stage.
    xf = original_inputs[name].cuda().float() / scaling[name]
    xq = current_inputs[name].cuda().float()
    h, cross, chol = statistics(xf, xq, damp=recipe['damping'])
    target, movement = qep_target(
        weights[name], h, cross, chol, alpha=recipe['qep_alpha'],
    )
    payload, report = refine(
        target, h, xq, rounds=recipe['daptq_rounds'],
        steps=recipe['code_polish_steps_per_round'],
    )
    receipt = save_matrix_like(payload, baseline_shard, new_shard)
```

Keep the folding inputs under their original per-linear names. Initialize the
weight dictionary exactly as in the controller, before any folds. The block
and dictionary are intentionally mutated; work on a separate block, not the
reference. The callback must use the shipping packed kernel rather than a dense
matrix approximation: AWQ selects using its real arithmetic. The original
factory is `reference/quality9_awq_qep_daptq.py::packed`.

`save_matrix_like` takes an existing **quality9 bare-key matrix shard** as the
shape/header template, and a fresh destination whose parent directory exists.
The desktop converter uses tensor-prefixed keys and a different manifest;
its shards cannot be passed to this adapter unchanged. Caller integration must
map the payload into the chosen manifest and maintain its own exact size check.
This version supplies the building blocks and measured reference, not a new
generic full-model CLI or an automatically wired desktop conversion option.

### Storage contract and previous failure

On disk, `codes` are uint8 values equal to signed K5 codes plus 121. Scales are
flat int16 codes and `meta` is the integer anchor record. Signed 2-D codes and
2-D scales used during fitting must be reshaped to the **original saved shapes**.
The first attempt wrote scales as a 2-D table, enlarging one padded header by
eight bytes. Two earlier files had changed shapes despite equal total sizes.
R2 restored every original shape and verified reloaded values, dtypes and bytes.

Both baseline and candidate total **11,097,277,432 weight-file bytes**, including
retained tensors and safetensors headers. This is not the size of all logs,
Hessians or metadata. The candidate layout references unchanged retained files
in its baseline sibling directory: copy that dependency too when restoring.
The payload is custom TK, not a drop-in standard Hugging Face or GGUF model.

## Measured result and evidence boundary

The saved/reloaded candidate passed `HumanEval/127` with the original prompt,
seed 196084012 and sampling: temperature 1, top-p 0.95, top-k 20, presence
penalty 1.5, thinking enabled. The uncapped answer ended normally after 1,548
tokens in 152.169 seconds. Full AI-assisted reading and the isolated code test
agreed. The original K5 answer failed. No human-independent grade is implied.

Independent saved-file checks covered all 248 matrices and 179 retained tensors.
Actual answer runtime counts: 383,656 K5 decode calls, 248 prefill calls, zero
dense fallback; carried cache. The summed conversion-block intervals were
897.68 seconds, excluding setup, answer and final backup wait.

This is **one matched sampled answer**, not a new full-suite score, a general
accuracy-retention percentage, a speedup or an ablation of the three components.
The earlier baseline manuscript's paired results are a separate experiment.
The raw save-verification record predates final reading and says reading was
pending; the later `answer-review.json` is the completed review of that answer.

## Original recovery and provenance

The frozen controller deliberately retains its historical paths and source
locks. It is a replay reference; do not launch it against a completed directory.
Its `--phase pilot` converts block zero, and `--phase full` resumes the saved
pilot, completes conversion and evaluates only the original interval question.
Future execution needs a fresh caller/configuration and the existing backup
requirements, not edits to the frozen reference.

- Successful source archive SHA256:
  `e2a5f93105da7111477358c537da1d64945802e7f2c73342ef2e6c49ad667257`.
- Original local archive and parent packet: `.cache/tk-quality9-1027` under the
  owner's home; `combined-code-r2.tar.gz` and `code-v6.tar.gz`. The source-lock
  contract lists every parent dependency; it is not an installer.
- Drive recovery: `MyDrive/taardis/tk-quality9-1027`, successful directories
  `awq-qep-daptq-code-r2`, `awq-qep-daptq-127-r2-artifacts`,
  `awq-qep-daptq-127-r2-evidence` and `awq-qep-daptq-127-r2`, plus baseline
  `artifacts` and pinned source inputs.
- Private evidence repository: `CodeMasterCody3D/tkintegers-9b-awq-qep-daptq-1028`,
  under `attempt-r2/`. Exact data, source, Hessians and review are backed up there;
  do not infer that the candidate weights were uploaded to HF. Weight recovery
  uses the confirmed Drive artifacts and their baseline dependencies.
- Final review commit: `8f9b4a48edb31813acb442caf2a54490afb67d95`.
  The independent receipt hashes the complete 13-file review package.
- Failed r1 code and partial artifacts remain preserved separately. No earlier
  kernel, answer, baseline result or manuscript is replaced by this recipe.

AWQ search reference: MIT Han Lab's
[`auto_scale.py` at d6e797a](https://github.com/mit-han-lab/llm-awq/blob/d6e797a42b9ef7778de8ee2352116e0f48a78d61/awq/quantize/auto_scale.py).
QEP and DAPTQ equations came from the existing Forge `ternary_gptq.py` functions
`_qep_iter`, `joint_scale_solve_` and `_flip_polish_incremental_`; the frozen K5
adapter generalizes the alphabet and gates on integer-encoded scales. DAPTQ is
Dixon Alternation Pre-Trained Transformer Quantization. The older source flag
spelling DPTQ does not name a separate method.
