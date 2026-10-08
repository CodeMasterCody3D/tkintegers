# Evidence and citation record for the Tk-Integers technical paper

Date: October 8, 2026.
Paper: [Tk-Integers: Explicit Base-3 Containers and Model-Aware Quantization Recipes](TKINTEGERS_AUTOGRID_PAPER.md).
Audited application: `4e3d91628302586c7a94d994641b391d656826c4`.

## Claim scope

This is an implementation and analytical paper. It contains no new model
quality, inference speed or training-memory benchmark. Calculated bytes are
labelled as calculated; successful UI/metadata checks are not model results.
The completed text has no missing experiment tables disguised as finished work.
Questions requiring model experiments are stated explicitly as limitations.

## Retained evidence

- [Storage calculations](evidence/storage-accounting.json): exact scalar examples
  from [verify_paper_accounting.py](verify_paper_accounting.py); no torch/model
  execution. Atlas byte counts also use the vendored `formats.storage` function.
- [Recipe tests](evidence/recipe-tests.log): fresh run, 8 passed, zero failures.
  Source: `tests/test_model_atlas.py` at the audited application commit.
- [Atlas architecture tests](evidence/atlas-architecture-tests.log): fresh run,
  18 passed, zero failures. Source: `vendor/model_atlas/tests/architecture.test.js`
  at the audited application commit.
- [Actual HTML integration log](evidence/atlas-html-check.log): retained from the
  preceding application-development session, not rerun as a numerical experiment.
  It reports ten models, 23 SVG nodes, the original layer dialog, review-button
  adjacency, k3 choices restored for 48 layers and the original HTML importer.
- [Recorded UI harness](evidence/atlas_html_check.recorded.py): exact historical
  test-script bytes retained with that log. It contains machine-specific paths
  and mocked save-dialog output, so it is an evidence artifact, not a portable
  installation command. It supplies no synthetic conversion result.
- [Embedded UI screenshot](figures/model-atlas-embedded.png): captured during
  the October 8 integration session; demonstrates the original flow diagram
  and inspector in the Tk tab. Not a benchmark or an independent usability study.

The metadata tests do not load the weights of the Qwen presets. The scalar
accounting script does not pack numeric tensors. The UI harness does not run
model conversion. These boundaries are retained in the manuscript.

## Source-to-claim audit

- `base3_pack.py:pack_ktrit` starts one stream per input row; the global converter
  explicitly reshapes its codes to one row first. Those are distinct layouts.
- `tkintegers_convert.py:quantize_tensor` pads the flattened source to complete
  groups and fits weight codes before its selected scale encoding.
- `tk_codec.py:pack_tensor` returns native-width integer arrays. It does not
  call `pack_ktrit`; its k5 matrix codes cost one byte each, and k10 scale codes
  cost two bytes each plus four int64 metadata entries.
- `tk_s12.py:pack12` uses `3*ceil(M/2)` bytes; the vendored Atlas helper uses
  `ceil(12*M/8)`. Swapping their unpackers is not valid.
- `vendor/model_atlas/quant/codec.py:pack` restarts the code stream per S12 group.
  Its non-S12 path restarts per matrix row. `fit_scales` chooses representable
  scales before final code assignment.
- `model_atlas.py` checks recipe resolution, known text wrappers, tied choices,
  source dimensions and unsupported selections. It does not implement every
  target available in Atlas's planning interface.
- `tkint_latent.py` retains dense `param`/`resid` tensors and dense integer codes
  while operating. Its state_dict packs only at serialization time. Its bpw
  helper counts code planes, not scales or resident memory.
- The earlier global decoder omitted division by `L_k`; the studied commit
  includes the fix. No pre-fix k>1 decoder result is cited as fidelity evidence.

## Primary-source verification

All sources below were accessed during preparation on October 8, 2026.
Bibliographic metadata for the three arXiv entries was retrieved via DOI content
negotiation (`Accept: application/x-bibtex`); the error-feedback entry uses the
publisher-provided BibTeX. Cleaned entries are in [references.bib](references.bib).

1. **BitNet b1.58**
   - Primary: https://arxiv.org/abs/2402.17764
   - Cross-check: https://github.com/microsoft/BitNet (official README links the paper).
   - DOI metadata: https://doi.org/10.48550/arXiv.2402.17764
   - Supported use: ternary-weight language-model training is prior work.
     Its reported accuracy/speed is not imported as a Tk-Integer result.
2. **GPTQ**
   - Primary: https://arxiv.org/abs/2210.17323
   - Cross-check: https://github.com/IST-DASLab/gptq (official paper link and citation).
   - DOI metadata: https://doi.org/10.48550/arXiv.2210.17323
   - Supported use: approximate second-order post-training weight quantization.
     The repository title says “Compression”; the current arXiv title says
     “Quantization.” The manuscript identifies the same arXiv ID explicitly.
3. **QLoRA**
   - Primary: https://arxiv.org/abs/2305.14314
   - Cross-check: https://github.com/artidoro/qlora (official README and citation).
   - DOI metadata: https://doi.org/10.48550/arXiv.2305.14314
   - Supported use: frozen-base low-rank adaptation and quantized quantization
     constants are prior work. No equivalence to the current converter is claimed.
4. **Error feedback**
   - Primary/publisher: https://proceedings.mlr.press/v97/karimireddy19a.html
   - Cross-check: https://arxiv.org/abs/1901.09847
   - Publisher supplies author list, ICML venue, volume 97 and pages 3252–3261.
   - Supported use: carrying compression error into later updates is prior work.
     Its convergence theorem is not asserted for TkIntLatent.
5. **Safetensors**
   - Documentation: https://huggingface.co/docs/safetensors/index
   - Source specification: https://github.com/huggingface/safetensors#format
   - Supported use: eight-byte header length, JSON header, tensor data offsets
     and payload structure. Custom codec semantics remain the application's job.

## Historical material reviewed but excluded from empirical results

Read bounded sections of the Forge contribution audit, latent-paper draft and
Tk-Integer memory note. Those records describe separate training, teacher and
lean-executor work. Their benchmark receipts were already marked missing or
unverified by that audit. This paper therefore describes the included helper's
update rule and storage contract without borrowing historical PPL, drift or
VRAM numbers as standalone-converter results.

Shared-memory MCP retrieval failed with `MCP error -32001: Request timed out`.
The preferred web-reader returned a quota error and one timeout. Primary-source
verification above used successful direct fetches and official repository
readmes instead. Anchor start reported an existing watcher; doctor reported
the compute shield enabled and zero tracked in-flight runs. No worker, model
job, training or numerical quantization run was launched for this paper.

## Revisions made to the old draft

- Replaced its single universal payload formula with separate planner, global,
  row-packed, group-packed and native-integer layouts.
- Distinguished k10-log's trit depth from its int16 storage width.
- Included anchor bytes and both odd-count S12 conventions.
- Corrected unqualified g12/g8 S12 rates for Atlas group-local byte padding.
- Separated integer payloads from floating reconstruction/execution.
- Specified dense-reference residual-state behavior and full serialized overhead.
- Added the literal HTML integration and saved-file review handoff.
- Added verified related work, bounded verification evidence and a reproducible
  model-evaluation protocol without inventing results.
