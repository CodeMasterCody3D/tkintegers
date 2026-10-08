# UltraMega + factorization release

Open **qwen-atlas-ultramega.html** and select **Quantize & factorize**. This release adds uniform TKinteger intermediate rates, per-layer binary/ternary factor ranks, head/embedding accounting, document context carry, and a GitHub Actions quantization workflow. Import old plans with the existing Import saved plan button.

See [quant/README.md](quant/README.md) for exact formats, the pipeline, GPU requirements, setup, supported targets and test results. The source ZIP is a repository-root overlay: its `.github/workflows/atlas-quant.yml` belongs at the GitHub repository root. The GitHub workflow is installed separately at the repository root.

Third-party and uploaded sources retain their own licensing/provenance; the app's MIT license does not assign a license to supplied files.

---

# Qwen Atlas

An interactive, config-driven model anatomy explorer. Open a decoder block,
follow its data-flow arrows, and hover or focus each component for an explanation.
Click a component to pin its module path, dimensions, precision notes, and sources.

**Default: Qwen3.8-27B.** Ten official Qwen presets are bundled with pinned config
revisions. No model weights, GPU, API key, build process, or JavaScript dependency
installation is required to use the app. Imported config files stay in your browser.

## Run locally

```bash
cd qwen-atlas
python3 -m http.server 8080
```

Open <http://localhost:8080>. Or run `npm start` with Python 3 installed.
The app uses ES modules, so serve it over HTTP instead of opening index.html as a
file. Any static web host can serve this folder unchanged. There are no external
font, analytics, or model-download requests when using the bundled presets.

## Explore

- Click numbered decoder blocks. Teal means DeltaNet; purple means attention.
- Hover or keyboard-focus a node for a short explanation; click or press Enter
  for pinned details, weight shapes, module paths, and source links.
- Follow the residual skips, parallel gates, recurrent state, KV cache, and MLP.
- Trace flow follows an explanatory order, not measured execution timing.
- Switch among dense, MoE, hybrid, and VL text-decoder presets.
- Import a config.json or paste JSON. Unknown architectures show their config
  overview and a clearly unsupported internal flow, rather than a guessed graph.
- Use the memory lab to manipulate a 2×2 delta-rule state and compare growing
  attention cache with fixed recurrent memory.
- Search the field guide and inspect the source trail.

## Coverage and boundaries

Verified text-decoder templates: Qwen2/2.5, Qwen2 MoE, Qwen3 dense/MoE,
Qwen3-Next, Qwen3.5/3.8 dense/MoE, Qwen2/2.5-VL and Qwen3-VL text decoders.
The bundled presets exercise Qwen2.5, Qwen3, Next, 3.5, 3.8 and VL families.
Qwen2-MoE and unbundled variants use the related template and should be checked
against their implementation before precision planning.

The 27B configs use 64 text decoder blocks, 48 DeltaNet and 16 full attention,
with separate embedding and output-head weights. The 0.8B preset ties those
weights. The schedule is read from layer_types, or derived using the documented
hybrid full_attention_interval when absent. It is not hardcoded to 3:1 globally.

Vision is a config-driven frontend overview. Detailed vision-block, audio,
talker, embedding-model output heads, and auxiliary MTP execution graphs are
not implemented. Original Qwen-1 and future unknown model types are not given
invented internals. Custom remote code may differ even with familiar config keys.
Thus “import any config” does not mean every Qwen architecture is fully verified.

## Precision evidence

Amber means a **conservative precaution**, not a universal sensitivity ranking.
Quantization sensitivity depends on recipe, calibration, runtime, and evaluation.
The source-linked NVFP4 study reports relatively robust a/b gate projections;
that does not establish ternary or low-rank robustness. Weight precision,
activation precision, accumulator precision, and runtime state precision are
separate concepts. No model quality or PPL is measured by this educational app.

The recurrence visualization follows the reference implementation’s ordering:
first decay S, then predict from decayed S, correct toward v, and read using q.
The lab is a deterministic 2D example, not a fitted approximation to a Qwen layer.

Memory estimates are tensor payloads at batch size one, with explicit dtype
assumptions. They omit weights, padding, allocator overhead and workspaces;
sliding-window attention estimates are upper bounds.

## Research and reproducibility

See [RESEARCH.md](RESEARCH.md), [data/manifest.json](data/manifest.json), and
[knowledge.js](knowledge.js). Config files are preserved as retrieved from
Qwen’s official Hugging Face repositories, with exact revision URLs. Operator
sources are official Transformers implementation files; source links using main
may evolve. Review date: 2026-10-02.

## Tests and extension

```bash
node --test tests/*.test.js
```

Tests cover pinned 27B dimensions, all bundled graph edges and help entries,
family differences, model tying, cache scaling, invalid configs, and the delta
recurrence. Add config fixtures in data/, add their entries to data/presets.js,
and extend architecture.js for a genuinely new family. Update the research
notes and tests with the implementation source used.

Files: architecture.js defines model adapters and graph structure; knowledge.js
contains sourced explanations; app.js handles interaction; style.css defines
responsive layout; index.html is the static entry point. UI supports pointer,
keyboard, touch (click to pin), reduced-motion preferences, and mobile layouts.

This is an independent educational project, not an official Qwen product.
App code is MIT licensed. Official config snapshots retain their upstream provenance.

## Single-file edition

Open **standalone.html** directly in a modern browser. It embeds the app, styles,
research notes, and all ten config presets and works offline. Regenerate it after
source changes with `python3 build-standalone.py`. Source-module development still
uses the local HTTP server. On narrow screens the graph scrolls horizontally to
keep node text readable; use Fit width or the zoom controls to explore.

## Automatic protection list

Open **Protection list** for a per-block, searchable list of conservative exclusions from low-bit weight quantization. Norms, DeltaNet convolutions/decay parameters, expert routers, and optional a/b gates and vocabulary matrices are included. Fused Next a/b projections are deduplicated; tied vocabulary aliases are both shown, with storage counted once conceptually. Block buttons jump to the diagram. Download JSON exports the entire selected policy, not just search matches.

These are config-derived recommendations, not measured sensitivity or a ready-to-run quantizer configuration. Resolve wrapper prefixes against the actual model. Protect module parameters at their original precision; preserve reference runtime defaults separately. Vision/audio/MTP internals remain outside the enumerated scope and should remain unchanged pending review. Unknown architectures produce no fabricated list. “Precision precaution” means a reason to consider retaining precision, not proof that quantization fails.

## Layer-level protection and precision

Use **Edit DeltaNet layers** or **Edit attention layers**. Click a box directly in the flow chart. A dialog shows a plain-language explanation, a custom protection checkbox, and a precision selector. Protected boxes show their status on the graph. A selected output projection protects that projection, not the whole decoder block. By default edits repeat only for the same layer path in blocks with the same mixer type. Turn off the repeat switch for single-block edits.

Automatic and custom protection remain separate. Automatic protection still applies when a custom checkbox is unchecked. Protected layers retain original precision. Clear custom choices for the current type with the dedicated button. Existing saved whole-block selections are migrated to explicit component entries so individual layers can be unchecked. Tied vocabulary precision is synchronized in the Precision list.

Settings persist per configuration in browser localStorage when available; export separate or combined JSON plans for portability. These are config-derived module groups, not an exact checkpoint tensor inventory. Fused projections share one setting and expert groups apply recursively. Vision/audio/MTP internals and runtime state precision are outside scope. No weights are modified by the app.

### Every flow box is editable

All flow boxes open editable controls, including normalization, gates, arithmetic, residual streams, KV cache, and recurrent state. Weight-bearing components receive weight targets; other boxes receive graph-level runtime targets under `@flow` identifiers (not checkpoint names). Runtime low-bit targets need a supporting implementation and validation; this app only saves the plan. Composite/unexpanded subsystem targets do not enumerate its internal tensors.

Precision selectors are never locked. Choosing a target removes custom protection for that selection and opts out of its automatic protection; the recommendation checkbox can restore automatic protection. Exports include opt-outs and distinguish weight/runtime entry kinds. Same-type propagation, persistence, and tied weight handling apply.

### TKinteger k1–k16

Every precision selector includes the user’s TKinteger base-3 format. k means trits per integer (1 through 16); each value has 3^k levels and a balanced integer range ±(3^k−1)/2. Five trits fit in one byte. For a continuous trit stream, N values need ceil(N*k/5) payload bytes; asymptotic code cost is 1.6*k bits/value, before scales, anchors, headers, alignment, and final padding. k1 is 1.6, k5 is 8.0, and k16 is 25.6 code bits/value. This is not k bits/value.

JSON plan schema v3 includes structured TKinteger format metadata for each selected target. This addition defines selectable format targets, not an encoder, scale recipe, reconstruction procedure, or integer arithmetic kernel. Runtime selections remain proposals and need implementation support; packed storage alone does not fix compute precision.

### Live size estimate

The live estimator counts represented text weight tensor dimensions, deduplicates fused modules and tied vocabulary weights, and applies each effective precision. TKinteger payloads use ceil(N*k/5) bytes per tensor. Optional scale storage defaults to 2 bytes per group of 128 values; change this to match the real codec. Expert groups count all experts, not only active experts. Protection is included at the selected original precision assumption.

This is not an exact checkpoint file prediction: unspecified biases, vision/audio/MTP internals, headers, and alignment must be supplied using the extras field. Those omissions are shown next to the estimate. Runtime operation/state targets do not contribute to weight file size. GB means 10^9 bytes and GiB means 2^30 bytes. The user's 5.9 GB k1 result is a comparison reference, not a hardcoded estimator output. Exports include size assumptions and estimated components.

### Import a saved plan

Use **Import saved plan** beside the model selector to upload an exported JSON. Combined, precision-only, custom-only, and legacy automatic-protection exports (schemas 1–4) are supported. The model configuration is restored from the file when present; older files resolve their named preset or require that same model to be selected. Validation checks paths, precisions, options, and size assumptions before changing state. Invalid files leave settings unchanged. Partial exports replace the sections they contain while preserving the other settings saved for that model. Imports use explicit paths, so single-block exceptions remain exceptions even with same-type editing enabled.

New combined exports include the raw editor choices, automatic protection settings and overrides, runtime targets, same-type editing toggle, and size assumptions. Older files restore their exported effective settings; unexported historical choices cannot be recovered. Imports recalculate size rather than trusting the file’s estimate.

### 20/44 k1 option

**TKinteger 20/44 k1 · 2.2 bits/value** is a separate user-defined format target: 20 ternary values per 44-bit block, with blocks packed consecutively. It is available in flow-box settings, the precision list, and the bulk size control. JSON plans preserve its identity and format metadata.

Storage is ceil(ceil(N/20)*44/8) bytes per tensor, plus separately configured scale storage and extras. The 2.2 bits/value rate is before padding and metadata; independently rounding every block to six bytes would instead cost 2.4 bits/value and is not the selected layout. This UI describes and estimates the format; it does not implement its encoder.

### Explicit head and embedding accounting
The live model calculator includes a visible body / embedding / head breakdown, including scale bytes. Head and embedding precision can be changed directly beside the size estimate. Tied models synchronize both selectors and count shared storage once. Existing plan imports and exports preserve these choices. The standalone release is qwen-atlas-head-embeddings.html.

### Uploaded TKinteger codec accounting

The k1–k16 options now follow `base3_pack.py`: for a matrix with O rows and I columns, payload is `O * ceil(I*k/5)` bytes. Each row starts a fresh five-trits-per-byte stream. Automatic scales cost `O*(I/g)*2` bytes for k1–k5 or `O*(I/g)*4` for k6–k16, where input width I must be divisible by group size g. The calculator flags incompatible shapes and estimates partial groups only as a hypothetical plan. Non-matrix targets also need an adapter. Expert gate/up and down projections use their respective orientations.

Select scale bytes **−1** for automatic codec widths. Existing numeric overrides and imported plans retain their chosen widths. Headers and other tensors still require the extras field. Embeddings and output head use the same matrix formulas; tied weights are counted once. The uploaded linear-only resident implementation does not automatically quantize embeddings. Packed size does not imply equivalent inference RAM: the supplied generic packed linear dequantizes for computation.

`tk_codec.py` is a separate format: its 2-D tensors use one int8 k5 code per weight, int16 k10 scale codes, and 32 bytes of anchor metadata per tensor. Small floating tensors use int32 k15 codes plus metadata. These are not the same storage rules as tightly packed any-k trits, and Atlas k1–k16 options do not represent that container. Legacy 20/44 remains an unverified planning format.

Validation: estimator regression tests cover row padding, k-dependent scale widths, manual overrides, incompatible groups, head/embedding choices, and tied storage. The current release also runs codec and tiny-model integration tests with CPU PyTorch; see quant/README.md. This update does not run model quantization or measure perplexity.

### Start directly from Atlas

In Quantize & factorize, enter a fine-grained GitHub token with **Contents: read/write** and **Actions: read/write** for the chosen repository, then click **Save plan & start run**. The token stays only in memory, is sent directly to api.github.com, and is cleared after the attempt. Do not paste tokens into chat. This standalone app cannot reuse ChatGPT's GitHub connection.

The app checks that the workflow exists and is active, commits a new uniquely named JSON snapshot, and dispatches the workflow with that exact commit. It reports submission separately from execution: a GPU job can remain queued until the required self-hosted runner is online. If a dispatch response is lost, check Actions before retrying to avoid duplicate runs. **Open GitHub Actions** remains a separate navigation button.


## GitHub Pages and GPU launch

The repository-root `.github/workflows/atlas-pages.yml` builds the standalone
interface and publishes only that file and its license. Plans, model files,
corpus data, and runner code are not included in the public website artifact.

One-time setup: repository Settings → Pages → Build and deployment → Source:
GitHub Actions. Then run **Publish Qwen Atlas** from Actions. Later Atlas changes
on main redeploy automatically. Private-repository Pages requires an eligible
GitHub plan; do not make the experiment repository public just to enable Pages.

In Atlas, click **Run quantization** at the top. Enter a fine-grained token
restricted to the target repository with Contents and Actions read/write, then
click **Save plan & start run**. The browser commits a unique plan snapshot and
dispatches `atlas-quant.yml`. Tokens are never stored. A self-hosted Linux x64
GPU runner with the `gpu` label must be online; Pages does not provide a GPU.

## Per-layer group scales and wider K1 budgets

Click a weight layer in the flow chart, then use **Group scales for this layer type**. Choose a group size (1–65536) and apply it to this block, matching block types, or that layer across all block types. Each tensor fits independent scales. Tied head/embedding aliases keep one shared setting. Protection stays in effect. Factorized layers retain their separate row/column scale scheme.

Schema 5 exports `group_size` on weight entries and retains editor overrides in `editor_state.group_sizes`. The runner consumes per-entry groups; older plans retain their global default. Fixed-group format names must agree with the row group size.

The format calculator now offers S12, FP16 and FP32 scales, including groups smaller than five. K1 always has three codes per scale group. With row-packed codes and FP16 scales, nominal cost is `1.6 + 16/G` bits per weight: G2 is 9.6 and G1 is 17.6. FP32 G1 is 33.6. These are storage budgets, not 9/17/33-bit ternary codes or guarantees of quality. There is no universal upper storage bound for K1.

S12 packs each group independently: full-group cost is `(8*ceil(G*k/5)+12)/G`; G2 K1 is 10 BPW and G1 is 20 BPW. The calculator counts partial-group and final-byte padding. Standard FP16/FP32 scale formats require matrix columns divisible by G in the executable runner. Non-matrix and runtime selections still require adapters.
