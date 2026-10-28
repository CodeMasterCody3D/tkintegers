# Tk-Integers

**TK stands for Ternary Kluster** — the name behind **TKintegers**.

A desktop safetensors converter with selectable weight/scale formats and the
original Model Atlas HTML interface embedded in a tab. Built on Autogrid.

**Read the paper:** [Tk-Integers: Explicit Base-3 Containers and Model-Aware Quantization Recipes](docs/TKINTEGERS_AUTOGRID_PAPER.md).
The completed technical report includes exact packing formulas, scale formats,
Model Atlas integration, related work and [recorded verification evidence](docs/PAPER_EVIDENCE.md).

**Saved quantization method:** [AWQ + QEP + DAPTQ K5, version 1](quantization/awq_qep_daptq_k5_v1/README.md)
includes the exact tested recipe, frozen implementation and reusable integration
functions. It passed one matched Qwen3.5-9B question at unchanged saved weight
size; broader accuracy is unmeasured. See the recipe for its dependencies and
source-only package verification.

## Quick install and launch

For the opt-in GPU inference kernels, see [Native k5 kernels](benchmarks/KERNELS.md).
The faster CUDA prefill implementation, its T4/L4 adapters, and the preserved
Triton decoder are included. These are low-level runtime modules, not an
automatic converter or desktop-runtime replacement.

**Supported full desktop setup:** Ubuntu 24.04+, Debian 12+, or KDE neon with an
X11 desktop or XWayland. Use the distribution's **Python 3.10+**, **Node.js 18+**,
Tkinter, and Qt5 WebEngine. The embedded Atlas tab currently requires X11/XWayland.

Clone this repository first, then run the following **in a terminal inside the
`tkintegers` folder**. This repository is public; reading or cloning it does not
require access to the private upstream Atlas repository. The Atlas files needed
by this build are included under `vendor/model_atlas/`.

```bash
git clone https://github.com/CodeMasterCody3D/tkintegers.git
cd tkintegers
```

**One copy-and-paste setup command:**

```bash
sudo apt-get update && \
sudo apt-get install -y python3-venv python3-tk python3-pyqt5.qtwebengine nodejs && \
/usr/bin/python3 -m venv --system-site-packages .venv && \
.venv/bin/python -m pip install -r requirements.txt && \
.venv/bin/python tkintegers_ui.py
```

This installs the desktop dependencies, creates `.venv`, installs the model-file
dependencies from `requirements.txt`, and opens the app. It does not download or
convert a model. Only the `apt-get` commands use `sudo`; the virtual environment
and Python packages belong to your user.

### Why this venv uses `--system-site-packages`

Tkinter and Qt WebEngine come from the system packages above. Creating the venv
with **`/usr/bin/python3`** and **`--system-site-packages`** makes those bindings
available to the Python that runs the app. The embedded Atlas renderer uses that
same interpreter. An isolated venv or a different Python installation may not
see the system's Qt bindings.

### Open it again later

From the repository folder, launch with the venv interpreter directly:

```bash
.venv/bin/python tkintegers_ui.py
```

Or open directly to the **Model Atlas** tab:

```bash
.venv/bin/python tkintegers_ui.py --atlas
```

If you prefer to activate the venv in your terminal:

```bash
source .venv/bin/activate
python tkintegers_ui.py
# When finished with this terminal environment:
deactivate
```

### Check the installation

```bash
.venv/bin/python -c "import sys, tkinter, torch, safetensors; from PyQt5 import QtWebEngineWidgets, QtWebChannel; print('Ready:', sys.executable)"
node --version
```

The Python path should end in `.venv/bin/python`; Node should report version 18
or newer. Run the app in your desktop session, not a headless SSH session.

- **`No module named venv` / `ensurepip is not available`:** install
  `python3-venv`, then repeat the venv creation command.
- **Missing `tkinter` or `PyQt5.QtWebEngineWidgets`:** install the system packages
  listed above and check that `.venv/pyvenv.cfg` contains
  `include-system-site-packages = true`. If this is an existing isolated venv,
  rerun `/usr/bin/python3 -m venv --system-site-packages .venv` with its matching
  Python version, then reinstall `requirements.txt`.
- **The app still says model dependencies are missing:** install with
  `.venv/bin/python -m pip install -r requirements.txt`, close the app, and
  reopen it with `.venv/bin/python` rather than a different system Python.
- **Atlas says Node is missing or too old:** install Node.js 18+ and verify that
  `node --version` works in the same terminal used to start the app.

For updates after pulling new source, rerun
`.venv/bin/python -m pip install -r requirements.txt` before reopening the app.

---

## Autogrid background

**Inspect balanced-ternary reconstruction error before choosing a conversion.**

AUTOGRID is a desktop app + CLI that scans every tensor of a safetensors
checkpoint, measures what it would cost to put that tensor on a
**balanced-ternary integer grid** (`w ≈ s · Σ dᵢ·3⁻ⁱ`, digits `dᵢ ∈ {−1,0,+1}`),
and compares the measured relative reconstruction error against a heuristic
threshold derived from the source dtype. The legacy path uses no retraining or
calibration data, but its rounding is generally lossy. A low error does not
guarantee identical values, logits or model answers.

The original in-place Autogrid rewrite retains the original tensor shapes and
dtypes; it does not itself produce an integer-only runtime or the new packed
Tk-Integer container. Its backup/revert mechanism can restore the original
bytes when the required originals are retained. That is different from a
lossless quantization map.

---

## Interpretation of the dtype threshold

For normal values rounded to nearest, BF16's conventional relative rounding
bound is about **0.39%**. This is not a constant error attached to every stored
value; values already representable in BF16 can be exact, and exceptional or
subnormal values need separate treatment.

Autogrid compares a tensor-level relative L2 reconstruction error with such a
threshold. It is a screening heuristic, not an elementwise error guarantee or
a theorem of model equivalence. A small tensor-level error can still alter a
sensitive component or change a model decision. Adding trits generally gives
a finer representable grid, but `3⁻ᵏ` is not an exact arbitrary-weight relative
error or a complete checkpoint-size formula.

Historical Forge scores are not validation of this standalone converter. The
[paper](docs/TKINTEGERS_AUTOGRID_PAPER.md) separates analytical storage results,
interface/metadata checks and the numerical model studies still required.

---

## What it does

For each tensor, AUTOGRID measures the per-digit-budget error and classifies it:

- 🟢 **FREE @ k** — a float tensor whose measured k-digit error is **at or below
  the configured dtype-derived threshold**. `FREE` is the legacy UI label for
  passing that heuristic, not a claim of zero loss or unchanged model quality.
  AUTOGRID recommends the smallest `k` passing the selected threshold.
- 🔵 **TERNARY** — a tensor whose *values* are already on the `{−1,0,+1}×scale`
  grid (e.g. a placed/quantized weight). AUTOGRID analyzes its **group scales**
  for digit-stack conversion instead.
- 🔴 **STEER** — no scanned k passes the threshold. A different placement or
  correction pipeline may be appropriate; this label alone is not a model-level
  verdict. The default automatic selection excludes these entries.

Then **CONVERT** applies the recommended `k` per tensor, **in place**, with the
originals backed up to one file first — every patch is byte-length-guarded and
**REVERT**-ible byte-for-byte.

### The `max k` dial = the two regimes

- **Higher k** gives a finer grid and may pass the chosen reconstruction-error
  threshold. It does not prove quality preservation or integer-only execution.
- **Lower k** gives fewer code levels. It can reduce a packed representation's
  code payload, but the final bytes depend on scales, groups, packing and retained
  tensors. The legacy in-place rewrite keeps the source storage widths.
- Inspect the actual per-tensor result and validate the converted model. There
  is no universal k boundary guaranteeing either success or collapse.

---

## Usage

**Requirements:** use the virtual environment from [Quick install and launch](#quick-install-and-launch).
The commands below assume it is activated with `source .venv/bin/activate`.

```bash
# Original Autogrid desktop UI:
python autogrid.py

# Headless scan (prints the class of every tensor + a recommended k):
python autogrid.py --scan /path/to/model            # HF dir or a .safetensors file
python autogrid.py --scan /path/to/model --max-k 2  # see what survives the compression grid
python autogrid.py --scan /path/to/model --json report.json

# Convert in place (backs up originals first), then revert if you want:
python autogrid.py --convert /path/to/model --plan plan.json
python autogrid.py --revert  /path/to/model --backup model.autogrid-backup.safetensors
```

`--scan` accepts a HuggingFace-layout directory (with
`model.safetensors.index.json` or a single `model.safetensors`) or a
`.safetensors` file directly.

### Tk-Integer Converter

Open the integrated converter from the Autogrid window with **Tk-Integer
Converter**, or launch it directly:

```bash
python autogrid.py --tk-ui
```

The native interface uses a charcoal/grey theme with cyan accents:

- **Tk-Integers**: separate file/folder pickers, output path, precision controls,
  contextual scale explanations, live format summary and persistent progress.
- **Tensor explorer**: searchable names, scrollable columns and full row details.
- **Density planner**: presets, optional residual planes, editable recipe parts
  and JSON export. This planner does not change the Converter tab's settings.
- **Model Atlas**: the original Qwen Atlas **HTML application embedded in the
  desktop tab**, with its own layout, SVG flow diagrams, clickable layer dialogs,
  source-linked information, model presets, protection and precision controls,
  memory lab, field guide, and import/export. It is not a native recreation.

### Model Atlas → Tk-Integers

1. Open **Model Atlas**, choose a Qwen preset or use the original import controls.
2. Edit the quant mix in Atlas's original flow diagram or precision list.
3. Click **Review in Tk-Integers →**, immediately beside **Import saved plan**.
4. Save the JSON recipe when prompted. The app reads that exported file and
   switches to **Tk-Integers**, showing the per-layer targets and protection.
5. Select and inspect the matching source model. Review resolved tensors and any
   unsupported settings before explicitly choosing **Convert reviewed mix**.

This handoff never starts a model job. The format-conversion adapter supports
original/protected tensors, explicit BF16/FP16/FP32 storage, and Atlas k-trit
formats with S12/FP16/FP32 scales. Unsupported INT4/INT8 weight methods,
factorization, runtime-state and operation targets remain visible and block
conversion rather than being silently replaced. Atlas's separate calibration,
Hessian, training and GitHub Actions options are retained in the recipe but are
not executed by the desktop format-conversion button.

The HTML embedding currently uses **Linux X11/XWayland**, Qt5 WebEngine and
WebChannel. On Debian/Ubuntu/KDE neon: install `python3-pyqt5.qtwebengine` in the
Python environment used for this app. Node.js 18+ runs the original Atlas plan
validation on import; the HTML itself runs in the embedded browser. Atlas uses
its own renderer process and local static-file server, not your Chrome profile.

Open directly on the Atlas tab with `.venv/bin/python tkintegers_ui.py --atlas`.
Vendored provenance is in `vendor/model_atlas/PROVENANCE.json`; original HTML,
CSS and JS are preserved. A thin integration button is appended when serving
`app.js`, and a WebChannel bootstrap is added to the served HTML.

You can also launch the interface directly with `.venv/bin/python tkintegers_ui.py`.
It opens without importing torch; if torch or safetensors is missing, it shows
setup guidance and disables model actions until the dependencies are installed
and the app is restarted. Layout and planner inspection do not load a model.

Model operations run in a background thread. The conversion inputs stay locked
during an operation so repeated clicks cannot start overlapping conversions.

The converter accepts either one `.safetensors` file or an HF-layout directory.
It inspects the tensors, lets you choose the weight `k`, weights-per-scale, and
scale container, then writes a new output directory. It refuses to write into a
non-empty directory and never modifies the source model.

The scale selector includes:

- existing Tk container k10-log scales with an integer anchor;
- TK S12 12-bit log2 scales;
- ternary-integer linear or log2 scales with a separately chosen scale `k1..k16`;
- int8, int16, and int32 linear scales with an integer anchor;
- bf16, fp16, fp32, and fp64 scale values.

The default conversion is weight `k5`, group `128`, with existing k10-log
container scales. The weight and scale `k` choices are independent.

The output is a packed Tk-Integer container, not a dense HF checkpoint. It
contains one safetensors shard per source tensor plus `manifest.json`, which
records the original shape/dtype, code packing, scale encoding, and output
payload keys. A runtime loader must read the manifest to reconstruct tensors.

Non-floating tensors are copied as raw payloads. Floating tensors are grouped,
quantized to balanced ternary k-trit codes, packed, and saved with the selected
scale stream.

The lower mixture panel remains available for designing different formats for
different model regions and exporting a `tkintegers-mixture-v1` recipe.

For a weight plane with k trits/value, `g` weights per scale, and `s` scale
bits, the ideal density is:

```text
ideal bpw = 1.6 * k + s / g
```

The density planner also reports a finite byte-payload estimate under its own
stream and scale-bit assumptions, including optional residual planes. It does
not include every converter's group padding, native-width codes, scale anchors
or file metadata. The remembered S12 choices are
visible as presets: `k1-g20-s12` is 2.2 ideal bpw and `k1-g15-s12` is 2.4 ideal
bpw. For an exact artifact count, use the writer-specific formulas in the
paper and count the files actually written; the planner is not a universal
serialized-byte measurement.

Mixture recipes export as `tkintegers-mixture-v1` JSON and can assign a
different k, scale density, scale precision, and residual to each model part.

---

## What integer storage establishes

- **Explicit stored values.** Integer codes and their matching metadata define
  a representation. They do not by themselves define the runtime arithmetic.
- **Checkable decoding.** Valid packed integer codes should round-trip with the
  matching decoder. That differs from recovering arbitrary source floats or
  obtaining identical cross-device model outputs.
- **A precision ladder.** Trit depth, scale precision and group geometry allow
  different representation choices. Lower precision still requires measured
  evaluation; the codec alone does not establish quality or speed gains.

---

## Part of TAARDIS

AUTOGRID is the noise-floor auto-detector from **TAARDIS** (*Ternary Adaptive
Alignment & Rotation for Dense Integer Stacking*) — a pipeline for full-ternary,
low-bit large language models. Forge's separate historical methods and runs are
not automatically results of this standalone application. The desktop app and
embedded Atlas are included here; numerical codec/model validation and packed
runtime integration remain separate work, as documented in the paper.

MIT licensed. By Cody Dixon, 2026.
