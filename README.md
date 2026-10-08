# Tk-Integers

A desktop safetensors converter with selectable weight/scale formats and the
original Model Atlas HTML interface embedded in a tab. Built on Autogrid.

## Quick install and launch

**Supported full desktop setup:** Ubuntu 24.04+, Debian 12+, or KDE neon with an
X11 desktop or XWayland. Use the distribution's **Python 3.10+**, **Node.js 18+**,
Tkinter, and Qt5 WebEngine. The embedded Atlas tab currently requires X11/XWayland.

Clone this repository first, then run the following **in a terminal inside the
`tkintegers` folder**. The repository is private, so cloning requires access and
your normal GitHub authentication.

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

**Turn any model into a full-size *integer* model — for free, where the math says it's free.**

AUTOGRID is a desktop app + CLI that scans every tensor of a safetensors
checkpoint, measures what it would cost to put that tensor on a
**balanced-ternary integer grid** (`w ≈ s · Σ dᵢ·3⁻ⁱ`, digits `dᵢ ∈ {−1,0,+1}`),
and converts the ones where the conversion is **below the noise the model was
already saved with.** No retraining. No calibration data. Fully reversible.

This is **not quantization** — it doesn't make the model smaller. It's a
**numeral-system change**: the same values, re-expressed as stacks of ternary
digits so the model's parameters become *integers* instead of floats.

---

## The idea: the conversion is quieter than the container

A float format has its own rounding noise just by existing. **bf16** keeps 8
significant bits, so it rounds every value it stores by up to **~0.39%**.

A **k-digit** balanced-ternary stack has a relative error of exactly **3⁻ᵏ**:

| digits `k` | error `3⁻ᵏ` | bits/value (group 128) |
|---|---|---|
| 6 | 0.14% | ~12.1 |
| 8 | 0.015% | ~16.1 |

At **k=8**, the ternary-integer error (0.015%) is **~25× smaller** than the bf16
container's own rounding (0.39%). So converting a bf16 tensor to a k8 digit stack
**discards less information than the checkpoint's storage format already did** —
the model cannot notice a perturbation smaller than the noise it was saved with.

**Measured proof:** on a 27-billion-parameter model, converting all 353
normalization / state-space tensors to k8 integer stacks changed wikitext
perplexity by **0.009%** over 274 chunks — one hundredth of the measurement's
error bar. Field convention says "never touch the norms"; the noise-floor math
says you can, and the measurement agrees.

---

## What it does

For each tensor, AUTOGRID measures the per-digit-budget error and classifies it:

- 🟢 **FREE @ k** — a float tensor whose measured k-digit error is **at or below
  its container's noise floor**. Converting it to a ternary-integer stack is free.
  AUTOGRID recommends the smallest sufficient `k`.
- 🔵 **TERNARY** — a tensor whose *values* are already on the `{−1,0,+1}×scale`
  grid (e.g. a placed/quantized weight). AUTOGRID analyzes its **group scales**
  for digit-stack conversion instead.
- 🔴 **STEER** — error is still above the floor even at k=8. This tensor needs a
  real correction pipeline (rotation / Hessian-aware placement / reconstruction),
  **not** a numeral-system change. **AUTOGRID refuses to touch it.**

Then **CONVERT** applies the recommended `k` per tensor, **in place**, with the
originals backed up to one file first — every patch is byte-length-guarded and
**REVERT**-ible byte-for-byte.

### The `max k` dial = the two regimes

- **k = 8 … 3** — the *free zone*. Convert to integer with no measurable quality
  cost. You get integer arithmetic and bit-exact reproducibility; the model is
  the **same size** (this is conversion, not compression).
- **k = 2 … 1** — the *compression regime* (2–4 bits/value). Here naive
  conversion **collapses** — this is where quantization lives, and where a
  steered pipeline (not AUTOGRID) is required. Set `max k` to 2 and watch the
  weight matrices flip to 🔴 STEER: that's the app telling you the truth about
  where "free" ends and "earned" begins.

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

The app also reports finite byte-payload bpw, which includes packed trit bytes,
scale-stream padding, and residual planes. The remembered S12 choices are
visible as presets: `k1-g20-s12` is 2.2 ideal bpw and `k1-g15-s12` is 2.4 ideal
bpw. The exact finite payload can be slightly higher when a tensor has an odd
number of scale codes or an incomplete group.

Mixture recipes export as `tkintegers-mixture-v1` JSON and can assign a
different k, scale density, scale precision, and residual to each model part.

---

## Why "integer, not smaller" is worth wanting

- **Integer arithmetic** end-to-end where it matters — no float in the values.
- **Bit-exact reproducibility** — integer values don't drift across machines.
- **A principled starting point for compression.** Once a model is on the integer
  ladder, dropping digit planes (k8 → … → k1) is a clean, exactly-known descent,
  and the dropped planes are themselves ternary corrections. AUTOGRID is the
  *front door*; the compression regime below k3 is where a steered pipeline
  takes over.

---

## Part of TAARDIS

AUTOGRID is the noise-floor auto-detector from **TAARDIS** (*Ternary Adaptive
Alignment & Rotation for Dense Integer Stacking*) — a pipeline for full-ternary,
fully-integer large language models. The digit-ladder math and the noise-floor
principle are validated there at 27B scale.

MIT licensed. By Cody Dixon, 2026.
