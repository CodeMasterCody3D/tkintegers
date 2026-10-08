# Tk-Integers work record

## October 8 — dark desktop redesign

Implemented charcoal/grey surfaces, cyan accents, contextual scale help,
separate file/folder selection, three tabs, tensor filtering, recipe editing,
operation input locking, dependency guidance and a persistent action area.

UI checks exposed root pack-layout starvation at 900×640: the notebook consumed
the footer space. Switched the root to weighted grid layout and fixed converter
column proportions; checks now pass at 900×640 and 1280×860.

The old planner rejected its default residual k of zero and rendered preset
scale bits as float strings that its integer parser rejected. Both UI input
paths are corrected. No model score or container result follows from this work.

Validation: UI-only interaction smoke, syntax and whitespace checks. Torch and
safetensors are missing in this launch environment; model conversion is untested.
No codec or conversion-engine file was changed by this UI refresh.

## October 8 — Model Atlas integration and correction

Cody asked for the Qwen Atlas from private `quant-test`. Initial implementation
reused its data/logic but replaced its HTML UI with native widgets. Cody corrected
this: use the **entire original HTML UI literally inside the tab**. That native
replacement has been removed. `model_atlas_ui.py` now hosts a real Qt WebEngine
child inside the Tk frame, serving original HTML/CSS/JS from the vendored files.

One button is added beside the original Import saved plan button: Review in
Tk-Integers. It exports the actual buildPlan('combined') JSON, loads that file
into the Tk-Integers review and switches tabs. It never starts conversion.
Original Atlas import/export, diagrams, layer dialogs, field guide, memory lab,
precision/protection controls and quantization planning remain in the HTML.

Embedding initially crashed Qt5 at QWidget.show after QWindow.setParent with a
foreign Tk parent. Native X11 reparenting after initialization fixed the renderer.
Verified native parent window plus a screenshot of the real SVG diagram in Tk.
Installed four Qt5 bindings/widget packages (311 KB download, 1.43 MB installed);
no existing packages upgraded or removed.

Recipe routing: original/protected/unlisted tensors stay raw; explicit BF16/FP16/
FP32 and supported Atlas TK formats have adapters. Unsupported factorization or
runtime targets block conversion. Atlas TK uses the upstream unweighted scale
fit/snap path; Atlas quant_job/GPTQ/training is not executed by this converter.

Code review found the older global loader omitted division by the k-trit level
range when rebuilding values. Added the missing division; pre-fix k>1 loader
results would be invalid. No model conversion or quality result had been verified.
The pre-Atlas engine is preserved in code_vault.

Checks: 18 original Atlas architecture tests, eight new metadata mapping tests,
actual HTML renderer with ten presets and 23 graph nodes, opening the original
layer-info dialog, editing QKV to k3 across 48 matching blocks, button adjacency,
saved-JSON handoff, tab switching and original HTML re-import of that recipe passed.
Numerical model conversion remains unverified on remote compute.

## October 8 — README virtual-environment setup

Added top-of-page quick installation for the Linux desktop app, using
`/usr/bin/python3 -m venv --system-site-packages .venv` so apt-installed Tk/Qt
bindings remain visible to the app and its renderer subprocess. Includes clone,
install, activation, direct launch, Atlas launch, import checks and troubleshooting.
Verified a fresh temporary venv imports Tkinter, QtWebEngineWidgets and QtWebChannel;
Node is version 22.23.0 here. Did not install torch or run model conversions for
this documentation check.
