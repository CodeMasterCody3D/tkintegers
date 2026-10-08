# PROJECT.md — Tk-Integers

## Purpose
Native Tkinter safetensors conversion front end with selectable weight and scale
formats. The density planner is secondary to model conversion.

## Owner decisions
1. Preserve the existing container code and source models.
2. Keep the standalone repository private; baseline is saved at commit fc13169.
3. Use a futuristic charcoal/grey desktop theme with cyan highlights and clear explanations.
4. Model Atlas means the **whole original HTML UI embedded as a tab**: diagrams,
   explanations, model selection, planning and original controls. Do not replace
   it with a native table or recreate a reduced interface.

## Existing components
- `README.md` starts with the quick install and `.venv` setup. Use distribution
  Python with `--system-site-packages` to expose the apt-installed Tk/Qt bindings.
- `tkintegers_ui.py`: desktop interface, launched directly without importing torch.
- `tkintegers_convert.py`: inspection, conversion and manifest-backed tensor loader.
- `tkintegers.py`: scale policies, format descriptions and density accounting.
- `base3_pack.py`, `tk_codec.py`, `tk_s12.py`, `tkint_latent.py`: container code.
- `autogrid.py`: original Autogrid scanner and integration entry point.
- `vendor/model_atlas/`: original Atlas from quant-test at
  `aea688d38e13bee505e608abf5125a8cef159682`; HTML/CSS/JS preserved.
- `model_atlas_ui.py` / `model_atlas_webview.py`: Qt WebEngine hosted inside a
  Tk X11 frame; includes XWayland. Original HTML gets one integration button.
- `model_atlas.py` / `model_atlas_bridge.mjs`: original Atlas plan validation
  and explicit tensor-matching review. No model job starts on recipe import.

## Validation boundary
The previous commit passed metadata/accounting tests and syntax checks. Numerical
conversion and model quality are not verified by those checks. Default python3
lacked torch in the previous session. Keep UI validation independent of model runs.

## Defaults
The existing converter selects weight k5, group128, existing k10-log scales.
The planner's exported mixtures are not applied by the global conversion button.
Output is a custom packed container with a manifest, not a standard HF model.

## UI refresh, October 8
- Converter, Tensor explorer and Density planner are separate tabs.
- The dark refresh reused the converter. The later Atlas integration adds recipe
  routing with upstream Atlas codecs; see BOARD.md for that separate change.
- UI-only checks cover scale help, conditional scale-k controls, presets, part
  editing, tensor filtering, busy state and 900×640 / 1280×860 layout.
- Fixed planner residual-k=0 rejection and preset scale-bits strings like `12.0`.
- Grid-managed root keeps the progress/buttons visible at smaller sizes.
- Previous UI bytes are archived in `code_vault/` before editing.

## Technical paper
`docs/TKINTEGERS_AUTOGRID_PAPER.md` is the completed version-1.0 implementation
and analytical paper, based on app commit `4e3d916`. It separates the actual
packing layouts and scale policies, covers Model Atlas and carried residuals,
and includes verified related work. `docs/PAPER_EVIDENCE.md` and `docs/evidence/`
retain the bounded checks. Do not turn its calculated byte counts or UI tests
into claims about model quality, inference speed or training memory.
