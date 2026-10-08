"""Native charcoal/cyan interface for the existing Tk-Integer converter."""
from __future__ import annotations

import importlib.util
import copy
import json
import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from tkintegers import (
    ContainerSpec, density_ladder, mixture_stats, preset_specs,
    resolve_scale_policy, scale_policy_choices,
)


COLORS = {
    "bg": "#101317", "panel": "#1b2026", "input": "#12171d",
    "border": "#303a44", "text": "#e5edf3", "muted": "#a0afbd",
    "cyan": "#52dced", "hover": "#293b45", "selected": "#164b59",
    "amber": "#e8bc79", "error": "#ff9797",
}
SCALE_HELP = {
    "existing-k10-log": "Uses the existing Tk k10-log scale encoding with an integer anchor. This selects that encoding style; it does not import scale values from another model.",
    "tk-s12": "Stores each scale as a 12-bit log2 code. With k1 weights, group 20 gives 2.2 ideal bits per weight; group 15 gives 2.4. Padding adds overhead.",
    "ternary-linear": "Stores scales as packed ternary integer codes with a shared anchor. Scale k controls their precision independently of weight k.",
    "ternary-log": "Encodes log2 scales as packed ternary integer codes with an integer anchor. Choose scale k separately from the weight precision.",
    "int8-linear": "Stores linear scale codes in int8, using positive codes 0–127 and a shared integer anchor. Widely differing group magnitudes can lose scale precision.",
    "int16-linear": "Stores linear scale codes in int16 with a shared integer anchor. More scale resolution than int8, at twice the code storage.",
    "int32-linear": "Stores linear scale codes in int32 with a shared integer anchor. Uses four bytes per scale code, plus anchor metadata.",
    "bf16": "Stores scales directly in bfloat16: two bytes each, with a wide exponent range and fewer significant bits than fp16.",
    "fp16": "Stores scales directly in float16: two bytes each. Very small or large scales may fall outside its representable range.",
    "fp32": "Stores scales directly in float32: four bytes each, with greater precision and range than fp16.",
    "fp64": "Stores scales in float64: eight bytes each. Storage precision does not change the converter's computation precision.",
}


def _parse_positive(value: str, label: str) -> int:
    try:
        number = int(value.replace(",", "").strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be a positive whole number.") from exc
    if number <= 0:
        raise ValueError(f"{label} must be greater than zero.")
    return number


def _format_bytes(value: int) -> str:
    for divisor, unit in ((1_000_000_000, "GB"), (1_000_000, "MB"), (1_000, "KB")):
        if value >= divisor:
            return f"{value / divisor:.2f} {unit}"
    return f"{value:,} B"


def _theme(window):
    """Use a named ttk theme without changing any model/container code."""
    style = ttk.Style(window)
    name = "tkintegers-dark"
    if name not in style.theme_names():
        style.theme_create(name, parent="clam")
    style.theme_use(name)
    c = COLORS
    style.configure(".", background=c["bg"], foreground=c["text"], font=("DejaVu Sans", 10))
    style.configure("TFrame", background=c["bg"])
    style.configure("Card.TFrame", background=c["panel"])
    style.configure("TLabel", background=c["bg"], foreground=c["text"])
    for prefix, bg in (("", c["bg"]), ("Card.", c["panel"])):
        style.configure(prefix + "Muted.TLabel", background=bg, foreground=c["muted"], font=("DejaVu Sans", 9))
        style.configure(prefix + "Title.TLabel", background=bg, foreground=c["text"], font=("DejaVu Sans", 14, "bold"))
        style.configure(prefix + "Cyan.TLabel", background=bg, foreground=c["cyan"], font=("DejaVu Sans", 9, "bold"))
    style.configure("Card.TLabel", background=c["panel"])
    style.configure("Hero.TLabel", font=("DejaVu Sans", 24, "bold"))
    style.configure("Badge.TLabel", background=c["selected"], foreground=c["cyan"], padding=(12, 7), font=("DejaVu Sans", 9, "bold"))
    style.configure("Warning.TLabel", foreground=c["amber"], background=c["panel"], font=("DejaVu Sans", 9))
    for control in ("TEntry", "TCombobox", "TSpinbox"):
        style.configure(control, fieldbackground=c["input"], background=c["border"], foreground=c["text"],
                        insertcolor=c["cyan"], bordercolor=c["border"], lightcolor=c["border"],
                        darkcolor=c["border"], arrowcolor=c["cyan"], padding=8)
        style.map(control, fieldbackground=[("readonly", c["input"]), ("disabled", c["panel"])],
                  foreground=[("disabled", c["muted"]), ("readonly", c["text"])],
                  bordercolor=[("focus", c["cyan"])], selectbackground=[("!disabled", c["selected"])])
    style.configure("TButton", background=c["border"], foreground=c["text"], borderwidth=0,
                    padding=(14, 10), font=("DejaVu Sans", 9, "bold"), focusthickness=1, focuscolor=c["cyan"])
    style.map("TButton", background=[("active", c["hover"])], foreground=[("disabled", "#677480")])
    style.configure("Primary.TButton", background=c["cyan"], foreground="#09212a", padding=(22, 12))
    style.map("Primary.TButton", background=[("disabled", c["border"]), ("active", "#8becf6")],
              foreground=[("disabled", c["muted"]), ("!disabled", "#09212a")])
    style.configure("TNotebook", borderwidth=0, tabmargins=(0, 0, 0, 12),
                    bordercolor=c["bg"], lightcolor=c["bg"], darkcolor=c["bg"])
    style.configure("TNotebook.Tab", padding=(23, 12), background=c["panel"], foreground=c["muted"],
                    bordercolor=c["bg"], lightcolor=c["bg"], darkcolor=c["bg"])
    style.map("TNotebook.Tab", background=[("selected", c["selected"]), ("active", c["hover"])],
              foreground=[("selected", c["cyan"])])
    style.configure("Treeview", background=c["input"], fieldbackground=c["input"], foreground=c["text"],
                    borderwidth=0, bordercolor=c["border"], lightcolor=c["border"], darkcolor=c["border"],
                    rowheight=32, font=("DejaVu Sans", 9))
    style.configure("Treeview.Heading", background=c["border"], foreground=c["muted"],
                    padding=(9, 10), relief="flat", font=("DejaVu Sans", 9, "bold"))
    style.map("Treeview", background=[("selected", c["selected"])], foreground=[("selected", "#d9fcff")])
    style.map("Treeview.Heading", background=[("active", c["hover"])])
    style.configure("Horizontal.TProgressbar", background=c["cyan"], troughcolor=c["input"],
                    borderwidth=0, bordercolor=c["bg"], lightcolor=c["cyan"], darkcolor=c["cyan"], thickness=5)
    for orient in ("Vertical", "Horizontal"):
        style.configure(orient + ".TScrollbar", background=c["border"], troughcolor=c["bg"],
                        arrowcolor=c["muted"], borderwidth=0, bordercolor=c["bg"],
                        lightcolor=c["border"], darkcolor=c["border"])
    window.configure(bg=c["bg"])
    window.option_add("*TCombobox*Listbox.background", c["input"])
    window.option_add("*TCombobox*Listbox.foreground", c["text"])
    window.option_add("*TCombobox*Listbox.selectBackground", c["selected"])
    window.option_add("*TCombobox*Listbox.selectForeground", c["cyan"])


class ConverterUI:
    def __init__(self, window):
        self.window = window
        _theme(window)
        window.title("Tk-Integers • Model Converter")
        width = min(1280, max(900, window.winfo_screenwidth() - 100))
        height = min(860, max(640, window.winfo_screenheight() - 100))
        window.geometry(f"{width}x{height}")
        window.minsize(900, 640)
        self.events = queue.Queue()
        self.busy = False
        self.rows = []
        self.inspected_source = None
        self.atlas_recipe = None
        self.recipe_path = None
        self.recipe_review = None
        self.last_atlas_export = None
        self.parts = []
        self.lockables = []
        self.choices = scale_policy_choices()
        self.scale_by_label = {p.label: p.key for p in self.choices}
        self.missing = [name for name in ("torch", "safetensors") if importlib.util.find_spec(name) is None]
        self.vars = {key: tk.StringVar(window, value=value) for key, value in {
            "source": "", "output": "", "k": "5", "group": "128", "scale_k": "5",
            "scale": self.choices[0].label, "search": "",
        }.items()}
        self.status = tk.StringVar(window, "Choose a model to get started.")
        self.summary = tk.StringVar(window)
        self.scale_help = tk.StringVar(window)
        self.density = tk.StringVar(window)
        self.counts = tk.StringVar(window, "No model inspected")
        self.details = tk.StringVar(window, "Select a tensor to see its full name, shape and source shard.")
        self.recipe_status = tk.StringVar(window, "No Atlas recipe loaded. Global settings are active.")
        self._build()
        for key in ("k", "group", "scale", "scale_k", "source", "output"):
            self.vars[key].trace_add("write", self._refresh)
        self.vars["search"].trace_add("write", self._filter_tensors)
        self._refresh()
        window.protocol("WM_DELETE_WINDOW", self.close)
        self.pump_id = window.after(100, self._pump)

    def _text(self, parent, text="", style="Card.Muted.TLabel", variable=None):
        label = ttk.Label(parent, text=text, textvariable=variable, style=style, justify="left", anchor="w", wraplength=400)
        label.pack(fill="x", pady=(0, 8))
        parent.bind("<Configure>", lambda e, w=label: w.configure(wraplength=max(180, e.width - 40)), add="+")
        return label

    def _card(self, parent, number, title, hint):
        outer = tk.Frame(parent, bg=COLORS["border"], padx=1, pady=1)
        frame = ttk.Frame(outer, style="Card.TFrame", padding=18)
        frame.pack(fill="both", expand=True)
        self._text(frame, number, "Card.Cyan.TLabel")
        self._text(frame, title, "Card.Title.TLabel")
        self._text(frame, hint)
        return outer, frame

    def _button(self, parent, text, command, primary=False):
        widget = ttk.Button(parent, text=text, command=command, style="Primary.TButton" if primary else "TButton")
        self.lockables.append((widget, "normal"))
        return widget

    def _entry(self, parent, variable, choices=None, width=15):
        if choices:
            widget = ttk.Combobox(parent, textvariable=variable, values=choices, state="readonly", width=width)
            normal = "readonly"
        else:
            widget = ttk.Entry(parent, textvariable=variable, width=width)
            normal = "normal"
        self.lockables.append((widget, normal))
        return widget

    def _scroll_page(self, notebook, title):
        shell = ttk.Frame(notebook)
        canvas = tk.Canvas(shell, bg=COLORS["bg"], highlightthickness=0, borderwidth=0)
        scroll = ttk.Scrollbar(shell, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        body = ttk.Frame(canvas)
        item = canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(item, width=e.width))
        # Bind this page only: a wheel over tables/combos keeps their own behavior.
        def wheel(event):
            if not str(event.widget).startswith(str(body)) or isinstance(event.widget, (ttk.Treeview, ttk.Combobox)):
                return
            if body.winfo_height() > canvas.winfo_height():
                delta = -1 if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0 else 1
                canvas.yview_scroll(delta * 3, "units")
        for event in ("<Button-4>", "<Button-5>", "<MouseWheel>"):
            self.window.bind(event, wheel, add="+")
        notebook.add(shell, text=title)
        return body

    def _build(self):
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(2, weight=1)
        header = ttk.Frame(self.window, padding=(26, 20, 26, 14))
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="TAARDIS  /  MODEL TOOLS", style="Cyan.TLabel").pack(anchor="w")
        self.badge = ttk.Label(header, text="SETUP NEEDED" if self.missing else "READY", style="Badge.TLabel")
        self.badge.pack(side="right")
        ttk.Label(header, text="Tk-Integers", style="Hero.TLabel").pack(anchor="w", pady=(5, 3))
        ttk.Label(header, text="Your model. Your precision. Your scale format.", style="Muted.TLabel").pack(anchor="w")
        tk.Frame(self.window, bg=COLORS["cyan"], height=2).grid(row=1, column=0, sticky="ew", padx=26)
        self.tabs = ttk.Notebook(self.window)
        self.tabs.grid(row=2, column=0, sticky="nsew", padx=26, pady=(16, 12))
        convert = self._scroll_page(self.tabs, "01   Tk-Integers")
        convert.columnconfigure(0, weight=3, minsize=435, uniform="converter")
        convert.columnconfigure(1, weight=2, minsize=275, uniform="converter")
        left, right = ttk.Frame(convert), ttk.Frame(convert)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        right.grid(row=0, column=1, sticky="nsew")

        outer, source = self._card(left, "01 / SOURCE & DESTINATION", "Choose your model", "Open a single safetensors file or a folder containing a sharded model.")
        outer.pack(fill="x", pady=(0, 12))
        self._text(source, "Source model", "Card.TLabel")
        row = ttk.Frame(source, style="Card.TFrame")
        row.pack(fill="x", pady=(0, 14))
        self._entry(row, self.vars["source"]).pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._button(row, "File…", lambda: self.browse_source(False)).pack(side="left", padx=(0, 8))
        self._button(row, "Folder…", lambda: self.browse_source(True)).pack(side="left")
        self._text(source, "Save converted model to", "Card.TLabel")
        row = ttk.Frame(source, style="Card.TFrame")
        row.pack(fill="x", pady=(0, 10))
        self._entry(row, self.vars["output"]).pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._button(row, "Browse…", self.browse_output).pack(side="left")
        self._text(source, "Use a new or empty folder. The converter writes a separate output and leaves the source in place.")

        recipe_tools = ttk.Frame(source, style="Card.TFrame")
        recipe_tools.pack(fill="x")
        self._button(recipe_tools, "Import Atlas recipe…", self.import_atlas_recipe).pack(side="left")
        self._button(recipe_tools, "Use global settings", self.clear_atlas_recipe).pack(side="left", padx=(8, 0))

        self.recipe_outer, recipe_card = self._card(left, "IMPORTED MIX / REVIEW BEFORE CONVERSION", "Atlas recipe assignments",
            "This recipe replaces the global format settings. Protected and unspecified tensors keep their original storage.")
        self.recipe_outer.pack(fill="x", pady=(0, 12))
        self._text(recipe_card, variable=self.recipe_status)
        self.recipe_table = self._table(recipe_card, (("name", "LAYER / MODULE", 260),
            ("precision", "TARGET", 170), ("group", "GROUP", 75), ("scales", "SCALES", 80),
            ("protection", "PROTECTION", 120), ("status", "MATCH / SUPPORT", 250)), height=6)
        self._text(recipe_card, "Conversion applies the format mix using Atlas's unweighted scale fitting and stored-scale snapping. It does not run the Atlas Hessian, training, factorization or GitHub job settings.")
        self.recipe_outer.pack_forget()

        outer, settings = self._card(left, "02 / FORMAT", "Choose your precision", "Weight precision and scale precision are independent settings.")
        self.settings_outer = outer
        outer.pack(fill="x", pady=(0, 12))
        grid = ttk.Frame(settings, style="Card.TFrame")
        grid.pack(fill="x")
        for col, (key, title, options, hint) in enumerate((
            ("k", "Weight precision · k", list(range(1, 17)), "Trits per weight. Higher k gives more levels and uses more space."),
            ("group", "Weights per scale", None, "Smaller groups store more scales. Larger groups reduce scale overhead."),
        )):
            cell = ttk.Frame(grid, style="Card.TFrame", padding=(0, 0, 10 if col == 0 else 0, 0))
            cell.grid(row=0, column=col, sticky="nsew")
            grid.columnconfigure(col, weight=1, uniform="precision")
            self._text(cell, title, "Card.TLabel")
            self._entry(cell, self.vars[key], options).pack(fill="x", pady=(0, 8))
            self._text(cell, hint)
        self._text(settings, "Scale format", "Card.TLabel")
        self._entry(settings, self.vars["scale"], [p.label for p in self.choices], width=20).pack(fill="x", pady=(0, 10))
        self._text(settings, variable=self.scale_help)
        self.scale_k_row = ttk.Frame(settings, style="Card.TFrame")
        self.scale_k_row.pack(fill="x")
        ttk.Label(self.scale_k_row, text="Scale precision · k", style="Card.TLabel").pack(side="left", padx=(0, 12))
        self.scale_k_box = self._entry(self.scale_k_row, self.vars["scale_k"], list(range(1, 17)), width=5)
        self.scale_k_box.pack(side="left")

        outer, review = self._card(right, "03 / REVIEW", "Conversion summary", "These settings apply globally to all floating tensors in the selected source.")
        outer.pack(fill="x", pady=(0, 12))
        self._text(review, variable=self.summary, style="Card.TLabel")
        self._text(review, variable=self.density, style="Card.Cyan.TLabel")
        self._text(review, "Ideal density is a format estimate. Padding, anchors, raw tensors and file metadata add storage. It is not a model-quality score.")
        self._text(review, "OUTPUT", "Card.Cyan.TLabel")
        self._text(review, "Packed safetensors + manifest.json. The output uses this project's loader; it is not a standard Hugging Face checkpoint.")
        outer, guide = self._card(right, "QUICK GUIDE", "What happens next?", "1. Inspect reads names, shapes and dtypes.\n\n2. Convert uses your selected format and saves each tensor.\n\n3. Progress shows which tensor has finished.")
        outer.pack(fill="x", pady=(0, 12))
        self._text(guide, "The density planner is a separate recipe tool. Its mixtures are not applied by this conversion button.")
        self._text(guide, "Runtime model conversion has not yet been verified for this version.", "Warning.TLabel")
        if self.missing:
            outer, setup = self._card(right, "ENVIRONMENT", "Conversion needs dependencies", "Missing from this Python environment: " + ", ".join(self.missing) + ". The interface and planner are available.")
            outer.pack(fill="x", pady=(0, 12))
            self._text(setup, "Install requirements.txt in the environment used to run conversion, then reopen the app.")
            self._button(setup, "Copy install command", self.copy_install).pack(anchor="w")

        explorer = ttk.Frame(self.tabs, padding=(0, 8))
        self.tabs.add(explorer, text="02   Tensor explorer")
        self._text(explorer, "Look inside your model", "Title.TLabel")
        self._text(explorer, "Inspect your source first, then filter tensor names. Selecting a row shows details; it does not change conversion settings.", "Muted.TLabel")
        filter_row = ttk.Frame(explorer)
        filter_row.pack(fill="x", pady=(4, 12))
        ttk.Label(filter_row, text="Filter by name", style="Muted.TLabel").pack(side="left", padx=(0, 12))
        ttk.Entry(filter_row, textvariable=self.vars["search"]).pack(side="left", fill="x", expand=True)
        ttk.Button(filter_row, text="Clear", command=lambda: self.vars["search"].set("")).pack(side="left", padx=(8, 0))
        self._text(explorer, variable=self.counts, style="Cyan.TLabel")
        self.tensor_table = self._table(explorer, (("name", "TENSOR", 360), ("shape", "SHAPE", 140),
            ("dtype", "DTYPE", 90), ("params", "PARAMETERS", 130), ("shard", "SOURCE SHARD", 180)))
        self.tensor_table.bind("<<TreeviewSelect>>", self._tensor_details)
        self._text(explorer, variable=self.details, style="Muted.TLabel")
        planner = self._scroll_page(self.tabs, "03   Density planner")
        self._build_planner(planner)
        atlas_page = ttk.Frame(self.tabs, padding=(0, 8))
        self.tabs.add(atlas_page, text="04   Model Atlas")
        from model_atlas_ui import ModelAtlasPanel
        self.atlas_panel = ModelAtlasPanel(self, atlas_page, COLORS)

        footer = ttk.Frame(self.window, padding=(26, 8, 26, 18))
        footer.grid(row=3, column=0, sticky="ew")
        self.footer = footer
        self.tabs.bind("<<NotebookTabChanged>>", self._tab_changed)
        self.progress = ttk.Progressbar(footer, mode="determinate")
        self.progress.pack(fill="x", pady=(0, 12))
        actions = ttk.Frame(footer)
        actions.pack(fill="x")
        self.inspect_button = self._button(actions, "Inspect source model", self.inspect)
        self.inspect_button.pack(side="right", padx=(10, 0))
        self.convert_button = self._button(actions, "Convert model", self.convert, primary=True)
        self.convert_button.pack(side="right", padx=(10, 0))
        self.status_label = ttk.Label(actions, textvariable=self.status, style="Muted.TLabel", wraplength=470, justify="left")
        self.status_label.pack(side="left", fill="x", expand=True)
        actions.bind("<Configure>", lambda e: self.status_label.configure(wraplength=max(180, e.width - 410)))

    def _tab_changed(self, _=None):
        if self.tabs.index(self.tabs.select()) == 3:
            self.footer.grid_remove()
        else:
            self.footer.grid()

    def _table(self, parent, columns, height=9):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True, pady=(0, 12))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        tree = ttk.Treeview(frame, columns=[c[0] for c in columns], show="headings", height=height, selectmode="browse")
        for key, title, width in columns:
            tree.heading(key, text=title)
            tree.column(key, width=width, minwidth=70, stretch=key in {"name", "format"})
        tree.grid(row=0, column=0, sticky="nsew")
        ys = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        xs = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        tree.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        tree.tag_configure("odd", background="#192129")
        tree.tag_configure("highlight", background=COLORS["selected"], foreground=COLORS["cyan"])
        return tree

    def _build_planner(self, parent):
        self._text(parent, "Explore storage density", "Title.TLabel")
        self._text(parent, "Plan mixtures and export recipes. This workspace estimates storage only; it does not configure the Converter tab.", "Muted.TLabel")
        outer, editor = self._card(parent, "RECIPE EDITOR", "Define a model part", "Give each region its own precision and group size. A residual is an optional second plane; set residual k to 0 to leave it out.")
        outer.pack(fill="x", pady=(0, 12))
        self.plan_vars = {key: tk.StringVar(self.window, value=value) for key, value in {
            "label": "Model body", "weights": "1000000", "k": "5", "group": "128",
            "bits": "10", "rk": "0", "rg": "", "rb": "",
        }.items()}
        fields = ttk.Frame(editor, style="Card.TFrame")
        fields.pack(fill="x")
        for index, (key, title) in enumerate((("label", "Part name"), ("weights", "Number of weights"),
                ("k", "Weight k · 1–16"), ("group", "Weights per scale"), ("bits", "Bits per scale"),
                ("rk", "Residual k · 0 = off"), ("rg", "Residual group · blank = inherit"), ("rb", "Residual bits · blank = inherit"))):
            row, col = divmod(index, 4)
            cell = ttk.Frame(fields, style="Card.TFrame", padding=(0, 0, 12, 10))
            cell.grid(row=row, column=col, sticky="ew")
            fields.columnconfigure(col, weight=1, uniform="planner")
            self._text(cell, title)
            ttk.Entry(cell, textvariable=self.plan_vars[key], width=12).pack(fill="x")
        self._text(editor, "Scale bits here describe storage, not a scale encoding. Ten bits is not the same thing as k10. This estimate excludes shared anchors and file headers.")
        presets = preset_specs()
        self.presets = {p.name: p for p in presets}
        self.preset = tk.StringVar(self.window, presets[4].name)
        bar = ttk.Frame(editor, style="Card.TFrame")
        bar.pack(fill="x", pady=(4, 0))
        ttk.Combobox(bar, textvariable=self.preset, values=list(self.presets), state="readonly", width=27).pack(side="left")
        ttk.Button(bar, text="Load preset", command=self.load_preset).pack(side="left", padx=8)
        ttk.Button(bar, text="Add part", command=self.add_part).pack(side="right")
        self.plan_summary = tk.StringVar(self.window, "No parts yet. Add a part to begin your recipe.")
        self._text(parent, variable=self.plan_summary, style="Cyan.TLabel")
        self.part_table = self._table(parent, (("name", "PART", 180), ("weights", "WEIGHTS", 120),
            ("format", "FORMAT", 280), ("ideal", "IDEAL BPW", 110), ("bytes", "PAYLOAD ESTIMATE", 160)), height=5)
        self.part_table.bind("<<TreeviewSelect>>", self.edit_part)
        bar = ttk.Frame(parent)
        bar.pack(fill="x", pady=(0, 20))
        ttk.Button(bar, text="Update selected", command=self.update_part).pack(side="left")
        ttk.Button(bar, text="Remove selected", command=self.remove_part).pack(side="left", padx=8)
        ttk.Button(bar, text="Export recipe…", command=self.export_recipe).pack(side="right")
        self._text(parent, "Scale-density ladder", "Title.TLabel")
        self._text(parent, "Every group size from 1 to 128, using the editor's weight k and scale bits. Double-click a row to use its group size. BPW means bits per weight.", "Muted.TLabel")
        self.ladder_hint = tk.StringVar(self.window)
        self._text(parent, variable=self.ladder_hint, style="Cyan.TLabel")
        self.ladder_table = self._table(parent, (("group", "WEIGHTS / SCALE", 190), ("scale", "SCALE BPW", 180),
            ("ideal", "IDEAL TOTAL BPW", 190), ("payload", "PADDED BPW ESTIMATE", 220)), height=7)
        self.ladder_table.bind("<Double-1>", self.pick_group)
        for key in ("k", "bits"):
            self.plan_vars[key].trace_add("write", self.refresh_ladder)
        self.refresh_ladder()

    def _config(self):
        from tkintegers_convert import ConversionConfig
        key = self.scale_by_label[self.vars["scale"].get()]
        return ConversionConfig(weight_k=_parse_positive(self.vars["k"].get(), "Weight k"),
            weights_per_scale=_parse_positive(self.vars["group"].get(), "Weights per scale"),
            scale_type=key, scale_k=_parse_positive(self.vars["scale_k"].get(), "Scale k"))

    def _refresh(self, *_):
        key = self.scale_by_label.get(self.vars["scale"].get(), "existing-k10-log")
        self.scale_help.set(SCALE_HELP[key])
        if key.startswith("ternary-"):
            self.scale_k_row.pack(fill="x")
        else:
            self.scale_k_row.pack_forget()
        valid = True
        try:
            config = self._config()
            policy = resolve_scale_policy(config.scale_type, config.scale_k)
            self.summary.set(f"Weights   k{config.weight_k}\nGrouping   {config.weights_per_scale:,} weights per scale\nScales   {policy.label}")
            self.density.set(f"{1.6 * config.weight_k + policy.bits / config.weights_per_scale:.4f} ideal bits / weight")
        except (ValueError, KeyError) as exc:
            valid = False
            self.summary.set(str(exc))
            self.density.set("Complete the format settings to see an estimate.")
        if self.atlas_recipe is not None:
            source = str(Path(self.vars["source"].get()).expanduser().resolve()) if self.vars["source"].get().strip() else None
            matched = self.inspected_source == source and self.recipe_review is not None and self.recipe_review["matched"]
            valid = matched and not self.recipe_review["issues"]
            self.summary.set(f"Atlas recipe: {self.atlas_recipe.get('model', 'Imported model')}\n{len(self.atlas_recipe['precision']):,} layer/runtime assignments\nSource match: {'reviewed' if matched else 'inspect the selected source'}")
            self.density.set("Per-layer mix active · review individual assignments")
            self.convert_button.configure(text="Convert reviewed mix")
        else:
            self.convert_button.configure(text="Convert model")
        ready = not self.busy and not self.missing
        self.inspect_button.configure(state="normal" if ready and self.vars["source"].get().strip() else "disabled")
        self.convert_button.configure(state="normal" if ready and valid and all(self.vars[k].get().strip() for k in ("source", "output")) else "disabled")

    def browse_source(self, folder):
        selected = (filedialog.askdirectory(parent=self.window, title="Choose model folder") if folder else
            filedialog.askopenfilename(parent=self.window, title="Choose safetensors model", filetypes=[("Safetensors", "*.safetensors")]))
        if selected:
            self.vars["source"].set(selected)
            path = Path(selected)
            if not self.vars["output"].get().strip():
                self.vars["output"].set(str(path.parent / f"{path.stem if not folder else path.name}.tkintegers"))
            self.rows = []
            self._filter_tensors()
            self.status.set("Source selected. Inspect it to view tensors, or review the format settings.")

    def browse_output(self):
        path = filedialog.askdirectory(parent=self.window, title="Choose empty output folder (or type a new path)")
        if path:
            self.vars["output"].set(path)

    def copy_install(self):
        self.window.clipboard_clear()
        self.window.clipboard_append(f'python3 -m pip install -r "{Path(__file__).with_name("requirements.txt")}"')
        self.status.set("Install command copied. Use it in the environment that will run conversion.")

    def import_atlas_recipe(self, path=None):
        if self.busy:
            return
        if path is None:
            path = filedialog.askopenfilename(parent=self.window, title="Import Atlas quant mix for review",
                                              filetypes=[("Atlas JSON recipe", "*.json")])
        if not path:
            return
        try:
            from model_atlas import atlas_request, read_recipe, review_recipe
            candidate = atlas_request(plan=read_recipe(path))
            plan = candidate["plan"]
            # Validate without changing the current recipe on malformed input.
            review_recipe(plan)
            self.atlas_recipe = copy.deepcopy(plan)
            self.recipe_path = str(path)
            self.settings_outer.pack_forget()
            self.recipe_outer.pack(fill="x", pady=(0,12))
            self._review_atlas()
            self.tabs.select(0)
            self.status.set("Atlas mix loaded for review. Inspect the matching source before converting.")
            self._refresh()
        except Exception as exc:
            messagebox.showerror("Could not import Atlas recipe", str(exc), parent=self.window)

    def review_html_atlas(self, plan):
        """Save/reuse the real HTML export, then load that file into review."""
        if self.busy:
            self.status.set("Wait for the current conversion before changing its recipe.")
            return
        try:
            path = self.last_atlas_export
            if path is not None:
                try:
                    if json.loads(Path(path).read_text()) != plan:
                        path = None
                except (OSError, ValueError):
                    path = None
            if path is None:
                path = filedialog.asksaveasfilename(parent=self.window,
                    title="Save Atlas recipe and review in Tk-Integers", initialfile="model-atlas-quant-mix.json",
                    defaultextension=".json", filetypes=[("Atlas JSON recipe", "*.json")])
                if not path:
                    return
                Path(path).write_text(json.dumps(plan, indent=2)+"\n")
            self.last_atlas_export = path
            self.import_atlas_recipe(path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not save Atlas recipe", str(exc), parent=self.window)

    def _review_atlas(self):
        from model_atlas import review_recipe
        source = str(Path(self.vars["source"].get()).expanduser().resolve()) if self.vars["source"].get().strip() else None
        tensors = self.rows if source is not None and source == self.inspected_source else None
        self.recipe_review = review_recipe(self.atlas_recipe, tensors)
        review = self.recipe_review
        self.recipe_table.delete(*self.recipe_table.get_children())
        for i,row in enumerate(review["rows"]):
            self.recipe_table.insert("","end",tags=("odd",) if i%2 else (), values=(row["path"],
                row["precision"],row["group"],row["scales"],row["protection"],row["status"]))
        state = "Source matched. Review the assignments before clicking Convert reviewed mix." if review["matched"] else "Choose the matching model and inspect its tensors to verify this mix."
        issues = review["issues"]
        if issues:
            state = f"{len(issues)} issue(s) block conversion:\n" + "\n".join(issues[:8])
            if len(issues)>8:
                state += "\nMore entries need attention; inspect the assignment table."
        self.recipe_status.set(f"{self.atlas_recipe.get('model', 'Model')}\nRecipe: {self.recipe_path}\n{state}")

    def clear_atlas_recipe(self):
        if self.busy:
            return
        self.atlas_recipe = self.recipe_review = self.recipe_path = None
        self.recipe_outer.pack_forget()
        self.settings_outer.pack(fill="x",pady=(0,12))
        self.status.set("Global format settings are active.")
        self._refresh()

    def _set_busy(self, busy, text):
        self.busy = busy
        for widget, normal in self.lockables:
            widget.configure(state="disabled" if busy else normal)
        self.badge.configure(text="WORKING" if busy else ("SETUP NEEDED" if self.missing else "READY"))
        self.status.set(text)
        self.atlas_panel.set_enabled(not busy)
        self._refresh()

    def _start(self, task, text):
        if self.busy:
            return
        self._set_busy(True, text)
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        def run():
            try:
                task()
            except Exception as exc:
                self.events.put(("error", str(exc)))
        threading.Thread(target=run, daemon=True).start()

    def inspect(self):
        source = self.vars["source"].get().strip()
        if not source or self.missing:
            return
        def task():
            from tkintegers_convert import inspect_model
            self.events.put(("inspected", inspect_model(source), source))
        self._start(task, "Reading tensor metadata…")

    def convert(self):
        if self.busy or self.missing:
            return
        try:
            from tkintegers_convert import ConversionConfig
            config = ConversionConfig() if self.atlas_recipe is not None else self._config()
            atlas_plan = copy.deepcopy(self.atlas_recipe)
            if atlas_plan is not None:
                self._review_atlas()
                if not self.recipe_review["matched"] or self.recipe_review["issues"]:
                    raise ValueError("Inspect the matching source and resolve the Atlas recipe issues before converting.")
            source_text, output_text = (self.vars[k].get().strip() for k in ("source", "output"))
            if not source_text or not output_text:
                raise ValueError("Choose a source and output folder first.")
            source, output = (Path(s).expanduser().resolve() for s in (source_text, output_text))
            if not source.exists():
                raise ValueError("The source path does not exist. Choose a model file or folder.")
            if source == output or (source.is_dir() and output.is_relative_to(source)):
                raise ValueError("Choose an output folder outside the source model.")
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                raise ValueError("The output is not an empty folder. Choose a new or empty folder.")
        except (OSError, ValueError) as exc:
            self.status.set(str(exc))
            messagebox.showerror("Check conversion settings", str(exc), parent=self.window)
            return
        def task():
            from tkintegers_convert import convert_model
            manifest = convert_model(source, output, config,
                progress=lambda i, n, name: self.events.put(("progress", i, n, name)), atlas_plan=atlas_plan)
            self.events.put(("converted", manifest, str(output)))
        self._start(task, "Preparing conversion…")

    def _pump(self):
        try:
            for _ in range(100):
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "atlas_web":
                    self.atlas_panel.handle(event[1])
                    continue
                if kind == "progress":
                    _, index, total, name = event
                    self.progress.stop()
                    self.progress.configure(mode="determinate", maximum=max(total, 1), value=index)
                    self.status.set(f"{index:,} / {total:,} tensors saved · {name}")
                else:
                    self.progress.stop()
                    self.progress.configure(mode="determinate", value=0)
                    if kind == "inspected":
                        self.rows = event[1]
                        self.inspected_source = str(Path(event[2]).expanduser().resolve())
                        self._filter_tensors()
                        if self.atlas_recipe is not None:
                            self._review_atlas()
                        self._set_busy(False, f"Inspected {len(self.rows):,} tensors from {Path(event[2]).name}.")
                        self.tabs.select(0 if self.atlas_recipe is not None else 1)
                    elif kind == "converted":
                        stats = event[1]["summary"]
                        self._set_busy(False, f"Saved {stats['quantized_tensors']:,} quantized + {stats['raw_tensors']:,} raw tensors to {event[2]}")
                        self.progress.configure(maximum=1, value=1)
                        self.badge.configure(text="SAVED")
                    elif kind == "error":
                        self._set_busy(False, "Could not finish: " + event[1])
                        self.badge.configure(text="NEEDS ATTENTION")
                        messagebox.showerror("Tk-Integers", event[1], parent=self.window)
        except queue.Empty:
            pass
        self.pump_id = self.window.after(100, self._pump)

    def _filter_tensors(self, *_):
        query = self.vars["search"].get().strip().lower()
        self.tensor_table.delete(*self.tensor_table.get_children())
        count = 0
        for index, row in enumerate(self.rows):
            if query not in row["tensor"].lower():
                continue
            self.tensor_table.insert("", "end", iid=str(index), tags=("odd",) if count % 2 else (),
                values=(row["tensor"], str(tuple(row["shape"])), row["dtype"], f"{row['params']:,}", Path(row["source_shard"]).name))
            count += 1
        self.counts.set(f"{count:,} of {len(self.rows):,} tensors · {sum(r['params'] for r in self.rows):,} total parameters" if self.rows else "No tensors yet. Choose a source in Converter, then click Inspect source model.")
        self.details.set("Select a tensor to see its full name, shape and source shard.")

    def _tensor_details(self, _=None):
        selected = self.tensor_table.selection()
        if selected:
            row = self.rows[int(selected[0])]
            self.details.set(f"{row['tensor']}\nShape {tuple(row['shape'])} · {row['dtype']} · {row['params']:,} parameters\n{row['source_shard']}")

    def _part(self):
        v = {k: var.get().strip() for k, var in self.plan_vars.items()}
        rk = int(v["rk"] or "0")
        return {"label": v["label"] or "Unnamed part", "weights": _parse_positive(v["weights"], "Weight count"),
            "spec": ContainerSpec(_parse_positive(v["k"], "Weight k"), _parse_positive(v["group"], "Group size"),
                _parse_positive(v["bits"], "Scale bits"), residual_k=rk,
                residual_weights_per_scale=_parse_positive(v["rg"], "Residual group") if v["rg"] else 0,
                residual_scale_bits=_parse_positive(v["rb"], "Residual scale bits") if v["rb"] else 0)}

    def add_part(self):
        try:
            self.parts.append(self._part())
            self._render_parts()
        except ValueError as exc:
            messagebox.showerror("Check recipe fields", str(exc), parent=self.window)

    def update_part(self):
        selected = self.part_table.selection()
        if not selected:
            self.plan_summary.set("Select a part to update, or choose Add part to create one.")
            return
        try:
            index = int(selected[0])
            self.parts[index] = self._part()
            self._render_parts()
            self.part_table.selection_set(str(index))
        except ValueError as exc:
            messagebox.showerror("Check recipe fields", str(exc), parent=self.window)

    def remove_part(self):
        selected = self.part_table.selection()
        if selected:
            self.parts.pop(int(selected[0]))
            self._render_parts()

    def _render_parts(self):
        self.part_table.delete(*self.part_table.get_children())
        for index, part in enumerate(self.parts):
            stats = part["spec"].storage(part["weights"])
            self.part_table.insert("", "end", iid=str(index), tags=("odd",) if index % 2 else (),
                values=(part["label"], f"{part['weights']:,}", part["spec"].format_name(), f"{stats['ideal_bpw']:.4f}", _format_bytes(stats["payload_bytes"])))
        stats = mixture_stats(self.parts)
        self.plan_summary.set(f"{len(self.parts)} parts · {stats['weights']:,} weights · {stats['ideal_bpw']:.4f} ideal bpw · {_format_bytes(stats['payload_bytes'])} estimated payload" if self.parts else "No parts yet. Add a part to begin your recipe.")

    def _load_spec(self, spec):
        for key, value in {"k": spec.k, "group": spec.weights_per_scale, "bits": spec.scale_bits,
                "rk": spec.residual_k, "rg": spec.residual_weights_per_scale or "", "rb": spec.residual_scale_bits or ""}.items():
            self.plan_vars[key].set(f"{value:g}" if isinstance(value, float) else str(value))

    def load_preset(self):
        self._load_spec(self.presets[self.preset.get()])

    def edit_part(self, _=None):
        selected = self.part_table.selection()
        if selected:
            part = self.parts[int(selected[0])]
            self._load_spec(part["spec"])
            self.plan_vars["label"].set(part["label"])
            self.plan_vars["weights"].set(str(part["weights"]))

    def refresh_ladder(self, *_):
        try:
            k = _parse_positive(self.plan_vars["k"].get(), "Weight k")
            bits = _parse_positive(self.plan_vars["bits"].get(), "Scale bits")
            rows = density_ladder(k, bits)
        except ValueError as exc:
            self.ladder_hint.set(str(exc))
            return
        self.ladder_table.delete(*self.ladder_table.get_children())
        for row in rows:
            group = row["group"]
            tag = "highlight" if k == 1 and bits == 12 and group in (15, 20) else ("odd" if group % 2 else "")
            self.ladder_table.insert("", "end", iid=str(group), tags=(tag,), values=(group,
                f"{row['scale_bpw']:.4f}", f"{row['ideal_bpw']:.4f}", f"{row['payload_bpw']:.4f}"))
        self.ladder_hint.set(f"k{k} · {bits}-bit scales · padded estimate uses 1,000,000 weights" + (" · Cyan: S12 g15 and g20" if k == 1 and bits == 12 else ""))

    def pick_group(self, _=None):
        selected = self.ladder_table.selection()
        if selected:
            self.plan_vars["group"].set(selected[0])

    def export_recipe(self):
        if not self.parts:
            self.plan_summary.set("Add at least one part before exporting a recipe.")
            return
        path = filedialog.asksaveasfilename(parent=self.window, title="Export mixture recipe", defaultextension=".json", filetypes=[("JSON recipe", "*.json")])
        if path:
            stats = mixture_stats(self.parts)
            payload = {"format": "tkintegers-mixture-v1", "parts": [
                {"label": p["label"], "weights": p["weights"], "spec": p["spec"].to_dict()} for p in self.parts],
                "summary": {key: value for key, value in stats.items() if key != "parts"}}
            try:
                Path(path).write_text(json.dumps(payload, indent=2) + "\n")
                self.plan_summary.set(f"Recipe saved to {path}")
            except OSError as exc:
                messagebox.showerror("Could not save recipe", str(exc), parent=self.window)

    def close(self):
        if self.busy:
            messagebox.showinfo("Work in progress", "Please let the current operation finish before closing this window.", parent=self.window)
            return
        self.window.after_cancel(self.pump_id)
        self.atlas_panel.close()
        self.window.destroy()


def launch_tkintegers_ui(parent=None, *, initial_tab=0):
    window = tk.Toplevel(parent) if parent is not None else tk.Tk()
    window.converter_ui = ConverterUI(window)
    window.converter_ui.tabs.select(initial_tab)
    if parent is None:
        window.mainloop()
    return window


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tk-Integers desktop converter with embedded Model Atlas")
    parser.add_argument("--atlas", action="store_true", help="open the original Atlas HTML tab")
    launch_tkintegers_ui(initial_tab=3 if parser.parse_args().atlas else 0)
