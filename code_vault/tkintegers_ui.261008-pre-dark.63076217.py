"""Tkinter safetensor converter and Tk-Integer mixture designer."""
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

from tkintegers import (
    ContainerSpec,
    density_ladder,
    mixture_stats,
    preset_specs,
    scale_policy_choices,
)


EXAMPLE_WEIGHTS = 1_000_000


def _parse_positive(value: str, label: str) -> int:
    try:
        number = int(value.replace(",", "").strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if number <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return number


def _format_bytes(value: int) -> str:
    if value < 1_000:
        return f"{value:,} B"
    if value < 1_000_000:
        return f"{value / 1_000:.2f} KB"
    if value < 1_000_000_000:
        return f"{value / 1_000_000:.2f} MB"
    return f"{value / 1_000_000_000:.2f} GB"


def launch_tkintegers_ui(parent=None):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    window = tk.Toplevel(parent) if parent is not None else tk.Tk()
    window.title("TAARDIS Tk-Integers Converter")
    window.geometry("1440x960")
    window.minsize(1120, 760)

    presets = preset_specs()
    preset_by_name = {spec.name: spec for spec in presets}
    parts = []

    header = ttk.Frame(window, padding=(16, 14, 16, 8))
    header.pack(fill="x")
    ttk.Label(header, text="TK-INTEGERS CONVERTER", font=("TkDefaultFont", 18, "bold")).pack(anchor="w")
    ttk.Label(
        header,
        text=("Select a safetensors model, choose the weight k and scale container, "
              "then write a manifest-backed packed Tk-Integer model."),
    ).pack(anchor="w", pady=(3, 0))

    conversion_queue = queue.Queue()
    conversion_frame = ttk.LabelFrame(window, text="Convert safetensors model", padding=10)
    conversion_frame.pack(fill="x", padx=12, pady=(0, 8))
    conversion_vars = {
        "model": tk.StringVar(),
        "output": tk.StringVar(),
        "weight_k": tk.StringVar(value="5"),
        "group": tk.StringVar(value="128"),
        "scale_k": tk.StringVar(value="5"),
        "scale_label": tk.StringVar(value=scale_policy_choices()[0].label),
        "status": tk.StringVar(value="Choose a source model and output directory."),
    }
    scale_choices = scale_policy_choices()
    scale_by_label = {choice.label: choice.key for choice in scale_choices}

    def browse_model() -> None:
        selected = filedialog.askopenfilename(
            parent=window,
            title="Select safetensors model or shard",
            filetypes=[("Safetensors", "*.safetensors"), ("All files", "*")],
        )
        if not selected:
            selected = filedialog.askdirectory(parent=window, title="Select HF safetensors model directory")
        if selected:
            conversion_vars["model"].set(selected)
            if not conversion_vars["output"].get().strip():
                source = Path(selected)
                stem = source.stem if source.is_file() else source.name
                conversion_vars["output"].set(str(source.parent / f"{stem}.tkintegers"))

    def browse_output() -> None:
        selected = filedialog.askdirectory(parent=window, title="Select empty Tk-Integer output directory")
        if selected:
            conversion_vars["output"].set(selected)

    path_row = ttk.Frame(conversion_frame)
    path_row.pack(fill="x")
    ttk.Label(path_row, text="Source model").pack(side="left")
    ttk.Entry(path_row, textvariable=conversion_vars["model"]).pack(side="left", fill="x", expand=True, padx=6)
    ttk.Button(path_row, text="Browse", command=browse_model).pack(side="left")
    ttk.Label(path_row, text="Output").pack(side="left", padx=(12, 0))
    ttk.Entry(path_row, textvariable=conversion_vars["output"], width=34).pack(side="left", padx=6)
    ttk.Button(path_row, text="Browse", command=browse_output).pack(side="left")

    option_row = ttk.Frame(conversion_frame)
    option_row.pack(fill="x", pady=(8, 0))
    ttk.Label(option_row, text="Weight k").pack(side="left")
    ttk.Combobox(
        option_row,
        textvariable=conversion_vars["weight_k"],
        values=[str(k) for k in range(1, 17)],
        state="readonly",
        width=5,
    ).pack(side="left", padx=(5, 14))
    ttk.Label(option_row, text="Weights / scale").pack(side="left")
    ttk.Entry(option_row, textvariable=conversion_vars["group"], width=8).pack(side="left", padx=(5, 14))
    ttk.Label(option_row, text="Scale container").pack(side="left")
    scale_box = ttk.Combobox(
        option_row,
        textvariable=conversion_vars["scale_label"],
        values=[choice.label for choice in scale_choices],
        state="readonly",
        width=42,
    )
    scale_box.pack(side="left", padx=5)
    ttk.Label(option_row, text="Scale k").pack(side="left", padx=(10, 0))
    scale_k_entry = ttk.Spinbox(
        option_row,
        from_=1,
        to=16,
        increment=1,
        textvariable=conversion_vars["scale_k"],
        width=5,
    )
    scale_k_entry.pack(side="left", padx=5)

    action_row = ttk.Frame(conversion_frame)
    action_row.pack(fill="x", pady=(8, 0))
    conversion_progress = ttk.Progressbar(action_row, mode="determinate")
    conversion_progress.pack(side="left", fill="x", expand=True, padx=(0, 8))
    ttk.Button(action_row, text="Inspect model", command=lambda: inspect_selected_model()).pack(side="left")
    ttk.Button(action_row, text="CONVERT MODEL", command=lambda: convert_selected_model()).pack(side="left", padx=6)
    ttk.Label(conversion_frame, textvariable=conversion_vars["status"], wraplength=1200).pack(anchor="w", pady=(6, 0))

    model_table_frame = ttk.LabelFrame(conversion_frame, text="Model tensors", padding=6)
    model_table_frame.pack(fill="x", pady=(8, 0))
    model_columns = ("tensor", "shape", "dtype", "params", "source")
    model_table = ttk.Treeview(model_table_frame, columns=model_columns, show="headings", height=5)
    for column, title, width in (
        ("tensor", "tensor", 410),
        ("shape", "shape", 170),
        ("dtype", "dtype", 90),
        ("params", "parameters", 120),
        ("source", "source shard", 420),
    ):
        model_table.heading(column, text=title)
        model_table.column(column, width=width, minwidth=80, stretch=column in {"tensor", "source"})
    model_table.pack(fill="x", expand=True)

    def update_scale_controls(*_event) -> None:
        key = scale_by_label.get(conversion_vars["scale_label"].get(), "")
        state = "normal" if key in {"ternary-linear", "ternary-log"} else "disabled"
        scale_k_entry.configure(state=state)

    scale_box.bind("<<ComboboxSelected>>", update_scale_controls)
    update_scale_controls()

    def conversion_config():
        from tkintegers_convert import ConversionConfig

        key = scale_by_label.get(conversion_vars["scale_label"].get())
        if not key:
            raise ValueError("choose a scale container")
        return ConversionConfig(
            weight_k=_parse_positive(conversion_vars["weight_k"].get(), "weight k"),
            weights_per_scale=_parse_positive(conversion_vars["group"].get(), "weights / scale"),
            scale_type=key,
            scale_k=_parse_positive(conversion_vars["scale_k"].get(), "scale k"),
        )

    def inspect_selected_model() -> None:
        model_path = conversion_vars["model"].get().strip()
        if not model_path:
            messagebox.showerror("Tk-Integers", "Select a safetensors model first.", parent=window)
            return
        model_table.delete(*model_table.get_children())
        conversion_vars["status"].set("Reading tensor metadata ...")

        def worker() -> None:
            try:
                from tkintegers_convert import inspect_model

                rows = inspect_model(model_path)
                conversion_queue.put(("inspect", rows))
            except Exception as exc:
                conversion_queue.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def convert_selected_model() -> None:
        model_path = conversion_vars["model"].get().strip()
        output_path = conversion_vars["output"].get().strip()
        if not model_path or not output_path:
            messagebox.showerror(
                "Tk-Integers",
                "Select both a safetensors source model and an output directory.",
                parent=window,
            )
            return
        try:
            config = conversion_config()
        except ValueError as exc:
            messagebox.showerror("Tk-Integers", str(exc), parent=window)
            return
        output = Path(output_path).expanduser()
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            messagebox.showerror(
                "Tk-Integers",
                "The output directory is not empty. Choose a new directory; the converter never overwrites it.",
                parent=window,
            )
            return
        conversion_vars["status"].set(
            f"Converting with weight k{config.weight_k}, g{config.weights_per_scale}, "
            f"{config.scale_type} scales ..."
        )
        conversion_progress["value"] = 0
        conversion_progress["maximum"] = max(1, len(model_table.get_children()))

        def worker() -> None:
            try:
                from tkintegers_convert import convert_model

                manifest = convert_model(
                    model_path,
                    output_path,
                    config,
                    progress=lambda index, total, name: conversion_queue.put(
                        ("progress", index, total, name)
                    ),
                )
                conversion_queue.put(("converted", manifest))
            except Exception as exc:
                conversion_queue.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    body = ttk.Panedwindow(window, orient="horizontal")
    body.pack(fill="both", expand=True, padx=12, pady=8)
    left = ttk.Frame(body, padding=8)
    right = ttk.Frame(body, padding=8)
    body.add(left, weight=3)
    body.add(right, weight=2)

    fields = ttk.LabelFrame(left, text="Mixture part", padding=10)
    fields.pack(fill="x")
    values = {
        "label": tk.StringVar(value="example body"),
        "weights": tk.StringVar(value=str(EXAMPLE_WEIGHTS)),
        "k": tk.StringVar(value="5"),
        "group": tk.StringVar(value="128"),
        "scale_bits": tk.StringVar(value="10"),
        "residual_k": tk.StringVar(value="0"),
        "residual_group": tk.StringVar(value=""),
        "residual_scale_bits": tk.StringVar(value=""),
    }
    labels = (
        ("label", "Part name"),
        ("weights", "Weights"),
        ("k", "Weight k"),
        ("group", "Weights / scale"),
        ("scale_bits", "Scale bits"),
        ("residual_k", "Residual k"),
        ("residual_group", "Residual weights / scale"),
        ("residual_scale_bits", "Residual scale bits"),
    )
    for index, (key, label) in enumerate(labels):
        row, col = divmod(index, 4)
        ttk.Label(fields, text=label).grid(row=row * 2, column=col, sticky="w", padx=4, pady=(0, 2))
        ttk.Entry(fields, textvariable=values[key], width=18).grid(
            row=row * 2 + 1, column=col, sticky="ew", padx=4, pady=(0, 7)
        )
    for col in range(4):
        fields.columnconfigure(col, weight=1)

    preset_row = ttk.Frame(fields)
    preset_row.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(2, 0))
    preset_var = tk.StringVar(value=presets[2].name)
    ttk.Label(preset_row, text="Presets").pack(side="left")
    preset_box = ttk.Combobox(
        preset_row,
        textvariable=preset_var,
        values=[spec.name for spec in presets],
        state="readonly",
        width=28,
    )
    preset_box.pack(side="left", padx=6, fill="x", expand=True)

    table_frame = ttk.LabelFrame(left, text="Mixture parts", padding=8)
    table_frame.pack(fill="both", expand=True, pady=(10, 0))
    columns = ("label", "weights", "format", "ideal", "payload", "bytes")
    table = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")
    headings = {
        "label": ("part", 150),
        "weights": ("weights", 105),
        "format": ("format", 240),
        "ideal": ("ideal bpw", 90),
        "payload": ("payload bpw", 100),
        "bytes": ("payload", 105),
    }
    for col, (title, width) in headings.items():
        table.heading(col, text=title)
        table.column(col, width=width, minwidth=70, stretch=col in {"label", "format"})
    table.pack(side="left", fill="both", expand=True)
    table_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=table.yview)
    table_scroll.pack(side="right", fill="y")
    table.configure(yscrollcommand=table_scroll.set)

    summary = tk.StringVar(value="Add a part to calculate the mixture.")
    ttk.Label(left, textvariable=summary, padding=(3, 9), wraplength=750).pack(fill="x")

    buttons = ttk.Frame(left)
    buttons.pack(fill="x", pady=(0, 4))

    ladder = ttk.LabelFrame(right, text="All scale densities", padding=8)
    ladder.pack(fill="both", expand=True)
    ttk.Label(
        ladder,
        text="Every group size 1..128 is shown. Use a custom group size in the part editor for any larger value.",
        wraplength=360,
    ).pack(anchor="w", pady=(0, 8))
    ladder_columns = ("group", "scale", "ideal", "payload")
    ladder_table = ttk.Treeview(ladder, columns=ladder_columns, show="headings")
    for col, title, width in (
        ("group", "weights / scale", 110),
        ("scale", "scale bpw", 92),
        ("ideal", "ideal bpw", 92),
        ("payload", "payload bpw", 105),
    ):
        ladder_table.heading(col, text=title)
        ladder_table.column(col, width=width, minwidth=80, stretch=False, anchor="e")
    ladder_table.pack(side="left", fill="both", expand=True)
    ladder_scroll = ttk.Scrollbar(ladder, orient="vertical", command=ladder_table.yview)
    ladder_scroll.pack(side="right", fill="y")
    ladder_table.configure(yscrollcommand=ladder_scroll.set)
    ladder_table.tag_configure("s12", background="#e8f1ff")

    ladder_hint = tk.StringVar()
    ttk.Label(right, textvariable=ladder_hint, padding=(3, 8), wraplength=380).pack(fill="x")

    def current_spec() -> ContainerSpec:
        residual_k = _parse_positive(values["residual_k"].get(), "residual k") if values["residual_k"].get().strip() else 0
        residual_group = (
            _parse_positive(values["residual_group"].get(), "residual weights / scale")
            if values["residual_group"].get().strip()
            else 0
        )
        residual_scale_bits = (
            _parse_positive(values["residual_scale_bits"].get(), "residual scale bits")
            if values["residual_scale_bits"].get().strip()
            else 0
        )
        return ContainerSpec(
            k=_parse_positive(values["k"].get(), "k"),
            weights_per_scale=_parse_positive(values["group"].get(), "weights / scale"),
            scale_bits=_parse_positive(values["scale_bits"].get(), "scale bits"),
            residual_k=residual_k,
            residual_weights_per_scale=residual_group,
            residual_scale_bits=residual_scale_bits,
        )

    def refresh_summary() -> None:
        stats = mixture_stats(parts)
        if not stats["weights"]:
            summary.set("Add a part to calculate the mixture.")
            return
        summary.set(
            f"{stats['weights']:,} weights | "
            f"ideal {stats['ideal_bpw']:.4f} bpw | "
            f"byte payload {stats['payload_bpw']:.4f} bpw | "
            f"{_format_bytes(stats['payload_bytes'])}"
        )

    def refresh_table() -> None:
        table.delete(*table.get_children())
        for index, part in enumerate(parts):
            stats = part["spec"].storage(part["weights"])
            table.insert(
                "",
                "end",
                iid=str(index),
                values=(
                    part["label"],
                    f"{part['weights']:,}",
                    part["spec"].format_name(),
                    f"{stats['ideal_bpw']:.4f}",
                    f"{stats['payload_bpw']:.4f}",
                    _format_bytes(stats["payload_bytes"]),
                ),
            )
        refresh_summary()

    def refresh_ladder(*_event) -> None:
        try:
            k = _parse_positive(values["k"].get(), "k")
            scale_bits = _parse_positive(values["scale_bits"].get(), "scale bits")
            rows = density_ladder(k, scale_bits)
        except ValueError as exc:
            ladder_hint.set(str(exc))
            return
        ladder_table.delete(*ladder_table.get_children())
        for row in rows:
            tags = ("s12",) if scale_bits == 12 and row["group"] in {15, 20} else ()
            ladder_table.insert(
                "",
                "end",
                iid=str(row["group"]),
                tags=tags,
                values=(
                    row["group"],
                    f"{row['scale_bpw']:.4f}",
                    f"{row['ideal_bpw']:.4f}",
                    f"{row['payload_bpw']:.4f}",
                ),
            )
        s12_note = " S12 g15 = 2.4 ideal bpw; g20 = 2.2 ideal bpw." if scale_bits == 12 and k == 1 else ""
        ladder_hint.set(f"k{k} with {scale_bits}-bit scales.{s12_note} Blue rows are the S12 choices.")

    def load_preset() -> None:
        spec = preset_by_name[preset_var.get()]
        values["k"].set(str(spec.k))
        values["group"].set(str(spec.weights_per_scale))
        values["scale_bits"].set(str(spec.scale_bits))
        values["residual_k"].set(str(spec.residual_k))
        values["residual_group"].set(str(spec.residual_weights_per_scale or ""))
        values["residual_scale_bits"].set(str(spec.residual_scale_bits or ""))
        refresh_ladder()

    def read_part() -> dict:
        return {
            "label": values["label"].get().strip() or "unnamed part",
            "weights": _parse_positive(values["weights"].get(), "weights"),
            "spec": current_spec(),
        }

    def add_part() -> None:
        try:
            parts.append(read_part())
        except ValueError as exc:
            messagebox.showerror("Tk-Integers", str(exc), parent=window)
            return
        refresh_table()

    def update_part() -> None:
        selected = table.selection()
        if not selected:
            add_part()
            return
        try:
            parts[int(selected[0])] = read_part()
        except ValueError as exc:
            messagebox.showerror("Tk-Integers", str(exc), parent=window)
            return
        refresh_table()

    def edit_selected(_event=None) -> None:
        selected = table.selection()
        if not selected:
            return
        part = parts[int(selected[0])]
        spec = part["spec"]
        values["label"].set(part["label"])
        values["weights"].set(str(part["weights"]))
        values["k"].set(str(spec.k))
        values["group"].set(str(spec.weights_per_scale))
        values["scale_bits"].set(str(spec.scale_bits))
        values["residual_k"].set(str(spec.residual_k))
        values["residual_group"].set(str(spec.residual_weights_per_scale or ""))
        values["residual_scale_bits"].set(str(spec.residual_scale_bits or ""))
        refresh_ladder()

    def remove_part() -> None:
        selected = table.selection()
        if selected:
            parts.pop(int(selected[0]))
            refresh_table()

    def export_mixture() -> None:
        if not parts:
            messagebox.showinfo("Tk-Integers", "Add at least one mixture part first.", parent=window)
            return
        path = filedialog.asksaveasfilename(
            parent=window,
            title="Export Tk-Integer mixture",
            defaultextension=".json",
            filetypes=[("Tk-Integer recipe", "*.json")],
        )
        if not path:
            return
        stats = mixture_stats(parts)
        payload = {
            "format": "tkintegers-mixture-v1",
            "parts": [
                {"label": p["label"], "weights": p["weights"], "spec": p["spec"].to_dict()}
                for p in parts
            ],
            "summary": {key: value for key, value in stats.items() if key != "parts"},
        }
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")
        messagebox.showinfo("Tk-Integers", f"Recipe exported to {path}", parent=window)

    ttk.Button(preset_row, text="Load preset", command=load_preset).pack(side="left")
    ttk.Button(buttons, text="Add part", command=add_part).pack(side="left")
    ttk.Button(buttons, text="Update selected", command=update_part).pack(side="left", padx=6)
    ttk.Button(buttons, text="Remove selected", command=remove_part).pack(side="left")
    ttk.Button(buttons, text="Export mixture", command=export_mixture).pack(side="right")
    table.bind("<<TreeviewSelect>>", edit_selected)
    for variable in (values["k"], values["scale_bits"]):
        variable.trace_add("write", refresh_ladder)

    def pump_conversion() -> None:
        try:
            while True:
                message = conversion_queue.get_nowait()
                if message[0] == "inspect":
                    rows = message[1]
                    model_table.delete(*model_table.get_children())
                    for row in rows:
                        model_table.insert(
                            "",
                            "end",
                            values=(
                                row["tensor"],
                                str(tuple(row["shape"])),
                                row["dtype"],
                                f"{row['params']:,}",
                                Path(row["source_shard"]).name,
                            ),
                        )
                    conversion_progress["value"] = 0
                    conversion_progress["maximum"] = max(1, len(rows))
                    conversion_vars["status"].set(f"Inspected {len(rows):,} tensors. Ready to convert.")
                elif message[0] == "progress":
                    _, index, total, name = message
                    conversion_progress["maximum"] = total
                    conversion_progress["value"] = index
                    conversion_vars["status"].set(f"Converted {index:,}/{total:,}: {name}")
                elif message[0] == "converted":
                    manifest = message[1]
                    summary_data = manifest["summary"]
                    conversion_progress["value"] = conversion_progress["maximum"]
                    conversion_vars["status"].set(
                        f"Finished: {summary_data['quantized_tensors']:,} quantized tensors, "
                        f"{summary_data['raw_tensors']:,} raw tensors. "
                        f"Manifest: {Path(conversion_vars['output'].get()) / 'manifest.json'}"
                    )
                elif message[0] == "error":
                    conversion_progress["value"] = 0
                    conversion_vars["status"].set(f"ERROR: {message[1]}")
        except queue.Empty:
            pass
        window.after(150, pump_conversion)

    refresh_ladder()
    add_part()
    pump_conversion()
    if parent is None:
        window.mainloop()
    return window


if __name__ == "__main__":
    launch_tkintegers_ui()
