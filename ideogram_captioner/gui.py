from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import tkinter as tk
from datetime import datetime, timezone
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from PIL import Image, ImageTk, UnidentifiedImageError

from .schema import (
    CAPTION_EXTENSIONS,
    COMMON_MEDIA,
    ELEMENT_TYPES,
    STYLE_MODES,
    default_caption,
    normalize_bbox,
    normalize_caption,
    parse_palette_text,
    palette_to_text,
)
from .store import CaptionStore
from .llm_captioning import (
    AutoCaptionError,
    CaptioningSettings,
    ModelJsonError,
    add_bboxes_to_caption,
    build_llama_server_command,
    default_models_dir,
    default_prompts_path,
    default_profiles_path,
    ensure_model_assets,
    find_llama_server,
    format_server_command,
    generate_json_from_image,
    generate_json_from_text,
    generate_plain_caption,
    is_server_ready,
    load_settings,
    profile_seed_data,
    profile_id_from_label,
    profile_label_from_id,
    profile_labels,
    profiles_for_task,
    runtime_config_for_task,
    save_settings,
    server_model_ids,
    start_server_process,
    stop_server_process,
    write_default_prompts,
)


try:
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - old Pillow fallback
    RESAMPLE = Image.LANCZOS

SHIFT_MASK = 0x0001
CORNER_CURSORS = {
    "nw": ("size_nw_se", "top_left_corner", "sizing"),
    "se": ("size_nw_se", "bottom_right_corner", "sizing"),
    "ne": ("size_ne_sw", "top_right_corner", "sizing"),
    "sw": ("size_ne_sw", "bottom_left_corner", "sizing"),
}
EDGE_CURSORS = {
    "n": ("sb_v_double_arrow", "size_ns", "top_side", "bottom_side", "sizing"),
    "s": ("sb_v_double_arrow", "size_ns", "top_side", "bottom_side", "sizing"),
    "e": ("sb_h_double_arrow", "size_we", "left_side", "right_side", "sizing"),
    "w": ("sb_h_double_arrow", "size_we", "left_side", "right_side", "sizing"),
}
MOVE_CURSORS = ("fleur", "size", "hand2")
RESIZE_HIT_TOLERANCE = 8
OBJ_BOX_COLORS = (
    "#4da3ff",
    "#3ddc97",
    "#ffd166",
    "#f15bb5",
    "#9bff78",
    "#00d4ff",
    "#ff8c42",
    "#c084fc",
)
TEXT_BOX_COLOR = "#ff9f1a"
BOX_LABEL_TEXT_COLOR = "#06101f"
ELEMENT_ROW_BASE_COLOR = "#1f2430"
ELEMENT_ROW_TEXT_COLOR = "#f8fbff"
ELEMENT_ROW_TEXT_TEXT_COLOR = "#fff2df"
ELEMENT_ROW_TINT = 0.22
CONTROL_MASK = 0x0004
ORIGINAL_NONE = "none"
ORIGINAL_FILE_EXTENSIONS = (".txt", ".original", ORIGINAL_NONE)
IMAGE_SORT_NAME_ASC = "Name A-Z"
IMAGE_SORT_NAME_DESC = "Name Z-A"
IMAGE_SORT_MODIFIED_NEWEST = "Modified newest"
IMAGE_SORT_MODIFIED_OLDEST = "Modified oldest"
IMAGE_SORT_CAPTION_MISSING = "Caption missing first"
IMAGE_SORT_ORIGINAL_MISSING = "Original missing first"
IMAGE_SORT_BBOX_MISSING = "Missing bboxes only"
IMAGE_SORT_FAILED = "Failed captions only"
IMAGE_SORT_OPTIONS = (
    IMAGE_SORT_NAME_ASC,
    IMAGE_SORT_NAME_DESC,
    IMAGE_SORT_MODIFIED_NEWEST,
    IMAGE_SORT_MODIFIED_OLDEST,
    IMAGE_SORT_CAPTION_MISSING,
    IMAGE_SORT_ORIGINAL_MISSING,
    IMAGE_SORT_BBOX_MISSING,
    IMAGE_SORT_FAILED,
)
RUNTIME_LOCAL_LABEL = "Local llama.cpp (recommended)"
RUNTIME_EXISTING_LABEL = "Connect to existing server"
RUNTIME_CUSTOM_LABEL = "Custom start commands"
RUNTIME_LABELS = (RUNTIME_LOCAL_LABEL, RUNTIME_EXISTING_LABEL, RUNTIME_CUSTOM_LABEL)
RUNTIME_MODE_TO_LABEL = {
    "local": RUNTIME_LOCAL_LABEL,
    "existing": RUNTIME_EXISTING_LABEL,
    "custom": RUNTIME_CUSTOM_LABEL,
}
RUNTIME_LABEL_TO_MODE = {label: mode for mode, label in RUNTIME_MODE_TO_LABEL.items()}


def mix_hex_color(color: str, base: str, amount: float) -> str:
    color = color.lstrip("#")
    base = base.lstrip("#")
    blended = []
    for index in range(0, 6, 2):
        foreground_part = int(color[index : index + 2], 16)
        base_part = int(base[index : index + 2], 16)
        blended_part = round(base_part + (foreground_part - base_part) * amount)
        blended.append(max(0, min(255, blended_part)))
    return "#{:02x}{:02x}{:02x}".format(*blended)


class ScrollFrame(ttk.Frame):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self.canvas = tk.Canvas(self, bg="#171a21", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.interior = ttk.Frame(self.canvas, padding=(10, 4))

        self.window_id = self.canvas.create_window((0, 0), window=self.interior, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.interior.bind("<Configure>", self._on_interior_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_interior_configure(self, _event: tk.Event) -> None:
        self._update_scroll_region()

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)
        self._update_scroll_region()

    def _update_scroll_region(self) -> None:
        bbox = self.canvas.bbox(self.window_id)
        if bbox is None:
            self.canvas.configure(scrollregion=(0, 0, self.canvas.winfo_width(), self.canvas.winfo_height()))
            return
        _left, _top, right, bottom = bbox
        right = max(right, self.canvas.winfo_width())
        bottom = max(bottom, self.canvas.winfo_height())
        self.canvas.configure(scrollregion=(0, 0, right, bottom))

    def _on_mousewheel(self, event: tk.Event) -> None:
        if self.winfo_containing(event.x_root, event.y_root) is None:
            return
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def scroll_to_top(self) -> None:
        self._update_scroll_region()
        self.canvas.yview_moveto(0.0)


class CaptioningPreferencesDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, settings: CaptioningSettings) -> None:
        super().__init__(parent)
        self.title("Auto Captioning Preferences")
        self.transient(parent)
        self.result: CaptioningSettings | None = None

        self.base_url_var = tk.StringVar(value=settings.base_url)
        self.api_key_var = tk.StringVar(value=settings.api_key)
        self.hf_token_var = tk.StringVar(value=settings.hf_token)
        self.models_dir_var = tk.StringVar(value=settings.models_dir or str(default_models_dir()))

        self.caption_profile_var = tk.StringVar(value=profile_label_from_id("caption", settings.caption_profile_id))
        self.caption_model_var = tk.StringVar(value=settings.caption_model)
        self.caption_hf_repo_var = tk.StringVar(value=settings.caption_hf_repo)
        self.caption_model_file_var = tk.StringVar(value=settings.caption_model_filename)
        self.caption_mmproj_file_var = tk.StringVar(value=settings.caption_mmproj_filename)
        self.caption_local_model_path_var = tk.StringVar(value=settings.caption_local_model_path)
        self.caption_local_mmproj_path_var = tk.StringVar(value=settings.caption_local_mmproj_path)

        self.bbox_profile_var = tk.StringVar(value=profile_label_from_id("bbox", settings.bbox_profile_id))
        self.bbox_model_var = tk.StringVar(value=settings.bbox_model)
        self.bbox_hf_repo_var = tk.StringVar(value=settings.bbox_hf_repo)
        self.bbox_model_file_var = tk.StringVar(value=settings.bbox_model_filename)
        self.bbox_mmproj_file_var = tk.StringVar(value=settings.bbox_mmproj_filename)
        self.bbox_local_model_path_var = tk.StringVar(value=settings.bbox_local_model_path)
        self.bbox_local_mmproj_path_var = tk.StringVar(value=settings.bbox_local_mmproj_path)

        self.add_bboxes_var = tk.BooleanVar(value=settings.add_bboxes_after_json)
        self.overwrite_bboxes_var = tk.BooleanVar(value=settings.overwrite_bboxes)
        self.filter_bbox_targets_var = tk.BooleanVar(value=settings.filter_bbox_targets)
        self.creative_var = tk.BooleanVar(value=settings.creative_json)
        self.disable_thinking_var = tk.BooleanVar(value=settings.disable_thinking)
        self.vision_format_var = tk.StringVar(value=settings.vision_image_format)
        self.max_caption_tokens_var = tk.StringVar(value=str(settings.max_tokens_caption))
        self.max_json_tokens_var = tk.StringVar(value=str(settings.max_tokens_json))
        self.max_bbox_tokens_var = tk.StringVar(value=str(settings.max_tokens_bboxes))
        self.context_chars_var = tk.StringVar(value=str(settings.context_chars))
        self.max_targets_var = tk.StringVar(value=str(settings.max_targets_per_call))

        self.runtime_mode_var = tk.StringVar(value=RUNTIME_MODE_TO_LABEL.get(settings.server_start_mode, RUNTIME_LOCAL_LABEL))
        self.auto_start_var = tk.BooleanVar(value=settings.auto_start_server)
        default_llama_server = find_llama_server()
        llama_server_path = settings.llama_server_path or (str(default_llama_server) if default_llama_server else "")
        self.llama_server_path_var = tk.StringVar(value=llama_server_path)
        self.llama_context_var = tk.StringVar(value=str(settings.llama_context))
        self.llama_gpu_layers_var = tk.StringVar(value=str(settings.llama_gpu_layers))
        self.llama_batch_var = tk.StringVar(value=str(settings.llama_batch))
        self.llama_ubatch_var = tk.StringVar(value=str(settings.llama_ubatch))
        self.llama_threads_var = tk.StringVar(value=str(settings.llama_threads))
        self.llama_extra_args_var = tk.StringVar(value=settings.llama_extra_args)
        self.llama_reasoning_budget_var = tk.StringVar(value=str(settings.llama_reasoning_budget))
        self.caption_server_command_var = tk.StringVar(value=settings.caption_server_command)
        self.bbox_server_command_var = tk.StringVar(value=settings.bbox_server_command)
        self.server_timeout_var = tk.StringVar(value=str(settings.server_startup_timeout))
        self.stop_server_var = tk.BooleanVar(value=settings.stop_server_after_job)
        self.custom_profile_frames: dict[str, dict[str, ttk.Frame]] = {}

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.grab_set()
        self.update_idletasks()
        self.geometry(f"820x760+{parent.winfo_rootx() + 80}+{parent.winfo_rooty() + 50}")

    def _build_ui(self) -> None:
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        notebook = ttk.Notebook(self)
        notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        connection = ttk.Frame(notebook, padding=10)
        models = ttk.Frame(notebook, padding=10)
        pipeline = ttk.Frame(notebook, padding=10)
        notebook.add(connection, text="Connection")
        notebook.add(models, text="Models")
        notebook.add(pipeline, text="Pipeline")

        self._build_connection_tab(connection)
        self._build_models_tab(models)
        self._build_pipeline_tab(pipeline)

        buttons = ttk.Frame(self, padding=(10, 0, 10, 10))
        buttons.grid(row=1, column=0, sticky="ew")
        buttons.columnconfigure(0, weight=1)
        ttk.Button(buttons, text="Cancel", command=self.destroy).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(buttons, text="Save", command=self._save, style="Accent.TButton").grid(row=0, column=2)

    def _add_entry(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        show: str | None = None,
    ) -> int:
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        entry = ttk.Entry(parent, textvariable=variable, show=show or "")
        entry.grid(row=row, column=1, sticky="ew", pady=4)
        return row + 1

    def _build_connection_tab(self, parent: ttk.Frame) -> None:
        row = 0
        parent.columnconfigure(1, weight=1)

        ttk.Label(parent, text="Runtime").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        runtime_combo = ttk.Combobox(parent, textvariable=self.runtime_mode_var, values=RUNTIME_LABELS, state="readonly")
        runtime_combo.grid(row=row, column=1, sticky="ew", pady=4)
        runtime_combo.bind("<<ComboboxSelected>>", lambda _event: self._on_runtime_mode_changed())
        row += 1

        ttk.Checkbutton(parent, text="Start runtime automatically before jobs", variable=self.auto_start_var).grid(
            row=row,
            column=0,
            columnspan=3,
            sticky="w",
            pady=4,
        )
        row += 1
        ttk.Checkbutton(parent, text="Stop auto-started runtime after job", variable=self.stop_server_var).grid(
            row=row,
            column=0,
            columnspan=3,
            sticky="w",
            pady=4,
        )
        row += 1
        row = self._add_entry(parent, row, "OpenAI-compatible base URL", self.base_url_var)
        row = self._add_entry(parent, row, "API key", self.api_key_var, show="*")
        row = self._add_entry(parent, row, "Hugging Face token", self.hf_token_var, show="*")

        ttk.Label(parent, text="Models folder").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(parent, textvariable=self.models_dir_var).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(parent, text="Browse", command=self._browse_models_dir).grid(row=row, column=2, padx=(6, 0), pady=4)
        row += 1

        ttk.Separator(parent).grid(row=row, column=0, columnspan=3, sticky="ew", pady=12)
        row += 1

        ttk.Label(parent, text="Local llama.cpp", style="Section.TLabel").grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1
        ttk.Label(parent, text="llama-server.exe").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(parent, textvariable=self.llama_server_path_var).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(parent, text="Browse", command=self._browse_llama_server).grid(row=row, column=2, padx=(6, 0), pady=4)
        row += 1
        row = self._add_entry(parent, row, "Context size", self.llama_context_var)
        row = self._add_entry(parent, row, "GPU layers", self.llama_gpu_layers_var)
        row = self._add_entry(parent, row, "Batch size", self.llama_batch_var)
        row = self._add_entry(parent, row, "Ubatch size", self.llama_ubatch_var)
        row = self._add_entry(parent, row, "CPU threads (0 = auto)", self.llama_threads_var)
        row = self._add_entry(parent, row, "Thinking token budget (-1 = unrestricted)", self.llama_reasoning_budget_var)
        row = self._add_entry(parent, row, "Extra llama args", self.llama_extra_args_var)
        row = self._add_entry(parent, row, "Startup timeout seconds", self.server_timeout_var)

        ttk.Separator(parent).grid(row=row, column=0, columnspan=3, sticky="ew", pady=12)
        row += 1
        ttk.Label(parent, text="Custom start commands", style="Section.TLabel").grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 6))
        row += 1
        row = self._add_entry(parent, row, "Caption server command", self.caption_server_command_var)
        row = self._add_entry(parent, row, "BBox server command", self.bbox_server_command_var)

        hint = (
            "Server command placeholders: {model_path}, {mmproj_path}, {models_dir}, "
            "{api_model}, {base_url}."
        )
        ttk.Label(parent, text=hint, wraplength=680).grid(row=row, column=0, columnspan=3, sticky="w", pady=(8, 0))
        row += 1
        ttk.Button(parent, text="Test Server", command=self._test_server).grid(row=row, column=0, sticky="w", pady=(12, 0))
        self._on_runtime_mode_changed()

    def _build_models_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        row = 0
        ttk.Label(parent, text="Caption / JSON model", style="Section.TLabel").grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(0, 8),
        )
        row += 1
        row = self._add_profile_controls(
            parent,
            row,
            "caption",
            self.caption_profile_var,
            self.caption_model_var,
            self.caption_hf_repo_var,
            self.caption_model_file_var,
            self.caption_mmproj_file_var,
            self.caption_local_model_path_var,
            self.caption_local_mmproj_path_var,
        )

        ttk.Separator(parent).grid(row=row, column=0, columnspan=2, sticky="ew", pady=14)
        row += 1
        ttk.Label(parent, text="BBox VLM", style="Section.TLabel").grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(0, 8),
        )
        row += 1
        row = self._add_profile_controls(
            parent,
            row,
            "bbox",
            self.bbox_profile_var,
            self.bbox_model_var,
            self.bbox_hf_repo_var,
            self.bbox_model_file_var,
            self.bbox_mmproj_file_var,
            self.bbox_local_model_path_var,
            self.bbox_local_mmproj_path_var,
        )
        ttk.Separator(parent).grid(row=row, column=0, columnspan=2, sticky="ew", pady=14)
        row += 1
        ttk.Button(parent, text="Open Profiles File", command=self._open_profiles_file).grid(row=row, column=0, sticky="w")

    def _add_profile_controls(
        self,
        parent: ttk.Frame,
        row: int,
        task: str,
        profile_var: tk.StringVar,
        model_var: tk.StringVar,
        repo_var: tk.StringVar,
        model_file_var: tk.StringVar,
        mmproj_file_var: tk.StringVar,
        local_model_path_var: tk.StringVar,
        local_mmproj_path_var: tk.StringVar,
    ) -> int:
        ttk.Label(parent, text="Profile").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        combo = ttk.Combobox(parent, textvariable=profile_var, values=profile_labels(task), state="readonly")
        combo.grid(row=row, column=1, sticky="ew", pady=4)
        combo.bind("<<ComboboxSelected>>", lambda _event, t=task: self._on_profile_changed(t))
        row += 1
        row = self._add_entry(parent, row, "API model name", model_var)

        hf_frame = ttk.Frame(parent)
        hf_frame.grid(row=row, column=0, columnspan=2, sticky="ew")
        hf_row = 0
        hf_row = self._add_entry(hf_frame, hf_row, "Custom HF repo", repo_var)
        hf_row = self._add_entry(hf_frame, hf_row, "Custom model file", model_file_var)
        self._add_entry(hf_frame, hf_row, "Custom mmproj file", mmproj_file_var)

        local_frame = ttk.Frame(parent)
        local_frame.grid(row=row, column=0, columnspan=2, sticky="ew")
        self._add_file_entry(local_frame, 0, "Local model GGUF", local_model_path_var, "Choose local GGUF model")
        self._add_file_entry(local_frame, 1, "Local mmproj file", local_mmproj_path_var, "Choose local mmproj file")

        self.custom_profile_frames[task] = {"hf": hf_frame, "local": local_frame}
        self._update_custom_profile_visibility(task)
        return row + 1

    def _add_file_entry(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, title: str) -> None:
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(parent, text="Browse", command=lambda: self._browse_model_file(variable, title)).grid(
            row=row,
            column=2,
            sticky="e",
            padx=(6, 0),
            pady=4,
        )

    def _browse_model_file(self, variable: tk.StringVar, title: str) -> None:
        initial = variable.get().strip()
        initial_dir = str(Path(initial).parent) if initial else self.models_dir_var.get() or str(default_models_dir())
        path = filedialog.askopenfilename(
            title=title,
            initialdir=initial_dir,
            filetypes=(("GGUF files", "*.gguf"), ("All files", "*.*")),
        )
        if path:
            variable.set(path)

    def _open_profiles_file(self) -> None:
        path = default_profiles_path()
        try:
            if not path.exists():
                path.write_text(json.dumps(profile_seed_data(), indent=2), encoding="utf-8")
            if hasattr(os, "startfile"):
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except OSError as exc:
            messagebox.showerror("Could not open profiles file", str(exc))

    def _open_prompts_file(self) -> None:
        path = default_prompts_path()
        try:
            if not path.exists():
                write_default_prompts(path)
            if hasattr(os, "startfile"):
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except OSError as exc:
            messagebox.showerror("Could not open prompts file", str(exc))

    def _build_pipeline_tab(self, parent: ttk.Frame) -> None:
        row = 0
        for text, variable in (
            ("Add/redo bboxes automatically after JSON generation", self.add_bboxes_var),
            ("Overwrite existing bboxes", self.overwrite_bboxes_var),
            ("Skip vague/ambient bbox targets", self.filter_bbox_targets_var),
            ("Creative text-to-JSON mode", self.creative_var),
            ("Disable model thinking/reasoning", self.disable_thinking_var),
        ):
            ttk.Checkbutton(parent, text=text, variable=variable).grid(row=row, column=0, columnspan=2, sticky="w", pady=4)
            row += 1

        ttk.Separator(parent).grid(row=row, column=0, columnspan=2, sticky="ew", pady=12)
        row += 1
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text="Vision image format").grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Combobox(
            parent,
            textvariable=self.vision_format_var,
            values=("auto", "original", "png", "jpeg", "jpg"),
            state="readonly",
        ).grid(row=row, column=1, sticky="ew", pady=4)
        row += 1

        for label, variable in (
            ("Plain caption max tokens", self.max_caption_tokens_var),
            ("JSON max tokens", self.max_json_tokens_var),
            ("BBox max tokens", self.max_bbox_tokens_var),
            ("BBox context chars", self.context_chars_var),
            ("Max bbox targets per call", self.max_targets_var),
        ):
            row = self._add_entry(parent, row, label, variable)

        ttk.Separator(parent).grid(row=row, column=0, columnspan=2, sticky="ew", pady=14)
        row += 1
        ttk.Button(parent, text="Open Prompts Folder", command=self._open_prompts_file).grid(row=row, column=0, sticky="w")

    def _browse_models_dir(self) -> None:
        folder = filedialog.askdirectory(title="Choose models folder", initialdir=self.models_dir_var.get() or str(default_models_dir()))
        if folder:
            self.models_dir_var.set(folder)

    def _browse_llama_server(self) -> None:
        initial = self.llama_server_path_var.get().strip()
        initial_dir = str(Path(initial).parent) if initial else str(default_models_dir().parent)
        path = filedialog.askopenfilename(
            title="Choose llama-server executable",
            initialdir=initial_dir,
            filetypes=(("llama-server", "llama-server.exe llama-server"), ("Executables", "*.exe"), ("All files", "*.*")),
        )
        if path:
            self.llama_server_path_var.set(path)

    def _runtime_mode(self) -> str:
        return RUNTIME_LABEL_TO_MODE.get(self.runtime_mode_var.get(), "local")

    def _on_runtime_mode_changed(self) -> None:
        if self._runtime_mode() == "existing":
            self.auto_start_var.set(False)

    def _profile_for_label(self, task: str, label: str):
        for profile in profiles_for_task(task):
            if profile.label == label:
                return profile
        return profiles_for_task(task)[0]

    def _update_custom_profile_visibility(self, task: str) -> None:
        frames = self.custom_profile_frames.get(task)
        if frames is None:
            return
        profile_var = self.bbox_profile_var if task == "bbox" else self.caption_profile_var
        profile = self._profile_for_label(task, profile_var.get())
        if profile.kind == "custom_hf":
            frames["hf"].grid()
        else:
            frames["hf"].grid_remove()
        if profile.kind == "custom_local":
            frames["local"].grid()
        else:
            frames["local"].grid_remove()

    def _on_profile_changed(self, task: str) -> None:
        if task == "bbox":
            profile = self._profile_for_label("bbox", self.bbox_profile_var.get())
            if profile.kind != "custom_hf":
                self.bbox_model_var.set(profile.api_model)
            if profile.kind == "server":
                self.runtime_mode_var.set(RUNTIME_EXISTING_LABEL)
                self._on_runtime_mode_changed()
            self._update_custom_profile_visibility("bbox")
            return
        profile = self._profile_for_label("caption", self.caption_profile_var.get())
        if profile.kind != "custom_hf":
            self.caption_model_var.set(profile.api_model)
        if profile.kind == "server":
            self.runtime_mode_var.set(RUNTIME_EXISTING_LABEL)
            self._on_runtime_mode_changed()
        self._update_custom_profile_visibility("caption")

    def _test_server(self) -> None:
        if is_server_ready(self.base_url_var.get().strip(), self.api_key_var.get().strip(), timeout=5.0):
            messagebox.showinfo("Server ready", "The OpenAI-compatible endpoint responded to /models.")
        else:
            messagebox.showwarning("Server unavailable", "The endpoint did not respond to /models.")

    def _parse_int(self, variable: tk.StringVar, label: str, minimum: int = 0) -> int:
        try:
            value = int(variable.get().strip())
        except ValueError as exc:
            raise ValueError(f"{label} must be an integer.") from exc
        if value < minimum:
            raise ValueError(f"{label} must be at least {minimum}.")
        return value

    def _parse_float(self, variable: tk.StringVar, label: str, minimum: float = 0.0) -> float:
        try:
            value = float(variable.get().strip())
        except ValueError as exc:
            raise ValueError(f"{label} must be a number.") from exc
        if value < minimum:
            raise ValueError(f"{label} must be at least {minimum}.")
        return value

    def _save(self) -> None:
        try:
            settings = CaptioningSettings(
                base_url=self.base_url_var.get().strip() or "http://127.0.0.1:8000/v1",
                api_key=self.api_key_var.get().strip() or "dummy",
                hf_token=self.hf_token_var.get().strip(),
                models_dir=self.models_dir_var.get().strip() or str(default_models_dir()),
                caption_profile_id=profile_id_from_label("caption", self.caption_profile_var.get()),
                caption_model=self.caption_model_var.get().strip(),
                caption_hf_repo=self.caption_hf_repo_var.get().strip(),
                caption_model_filename=self.caption_model_file_var.get().strip(),
                caption_mmproj_filename=self.caption_mmproj_file_var.get().strip(),
                caption_local_model_path=self.caption_local_model_path_var.get().strip(),
                caption_local_mmproj_path=self.caption_local_mmproj_path_var.get().strip(),
                bbox_profile_id=profile_id_from_label("bbox", self.bbox_profile_var.get()),
                bbox_model=self.bbox_model_var.get().strip(),
                bbox_hf_repo=self.bbox_hf_repo_var.get().strip(),
                bbox_model_filename=self.bbox_model_file_var.get().strip(),
                bbox_mmproj_filename=self.bbox_mmproj_file_var.get().strip(),
                bbox_local_model_path=self.bbox_local_model_path_var.get().strip(),
                bbox_local_mmproj_path=self.bbox_local_mmproj_path_var.get().strip(),
                add_bboxes_after_json=self.add_bboxes_var.get(),
                overwrite_bboxes=self.overwrite_bboxes_var.get(),
                filter_bbox_targets=self.filter_bbox_targets_var.get(),
                use_caption_model_for_bboxes=False,
                creative_json=self.creative_var.get(),
                disable_thinking=self.disable_thinking_var.get(),
                vision_image_format=self.vision_format_var.get(),
                max_tokens_caption=self._parse_int(self.max_caption_tokens_var, "Plain caption max tokens", 1),
                max_tokens_json=self._parse_int(self.max_json_tokens_var, "JSON max tokens", 1),
                max_tokens_bboxes=self._parse_int(self.max_bbox_tokens_var, "BBox max tokens", 1),
                context_chars=self._parse_int(self.context_chars_var, "BBox context chars", 0),
                max_targets_per_call=self._parse_int(self.max_targets_var, "Max bbox targets per call", 0),
                server_start_mode=self._runtime_mode(),
                auto_start_server=self.auto_start_var.get() and self._runtime_mode() != "existing",
                llama_server_path=self.llama_server_path_var.get().strip(),
                llama_context=self._parse_int(self.llama_context_var, "Context size", 512),
                llama_gpu_layers=self._parse_int(self.llama_gpu_layers_var, "GPU layers", 0),
                llama_batch=self._parse_int(self.llama_batch_var, "Batch size", 1),
                llama_ubatch=self._parse_int(self.llama_ubatch_var, "Ubatch size", 1),
                llama_threads=self._parse_int(self.llama_threads_var, "CPU threads", 0),
                llama_reasoning_budget=self._parse_int(
                    self.llama_reasoning_budget_var,
                    "Thinking token budget",
                    -1,
                ),
                llama_extra_args=self.llama_extra_args_var.get().strip(),
                caption_server_command=self.caption_server_command_var.get().strip(),
                bbox_server_command=self.bbox_server_command_var.get().strip(),
                server_startup_timeout=self._parse_float(self.server_timeout_var, "Startup timeout seconds", 1.0),
                stop_server_after_job=self.stop_server_var.get(),
            )
        except ValueError as exc:
            messagebox.showerror("Invalid preferences", str(exc))
            return

        self.result = settings
        self.destroy()


class CaptionEditorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Ideogram Captioner")
        self.geometry("1500x950")
        try:
            self.state("zoomed")
        except tk.TclError:
            pass

        self.folder: Path | None = None
        self.store: CaptionStore | None = None
        self.all_images: list[Path] = []
        self.images: list[Path] = []
        self.current_index = -1
        self.current_caption: dict[str, Any] = default_caption()
        self.selected_element_index: int | None = None
        self.pil_image: Image.Image | None = None
        self.tk_image: ImageTk.PhotoImage | None = None
        self.canvas_image_id: int | None = None
        self.image_render_key: tuple[int, int, int] | None = None
        self.image_bounds = (0.0, 0.0, 1.0, 1.0)
        self.dirty = False
        self.autosave_job: str | None = None
        self.original_dirty = False
        self.original_autosave_job: str | None = None
        self.current_original_path: Path | None = None
        self.loading_form = False
        self.loading_element = False
        self.loading_list = False
        self.loading_original = False
        self.draw_mode = tk.BooleanVar(value=False)
        self.drag_state: dict[str, Any] | None = None
        self.eyedrop_target: dict[str, Any] | None = None
        self.palette_selected_index: dict[str, int | None] = {"style": None, "element": None}
        self.current_canvas_cursor = ""
        self.fullscreen = False
        self.drag_threshold = 4
        self.captioning_settings = load_settings()
        self.ai_worker: threading.Thread | None = None
        self.ai_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self.ai_undo_stack: list[dict[str, Any]] = []
        self.ai_buttons: list[ttk.Button] = []
        self.ai_cancel_button: ttk.Button | None = None
        self.ai_cancel_event: threading.Event | None = None
        self.ai_server_process: subprocess.Popen | None = None
        self.ai_server_command: str | None = None
        self.ai_server_phase: str | None = None
        self.pending_image_select_job: str | None = None
        self.last_image_click_index: int | None = None
        self.image_selection_anchor_path: Path | None = None

        self.caption_extension_var = tk.StringVar(value=".json")
        self.original_extension_var = tk.StringVar(value=".txt")
        self.image_sort_var = tk.StringVar(value=IMAGE_SORT_NAME_ASC)
        self.status_var = tk.StringVar(value="Open a folder to begin.")
        self.ai_progress_var = tk.DoubleVar(value=0.0)
        self.ai_progress_text_var = tk.StringVar(value="")
        self.original_status_var = tk.StringVar(value="Original: .txt")
        self.image_title_var = tk.StringVar(value="No image loaded")
        self.style_mode_var = tk.StringVar(value="photo")
        self.medium_var = tk.StringVar(value="photograph")
        self.palette_var = tk.StringVar(value="")
        self.element_type_var = tk.StringVar(value="obj")
        self.element_palette_var = tk.StringVar(value="")
        self.bbox_vars = {
            "y1": tk.StringVar(value=""),
            "x1": tk.StringVar(value=""),
            "y2": tk.StringVar(value=""),
            "x2": tk.StringVar(value=""),
        }

        self._configure_theme()
        self._build_ui()
        self._bind_events()

    def _configure_theme(self) -> None:
        self.configure(bg="#111318")
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background="#171a21")
        style.configure("Toolbar.TFrame", background="#0d0f14")
        style.configure("Panel.TFrame", background="#171a21")
        style.configure("TLabel", background="#171a21", foreground="#d9dee9")
        style.configure("Section.TLabel", font=("Segoe UI", 10, "bold"), foreground="#ffffff")
        style.configure("Status.TLabel", background="#0d0f14", foreground="#9ba5b7")
        style.configure("TButton", padding=(10, 6), background="#252b36", foreground="#edf2ff", borderwidth=0)
        style.map("TButton", background=[("active", "#30394a"), ("pressed", "#1d2430")])
        style.configure("Accent.TButton", background="#2f6fed", foreground="#ffffff")
        style.map("Accent.TButton", background=[("active", "#407dff"), ("pressed", "#255fcf")])
        style.configure("TCheckbutton", background="#171a21", foreground="#d9dee9")
        style.configure("TEntry", fieldbackground="#222733", foreground="#f2f5fb", insertcolor="#ffffff", bordercolor="#333b4b")
        style.configure("TCombobox", fieldbackground="#222733", foreground="#f2f5fb", arrowcolor="#d9dee9", bordercolor="#333b4b")
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", "#222733"), ("!disabled", "#222733")],
            foreground=[("readonly", "#f2f5fb"), ("!disabled", "#f2f5fb")],
            background=[("readonly", "#222733"), ("!disabled", "#222733")],
            selectbackground=[("readonly", "#315fbd"), ("!disabled", "#315fbd")],
            selectforeground=[("readonly", "#ffffff"), ("!disabled", "#ffffff")],
        )
        self.option_add("*TCombobox*Listbox.background", "#1f2430")
        self.option_add("*TCombobox*Listbox.foreground", "#f2f5fb")
        self.option_add("*TCombobox*Listbox.selectBackground", "#315fbd")
        self.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        style.configure("TSpinbox", fieldbackground="#222733", foreground="#f2f5fb", arrowcolor="#d9dee9", bordercolor="#333b4b")
        style.configure(
            "Treeview",
            rowheight=26,
            background="#1f2430",
            fieldbackground="#1f2430",
            foreground="#e8edf7",
            borderwidth=0,
        )
        style.map("Treeview", background=[("selected", "#315fbd")], foreground=[("selected", "#ffffff")])
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"), background="#252b36", foreground="#d9dee9")

    def _build_ui(self) -> None:
        self.rowconfigure(2, weight=1)
        self.columnconfigure(0, weight=1)

        toolbar = ttk.Frame(self, style="Toolbar.TFrame", padding=(8, 6))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(10, weight=1)

        ttk.Button(toolbar, text="Open Folder", command=self.open_folder_dialog).grid(row=0, column=0, padx=(0, 8))
        ttk.Label(toolbar, text="Caption files", style="Status.TLabel").grid(row=0, column=1, padx=(0, 6))
        self.extension_combo = ttk.Combobox(
            toolbar,
            textvariable=self.caption_extension_var,
            values=CAPTION_EXTENSIONS,
            state="readonly",
            width=10,
        )
        self.extension_combo.grid(row=0, column=2, padx=(0, 8))
        ttk.Label(toolbar, text="Original files", style="Status.TLabel").grid(row=0, column=3, padx=(0, 6))
        self.original_extension_combo = ttk.Combobox(
            toolbar,
            textvariable=self.original_extension_var,
            values=ORIGINAL_FILE_EXTENSIONS,
            state="readonly",
            width=10,
        )
        self.original_extension_combo.grid(row=0, column=4, padx=(0, 8))
        ttk.Button(toolbar, text="Save", command=self.save_current, style="Accent.TButton").grid(row=0, column=5, padx=(0, 8))
        ttk.Button(toolbar, text="Copy to edit", command=self.copy_current_image_to_edit).grid(row=0, column=6, padx=(0, 8))
        ttk.Button(toolbar, text="Previous", command=self.previous_image).grid(row=0, column=7, padx=(0, 4))
        ttk.Button(toolbar, text="Next", command=self.next_image).grid(row=0, column=8, padx=(0, 8))
        ttk.Button(toolbar, text="Preferences", command=self.open_captioning_preferences).grid(row=0, column=9, padx=(0, 8))
        ttk.Label(toolbar, textvariable=self.status_var, style="Status.TLabel", anchor="e").grid(row=0, column=10, sticky="ew")

        self.ai_progress_frame = ttk.Frame(self, style="Toolbar.TFrame", padding=(8, 0, 8, 6))
        self.ai_progress_frame.grid(row=1, column=0, sticky="ew")
        self.ai_progress_frame.columnconfigure(1, weight=1)
        ttk.Label(self.ai_progress_frame, textvariable=self.ai_progress_text_var, style="Status.TLabel", width=26).grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        self.ai_progress_bar = ttk.Progressbar(
            self.ai_progress_frame,
            variable=self.ai_progress_var,
            mode="determinate",
            maximum=1,
        )
        self.ai_progress_bar.grid(row=0, column=1, sticky="ew")
        self.ai_progress_frame.grid_remove()

        self.paned = ttk.PanedWindow(self, orient="horizontal")
        self.paned.grid(row=2, column=0, sticky="nsew")

        list_frame = ttk.Frame(self.paned, width=180, padding=(10, 10), style="Panel.TFrame")
        list_frame.rowconfigure(2, weight=1)
        list_frame.columnconfigure(0, weight=1)
        ttk.Label(list_frame, text="Images", style="Section.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        sort_frame = ttk.Frame(list_frame)
        sort_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        sort_frame.columnconfigure(1, weight=1)
        ttk.Label(sort_frame, text="Sort").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.image_sort_combo = ttk.Combobox(
            sort_frame,
            textvariable=self.image_sort_var,
            values=IMAGE_SORT_OPTIONS,
            state="readonly",
            width=20,
        )
        self.image_sort_combo.grid(row=0, column=1, sticky="ew")
        self.image_list = tk.Listbox(
            list_frame,
            width=24,
            activestyle="dotbox",
            bg="#1f2430",
            fg="#e8edf7",
            selectbackground="#315fbd",
            selectforeground="#ffffff",
            highlightthickness=0,
            relief="flat",
            exportselection=False,
            selectmode="extended",
        )
        image_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.image_list.yview)
        self.image_list.configure(yscrollcommand=image_scroll.set)
        self.image_list.grid(row=2, column=0, sticky="nsew")
        image_scroll.grid(row=2, column=1, sticky="ns")

        canvas_frame = ttk.Frame(self.paned)
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(canvas_frame, bg="#05070a", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        editor_outer = ttk.Frame(self.paned, width=440, style="Panel.TFrame")
        editor_outer.rowconfigure(0, weight=1)
        editor_outer.columnconfigure(0, weight=1)
        self.editor = ScrollFrame(editor_outer)
        self.editor.grid(row=0, column=0, sticky="nsew")
        self._build_editor(self.editor.interior)

        self.paned.add(list_frame, weight=0)
        self.paned.add(canvas_frame, weight=1)
        self.paned.add(editor_outer, weight=0)

    def _build_editor(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        row = 0

        self.image_title_label = ttk.Label(parent, textvariable=self.image_title_var, style="Section.TLabel")
        self.image_title_label.grid(row=row, column=0, sticky="w", pady=(0, 6))
        row += 1

        row = self._build_ai_controls(parent, row)

        self.high_text, row = self._add_text(
            parent,
            row,
            "High-level description",
            5,
            self._on_form_change,
            autofit=True,
            max_height=16,
        )

        ttk.Separator(parent).grid(row=row, column=0, sticky="ew", pady=12)
        row += 1
        ttk.Label(parent, text="Style", style="Section.TLabel").grid(row=row, column=0, sticky="w", pady=(0, 6))
        row += 1

        mode_frame = ttk.Frame(parent)
        mode_frame.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        mode_frame.columnconfigure(1, weight=1)
        ttk.Label(mode_frame, text="Mode").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.style_mode_combo = ttk.Combobox(
            mode_frame,
            textvariable=self.style_mode_var,
            values=STYLE_MODES,
            state="readonly",
            width=14,
        )
        self.style_mode_combo.grid(row=0, column=1, sticky="ew")
        row += 1

        self.aesthetics_text, row = self._add_text(parent, row, "Aesthetics", 2, self._on_form_change)
        self.lighting_text, row = self._add_text(parent, row, "Lighting", 2, self._on_form_change)

        medium_frame = ttk.Frame(parent)
        medium_frame.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        medium_frame.columnconfigure(1, weight=1)
        ttk.Label(medium_frame, text="Medium").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.medium_combo = ttk.Combobox(medium_frame, textvariable=self.medium_var, values=COMMON_MEDIA)
        self.medium_combo.grid(row=0, column=1, sticky="ew")
        row += 1

        self.style_detail_label = ttk.Label(parent, text="Photo")
        self.style_detail_label.grid(row=row, column=0, sticky="w", pady=(0, 2))
        row += 1
        self.style_detail_text = self._make_text(parent, 2)
        self.style_detail_text.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        self._bind_text(self.style_detail_text, self._on_form_change)
        row += 1

        self.palette_entry, row = self._add_entry(parent, row, "Color palette (max 16)", self.palette_var, self._on_form_change)
        self.palette_entry.grid_remove()
        self.style_palette_list, row = self._add_palette_tools(parent, row, "style")

        ttk.Separator(parent).grid(row=row, column=0, sticky="ew", pady=12)
        row += 1
        self.background_text, row = self._add_text(parent, row, "Background", 5, self._on_form_change)

        ttk.Separator(parent).grid(row=row, column=0, sticky="ew", pady=12)
        row += 1
        ttk.Label(parent, text="Elements", style="Section.TLabel").grid(row=row, column=0, sticky="w", pady=(0, 6))
        row += 1

        element_buttons = ttk.Frame(parent)
        element_buttons.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(element_buttons, text="+ Obj", command=lambda: self.add_element("obj")).grid(row=0, column=0, padx=(0, 4))
        ttk.Button(element_buttons, text="+ Text", command=lambda: self.add_element("text")).grid(row=0, column=1, padx=(0, 4))
        ttk.Checkbutton(element_buttons, text="Draw box", variable=self.draw_mode).grid(row=0, column=2, padx=(8, 0))
        row += 1

        self.elements_tree = ttk.Treeview(
            parent,
            columns=("type", "bbox", "summary"),
            show="headings",
            height=7,
            selectmode="browse",
        )
        self.elements_tree.heading("type", text="Type")
        self.elements_tree.heading("bbox", text="BBox")
        self.elements_tree.heading("summary", text="Description")
        self.elements_tree.column("type", width=50, stretch=False)
        self.elements_tree.column("bbox", width=105, stretch=False)
        self.elements_tree.column("summary", width=230, stretch=True)
        self.elements_tree.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        self.configure_element_tree_tags()
        row += 1

        detail = ttk.Frame(parent)
        detail.grid(row=row, column=0, sticky="ew")
        detail.columnconfigure(0, weight=1)
        self.element_detail = detail
        row += 1

        detail_row = 0
        type_frame = ttk.Frame(detail)
        type_frame.grid(row=detail_row, column=0, sticky="ew", pady=(0, 6))
        type_frame.columnconfigure(1, weight=1)
        ttk.Label(type_frame, text="Selected type").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.element_type_combo = ttk.Combobox(
            type_frame,
            textvariable=self.element_type_var,
            values=ELEMENT_TYPES,
            state="readonly",
            width=10,
        )
        self.element_type_combo.grid(row=0, column=1, sticky="ew")
        ttk.Button(type_frame, text="Delete", command=self.delete_selected_element).grid(row=0, column=2, padx=(8, 0))
        detail_row += 1

        self.element_text_label = ttk.Label(detail, text="Text rendered in image")
        self.element_text_label.grid(row=detail_row, column=0, sticky="w", pady=(0, 2))
        detail_row += 1
        self.element_text_text = self._make_text(detail, 2)
        self.element_text_text.grid(row=detail_row, column=0, sticky="ew", pady=(0, 6))
        self._bind_text(self.element_text_text, self._on_element_change)
        detail_row += 1

        self.element_desc_text, detail_row = self._add_text(
            detail,
            detail_row,
            "Description",
            6,
            self._on_element_change,
            autofit=True,
            max_height=20,
        )

        bbox_outer = ttk.Frame(detail)
        bbox_outer.grid(row=detail_row, column=0, sticky="ew", pady=(2, 6))
        for index, (key, label) in enumerate((("y1", "y1"), ("x1", "x1"), ("y2", "y2"), ("x2", "x2"))):
            ttk.Label(bbox_outer, text=label).grid(row=0, column=index, sticky="w", padx=(0 if index == 0 else 6, 2))
            spin = ttk.Spinbox(
                bbox_outer,
                from_=0,
                to=1000,
                increment=1,
                textvariable=self.bbox_vars[key],
                width=6,
                command=self._on_element_change,
            )
            spin.grid(row=1, column=index, sticky="ew", padx=(0 if index == 0 else 6, 0))
        detail_row += 1

        bbox_buttons = ttk.Frame(detail)
        bbox_buttons.grid(row=detail_row, column=0, sticky="ew", pady=(0, 6))
        ttk.Button(bbox_buttons, text="Remove bbox", command=self.remove_selected_bbox).grid(row=0, column=0, padx=(0, 4))
        ttk.Button(bbox_buttons, text="Fit image", command=self.fit_selected_bbox_to_image).grid(row=0, column=1)
        detail_row += 1

        self.element_palette_entry, detail_row = self._add_entry(
            detail,
            detail_row,
            "Element palette (max 5)",
            self.element_palette_var,
            self._on_element_change,
        )
        self.element_palette_entry.grid_remove()
        self.element_palette_list, detail_row = self._add_palette_tools(detail, detail_row, "element")

        ttk.Separator(parent).grid(row=row, column=0, sticky="ew", pady=12)
        row += 1
        ttk.Label(parent, text="Original Caption", style="Section.TLabel").grid(row=row, column=0, sticky="w", pady=(0, 6))
        row += 1
        self.original_status_label = ttk.Label(parent, textvariable=self.original_status_var)
        self.original_status_label.grid(row=row, column=0, sticky="w", pady=(0, 2))
        row += 1
        self.original_text = self._make_text(parent, 8)
        self.original_text.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        self._bind_text(self.original_text, self._on_original_change)
        self._enable_text_autofit(self.original_text, 8, 24)
        self.update_original_text_state()

    def _build_ai_controls(self, parent: ttk.Frame, row: int) -> int:
        ttk.Label(parent, text="Auto Captioning", style="Section.TLabel").grid(row=row, column=0, sticky="w", pady=(0, 6))
        row += 1
        ai_frame = ttk.Frame(parent)
        ai_frame.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        for column in range(2):
            ai_frame.columnconfigure(column, weight=1)

        buttons = (
            ("Text Caption", self.auto_generate_text_caption),
            ("JSON from Text", self.auto_generate_json_from_text),
            ("JSON from Image", self.auto_generate_json_from_image),
            ("Add/redo BBoxes", self.auto_generate_bboxes),
            ("Retry Failed", self.retry_failed_captions),
            ("Clear Failed", self.clear_failed_markers),
            ("Undo AI", self.undo_last_ai_job),
            ("Prefs", self.open_captioning_preferences),
        )
        self.ai_buttons = []
        for index, (label, command) in enumerate(buttons):
            button = ttk.Button(ai_frame, text=label, command=command)
            button.grid(row=index // 2, column=index % 2, sticky="ew", padx=(0 if index % 2 == 0 else 4, 0), pady=(0, 4))
            self.ai_buttons.append(button)
        cancel_row = (len(buttons) + 1) // 2
        self.ai_cancel_button = ttk.Button(ai_frame, text="Cancel Run", command=self.cancel_ai_job, state="disabled")
        self.ai_cancel_button.grid(row=cancel_row, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        row += 1

        ttk.Separator(parent).grid(row=row, column=0, sticky="ew", pady=(4, 12))
        return row + 1

    def _make_text(self, parent: tk.Widget, height: int) -> tk.Text:
        return tk.Text(
            parent,
            height=height,
            wrap="word",
            undo=True,
            bg="#222733",
            fg="#f2f5fb",
            insertbackground="#ffffff",
            selectbackground="#315fbd",
            relief="solid",
            borderwidth=1,
            highlightthickness=0,
            font=("Segoe UI", 10),
        )

    def _add_text(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        height: int,
        callback: callable,
        autofit: bool = False,
        max_height: int | None = None,
    ) -> tuple[tk.Text, int]:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(0, 2))
        row += 1
        widget = self._make_text(parent, height)
        widget.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        self._bind_text(widget, callback)
        if autofit:
            self._enable_text_autofit(widget, height, max_height)
        return widget, row + 1

    def _add_entry(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        callback: callable,
    ) -> tuple[ttk.Entry, int]:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=(0, 2))
        row += 1
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=0, sticky="ew", pady=(0, 6))
        variable.trace_add("write", lambda *_args: callback())
        return entry, row + 1

    def _add_palette_tools(self, parent: ttk.Frame, row: int, scope: str) -> tuple[tk.Canvas, int]:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        frame.columnconfigure(0, weight=1)
        colors = tk.Canvas(
            frame,
            height=30,
            bg="#1f2430",
            highlightthickness=0,
            relief="flat",
        )
        colors.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        colors.bind("<Button-1>", lambda event: self.on_palette_swatch_click(scope, event))
        ttk.Button(frame, text="Add", command=lambda: self.start_palette_eyedrop(scope, add=True)).grid(
            row=0, column=1, sticky="ew", padx=(0, 4)
        )
        ttk.Button(frame, text="Delete", command=lambda: self.remove_palette_color(scope)).grid(row=0, column=2, sticky="ew")
        return colors, row + 1

    def palette_var_for_scope(self, scope: str) -> tk.StringVar:
        return self.palette_var if scope == "style" else self.element_palette_var

    def palette_canvas_for_scope(self, scope: str) -> tk.Canvas:
        return self.style_palette_list if scope == "style" else self.element_palette_list

    def palette_limit_for_scope(self, scope: str) -> int:
        return 16 if scope == "style" else 5

    def get_palette_colors(self, scope: str) -> list[str]:
        colors, _invalid = parse_palette_text(self.palette_var_for_scope(scope).get(), self.palette_limit_for_scope(scope))
        return colors

    def set_palette_colors(self, scope: str, colors: list[str]) -> None:
        limit = self.palette_limit_for_scope(scope)
        self.palette_var_for_scope(scope).set(", ".join(colors[:limit]))
        self.refresh_palette_list(scope)

    def refresh_palette_list(self, scope: str) -> None:
        canvas = self.palette_canvas_for_scope(scope)
        colors = self.get_palette_colors(scope)
        selected = self.palette_selected_index.get(scope)
        if selected is not None and selected >= len(colors):
            selected = None
            self.palette_selected_index[scope] = None
        canvas.delete("all")
        swatch = 20
        gap = 4
        x = 3
        y = 5
        for index, color in enumerate(colors):
            outline = "#ffffff" if index == selected else "#566174"
            width = 3 if index == selected else 1
            canvas.create_rectangle(
                x,
                y,
                x + swatch,
                y + swatch,
                fill=color,
                outline=outline,
                width=width,
                tags=(f"swatch:{index}",),
            )
            x += swatch + gap
        if not colors:
            canvas.create_text(4, 16, anchor="w", text="No colors", fill="#7f8898", font=("Segoe UI", 9))

    def refresh_palette_lists(self) -> None:
        self.refresh_palette_list("style")
        self.refresh_palette_list("element")

    def selected_palette_index(self, scope: str) -> int | None:
        return self.palette_selected_index.get(scope)

    def on_palette_swatch_click(self, scope: str, event: tk.Event) -> None:
        index = self.palette_swatch_index_at(scope, event.x, event.y)
        if index is None:
            return
        self.palette_selected_index[scope] = index
        self.refresh_palette_list(scope)
        self.start_palette_eyedrop(scope, add=False)

    def palette_swatch_index_at(self, scope: str, x: float, y: float) -> int | None:
        if y < 5 or y > 25:
            return None
        index = int((x - 3) // 24)
        if index < 0:
            return None
        swatch_left = 3 + index * 24
        if x < swatch_left or x > swatch_left + 20:
            return None
        if index >= len(self.get_palette_colors(scope)):
            return None
        return index

    def start_palette_eyedrop(self, scope: str, add: bool) -> None:
        if scope == "element" and self.get_selected_element() is None:
            self.status_var.set("Select an element before editing its palette.")
            return
        colors = self.get_palette_colors(scope)
        limit = self.palette_limit_for_scope(scope)
        if add:
            if len(colors) >= limit:
                self.status_var.set(f"{scope.capitalize()} palette already has the maximum {limit} colors.")
                return
            index = len(colors)
        else:
            selected = self.selected_palette_index(scope)
            if selected is None:
                self.status_var.set("Select a palette color to replace.")
                return
            index = selected
        self.eyedrop_target = {"scope": scope, "index": index}
        self.palette_selected_index[scope] = index if not add else None
        self.refresh_palette_list(scope)
        self.canvas.configure(cursor="crosshair")
        self.current_canvas_cursor = "crosshair"
        action = "new color" if add else f"color #{index + 1}"
        self.status_var.set(f"Eyedrop armed for {scope} {action}. Click the image to sample.")

    def remove_palette_color(self, scope: str) -> None:
        selected = self.selected_palette_index(scope)
        if selected is None:
            self.status_var.set("Select a palette color to remove.")
            return
        colors = self.get_palette_colors(scope)
        if not (0 <= selected < len(colors)):
            return
        del colors[selected]
        self.palette_selected_index[scope] = None
        self.cancel_eyedrop()
        self.set_palette_colors(scope, colors)

    def cancel_eyedrop(self, update_status: bool = False) -> bool:
        if self.eyedrop_target is None:
            return False
        scope = self.eyedrop_target["scope"]
        self.eyedrop_target = None
        self.canvas.configure(cursor="")
        self.current_canvas_cursor = ""
        self.palette_selected_index[scope] = None
        self.refresh_palette_list(scope)
        if update_status:
            self.status_var.set("Eyedrop canceled.")
        return True

    def sample_color_at_canvas(self, x: float, y: float) -> str | None:
        if self.pil_image is None:
            return None
        left, top, width, height = self.image_bounds
        if x < left or y < top or x > left + width or y > top + height:
            return None
        image_x = max(0, min(self.pil_image.width - 1, int(((x - left) / width) * self.pil_image.width)))
        image_y = max(0, min(self.pil_image.height - 1, int(((y - top) / height) * self.pil_image.height)))
        red, green, blue = self.pil_image.getpixel((image_x, image_y))[:3]
        return f"#{red:02X}{green:02X}{blue:02X}"

    def apply_eyedrop_sample(self, x: float, y: float) -> bool:
        if self.eyedrop_target is None:
            return False
        color = self.sample_color_at_canvas(x, y)
        if color is None:
            self.status_var.set("Click inside the image to sample a color.")
            return True
        scope = self.eyedrop_target["scope"]
        index = self.eyedrop_target["index"]
        colors = self.get_palette_colors(scope)
        if index >= len(colors):
            colors.append(color)
        else:
            colors[index] = color
        self.eyedrop_target = None
        self.canvas.configure(cursor="")
        self.current_canvas_cursor = ""
        self.palette_selected_index[scope] = min(index, len(colors) - 1)
        self.set_palette_colors(scope, colors)
        self.status_var.set(f"Sampled {color}.")
        return True

    def _bind_events(self) -> None:
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.extension_combo.bind("<<ComboboxSelected>>", self.on_caption_extension_changed)
        self.original_extension_combo.bind("<<ComboboxSelected>>", self.on_original_extension_changed)
        self.image_sort_combo.bind("<<ComboboxSelected>>", self.on_image_sort_changed)
        self.style_mode_var.trace_add("write", lambda *_args: self.on_style_mode_changed())
        self.medium_var.trace_add("write", lambda *_args: self._on_form_change())
        self.element_type_var.trace_add("write", lambda *_args: self.on_element_type_changed())
        for var in self.bbox_vars.values():
            var.trace_add("write", lambda *_args: self._on_element_change())

        self.image_list.bind("<<ListboxSelect>>", self.on_image_list_select)
        self.image_list.bind("<ButtonPress-1>", self.remember_image_list_click, add="+")
        self.image_list.bind("<Control-a>", self.select_all_images)
        self.image_list.bind("<Control-A>", self.select_all_images)
        self.elements_tree.bind("<<TreeviewSelect>>", self.on_element_tree_select)
        self.canvas.bind("<Configure>", lambda _event: self.render_image())
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        self.canvas.bind("<Leave>", lambda _event: self.set_canvas_cursor(""))

        self.bind_all("<KeyPress-Right>", self.next_image_from_key)
        self.bind_all("<KeyPress-Down>", self.next_image_from_key)
        self.bind_all("<KeyPress-Left>", self.previous_image_from_key)
        self.bind_all("<KeyPress-Up>", self.previous_image_from_key)
        self.bind_all("<Control-KeyPress-Down>", self.next_image_always)
        self.bind_all("<Control-KeyPress-Up>", self.previous_image_always)
        self.bind_all("<KeyPress-Return>", self.on_return_key)
        self.bind_all("<F11>", lambda _event: self.toggle_fullscreen())
        self.bind_all("<Escape>", self.on_escape_key)
        self.bind_all("<Control-s>", lambda _event: self.save_current())
        self.bind_child_navigation_shortcuts(self)

    def _bind_text(self, widget: tk.Text, callback: callable) -> None:
        widget.bind("<<Modified>>", lambda event: self._on_text_modified(event, callback))
        widget.bind("<KeyPress-Return>", self.on_text_return_key)
        widget.bind("<Tab>", self.focus_next_widget)
        widget.bind("<Shift-Tab>", self.focus_previous_widget)
        try:
            widget.bind("<ISO_Left_Tab>", self.focus_previous_widget)
        except tk.TclError:
            pass
        widget.bind("<Control-KeyPress-Down>", self.next_image_always)
        widget.bind("<Control-KeyPress-Up>", self.previous_image_always)

    def focus_next_widget(self, event: tk.Event) -> str:
        next_widget = event.widget.tk_focusNext()
        if next_widget is not None:
            next_widget.focus_set()
        return "break"

    def focus_previous_widget(self, event: tk.Event) -> str:
        previous_widget = event.widget.tk_focusPrev()
        if previous_widget is not None:
            previous_widget.focus_set()
        return "break"

    def bind_child_navigation_shortcuts(self, widget: tk.Widget) -> None:
        widget.bind("<Control-KeyPress-Down>", self.next_image_always)
        widget.bind("<Control-KeyPress-Up>", self.previous_image_always)
        for child in widget.winfo_children():
            self.bind_child_navigation_shortcuts(child)

    def _enable_text_autofit(self, widget: tk.Text, min_height: int, max_height: int | None) -> None:
        widget.autofit_bounds = (min_height, max_height)
        widget.bind("<Configure>", lambda event: self._queue_text_autofit(event.widget), add="+")
        self._queue_text_autofit(widget)

    def _queue_text_autofit(self, widget: tk.Text) -> None:
        widget.after_idle(lambda: self._fit_text_height(widget))

    def _fit_text_height(self, widget: tk.Text) -> None:
        bounds = getattr(widget, "autofit_bounds", None)
        if not bounds:
            return
        min_height, max_height = bounds
        try:
            display_count = widget.count("1.0", "end-1c", "update", "displaylines")
            if isinstance(display_count, tuple):
                display_lines = int(display_count[-1]) if display_count else 1
            else:
                display_lines = int(display_count) if display_count else 1
            target_height = max(min_height, display_lines + 1)
            if max_height is not None:
                target_height = min(max_height, target_height)
            if int(widget.cget("height")) != target_height:
                widget.configure(height=target_height)
        except tk.TclError:
            return

    def _on_text_modified(self, event: tk.Event, callback: callable) -> None:
        widget = event.widget
        if isinstance(widget, tk.Text) and widget.edit_modified():
            widget.edit_modified(False)
            callback()
            self._queue_text_autofit(widget)

    def _get_text(self, widget: tk.Text) -> str:
        return widget.get("1.0", "end-1c")

    def _set_text(self, widget: tk.Text, value: Any) -> None:
        old_state = str(widget.cget("state"))
        if old_state == "disabled":
            widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", "" if value is None else str(value))
        widget.edit_modified(False)
        if old_state == "disabled":
            widget.configure(state="disabled")
        self._queue_text_autofit(widget)

    def selected_original_extension(self) -> str | None:
        extension = self.original_extension_var.get()
        if extension == ORIGINAL_NONE:
            return None
        return extension if extension in ORIGINAL_FILE_EXTENSIONS else ".txt"

    def original_path_for_image(self, image_path: Path) -> Path | None:
        extension = self.selected_original_extension()
        if extension is None:
            return None
        return image_path.with_suffix(extension)

    def update_original_text_state(self) -> None:
        enabled = self.current_original_path is not None and self.current_index >= 0
        self.original_text.configure(state="normal" if enabled else "disabled")

    def clear_original_text(self, status: str) -> None:
        self.loading_original = True
        self.current_original_path = None
        self._set_text(self.original_text, "")
        self.original_status_var.set(status)
        self.original_dirty = False
        self.update_original_text_state()
        self.loading_original = False

    def load_original_text(self, image_path: Path) -> None:
        original_path = self.original_path_for_image(image_path)
        if original_path is None:
            self.clear_original_text("Original: none")
            return
        if self.store is not None and self.store.caption_path(image_path) == original_path:
            self.clear_original_text(f"Original: {original_path.name} is the active caption file")
            return

        self.loading_original = True
        self.current_original_path = original_path
        try:
            text = original_path.read_text(encoding="utf-8-sig") if original_path.exists() else ""
        except OSError as exc:
            text = ""
            self.original_status_var.set(f"Original: could not read {original_path.name}: {exc}")
            self.current_original_path = None
        else:
            if original_path.exists():
                self.original_status_var.set(f"Original: {original_path.name}")
            else:
                self.original_status_var.set(f"Original: no {original_path.name} yet")
        self._set_text(self.original_text, text)
        self.original_dirty = False
        self.update_original_text_state()
        self.loading_original = False

    def open_folder_dialog(self) -> None:
        folder = filedialog.askdirectory(title="Open image folder")
        if folder:
            self.open_folder(Path(folder))

    def image_modified_time(self, image_path: Path) -> float:
        try:
            return image_path.stat().st_mtime
        except OSError:
            return 0.0

    def image_name_key(self, image_path: Path) -> str:
        return image_path.name.lower()

    def image_has_caption(self, image_path: Path) -> bool:
        return self.store is not None and self.store.caption_path(image_path).exists()

    def image_has_failure_marker(self, image_path: Path) -> bool:
        return self.store is not None and self.store.has_failure_marker(image_path)

    def image_has_original(self, image_path: Path) -> bool:
        original_path = self.original_path_for_image(image_path)
        return original_path is not None and original_path.exists()

    def image_has_missing_bboxes(self, image_path: Path) -> bool:
        if self.store is None:
            return False
        caption_path = self.store.caption_path(image_path)
        if not caption_path.exists():
            return False
        try:
            caption, message = self.store.load_caption(image_path)
        except OSError:
            return False
        if message and "Could not parse" in message:
            return False
        elements = normalize_caption(caption).get("compositional_deconstruction", {}).get("elements", [])
        return any(
            isinstance(element, dict)
            and element.get("type") in {"obj", "text"}
            and normalize_bbox(element.get("bbox")) is None
            for element in elements
        )

    def visible_images_for_current_mode(self) -> list[Path]:
        images = list(self.all_images)
        if self.image_sort_var.get() == IMAGE_SORT_BBOX_MISSING:
            images = [image_path for image_path in images if self.image_has_missing_bboxes(image_path)]
        elif self.image_sort_var.get() == IMAGE_SORT_FAILED:
            images = [image_path for image_path in images if self.image_has_failure_marker(image_path)]
        return images

    def sorted_images(self, images: list[Path]) -> list[Path]:
        sort_mode = self.image_sort_var.get()
        if sort_mode == IMAGE_SORT_NAME_DESC:
            return sorted(images, key=self.image_name_key, reverse=True)
        if sort_mode == IMAGE_SORT_MODIFIED_NEWEST:
            return sorted(images, key=lambda path: (-self.image_modified_time(path), self.image_name_key(path)))
        if sort_mode == IMAGE_SORT_MODIFIED_OLDEST:
            return sorted(images, key=lambda path: (self.image_modified_time(path), self.image_name_key(path)))
        if sort_mode == IMAGE_SORT_CAPTION_MISSING:
            return sorted(images, key=lambda path: (self.image_has_caption(path), self.image_name_key(path)))
        if sort_mode == IMAGE_SORT_ORIGINAL_MISSING:
            return sorted(images, key=lambda path: (self.image_has_original(path), self.image_name_key(path)))
        return sorted(images, key=self.image_name_key)

    def apply_image_sort(self, preserve_current: bool = True) -> None:
        current_path = None
        if preserve_current and 0 <= self.current_index < len(self.images):
            current_path = self.images[self.current_index]
        self.images = self.sorted_images(self.visible_images_for_current_mode())
        if current_path is not None and current_path in self.images:
            self.current_index = self.images.index(current_path)
        elif not self.images:
            self.current_index = -1
        elif not preserve_current:
            self.current_index = -1
        else:
            self.current_index = min(max(self.current_index, 0), len(self.images) - 1)

    def clear_loaded_image(self, message: str) -> None:
        self.current_index = -1
        self.image_title_var.set("No image loaded")
        self.current_caption = default_caption()
        self.selected_element_index = None
        if self.pil_image is not None:
            self.pil_image.close()
        self.pil_image = None
        self.canvas_image_id = None
        self.image_render_key = None
        self.clear_original_text("Original: no image loaded")
        self.populate_form()
        self.render_image()
        self.status_var.set(message)

    def open_folder(self, folder: Path) -> None:
        self.save_current()
        self.folder = folder
        self.store = CaptionStore(folder, self.caption_extension_var.get())
        self.all_images = self.store.images()
        self.images = []
        self.current_index = -1
        self.apply_image_sort(preserve_current=False)
        self.populate_image_list()
        if not self.all_images:
            self.status_var.set(f"No images found in {folder}")
            self.clear_loaded_image(f"No images found in {folder}")
            return
        if not self.images:
            self.clear_loaded_image(f"No images match {self.image_sort_var.get()}.")
            return
        self.load_image_at(0)

    def populate_image_list(self) -> None:
        self.loading_list = True
        self.image_list.delete(0, "end")
        if self.store is None:
            self.loading_list = False
            return
        for index, image_path in enumerate(self.images, start=1):
            if self.store.has_failure_marker(image_path):
                marker = "!"
            else:
                marker = " " if self.store.caption_path(image_path).exists() else "+"
            self.image_list.insert("end", f"{index:04d} {marker} {image_path.name}")
        self.loading_list = False

    def on_image_list_select(self, _event: tk.Event) -> None:
        if self.loading_list:
            return
        if self.pending_image_select_job is not None:
            self.after_cancel(self.pending_image_select_job)
        self.pending_image_select_job = self.after_idle(self.finish_image_list_select)

    def image_list_index_from_event(self, event: tk.Event) -> int | None:
        if not self.images:
            return None
        index = int(self.image_list.nearest(event.y))
        if not (0 <= index < len(self.images)):
            return None
        item_bbox = self.image_list.bbox(index)
        if item_bbox is not None:
            _x, y, _width, height = item_bbox
            if event.y < y or event.y > y + height:
                return None
        return index

    def current_image_selection_anchor_index(self) -> int | None:
        if self.image_selection_anchor_path is not None:
            try:
                return self.images.index(self.image_selection_anchor_path)
            except ValueError:
                pass
        if 0 <= self.current_index < len(self.images):
            return self.current_index
        selection = self.image_list.curselection()
        if selection:
            index = int(selection[0])
            if 0 <= index < len(self.images):
                return index
        return None

    def remember_image_list_click(self, event: tk.Event) -> str | None:
        self.last_image_click_index = None
        index = self.image_list_index_from_event(event)
        if index is None:
            return None
        self.last_image_click_index = index
        if event.state & SHIFT_MASK:
            anchor = self.current_image_selection_anchor_index()
            if anchor is None:
                anchor = index
                self.image_selection_anchor_path = self.images[index]
            first, last = sorted((anchor, index))
            self.loading_list = True
            if not (event.state & CONTROL_MASK):
                self.image_list.selection_clear(0, "end")
            self.image_list.selection_set(first, last)
            self.image_list.activate(index)
            self.image_list.see(index)
            if hasattr(self.image_list, "selection_anchor"):
                self.image_list.selection_anchor(anchor)
            self.loading_list = False
            self.on_image_list_select(event)
            return "break"
        self.image_selection_anchor_path = self.images[index]
        return None

    def current_image_selection_paths(self) -> list[Path]:
        if not self.images:
            return []
        paths: list[Path] = []
        seen: set[Path] = set()
        for raw_index in self.image_list.curselection():
            index = int(raw_index)
            if 0 <= index < len(self.images):
                image_path = self.images[index]
                if image_path not in seen:
                    paths.append(image_path)
                    seen.add(image_path)
        return paths

    def image_indices_for_paths(self, image_paths: list[Path] | tuple[Path, ...] | None) -> tuple[int, ...]:
        if not image_paths:
            return ()
        selected_paths = set(image_paths)
        return tuple(index for index, image_path in enumerate(self.images) if image_path in selected_paths)

    def flush_pending_image_selection(self) -> None:
        if self.pending_image_select_job is None:
            return
        pending_job = self.pending_image_select_job
        self.pending_image_select_job = None
        try:
            self.after_cancel(pending_job)
        except tk.TclError:
            pass
        self.finish_image_list_select()

    def finish_image_list_select(self) -> None:
        self.pending_image_select_job = None
        if self.loading_list:
            return
        selection = tuple(int(index) for index in self.image_list.curselection())
        clicked_index = self.last_image_click_index
        self.last_image_click_index = None
        if selection:
            if clicked_index is not None and clicked_index in selection:
                index = clicked_index
            else:
                try:
                    active = int(self.image_list.index("active"))
                except tk.TclError:
                    active = selection[0]
                index = active if active in selection else selection[0]
            self.load_image_at(index, preserve_selection=True)

    def select_all_images(self, _event: tk.Event | None = None) -> str:
        if not self.images:
            return "break"
        self.loading_list = True
        self.image_list.selection_set(0, "end")
        if self.current_index >= 0:
            self.image_selection_anchor_path = self.images[self.current_index]
            self.image_list.activate(self.current_index)
            self.image_list.see(self.current_index)
        self.loading_list = False
        self.status_var.set(f"Selected {len(self.images)} images.")
        return "break"

    def on_image_sort_changed(self, _event: tk.Event) -> None:
        selected_paths = self.current_image_selection_paths()
        if self.dirty or self.original_dirty:
            self.save_current()
        self.apply_image_sort(preserve_current=True)
        self.populate_image_list()
        if self.images:
            self.populate_image_selection(selected_paths=selected_paths)
            self.load_image_at(self.current_index if self.current_index >= 0 else 0, skip_save=True, force_reload=True)
            self.status_var.set(f"Sorted images by {self.image_sort_var.get()}.")
        else:
            self.clear_loaded_image(f"No images match {self.image_sort_var.get()}.")

    def load_image_at(self, index: int, preserve_selection: bool = False, skip_save: bool = False, force_reload: bool = False) -> None:
        if index < 0 or index >= len(self.images):
            return
        if index == self.current_index and not force_reload:
            return
        selected_paths = self.current_image_selection_paths() if preserve_selection else []
        self.cancel_eyedrop()
        if not skip_save:
            self.save_current()
        self.current_index = index
        image_path = self.images[index]
        self.image_title_var.set(image_path.name)

        try:
            if self.pil_image is not None:
                self.pil_image.close()
            self.pil_image = Image.open(image_path).convert("RGB")
            self.canvas_image_id = None
            self.image_render_key = None
        except (OSError, UnidentifiedImageError) as exc:
            self.pil_image = None
            self.canvas_image_id = None
            self.image_render_key = None
            self.current_caption = default_caption()
            self.clear_original_text("Original: image unavailable")
            self.status_var.set(f"Could not open {image_path.name}: {exc}")
            self.render_image()
            return

        if self.store is None:
            self.current_caption = default_caption()
            message = None
        else:
            self.current_caption, message = self.store.load_caption(image_path)
            self.current_caption = normalize_caption(self.current_caption)

        elements = self.current_caption["compositional_deconstruction"]["elements"]
        self.selected_element_index = 0 if elements else None
        self.load_original_text(image_path)
        self.populate_form()
        self.populate_image_selection(selected_paths=selected_paths if preserve_selection else None)
        self.editor.scroll_to_top()
        self.render_image()
        if message:
            self.status_var.set(message)
        else:
            self.status_var.set(f"{index + 1}/{len(self.images)} {image_path.name}")

    def populate_image_selection(
        self,
        selected_indices: tuple[int, ...] | None = None,
        selected_paths: list[Path] | tuple[Path, ...] | None = None,
    ) -> None:
        if selected_paths is not None:
            selected_indices = self.image_indices_for_paths(selected_paths)
        self.loading_list = True
        self.image_list.selection_clear(0, "end")
        selected_any = False
        if selected_indices:
            for index in selected_indices:
                if 0 <= index < len(self.images):
                    self.image_list.selection_set(index)
                    selected_any = True
            if self.current_index >= 0:
                self.image_list.activate(self.current_index)
                self.image_list.see(self.current_index)
        if not selected_any and self.current_index >= 0:
            self.image_list.selection_set(self.current_index)
            self.image_list.activate(self.current_index)
            self.image_list.see(self.current_index)
            self.image_selection_anchor_path = self.images[self.current_index]
        anchor_index = self.current_image_selection_anchor_index()
        if anchor_index is not None and hasattr(self.image_list, "selection_anchor"):
            self.image_list.selection_anchor(anchor_index)
        self.loading_list = False

    def populate_form(self) -> None:
        self.loading_form = True
        self.loading_element = True
        caption = normalize_caption(self.current_caption)
        self.current_caption = caption

        self._set_text(self.high_text, caption.get("high_level_description", ""))
        style = caption.get("style_description", {})
        mode = "art_style" if "art_style" in style and "photo" not in style else "photo"
        self.style_mode_var.set(mode)
        self._set_text(self.aesthetics_text, style.get("aesthetics", ""))
        self._set_text(self.lighting_text, style.get("lighting", ""))
        self.medium_var.set(style.get("medium", "photograph" if mode == "photo" else "illustration"))
        self._set_text(self.style_detail_text, style.get("photo" if mode == "photo" else "art_style", ""))
        self.palette_var.set(palette_to_text(style.get("color_palette", [])))
        self.update_style_detail_label()
        self.refresh_palette_list("style")

        comp = caption.get("compositional_deconstruction", {})
        self._set_text(self.background_text, comp.get("background", ""))
        self.populate_elements_tree()
        self.populate_element_detail()
        self.dirty = False
        self.loading_element = False
        self.loading_form = False

    def populate_elements_tree(self) -> None:
        self.elements_tree.delete(*self.elements_tree.get_children())
        elements = self.current_caption.get("compositional_deconstruction", {}).get("elements", [])
        obj_color_index = 0
        for index, element in enumerate(elements):
            tag = self.element_tree_tag(element, obj_color_index)
            self.elements_tree.insert("", "end", iid=str(index), values=self.element_tree_values(index, element), tags=(tag,))
            if element.get("type", "obj") == "obj":
                obj_color_index += 1
        if self.selected_element_index is not None and 0 <= self.selected_element_index < len(elements):
            self.elements_tree.selection_set(str(self.selected_element_index))
            self.elements_tree.see(str(self.selected_element_index))

    def configure_element_tree_tags(self) -> None:
        for color_index, color in enumerate(OBJ_BOX_COLORS):
            self.elements_tree.tag_configure(
                f"obj_color_{color_index}",
                background=mix_hex_color(color, ELEMENT_ROW_BASE_COLOR, ELEMENT_ROW_TINT),
                foreground=ELEMENT_ROW_TEXT_COLOR,
            )
        self.elements_tree.tag_configure(
            "text_color",
            background=mix_hex_color(TEXT_BOX_COLOR, ELEMENT_ROW_BASE_COLOR, ELEMENT_ROW_TINT),
            foreground=ELEMENT_ROW_TEXT_TEXT_COLOR,
        )

    def element_tree_tag(self, element: dict[str, Any], obj_color_index: int) -> str:
        if element.get("type", "obj") == "obj":
            return f"obj_color_{obj_color_index % len(OBJ_BOX_COLORS)}"
        return "text_color"

    def element_box_color(self, element: dict[str, Any], obj_color_index: int) -> str:
        if element.get("type", "obj") == "obj":
            return OBJ_BOX_COLORS[obj_color_index % len(OBJ_BOX_COLORS)]
        return TEXT_BOX_COLOR

    def refresh_element_tree_tags(self) -> None:
        elements = self.current_caption.get("compositional_deconstruction", {}).get("elements", [])
        obj_color_index = 0
        for index, element in enumerate(elements):
            iid = str(index)
            if self.elements_tree.exists(iid):
                self.elements_tree.item(iid, tags=(self.element_tree_tag(element, obj_color_index),))
            if element.get("type", "obj") == "obj":
                obj_color_index += 1

    def element_tree_values(self, _index: int, element: dict[str, Any]) -> tuple[str, str, str]:
        bbox = element.get("bbox")
        bbox_text = ",".join(str(v) for v in bbox) if bbox else ""
        summary = element.get("text") if element.get("type") == "text" else element.get("desc", "")
        if not summary:
            summary = element.get("desc", "")
        return element.get("type", "obj"), bbox_text, summary[:80]

    def update_selected_element_tree_row(self) -> None:
        if self.selected_element_index is None:
            return
        elements = self.current_caption.get("compositional_deconstruction", {}).get("elements", [])
        if not (0 <= self.selected_element_index < len(elements)):
            return
        iid = str(self.selected_element_index)
        if self.elements_tree.exists(iid):
            self.elements_tree.item(iid, values=self.element_tree_values(self.selected_element_index, elements[self.selected_element_index]))
            self.refresh_element_tree_tags()

    def populate_element_detail(self) -> None:
        self.loading_element = True
        element = self.get_selected_element()
        if element is None:
            self.element_type_var.set("obj")
            self._set_text(self.element_text_text, "")
            self._set_text(self.element_desc_text, "")
            self.element_palette_var.set("")
            self.refresh_palette_list("element")
            for var in self.bbox_vars.values():
                var.set("")
            self.update_element_detail_state()
            self.loading_element = False
            return

        self.element_type_var.set(element.get("type", "obj"))
        self._set_text(self.element_text_text, element.get("text", ""))
        self._set_text(self.element_desc_text, element.get("desc", ""))
        self.element_palette_var.set(palette_to_text(element.get("color_palette", [])))
        self.refresh_palette_list("element")
        bbox = element.get("bbox") or ["", "", "", ""]
        for key, value in zip(("y1", "x1", "y2", "x2"), bbox):
            self.bbox_vars[key].set(str(value))
        self.update_element_detail_state()
        self.loading_element = False

    def update_style_detail_label(self) -> None:
        mode = self.style_mode_var.get()
        self.style_detail_label.configure(text="Photo" if mode == "photo" else "Art style")

    def update_element_detail_state(self) -> None:
        element = self.get_selected_element()
        has_element = element is not None
        state = "normal" if has_element else "disabled"
        text_state = "normal" if has_element and self.element_type_var.get() == "text" else "disabled"
        self.element_type_combo.configure(state="readonly" if has_element else "disabled")
        if has_element and self.element_type_var.get() == "text":
            self.element_text_label.grid()
            self.element_text_text.grid()
            self.element_text_text.configure(state=text_state)
        else:
            self.element_text_label.grid_remove()
            self.element_text_text.grid_remove()
        self.element_desc_text.configure(state=state)
        self.element_palette_entry.configure(state=state)
        for child in self.element_detail.winfo_children():
            if isinstance(child, ttk.Frame):
                for grandchild in child.winfo_children():
                    if isinstance(grandchild, (ttk.Button, ttk.Spinbox)):
                        grandchild.configure(state=state)

    def _on_form_change(self) -> None:
        if self.loading_form:
            return
        self.sync_caption_from_form()
        self.refresh_palette_list("style")
        self.mark_dirty()

    def _on_element_change(self) -> None:
        if self.loading_form or self.loading_element:
            return
        self.sync_selected_element_from_form()
        self.update_selected_element_tree_row()
        self.refresh_palette_list("element")
        self.redraw_overlays()
        self.mark_dirty()

    def _on_original_change(self) -> None:
        if self.loading_original:
            return
        self.mark_original_dirty()

    def on_style_mode_changed(self) -> None:
        self.update_style_detail_label()
        if self.loading_form:
            return
        if self.style_mode_var.get() == "photo":
            if not self.medium_var.get().strip():
                self.medium_var.set("photograph")
        elif not self.medium_var.get().strip() or self.medium_var.get() == "photograph":
            self.medium_var.set("illustration")
        self._on_form_change()

    def on_element_type_changed(self) -> None:
        self.update_element_detail_state()
        self._on_element_change()

    def sync_caption_from_form(self) -> None:
        if self.current_index < 0:
            return
        self.sync_selected_element_from_form()
        elements = self.current_caption.get("compositional_deconstruction", {}).get("elements", [])

        mode = self.style_mode_var.get() if self.style_mode_var.get() in STYLE_MODES else "photo"
        style: dict[str, Any] = {
            "aesthetics": self._get_text(self.aesthetics_text),
            "lighting": self._get_text(self.lighting_text),
        }
        if mode == "photo":
            style["photo"] = self._get_text(self.style_detail_text)
            style["medium"] = self.medium_var.get().strip() or "photograph"
        else:
            style["medium"] = self.medium_var.get().strip() or "illustration"
            style["art_style"] = self._get_text(self.style_detail_text)

        palette, invalid = parse_palette_text(self.palette_var.get(), 16)
        if palette:
            style["color_palette"] = palette
        if invalid:
            self.status_var.set("Ignored invalid global palette entries: " + ", ".join(invalid[:4]))

        self.current_caption = {
            "high_level_description": self._get_text(self.high_text),
            "style_description": style,
            "compositional_deconstruction": {
                "background": self._get_text(self.background_text),
                "elements": elements,
            },
        }

    def sync_selected_element_from_form(self) -> None:
        if self.selected_element_index is None:
            return
        elements = self.current_caption.get("compositional_deconstruction", {}).get("elements", [])
        if self.selected_element_index < 0 or self.selected_element_index >= len(elements):
            return

        element_type = self.element_type_var.get() if self.element_type_var.get() in ELEMENT_TYPES else "obj"
        element: dict[str, Any] = {"type": element_type}
        bbox = self._bbox_from_vars()
        if bbox:
            element["bbox"] = bbox
        if element_type == "text":
            element["text"] = self._get_text(self.element_text_text)
        element["desc"] = self._get_text(self.element_desc_text)
        palette, invalid = parse_palette_text(self.element_palette_var.get(), 5)
        if palette:
            element["color_palette"] = palette
        if invalid:
            self.status_var.set("Ignored invalid element palette entries: " + ", ".join(invalid[:4]))

        elements[self.selected_element_index] = element

    def _bbox_from_vars(self) -> list[int] | None:
        values = [self.bbox_vars[key].get().strip() for key in ("y1", "x1", "y2", "x2")]
        if not any(values):
            return None
        if not all(values):
            return None
        return normalize_bbox(values)

    def get_selected_element(self) -> dict[str, Any] | None:
        if self.selected_element_index is None:
            return None
        elements = self.current_caption.get("compositional_deconstruction", {}).get("elements", [])
        if self.selected_element_index < 0 or self.selected_element_index >= len(elements):
            return None
        return elements[self.selected_element_index]

    def on_element_tree_select(self, _event: tk.Event) -> None:
        if self.loading_form:
            return
        selection = self.elements_tree.selection()
        if not selection:
            return
        selected_index = int(selection[0])
        if selected_index == self.selected_element_index:
            return
        self.sync_selected_element_from_form()
        self.selected_element_index = selected_index
        self.populate_element_detail()
        self.redraw_overlays()

    def add_element(self, element_type: str, immediate_save: bool = True) -> None:
        if element_type not in ELEMENT_TYPES:
            element_type = "obj"
        self.sync_caption_from_form()
        elements = self.current_caption["compositional_deconstruction"]["elements"]
        element: dict[str, Any] = {"type": element_type, "desc": ""}
        if element_type == "text":
            element["text"] = ""
        elements.append(element)
        self.selected_element_index = len(elements) - 1
        self.draw_mode.set(True)
        self.populate_elements_tree()
        self.populate_element_detail()
        self.redraw_overlays()
        if immediate_save:
            self.mark_dirty(immediate=True)
            self.focus_element_description()
        else:
            self.dirty = True
            self.status_var.set("Unsaved changes...")

    def focus_element_description(self) -> None:
        self.element_desc_text.configure(state="normal")
        self.element_desc_text.focus_set()
        self.element_desc_text.mark_set("insert", "end-1c")
        self.element_desc_text.see("insert")

    def delete_selected_element(self) -> None:
        if self.selected_element_index is None:
            return
        elements = self.current_caption["compositional_deconstruction"]["elements"]
        if not (0 <= self.selected_element_index < len(elements)):
            return
        del elements[self.selected_element_index]
        if not elements:
            self.selected_element_index = None
        else:
            self.selected_element_index = min(self.selected_element_index, len(elements) - 1)
        self.populate_elements_tree()
        self.populate_element_detail()
        self.redraw_overlays()
        self.mark_dirty(immediate=True)

    def remove_selected_bbox(self) -> None:
        element = self.get_selected_element()
        if element is None:
            return
        element.pop("bbox", None)
        self.populate_element_detail()
        self.populate_elements_tree()
        self.redraw_overlays()
        self.mark_dirty(immediate=True)

    def fit_selected_bbox_to_image(self) -> None:
        self.set_selected_bbox([0, 0, 1000, 1000], mark=True)

    def open_captioning_preferences(self) -> None:
        dialog = CaptioningPreferencesDialog(self, self.captioning_settings)
        self.wait_window(dialog)
        if dialog.result is None:
            return
        self.captioning_settings = dialog.result
        try:
            path = save_settings(self.captioning_settings)
        except OSError as exc:
            messagebox.showerror("Preferences not saved", str(exc))
            return
        self.status_var.set(f"Saved auto-captioning preferences to {path.name}.")

    def auto_generate_text_caption(self) -> None:
        self.save_current()
        self.ensure_original_extension_for_ai()
        self.start_ai_job("text", "text captions")

    def auto_generate_json_from_text(self) -> None:
        self.save_current()
        self.ensure_original_extension_for_ai()
        self.start_ai_job("json_text", "JSON captions from text")

    def auto_generate_json_from_image(self) -> None:
        self.start_ai_job("json_image", "JSON captions from image")

    def auto_generate_bboxes(self) -> None:
        self.start_ai_job("bboxes", "bboxes")

    def failed_image_paths_for_ai(self) -> list[Path]:
        if self.store is None or not self.images:
            return []
        selection = self.current_image_selection_paths()
        candidates = selection if selection else []
        if not candidates and self.image_sort_var.get() == IMAGE_SORT_FAILED:
            candidates = list(self.images)
        if not candidates and 0 <= self.current_index < len(self.images):
            candidates = [self.images[self.current_index]]
        return [image_path for image_path in candidates if self.store.has_failure_marker(image_path)]

    def retry_failed_captions(self) -> None:
        image_paths = self.failed_image_paths_for_ai()
        if not image_paths:
            self.status_var.set("Select one or more images with failure markers.")
            return
        self.start_ai_job("retry_failed", "retry failed captions", image_paths=image_paths)

    def clear_failed_markers(self) -> None:
        if self.store is None:
            self.status_var.set("Open an image folder before clearing failure markers.")
            return
        image_paths = self.failed_image_paths_for_ai()
        if not image_paths:
            self.status_var.set("Select one or more images with failure markers.")
            return
        if len(image_paths) > 1 and not messagebox.askyesno(
            "Clear failure markers",
            f"Clear failure markers for {len(image_paths)} image(s)?",
        ):
            return
        cleared = 0
        try:
            for image_path in image_paths:
                if self.store.clear_failure_marker(image_path):
                    cleared += 1
        except OSError as exc:
            messagebox.showerror("Clear failed", str(exc))
            return
        self.refresh_after_ai_job(f"Cleared {cleared} failure marker(s).")

    def ensure_original_extension_for_ai(self) -> None:
        if self.current_index < 0 or self.current_index >= len(self.images):
            return
        extension = self.selected_original_extension()
        if extension is None:
            self.original_extension_var.set(".txt")
            extension = ".txt"
        image_path = self.images[self.current_index]
        if self.store is not None and image_path.with_suffix(extension) == self.store.caption_path(image_path):
            self.original_extension_var.set(".original")
        self.load_original_text(image_path)

    def set_ai_controls_running(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        for button in self.ai_buttons:
            button.configure(state=state)
        if self.ai_cancel_button is not None:
            self.ai_cancel_button.configure(state="normal" if running else "disabled")

    def cancel_ai_job(self) -> None:
        if self.ai_worker is None or not self.ai_worker.is_alive() or self.ai_cancel_event is None:
            self.status_var.set("No auto-captioning job is running.")
            return
        self.ai_cancel_event.set()
        if self.ai_cancel_button is not None:
            self.ai_cancel_button.configure(state="disabled")
        self.status_var.set("Cancel requested. Waiting for the current model request to finish...")

    def show_ai_progress(self, label: str, total: int) -> None:
        maximum = max(1, total)
        self.ai_progress_bar.configure(maximum=maximum)
        self.ai_progress_var.set(0)
        self.ai_progress_text_var.set(f"{label}: 0/{total}")
        self.ai_progress_frame.grid()

    def update_ai_progress(self, label: str, current: int, total: int) -> None:
        total = max(1, total)
        current = max(0, min(current, total))
        self.ai_progress_bar.configure(maximum=total)
        self.ai_progress_var.set(current)
        self.ai_progress_text_var.set(f"{label}: {current}/{total}")
        self.ai_progress_frame.grid()

    def selected_image_paths_for_ai(self) -> list[Path]:
        if not self.images:
            return []
        selection = self.current_image_selection_paths()
        if selection:
            return selection
        if 0 <= self.current_index < len(self.images):
            return [self.images[self.current_index]]
        return []

    def writable_original_path_for_image(self, image_path: Path) -> Path:
        original_path = self.original_path_for_image(image_path)
        if original_path is not None and (self.store is None or self.store.caption_path(image_path) != original_path):
            return original_path
        extension = ".txt"
        if self.store is not None and self.store.caption_path(image_path).suffix.lower() == extension:
            extension = ".original"
        return image_path.with_suffix(extension)

    def read_text_file_if_exists(self, path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8-sig").strip()

    def file_snapshot(self, path: Path) -> dict[str, Any]:
        if path.exists():
            return {"path": str(path), "exists": True, "text": path.read_text(encoding="utf-8-sig")}
        return {"path": str(path), "exists": False, "text": ""}

    def capture_ai_undo_snapshot(self, targets: list[dict[str, Path]], label: str) -> dict[str, Any]:
        return {
            "label": label,
            "items": [
                {
                    "image_path": str(target["image_path"]),
                    "caption": self.file_snapshot(target["caption_path"]),
                    "original": self.file_snapshot(target["original_path"]),
                    "failure": self.file_snapshot(target["failure_path"]),
                }
                for target in targets
            ],
        }

    def restore_file_snapshot(self, snapshot: dict[str, Any]) -> None:
        path = Path(snapshot["path"])
        if snapshot.get("exists"):
            path.write_text(str(snapshot.get("text", "")), encoding="utf-8")
        elif path.exists():
            path.unlink()

    def undo_last_ai_job(self) -> None:
        if self.ai_worker is not None and self.ai_worker.is_alive():
            self.status_var.set("Wait for the current auto-captioning job to finish before undoing.")
            return
        if not self.ai_undo_stack:
            self.status_var.set("No AI captioning job to undo.")
            return
        snapshot = self.ai_undo_stack[-1]
        if not messagebox.askyesno("Undo AI captioning", f"Restore sidecars changed by the last {snapshot['label']} job?"):
            return
        try:
            for item in snapshot["items"]:
                self.restore_file_snapshot(item["caption"])
                self.restore_file_snapshot(item["original"])
                if "failure" in item:
                    self.restore_file_snapshot(item["failure"])
        except OSError as exc:
            messagebox.showerror("Undo failed", str(exc))
            return
        self.ai_undo_stack.pop()
        self.refresh_after_ai_job(f"Undid last {snapshot['label']} job.")

    def start_ai_job(self, operation: str, label: str, image_paths: list[Path] | None = None) -> None:
        if self.ai_worker is not None and self.ai_worker.is_alive():
            self.status_var.set("Auto-captioning is already running.")
            return
        if self.store is None or self.folder is None:
            self.status_var.set("Open an image folder before auto-captioning.")
            return

        self.flush_pending_image_selection()
        image_paths = image_paths if image_paths is not None else self.selected_image_paths_for_ai()
        if not image_paths:
            self.status_var.set("Select at least one image.")
            return
        if len(image_paths) > 1:
            confirmed = messagebox.askyesno(
                "Confirm batch auto-captioning",
                f"Run {label} on {len(image_paths)} selected images?",
            )
            if not confirmed:
                return

        self.save_current()
        caption_extension = self.caption_extension_var.get()
        targets = [
            {
                "image_path": image_path,
                "caption_path": image_path.with_suffix(caption_extension),
                "original_path": self.writable_original_path_for_image(image_path),
                "failure_path": image_path.with_suffix(".caption_failed.json"),
            }
            for image_path in image_paths
        ]
        snapshot = self.capture_ai_undo_snapshot(targets, label)
        settings = self.captioning_settings
        folder = self.folder
        managed_process = self.ai_server_process
        managed_command = self.ai_server_command
        managed_phase = self.ai_server_phase
        self.ai_server_process = None
        self.ai_server_command = None
        self.ai_server_phase = None
        self.ai_cancel_event = threading.Event()

        self.set_ai_controls_running(True)
        self.show_ai_progress(label, len(targets))
        self.status_var.set(f"Starting {label} for {len(targets)} image(s)...")
        self.ai_worker = threading.Thread(
            target=self.ai_worker_main,
            args=(
                operation,
                label,
                targets,
                snapshot,
                settings,
                folder,
                caption_extension,
                managed_process,
                managed_command,
                managed_phase,
                self.ai_cancel_event,
            ),
            daemon=True,
        )
        self.ai_worker.start()
        self.after(100, self.poll_ai_queue)

    def ai_worker_main(
        self,
        operation: str,
        label: str,
        targets: list[dict[str, Path]],
        snapshot: dict[str, Any],
        settings: CaptioningSettings,
        folder: Path,
        caption_extension: str,
        managed_process: subprocess.Popen | None,
        managed_command: str | None,
        managed_phase: str | None,
        cancel_event: threading.Event,
    ) -> None:
        store = CaptionStore(folder, caption_extension)
        ok = 0
        failed = 0
        changed = 0
        bbox_attempted_total = 0
        bbox_written_total = 0
        bbox_reason_totals: dict[str, int] = {}
        errors: list[str] = []
        canceled = False

        class AiJobCanceled(Exception):
            pass

        def progress(message: str | None = None, current: int | None = None, total: int | None = None) -> None:
            event: dict[str, Any] = {"type": "progress", "label": label}
            if message is not None:
                event["message"] = message
            if current is not None:
                event["current"] = current
            if total is not None:
                event["total"] = total
            self.ai_queue.put(event)

        def ensure_endpoint_exposes_model(config_label: str, api_model: str) -> None:
            try:
                model_ids = server_model_ids(settings.base_url, settings.api_key, timeout=5.0)
            except Exception as exc:
                raise AutoCaptionError(f"Captioning server is not reachable at {settings.base_url}: {exc}") from exc
            if api_model and model_ids and api_model not in model_ids:
                available = ", ".join(sorted(model_ids)) or "(none)"
                raise AutoCaptionError(
                    f"Server at {settings.base_url} is running, but it does not expose the selected model alias "
                    f"'{api_model}' for {config_label}. Available aliases: {available}. "
                    "Stop the existing server, change the API model name/profile, or switch Preferences to "
                    "'Connect to existing server' with the matching alias."
                )

        def format_bbox_reasons(reason_counts: dict[str, int]) -> str:
            if not reason_counts:
                return ""
            parts = [f"{reason}={count}" for reason, count in sorted(reason_counts.items())]
            return " (" + "; ".join(parts) + ")"

        def record_bbox_result(image_name: str, attempted: int, written: int, reason_counts: dict[str, int]) -> None:
            nonlocal bbox_attempted_total, bbox_written_total
            bbox_attempted_total += attempted
            bbox_written_total += written
            for reason, count in reason_counts.items():
                bbox_reason_totals[reason] = bbox_reason_totals.get(reason, 0) + count
            progress(
                f"{image_name}: bbox targets={attempted}, written={written}, "
                f"skipped={attempted - written}{format_bbox_reasons(reason_counts)}"
            )

        def prepare_task(task: str) -> None:
            nonlocal managed_process, managed_command, managed_phase
            if cancel_event.is_set():
                raise AiJobCanceled()
            config = runtime_config_for_task(settings, task)
            if settings.server_start_mode == "existing" or not settings.auto_start_server:
                ensure_endpoint_exposes_model(config.label, config.api_model)
                return

            command_task = task
            if managed_process is not None and managed_process.poll() is not None:
                managed_process = None
                managed_command = None
                managed_phase = None
            if managed_process is None and is_server_ready(settings.base_url, settings.api_key, timeout=3.0):
                ensure_endpoint_exposes_model(config.label, config.api_model)
                progress(f"Endpoint is already running with {config.api_model}.")
                return
            if settings.server_start_mode == "local":
                server_path = Path(settings.llama_server_path).expanduser() if settings.llama_server_path.strip() else find_llama_server()
                if server_path is None or not server_path.exists():
                    raise AutoCaptionError("Choose llama-server.exe in Preferences before using local captioning.")
                if config.kind == "server":
                    raise AutoCaptionError("Existing server alias profiles require the 'Connect to existing server' runtime.")

            progress(f"Checking {config.label} assets...")
            assets = ensure_model_assets(settings, task, progress)
            if settings.server_start_mode == "local":
                command = build_llama_server_command(settings, task, assets)
            else:
                template = settings.caption_server_command if command_task == "caption" else settings.bbox_server_command
                command = format_server_command(template, settings, task, assets)

            if managed_process is not None and managed_command != command:
                progress(f"Stopping {managed_phase or 'previous'} server...")
                stop_server_process(managed_process)
                managed_process = None
                managed_command = None
                managed_phase = None
            if managed_process is not None:
                return
            name = "caption" if command_task == "caption" else "bbox"
            managed_process = start_server_process(
                command=command,
                base_url=settings.base_url,
                api_key=settings.api_key,
                log_dir=Path(settings.models_dir).expanduser().resolve() / "server_logs",
                name=f"{name}_server",
                startup_timeout=settings.server_startup_timeout,
                progress=progress,
            )
            managed_command = command
            managed_phase = name

        def preview_text(text: str, limit: int = 4000) -> str:
            text = text.strip()
            if len(text) <= limit:
                return text
            return text[:limit].rstrip() + f"\n... truncated {len(text) - limit} character(s)"

        def model_alias_for_task(task: str) -> str:
            try:
                return runtime_config_for_task(settings, task).api_model
            except Exception:
                return ""

        def classify_failure(exc: Exception, task: str) -> str:
            if isinstance(exc, ModelJsonError):
                return "bbox_json_parse_failed" if task == "bbox" else "caption_json_parse_failed"
            text = str(exc).lower()
            if "no source text caption" in text:
                return "missing_source_text"
            if task == "bbox" and "response must contain" in text:
                return "bbox_json_parse_failed"
            if "server" in text or "endpoint" in text or "llama" in text:
                return "server_failed"
            if "model request failed" in text or "returned an empty response" in text:
                return "model_request_failed"
            return "auto_caption_failed"

        def build_failure_marker(
            image_path: Path,
            target: dict[str, Path],
            effective_operation: str,
            task: str,
            exc: Exception,
        ) -> dict[str, Any]:
            retry_operation = (
                "bboxes" if task == "bbox" and effective_operation in {"json_text", "json_image"} else effective_operation
            )
            marker: dict[str, Any] = {
                "version": 1,
                "image": image_path.name,
                "caption_path": target["caption_path"].name,
                "operation": retry_operation,
                "source_operation": effective_operation,
                "label": label,
                "task": task,
                "reason": classify_failure(exc, task),
                "model": model_alias_for_task(task),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
                "can_retry": True,
            }
            if isinstance(exc, ModelJsonError):
                if exc.raw_output:
                    marker["raw_output_preview"] = preview_text(exc.raw_output)
                if exc.repair_output:
                    marker["repair_output_preview"] = preview_text(exc.repair_output)
            return marker

        def retry_operation_for_marker(image_path: Path) -> str:
            marker = store.load_failure_marker(image_path)
            operation_from_marker = str(marker.get("operation", "")).strip() if marker else ""
            if operation_from_marker in {"text", "json_text", "json_image", "bboxes"}:
                return operation_from_marker
            return "json_image"

        try:
            for index, target in enumerate(targets, start=1):
                if cancel_event.is_set():
                    canceled = True
                    progress(f"Canceled {label} after {ok + failed}/{len(targets)} image(s).", current=ok + failed, total=len(targets))
                    break
                image_path = target["image_path"]
                effective_operation = retry_operation_for_marker(image_path) if operation == "retry_failed" else operation
                failure_task = "caption"
                item_changed = False
                try:
                    progress(f"{label}: {index}/{len(targets)} {image_path.name}", current=index - 1, total=len(targets))
                    if operation == "retry_failed":
                        progress(f"{image_path.name}: retrying previous {effective_operation} failure.")
                    if effective_operation == "text":
                        failure_task = "caption"
                        prepare_task("caption")
                        caption_text = generate_plain_caption(settings, image_path)
                        target["original_path"].write_text(caption_text, encoding="utf-8")
                        item_changed = True
                    elif effective_operation == "json_text":
                        failure_task = "caption"
                        source_text = self.read_text_file_if_exists(target["original_path"])
                        if not source_text:
                            raise AutoCaptionError(f"No source text caption found for {image_path.name}.")
                        prepare_task("caption")
                        caption = generate_json_from_text(settings, source_text, progress=progress)
                        store.save_caption(image_path, caption)
                        item_changed = True
                        if settings.add_bboxes_after_json:
                            failure_task = "bbox"
                            prepare_task("bbox")
                            caption, attempted, added, reasons = add_bboxes_to_caption(
                                settings,
                                image_path,
                                caption,
                                progress=progress,
                            )
                            store.save_caption(image_path, caption)
                            record_bbox_result(image_path.name, attempted, added, reasons)
                    elif effective_operation == "json_image":
                        failure_task = "caption"
                        prepare_task("caption")
                        caption = generate_json_from_image(settings, image_path, progress=progress)
                        store.save_caption(image_path, caption)
                        item_changed = True
                        if settings.add_bboxes_after_json:
                            failure_task = "bbox"
                            prepare_task("bbox")
                            caption, attempted, added, reasons = add_bboxes_to_caption(
                                settings,
                                image_path,
                                caption,
                                progress=progress,
                            )
                            store.save_caption(image_path, caption)
                            record_bbox_result(image_path.name, attempted, added, reasons)
                    elif effective_operation == "bboxes":
                        failure_task = "bbox"
                        if not target["caption_path"].exists() or not target["caption_path"].read_text(encoding="utf-8-sig").strip():
                            raise AutoCaptionError(f"No structured caption found for {image_path.name}.")
                        caption, message = store.load_caption(image_path)
                        if message and "Could not parse" in message:
                            raise AutoCaptionError(message)
                        prepare_task("bbox")
                        caption, attempted, added, reasons = add_bboxes_to_caption(
                            settings,
                            image_path,
                            caption,
                            progress=progress,
                        )
                        store.save_caption(image_path, caption)
                        item_changed = True
                        record_bbox_result(image_path.name, attempted, added, reasons)
                    else:
                        raise AutoCaptionError(f"Unknown auto-captioning operation: {effective_operation}")
                    if store.clear_failure_marker(image_path):
                        item_changed = True
                    ok += 1
                    if item_changed:
                        changed += 1
                except AiJobCanceled:
                    canceled = True
                    progress(f"Canceled {label} after {ok + failed}/{len(targets)} image(s).", current=ok + failed, total=len(targets))
                    break
                except Exception as exc:
                    failed += 1
                    error = f"{image_path.name}: {exc}"
                    try:
                        store.save_failure_marker(
                            image_path,
                            build_failure_marker(image_path, target, effective_operation, failure_task, exc),
                        )
                        item_changed = True
                    except OSError as marker_exc:
                        error += f" (also could not write failure marker: {marker_exc})"
                    errors.append(error)
                    progress(error)
                    if item_changed:
                        changed += 1
                progress(current=index, total=len(targets))
        finally:
            if settings.stop_server_after_job and managed_process is not None:
                progress("Stopping auto-started server...")
                stop_server_process(managed_process)
                managed_process = None
                managed_command = None
                managed_phase = None

        self.ai_queue.put(
            {
                "type": "done",
                "label": label,
                "total": len(targets),
                "ok": ok,
                "failed": failed,
                "errors": errors,
                "changed": changed,
                "canceled": canceled,
                "bbox_attempted": bbox_attempted_total,
                "bbox_written": bbox_written_total,
                "bbox_reasons": bbox_reason_totals,
                "snapshot": snapshot,
                "server_process": managed_process,
                "server_command": managed_command,
                "server_phase": managed_phase,
            }
        )

    def poll_ai_queue(self) -> None:
        final_message: str | None = None
        while True:
            try:
                event = self.ai_queue.get_nowait()
            except queue.Empty:
                break

            if event.get("type") == "progress":
                if "message" in event:
                    self.status_var.set(str(event.get("message", "")))
                if "current" in event and "total" in event:
                    self.update_ai_progress(
                        str(event.get("label", "Auto-captioning")),
                        int(event.get("current", 0)),
                        int(event.get("total", 1)),
                    )
            elif event.get("type") == "done":
                self.ai_server_process = event.get("server_process")
                self.ai_server_command = event.get("server_command")
                self.ai_server_phase = event.get("server_phase")
                changed = int(event.get("changed", 0))
                if changed:
                    self.ai_undo_stack.append(event["snapshot"])
                ok = int(event.get("ok", 0))
                failed = int(event.get("failed", 0))
                total = int(event.get("total", ok + failed))
                canceled = bool(event.get("canceled", False))
                errors = [str(error) for error in event.get("errors", [])]
                bbox_attempted = int(event.get("bbox_attempted", 0))
                bbox_written = int(event.get("bbox_written", 0))
                bbox_reasons = event.get("bbox_reasons", {})
                bbox_summary = ""
                if bbox_attempted:
                    reason_summary = ""
                    if isinstance(bbox_reasons, dict) and bbox_reasons:
                        parts = [f"{reason}={count}" for reason, count in sorted(bbox_reasons.items())]
                        reason_summary = " Reasons: " + "; ".join(parts) + "."
                    bbox_summary = (
                        f" BBoxes: targets={bbox_attempted}, written={bbox_written}, "
                        f"skipped={bbox_attempted - bbox_written}.{reason_summary}"
                    )
                if errors:
                    first_error = errors[0]
                    final_message = f"Finished {event['label']}: ok={ok}, failed={failed}.{bbox_summary} First error: {first_error}"
                    details = "\n".join(errors[:10])
                    if len(errors) > 10:
                        details += f"\n... plus {len(errors) - 10} more"
                    messagebox.showerror("Auto-captioning failed", details)
                elif canceled:
                    final_message = f"Canceled {event['label']}: ok={ok}, failed={failed}.{bbox_summary}"
                else:
                    final_message = f"Finished {event['label']}: ok={ok}, failed={failed}.{bbox_summary}"
                self.update_ai_progress(str(event["label"]), ok + failed, total)

        if self.ai_worker is not None and self.ai_worker.is_alive():
            self.after(100, self.poll_ai_queue)
            return

        self.set_ai_controls_running(False)
        if final_message:
            self.refresh_after_ai_job(final_message)
        self.ai_worker = None
        self.ai_cancel_event = None

    def refresh_after_ai_job(self, message: str) -> None:
        selected_paths = self.current_image_selection_paths()
        self.apply_image_sort(preserve_current=True)
        self.populate_image_list()
        self.populate_image_selection(selected_paths=selected_paths)
        if 0 <= self.current_index < len(self.images):
            self.load_image_at(self.current_index, preserve_selection=True, skip_save=True, force_reload=True)
        elif not self.images:
            self.clear_loaded_image(f"{message} No images match {self.image_sort_var.get()}.")
            return
        self.status_var.set(message)

    def mark_dirty(self, immediate: bool = False) -> None:
        if self.loading_form:
            return
        self.dirty = True
        if immediate:
            self.save_current()
            return
        self.status_var.set("Unsaved changes...")
        if self.autosave_job:
            self.after_cancel(self.autosave_job)
        self.autosave_job = self.after(700, self.autosave)

    def autosave(self) -> None:
        self.autosave_job = None
        self.save_current()

    def mark_original_dirty(self, immediate: bool = False) -> None:
        if self.loading_original:
            return
        self.original_dirty = True
        if immediate:
            self.save_current()
            return
        self.status_var.set("Unsaved original changes...")
        if self.original_autosave_job:
            self.after_cancel(self.original_autosave_job)
        self.original_autosave_job = self.after(700, self.autosave_original)

    def autosave_original(self) -> None:
        self.original_autosave_job = None
        self.save_current()

    def save_current_original(self) -> Path | None:
        if not self.original_dirty:
            return None
        original_path = self.current_original_path
        if original_path is None:
            self.original_dirty = False
            return None
        original_path.write_text(self._get_text(self.original_text), encoding="utf-8")
        self.original_dirty = False
        self.original_status_var.set(f"Original: {original_path.name}")
        return original_path

    def save_current(self) -> None:
        if self.autosave_job:
            self.after_cancel(self.autosave_job)
            self.autosave_job = None
        if self.original_autosave_job:
            self.after_cancel(self.original_autosave_job)
            self.original_autosave_job = None
        if self.store is None or self.current_index < 0 or self.current_index >= len(self.images):
            return
        selected_paths = self.current_image_selection_paths()
        self.sync_caption_from_form()
        image_path = self.images[self.current_index]
        saved_names: list[str] = []
        try:
            caption_path = self.store.save_caption(image_path, self.current_caption)
            saved_names.append(caption_path.name)
            original_path = self.save_current_original()
            if original_path is not None:
                saved_names.append(original_path.name)
        except OSError as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        self.dirty = False
        self.populate_image_list()
        self.populate_image_selection(selected_paths=selected_paths)
        self.status_var.set(f"Saved {', '.join(saved_names)} at {time.strftime('%H:%M:%S')}")

    def copy_current_image_to_edit(self) -> None:
        if self.current_index < 0 or self.current_index >= len(self.images):
            self.status_var.set("Select an image to copy.")
            return
        source_path = self.images[self.current_index]
        edit_folder = source_path.parent / "edit"
        target_path = edit_folder / source_path.name
        existed = target_path.exists()
        try:
            edit_folder.mkdir(exist_ok=True)
            shutil.copy2(source_path, target_path)
        except OSError as exc:
            messagebox.showerror("Copy failed", str(exc))
            return
        action = "Updated" if existed else "Copied"
        self.status_var.set(f"{action} {source_path.name} in {edit_folder.name}.")

    def on_caption_extension_changed(self, _event: tk.Event) -> None:
        self.save_current()
        if self.folder is None:
            return
        current_path = self.images[self.current_index] if 0 <= self.current_index < len(self.images) else None
        self.store = CaptionStore(self.folder, self.caption_extension_var.get())
        self.apply_image_sort(preserve_current=True)
        self.populate_image_list()
        if current_path is not None and current_path in self.images:
            index = self.images.index(current_path)
            self.current_index = -1
            self.load_image_at(index)
        elif self.images:
            self.current_index = -1
            self.load_image_at(0)
        else:
            self.clear_loaded_image(f"No images match {self.image_sort_var.get()}.")

    def on_original_extension_changed(self, _event: tk.Event) -> None:
        try:
            self.save_current_original()
        except OSError as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        if self.current_index >= 0 and self.current_index < len(self.images):
            self.load_original_text(self.images[self.current_index])
        else:
            extension = self.selected_original_extension()
            self.clear_original_text("Original: none" if extension is None else f"Original: {extension}")
        if self.image_sort_var.get() == IMAGE_SORT_ORIGINAL_MISSING:
            selected_paths = self.current_image_selection_paths()
            self.apply_image_sort(preserve_current=True)
            self.populate_image_list()
            self.populate_image_selection(selected_paths=selected_paths)

    def next_image(self) -> None:
        self.load_image_at(min(self.current_index + 1, len(self.images) - 1))

    def previous_image(self) -> None:
        self.load_image_at(max(self.current_index - 1, 0))

    def next_image_from_key(self, event: tk.Event | None = None) -> str | None:
        if event is not None and event.state & CONTROL_MASK:
            return None
        if self.focus_allows_navigation():
            self.next_image()
        return None

    def previous_image_from_key(self, event: tk.Event | None = None) -> str | None:
        if event is not None and event.state & CONTROL_MASK:
            return None
        if self.focus_allows_navigation():
            self.previous_image()
        return None

    def next_image_always(self, _event: tk.Event | None = None) -> str:
        self.save_and_next()
        return "break"

    def previous_image_always(self, _event: tk.Event | None = None) -> str:
        self.save_and_previous()
        return "break"

    def save_and_next(self) -> None:
        self.save_current()
        if self.current_index < len(self.images) - 1:
            self.load_image_at(self.current_index + 1)

    def save_and_previous(self) -> None:
        self.save_current()
        if self.current_index > 0:
            self.load_image_at(self.current_index - 1)

    def on_return_key(self, event: tk.Event) -> str | None:
        if event.state & SHIFT_MASK:
            return None
        widget = self.focus_get()
        if widget is not None and widget.winfo_class() in {"Button", "TButton"}:
            return None
        self.save_and_next()
        return "break"

    def on_text_return_key(self, event: tk.Event) -> str | None:
        if event.state & SHIFT_MASK:
            return None
        self.save_and_next()
        return "break"

    def on_escape_key(self, _event: tk.Event) -> str | None:
        if self.cancel_eyedrop(update_status=True):
            return "break"
        self.exit_fullscreen()
        return None

    def focus_allows_navigation(self) -> bool:
        widget = self.focus_get()
        if widget is None:
            return True
        blocked = {"Text", "Entry", "TEntry", "TCombobox", "Spinbox", "TSpinbox"}
        return widget.winfo_class() not in blocked

    def render_image(self) -> None:
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())

        if self.pil_image is None:
            self.canvas.delete("all")
            self.canvas_image_id = None
            self.image_render_key = None
            self.tk_image = None
            self.canvas.create_text(
                width / 2,
                height / 2,
                text="Open a folder of images",
                fill="#8f8f8f",
                font=("Segoe UI", 16),
            )
            return

        image_width, image_height = self.pil_image.size
        scale = min(width / image_width, height / image_height)
        display_width = max(1, int(image_width * scale))
        display_height = max(1, int(image_height * scale))
        left = (width - display_width) / 2
        top = (height - display_height) / 2
        self.image_bounds = (left, top, display_width, display_height)

        render_key = (id(self.pil_image), width, height)
        if render_key != self.image_render_key or self.canvas_image_id is None:
            self.canvas.delete("all")
            resized = self.pil_image.resize((display_width, display_height), RESAMPLE)
            self.tk_image = ImageTk.PhotoImage(resized)
            self.canvas_image_id = self.canvas.create_image(left, top, anchor="nw", image=self.tk_image)
            self.image_render_key = render_key
        else:
            self.canvas.coords(self.canvas_image_id, left, top)
        self.redraw_overlays()

    def redraw_overlays(self) -> None:
        self.canvas.delete("overlay")
        self.draw_bboxes()

    def draw_bboxes(self) -> None:
        elements = self.current_caption.get("compositional_deconstruction", {}).get("elements", [])
        obj_color_index = 0
        for index, element in enumerate(elements):
            element_type = element.get("type", "obj")
            color = self.element_box_color(element, obj_color_index)
            if element_type == "obj":
                obj_color_index += 1

            bbox = normalize_bbox(element.get("bbox"))
            if not bbox:
                continue
            x1, y1, x2, y2 = self.bbox_to_canvas(bbox)
            selected = index == self.selected_element_index
            width = 3 if selected else 2
            if selected:
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="#ffffff", width=5, tags=("overlay",))
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width, tags=("overlay",))
            label = f"#{index + 1} {element_type}"
            if element_type == "text" and element.get("text"):
                label = f"#{index + 1} {element['text'][:24]}"
            label_width = min(280, 8 * len(label) + 18)
            self.canvas.create_rectangle(
                x1,
                max(0, y1 - 23),
                x1 + label_width,
                y1,
                fill=color,
                outline="#ffffff" if selected else "",
                tags=("overlay",),
            )
            self.canvas.create_text(
                x1 + 7,
                y1 - 12,
                anchor="w",
                text=label,
                fill=BOX_LABEL_TEXT_COLOR,
                font=("Segoe UI", 9, "bold"),
                tags=("overlay",),
            )
            if selected:
                for hx, hy in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
                    self.canvas.create_rectangle(
                        hx - 5,
                        hy - 5,
                        hx + 5,
                        hy + 5,
                        fill="#ffffff",
                        outline=BOX_LABEL_TEXT_COLOR,
                        tags=("overlay",),
                    )
                for hx, hy, horizontal in (
                    ((x1 + x2) / 2, y1, True),
                    ((x1 + x2) / 2, y2, True),
                    (x1, (y1 + y2) / 2, False),
                    (x2, (y1 + y2) / 2, False),
                ):
                    if horizontal:
                        self.canvas.create_rectangle(
                            hx - 8,
                            hy - 4,
                            hx + 8,
                            hy + 4,
                            fill="#ffffff",
                            outline=BOX_LABEL_TEXT_COLOR,
                            tags=("overlay",),
                        )
                    else:
                        self.canvas.create_rectangle(
                            hx - 4,
                            hy - 8,
                            hx + 4,
                            hy + 8,
                            fill="#ffffff",
                            outline=BOX_LABEL_TEXT_COLOR,
                            tags=("overlay",),
                        )

    def bbox_to_canvas(self, bbox: list[int]) -> tuple[float, float, float, float]:
        left, top, width, height = self.image_bounds
        y1, x1, y2, x2 = bbox
        return (
            left + (x1 / 1000) * width,
            top + (y1 / 1000) * height,
            left + (x2 / 1000) * width,
            top + (y2 / 1000) * height,
        )

    def canvas_to_norm(self, x: float, y: float) -> tuple[int, int]:
        left, top, width, height = self.image_bounds
        norm_x = int(round(((x - left) / width) * 1000))
        norm_y = int(round(((y - top) / height) * 1000))
        return max(0, min(1000, norm_y)), max(0, min(1000, norm_x))

    def should_draw_selected_box(self) -> bool:
        if not self.draw_mode.get():
            return False
        element = self.get_selected_element()
        if element is None:
            return False
        return normalize_bbox(element.get("bbox")) is None

    def point_inside_image(self, x: float, y: float) -> bool:
        left, top, width, height = self.image_bounds
        return left <= x <= left + width and top <= y <= top + height

    def elements_at_canvas(self, x: float, y: float) -> list[int]:
        elements = self.current_caption.get("compositional_deconstruction", {}).get("elements", [])
        hits: list[int] = []
        for index in reversed(range(len(elements))):
            bbox = normalize_bbox(elements[index].get("bbox"))
            if not bbox:
                continue
            x1, y1, x2, y2 = self.bbox_to_canvas(bbox)
            if x1 <= x <= x2 and y1 <= y <= y2:
                hits.append(index)
        return hits

    def element_at_canvas(self, x: float, y: float) -> int | None:
        hits = self.elements_at_canvas(x, y)
        return hits[0] if hits else None

    def cycle_element_at_canvas(self, x: float, y: float) -> None:
        hits = self.elements_at_canvas(x, y)
        if not hits:
            self.selected_element_index = None
            self.populate_elements_tree()
            self.populate_element_detail()
            self.redraw_overlays()
            return

        if self.selected_element_index in hits:
            next_index = hits[(hits.index(self.selected_element_index) + 1) % len(hits)]
        else:
            next_index = hits[0]
        self.select_element(next_index)
        if len(hits) > 1:
            self.status_var.set(f"Selected element #{next_index + 1}; {len(hits)} boxes overlap here.")

    def hit_handle(self, x: float, y: float) -> str | None:
        element = self.get_selected_element()
        if element is None:
            return None
        bbox = normalize_bbox(element.get("bbox"))
        if not bbox:
            return None
        x1, y1, x2, y2 = self.bbox_to_canvas(bbox)
        handles = {
            "nw": (x1, y1),
            "ne": (x2, y1),
            "sw": (x1, y2),
            "se": (x2, y2),
        }
        for name, (hx, hy) in handles.items():
            if abs(x - hx) <= RESIZE_HIT_TOLERANCE and abs(y - hy) <= RESIZE_HIT_TOLERANCE:
                return name
        if x1 <= x <= x2:
            if abs(y - y1) <= RESIZE_HIT_TOLERANCE:
                return "n"
            if abs(y - y2) <= RESIZE_HIT_TOLERANCE:
                return "s"
        if y1 <= y <= y2:
            if abs(x - x1) <= RESIZE_HIT_TOLERANCE:
                return "w"
            if abs(x - x2) <= RESIZE_HIT_TOLERANCE:
                return "e"
        if x1 <= x <= x2 and y1 <= y <= y2:
            return "move"
        return None

    def set_canvas_cursor(self, cursor: str) -> None:
        if self.eyedrop_target is not None:
            cursor = "crosshair"
        if cursor == self.current_canvas_cursor:
            return
        try:
            self.canvas.configure(cursor=cursor)
            self.current_canvas_cursor = cursor
        except tk.TclError:
            self.canvas.configure(cursor="")
            self.current_canvas_cursor = ""

    def set_canvas_cursor_from_candidates(self, candidates: tuple[str, ...]) -> None:
        for cursor in candidates:
            try:
                self.canvas.configure(cursor=cursor)
                self.current_canvas_cursor = cursor
                return
            except tk.TclError:
                continue
        self.set_canvas_cursor("")

    def cursor_for_handle(self, handle: str) -> tuple[str, ...]:
        if handle in CORNER_CURSORS:
            return CORNER_CURSORS[handle]
        if handle in EDGE_CURSORS:
            return EDGE_CURSORS[handle]
        if handle == "move":
            return MOVE_CURSORS
        return ("",)

    def on_canvas_motion(self, event: tk.Event) -> None:
        if self.drag_state:
            return
        if self.eyedrop_target is not None:
            self.set_canvas_cursor("crosshair")
            return
        handle = self.hit_handle(event.x, event.y)
        if handle:
            self.set_canvas_cursor_from_candidates(self.cursor_for_handle(handle))
            return
        if self.elements_at_canvas(event.x, event.y):
            self.set_canvas_cursor_from_candidates(MOVE_CURSORS)
            return
        if self.point_inside_image(event.x, event.y):
            self.set_canvas_cursor("crosshair")
            return
        self.set_canvas_cursor("")

    def on_canvas_press(self, event: tk.Event) -> None:
        self.canvas.focus_set()
        if self.pil_image is None:
            return
        if self.apply_eyedrop_sample(event.x, event.y):
            return
        if not self.point_inside_image(event.x, event.y):
            self.selected_element_index = None
            self.populate_elements_tree()
            self.populate_element_detail()
            self.redraw_overlays()
            return
        norm_y, norm_x = self.canvas_to_norm(event.x, event.y)

        if self.should_draw_selected_box():
            self.drag_state = {"mode": "draw", "start": (norm_y, norm_x)}
            self.set_selected_bbox([norm_y, norm_x, norm_y + 1, norm_x + 1], mark=False)
            return

        handle = self.hit_handle(event.x, event.y)
        if handle and handle != "move":
            element = self.get_selected_element()
            bbox = normalize_bbox(element.get("bbox")) if element else None
            if bbox:
                self.drag_state = {"mode": handle, "start": (norm_y, norm_x), "bbox": bbox}
            return

        hits = self.elements_at_canvas(event.x, event.y)
        if hits:
            self.drag_state = {
                "mode": "pending_click",
                "start": (norm_y, norm_x),
                "canvas_start": (event.x, event.y),
                "hits": hits,
            }
            return

        if handle == "move":
            return

        self.drag_state = {
            "mode": "pending_new_box",
            "start": (norm_y, norm_x),
            "canvas_start": (event.x, event.y),
        }

    def on_canvas_drag(self, event: tk.Event) -> None:
        if not self.drag_state:
            return
        if self.drag_state["mode"] == "pending_click":
            start_x, start_y = self.drag_state["canvas_start"]
            if abs(event.x - start_x) < self.drag_threshold and abs(event.y - start_y) < self.drag_threshold:
                return
            hits = self.drag_state["hits"]
            if self.selected_element_index in hits:
                moving_index = self.selected_element_index
            else:
                moving_index = hits[0]
                self.select_element(moving_index)
            element = self.get_selected_element()
            bbox = normalize_bbox(element.get("bbox")) if element else None
            if not bbox:
                self.drag_state = None
                return
            self.drag_state["mode"] = "move"
            self.drag_state["bbox"] = bbox

        if self.drag_state["mode"] == "pending_new_box":
            start_x, start_y = self.drag_state["canvas_start"]
            if abs(event.x - start_x) < self.drag_threshold and abs(event.y - start_y) < self.drag_threshold:
                return
            self.add_element("obj", immediate_save=False)
            self.drag_state["mode"] = "draw"
            self.drag_state["created_element"] = True

        norm_y, norm_x = self.canvas_to_norm(event.x, event.y)
        mode = self.drag_state["mode"]
        if mode == "draw":
            start_y, start_x = self.drag_state["start"]
            self.set_selected_bbox([start_y, start_x, norm_y, norm_x], mark=False)
            return

        original = self.drag_state["bbox"]
        start_y, start_x = self.drag_state["start"]
        y1, x1, y2, x2 = original
        if mode == "move":
            delta_y = norm_y - start_y
            delta_x = norm_x - start_x
            height = y2 - y1
            width = x2 - x1
            new_y1 = max(0, min(1000 - height, y1 + delta_y))
            new_x1 = max(0, min(1000 - width, x1 + delta_x))
            self.set_selected_bbox([new_y1, new_x1, new_y1 + height, new_x1 + width], mark=False)
            return

        if "n" in mode:
            y1 = norm_y
        if "s" in mode:
            y2 = norm_y
        if "w" in mode:
            x1 = norm_x
        if "e" in mode:
            x2 = norm_x
        self.set_selected_bbox([y1, x1, y2, x2], mark=False)

    def on_canvas_release(self, event: tk.Event) -> None:
        if not self.drag_state:
            return
        if self.drag_state["mode"] == "pending_click":
            self.drag_state = None
            self.cycle_element_at_canvas(event.x, event.y)
            return
        if self.drag_state["mode"] == "pending_new_box":
            self.drag_state = None
            self.selected_element_index = None
            self.populate_elements_tree()
            self.populate_element_detail()
            self.redraw_overlays()
            return
        focus_description = bool(self.drag_state.get("created_element"))
        self.drag_state = None
        self.sync_selected_element_from_form()
        self.populate_elements_tree()
        self.redraw_overlays()
        self.mark_dirty(immediate=True)
        if focus_description:
            self.focus_element_description()

    def set_selected_bbox(self, bbox: list[int], mark: bool) -> None:
        element = self.get_selected_element()
        normalized = normalize_bbox(bbox)
        if element is None or normalized is None:
            return
        element["bbox"] = normalized
        self.loading_element = True
        for key, value in zip(("y1", "x1", "y2", "x2"), normalized):
            self.bbox_vars[key].set(str(value))
        self.loading_element = False
        self.redraw_overlays()
        if mark:
            self.populate_elements_tree()
            self.mark_dirty(immediate=True)

    def select_element(self, index: int) -> None:
        self.sync_selected_element_from_form()
        self.selected_element_index = index
        self.populate_elements_tree()
        self.populate_element_detail()
        self.redraw_overlays()

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        self.attributes("-fullscreen", self.fullscreen)

    def exit_fullscreen(self) -> None:
        if self.fullscreen:
            self.fullscreen = False
            self.attributes("-fullscreen", False)

    def on_close(self) -> None:
        self.save_current()
        if self.ai_server_process is not None:
            stop_server_process(self.ai_server_process)
            self.ai_server_process = None
        self.destroy()


def main() -> None:
    try:
        app = CaptionEditorApp()
    except Exception as exc:  # pragma: no cover - startup guard for missing Tk/Pillow issues
        print(f"Failed to start Ideogram Captioner: {exc}", file=sys.stderr)
        raise
    app.mainloop()
