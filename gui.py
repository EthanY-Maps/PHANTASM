"""Consolidated tkinter GUI for the historic map pipeline."""

import os
import sys
import subprocess
import platform
import threading
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Dict, Optional, Any
from PIL import Image, ImageTk

from config import Config
from pipeline import Pipeline
from utils import validate_path
from model_parameters import DEFAULT_PARAMS_FILE, load_parameters
from stages import STAGES
from gui_styles import PALETTE as P, apply_styles
from gui_widgets import LogConsole
from model_params_dialog import open_model_params_dialog


class PipelineGUI:
    WINDOW_SIZE = "1400x920"
    MIN_SIZE    = (1100, 780)
    TITLE       = "Historic Map Pipeline"
    CONFIG_FILE = "config/map_pipeline_config.json"

    def __init__(self) -> None:
        self.root = tk.Tk()
        apply_styles()
        self.setup_window()

        self.config_manager   = Config(self.CONFIG_FILE)
        self.message_queue    = queue.Queue()
        self.pipeline_thread  = None
        self.pipeline_running = False
        self.validation_errors: Dict[str, bool] = {}

        self.setup_variables()
        self.create_widgets()
        self.setup_event_handlers()
        self.process_queue()

        params_file = DEFAULT_PARAMS_FILE
        loaded_config = self.config_manager.load()
        if loaded_config and 'model_parameters_file' in loaded_config:
            params_file = loaded_config['model_parameters_file']

        try:
            self.model_params = load_parameters(params_file)
            self.log_console.log(f"[INFO] Parameters loaded from {params_file}")
        except Exception as e:
            from model_parameters import get_parameters
            self.model_params = get_parameters()
            self.log_console.log(f"[WARNING] Using defaults: {e}", "WARNING")

        self.load_config()
        self._update_submit_button()

    # ── Window ────────────────────────────────
    def setup_window(self) -> None:
        self.root.title(self.TITLE)
        self.root.geometry(self.WINDOW_SIZE)
        self.root.minsize(*self.MIN_SIZE)
        self.root.configure(bg=P['bg'])

    # ── Variables ─────────────────────────────
    def setup_variables(self) -> None:
        self.entry_vars = {
            k: tk.StringVar() for k in [
                "input", "output", "model", "osm", "osm_road",
                "text_model", "ocr", "expt",
                "bounding_coordinate_1", "bounding_coordinate_2"
            ]
        }
        stage_keys = [s.key for s in STAGES]
        self.bool_vars = {
            k: tk.BooleanVar() for k in stage_keys + ["crop", "dim"]
        }
        for key in ["input", "output", "expt"]:
            self.entry_vars[key].trace_add(
                "write", lambda *args, k=key: self._validate_field(k)
            )

    # ── Widget tree ───────────────────────────
    def create_widgets(self) -> None:
        self._create_header()
        self._create_main_container()
        self._create_log_console()
        self._create_status_bar()

    def _create_header(self) -> None:
        """Dark banner with title + subtitle."""
        banner = tk.Frame(self.root, bg=P['header_bg'])
        banner.pack(fill=tk.X)

        inner = tk.Frame(banner, bg=P['header_bg'])
        inner.pack(padx=20, pady=14)

        tk.Label(inner, text=self.TITLE, bg=P['header_bg'], fg=P['header_fg'],
                 font=("Helvetica", 18, "bold")).pack(anchor="w")
        tk.Label(inner, text="Road extraction  •  Text detection  •  Geolocation  •  Similarity",
                 bg=P['header_bg'], fg="#8899BB",
                 font=("Helvetica", 10)).pack(anchor="w")

        tk.Frame(self.root, bg=P['accent'], height=3).pack(fill=tk.X)

    def _create_main_container(self) -> None:
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=(10, 4))
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        pane = ttk.Frame(main)
        pane.grid(row=0, column=0, sticky="nsew")
        self._create_scrollable_pane(pane)

    def _create_scrollable_pane(self, parent: tk.Widget) -> None:
        self.scroll_canvas = tk.Canvas(parent, highlightthickness=0, bg=P['bg'])
        scrollbar = ttk.Scrollbar(parent, orient="vertical",
                                   command=self.scroll_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.scroll_canvas)

        def _update_region(event=None):
            self.scroll_canvas.configure(
                scrollregion=self.scroll_canvas.bbox("all"))

        self.scrollable_frame.bind("<Configure>", _update_region)
        win = self.scroll_canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw")

        self.scroll_canvas.bind(
            "<Configure>",
            lambda e: self.scroll_canvas.itemconfig(win, width=e.width)
        )
        self.scroll_canvas.configure(yscrollcommand=scrollbar.set)
        self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        def _on_scroll(event):
            if not self.scroll_canvas.winfo_containing(event.x_root, event.y_root):
                return
            if hasattr(event, 'num'):
                self.scroll_canvas.yview_scroll(-1 if event.num == 4 else 1, "units")
            elif hasattr(event, 'delta'):
                self.scroll_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

        for widget in (self.scroll_canvas, self.scrollable_frame):
            widget.bind("<MouseWheel>", _on_scroll)
            widget.bind("<Button-4>",   _on_scroll)
            widget.bind("<Button-5>",   _on_scroll)

        self._create_main_content(self.scrollable_frame)
        self._create_buttons(self.scrollable_frame)
        self._create_progress_bar(self.scrollable_frame)
        ttk.Frame(self.scrollable_frame).pack(fill=tk.X, pady=(0, 16))
        self.update_scrollregion = _update_region

    def _create_main_content(self, parent: tk.Widget) -> None:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        frame.columnconfigure(0, weight=3)
        frame.columnconfigure(1, weight=1)

        self._create_directories_frame(frame)
        self._create_modules_frame(frame)

    # ── Directories panel ─────────────────────
    def _create_directories_frame(self, parent: tk.Widget) -> None:
        card = ttk.LabelFrame(parent, text="  Configuration")
        card.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=0)
        card.columnconfigure(1, weight=1)

        paths = [
            ("Input (File or Folder)",    "input",      "any",    True),
            ("Output Folder",             "output",     "folder", True),
            ("Road Detection Model",      "model",      "file",   False),
            ("OSM Intersection GeoJSON",  "osm",        "file",   False),
            ("OSM Road Graph (GraphML)",  "osm_road",   "file",   False),
            ("Text Detection Model",      "text_model", "file",   False),
            ("OCR Model Folder",          "ocr",        "folder", False),
        ]

        for i, (label, key, btype, required) in enumerate(paths):
            self._browse_row(card, label, i, self.entry_vars[key], btype, required)

        # Separator before experiment / coords
        sep_row = len(paths)
        ttk.Separator(card, orient=tk.HORIZONTAL).grid(
            row=sep_row, column=0, columnspan=4, sticky="ew", pady=6, padx=8)

        # Experiment name
        r = sep_row + 1
        ttk.Label(card, text="Experiment Name  *").grid(
            row=r, column=0, sticky="w", padx=10, pady=5)
        ttk.Entry(card, textvariable=self.entry_vars["expt"]).grid(
            row=r, column=1, columnspan=2, sticky="ew", padx=8, pady=5)

        # Bbox tool
        r += 1
        tool_row = tk.Frame(card, bg=P['surface'])
        tool_row.grid(row=r, column=0, columnspan=4, sticky="w", padx=10, pady=(2, 6))
        tk.Label(tool_row, text="Need bounding coordinates?",
                 bg=P['surface'], fg=P['text_muted'],
                 font=("Helvetica", 9)).pack(side=tk.LEFT)
        tk.Button(
            tool_row, text="  Launch Bounding Box Tool  ↗",
            command=self.launch_bbox_tool,
            bg=P['bg'], fg=P['accent'], relief=tk.FLAT,
            font=("Helvetica", 9, "bold"), cursor="hand2",
            activeforeground=P['accent_dark'], activebackground=P['border']
        ).pack(side=tk.LEFT, padx=6)

        # Coordinates
        for label, key in [
            ("Bounding Coord SW  (lon,lat)", "bounding_coordinate_1"),
            ("Bounding Coord NE  (lon,lat)", "bounding_coordinate_2"),
        ]:
            r += 1
            ttk.Label(card, text=label).grid(
                row=r, column=0, sticky="w", padx=10, pady=4)
            ttk.Entry(card, textvariable=self.entry_vars[key]).grid(
                row=r, column=1, columnspan=2, sticky="ew", padx=8, pady=4)

    def _browse_row(self, parent, label_text, row, entry_var, browse_type, required):
        display = label_text + ("  *" if required else "")
        ttk.Label(parent, text=display).grid(
            row=row, column=0, sticky="w", padx=10, pady=4)

        entry = ttk.Entry(parent, textvariable=entry_var)
        entry.grid(row=row, column=1, sticky="ew", padx=6, pady=4)

        tick = ttk.Label(parent, text="", width=2)
        tick.grid(row=row, column=2, padx=2, pady=4)

        def _check(*_):
            path = entry_var.get().strip()
            if not path:
                tick.config(text="", foreground=P['text_muted'])
            elif validate_path(path, browse_type):
                tick.config(text="✓", foreground=P['success'])
            else:
                tick.config(text="✗", foreground=P['error'])

        entry_var.trace_add("write", _check)
        _check()

        if browse_type == "any":
            btn_frame = ttk.Frame(parent)
            btn_frame.grid(row=row, column=3, padx=6, pady=4)
            ttk.Button(btn_frame, text="File",
                       command=lambda: self._browse_file(entry_var, tick, browse_type)
                       ).pack(side=tk.LEFT, padx=(0, 2))
            ttk.Button(btn_frame, text="Folder",
                       command=lambda: self._browse_folder(entry_var, tick, browse_type)
                       ).pack(side=tk.LEFT)
        else:
            cmd = (lambda: self._browse_file(entry_var, tick, browse_type)
                   if browse_type == "file"
                   else lambda: self._browse_folder(entry_var, tick, browse_type))
            ttk.Button(parent, text="Browse", command=cmd).grid(
                row=row, column=3, padx=6, pady=4)

    # ── Modules panel ─────────────────────────
    def _create_modules_frame(self, parent: tk.Widget) -> None:
        card = ttk.LabelFrame(parent, text="  Pipeline Stages")
        card.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=0)
        card.columnconfigure(0, weight=1)

        for i, stage in enumerate(STAGES):
            self._module_card(card, stage.key, stage.display, stage.description, i)

    def _module_card(self, parent, key, name, desc, row):
        """A small card per pipeline stage."""
        bg = P['surface']
        cell = tk.Frame(parent, bg=bg, bd=0, highlightthickness=1,
                        highlightbackground=P['border'])
        cell.grid(row=row, column=0, sticky="ew", padx=10, pady=4)
        cell.columnconfigure(1, weight=1)

        chk = tk.Checkbutton(
            cell, variable=self.bool_vars[key],
            bg=bg, activebackground=bg, selectcolor=bg,
            cursor="hand2"
        )
        chk.grid(row=0, column=0, rowspan=2, padx=(8, 4), pady=6)

        tk.Label(cell, text=name, bg=bg, fg=P['text'],
                 font=("Helvetica", 10, "bold")).grid(
            row=0, column=1, sticky="w", pady=(6, 0))
        tk.Label(cell, text=desc, bg=bg, fg=P['text_muted'],
                 font=("Helvetica", 8), wraplength=200, justify=tk.LEFT).grid(
            row=1, column=1, sticky="w", pady=(0, 6))

    # ── Dialogs ───────────────────────────────
    def _browse_folder(self, entry_var, tick, browse_type):
        folder = filedialog.askdirectory(title="Select Folder", parent=self.root)
        if folder:
            entry_var.set(folder)
            tick.config(
                text="✓" if validate_path(folder, browse_type) else "✗",
                foreground=P['success'] if validate_path(folder, browse_type) else P['error']
            )

    def _browse_file(self, entry_var, tick, browse_type):
        f = filedialog.askopenfilename(title="Select File", parent=self.root)
        if f:
            entry_var.set(f)
            tick.config(
                text="✓" if validate_path(f, browse_type) else "✗",
                foreground=P['success'] if validate_path(f, browse_type) else P['error']
            )

    def _show_tooltip(self, event, text):
        if hasattr(event.widget, '_tip'):
            try:
                event.widget._tip.destroy()
            except Exception:
                pass
        tip = tk.Toplevel()
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{event.x_root+12}+{event.y_root+12}")
        tk.Label(tip, text=text, background="#FFFDE7", foreground=P['text'],
                 relief="solid", borderwidth=1, font=("Helvetica", 9),
                 wraplength=220, padx=6, pady=4).pack()
        event.widget._tip = tip

    def _hide_tooltip(self, event):
        if hasattr(event.widget, '_tip'):
            try:
                event.widget._tip.destroy()
                del event.widget._tip
            except Exception:
                pass

    # ── Buttons ───────────────────────────────
    def _create_buttons(self, parent: tk.Widget) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=10, pady=(10, 6))

        self.submit_button = ttk.Button(
            row, text="▶  Run Pipeline", command=self.submit_pipeline,
            style="Primary.TButton"
        )
        self.submit_button.pack(side=tk.LEFT, padx=(0, 8))

        for text, cmd in [
            ("Reset",        self.reset_form),
            ("Save Config",  self.save_config),
            ("Load Config",  self.load_config),
            ("Model Params", self.show_model_params),
            ("Help",         self.show_help),
        ]:
            ttk.Button(row, text=text, command=cmd).pack(side=tk.LEFT, padx=3)

    # ── Progress ──────────────────────────────
    def _create_progress_bar(self, parent: tk.Widget) -> None:
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=10, pady=(0, 6))

        self.progress_label = ttk.Label(row, text="", style="Muted.TLabel")
        self.progress_label.pack(side=tk.LEFT, padx=(0, 8))

        self.progress = ttk.Progressbar(row, mode='indeterminate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.cancel_button = ttk.Button(
            row, text="Cancel", command=self.cancel_pipeline, state=tk.DISABLED
        )
        self.cancel_button.pack(side=tk.LEFT, padx=(8, 0))

    # ── Log + status bar ─────────────────────
    def _create_log_console(self) -> None:
        self.log_console = LogConsole(self.root)

    def _create_status_bar(self) -> None:
        bar = tk.Frame(self.root, bg=P['header_bg'], height=24)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        bar.pack_propagate(False)
        self.status_bar = tk.Label(
            bar, text="  Ready", bg=P['header_bg'], fg="#8899BB",
            font=("Helvetica", 9), anchor="w"
        )
        self.status_bar.pack(fill=tk.X, padx=8)

    # ── Events ────────────────────────────────
    def setup_event_handlers(self) -> None:
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.bind('<Control-s>',     lambda e: self.save_config())
        self.root.bind('<Control-o>',     lambda e: self.load_config())
        self.root.bind('<Control-r>',     lambda e: self.reset_form())
        self.root.bind('<Control-Return>', lambda e: self.submit_pipeline())
        self.root.bind('<F1>',            lambda e: self.show_help())

    # ── Validation ───────────────────────────
    def _validate_field(self, field_name: str) -> None:
        value = self.entry_vars[field_name].get().strip()
        if field_name == "input":
            ok = bool(value) and validate_path(value, "any")
        elif field_name == "output":
            ok = bool(value) and validate_path(value, "folder")
        else:
            ok = bool(value)
        self.validation_errors[field_name] = not ok
        self._update_submit_button()

    def _update_submit_button(self) -> None:
        all_ok = all(
            not self.validation_errors.get(f, False)
            for f in ["input", "output", "expt"]
        )
        self.submit_button.config(state=tk.NORMAL if all_ok else tk.DISABLED)

    # ── Queue processing ─────────────────────
    def process_queue(self) -> None:
        try:
            while True:
                msg = self.message_queue.get_nowait()
                t   = msg.get('type')
                if t == 'log':
                    self.log_console.log(msg['message'], msg.get('level', 'INFO'))
                elif t == 'status':
                    self.update_status(msg['message'])
                elif t == 'pipeline_complete':
                    self._on_pipeline_complete(msg.get('success', True), msg.get('error'))
                elif t == 'pipeline_start':
                    self.progress.start(12)
                    self.progress_label.config(text="Pipeline running…")
                    self.cancel_button.config(state=tk.NORMAL)
                    self.submit_button.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        self.root.after(50, self.process_queue)

    def _thread_safe_log(self, message: str, level: str = "INFO") -> None:
        self.message_queue.put({'type': 'log', 'message': message, 'level': level})

    def _thread_safe_status(self, message: str) -> None:
        self.message_queue.put({'type': 'status', 'message': message})

    def _run_pipeline_thread(self, config: Dict[str, Any]) -> None:
        try:
            self.pipeline_running = True
            callbacks = {
                'log':           self._thread_safe_log,
                'update_status': self._thread_safe_status,
            }
            Pipeline(config, callbacks, params=self.model_params).run()
            self.message_queue.put({'type': 'pipeline_complete', 'success': True})
        except Exception as e:
            self.message_queue.put({'type': 'pipeline_complete', 'success': False, 'error': str(e)})
        finally:
            self.pipeline_running = False

    def _on_pipeline_complete(self, success: bool, error: Optional[str] = None) -> None:
        self.progress.stop()
        self.progress_label.config(text="")
        self.cancel_button.config(state=tk.DISABLED)
        self.submit_button.config(state=tk.NORMAL)
        if success:
            self.log_console.log("Pipeline completed successfully!", "SUCCESS")
            self.update_status("Pipeline completed successfully")
        else:
            messagebox.showerror("Pipeline Error", f"Pipeline failed:\n{error}")
            self.log_console.log(f"Pipeline failed: {error}", "ERROR")

    # ── Pipeline ─────────────────────────────
    def submit_pipeline(self) -> None:
        if not self.validate_inputs():
            return
        if self.pipeline_running:
            messagebox.showwarning("Already Running", "A pipeline is already running.")
            return
        config = self.get_pipeline_config()
        self.message_queue.put({'type': 'pipeline_start'})
        self.pipeline_thread = threading.Thread(
            target=self._run_pipeline_thread, args=(config,), daemon=True
        )
        self.pipeline_thread.start()

    def launch_bbox_tool(self) -> None:
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bbox.py")
        if not os.path.exists(script_path):
            messagebox.showerror("Error", f"bbox.py not found:\n{script_path}")
            return
        streamlit_exe = os.path.join(os.path.dirname(sys.executable), "streamlit")
        cmd = f"'{streamlit_exe}' run '{script_path}'"
        try:
            plat = platform.system()
            if plat == "Darwin":
                subprocess.Popen(['osascript', '-e',
                                   f'tell application "Terminal" to do script "{cmd}" activate'])
            elif plat == "Windows":
                subprocess.Popen(f'start cmd /k "{cmd}"', shell=True)
            elif plat == "Linux":
                try:
                    subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', f'{cmd}; exec bash'])
                except FileNotFoundError:
                    subprocess.Popen(['xterm', '-e', f'{cmd}; bash'])
            self.log_console.log("Launched Bounding Box Tool in external terminal.")
        except Exception as e:
            messagebox.showerror("Launch Error", f"Failed to launch tool:\n{e}")

    def cancel_pipeline(self) -> None:
        if self.pipeline_running:
            self.log_console.log("Cancellation requested (not yet implemented)", "WARNING")

    def validate_inputs(self) -> bool:
        errors = []
        for field in ["input", "output", "expt"]:
            value = self.entry_vars[field].get().strip()
            if not value:
                errors.append(f"'{field}' is required.")
            elif field == "input" and not validate_path(value, "any"):
                errors.append("Input path does not exist or is not a supported image file/folder.")
            elif field == "output" and not validate_path(value, "folder"):
                errors.append(f"Output path does not exist or is not a folder.")
        if errors:
            messagebox.showerror("Validation Error", "\n".join(errors))
            return False
        return True

    def get_pipeline_config(self) -> Dict[str, Any]:
        return {
            'entries': {k: v.get().strip() for k, v in self.entry_vars.items()},
            'modules': {k: v.get()         for k, v in self.bool_vars.items()},
        }

    def reset_form(self) -> None:
        for v in self.entry_vars.values():
            v.set("")
        for v in self.bool_vars.values():
            v.set(False)
        self.validation_errors.clear()
        self._update_submit_button()
        self.log_console.log("Form reset.")

    def save_config(self) -> None:
        if self.config_manager.save(self.get_pipeline_config()):
            self.log_console.log("Configuration saved.")
            self.update_status("Configuration saved")
        else:
            self.log_console.log("Failed to save configuration.", "ERROR")

    def load_config(self) -> None:
        config = self.config_manager.load()
        if not config:
            self.log_console.log("No saved configuration found.")
            return
        for k, v in config.get("entries", {}).items():
            if k in self.entry_vars:
                self.entry_vars[k].set(v)
        for k, v in config.get("modules", {}).items():
            if k in self.bool_vars:
                self.bool_vars[k].set(v)
        self._update_submit_button()
        self.log_console.log("Configuration loaded.")

    def update_status(self, message: str) -> None:
        self.status_bar.config(text=f"  {message}")
        self.progress_label.config(text=message)
        self.root.update_idletasks()

    # ── Model params dialog ───────────────────
    def show_model_params(self) -> None:
        def _on_save(new_params):
            self.model_params = new_params
        open_model_params_dialog(self.root, self.model_params, _on_save)

    # ── Help dialog ───────────────────────────
    def show_help(self) -> None:
        messagebox.showinfo("Help", (
            "Historic Map Pipeline\n\n"
            "Keyboard shortcuts:\n"
            "  Ctrl+Enter  Run pipeline\n"
            "  Ctrl+S      Save config\n"
            "  Ctrl+O      Load config\n"
            "  Ctrl+R      Reset form\n"
            "  F1          This help\n\n"
            "Required fields:\n"
            "  Input Image File  *\n"
            "  Output Folder     *\n"
            "  Experiment Name   *\n\n"
            "Pipeline stages can be toggled independently.\n"
            "Model hyperparameters are editable via 'Model Params'."
        ))

    def on_closing(self) -> None:
        self.save_config()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    PipelineGUI().run()
