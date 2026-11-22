"""Modal dialog for editing pipeline model parameters as JSON."""

import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from typing import Callable

from gui_styles import PALETTE as P
from model_parameters import ModelParameters, DEFAULT_PARAMS_FILE, load_parameters, save_parameters


def open_model_params_dialog(root: tk.Tk, current_params: ModelParameters,
                             on_save: Callable[[ModelParameters], None]) -> None:
    """`on_save(new_params)` is invoked after a successful Save so the parent GUI updates."""
    win = tk.Toplevel(root)
    win.title("Model Parameters")
    win.geometry("820x620")
    win.configure(bg=P['bg'])

    hdr = tk.Frame(win, bg=P['header_bg'])
    hdr.pack(fill=tk.X)
    tk.Label(hdr, text="  Model Parameters", bg=P['header_bg'], fg=P['header_fg'],
             font=("Helvetica", 12, "bold"), pady=10).pack(anchor="w")
    tk.Frame(win, bg=P['accent'], height=2).pack(fill=tk.X)

    ttk.Label(win, text="Edit parameters as JSON. Changes take effect on next pipeline run.",
              style="Muted.TLabel").pack(anchor="w", padx=12, pady=(8, 4))

    text_frame = ttk.Frame(win)
    text_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)

    text_widget = ScrolledText(
        text_frame, wrap=tk.NONE, font=("Menlo", 9),
        background=P['log_bg'], foreground=P['log_fg'],
        insertbackground=P['log_fg'], relief=tk.FLAT, padx=8, pady=6,
    )
    text_widget.pack(fill=tk.BOTH, expand=True)
    text_widget.insert('1.0', json.dumps(current_params.to_dict(), indent=2))

    def save_params():
        try:
            new_params = ModelParameters.from_dict(
                json.loads(text_widget.get('1.0', tk.END).strip())
            )
            save_parameters(DEFAULT_PARAMS_FILE, new_params)
            on_save(new_params)
            messagebox.showinfo("Saved", "Parameters saved successfully.")
            win.destroy()
        except json.JSONDecodeError as e:
            messagebox.showerror("JSON Error", f"Invalid JSON:\n{e}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load_from_file():
        fname = filedialog.askopenfilename(
            title="Load Parameters",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not fname:
            return
        try:
            loaded = load_parameters(fname)
            text_widget.delete('1.0', tk.END)
            text_widget.insert('1.0', json.dumps(loaded.to_dict(), indent=2))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def export_to_file():
        fname = filedialog.asksaveasfilename(
            title="Export Parameters", defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        if not fname:
            return
        try:
            save_parameters(fname, ModelParameters.from_dict(
                json.loads(text_widget.get('1.0', tk.END).strip())
            ))
            messagebox.showinfo("Exported", f"Saved to {fname}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    btn_row = ttk.Frame(win)
    btn_row.pack(fill=tk.X, padx=12, pady=8)
    ttk.Button(btn_row, text="Save",      command=save_params,    style="Primary.TButton").pack(side=tk.LEFT, padx=(0, 6))
    ttk.Button(btn_row, text="Load File", command=load_from_file).pack(side=tk.LEFT, padx=3)
    ttk.Button(btn_row, text="Export",    command=export_to_file).pack(side=tk.LEFT, padx=3)
    ttk.Button(btn_row, text="Close",     command=win.destroy).pack(side=tk.RIGHT)
