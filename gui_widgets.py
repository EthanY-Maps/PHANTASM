"""Reusable widgets for the pipeline GUI."""

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from datetime import datetime
from typing import Dict, List, Any

from gui_styles import PALETTE as P


class LogConsole:
    """Filterable, exportable log panel."""

    def __init__(self, parent: tk.Widget) -> None:
        outer = tk.Frame(parent, bg=P['bg'])
        outer.pack(fill=tk.BOTH, expand=False, padx=16, pady=(0, 12))

        bar = tk.Frame(outer, bg=P['header_bg'], height=32)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)
        tk.Label(bar, text="  Log Console", bg=P['header_bg'], fg=P['header_fg'],
                 font=("Helvetica", 10, "bold")).pack(side=tk.LEFT)

        toolbar = tk.Frame(outer, bg=P['surface'], pady=4)
        toolbar.pack(fill=tk.X)

        tk.Label(toolbar, text="  Filter:", bg=P['surface'],
                 fg=P['text_muted'], font=("Helvetica", 9)).pack(side=tk.LEFT)

        self.filter_var = tk.StringVar(value="ALL")
        for level in ["ALL", "INFO", "WARNING", "ERROR"]:
            tk.Radiobutton(
                toolbar, text=level, variable=self.filter_var, value=level,
                command=self._apply_filter, bg=P['surface'], fg=P['text'],
                selectcolor=P['surface'], activebackground=P['surface'],
                font=("Helvetica", 9)
            ).pack(side=tk.LEFT, padx=3)

        tk.Frame(toolbar, bg=P['border'], width=1).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        for text, cmd in [("Clear", self.clear), ("Export", self.export_log)]:
            tk.Button(
                toolbar, text=text, command=cmd,
                bg=P['bg'], fg=P['text'], relief=tk.FLAT,
                font=("Helvetica", 9), padx=8, pady=2,
                cursor="hand2", activebackground=P['border']
            ).pack(side=tk.LEFT, padx=2)

        self.auto_scroll_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            toolbar, text="Auto-scroll", variable=self.auto_scroll_var,
            bg=P['surface'], fg=P['text_muted'], selectcolor=P['surface'],
            activebackground=P['surface'], font=("Helvetica", 9)
        ).pack(side=tk.LEFT, padx=8)

        self.text_widget = ScrolledText(
            outer, height=7, state='disabled', wrap=tk.WORD,
            font=("Menlo", 9), background=P['log_bg'], foreground=P['log_fg'],
            insertbackground=P['log_fg'], relief=tk.FLAT, padx=8, pady=6, bd=0,
        )
        self.text_widget.pack(fill=tk.BOTH, expand=False)

        self.text_widget.tag_config("INFO",    foreground=P['tag_info'])
        self.text_widget.tag_config("WARNING", foreground=P['tag_warn'])
        self.text_widget.tag_config("ERROR",   foreground=P['tag_err'])
        self.text_widget.tag_config("SUCCESS", foreground=P['tag_ok'])

        self.log_entries: List[Dict[str, Any]] = []
        self.current_filter = "ALL"

    def log(self, message: str, level: str = "INFO") -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = {
            "timestamp": timestamp, "level": level, "message": message,
            "full_text": f"[{timestamp}] {message}",
        }
        self.log_entries.append(entry)
        if self._should_show(level):
            self._append_log(entry)

    def _should_show(self, level: str) -> bool:
        return self.current_filter == "ALL" or level == self.current_filter

    def _append_log(self, entry: Dict[str, Any]) -> None:
        self.text_widget.config(state='normal')
        self.text_widget.insert(tk.END, entry["full_text"] + "\n", entry["level"])
        if self.auto_scroll_var.get():
            self.text_widget.yview_moveto(1.0)
        self.text_widget.config(state='disabled')
        try:
            self.text_widget.update_idletasks()
        except Exception:
            pass

    def _apply_filter(self) -> None:
        self.current_filter = self.filter_var.get()
        self.text_widget.config(state='normal')
        self.text_widget.delete(1.0, tk.END)
        for entry in self.log_entries:
            if self._should_show(entry["level"]):
                self._append_log(entry)
        self.text_widget.config(state='disabled')

    def clear(self) -> None:
        self.log_entries.clear()
        self.text_widget.config(state='normal')
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.config(state='disabled')

    def export_log(self) -> None:
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not filename:
            return
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for entry in self.log_entries:
                    f.write(entry["full_text"] + "\n")
            self.log(f"Log exported to {filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export log: {e}")
