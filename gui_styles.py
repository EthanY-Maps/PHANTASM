"""Colour palette and ttk style configuration for the pipeline GUI."""

import tkinter as tk
from tkinter import ttk


PALETTE = {
    'bg':           '#F0F4F8',
    'surface':      '#FFFFFF',
    'header_bg':    '#1A233A',
    'header_fg':    '#FFFFFF',
    'accent':       '#4F6AF5',
    'accent_dark':  '#3B54D4',
    'accent_fg':    '#FFFFFF',
    'border':       '#D6DCE8',
    'text':         '#1E293B',
    'text_muted':   '#64748B',
    'success':      '#16A34A',
    'warning':      '#D97706',
    'error':        '#DC2626',
    'tag_info':     '#60AEFF',
    'tag_warn':     '#FFD166',
    'tag_err':      '#FF6B6B',
    'tag_ok':       '#6BFFB8',
    'log_bg':       '#0F172A',
    'log_fg':       '#CBD5E1',
}


def apply_styles() -> None:
    P = PALETTE
    style = ttk.Style()
    style.theme_use("clam")

    style.configure("TFrame", background=P['bg'])
    style.configure("TLabel", background=P['bg'], foreground=P['text'],
                    font=("Helvetica", 10))
    style.configure("Muted.TLabel", background=P['bg'], foreground=P['text_muted'],
                    font=("Helvetica", 9))
    style.configure("TLabelframe", background=P['surface'], foreground=P['text'],
                    font=("Helvetica", 10, "bold"),
                    bordercolor=P['border'], relief=tk.GROOVE, borderwidth=1)
    style.configure("TLabelframe.Label", background=P['surface'], foreground=P['text'],
                    font=("Helvetica", 10, "bold"))

    style.configure("TEntry", fieldbackground=P['surface'], foreground=P['text'],
                    bordercolor=P['border'], relief=tk.FLAT, padding=5)
    style.map("TEntry",
              bordercolor=[("focus", P['accent']), ("!focus", P['border'])])

    style.configure("TButton", background=P['bg'], foreground=P['text'],
                    font=("Helvetica", 9), borderwidth=1,
                    relief=tk.FLAT, padding=(10, 5))
    style.map("TButton",
              background=[("active", P['border']), ("!active", P['bg'])],
              relief=[("active", tk.FLAT)])

    style.configure("Primary.TButton", background=P['accent'], foreground=P['accent_fg'],
                    font=("Helvetica", 10, "bold"), relief=tk.FLAT, padding=(14, 7))
    style.map("Primary.TButton",
              background=[("disabled", P['border']),
                          ("active",   P['accent_dark']),
                          ("!active",  P['accent'])],
              foreground=[("disabled", P['text_muted'])])

    style.configure("TCheckbutton", background=P['surface'], foreground=P['text'],
                    font=("Helvetica", 10))
    style.map("TCheckbutton",
              background=[("active", P['surface'])],
              foreground=[("active", P['accent'])])

    style.configure("TRadiobutton", background=P['bg'], foreground=P['text'],
                    font=("Helvetica", 9))

    style.configure("TProgressbar", troughcolor=P['border'],
                    background=P['accent'], thickness=6)

    style.configure("TSeparator", background=P['border'])

    style.configure("TScrollbar", background=P['border'],
                    troughcolor=P['bg'], borderwidth=0, arrowsize=12)
