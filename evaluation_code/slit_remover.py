#!/usr/bin/env python3
"""
Map slit remover — polygon edition.

Draw a polygon over each fold/slit in the map.
Set each polygon's closing direction independently:
  • Vertical fold (crease runs top→bottom)  → close Horizontally (push left/right)
  • Horizontal fold (crease runs left→right) → close Vertically   (push up/down)

All horizontal-closing polygons are applied first, then vertical-closing.

Controls:
    Left-click           — add vertex
    Double-click / Enter — close & finalise polygon
    Escape               — cancel current polygon
    Scroll               — zoom centred on cursor
    Right-click drag     — pan
    +  /  -  /  f        — zoom in / out / fit
"""

import sys
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageTk

Image.MAX_IMAGE_PIXELS = None


# ── stitching ─────────────────────────────────────────────────────────────────

def _build_mask(w: int, h: int, pts: list[tuple]) -> np.ndarray:
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).polygon(pts, fill=255)
    return np.array(m) > 0


def _remove_polygon(arr: np.ndarray, pts: list[tuple], direction: str) -> np.ndarray:
    h, w = arr.shape[:2]
    mask = _build_mask(w, h, pts)

    if direction == "horizontal":
        new_rows = []; max_rm = 0
        for y in range(h):
            filled = np.where(mask[y])[0]
            if len(filled) == 0:
                new_rows.append(arr[y])
            else:
                xl, xr = int(filled[0]), int(filled[-1]) + 1
                max_rm = max(max_rm, xr - xl)
                new_rows.append(np.concatenate([arr[y, :xl], arr[y, xr:]]))
        if max_rm == 0:
            return arr.copy()
        tw = max(1, w - max_rm)
        res = np.full((h, tw, 3), 255, dtype=arr.dtype)
        for y, row in enumerate(new_rows):
            cw = min(len(row), tw)
            res[y, :cw] = row[:cw]
        return res

    else:  # vertical
        new_cols = []; max_rm = 0
        for x in range(w):
            filled = np.where(mask[:, x])[0]
            if len(filled) == 0:
                new_cols.append(arr[:, x])
            else:
                yt, yb = int(filled[0]), int(filled[-1]) + 1
                max_rm = max(max_rm, yb - yt)
                new_cols.append(np.concatenate([arr[:yt, x], arr[yb:, x]]))
        if max_rm == 0:
            return arr.copy()
        th = max(1, h - max_rm)
        res = np.full((th, w, 3), 255, dtype=arr.dtype)
        for x, col in enumerate(new_cols):
            ch = min(len(col), th)
            res[:ch, x] = col[:ch]
        return res


def _stitch(image: Image.Image, polygons: list[dict]) -> Image.Image:
    """
    polygons: list of {"pts": [...], "direction": "horizontal"|"vertical"}
    Apply horizontal-closing first, then vertical-closing.
    """
    arr = np.array(image.convert("RGB"))
    for p in sorted(polygons, key=lambda x: 0 if x["direction"] == "horizontal" else 1):
        if len(p["pts"]) >= 3:
            arr = _remove_polygon(arr, p["pts"], p["direction"])
    return Image.fromarray(arr)


def _mask_preview(image: Image.Image, polygons: list[dict]) -> Image.Image:
    base = image.convert("RGBA").copy()
    ov   = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov)
    for p in polygons:
        if len(p["pts"]) >= 3:
            col = (255, 80, 80, 150) if p["direction"] == "horizontal" else (80, 80, 255, 150)
            draw.polygon(p["pts"], fill=col)
    base.alpha_composite(ov)
    return base.convert("RGB")


# ── App ───────────────────────────────────────────────────────────────────────

DIRECTIONS = ("horizontal", "vertical")
DIR_LABELS = {
    "horizontal": "Horizontal close  (vertical fold / crease top→bottom)",
    "vertical":   "Vertical close    (horizontal fold / crease left→right)",
}
DIR_COLOUR = {"horizontal": "#ff9900", "vertical": "#4488ff"}


class SlitRemoverApp:

    ZOOM_MIN  = 0.01
    ZOOM_MAX  = 20.0
    ZOOM_STEP = 1.5

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Slit Remover")
        self.root.geometry("1340x900")

        self.image:    Image.Image | None = None
        self.filepath: str                = ""
        self._photo                       = None

        self.zoom = 1.0

        # Each polygon: {"pts": [(x,y),...], "direction": "horizontal"|"vertical"}
        self.polygons: list[dict] = []
        self.current:  list[tuple] = []

        self._zoom_pct    = tk.StringVar(value="—")
        self._cursor_pos  = tk.StringVar(value="x: —  y: —")
        self._status      = tk.StringVar(value="No image loaded.")
        self._hint        = tk.StringVar(
            value="Click to place vertices  •  Double-click or Enter to close  •  Esc to cancel")
        self._new_dir     = tk.StringVar(value="horizontal")  # direction for next polygon

        self._build_ui()
        self._bind_keys()

        if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
            self._load_image(sys.argv[1])

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(0, weight=1)

        cf = ttk.Frame(self.root)
        cf.grid(row=0, column=0, sticky="nsew")
        cf.columnconfigure(0, weight=1)
        cf.rowconfigure(0, weight=0); cf.rowconfigure(1, weight=1); cf.rowconfigure(2, weight=0)

        # ── toolbar ───────────────────────────────────────────────────────
        tb = ttk.Frame(cf, padding=(4, 3))
        tb.grid(row=0, column=0, columnspan=2, sticky="ew")

        ttk.Button(tb, text="Open…", command=self._open_dialog, width=7
                   ).pack(side="left", padx=(0, 4))
        ttk.Separator(tb, orient="vertical").pack(side="left", fill="y", padx=4)
        ttk.Button(tb, text="−", width=3, command=self._zoom_out).pack(side="left", padx=(4,1))
        ttk.Label(tb, textvariable=self._zoom_pct, width=6,
                  anchor="center", relief="sunken", padding=(2,1)
                  ).pack(side="left", padx=1)
        ttk.Button(tb, text="+", width=3, command=self._zoom_in).pack(side="left", padx=(1,2))
        ttk.Button(tb, text="Fit", width=4, command=self._fit_to_window
                   ).pack(side="left", padx=(2,8))
        ttk.Separator(tb, orient="vertical").pack(side="left", fill="y", padx=4)
        ttk.Label(tb, textvariable=self._cursor_pos, foreground="#555", width=24
                  ).pack(side="left", padx=6)
        ttk.Label(tb, textvariable=self._hint, foreground="#888"
                  ).pack(side="right", padx=8)

        # ── canvas ────────────────────────────────────────────────────────
        self.canvas = tk.Canvas(cf, bg="#2b2b2b", highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=1, column=0, sticky="nsew")
        hbar = ttk.Scrollbar(cf, orient="horizontal", command=self.canvas.xview)
        vbar = ttk.Scrollbar(cf, orient="vertical",   command=self.canvas.yview)
        hbar.grid(row=2, column=0, sticky="ew"); vbar.grid(row=1, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

        self.canvas.bind("<ButtonPress-1>",   self._on_click)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)
        self.canvas.bind("<ButtonPress-3>",   self._pan_start)
        self.canvas.bind("<B3-Motion>",       self._pan_move)
        self.canvas.bind("<ButtonPress-2>",   self._pan_start)
        self.canvas.bind("<B2-Motion>",       self._pan_move)
        self.canvas.bind("<MouseWheel>",      self._on_scroll)
        self.canvas.bind("<Button-4>",        self._on_scroll)
        self.canvas.bind("<Button-5>",        self._on_scroll)
        self.canvas.bind("<Motion>",          self._on_motion)
        self.canvas.bind("<Leave>",           lambda _: self._cursor_pos.set("x: —  y: —"))

        # ── side panel ────────────────────────────────────────────────────
        ctrl = ttk.Frame(self.root, padding=10, width=280)
        ctrl.grid(row=0, column=1, sticky="nsew")
        ctrl.columnconfigure(0, weight=1)
        r = 0

        ttk.Label(ctrl, text="Next polygon closing direction:",
                  font=("", 10, "bold")).grid(row=r, column=0, sticky="w"); r += 1
        ttk.Label(ctrl,
                  text="Set BEFORE drawing each polygon.\n"
                       "Orange = horizontal close  (vertical fold)\n"
                       "Blue   = vertical close    (horizontal fold)",
                  foreground="#555", wraplength=255, justify="left"
                  ).grid(row=r, column=0, sticky="w", pady=(0, 4)); r += 1

        for val in DIRECTIONS:
            ttk.Radiobutton(ctrl, text=DIR_LABELS[val],
                            variable=self._new_dir, value=val
                            ).grid(row=r, column=0, sticky="w"); r += 1

        ttk.Separator(ctrl).grid(row=r, column=0, sticky="ew", pady=8); r += 1

        # Polygon list
        ttk.Label(ctrl, text="Polygons", font=("", 10, "bold")
                  ).grid(row=r, column=0, sticky="w"); r += 1

        # Direction change for selected polygons
        drow = ttk.Frame(ctrl); drow.grid(row=r, column=0, sticky="ew"); r += 1
        ttk.Label(drow, text="Change selected → ").pack(side="left")
        ttk.Button(drow, text="Horiz", width=6,
                   command=lambda: self._change_dir_selected("horizontal")
                   ).pack(side="left", padx=(0, 2))
        ttk.Button(drow, text="Vert", width=6,
                   command=lambda: self._change_dir_selected("vertical")
                   ).pack(side="left")

        lf = ttk.Frame(ctrl)
        lf.grid(row=r, column=0, sticky="nsew"); r += 1
        ctrl.rowconfigure(r - 1, weight=1)
        lf.columnconfigure(0, weight=1); lf.rowconfigure(0, weight=1)

        self._lb = tk.Listbox(lf, selectmode="extended", height=14, font=("Courier", 9))
        self._lb.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(lf, command=self._lb.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self._lb.configure(yscrollcommand=sb.set)
        self._lb.bind("<<ListboxSelect>>", lambda _: self._draw_all())

        brow = ttk.Frame(ctrl); brow.grid(row=r, column=0, sticky="ew"); r += 1
        ttk.Button(brow, text="Delete Selected",
                   command=self._delete_selected
                   ).pack(side="left", expand=True, fill="x", padx=(0,2))
        ttk.Button(brow, text="Clear All",
                   command=self._clear_all).pack(side="left", expand=True, fill="x")

        ttk.Separator(ctrl).grid(row=r, column=0, sticky="ew", pady=8); r += 1

        ttk.Button(ctrl, text="Show Mask Preview",
                   command=self._show_mask).grid(row=r, column=0, sticky="ew"); r += 1
        ttk.Button(ctrl, text="Preview Stitched Result",
                   command=self._preview).grid(row=r, column=0, sticky="ew"); r += 1
        ttk.Button(ctrl, text="Save Result As…",
                   command=self._save).grid(row=r, column=0, sticky="ew"); r += 1
        ttk.Separator(ctrl).grid(row=r, column=0, sticky="ew", pady=8); r += 1
        ttk.Label(ctrl, textvariable=self._status, wraplength=255,
                  justify="left", foreground="#555"
                  ).grid(row=r, column=0, sticky="w")

    def _bind_keys(self):
        for k in ("Return", "KP_Enter"):
            self.root.bind(f"<{k}>", lambda _: self._close_polygon())
        self.root.bind("<Escape>", lambda _: self._cancel_polygon())
        for k in ("equal", "plus"):
            self.root.bind(f"<{k}>", lambda _: self._zoom_in())
        self.root.bind("<minus>", lambda _: self._zoom_out())
        self.root.bind("f",       lambda _: self._fit_to_window())

    # ── image loading ─────────────────────────────────────────────────────────

    def _open_dialog(self):
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[("Images", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
                       ("All files", "*.*")])
        if path:
            self._load_image(path)

    def _load_image(self, path: str):
        self._status.set("Loading…"); self.root.update_idletasks()
        try:
            img = Image.open(path); img.load()
            self.image = img; self.filepath = path
            self.polygons = []; self.current = []
            self._refresh_list()
            self.root.after(50, self._fit_to_window)
            self._status.set(f"{os.path.basename(path)}\n{img.width} × {img.height} px")
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    # ── rendering ─────────────────────────────────────────────────────────────

    def _fit_to_window(self):
        if self.image is None: return
        self.root.update_idletasks()
        cw = self.canvas.winfo_width()  or 900
        ch = self.canvas.winfo_height() or 700
        self.zoom = min(cw / self.image.width, ch / self.image.height, 1.0)
        self._redraw()

    def _redraw(self):
        if self.image is None: return
        w = max(1, int(self.image.width  * self.zoom))
        h = max(1, int(self.image.height * self.zoom))
        thumb = self.image.copy(); thumb.thumbnail((w, h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(thumb)
        self.canvas.delete("all")
        self.canvas.configure(scrollregion=(0, 0, w, h))
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self._draw_all()
        self._zoom_pct.set(f"{self.zoom * 100:.0f}%")

    def _i2c(self, ix, iy):
        return ix * self.zoom, iy * self.zoom

    def _draw_all(self):
        if self.image is None: return
        self.canvas.delete("poly")
        selected = set(self._lb.curselection())

        for i, poly in enumerate(self.polygons):
            pts = poly["pts"]
            if len(pts) < 2: continue
            base_col = DIR_COLOUR[poly["direction"]]
            col = "#ff4444" if i in selected else base_col
            cpts = [c for p in pts for c in self._i2c(*p)]
            self.canvas.create_polygon(*cpts,
                fill=col, stipple="gray25", outline=col, width=2, tags="poly")
            for ix, iy in pts:
                cx, cy = self._i2c(ix, iy)
                self.canvas.create_oval(cx-4, cy-4, cx+4, cy+4,
                    fill=col, outline="white", width=1, tags="poly")
            mx = sum(p[0] for p in pts) / len(pts) * self.zoom
            my = sum(p[1] for p in pts) / len(pts) * self.zoom
            self.canvas.create_text(mx, my, text=str(i + 1),
                fill="white", font=("Arial", 8, "bold"), tags="poly")

        # in-progress
        if self.current:
            col = DIR_COLOUR[self._new_dir.get()]
            cpts = [c for p in self.current for c in self._i2c(*p)]
            if len(cpts) >= 4:
                self.canvas.create_line(*cpts, fill=col, width=2, dash=(6,3), tags="poly")
            for ix, iy in self.current:
                cx, cy = self._i2c(ix, iy)
                self.canvas.create_oval(cx-4, cy-4, cx+4, cy+4,
                    fill=col, outline="white", width=1, tags="poly")

    # ── pan / zoom ────────────────────────────────────────────────────────────

    def _pan_start(self, e): self.canvas.scan_mark(e.x, e.y)
    def _pan_move(self,  e): self.canvas.scan_dragto(e.x, e.y, gain=1)

    def _on_scroll(self, e):
        f = self.ZOOM_STEP if (e.num == 4 or e.delta > 0) else 1/self.ZOOM_STEP
        self._zoom_at(f, e.x, e.y)

    def _zoom_at(self, factor, cx, cy):
        if self.image is None: return
        old = self.zoom
        new = max(self.ZOOM_MIN, min(self.ZOOM_MAX, old * factor))
        if new == old: return
        ix = self.canvas.canvasx(cx) / old; iy = self.canvas.canvasy(cy) / old
        self.zoom = new; self._redraw()
        tw = self.image.width * new; th = self.image.height * new
        self.canvas.xview_moveto(max(0, (ix*new - cx) / tw))
        self.canvas.yview_moveto(max(0, (iy*new - cy) / th))

    def _zoom_in(self):
        self._zoom_at(self.ZOOM_STEP, self.canvas.winfo_width()//2, self.canvas.winfo_height()//2)
    def _zoom_out(self):
        self._zoom_at(1/self.ZOOM_STEP, self.canvas.winfo_width()//2, self.canvas.winfo_height()//2)

    # ── motion / rubber-band ──────────────────────────────────────────────────

    def _on_motion(self, e):
        if self.image is None: return
        ix = max(0, min(self.image.width  - 1, int(self.canvas.canvasx(e.x) / self.zoom)))
        iy = max(0, min(self.image.height - 1, int(self.canvas.canvasy(e.y) / self.zoom)))
        self._cursor_pos.set(f"x: {ix:5d}  y: {iy:5d}")
        self.canvas.delete("rubber")
        if self.current:
            col = DIR_COLOUR[self._new_dir.get()]
            lx, ly = self._i2c(*self.current[-1])
            cx = self.canvas.canvasx(e.x); cy = self.canvas.canvasy(e.y)
            self.canvas.create_line(lx, ly, cx, cy,
                fill=col, width=1, dash=(5,4), tags="rubber")
            if len(self.current) >= 3:
                fx, fy = self._i2c(*self.current[0])
                self.canvas.create_line(cx, cy, fx, fy,
                    fill=col, width=1, dash=(2,6), tags="rubber")

    # ── drawing ───────────────────────────────────────────────────────────────

    def _c2i(self, ex, ey):
        return (max(0, min(self.image.width  - 1, int(self.canvas.canvasx(ex) / self.zoom))),
                max(0, min(self.image.height - 1, int(self.canvas.canvasy(ey) / self.zoom))))

    def _on_click(self, e):
        if self.image is None: return
        self.current.append(self._c2i(e.x, e.y))
        self._draw_all()
        if len(self.current) == 1:
            self._hint.set("Keep clicking  •  Double-click or Enter to close  •  Esc to cancel")

    def _on_double_click(self, e):
        if self.image is None: return
        if len(self.current) > 1:
            self.current.pop()
        self._close_polygon()

    def _close_polygon(self):
        if len(self.current) < 3:
            if self.current: messagebox.showinfo("Too few points", "Need at least 3 vertices.")
            return
        self.polygons.append({"pts": list(self.current), "direction": self._new_dir.get()})
        self.current = []
        self._refresh_list(); self._draw_all()
        self._hint.set("Click to place vertices  •  Double-click or Enter to close  •  Esc to cancel")

    def _cancel_polygon(self):
        self.current = []
        self.canvas.delete("rubber"); self._draw_all()
        self._hint.set("Click to place vertices  •  Double-click or Enter to close  •  Esc to cancel")

    # ── list management ───────────────────────────────────────────────────────

    def _refresh_list(self):
        self._lb.delete(0, "end")
        for i, p in enumerate(self.polygons):
            pts = p["pts"]
            xs = [q[0] for q in pts]; ys = [q[1] for q in pts]
            span_w = max(xs) - min(xs); span_h = max(ys) - min(ys)
            d = "H" if p["direction"] == "horizontal" else "V"
            self._lb.insert("end",
                f"  {i+1:3d} [{d}]  {len(pts)}-pt  {span_w}×{span_h}px")

    def _delete_selected(self):
        for i in sorted(self._lb.curselection(), reverse=True):
            self.polygons.pop(i)
        self._refresh_list(); self._draw_all()

    def _clear_all(self):
        self.polygons = []; self.current = []
        self._refresh_list(); self.canvas.delete("poly"); self.canvas.delete("rubber")

    def _change_dir_selected(self, direction: str):
        for i in self._lb.curselection():
            self.polygons[i]["direction"] = direction
        self._refresh_list(); self._draw_all()

    # ── mask preview ──────────────────────────────────────────────────────────

    def _show_mask(self):
        if self.image is None: return
        if not self.polygons:
            messagebox.showinfo("No polygons", "Draw at least one polygon first."); return
        self._status.set("Building mask…"); self.root.update_idletasks()
        preview = _mask_preview(self.image, self.polygons)
        self._preview_window(preview,
            "Mask Preview — orange = horizontal close, blue = vertical close")
        self._status.set("Mask preview shown.")

    def _preview_window(self, img: Image.Image, title: str):
        win = tk.Toplevel(self.root); win.title(title); win.geometry("980x740")
        scale = min(940/img.width, 680/img.height, 1.0)
        thumb = img.copy()
        thumb.thumbnail((max(1,int(img.width*scale)), max(1,int(img.height*scale))), Image.LANCZOS)
        photo = ImageTk.PhotoImage(thumb)
        lbl = ttk.Label(win, image=photo); lbl.image = photo; lbl.pack(padx=10, pady=10)
        ttk.Label(win, text=f"{img.width}×{img.height} px  (shown at {scale*100:.0f}%)"
                  ).pack(pady=(0,8))

    # ── stitch preview & save ─────────────────────────────────────────────────

    def _preview(self):
        if self.image is None: return
        if not self.polygons:
            messagebox.showinfo("No polygons", "Draw at least one polygon first."); return
        if not self._safety_check(): return
        self._status.set("Stitching…"); self.root.update_idletasks()
        try:
            result = _stitch(self.image, self.polygons)
        except Exception as e:
            messagebox.showerror("Error", str(e)); return
        self._preview_window(result, "Preview — Stitched Result")
        self._status.set(f"Preview: {result.width}×{result.height} px")

    def _safety_check(self) -> bool:
        """Warn if any polygon removes an unreasonably large fraction of the image."""
        w, h = self.image.width, self.image.height
        for i, p in enumerate(self.polygons):
            mask = _build_mask(w, h, p["pts"])
            if p["direction"] == "horizontal":
                spans = [int(r[-1]) - int(r[0]) + 1
                         for r in (np.where(mask[y])[0] for y in range(h)) if len(r)]
                limit = w
            else:
                spans = [int(c[-1]) - int(c[0]) + 1
                         for c in (np.where(mask[:, x])[0] for x in range(w)) if len(c)]
                limit = h
            if not spans: continue
            pct = max(spans) / limit * 100
            if pct > 60:
                ans = messagebox.askyesno("Large removal warning",
                    f"Polygon {i+1} ({p['direction']} close) would remove up to {pct:.0f}% "
                    f"of the image {'width' if p['direction']=='horizontal' else 'height'}.\n\n"
                    f"This is usually caused by drawing the polygon too large, or choosing "
                    f"the wrong closing direction.\n\nContinue anyway?")
                if not ans: return False
        return True

    def _save(self):
        if self.image is None: return
        if not self.polygons:
            messagebox.showinfo("No polygons", "Draw at least one polygon first."); return
        if not self._safety_check(): return
        stem, ext = os.path.splitext(self.filepath)
        default = (stem + "_stitched" + ext) if self.filepath else "stitched.tif"
        out = filedialog.asksaveasfilename(
            title="Save", initialfile=os.path.basename(default),
            defaultextension=".tif",
            filetypes=[("TIFF","*.tif *.tiff"),("PNG","*.png"),("JPEG","*.jpg"),("All","*.*")])
        if not out: return
        self._status.set("Saving…"); self.root.update_idletasks()

        def _work():
            try:
                result = _stitch(self.image, self.polygons)
                kw = {}
                if out.lower().endswith((".tif",".tiff")): kw = {"compression":"lzw"}
                elif out.lower().endswith((".jpg",".jpeg")): kw = {"quality":95}
                result.save(out, **kw)
                self.root.after(0, lambda: self._status.set(
                    f"Saved: {os.path.basename(out)}\n{result.width}×{result.height} px"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Save error", str(e)))

        threading.Thread(target=_work, daemon=True).start()


# ── entry ─────────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    SlitRemoverApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
