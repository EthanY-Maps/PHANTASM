#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Map Labeler - A tool for annotating text on historic maps.
Supports large images via viewport rendering, zooming, panning, polygon annotation,
selecting/deleting specific annotations, and loading PDFs.
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import json
import os
import sys

# Attempt to import pdf2image for PDF support
try:
    from pdf2image import convert_from_path
    HAS_PDF_SUPPORT = True
except ImportError:
    HAS_PDF_SUPPORT = False

# Increase PIL limit for large maps
Image.MAX_IMAGE_PIXELS = None

def is_point_in_polygon(x, y, poly_points):
    """
    Ray-casting algorithm to check if point (x,y) is inside the polygon.
    """
    n = len(poly_points)
    inside = False
    p1x, p1y = poly_points[0]
    for i in range(n + 1):
        p2x, p2y = poly_points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

class LabelerCanvas(tk.Frame):
    """Canvas with viewport-based zoom, pan, drawing, and selection capabilities."""
    
    def __init__(self, parent, status_callback=None):
        super().__init__(parent)
        self.status_callback = status_callback
        
        # State
        self.image_path = None
        self.original_image = None  # Full resolution PIL Image
        self.tk_image = None        # ImageTk for the CURRENT viewport
        
        self.scale = 1.0
        
        self.annotations = []       # List of polygons
        self.current_points = []    # Points for the polygon currently being drawn
        self.selected_index = None  # Index of the currently selected polygon
        
        self.mode = "view"          # 'view', 'select', 'draw_poly', 'draw_rect', 'draw_tri'
        
        # UI Setup
        self._setup_ui()
        self._bind_events()
        
    def _setup_ui(self):
        # Scrollbars
        self.v_scroll = ttk.Scrollbar(self, orient=tk.VERTICAL)
        self.h_scroll = ttk.Scrollbar(self, orient=tk.HORIZONTAL)
        
        # Canvas
        self.canvas = tk.Canvas(self, bg="#333333", highlightthickness=0,
                                xscrollcommand=self.h_scroll.set,
                                yscrollcommand=self.v_scroll.set)
        
        self.v_scroll.config(command=self.canvas.yview)
        self.h_scroll.config(command=self.canvas.xview)
        
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _bind_events(self):
        self.canvas.bind("<Configure>", self.on_resize)
        self.canvas.bind("<ButtonPress-1>", self.on_left_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        
        # Right click
        self.canvas.bind("<ButtonPress-2>", self.on_right_click) 
        self.canvas.bind("<ButtonPress-3>", self.on_right_click)
        self.canvas.bind("<Control-Button-1>", self.on_right_click) 
        
        # Zoom
        self.canvas.bind("<MouseWheel>", self.on_wheel)
        self.canvas.bind("<Button-4>", self.on_wheel)
        self.canvas.bind("<Button-5>", self.on_wheel)
        
        # Keyboard shortcuts
        self.bind_all("<Control-z>", self.undo_last_point)
        self.bind_all("<Delete>", self.handle_delete_key)
        self.bind_all("<BackSpace>", self.handle_delete_key)
        self.bind_all("<Escape>", self.cancel_drawing)
        self.bind_all("<Return>", lambda e: self.finish_polygon()) 

    def cancel_drawing(self, event=None):
        self.current_points = []
        self.canvas.delete("temp_line")
        self.refresh_display()
        if self.status_callback:
            self.status_callback("Drawing cancelled")

    def undo_last_point(self, event=None):
        if self.current_points:
            self.current_points.pop()
            self.refresh_display()
            if self.status_callback:
                self.status_callback("Removed last point")

    def handle_delete_key(self, event=None):
        if self.selected_index is not None:
            self.delete_selected_polygon()
        else:
            self.delete_last_polygon()

    def delete_selected_polygon(self):
        if self.selected_index is not None and 0 <= self.selected_index < len(self.annotations):
            self.annotations.pop(self.selected_index)
            self.selected_index = None
            self.refresh_display()
            if self.status_callback:
                self.status_callback("Deleted selected polygon")
        else:
            if self.status_callback:
                self.status_callback("No polygon selected to delete")

    def delete_last_polygon(self, event=None):
        if self.annotations and self.selected_index is None:
            self.annotations.pop()
            self.refresh_display()
            if self.status_callback:
                self.status_callback("Deleted last polygon")

    def load_image(self, path):
        try:
            self.image_path = path
            
            # --- PDF Handling Logic ---
            if path.lower().endswith(".pdf"):
                if not HAS_PDF_SUPPORT:
                    messagebox.showerror("Error", "pdf2image library not found.\nPlease install it via: pip install pdf2image\nYou also need Poppler installed on your system.")
                    return
                
                if self.status_callback:
                    self.status_callback("Converting PDF (this may take a moment)...")
                    # Force update to show the loading message
                    self.update_idletasks() 

                # Convert first page only, use high DPI (300) for map detail
                images = convert_from_path(path, dpi=300, first_page=1, last_page=1)
                
                if not images:
                    raise ValueError("Could not extract image from PDF")
                    
                self.original_image = images[0]
            else:
                # Standard Image Load
                self.original_image = Image.open(path)

            self.scale = 1.0
            self.annotations = []
            self.current_points = []
            self.selected_index = None
            
            self.update_scrollregion()
            self.refresh_display()
            
            if self.status_callback:
                self.status_callback(f"Loaded {os.path.basename(path)} ({self.original_image.size[0]}x{self.original_image.size[1]})")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
            if self.status_callback:
                self.status_callback("Load failed")

    def update_scrollregion(self):
        if not self.original_image:
            return
        w, h = self.original_image.size
        sw = int(w * self.scale)
        sh = int(h * self.scale)
        self.canvas.config(scrollregion=(0, 0, sw, sh))

    def on_resize(self, event):
        self.refresh_display()

    def refresh_display(self):
        if not self.original_image:
            return

        x1 = self.canvas.canvasx(0)
        y1 = self.canvas.canvasy(0)
        vw = self.canvas.winfo_width()
        vh = self.canvas.winfo_height()
        
        if vw <= 1 or vh <= 1:
            return
            
        x2 = x1 + vw
        y2 = y1 + vh
        
        ix1 = max(0, int(x1 / self.scale))
        iy1 = max(0, int(y1 / self.scale))
        ix2 = int(x2 / self.scale) + 1
        iy2 = int(y2 / self.scale) + 1
        
        img_w, img_h = self.original_image.size
        ix2 = min(img_w, ix2)
        iy2 = min(img_h, iy2)
        
        if ix2 <= ix1 or iy2 <= iy1:
            return
            
        crop_region = (ix1, iy1, ix2, iy2)
        try:
            cropped = self.original_image.crop(crop_region)
        except Exception:
            return
            
        sx1 = ix1 * self.scale
        sy1 = iy1 * self.scale
        
        target_w = int((ix2 - ix1) * self.scale)
        target_h = int((iy2 - iy1) * self.scale)
        
        if target_w <= 0 or target_h <= 0:
            return
            
        resized = cropped.resize((target_w, target_h), Image.Resampling.NEAREST)
        self.tk_image = ImageTk.PhotoImage(resized)
        
        self.canvas.delete("all")
        self.canvas.create_image(sx1, sy1, image=self.tk_image, anchor="nw")
        
        self.redraw_annotations()

    def redraw_annotations(self):
        for i, poly in enumerate(self.annotations):
            pts = [(x * self.scale, y * self.scale) for x, y in poly['points']]
            flat_pts = [coord for pt in pts for coord in pt]
            
            if i == self.selected_index:
                outline_color = "cyan"
                width = 3
                fill_color = "" 
            else:
                outline_color = "red"
                width = 2
                fill_color = ""

            if len(flat_pts) >= 4:
                self.canvas.create_polygon(flat_pts, outline=outline_color, 
                                           fill=fill_color, width=width, 
                                           tags=("annotation", f"poly_{i}"))
        
        if self.current_points:
            pts = [(x * self.scale, y * self.scale) for x, y in self.current_points]
            flat_pts = [coord for pt in pts for coord in pt]
            
            if len(pts) > 1:
                self.canvas.create_line(flat_pts, fill="yellow", width=2, tags="current")
            
            for pt in pts:
                self.canvas.create_oval(pt[0]-3, pt[1]-3, pt[0]+3, pt[1]+3, fill="yellow", tags="current")

    def screen_to_image(self, sx, sy):
        cx = self.canvas.canvasx(sx)
        cy = self.canvas.canvasy(sy)
        ix = cx / self.scale
        iy = cy / self.scale
        return ix, iy

    def on_left_click(self, event):
        ix, iy = self.screen_to_image(event.x, event.y)

        if self.mode == "select":
            self.handle_selection_click(ix, iy)
            return

        if self.mode == "view":
            self.canvas.scan_mark(event.x, event.y)
        else:
            if self.mode == "draw_poly" and len(self.current_points) >= 3:
                start_pt = self.current_points[0]
                sx = start_pt[0] * self.scale
                sy = start_pt[1] * self.scale
                cx = self.canvas.canvasx(event.x)
                cy = self.canvas.canvasy(event.y)
                dist = ((sx - cx)**2 + (sy - cy)**2)**0.5
                if dist < 10: 
                    self.finish_polygon()
                    return

            self.current_points.append((ix, iy))
            
            if self.mode == "draw_tri" and len(self.current_points) == 3:
                self.finish_polygon()
            elif self.mode == "draw_rect" and len(self.current_points) == 4:
                self.finish_polygon()
            else:
                self.refresh_display()

    def handle_selection_click(self, ix, iy):
        found_index = None
        for i in range(len(self.annotations) - 1, -1, -1):
            poly_points = self.annotations[i]['points']
            if is_point_in_polygon(ix, iy, poly_points):
                found_index = i
                break
        
        if found_index != self.selected_index:
            self.selected_index = found_index
            self.refresh_display()
            if self.status_callback:
                if found_index is not None:
                    self.status_callback(f"Selected Polygon #{found_index}")
                else:
                    self.status_callback("Deselected")
        elif found_index is None:
            self.selected_index = None
            self.refresh_display()

    def on_drag(self, event):
        if self.mode == "view":
            self.canvas.scan_dragto(event.x, event.y, gain=1)
            self.refresh_display()
    
    def on_mouse_move(self, event):
        if self.mode not in ["view", "select"] and self.current_points:
            self.canvas.delete("temp_line")
            last_pt = self.current_points[-1]
            lx = last_pt[0] * self.scale
            ly = last_pt[1] * self.scale
            cx = self.canvas.canvasx(event.x)
            cy = self.canvas.canvasy(event.y)
            self.canvas.create_line(lx, ly, cx, cy, fill="yellow", dash=(4, 2), tags="temp_line")

    def on_release(self, event):
        if self.mode == "view":
            self.refresh_display()

    def on_right_click(self, event):
        if self.mode not in ["view", "select"] and len(self.current_points) > 2:
            self.finish_polygon()
        else:
            self.current_points = []
            self.refresh_display()

    def finish_polygon(self):
        if len(self.current_points) >= 3:
            self.annotations.append({'points': list(self.current_points), 'label': 'text'})
            self.current_points = []
            self.selected_index = len(self.annotations) - 1
            self.refresh_display()
            if self.status_callback:
                self.status_callback(f"Added polygon. Total: {len(self.annotations)}")

    def on_wheel(self, event):
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        ix = cx / self.scale
        iy = cy / self.scale
        
        if event.delta > 0 or event.num == 5:
            self.scale = self.scale * 1.1
        else:
            self.scale = self.scale * 0.9
            
        if self.scale < 0.01: self.scale = 0.01
        if self.scale > 50.0: self.scale = 50.0
        
        self.update_scrollregion()
        
        new_cx = ix * self.scale
        new_cy = iy * self.scale
        
        sr = self.canvas.config("scrollregion")[4]
        if sr:
            total_w = float(sr.split()[2])
            total_h = float(sr.split()[3])
            
            desired_left_x = new_cx - event.x
            desired_top_y = new_cy - event.y
            
            if total_w > 0: self.canvas.xview_moveto(desired_left_x / total_w)
            if total_h > 0: self.canvas.yview_moveto(desired_top_y / total_h)
            
        self.refresh_display()
        if self.status_callback:
            self.status_callback(f"Zoom: {self.scale:.2f}x")

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Historic Map Labeler")
        self.root.geometry("1200x800")
        self._setup_layout()
        
    def _setup_layout(self):
        # Toolbar
        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        # Updated Buttons
        ttk.Button(toolbar, text="Load Image/PDF", command=self.load_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Load JSON", command=self.load_json).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Tools
        self.mode_var = tk.StringVar(value="view")
        modes = [
            ("View/Pan", "view"),
            ("Select", "select"),
            ("Polygon", "draw_poly"),
            ("Rectangle", "draw_rect"),
            ("Triangle", "draw_tri")
        ]
        
        for text, mode in modes:
            ttk.Radiobutton(toolbar, text=text, variable=self.mode_var, 
                           value=mode, command=self.change_mode).pack(side=tk.LEFT, padx=5)
            
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        ttk.Button(toolbar, text="Finish Poly", command=self.finish_polygon_action).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Delete Selected", command=self.delete_selected_action).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Save JSON", command=self.save_json).pack(side=tk.LEFT, padx=2)
        
        # Status
        self.status = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Canvas
        self.labeler = LabelerCanvas(self.root, status_callback=self.update_status)
        self.labeler.pack(fill=tk.BOTH, expand=True)

    def finish_polygon_action(self):
        self.labeler.finish_polygon()
        
    def delete_selected_action(self):
        if self.labeler.selected_index is not None:
            self.labeler.delete_selected_polygon()
        else:
            messagebox.showinfo("Info", "Select a polygon first.")

    def load_image(self):
        # Updated filetypes to include PDF
        path = filedialog.askopenfilename(
            filetypes=[("All Supported", "*.tif *.tiff *.jpg *.png *.bmp *.pdf"),
                       ("Images", "*.tif *.tiff *.jpg *.png *.bmp"),
                       ("PDF", "*.pdf")]
        )
        if path:
            self.labeler.load_image(path)
            
    def load_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if path:
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                
                if "annotations" in data:
                    self.labeler.annotations = data["annotations"]
                    self.labeler.selected_index = None
                    self.labeler.refresh_display()
                    self.update_status(f"Loaded {len(data['annotations'])} annotations from {os.path.basename(path)}")
                else:
                    messagebox.showwarning("Error", "Invalid JSON format: missing 'annotations' key")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load JSON: {e}")
            
    def change_mode(self):
        self.labeler.mode = self.mode_var.get()
        self.labeler.selected_index = None
        self.labeler.refresh_display()
        self.update_status(f"Mode changed to: {self.labeler.mode}")
        
    def save_json(self):
        if not self.labeler.annotations:
            messagebox.showwarning("Warning", "No annotations to save")
            return
            
        initial_file = "annotations.json"
        if self.labeler.image_path:
             initial_file = os.path.splitext(os.path.basename(self.labeler.image_path))[0] + ".json"
             
        path = filedialog.asksaveasfilename(initialfile=initial_file,
                                          defaultextension=".json",
                                          filetypes=[("JSON", "*.json")])
        if path:
            data = {
                "image": self.labeler.image_path,
                "annotations": self.labeler.annotations
            }
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            self.update_status(f"Saved to {path}")

    def update_status(self, msg):
        self.status.config(text=msg)

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()