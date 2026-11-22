import tkinter as tk
from tkinter import filedialog, Scale, HORIZONTAL, Button, Label, Frame
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import cv2

class AdvancedMapEditor:
    def __init__(self, root, img_path=None, mask_path=None):
        self.root = root
        self.root.title("Map Segmentation Tool: Brush, Polygon & Zoom")
        
        # --- State ---
        self.brush_size = 20
        self.mode = "brush_draw" # Modes: brush_draw, brush_erase, poly_draw, poly_erase
        self.alpha = 0.4
        self.pan_speed = 50
        
        self.zoom = 1.0
        
        # Viewport Camera (Absolute Top-Left of the visible area)
        self.offset_x = 0  
        self.offset_y = 0
        
        # Current Viewport Size (Screen Pixels)
        self.view_w = 800  
        self.view_h = 600
        
        # Interaction State
        self.last_draw_x, self.last_draw_y = None, None # For Brush
        self.poly_points = [] # List of (abs_x, abs_y) for current polygon
        self.resize_timer = None # For debouncing window resizes
        
        # --- Load Data ---
        self.img_path = img_path if img_path else filedialog.askopenfilename(title="Select Base Map Image")
        if not self.img_path: return
        self.mask_path = mask_path if mask_path else filedialog.askopenfilename(title="Select .npy Prediction File")
        if not self.mask_path: return

        print("Loading data...")
        
        # Load Base (OpenCV -> RGB -> PIL)
        base_cv = cv2.imread(self.img_path)
        base_cv = cv2.cvtColor(base_cv, cv2.COLOR_BGR2RGB)
        self.full_height, self.full_width, _ = base_cv.shape
        self.base_pil_full = Image.fromarray(base_cv)
        
        # Load Mask
        raw_mask = np.load(self.mask_path)
        raw_mask = np.squeeze(raw_mask)
        if raw_mask.max() <= 1.0:
            raw_mask = (raw_mask > 0.5).astype(np.uint8) * 255
        else:
            raw_mask = raw_mask.astype(np.uint8)
            
        if raw_mask.shape != (self.full_height, self.full_width):
            raw_mask = cv2.resize(raw_mask, (self.full_width, self.full_height), interpolation=cv2.INTER_NEAREST)

        self.mask_pil_full = Image.fromarray(raw_mask, mode='L')
        self.draw_engine = ImageDraw.Draw(self.mask_pil_full)

        # --- GUI Layout ---
        
        # Toolbar Row 1: Zoom & Navigation
        self.row1 = Frame(root, bg="#ddd")
        self.row1.pack(side=tk.TOP, fill=tk.X)
        Label(self.row1, text="View:", bg="#ddd").pack(side=tk.LEFT, padx=5)
        Button(self.row1, text="Zoom In (+)", command=self.zoom_in).pack(side=tk.LEFT, padx=2)
        Button(self.row1, text="Zoom Out (-)", command=self.zoom_out).pack(side=tk.LEFT, padx=2)
        Label(self.row1, text="   (Use WASD to Pan)", bg="#ddd").pack(side=tk.LEFT, padx=10)
        
        # Toolbar Row 2: Tools
        self.row2 = Frame(root, bg="#f0f0f0")
        self.row2.pack(side=tk.TOP, fill=tk.X)
        
        Label(self.row2, text="Brush:", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        Button(self.row2, text="Paint", bg="#90ee90", command=lambda: self.set_mode("brush_draw")).pack(side=tk.LEFT, padx=2)
        Button(self.row2, text="Erase", bg="#ffcccb", command=lambda: self.set_mode("brush_erase")).pack(side=tk.LEFT, padx=2)
        
        Label(self.row2, text="  |  Polygon:", bg="#f0f0f0").pack(side=tk.LEFT, padx=5)
        Button(self.row2, text="Add Area", bg="#90ee90", command=lambda: self.set_mode("poly_draw")).pack(side=tk.LEFT, padx=2)
        Button(self.row2, text="Remove Area", bg="#ffcccb", command=lambda: self.set_mode("poly_erase")).pack(side=tk.LEFT, padx=2)
        
        Button(self.row2, text="SAVE .NPY", bg="lightblue", command=self.save_result).pack(side=tk.RIGHT, padx=10)

        # Toolbar Row 3: Sliders & Info
        self.row3 = Frame(root, bg="#f0f0f0")
        self.row3.pack(side=tk.TOP, fill=tk.X)
        
        Label(self.row3, text="Brush Size:").pack(side=tk.LEFT, padx=5)
        self.s_size = Scale(self.row3, from_=5, to=200, orient=HORIZONTAL, command=self.update_params)
        self.s_size.set(self.brush_size)
        self.s_size.pack(side=tk.LEFT)
        
        Label(self.row3, text="Move Speed:").pack(side=tk.LEFT, padx=5)
        self.s_speed = Scale(self.row3, from_=10, to=300, orient=HORIZONTAL, command=self.update_params)
        self.s_speed.set(self.pan_speed)
        self.s_speed.pack(side=tk.LEFT)
        
        self.lbl_mode = Label(self.row3, text=f"Mode: {self.mode}", fg="blue", font=("Arial", 10, "bold"))
        self.lbl_mode.pack(side=tk.RIGHT, padx=10)

        # Canvas
        self.canvas = tk.Canvas(root, bg="#333", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # --- Bindings ---
        self.canvas.bind("<Configure>", self.on_resize)
        
        # Mouse
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", self.on_right_click) # Right click to close poly
        self.canvas.bind("<Motion>", self.on_mouse_move)    # For elastic lines
        
        # Keys
        root.bind("<w>", lambda e: self.pan_key(0, -1))
        root.bind("<a>", lambda e: self.pan_key(-1, 0))
        root.bind("<s>", lambda e: self.pan_key(0, 1))
        root.bind("<d>", lambda e: self.pan_key(1, 0))
        root.bind("<Return>", self.on_right_click) # Enter to close poly
        root.bind("<Escape>", self.cancel_poly)

        self.update_viewport()

    # --- Mode & Params ---
    def set_mode(self, mode):
        self.mode = mode
        self.lbl_mode.config(text=f"Mode: {mode}")
        self.poly_points = []
        self.update_viewport()

    def update_params(self, _):
        self.brush_size = self.s_size.get()
        self.pan_speed = self.s_speed.get()

    # --- Coordinate Helpers ---
    def screen_to_abs(self, sx, sy):
        ax = self.offset_x + (sx / self.zoom)
        ay = self.offset_y + (sy / self.zoom)
        return ax, ay

    def abs_to_screen(self, ax, ay):
        sx = (ax - self.offset_x) * self.zoom
        sy = (ay - self.offset_y) * self.zoom
        return sx, sy

    # --- Zoom Logic (Now Center-Targeted) ---
    def _apply_zoom(self, zoom_factor):
        # 1. Find the absolute coordinates of the center of the current screen
        center_ax = self.offset_x + (self.view_w / self.zoom) / 2
        center_ay = self.offset_y + (self.view_h / self.zoom) / 2
        
        # 2. Apply zoom
        self.zoom *= zoom_factor
        
        # 3. Adjust offset so the center stays in the same place
        self.offset_x = center_ax - (self.view_w / self.zoom) / 2
        self.offset_y = center_ay - (self.view_h / self.zoom) / 2
        
        # Clamp offsets to prevent panning infinitely out of bounds
        self.clamp_offsets()
        self.update_viewport()

    def zoom_in(self):
        self._apply_zoom(1.25)

    def zoom_out(self):
        self._apply_zoom(1 / 1.25)

    def clamp_offsets(self):
        src_w = self.view_w / self.zoom
        src_h = self.view_h / self.zoom
        self.offset_x = max(0, min(self.offset_x, self.full_width - src_w))
        self.offset_y = max(0, min(self.offset_y, self.full_height - src_h))

    # --- Core Viewport Renderer ---
    def update_viewport(self):
        src_w = self.view_w / self.zoom
        src_h = self.view_h / self.zoom
        
        x1 = max(0, int(self.offset_x))
        y1 = max(0, int(self.offset_y))
        x2 = min(self.full_width, int(self.offset_x + src_w))
        y2 = min(self.full_height, int(self.offset_y + src_h))
        
        crop_box = (x1, y1, x2, y2)
        
        try:
            base_crop = self.base_pil_full.crop(crop_box)
            mask_crop = self.mask_pil_full.crop(crop_box)
            
            target_w = int((x2 - x1) * self.zoom)
            target_h = int((y2 - y1) * self.zoom)
            
            if target_w <= 0 or target_h <= 0: return

            base_view = base_crop.resize((target_w, target_h), Image.NEAREST)
            mask_view = mask_crop.resize((target_w, target_h), Image.NEAREST)
            
        except Exception as e:
            print(f"Render error: {e}")
            return

        mask_np = np.array(mask_view)
        zeros = np.zeros_like(mask_np)
        base_view = base_view.convert("RGBA")
        
        red_layer = Image.new('RGBA', base_view.size, (0, 0, 255, 0))
        d = ImageDraw.Draw(red_layer)
        d.bitmap((0,0), mask_view, fill=(0, 0, 255, int(255 * self.alpha)))
        
        final_view = Image.alpha_composite(base_view, red_layer)
        
        self.tk_image = ImageTk.PhotoImage(final_view)
        self.canvas.delete("all") 
        self.canvas.create_image(0, 0, image=self.tk_image, anchor=tk.NW, tag="bg_map")
        
        if "poly" in self.mode and len(self.poly_points) > 0:
            screen_pts = [self.abs_to_screen(ax, ay) for ax, ay in self.poly_points]
            if len(screen_pts) > 1:
                self.canvas.create_line(screen_pts, fill="yellow", width=2, tag="poly_line")
            for sx, sy in screen_pts:
                self.canvas.create_oval(sx-3, sy-3, sx+3, sy+3, fill="yellow", outline="black", tag="poly_node")

    # --- Interaction Logic ---
    def on_click(self, event):
        if "brush" in self.mode:
            self.last_draw_x, self.last_draw_y = event.x, event.y
            self.paint_brush(event)
        elif "poly" in self.mode:
            ax, ay = self.screen_to_abs(event.x, event.y)
            self.poly_points.append((ax, ay))
            self.update_viewport()

    def on_drag(self, event):
        if "brush" in self.mode:
            self.paint_brush(event)

    def on_release(self, event):
        if "brush" in self.mode:
            self.last_draw_x = None
            self.update_viewport()

    def on_mouse_move(self, event):
        if "poly" in self.mode and len(self.poly_points) > 0:
            self.canvas.delete("elastic")
            last_ax, last_ay = self.poly_points[-1]
            lx, ly = self.abs_to_screen(last_ax, last_ay)
            self.canvas.create_line(lx, ly, event.x, event.y, fill="yellow", dash=(4, 4), tag="elastic")

    def on_right_click(self, event):
        if "poly" in self.mode and len(self.poly_points) > 2:
            fill_val = 255 if "draw" in self.mode else 0
            self.draw_engine.polygon(self.poly_points, fill=fill_val)
            print(f"Polygon finished with {len(self.poly_points)} points.")
            self.poly_points = [] 
            self.canvas.delete("elastic")
            self.update_viewport()

    def cancel_poly(self, event):
        self.poly_points = []
        self.canvas.delete("elastic")
        self.update_viewport()
        print("Polygon cancelled")

    def paint_brush(self, event):
        # 1. Visual Feedback (Screen Coords)
        color = "red" if "draw" in self.mode else "blue"
        screen_diameter = self.brush_size * self.zoom 
        
        if self.last_draw_x:
            self.canvas.create_line(self.last_draw_x, self.last_draw_y, event.x, event.y, 
                                  fill=color, width=screen_diameter, capstyle=tk.ROUND, tag="temp_stroke")
        
        # 2. Data Update (Absolute Coords)
        ax, ay = self.screen_to_abs(event.x, event.y)
        lax, lay = self.screen_to_abs(self.last_draw_x, self.last_draw_y)
        
        fill_val = 255 if "draw" in self.mode else 0
        absolute_r = self.brush_size / 2 # Removed redundant math here
        
        self.draw_engine.line([lax, lay, ax, ay], fill=fill_val, width=self.brush_size)
        self.draw_engine.ellipse([ax - absolute_r, ay - absolute_r, ax + absolute_r, ay + absolute_r], fill=fill_val)

        self.last_draw_x = event.x
        self.last_draw_y = event.y

    # --- Pan Logic ---
    def pan_key(self, x_dir, y_dir):
        self.offset_x += x_dir * self.pan_speed
        self.offset_y += y_dir * self.pan_speed
        self.clamp_offsets()
        self.update_viewport()

    # --- Debounced Resize Logic ---
    def on_resize(self, event):
        # Ignore configure events from child widgets (like frames/buttons)
        if event.widget == self.canvas:
            self.view_w = event.width
            self.view_h = event.height
            
            # Cancel previous timer if it exists
            if self.resize_timer:
                self.root.after_cancel(self.resize_timer)
                
            # Wait 150ms after user stops resizing before redrawing
            self.resize_timer = self.root.after(150, self.update_viewport)

    def save_result(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".npy", title="Save Corrected Mask")
        if save_path:
            final_mask = np.array(self.mask_pil_full)
            final_binary = (final_mask > 127).astype(np.uint8) # Changed to standard uint8
            np.save(save_path, final_binary)
            print(f"Saved to {save_path}")

if __name__ == "__main__":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except: pass

    root = tk.Tk()
    root.geometry("1000x800")
    app = AdvancedMapEditor(root)
    root.mainloop()