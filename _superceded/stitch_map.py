import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

# Allow Pillow to load massive .tif map files without throwing DecompressionBomb errors
Image.MAX_IMAGE_PIXELS = None

# ==========================================
#  MATH HELPER: Robust Matrix Solver
# ==========================================
def find_coeffs(pa, pb):
    """
    Calculates coefficients for perspective transform.
    Uses Gaussian elimination with Partial Pivoting to avoid divide-by-zero errors.
    """
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = matrix
    B = [p for p1, p2 in zip(pa, pb) for p in p2]
    
    n = 8
    
    for i in range(n):
        max_row = i
        for k in range(i + 1, n):
            if abs(A[k][i]) > abs(A[max_row][i]):
                max_row = k
        
        A[i], A[max_row] = A[max_row], A[i]
        B[i], B[max_row] = B[max_row], B[i]
        
        pivot = A[i][i]
        if abs(pivot) < 1e-10: 
            continue 
            
        for j in range(i + 1, n):
            factor = A[j][i] / pivot
            B[j] -= factor * B[i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]

    x = [0] * n
    for i in range(n - 1, -1, -1):
        if abs(A[i][i]) < 1e-10: continue
        sum_ax = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (B[i] - sum_ax) / A[i][i]
        
    return x

# ==========================================
#  MAIN APPLICATION
# ==========================================
class JigsawStudio:
    def __init__(self, root):
        self.root = root
        self.root.title("Jigsaw Studio: Map Mapper")
        self.root.geometry("1400x900")
        
        # --- State ---
        self.pieces = []  
        self.zoom_level = 1.0
        self.selected_index = -1
        self.drag_start = None
        self.library_paths = {} 
        
        # Warp State
        self.warp_mode = False       
        self.warp_handles = []       
        self.active_handle = -1      

        self.setup_layout()
        self.bind_controls()

    def setup_layout(self):
        self.paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=5, bg="#333")
        self.paned.pack(fill=tk.BOTH, expand=True)

        # Sidebar
        self.sidebar = tk.Frame(self.paned, bg="#444", width=300)
        self.paned.add(self.sidebar)

        tk.Label(self.sidebar, text="Library", bg="#444", fg="white", font=("Arial", 12, "bold")).pack(pady=5)
        
        btn_frame = tk.Frame(self.sidebar, bg="#444")
        btn_frame.pack(fill=tk.X, padx=5)
        tk.Button(btn_frame, text="📂 Folder", command=self.load_folder, bg="#666", fg="white").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        tk.Button(btn_frame, text="💾 Save", command=self.save_result, bg="#666", fg="white").pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)

        self.file_list = tk.Listbox(self.sidebar, bg="#222", fg="#eee", bd=0, highlightthickness=0)
        self.file_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.file_list.bind("<Double-Button-1>", self.on_library_double_click)

        help_text = (
            "CONTROLS (Keyboard Only):\n"
            "WASD : Pan Camera\n"
            "+ / - : Zoom Camera\n"
            "T : Toggle Warp Mode\n"
            "R : Re-Crop Selected\n"
            "[ / ] : Resize Piece\n"
            ", / . : Layer Back/Front\n"
            "Del : Delete Piece"
        )
        tk.Label(self.sidebar, text=help_text, bg="#444", fg="#aaa", justify=tk.LEFT, font=("Consolas", 10)).pack(pady=10, padx=5, anchor="w")

        # Canvas
        self.canvas_frame = tk.Frame(self.paned, bg="#111")
        self.paned.add(self.canvas_frame)

        self.canvas = tk.Canvas(self.canvas_frame, bg="#111", highlightthickness=0, scrollregion=(-5000, -5000, 5000, 5000))
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.xview_moveto(0.5)
        self.canvas.yview_moveto(0.5)

    def bind_controls(self):
        self.root.bind("-", self.zoom_out)
        self.root.bind("=", self.zoom_in)
        for k in ["w", "s", "a", "d"]:
            self.root.bind(k, self.pan_logic)
            
        self.root.bind("[", lambda e: self.scale_piece(0.9)) 
        self.root.bind("]", lambda e: self.scale_piece(1.1))
        self.root.bind("t", lambda e: self.toggle_warp_mode())
        self.root.bind("r", lambda e: self.recrop_selected())
        self.root.bind("<Delete>", lambda e: self.delete_piece())
        self.root.bind("<BackSpace>", lambda e: self.delete_piece())
        self.root.bind(",", lambda e: self.layer_down())
        self.root.bind(".", lambda e: self.layer_up())

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Motion>", self.on_hover) 

    def pan_logic(self, event):
        key = event.keysym.lower()
        if key == 'w': self.canvas.yview_scroll(-2, "units")
        elif key == 's': self.canvas.yview_scroll(2, "units")
        elif key == 'a': self.canvas.xview_scroll(-2, "units")
        elif key == 'd': self.canvas.xview_scroll(2, "units")

    # ==========================
    # Logic: Scaling
    # ==========================
    def scale_piece(self, factor):
        if self.selected_index == -1: return
        p = self.pieces[self.selected_index]
        w, h = p['img_data'].size
        cx, cy = p['x'] + w/2, p['y'] + h/2
        
        new_w, new_h = int(w * factor), int(h * factor)
        if new_w < 5 or new_h < 5: return 
        
        p['img_data'] = p['img_data'].resize((new_w, new_h), Image.Resampling.LANCZOS)
        p['x'], p['y'] = cx - new_w/2, cy - new_h/2
        
        if 'base_img' in p:
            del p['base_img'] 
            p['warp_corners'] = [(0,0), (new_w,0), (new_w,new_h), (0,new_h)]
        
        if self.warp_mode: self.exit_warp_mode()
        self.redraw()

    # ==========================
    # Logic: Warping
    # ==========================
    def toggle_warp_mode(self):
        if self.selected_index == -1: return
        if self.warp_mode:
            self.exit_warp_mode()
        else:
            self.warp_mode = True
            p = self.pieces[self.selected_index]
            if 'base_img' not in p:
                p['base_img'] = p['img_data'].copy()
                w, h = p['base_img'].size
                p['warp_corners'] = [(0,0), (w,0), (w,h), (0,h)]
            self.draw_warp_handles()
            self.root.config(cursor="fleur")

    def exit_warp_mode(self):
        self.warp_mode = False
        self.canvas.delete("warp_handle")
        self.warp_handles = []
        self.active_handle = -1
        self.root.config(cursor="")
        self.redraw()

    def draw_warp_handles(self):
        self.canvas.delete("warp_handle")
        self.warp_handles = []
        if self.selected_index == -1: return
        p = self.pieces[self.selected_index]
        bx, by = p['x'], p['y']
        
        for i, (cx, cy) in enumerate(p['warp_corners']):
            screen_x, screen_y = (bx + cx) * self.zoom_level, (by + cy) * self.zoom_level
            r = 8
            hid = self.canvas.create_oval(screen_x-r, screen_y-r, screen_x+r, screen_y+r, 
                                          fill="#ff4444", outline="white", width=2, tags="warp_handle")
            self.warp_handles.append(hid)

        coords = []
        for (cx, cy) in p['warp_corners']:
            coords.append((bx + cx) * self.zoom_level)
            coords.append((by + cy) * self.zoom_level)
        coords.append(coords[0])
        coords.append(coords[1])
        self.canvas.create_line(coords, fill="#ff4444", dash=(2,2), tags="warp_handle")
        self.canvas.tag_raise("warp_handle")

    def apply_warp(self):
        p = self.pieces[self.selected_index]
        base = p['base_img']
        w, h = base.size
        
        src_pts = [(0,0), (w,0), (w,h), (0,h)]
        corners = p['warp_corners'] 
        
        min_x = min(c[0] for c in corners)
        min_y = min(c[1] for c in corners)
        max_x = max(c[0] for c in corners)
        max_y = max(c[1] for c in corners)
        
        new_w = max(1, int(max_x - min_x))
        new_h = max(1, int(max_y - min_y))
        
        dst_pts = [(c[0] - min_x, c[1] - min_y) for c in corners]
        
        try:
            coeffs = find_coeffs(dst_pts, src_pts)
            new_img = base.transform((new_w, new_h), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
            p['img_data'] = new_img
            
            p['x'] += min_x
            p['y'] += min_y
            p['warp_corners'] = [(c[0] - min_x, c[1] - min_y) for c in corners]
            
            self.redraw()
            self.draw_warp_handles()
            return True
        except Exception as e:
            print(f"Warp Rejected: Math produced impossible geometry. ({e})")
            return False

    # ==========================
    # Logic: General
    # ==========================
    def load_folder(self):
        d = filedialog.askdirectory()
        if not d: return
        self.file_list.delete(0, tk.END)
        self.library_paths = {}
        exts = ('.jpg', '.png', '.jpeg', '.gif', '.tif', '.tiff')
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(exts):
                self.file_list.insert(tk.END, f)
                self.library_paths[f] = os.path.join(d, f)

    def on_library_double_click(self, event):
        sel = self.file_list.curselection()
        if not sel: return
        path = self.library_paths[self.file_list.get(sel[0])]
        self.open_crop_window(path, False)

    # ---------------------------------------------------------
    # UPDATED RECTANGULAR CROP TOOL 
    # ---------------------------------------------------------
    def open_crop_window(self, path, is_edit):
        cw = tk.Toplevel(self.root)
        cw.title("DRAG-TO-CROP TOOL")
        
        sw_screen, sh_screen = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        cw_w, cw_h = int(sw_screen * 0.8), int(sh_screen * 0.8)
        cw.geometry(f"{cw_w}x{cw_h}")
        cw.focus_force() 
        
        orig = Image.open(path).convert("RGBA")
        scale = min((cw_w-50)/orig.width, (cw_h-100)/orig.height, 1.0)
        dw, dh = int(orig.width*scale), int(orig.height*scale)
        
        disp = orig.resize((dw, dh), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(disp)
        
        toolbar = tk.Frame(cw, bg="#333", height=40)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        c = tk.Canvas(cw, width=dw, height=dh, bg="black", cursor="crosshair")
        c.pack(pady=10)
        c.create_image(0,0, image=tk_img, anchor=tk.NW)
        c.image = tk_img
        
        # State for our drag-to-crop bounding box
        crop_state = {'start_x': 0, 'start_y': 0, 'rect_id': None, 'bbox': None}
        
        def on_press(e):
            crop_state['start_x'] = c.canvasx(e.x)
            crop_state['start_y'] = c.canvasy(e.y)
            if crop_state['rect_id']:
                c.delete(crop_state['rect_id'])
            crop_state['rect_id'] = c.create_rectangle(
                crop_state['start_x'], crop_state['start_y'], 
                crop_state['start_x'], crop_state['start_y'], 
                outline="cyan", width=2, dash=(4,4)
            )

        def on_drag(e):
            if crop_state['rect_id']:
                cur_x = c.canvasx(e.x)
                cur_y = c.canvasy(e.y)
                c.coords(crop_state['rect_id'], crop_state['start_x'], crop_state['start_y'], cur_x, cur_y)

        def on_release(e):
            if not crop_state['rect_id']: return
            cur_x = c.canvasx(e.x)
            cur_y = c.canvasy(e.y)
            x1 = min(crop_state['start_x'], cur_x)
            y1 = min(crop_state['start_y'], cur_y)
            x2 = max(crop_state['start_x'], cur_x)
            y2 = max(crop_state['start_y'], cur_y)
            crop_state['bbox'] = (x1, y1, x2, y2)

        c.bind("<ButtonPress-1>", on_press)
        c.bind("<B1-Motion>", on_drag)
        c.bind("<ButtonRelease-1>", on_release)
        
        def finish(e=None):
            box = crop_state['bbox']
            # Make sure a valid box was drawn (at least 5x5 pixels)
            if not box or (box[2] - box[0] < 5) or (box[3] - box[1] < 5): 
                messagebox.showwarning("Crop", "Please click and drag to draw a valid rectangle!")
                return
            
            # Convert screen coordinates back to original image scale
            x1 = max(0, box[0] / scale)
            y1 = max(0, box[1] / scale)
            x2 = min(orig.width, box[2] / scale)
            y2 = min(orig.height, box[3] / scale)

            # Crop it!
            res = orig.crop((int(x1), int(y1), int(x2), int(y2)))
            
            if is_edit:
                if self.selected_index != -1:
                    self.exit_warp_mode()
                    p = self.pieces[self.selected_index]
                    p['img_data'] = res
                    p.pop('base_img', None)
                    w, h = res.size
                    p['warp_corners'] = [(0,0), (w,0), (w,h), (0,h)]
                    self.redraw()
            else:
                self.add_piece(res, path)
            cw.destroy()

        tk.Button(toolbar, text="✅ FINISH CROP (Enter)", command=finish, bg="#0078D7", fg="white", font=("Arial", 12)).pack(pady=5)
        cw.bind("<Return>", finish)

    def add_piece(self, img, path):
        self.canvas.update_idletasks()
        
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        w = w if w > 1 else 1000
        h = h if h > 1 else 800

        cx = self.canvas.canvasx(w/2)
        cy = self.canvas.canvasy(h/2)
        
        self.pieces.append({
            'img_data': img,
            'x': cx/self.zoom_level - img.width/2,
            'y': cy/self.zoom_level - img.height/2,
            'source_path': path,
            'warp_corners': [(0,0), (img.width,0), (img.width,img.height), (0,img.height)]
        })
        self.selected_index = len(self.pieces)-1
        self.redraw()

    def redraw(self):
        self.canvas.delete("img_obj")
        self.canvas.delete("sel_outline")
        
        for i, p in enumerate(self.pieces):
            w, h = p['img_data'].size
            dw, dh = int(w*self.zoom_level), int(h*self.zoom_level)
            if dw<1 or dh<1: continue
            
            disp = p['img_data'].resize((dw, dh), Image.Resampling.NEAREST)
            tk_img = ImageTk.PhotoImage(disp)
            p['tk_ref'] = tk_img
            
            dx, dy = p['x']*self.zoom_level, p['y']*self.zoom_level
            p['id'] = self.canvas.create_image(dx, dy, image=tk_img, anchor=tk.NW, tags="img_obj")
            
            if i == self.selected_index and not self.warp_mode:
                self.canvas.create_rectangle(dx, dy, dx+dw, dy+dh, outline="cyan", width=2, dash=(4,4), tags="sel_outline")

        if self.warp_mode:
            self.canvas.tag_raise("warp_handle")

    # ==========================
    # Interaction
    # ==========================
    def on_hover(self, event):
        if not self.warp_mode: 
            self.canvas.config(cursor="")
            return
            
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        items = self.canvas.find_overlapping(x-5, y-5, x+5, y+5)
        is_over_handle = any(i in self.warp_handles for i in items)
        
        if is_over_handle:
            self.canvas.config(cursor="crosshair") 
        else:
            self.canvas.config(cursor="fleur")

    def on_click(self, event):
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        
        if self.warp_mode:
            items = self.canvas.find_overlapping(x-10, y-10, x+10, y+10)
            found_handles = [i for i in items if i in self.warp_handles]
            
            if found_handles:
                self.active_handle = self.warp_handles.index(found_handles[-1])
                self.drag_start = (event.x, event.y)
                return 

        items = self.canvas.find_overlapping(x, y, x+1, y+1)
        img_ids = [pid['id'] for pid in self.pieces if pid.get('id')]
        clicked_imgs = [i for i in items if i in img_ids]
        
        if clicked_imgs:
            top_id = clicked_imgs[-1]
            for i, p in enumerate(self.pieces):
                if p['id'] == top_id:
                    if self.selected_index != i:
                        self.exit_warp_mode() 
                    self.selected_index = i
                    break
        else:
            self.exit_warp_mode()
            self.selected_index = -1
            
        self.drag_start = (event.x, event.y)
        self.redraw()

    def on_drag(self, event):
        if not self.drag_start: return
        dx = (event.x - self.drag_start[0]) / self.zoom_level
        dy = (event.y - self.drag_start[1]) / self.zoom_level
        
        if self.warp_mode and self.active_handle != -1:
            p = self.pieces[self.selected_index]
            
            original_corner = p['warp_corners'][self.active_handle]
            cx, cy = original_corner
            p['warp_corners'][self.active_handle] = (cx + dx, cy + dy)
            
            success = self.apply_warp() 
            if not success:
                p['warp_corners'][self.active_handle] = original_corner
                self.canvas.config(bg="#551111")
                self.root.after(100, lambda: self.canvas.config(bg="#111"))

            self.drag_start = (event.x, event.y)
            return

        elif self.selected_index != -1 and not self.warp_mode:
            p = self.pieces[self.selected_index]
            p['x'] += dx
            p['y'] += dy
            self.drag_start = (event.x, event.y)
            self.redraw()

    def on_release(self, event):
        self.drag_start = None
        self.active_handle = -1

    def recrop_selected(self):
        if self.selected_index != -1:
            self.open_crop_window(self.pieces[self.selected_index]['source_path'], True)

    def delete_piece(self):
        if self.selected_index != -1:
            del self.pieces[self.selected_index]
            self.exit_warp_mode()
            self.selected_index = -1
            self.redraw()

    def layer_up(self):
        if self.selected_index < len(self.pieces)-1 and self.selected_index != -1:
            i = self.selected_index
            self.pieces[i], self.pieces[i+1] = self.pieces[i+1], self.pieces[i]
            self.selected_index += 1
            self.redraw()

    def layer_down(self):
        if self.selected_index > 0:
            i = self.selected_index
            self.pieces[i], self.pieces[i-1] = self.pieces[i-1], self.pieces[i]
            self.selected_index -= 1
            self.redraw()

    def zoom_in(self, e=None):
        self.zoom_level *= 1.1
        self.redraw()
        if self.warp_mode: self.draw_warp_handles()

    def zoom_out(self, e=None):
        self.zoom_level *= 0.9
        self.redraw()
        if self.warp_mode: self.draw_warp_handles()

    def save_result(self):
        if not self.pieces: return
        min_x = min(p['x'] for p in self.pieces)
        min_y = min(p['y'] for p in self.pieces)
        max_x = max(p['x'] + p['img_data'].width for p in self.pieces)
        max_y = max(p['y'] + p['img_data'].height for p in self.pieces)
        w, h = int(max_x - min_x), int(max_y - min_y)
        bg = Image.new("RGBA", (w, h), (0,0,0,0))
        for p in self.pieces:
            bg.paste(p['img_data'], (int(p['x']-min_x), int(p['y']-min_y)), p['img_data'])
        f = filedialog.asksaveasfilename(defaultextension=".png")
        if f: bg.save(f)

if __name__ == "__main__":
    root = tk.Tk()
    app = JigsawStudio(root)
    root.mainloop()