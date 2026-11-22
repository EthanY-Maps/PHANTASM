import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from PIL import Image, ImageTk
from pdf2image import convert_from_path
import json
from PIL import Image, ImageTk, ImageDraw
import os
import sys

# --- CONFIGURATION ---
BUFFER_SIZE = 10        # Pixels of context to show around the polygon
CONTEXT_BUFFER = 200    # Pixels to show in the larger context tile  <-- ADD THIS
DEFAULT_CLASS = "text"  # If 'label' is missing, this value will be added
BOX_COLOR = "red"       # Color of the bounding box
BOX_WIDTH = 1          # Thickness of the bounding box line
PROPERTY_KEY = "truth_text" # The property where your typed text is saved
ROUND_DECIMALS = 4      # Round coordinates for cleaner JSON

# IMPORTANT: Does your JSON imply a specific page? 
# If your JSON covers the whole PDF, it likely needs a 'page' or 'group_id' key.
# If the JSON is ONLY for Page 1, set this to 0.
DEFAULT_PAGE_INDEX = 0 
# ---------------------

class LabelingApp:
    def __init__(self, root, pdf_path, json_path, buffer):
        self.root = root
        self.root.title("PDF Truth Text Labeler")
        self.buffer = buffer
        self.json_path = json_path
        
        # 1. Check Files
        if not os.path.exists(json_path):
            messagebox.showerror("Error", f"JSON file not found: {json_path}")
            sys.exit(1)
        if not os.path.exists(pdf_path):
            messagebox.showerror("Error", f"Document file not found: {pdf_path}")
            sys.exit(1)

        # 2. Load JSON Data
        try:
            with open(json_path, 'r') as f:
                self.data = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load JSON: {e}")
            sys.exit(1)

        # 3. Load PDF/Image
        if pdf_path.lower().endswith(".pdf"):
            try:
                print("Converting PDF to images... this may take a moment.")
                self.pages = convert_from_path(pdf_path, dpi=300) 
                print(f"Loaded {len(self.pages)} pages.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load PDF (Poppler installed?): {e}")
                sys.exit(1)
        else: 
            try:
                print("Loading Image file...")
                self.pages = [Image.open(pdf_path)]
                print(f"Loaded {len(self.pages)} pages.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load Image: {e}")
                sys.exit(1)

        # 4. Locate Annotations List
        if isinstance(self.data, dict) and 'annotations' in self.data:
            self.shapes = self.data['annotations']
        elif isinstance(self.data, list):
            self.shapes = self.data
        else:
            messagebox.showerror("Error", "JSON structure not recognized")
            sys.exit(1)

        self.current_index = 0
        
        # 5. Setup GUI
        self.image_panel = tk.Label(root)
        self.image_panel.pack(pady=10, padx=10)

        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(root, textvariable=self.entry_var, font=("Arial", 16), width=50)
        self.entry.pack(pady=5)
        
        # --- KEY BINDINGS ---
        self.entry.bind('<Return>', self.save_and_next)     # Enter = Save & Next
        self.root.bind('<Escape>', self.next_shape)         # Esc = Skip (Next without saving)
        self.root.bind('<Up>', self.prev_shape)             # Up Arrow = Previous shape
        self.root.bind('<Control-space>', self.show_context) # Ctrl+Space = Show Context
        self.entry.focus_set()

        self.info_label = tk.Label(root, text="", font=("Arial", 12))
        self.info_label.pack(pady=5)

        # --- BUTTONS ---
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Previous (Up)", command=self.prev_shape).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Context (Ctrl+Space)", command=self.show_context).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Skip (Esc)", command=self.next_shape).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Save & Next (Enter)", command=self.save_and_next, bg="#dddddd").pack(side=tk.LEFT, padx=5)
        # Start
        self.find_first_unlabeled()
        self.load_current_shape()

    def find_first_unlabeled(self):
        """Finds first shape without the text property."""
        for i, shape in enumerate(self.shapes):
            if PROPERTY_KEY not in shape or str(shape[PROPERTY_KEY]).strip() == "":
                self.current_index = i
                break

    def get_bbox(self, points):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return min(xs), min(ys), max(xs), max(ys)

    def load_current_shape(self):
        if self.current_index >= len(self.shapes):
            messagebox.showinfo("Done", "All annotations labeled!")
            self.root.destroy()
            return

        shape = self.shapes[self.current_index]
        if 'points' not in shape:
            self.next_shape()
            return

        points = shape['points']
        
        # --- PAGE HANDLING ---
        page_idx = shape.get('page_index', DEFAULT_PAGE_INDEX) 
        
        if page_idx >= len(self.pages):
            print(f"Warning: JSON asks for page {page_idx}, but document only has {len(self.pages)} pages. using last page.")
            page_idx = len(self.pages) - 1

        target_image = self.pages[page_idx]
        # ---------------------

        x_min, y_min, x_max, y_max = self.get_bbox(points)
        
        # Crop with buffer
        w, h = target_image.size
        left = max(0, int(x_min) - self.buffer)
        top = max(0, int(y_min) - self.buffer)
        right = min(w, int(x_max) + self.buffer)
        bottom = min(h, int(y_max) + self.buffer)

        cropped = target_image.crop((left, top, right, bottom))

        # Scale up if tiny
        target_height = 150
        if cropped.height < target_height:
            scale = target_height / cropped.height
            new_size = (int(cropped.width * scale), int(cropped.height * scale))
            cropped = cropped.resize(new_size, Image.Resampling.LANCZOS)

        self.tk_image = ImageTk.PhotoImage(cropped)
        self.image_panel.config(image=self.tk_image)
        
        self.entry_var.set(shape.get(PROPERTY_KEY, ''))
        self.entry.select_range(0, tk.END) 
        
        lbl = shape.get('label', '[Missing]')
        self.info_label.config(text=f"Item {self.current_index + 1}/{len(self.shapes)} | Page: {page_idx} | Label: {lbl}")

    def save_and_next(self, event=None):
        text = self.entry_var.get()
        current_shape = self.shapes[self.current_index]
        
        # Update text
        current_shape[PROPERTY_KEY] = text
        
        # Add label if missing
        if 'label' not in current_shape or not current_shape['label']:
            current_shape['label'] = DEFAULT_CLASS

        # CLEAN DATA: Round coordinates
        if 'points' in current_shape:
            cleaned_points = []
            for pt in current_shape['points']:
                cleaned_points.append([round(float(pt[0]), ROUND_DECIMALS), round(float(pt[1]), ROUND_DECIMALS)])
            current_shape['points'] = cleaned_points

        self.save_json()
        self.next_shape()

    def next_shape(self, event=None):
        """Moves to the next shape without saving."""
        if self.current_index < len(self.shapes) - 1:
            self.current_index += 1
            self.load_current_shape()
        else:
            messagebox.showinfo("Done", "You are at the last annotation!")

    def prev_shape(self, event=None):
        """Moves to the previous shape."""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_shape()
        else:
            messagebox.showinfo("Start", "You are already at the first annotation!")

    def show_context(self, event=None):
        """Opens a temporary pop-up window showing a larger area, with the target text outlined."""
        # Close existing context window if one is already open
        if hasattr(self, 'context_win') and self.context_win.winfo_exists():
            self.context_win.destroy()

        shape = self.shapes[self.current_index]
        if 'points' not in shape: return
        
        page_idx = shape.get('page_index', DEFAULT_PAGE_INDEX)
        if page_idx >= len(self.pages): 
            page_idx = len(self.pages) - 1
            
        target_image = self.pages[page_idx]
        x_min, y_min, x_max, y_max = self.get_bbox(shape['points'])
        
        # Calculate larger crop using CONTEXT_BUFFER
        w, h = target_image.size
        left = max(0, int(x_min) - CONTEXT_BUFFER)
        top = max(0, int(y_min) - CONTEXT_BUFFER)
        right = min(w, int(x_max) + CONTEXT_BUFFER)
        bottom = min(h, int(y_max) + CONTEXT_BUFFER)
        
        cropped = target_image.crop((left, top, right, bottom))
        
        # --- NEW POLYGON DRAWING LOGIC ---
        # Convert absolute polygon points to coordinates relative to this crop
        relative_points = []
        for p in shape['points']:
            # subtract the top-left coordinate of the crop
            relative_points.append((p[0] - left, p[1] - top))
        
        # Draw the polygon outline onto the cropped image
        draw = ImageDraw.Draw(cropped)
        # We use polygon to match your JSON format; 
        # use draw.rectangle if your JSON always stores points as a rect.
        draw.polygon(relative_points, outline=BOX_COLOR, width=BOX_WIDTH)
        # ----------------------------------

        # Prevent the context image from being too massive on screen
        max_size = 800
        if cropped.width > max_size or cropped.height > max_size:
            cropped.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
        # Create a new TopLevel (pop-up) window
        self.context_win = tk.Toplevel(self.root)
        self.context_win.title("Context View")
        
        tk_img = ImageTk.PhotoImage(cropped)
        lbl = tk.Label(self.context_win, image=tk_img)
        lbl.image = tk_img # Keep a reference to prevent garbage collection!
        lbl.pack(padx=10, pady=10)
        
        # Convenience: Close the pop-up instantly if the user clicks it or presses any key
        self.context_win.bind('<Key>', lambda e: self.context_win.destroy())
        lbl.bind('<Button-1>', lambda e: self.context_win.destroy())

    def save_json(self):
        with open(self.json_path, 'w') as f:
            json.dump(self.data, f, indent=2)

if __name__ == "__main__":
    # Hide the main window while we prompt for files
    root = tk.Tk()
    root.withdraw() 
    
    # 1. Prompt for the PDF/Image
    selected_pdf = filedialog.askopenfilename(
        title="Select PDF or TIF File",
        filetypes=[("Document Files", "*.pdf *.tif *.tiff"), ("All Files", "*.*")]
    )
    
    # Exit if the user clicks "Cancel"
    if not selected_pdf:
        print("No image/PDF selected. Exiting...")
        sys.exit(0)
        
    # 2. Prompt for the JSON
    selected_json = filedialog.askopenfilename(
        title="Select JSON Annotation File",
        filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
    )
    
    # Exit if the user clicks "Cancel"
    if not selected_json:
        print("No JSON selected. Exiting...")
        sys.exit(0)

    # Bring the main window back and start the app
    root.deiconify()
    root.geometry("900x500")
    
    # Pass the user-selected paths into the app
    app = LabelingApp(root, selected_pdf, selected_json, BUFFER_SIZE)
    
    # Run the application
    root.mainloop()