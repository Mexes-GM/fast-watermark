import os
import sys
import json
import random
import struct
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Windows AppUserModelID (must run BEFORE any Tk window) ──────────
if sys.platform == 'win32':
    import ctypes
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            'FastWatermark.WatermarkApp'
        )
    except Exception:
        pass

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from tkinterdnd2 import DND_FILES, TkinterDnD

from post_filters import apply_pipeline, DEFAULT_PIPELINE
from comfy_metadata import get_candidates_for_image, sanitize_for_filename


# Configuration from original script
MARGIN = 10
WATERMARK_RELATIVE_WIDTH = 0.284
WATERMARK_MIN_RELATIVE = 0.06
WATERMARK_MAX_RELATIVE = 0.50
WATERMARK_OPACITY = 0.8

class FastWatermarkApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fast Watermark & Metadata Remover")
        self.root.geometry("1180x820")
        
        # Window icon (title bar + taskbar, all resolutions)
        self._set_window_icon()
        
        # Config file path (next to .exe when frozen, next to script otherwise)
        self.config_file = os.path.join(self._app_dir(), "watermark_config.json")
        self.library_file = os.path.join(self._app_dir(), "character_library.json")
        
        self.files_to_process = []
        self.watermark_path = tk.StringVar()
        self.output_dir = tk.StringVar()  # empty = <input_dir>/watermarked_clean
        self.status_var = tk.StringVar(value="Ready")
        
        # New watermark customization variables
        self.watermark_size = tk.DoubleVar(value=WATERMARK_RELATIVE_WIDTH)
        self.watermark_opacity = tk.DoubleVar(value=WATERMARK_OPACITY)
        self.watermark_corner = tk.StringVar(value="bottom-left")
        self.randomize_corner = tk.BooleanVar(value=False)

        # ===== Post-processing filter variables =====
        self.pp_enabled = tk.BooleanVar(value=DEFAULT_PIPELINE["enabled"])
        self.pp_upscale = tk.DoubleVar(value=DEFAULT_PIPELINE["upscale"])
        self.pp_kuwahara_radius = tk.IntVar(value=DEFAULT_PIPELINE["kuwahara_radius"])
        self.pp_median_size = tk.IntVar(value=DEFAULT_PIPELINE["median_size"])
        self.pp_downscale = tk.DoubleVar(value=DEFAULT_PIPELINE["downscale"])
        self.pp_noise_strength = tk.DoubleVar(value=DEFAULT_PIPELINE["noise_strength"])
        self.pp_noise_mono = tk.BooleanVar(value=DEFAULT_PIPELINE["noise_monochromatic"])
        self.pp_noise_invert = tk.BooleanVar(value=DEFAULT_PIPELINE["noise_invert"])
        self.pp_noise_channels = tk.StringVar(value=DEFAULT_PIPELINE["noise_channels"])

        # ===== Auto-naming variables =====
        self.autoname_enabled = tk.BooleanVar(value=False)
        # Map file_path -> list of chosen characters (up to 2, filled by dialog before worker)
        self.autoname_map = {}
        
        # Cache for preview image
        self.preview_photo = None

        # Character library — learned tags from user selections
        # { "tag_lowercase": {"tag": "original casing", "count": N, "last_used": "ISO date"} }
        self.character_library = {}

        # Per-character index for auto-naming (filled at processing time)
        self.autoname_counters = defaultdict(int)
        
        # Load saved options
        self.load_options()
        self._load_character_library()
        
        # Try to find default watermark
        default_wm = os.path.join(os.path.dirname(__file__), "watermark.png")
        if os.path.exists(default_wm) and not self.watermark_path.get():
            self.watermark_path.set(default_wm)

        self.create_widgets()
        
        # Save options when closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _set_window_icon(self):
        """Load multi-resolution .ico and set window icon correctly."""
        icon_path = os.path.join(self._app_dir(), "icon.ico")
        if not os.path.exists(icon_path):
            return
        try:
            # Title bar icon (16x16)
            self.root.iconbitmap(bitmap=icon_path)
            # Taskbar/Alt+Tab icon (all resolutions via Tcl)
            photos = self._load_ico_photos(icon_path)
            if photos:
                self.root.tk.call('wm', 'iconphoto', self.root._w,
                                  '-default', *photos)
        except Exception:
            pass

    @staticmethod
    def _app_dir():
        """Directory containing the script or frozen executable."""
        if getattr(sys, 'frozen', False):
            return os.path.dirname(sys.executable)
        return os.path.dirname(os.path.abspath(__file__))

    @staticmethod
    def _load_ico_photos(ico_path):
        """Parse ICO binary and return list of PhotoImage for all sizes."""
        photos = []
        with open(ico_path, 'rb') as f:
            data = f.read()
        # ICO header: reserved(2) + type(2) + count(2)
        count = struct.unpack_from('<H', data, 4)[0]
        for i in range(count):
            off = 6 + i * 16
            w, h = data[off], data[off + 1]
            w = 256 if w == 0 else w
            h = 256 if h == 0 else h
            size = struct.unpack_from('<I', data, off + 8)[0]
            img_off = struct.unpack_from('<I', data, off + 12)[0]
            # Read BMP data (skip 40-byte BITMAPINFOHEADER)
            bmp_data = data[img_off:img_off + size]
            # The BMP inside ICO is height*2 (includes AND mask)
            bmp_h = struct.unpack_from('<i', bmp_data, 8)[0] // 2
            # Read 32-bit BGRA pixels
            pixel_data = bmp_data[40:40 + w * bmp_h * 4]
            # Convert BGRA → RGBA for PPM → PhotoImage
            ppm = b'P6\n%d %d\n255\n' % (w, h)
            rgba = bytearray(w * h * 3)
            for row in range(h):
                src_row = (bmp_h - 1 - row) * w * 4
                dst_row = (h - 1 - row) * w * 3
                for x in range(w):
                    b = pixel_data[src_row + x * 4]
                    g = pixel_data[src_row + x * 4 + 1]
                    r = pixel_data[src_row + x * 4 + 2]
                    rgba[dst_row + x * 3] = r
                    rgba[dst_row + x * 3 + 1] = g
                    rgba[dst_row + x * 3 + 2] = b
            photo = tk.PhotoImage(data=ppm + bytes(rgba))
            photos.append(photo)
        return photos

    def create_widgets(self):
        # Create main frames
        left_frame = tk.Frame(self.root)
        left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        right_frame = tk.Frame(self.root, width=460)
        right_frame.pack(side="right", fill="both", expand=False, padx=10, pady=10)
        right_frame.pack_propagate(False)

        # ===== LEFT SIDE =====
        
        # Watermark Selection
        frame_wm = tk.LabelFrame(left_frame, text="Watermark Image", padx=10, pady=10)
        frame_wm.pack(fill="x", pady=5)
        
        entry_wm = tk.Entry(frame_wm, textvariable=self.watermark_path, width=50)
        entry_wm.pack(side="left", fill="x", expand=True)
        
        # Enable Drag & Drop for Watermark
        entry_wm.drop_target_register(DND_FILES)
        entry_wm.dnd_bind('<<Drop>>', self.drop_watermark)
        
        btn_wm = tk.Button(frame_wm, text="Browse...", command=self.browse_watermark)
        btn_wm.pack(side="right", padx=5)

        # Output directory
        frame_out = tk.LabelFrame(left_frame, text="Output folder (empty = <source>/watermarked_clean)",
                                  padx=10, pady=10)
        frame_out.pack(fill="x", pady=5)
        entry_out = tk.Entry(frame_out, textvariable=self.output_dir, width=50)
        entry_out.pack(side="left", fill="x", expand=True)
        entry_out.drop_target_register(DND_FILES)
        entry_out.dnd_bind('<<Drop>>', self.drop_output_dir)
        tk.Button(frame_out, text="Browse...",
                  command=self.browse_output_dir).pack(side="right", padx=5)
        tk.Button(frame_out, text="Clear",
                  command=lambda: self.output_dir.set("")).pack(side="right")

        # Drop Zone (Listbox)
        frame_drop = tk.LabelFrame(left_frame, text="Drag & Drop Images Here", padx=10, pady=10)
        frame_drop.pack(fill="both", expand=True, pady=5)

        self.file_listbox = tk.Listbox(frame_drop, selectmode=tk.EXTENDED)
        self.file_listbox.pack(fill="both", expand=True, side="left")
        
        scrollbar = tk.Scrollbar(frame_drop, orient="vertical", command=self.file_listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.file_listbox.config(yscrollcommand=scrollbar.set)

        # Enable Drag & Drop for Listbox
        self.file_listbox.drop_target_register(DND_FILES)
        self.file_listbox.dnd_bind('<<Drop>>', self.drop_files)

        # Clear List Button
        btn_clear = tk.Button(left_frame, text="Clear List", command=self.clear_list)
        btn_clear.pack(pady=5)

        # Progress Bar
        self.progress = ttk.Progressbar(left_frame, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=10, fill="x")

        # Status Label
        lbl_status = tk.Label(left_frame, textvariable=self.status_var)
        lbl_status.pack()

        # Process Button
        btn_process = tk.Button(left_frame, text="Start Batch Processing", command=self.start_processing_thread, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
        btn_process.pack(pady=10, ipadx=20, ipady=5)

        # ===== RIGHT SIDE (tabbed) =====
        notebook = ttk.Notebook(right_frame)
        notebook.pack(fill="both", expand=True)

        wm_tab = tk.Frame(notebook)
        pp_tab = tk.Frame(notebook)
        an_tab = tk.Frame(notebook)
        notebook.add(wm_tab, text="Watermark")
        notebook.add(pp_tab, text="Post-Processing")
        notebook.add(an_tab, text="Auto-Name")

        # ---- Watermark tab ----
        # Watermark Size
        frame_size = tk.LabelFrame(wm_tab, text="Watermark Size", padx=10, pady=10)
        frame_size.pack(fill="x", pady=5)
        tk.Label(frame_size, text="Size (%):").pack(anchor="w")
        self.size_scale = tk.Scale(frame_size, from_=6, to=50, orient="horizontal",
                                    variable=self.watermark_size, command=self.update_preview)
        self.size_scale.pack(fill="x", pady=5)
        self.size_label = tk.Label(frame_size, text=f"28.4%")
        self.size_label.pack(anchor="w")

        # Watermark Opacity
        frame_opacity = tk.LabelFrame(wm_tab, text="Transparency", padx=10, pady=10)
        frame_opacity.pack(fill="x", pady=5)
        tk.Label(frame_opacity, text="Opacity:").pack(anchor="w")
        self.opacity_scale = tk.Scale(frame_opacity, from_=0, to=100, orient="horizontal",
                                       variable=self.watermark_opacity, command=self.update_preview)
        self.opacity_scale.pack(fill="x", pady=5)
        self.opacity_label = tk.Label(frame_opacity, text=f"80%")
        self.opacity_label.pack(anchor="w")

        # Watermark Position
        frame_position = tk.LabelFrame(wm_tab, text="Position", padx=10, pady=10)
        frame_position.pack(fill="x", pady=5)
        tk.Label(frame_position, text="Corner:").pack(anchor="w")
        corner_frame = tk.Frame(frame_position)
        corner_frame.pack(fill="x", pady=5)
        corners = [("Bot. Left", "bottom-left"), ("Bot. Right", "bottom-right"),
                   ("Top Left", "top-left"), ("Top Right", "top-right")]
        for label, value in corners:
            tk.Radiobutton(corner_frame, text=label, variable=self.watermark_corner,
                          value=value, command=self.update_preview).pack(anchor="w")
        tk.Checkbutton(frame_position, text="Randomize corner",
                      variable=self.randomize_corner, command=self.update_preview).pack(anchor="w", pady=5)

        # Preview
        frame_preview = tk.LabelFrame(wm_tab, text="Preview", padx=10, pady=10)
        frame_preview.pack(fill="both", expand=True, pady=5)
        self.preview_canvas = tk.Canvas(frame_preview, bg="#f0f0f0", highlightthickness=1, highlightbackground="#cccccc")
        self.preview_canvas.pack(fill="both", expand=True)
        self.preview_canvas.bind("<Configure>", self.on_preview_resize)

        # ---- Post-Processing tab ----
        self._build_postprocessing_tab(pp_tab)
        # ---- Auto-Naming tab ----
        self._build_autoname_tab(an_tab)
        
        # Add traces to save options when modified
        for v in (self.watermark_path, self.output_dir,
                  self.watermark_size, self.watermark_opacity,
                  self.watermark_corner, self.randomize_corner,
                  self.pp_enabled, self.pp_upscale, self.pp_kuwahara_radius,
                  self.pp_median_size, self.pp_downscale, self.pp_noise_strength,
                  self.pp_noise_mono, self.pp_noise_invert, self.pp_noise_channels,
                  self.autoname_enabled):
            v.trace_add('write', lambda *args: self.on_option_changed())

    def load_options(self):
        """Load saved options from config file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    
                # Restore saved options
                if "watermark_path" in config:
                    self.watermark_path.set(config["watermark_path"])
                if "watermark_size" in config:
                    self.watermark_size.set(config["watermark_size"])
                if "watermark_opacity" in config:
                    self.watermark_opacity.set(config["watermark_opacity"])
                if "watermark_corner" in config:
                    self.watermark_corner.set(config["watermark_corner"])
                if "randomize_corner" in config:
                    self.randomize_corner.set(config["randomize_corner"])
                if "output_dir" in config:
                    self.output_dir.set(config["output_dir"])

                # Post-processing
                pp = config.get("post_processing", {}) or {}
                if "enabled" in pp: self.pp_enabled.set(pp["enabled"])
                if "upscale" in pp: self.pp_upscale.set(pp["upscale"])
                if "kuwahara_radius" in pp: self.pp_kuwahara_radius.set(pp["kuwahara_radius"])
                if "median_size" in pp: self.pp_median_size.set(pp["median_size"])
                if "downscale" in pp: self.pp_downscale.set(pp["downscale"])
                if "noise_strength" in pp: self.pp_noise_strength.set(pp["noise_strength"])
                if "noise_monochromatic" in pp: self.pp_noise_mono.set(pp["noise_monochromatic"])
                if "noise_invert" in pp: self.pp_noise_invert.set(pp["noise_invert"])
                if "noise_channels" in pp: self.pp_noise_channels.set(pp["noise_channels"])
                # Auto-name
                if "autoname_enabled" in config:
                    self.autoname_enabled.set(config["autoname_enabled"])

                print(f"Options loaded from {self.config_file}")
            except Exception as e:
                print(f"Error loading options: {e}")

    def _load_character_library(self):
        """Load learned character tags from JSON file."""
        if os.path.exists(self.library_file):
            try:
                with open(self.library_file, 'r') as f:
                    data = json.load(f)
                self.character_library = data.get("characters", {})
                print(f"Character library loaded: {len(self.character_library)} tags")
            except Exception as e:
                print(f"Error loading character library: {e}")
                self.character_library = {}

    def _save_character_library(self):
        """Persist learned character tags to JSON file."""
        try:
            data = {"characters": self.character_library}
            with open(self.library_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Character library saved: {len(self.character_library)} tags")
        except Exception as e:
            print(f"Error saving character library: {e}")

    def save_options(self):
        """Save current options to config file."""
        try:
            config = {
                "watermark_path": self.watermark_path.get(),
                "watermark_size": self.watermark_size.get(),
                "watermark_opacity": self.watermark_opacity.get(),
                "watermark_corner": self.watermark_corner.get(),
                "randomize_corner": self.randomize_corner.get(),
                "output_dir": self.output_dir.get(),
                "autoname_enabled": self.autoname_enabled.get(),
                "post_processing": {
                    "enabled": self.pp_enabled.get(),
                    "upscale": self.pp_upscale.get(),
                    "kuwahara_radius": self.pp_kuwahara_radius.get(),
                    "median_size": self.pp_median_size.get(),
                    "downscale": self.pp_downscale.get(),
                    "noise_strength": self.pp_noise_strength.get(),
                    "noise_monochromatic": self.pp_noise_mono.get(),
                    "noise_invert": self.pp_noise_invert.get(),
                    "noise_channels": self.pp_noise_channels.get(),
                },
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
                
            print(f"Options saved to {self.config_file}")
        except Exception as e:
            print(f"Error saving options: {e}")

    def on_closing(self):
        """Save options and close application."""
        self.save_options()
        self.root.destroy()

    def on_option_changed(self):
        """Called when any option changes."""
        self.update_preview()
        self.save_options()

    def drop_files(self, event):
        # Parse the dropped data using Tkinter's splitlist to handle spaces and braces
        files = self.root.tk.splitlist(event.data)
        for f in files:
            if os.path.isfile(f):
                ext = os.path.splitext(f)[1].lower()
                if ext in {'.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov', '.mkv'}:
                    if f not in self.files_to_process:
                        self.files_to_process.append(f)
                        self.file_listbox.insert(tk.END, f)
            elif os.path.isdir(f):
                for root_dir, _, filenames in os.walk(f):
                    for filename in filenames:
                        if os.path.splitext(filename)[1].lower() in {'.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov', '.mkv'}:
                            full_path = os.path.join(root_dir, filename)
                            if full_path not in self.files_to_process:
                                self.files_to_process.append(full_path)
                                self.file_listbox.insert(tk.END, full_path)


    def drop_watermark(self, event):
        path = self.clean_path(event.data)
        if os.path.isfile(path):
            self.watermark_path.set(path)

    def update_preview(self, *args):
        """Update real-time preview."""
        wm_path = self.watermark_path.get()
        
        if not wm_path or not os.path.exists(wm_path):
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(
                self.preview_canvas.winfo_width() // 2,
                self.preview_canvas.winfo_height() // 2,
                text="Loading watermark...",
                fill="#666666"
            )
            return
        
        try:
            # Get canvas dimensions
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()
            
            # Use minimum dimensions if not yet allocated
            if canvas_width <= 1:
                canvas_width = 300
            if canvas_height <= 1:
                canvas_height = 300
            
            # Load watermark
            watermark = Image.open(wm_path).convert("RGBA")
            
            # Create preview image scaled to canvas size with some padding
            padding = 20
            preview_width = canvas_width - padding
            preview_height = canvas_height - padding
            
            if preview_width > 0 and preview_height > 0:
                preview_img = Image.new("RGBA", (preview_width, preview_height), (200, 200, 200, 255))
                
                # Calculate watermark size based on slider
                wm_width, wm_height = watermark.size
                size_percent = self.watermark_size.get()
                target_w = int(preview_width * (size_percent / 100))
                
                if target_w > 0 and wm_width > 0:
                    scale = target_w / wm_width
                    
                    wm_resized = watermark.resize((int(wm_width * scale), int(wm_height * scale)), resample=Image.LANCZOS)
                    
                    # Apply opacity
                    opacity_percent = self.watermark_opacity.get()
                    if opacity_percent < 100:
                        wm_resized = wm_resized.copy()
                        alpha = wm_resized.split()[3].point(lambda p: int(p * (opacity_percent / 100)))
                        wm_resized.putalpha(alpha)
                    
                    # Determine corner position
                    corner = self.watermark_corner.get()
                    if self.randomize_corner.get():
                        # For preview, choose a random corner locally without updating the variable
                        corner = random.choice(["bottom-left", "bottom-right", "top-left", "top-right"])
                    
                    position = self.get_watermark_position(preview_img, wm_resized, corner)
                    
                    # Paste watermark
                    preview_img.paste(wm_resized, position, wm_resized)
                
                # Convert to PhotoImage and display on canvas
                self.preview_photo = ImageTk.PhotoImage(preview_img)
                self.preview_canvas.delete("all")
                
                # Center image on canvas
                canvas_id = self.preview_canvas.create_image(
                    padding // 2, padding // 2,
                    image=self.preview_photo,
                    anchor="nw"
                )
                
                # Update labels
                self.size_label.config(text=f"{size_percent:.1f}%")
                self.opacity_label.config(text=f"{opacity_percent:.0f}%")
            
        except Exception as e:
            self.preview_canvas.delete("all")
            self.preview_canvas.create_text(
                self.preview_canvas.winfo_width() // 2,
                self.preview_canvas.winfo_height() // 2,
                text=f"Error: {str(e)}",
                fill="#ff0000"
            )

    def on_preview_resize(self, event):
        """Fires when preview canvas resizes."""
        self.update_preview()

    def get_watermark_position(self, base_img, watermark_img, corner):
        """Calculate watermark position based on selected corner."""
        base_w, base_h = base_img.size
        wm_w, wm_h = watermark_img.size
        margin = 10
        
        if corner == "bottom-left":
            x = margin
            y = base_h - wm_h - margin
        elif corner == "bottom-right":
            x = base_w - wm_w - margin
            y = base_h - wm_h - margin
        elif corner == "top-left":
            x = margin
            y = margin
        elif corner == "top-right":
            x = base_w - wm_w - margin
            y = margin
        else:  # default to bottom-left
            x = margin
            y = base_h - wm_h - margin
        
        return (max(0, x), max(0, y))

    def clean_path(self, path):
        path = path.strip()
        if path.startswith('{') and path.endswith('}'):
            path = path[1:-1]
        return path

    def clear_list(self):
        self.files_to_process = []
        self.file_listbox.delete(0, tk.END)

    def browse_watermark(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.watermark_path.set(file_path)

    def browse_output_dir(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.output_dir.set(d)

    def drop_output_dir(self, event):
        path = self.clean_path(event.data)
        if os.path.isdir(path):
            self.output_dir.set(path)

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------
    def _build_postprocessing_tab(self, parent):
        tk.Checkbutton(parent, text="Enable Post-Processing",
                       variable=self.pp_enabled).pack(anchor="w", padx=10, pady=8)

        # Upscale
        f = tk.LabelFrame(parent, text="1) Resize Relative (Upscale)", padx=10, pady=6)
        f.pack(fill="x", padx=10, pady=4)
        tk.Label(f, text="Scale (W=H):").pack(anchor="w")
        tk.Scale(f, from_=1.0, to=4.0, resolution=0.1, orient="horizontal",
                 variable=self.pp_upscale).pack(fill="x")

        # Kuwahara
        f = tk.LabelFrame(parent, text="2) Kuwahara Blur (mean)", padx=10, pady=6)
        f.pack(fill="x", padx=10, pady=4)
        tk.Label(f, text="Radius (0 = disabled):").pack(anchor="w")
        tk.Scale(f, from_=0, to=8, orient="horizontal",
                 variable=self.pp_kuwahara_radius).pack(fill="x")

        # Median
        f = tk.LabelFrame(parent, text="3) Median Filter", padx=10, pady=6)
        f.pack(fill="x", padx=10, pady=4)
        tk.Label(f, text="Size (kernel = 2*size+1, 0 = disabled):").pack(anchor="w")
        tk.Scale(f, from_=0, to=5, orient="horizontal",
                 variable=self.pp_median_size).pack(fill="x")

        # Downscale
        f = tk.LabelFrame(parent, text="4) Resize Relative (Downscale)", padx=10, pady=6)
        f.pack(fill="x", padx=10, pady=4)
        tk.Label(f, text="Scale (W=H):").pack(anchor="w")
        tk.Scale(f, from_=0.1, to=2.0, resolution=0.05, orient="horizontal",
                 variable=self.pp_downscale).pack(fill="x")

        # Noise
        f = tk.LabelFrame(parent, text="5) Gaussian Noise", padx=10, pady=6)
        f.pack(fill="x", padx=10, pady=4)
        tk.Label(f, text="Strength (0 = disabled):").pack(anchor="w")
        tk.Scale(f, from_=0.0, to=0.5, resolution=0.01, orient="horizontal",
                 variable=self.pp_noise_strength).pack(fill="x")
        tk.Checkbutton(f, text="Monochromatic", variable=self.pp_noise_mono).pack(anchor="w")
        tk.Checkbutton(f, text="Invert", variable=self.pp_noise_invert).pack(anchor="w")
        ch_frame = tk.Frame(f); ch_frame.pack(anchor="w")
        tk.Label(ch_frame, text="Channels:").pack(side="left")
        ttk.Combobox(ch_frame, textvariable=self.pp_noise_channels, width=6,
                     values=("rgb", "r", "g", "b", "rg", "rb", "gb"),
                     state="readonly").pack(side="left", padx=4)

        tk.Button(parent, text="Reset to defaults",
                  command=self._reset_postprocessing).pack(pady=8)

    def _reset_postprocessing(self):
        d = DEFAULT_PIPELINE
        self.pp_enabled.set(d["enabled"])
        self.pp_upscale.set(d["upscale"])
        self.pp_kuwahara_radius.set(d["kuwahara_radius"])
        self.pp_median_size.set(d["median_size"])
        self.pp_downscale.set(d["downscale"])
        self.pp_noise_strength.set(d["noise_strength"])
        self.pp_noise_mono.set(d["noise_monochromatic"])
        self.pp_noise_invert.set(d["noise_invert"])
        self.pp_noise_channels.set(d["noise_channels"])

    def _build_autoname_tab(self, parent):
        tk.Checkbutton(parent, text="Enable metadata auto-naming",
                       variable=self.autoname_enabled).pack(anchor="w", padx=10, pady=8)
        info = (
            "When enabled, before processing PNG images with metadata,\n"
            "a dialog will show candidate tags extracted from the\n"
            "positive prompt (tags that appear before '1girl' / '1boy',\n"
            "after LoRA triggers).\n\n"
            "Check all character tags that apply, or type your own\n"
            "(comma-separated), or press 'Skip' to keep the\n"
            "original filename.\n\n"
            "The original name is REPLACED with:\n"
            "  <char1>_<N>.png\n"
            "  <char1>+<char2>+<char3>_<N>.png\n"
            "Characters are joined with '+' for easy automated parsing.\n"
            "<N> is a per-combination counter within the batch."
        )
        tk.Label(parent, text=info, justify="left", anchor="w",
                 wraplength=420).pack(fill="x", padx=10, pady=6)

        # Library status + manage button
        lib_frame = tk.Frame(parent)
        lib_frame.pack(fill="x", padx=10, pady=6)
        self.lib_status_lbl = tk.Label(lib_frame, text="", fg="#555")
        self.lib_status_lbl.pack(side="left")
        tk.Button(lib_frame, text="Manage Library",
                  command=self._manage_library).pack(side="right")
        self._update_library_status()

    def _update_library_status(self):
        n = len(self.character_library)
        if n == 0:
            self.lib_status_lbl.config(
                text="📚 No learned characters yet. Selected tags will be auto-learned.")
        else:
            top = sorted(self.character_library.values(),
                         key=lambda x: -x["count"])[:3]
            names = ", ".join(e["tag"] for e in top)
            self.lib_status_lbl.config(
                text=f"📚 {n} learned: {names}...")

    def _manage_library(self):
        """Dialog to view and remove learned character tags."""
        dlg = tk.Toplevel(self.root)
        dlg.title("Character Library")
        dlg.geometry("500x520")
        dlg.transient(self.root)
        dlg.grab_set()

        tk.Label(dlg, text="Learned character tags (★ = auto-selected in future)",
                 font=("Arial", 10, "bold")).pack(pady=10)

        list_frame = tk.Frame(dlg)
        list_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Sort by count desc
        entries = sorted(self.character_library.items(),
                         key=lambda kv: -kv[1]["count"])

        if not entries:
            tk.Label(list_frame, text="No characters learned yet. Select tags during\n"
                     "auto-naming and they'll appear here.").pack(pady=20)
        else:
            canvas = tk.Canvas(list_frame, highlightthickness=0)
            scrollbar = tk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
            scroll_frame = tk.Frame(canvas)

            scroll_frame.bind("<Configure>",
                              lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            for key, entry in entries:
                row = tk.Frame(scroll_frame)
                row.pack(fill="x", pady=2)
                tk.Label(row, text=f"★ {entry['tag']}",
                         font=("Arial", 10, "bold"), fg="#2e7d32",
                         width=30, anchor="w").pack(side="left")
                tk.Label(row, text=f"×{entry['count']}",
                         fg="#888", width=6).pack(side="left")
                btn = tk.Button(row, text="✕", fg="#c0392b",
                                command=lambda k=key: self._remove_from_library(k, dlg))
                btn.pack(side="right", padx=2)

        # Buttons
        btns = tk.Frame(dlg)
        btns.pack(fill="x", padx=10, pady=10)
        tk.Button(btns, text="Clear All",
                  command=lambda: self._clear_library(dlg),
                  fg="#c0392b").pack(side="left")
        tk.Button(btns, text="Close", command=dlg.destroy,
                  width=10).pack(side="right")

    def _remove_from_library(self, key, dialog):
        """Remove a single tag from the library."""
        if key in self.character_library:
            del self.character_library[key]
            self._save_character_library()
            self._update_library_status()
            dialog.destroy()
            self._manage_library()  # reopen to refresh

    def _clear_library(self, dialog):
        """Clear all learned tags after confirmation."""
        if messagebox.askyesno("Clear Library",
                               "Remove ALL learned character tags?\nThis cannot be undone.",
                               parent=dialog):
            self.character_library.clear()
            self._save_character_library()
            self._update_library_status()
            dialog.destroy()
            self._manage_library()  # reopen to refresh

    # ------------------------------------------------------------------
    # Auto-name dialog flow
    # ------------------------------------------------------------------
    def _prompt_character_for_image(self, image_path, candidates, prompt_text):
        """Modal dialog: returns list of chosen character names (multi-select),
        empty list to skip, or None to cancel batch."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Select characters")
        dialog.geometry("720x560")
        dialog.transient(self.root)
        dialog.grab_set()

        result = {"value": []}
        cancel_all = {"value": False}

        # Thumbnail + filename
        top = tk.Frame(dialog); top.pack(fill="x", padx=10, pady=10)
        try:
            with Image.open(image_path) as im:
                im.thumbnail((240, 240))
                photo = ImageTk.PhotoImage(im.copy())
            lbl_img = tk.Label(top, image=photo)
            lbl_img.image = photo
            lbl_img.pack(side="left")
        except Exception:
            tk.Label(top, text="(no preview)").pack(side="left")
        tk.Label(top, text=os.path.basename(image_path), font=("Arial", 11, "bold"),
                 wraplength=420, justify="left").pack(side="left", padx=10)

        # Candidates (checkboxes, unlimited)
        f_cand = tk.LabelFrame(dialog, text="Candidate tags — check all that apply",
                                padx=10, pady=8)
        f_cand.pack(fill="x", padx=10, pady=4)
        check_vars = []  # list of (tk.BooleanVar, tag_string)
        count_lbl = tk.Label(f_cand, text="", fg="#666")
        count_lbl.pack(anchor="w")

        def _update_count():
            n = sum(1 for v, t in check_vars if v.get())
            if n == 0:
                count_lbl.config(text="")
            elif n == 1:
                count_lbl.config(text="✓ 1 character selected")
            else:
                count_lbl.config(text=f"✓ {n} characters selected")

        if candidates:
            # ── Sort by library confidence: known tags first, then by usage count ──
            def _sort_key(tag):
                lib_entry = self.character_library.get(tag.lower())
                if lib_entry:
                    # Known character: sort by count desc (higher = more confident)
                    return (0, -lib_entry.get("count", 0), -len(tag))
                # Unknown: after all known tags, then by length
                return (1, 0, -len(tag))

            sorted_candidates = sorted(candidates, key=_sort_key)

            # Auto-select: pre-check tags that match the library AND are the
            # clear top pick (count >= 2, and at least 2x the next known tag).
            auto_select = set()
            known = [(c, self.character_library.get(c.lower())) for c in sorted_candidates
                     if self.character_library.get(c.lower())]
            if known:
                best_tag, best_entry = known[0]
                best_count = best_entry.get("count", 0)
                if best_count >= 1:
                    second_count = known[1][1].get("count", 0) if len(known) > 1 else 0
                    if best_count >= second_count * 2:
                        auto_select.add(best_tag.lower())

            for c in sorted_candidates:
                lib = self.character_library.get(c.lower())
                label = f"★ {c}  (×{lib['count']})" if lib else c
                prechecked = c.lower() in auto_select
                var = tk.BooleanVar(value=prechecked)
                check_vars.append((var, c))
                cb = tk.Checkbutton(f_cand, text=label, variable=var, anchor="w",
                                    command=_update_count)
                if lib:
                    cb.config(fg="#2e7d32", font=("Arial", 9, "bold"))
                cb.pack(fill="x", anchor="w")
            _update_count()  # show count for auto-selected items
        else:
            tk.Label(f_cand, text="(no candidates detected)").pack()

        # Custom entry
        f_cust = tk.LabelFrame(dialog, text="Or type custom (comma-separated)",
                               padx=10, pady=6)
        f_cust.pack(fill="x", padx=10, pady=4)
        custom_var = tk.StringVar()
        tk.Entry(f_cust, textvariable=custom_var).pack(fill="x")

        # Buttons
        btns = tk.Frame(dialog)
        btns.pack(side="bottom", fill="x", padx=10, pady=10)
        def on_apply():
            custom_text = custom_var.get().strip()
            if custom_text:
                result["value"] = [p.strip() for p in custom_text.split(",") if p.strip()]
            else:
                result["value"] = [t for v, t in check_vars if v.get()]
            dialog.destroy()
        def on_skip():
            result["value"] = []
            dialog.destroy()
        def on_cancel_all():
            cancel_all["value"] = True
            dialog.destroy()
        tk.Button(btns, text="Apply", command=on_apply, width=14,
                  bg="#4CAF50", fg="white").pack(side="right", padx=4)
        tk.Button(btns, text="Skip", command=on_skip, width=10).pack(side="right", padx=4)
        tk.Button(btns, text="Cancel batch", command=on_cancel_all,
                  width=14).pack(side="left", padx=4)

        # Prompt preview
        f_pp = tk.LabelFrame(dialog, text="Positive prompt (excerpt)", padx=8, pady=6)
        f_pp.pack(fill="both", expand=True, padx=10, pady=4)
        txt = tk.Text(f_pp, height=6, wrap="word")
        txt.insert("1.0", (prompt_text or "")[:1500])
        txt.config(state="disabled")
        txt.pack(fill="both", expand=True)

        dialog.wait_window()
        if cancel_all["value"]:
            return None
        return result["value"]

    def _collect_autoname_choices(self):
        """Show dialog for every image; populate self.autoname_map.
        Returns False if user cancelled the batch.
        
        Shows dialog for ALL files (not just PNGs with ComfyUI metadata)
        so the user can always type custom character names manually,
        even for images/videos without embedded prompt data.
        """
        self.autoname_map = {}
        self.autoname_counters = defaultdict(int)
        if not self.autoname_enabled.get():
            return True

        library_changed = False
        for path in list(self.files_to_process):
            prompt, candidates = get_candidates_for_image(path)
            choice = self._prompt_character_for_image(path, candidates, prompt)
            if choice is None:
                return False  # user cancelled batch
            if choice:  # non-empty list
                self.autoname_map[path] = choice
                # ── Learn: add selected tags to character library ──
                for tag in choice:
                    key = tag.lower()
                    entry = self.character_library.get(key, {"tag": tag, "count": 0})
                    entry["count"] = entry.get("count", 0) + 1
                    if entry["tag"] != tag and len(tag) > len(entry["tag"]):
                        # Prefer longer/original casing
                        entry["tag"] = tag
                    self.character_library[key] = entry
                    library_changed = True

        if library_changed:
            self._save_character_library()
        return True

    # ------------------------------------------------------------------
    def start_processing_thread(self):
        if not self.files_to_process:
            messagebox.showerror("Error", "Please drag and drop images to process.")
            return
        if not self.watermark_path.get():
            messagebox.showerror("Error", "Please select a watermark image.")
            return

        # Run auto-name dialogs synchronously in the main thread first.
        if not self._collect_autoname_choices():
            self.status_var.set("Cancelled by user.")
            return

        threading.Thread(target=self.process_images, daemon=True).start()

    def process_images(self):
        wm_path = self.watermark_path.get()
        custom_out = self.output_dir.get().strip()

        try:
            watermark = Image.open(wm_path).convert("RGBA")
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load watermark: {e}"))
            return

        # ── Snapshot config so background threads read consistent values ──
        cfg = {
            "watermark_size": self.watermark_size.get(),
            "watermark_opacity": self.watermark_opacity.get(),
            "watermark_corner": self.watermark_corner.get(),
            "randomize_corner": self.randomize_corner.get(),
            "pp_cfg": self._current_pp_config(),
            "custom_out": custom_out,
        }

        files = self.files_to_process
        total_files = len(files)
        counters_lock = threading.Lock()

        self.root.after(0, lambda: self.status_var.set(f"Processing 0/{total_files}..."))
        self.root.after(0, lambda: self.progress.configure(maximum=total_files, value=0))

        processed = 0
        max_workers = min(os.cpu_count() or 4, 8)  # I/O-bound, 4-8 threads ideal

        # Pre-create output dirs to avoid race conditions
        for file_path in files:
            out_dir = custom_out or os.path.join(os.path.dirname(file_path),
                                                  "watermarked_clean")
            os.makedirs(out_dir, exist_ok=True)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for file_path in files:
                fut = executor.submit(
                    self._process_single_image,
                    file_path, watermark, cfg, counters_lock,
                )
                futures[fut] = file_path

            for fut in as_completed(futures):
                file_path = futures[fut]
                try:
                    fut.result()
                    processed += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

                # Update progress from background thread (root.after is thread-safe)
                self.root.after(0, lambda v=processed: self.progress.configure(value=v))
                self.root.after(0, lambda v=processed: self.status_var.set(
                    f"Processing {v}/{total_files}..."))

        self.root.after(0, lambda: self.status_var.set("Completed!"))
        self.root.after(0, lambda: messagebox.showinfo("Success", f"Processed {processed} files."))

    def _process_single_image(self, file_path, watermark, cfg, counters_lock):
        """Process one image. Called from worker threads — no GUI access."""
        out_dir = cfg["custom_out"] or os.path.join(os.path.dirname(file_path),
                                                     "watermarked_clean")

        if os.path.splitext(file_path)[1].lower() in {'.mp4', '.avi', '.mov', '.mkv'}:
            self._overlay_watermark_video_worker(file_path, watermark, out_dir, cfg)
        else:
            self._overlay_watermark_worker(file_path, watermark, out_dir, cfg, counters_lock)

    # ------------------------------------------------------------------
    # Worker methods (called from thread pool — no tkinter variable access)
    # ------------------------------------------------------------------
    def _overlay_watermark_video_worker(self, video_path, watermark, save_folder, cfg):
        """Thread-safe video watermarking. Reads all params from cfg dict."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        filename = os.path.basename(video_path)
        save_path = os.path.join(save_folder, filename)

        out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

        wm_width, wm_height = watermark.size
        ref_dim = min(width, height)
        target_w = int(ref_dim * (cfg["watermark_size"] / 100))
        min_w = int(ref_dim * WATERMARK_MIN_RELATIVE)
        max_w = int(ref_dim * WATERMARK_MAX_RELATIVE)
        target_w = max(min_w, min(max_w, target_w))
        scale = target_w / wm_width

        wm_resized = watermark.resize(
            (int(wm_width * scale), int(wm_height * scale)), resample=Image.LANCZOS)

        wm_np = np.array(wm_resized)
        wm_bgr = wm_np[:, :, :3][:, :, ::-1]
        wm_mask = wm_np[:, :, 3]

        wm_h, wm_w = wm_bgr.shape[:2]

        corner = cfg["watermark_corner"]
        if cfg["randomize_corner"]:
            corner = random.choice(["bottom-left", "bottom-right", "top-left", "top-right"])

        x, y = self.get_watermark_position_video(width, height, wm_w, wm_h, corner)

        wm_bgr_f = wm_bgr.astype(float)
        wm_mask_f = (wm_mask.astype(float) / 255.0) * (cfg["watermark_opacity"] / 100)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame.shape[0] < y + wm_h or frame.shape[1] < x + wm_w:
                out.write(frame)
                continue

            roi = frame[y:y+wm_h, x:x+wm_w].astype(float)
            for c in range(3):
                roi[:, :, c] = wm_bgr_f[:, :, c] * wm_mask_f + roi[:, :, c] * (1 - wm_mask_f)
            frame[y:y+wm_h, x:x+wm_w] = roi.astype(np.uint8)
            out.write(frame)

        cap.release()
        out.release()

    def get_watermark_position_video(self, video_w, video_h, wm_w, wm_h, corner):
        """Calculate video watermark position based on selected corner."""
        margin = MARGIN
        
        if corner == "bottom-left":
            x = margin
            y = video_h - wm_h - margin
        elif corner == "bottom-right":
            x = video_w - wm_w - margin
            y = video_h - wm_h - margin
        elif corner == "top-left":
            x = margin
            y = margin
        elif corner == "top-right":
            x = video_w - wm_w - margin
            y = margin
        else:  # default to bottom-left
            x = margin
            y = video_h - wm_h - margin
        
        return (max(0, x), max(0, y))

    def _current_pp_config(self):
        return {
            "enabled": self.pp_enabled.get(),
            "upscale": self.pp_upscale.get(),
            "upscale_method": "lanczos",
            "kuwahara_radius": self.pp_kuwahara_radius.get(),
            "kuwahara_method": "mean",
            "median_size": self.pp_median_size.get(),
            "downscale": self.pp_downscale.get(),
            "downscale_method": "lanczos",
            "noise_strength": self.pp_noise_strength.get(),
            "noise_monochromatic": self.pp_noise_mono.get(),
            "noise_invert": self.pp_noise_invert.get(),
            "noise_channels": self.pp_noise_channels.get(),
        }

    def _build_output_filename(self, image_path, counters_lock=None):
        """
        Auto-naming: thread-safe version. Pass counters_lock when called
        from worker threads.
        """
        original = os.path.basename(image_path)
        characters = self.autoname_map.get(image_path)
        if not characters:
            return original
        ext = os.path.splitext(original)[1]
        safe_names = [sanitize_for_filename(c) for c in characters]
        joined = "+".join(safe_names)
        if counters_lock:
            with counters_lock:
                self.autoname_counters[joined] += 1
                n = self.autoname_counters[joined]
        else:
            self.autoname_counters[joined] += 1
            n = self.autoname_counters[joined]
        return f"{joined}_{n}{ext}"

    def _overlay_watermark_worker(self, image_path, watermark, save_folder, cfg, counters_lock):
        """Thread-safe image watermarking. Reads all params from cfg dict."""
        with Image.open(image_path) as im:
            pp_cfg = cfg["pp_cfg"]
            if pp_cfg["enabled"]:
                if im.mode not in ("RGB", "RGBA"):
                    im_proc = im.convert("RGB")
                else:
                    im_proc = im.copy()
                im_proc = apply_pipeline(im_proc, pp_cfg)
            else:
                im_proc = im.copy()

            im_width, im_height = im_proc.size
            wm_width, wm_height = watermark.size

            ref_dim = min(im_width, im_height)
            target_w = int(ref_dim * (cfg["watermark_size"] / 100))
            min_w = int(ref_dim * WATERMARK_MIN_RELATIVE)
            max_w = int(ref_dim * WATERMARK_MAX_RELATIVE)
            target_w = max(min_w, min(max_w, target_w))
            scale_factor = target_w / wm_width

            wm_resized = watermark.resize(
                (int(wm_width * scale_factor), int(wm_height * scale_factor)),
                resample=Image.LANCZOS)

            if cfg["watermark_opacity"] < 100:
                wm_resized = wm_resized.copy()
                alpha = wm_resized.split()[3].point(
                    lambda p: int(p * (cfg["watermark_opacity"] / 100)))
                wm_resized.putalpha(alpha)

            if im_proc.mode != 'RGBA':
                base = im_proc.convert("RGBA")
            else:
                base = im_proc

            layer = Image.new("RGBA", base.size, (0, 0, 0, 0))

            corner = cfg["watermark_corner"]
            if cfg["randomize_corner"]:
                corner = random.choice(["bottom-left", "bottom-right", "top-left", "top-right"])

            position = self.get_watermark_position(base, wm_resized, corner)
            layer.paste(wm_resized, position, wm_resized)
            result = Image.alpha_composite(base, layer)

            filename = self._build_output_filename(image_path, counters_lock)
            ext = os.path.splitext(filename)[1].lower()
            save_path = os.path.join(save_folder, filename)

            # Safety: never overwrite the original input file.
            if os.path.normpath(save_path) == os.path.normpath(image_path):
                base_name = os.path.splitext(filename)[0]
                save_path = os.path.join(save_folder, f"{base_name}_wm{ext}")
                print(f"WARNING: output would overwrite input. "
                      f"Saved as: {os.path.basename(save_path)}")

            # Save WITHOUT metadata — PIL doesn't embed EXIF/pnginfo unless
            # explicitly passed, so no strip_metadata() needed.
            if ext in ['.jpg', '.jpeg']:
                rgb = result.convert("RGB")
                rgb.save(save_path, "JPEG", quality=95, optimize=True)
            else:
                result.save(save_path, "PNG", optimize=True)

if __name__ == "__main__":
    root = TkinterDnD.Tk()
    app = FastWatermarkApp(root)
    root.mainloop()
