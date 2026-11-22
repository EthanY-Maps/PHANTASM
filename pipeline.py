"""Main pipeline orchestrator — drives execution of all stages for one or many maps."""

import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import json
import os

# Historic maps frequently exceed PIL's default decompression-bomb limit.
try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
except ImportError:
    pass

from config import Config
from utils import (
    setup_output_directories, get_map_name_from_input, convert_pdf_to_png,
    collect_image_files, deep_merge_missing, load_json, save_json,
)
from model_parameters import ModelParameters, get_parameters, load_parameters, DEFAULT_PARAMS_FILE
from stages import STAGES
import modules


class Pipeline:
    def __init__(self, config: Dict[str, Any], callbacks: Optional[Dict[str, Callable]] = None,
                 params: Optional[ModelParameters] = None, params_file: Optional[str] = None):
        self.config = config
        self.callbacks = callbacks or {}
        self.log = self.callbacks.get('log', print)
        self.update_status = self.callbacks.get('update_status', lambda x: None)
        self.show_image = self.callbacks.get('show_image', lambda x: None)

        if params is not None:
            self.params = params
        elif params_file:
            self.params = load_parameters(params_file)
        else:
            try:
                self.params = load_parameters(DEFAULT_PARAMS_FILE)
                self.log(f"[INFO] Loaded parameters from {DEFAULT_PARAMS_FILE}")
            except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
                self.params = get_parameters()
                self.log(f"[INFO] Using default parameters ({e.__class__.__name__})")
        
    def run(self) -> None:
        """Run the mapping pipeline for one file, a flat folder, or a folder of folders."""
        pipeline_start_time = time.time()

        input_path = self.config['entries']['input']
        image_files = collect_image_files(input_path)

        if not image_files:
            raise ValueError(f"No supported image files found at: {input_path}")

        is_batch = Path(input_path).is_dir()
        total_files = len(image_files)
        self.log(f"[PIPELINE] Found {total_files} image file(s) to process.")

        try:
            for i, img_file in enumerate(image_files):
                if is_batch:
                    self.log(f"[PIPELINE] ── {i+1}/{total_files}: {Path(img_file).name}")
                self._run_single(img_file, is_batch)

            total_time = time.time() - pipeline_start_time
            self.log(f"[SUCCESS] Pipeline completed in {total_time:.2f}s "
                     f"({total_files} file(s))")
            self.update_status(f"Pipeline completed in {total_time:.2f}s")

        except Exception as e:
            total_time = time.time() - pipeline_start_time
            self.log(f"[ERROR] Pipeline failed after {total_time:.2f}s: {e}")
            self.update_status(f"Pipeline failed after {total_time:.2f}s")
            raise

    def _run_single(self, input_file: str, is_batch: bool) -> None:
        """Run all pipeline stages for one image file."""
        start = time.time()
        original_input = self.config['entries']['input']
        self.config['entries']['input'] = input_file

        try:
            if input_file.lower().endswith('.pdf'):
                dpi = self.params.preprocessing.pdf_dpi
                self.log(f"[INFO] Converting PDF to PNG at {dpi} DPI...")
                map_name = get_map_name_from_input(input_file)
                # Place the converted PNG next to the PDF so future runs reuse it.
                png_path = os.path.join(os.path.dirname(input_file), map_name + ".png")
                convert_pdf_to_png(input_file, png_path, dpi=dpi, page_num=0)
                input_file = png_path
                self.config['entries']['input'] = input_file
                self.log(f"[INFO] PDF converted to {png_path}")

            dirs = self._setup_directories(input_file, is_batch)
            self._create_local_config(dirs)

            for stage in STAGES:
                self._run_stage(stage, dirs)

            self.log(f"[INFO] {Path(input_file).name} done in {time.time()-start:.2f}s")

        finally:
            self.config['entries']['input'] = original_input

    def _setup_directories(self, input_file: str = None, is_batch: bool = False) -> Dict[str, Path]:
        """Setup output directories, optionally namespaced by map name in batch mode."""
        entries = self.config['entries']
        output_dir = entries['output']
        expt_name = entries['expt']

        if is_batch and input_file:
            map_name = get_map_name_from_input(input_file)
            sub = str(Path(expt_name) / map_name)
        else:
            sub = expt_name

        return setup_output_directories(output_dir, sub, self.log)

    def _build_local_config_template(self) -> Dict[str, Any]:
        """Build the fresh-blueprint dict from current config + parameters."""
        init_config = self.config['entries']
        params_dict = self.params.to_dict()
        template = {
            "Map": get_map_name_from_input(init_config['input']),
            "Creation Date": datetime.now().isoformat(),
            "Initialisation Data": init_config,
            "_Pipeline Modules": "Parameters and settings for each module",
        }
        for stage in STAGES:
            template[stage.display] = {
                "Parameters": params_dict.get(stage.module, {}),
                "Output Parameter": {},
            }
        return template

    def _create_local_config(self, dirs):
        file_path = os.path.join(dirs['base_output'], "Local_config.json")
        template = self._build_local_config_template()

        if os.path.exists(file_path):
            try:
                config_data = deep_merge_missing(load_json(file_path), template)
                self.log(f"[INFO] Loaded and updated existing configuration from {file_path}")
            except json.JSONDecodeError:
                self.log(f"[WARNING] {file_path} contains invalid JSON. Overwriting with new config.")
                config_data = template
        else:
            config_data = template

        save_json(file_path, config_data, indent=4)

        self.log(f"[INFO] Configuration written to {file_path}")
        self.config['entries']['local_config'] = file_path
        
    def _run_stage(self, stage, dirs: Dict[str, Path]) -> None:
        """Run a single pipeline stage."""
        if not self.config['modules'].get(stage.key, False):
            self.log(f"[SKIP] {stage.display} disabled, skipping...")
            return

        start_time = time.time()
        self.log(f"[STAGE] Starting {stage.display.lower()}...")
        self.update_status(f"Running {stage.display.lower()}...")

        try:
            module = getattr(modules, stage.module)
            module.run(self.config, dirs, self.callbacks, self.params)
            elapsed = time.time() - start_time
            self.log(f"[INFO] {stage.display} completed in {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            self.log(f"[ERROR] {stage.display} failed after {elapsed:.2f}s: {e}")
            raise
    
