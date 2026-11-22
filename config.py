"""GUI configuration manager — read/write/migrate `map_pipeline_config.json`."""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

from stages import default_modules_config


class Config:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.ensure_config_directory()

    def ensure_config_directory(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, config_data: Dict[str, Any]) -> bool:
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False

    def load(self) -> Optional[Dict[str, Any]]:
        if not self.config_path.exists():
            return None
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return None

    def get_default(self) -> Dict[str, Any]:
        return {
            'entries': {
                'input': '',
                'output': '',
                'model': '',
                'osm': '',
                'osm_road': '',
                'ocr': '',
                'text_model': '',
                'bounding_coordinate_1': '',
                'bounding_coordinate_2': '',
            },
            'modules': default_modules_config(),
            'model_parameters_file': 'config/model_parameters.json',
        }

    # Pre-stages-registry configs used PascalCase keys for these three stages.
    _LEGACY_MODULE_KEY_MAP = {
        'Text_Detection': 'text',
        'Geolocation':    'geoloc',
        'Similarity':     'similarity',
    }

    def merge_with_defaults(self, config_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge loaded configuration with defaults; migrate legacy module keys."""
        default_config = self.get_default()
        if config_data is None:
            return default_config

        for key, default_value in default_config['entries'].items():
            if key not in config_data.get('entries', {}):
                config_data.setdefault('entries', {})[key] = default_value

        modules = config_data.setdefault('modules', {})
        for legacy_key, current_key in self._LEGACY_MODULE_KEY_MAP.items():
            if legacy_key in modules and current_key not in modules:
                modules[current_key] = modules.pop(legacy_key)

        for key, default_value in default_config['modules'].items():
            if key not in modules:
                modules[key] = default_value

        return config_data

    def validate(self, config_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []

        for section in ('entries', 'modules'):
            if section not in config_data:
                errors.append(f"Missing required section: {section}")

        if 'entries' in config_data:
            for entry in ('input', 'output', 'expt'):
                if not config_data['entries'].get(entry, '').strip():
                    errors.append(f"Required entry '{entry}' is empty")

            for entry in ('input', 'output', 'model', 'osm', 'ocr', 'text_model'):
                path = config_data['entries'].get(entry, '').strip()
                if path and not os.path.exists(path):
                    errors.append(f"Path does not exist: {entry} = {path}")

        return len(errors) == 0, errors
