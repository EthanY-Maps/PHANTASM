"""
Shared scaffolding for module-level `run()` functions.

Each `modules/<stage>.py` is a thin wrapper around an underlying impl.
This module absorbs the boilerplate they all repeat: resolving callbacks,
loading default parameters when none are supplied, and logging errors
with the correct stage name.
"""

import sys
from pathlib import Path
from functools import wraps
from typing import Dict, Any, Optional, Callable

from stages import STAGES_BY_MODULE


_MODULES_DIR = Path(__file__).parent
_PARENT_DIR = _MODULES_DIR.parent


def add_paths(*subdirs: str) -> None:
    """Insert the project root and given `modules/<sub>` dirs onto sys.path. Idempotent."""
    if str(_PARENT_DIR) not in sys.path:
        sys.path.insert(0, str(_PARENT_DIR))
    for sub in subdirs:
        p = _MODULES_DIR / sub
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def stage_runner(fn: Callable) -> Callable:
    """
    Decorator for module-level `run()` functions.

    The inner function is invoked with the unpacked signature
    `(config, dirs, log, update_status, params)` and may assume `params`
    is non-None and `log` / `update_status` are real callables.
    """
    module_basename = fn.__module__.split('.')[-1]
    display = STAGES_BY_MODULE[module_basename].display

    @wraps(fn)
    def wrapper(config: Dict[str, Any], dirs: Dict[str, Path],
                callbacks: Optional[Dict[str, Callable]] = None,
                params=None) -> None:
        cb = callbacks or {}
        log = cb.get('log', print)
        update_status = cb.get('update_status', lambda x: None)

        if params is None:
            from model_parameters import get_parameters
            params = get_parameters()

        try:
            return fn(config, dirs, log, update_status, params)
        except Exception as e:
            log(f"[ERROR] {display} failed: {e}")
            raise

    return wrapper
