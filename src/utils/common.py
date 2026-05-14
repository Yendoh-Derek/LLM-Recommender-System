"""
common.py
Shared utility functions.
"""
from pathlib import Path
from typing import Any, Dict

import json

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        text = f.read()

    if config_path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise ImportError("PyYAML is required to load YAML config files")
        content = yaml.safe_load(text)
    elif config_path.suffix.lower() == ".json":
        content = json.loads(text)
    else:
        if yaml is not None:
            content = yaml.safe_load(text)
        else:
            content = json.loads(text)

    return content or {}
