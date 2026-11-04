"""
Configuration Loader
====================
Loads experiment parameters from YAML configuration files.
"""

from typing import Dict, Any
import os
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file into a dictionary."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config or {}
