"""
Configuration Loader
Loads business rules from rules.yaml at startup (no hot-reload).
"""

import os
import yaml
from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parent
_CONFIG_PATH = _CONFIG_DIR / "rules.yaml"
_cached_config = None


def load_config(force_reload: bool = False) -> dict:
    """Load the YAML configuration. Cached after first load."""
    global _cached_config
    if _cached_config is not None and not force_reload:
        return _cached_config

    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found: {_CONFIG_PATH}")

    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        _cached_config = yaml.safe_load(f)

    return _cached_config


def get_uniform_rules() -> dict:
    """Return uniform/tshirt rules from config."""
    cfg = load_config()
    return cfg.get("uniform", {})


def get_cap_rules() -> dict:
    """Return cap detection rules."""
    cfg = load_config()
    return cfg.get("cap", {})


def get_accessory_rules() -> dict:
    """Return all accessory definitions."""
    cfg = load_config()
    known_accessories = {"gloves", "apron", "mask", "hairnet", "id_card", "jacket"}
    return {k: v for k, v in cfg.items() if k in known_accessories}


def get_thresholds() -> dict:
    """Return detection thresholds."""
    cfg = load_config()
    return {
        "minimum_face_size": cfg.get("minimum_face_size", 30),
        "maximum_face_size": cfg.get("maximum_face_size", 70),
        "minimum_blur_score": cfg.get("minimum_blur_score", 60),
        "minimum_visibility": cfg.get("minimum_visibility", 75),
        "minimum_brightness": cfg.get("minimum_brightness", 50),
        "minimum_overall_score": cfg.get("minimum_overall_score", 80),
    }
