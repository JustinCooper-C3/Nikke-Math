"""
Settings Module for Nikke Math Solver

Provides persistent storage for user preferences using JSON.
Settings are stored in config.json in the project root.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Settings file location (project root)
SETTINGS_FILE = Path("config.json")

# Default settings
DEFAULT_SETTINGS: Dict[str, Any] = {
    "debug_enabled": False,
    "strategy_name": "greedy"
}


def load_settings() -> Dict[str, Any]:
    """
    Load settings from config.json.

    Returns:
        Settings dictionary. Returns defaults if file missing or invalid.
    """
    if not SETTINGS_FILE.exists():
        logger.debug("Settings file not found, using defaults")
        return DEFAULT_SETTINGS.copy()

    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            settings = json.load(f)

        # Merge with defaults to handle missing keys
        result = DEFAULT_SETTINGS.copy()
        result.update(settings)
        logger.debug(f"Settings loaded: {result}")
        return result

    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load settings: {e}, using defaults")
        return DEFAULT_SETTINGS.copy()


def save_settings(settings: Dict[str, Any]) -> None:
    """
    Save settings to config.json.

    Args:
        settings: Settings dictionary to save
    """
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)
        logger.debug(f"Settings saved: {settings}")
    except IOError as e:
        logger.error(f"Failed to save settings: {e}")
