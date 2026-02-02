"""
Internationalization (i18n) Module
===================================
Provides multi-language support for MindBrush.

Loads translations based on the language setting in .chainlit/config.toml
and provides a simple translation function.
"""

import json
import tomli
from pathlib import Path
from functools import lru_cache
from typing import Dict, Any, Optional


# ==============================================================================
# Configuration Loading
# ==============================================================================

def _load_chainlit_config() -> Dict[str, Any]:
    """Load Chainlit configuration to get language setting."""
    config_path = Path(".chainlit/config.toml")
    
    if not config_path.exists():
        return {}
    
    with open(config_path, "rb") as f:
        return tomli.load(f)


def _get_language() -> str:
    """Get the configured language from Chainlit config."""
    config = _load_chainlit_config()
    ui_config = config.get("UI", {})
    language = ui_config.get("language", "en-US")
    
    # Normalize language code
    if language:
        return language
    return "en-US"


# ==============================================================================
# Message Loading
# ==============================================================================

@lru_cache(maxsize=2)
def _load_messages(language: str) -> Dict[str, Any]:
    """
    Load messages for a specific language.
    Falls back to en-US if language not found.
    """
    locales_dir = Path("locales")
    message_file = locales_dir / language / "messages.json"
    
    # Try to load requested language
    if message_file.exists():
        with open(message_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # Fallback to en-US
    fallback_file = locales_dir / "en-US" / "messages.json"
    if fallback_file.exists():
        with open(fallback_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    # If even fallback doesn't exist, return empty dict
    return {}


def _get_nested_value(data: Dict, key_path: str) -> Optional[str]:
    """
    Get a nested value from dictionary using dot notation.
    
    Example:
        _get_nested_value(data, "welcome.title")
        -> data["welcome"]["title"]
    """
    keys = key_path.split(".")
    value = data
    
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return None
    
    return value


# ==============================================================================
# Translation Function
# ==============================================================================

def t(key: str, **kwargs) -> str:
    """
    Translate a message key to the current language.
    
    Args:
        key: Message key in dot notation (e.g., "welcome.title")
        **kwargs: Optional format arguments for string interpolation
        
    Returns:
        Translated string, or the key itself if not found
        
    Example:
        >>> t("welcome.title")
        "Welcome to **MindBrush**"
        
        >>> t("completion.total_steps")
        "Total Steps"
    """
    language = _get_language()
    messages = _load_messages(language)
    
    value = _get_nested_value(messages, key)
    
    if value is None:
        # Return key if translation not found
        return key
    
    # Apply string formatting if kwargs provided
    if kwargs:
        try:
            return value.format(**kwargs)
        except (KeyError, ValueError):
            return value
    
    return value


def get_current_language() -> str:
    """Get the current language code."""
    return _get_language()


def get_chainlit_md_path() -> Path:
    """
    Get the path to the chainlit.md file for the current language.
    
    Returns:
        Path to the localized chainlit.md file
    """
    language = _get_language()
    locales_dir = Path("locales")
    md_file = locales_dir / language / "chainlit.md"
    
    if md_file.exists():
        return md_file
    
    # Fallback to en-US
    fallback_file = locales_dir / "en-US" / "chainlit.md"
    if fallback_file.exists():
        return fallback_file
    
    # Fallback to root chainlit.md
    return Path("chainlit.md")


def setup_chainlit_md() -> None:
    """
    Copy the appropriate chainlit.md to the root directory.
    Should be called on application startup.
    """
    import shutil
    
    source = get_chainlit_md_path()
    target = Path("chainlit.md")
    
    if source.exists() and source != target:
        shutil.copy(source, target)


# ==============================================================================
# Reload Function
# ==============================================================================

def reload_translations() -> None:
    """
    Force reload of translations.
    Clears the cache and reloads messages.
    """
    _load_messages.cache_clear()
