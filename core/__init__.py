"""
MindBrush Core Module
=====================
This module provides the core infrastructure for the MindBrush agent system.

Components:
- config_loader: Pydantic-based configuration management
- mcp_client: MCP protocol client for tool invocation
- model_provider: Unified LLM interface
"""

from .config_loader import get_settings, Settings
from .model_provider import ModelProvider
from .session_manager import SessionManager, get_session_manager

__all__ = [
    "get_settings",
    "Settings", 
    "ModelProvider",
    "SessionManager",
    "get_session_manager",
]
