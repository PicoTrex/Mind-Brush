"""
Output Formatters
=================
Utilities for formatting step outputs in the Chainlit UI.
Provides rich markdown formatting with hierarchical display and configurable styles.
"""

import os
from typing import Any, Dict, List, Union, Optional


class OutputFormatter:
    """
    Configurable output formatter for step results.
    Supports different display styles per field.
    """
    
    def __init__(self, display_config: Optional[Dict] = None, field_config: Optional[Dict] = None):
        """
        Args:
            display_config: Global display settings (heading levels, etc.)
            field_config: Per-field display settings from step config
        """
        self.display_config = display_config or {}
        self.field_config = field_config or {}
        
        # Heading levels
        self.h1 = self.display_config.get("heading_level_1", "###")
        self.h2 = self.display_config.get("heading_level_2", "####")
        self.h3 = self.display_config.get("heading_level_3", "#####")
    
    def get_field_style(self, field_name: str) -> Dict:
        """Get display style for a field."""
        return self.field_config.get(field_name, {})
    
    def format_field_name(self, name: str, level: int = 1) -> str:
        """Format field name with appropriate heading level."""
        # Convert snake_case to Title Case
        display_name = name.replace("_", " ").title()
        
        if level == 1:
            return f"{self.h1} {display_name}"
        elif level == 2:
            return f"{self.h2} {display_name}"
        else:
            return f"{self.h3} {display_name}"
    
    def format_string_value(self, value: str, style: str) -> str:
        """Format a string value with the specified style."""
        if style == "code_block":
            return f"`{value}`"
        elif style == "quote_block":
            lines = value.split("\n")
            return "\n".join(f"> {line}" for line in lines)
        else:
            return value
    
    def format_list_items(self, items: List, field_name: str, style: str) -> str:
        """Format list items with the specified style."""
        if not items:
            return "_Empty_"
        
        lines = []
        for item in items:
            if isinstance(item, dict):
                lines.append(self._format_dict(item, 2))
            elif isinstance(item, str):
                if style == "code_block":
                    lines.append(f"> `{item}`")
                elif style == "quote_block":
                    lines.append(f"> {item}")
                else:
                    lines.append(f"- {item}")
            else:
                lines.append(f"- {item}")
        
        return "\n".join(lines)
    
    def _format_dict(self, data: Dict, level: int) -> str:
        """Format a dictionary with proper structure."""
        lines = []
        
        for key, value in data.items():
            field_style = self.get_field_style(key)
            style_type = field_style.get("style", "plain")
            
            # Always use heading for field name
            lines.append(f"\n{self.format_field_name(key, level)}\n")
            
            if isinstance(value, dict):
                lines.append(self._format_dict(value, level + 1))
            elif isinstance(value, list):
                formatted_items = self.format_list_items(value, key, style_type)
                lines.append(formatted_items)
            elif isinstance(value, str):
                formatted_value = self.format_string_value(value, style_type)
                lines.append(formatted_value)
            elif value is None:
                lines.append("_None_")
            elif isinstance(value, bool):
                lines.append("✓" if value else "✗")
            else:
                lines.append(str(value))
        
        return "\n".join(lines)
    
    def format(self, data: Union[Dict, List, str]) -> str:
        """
        Format data as markdown with configured styles.
        
        Args:
            data: Dictionary, list, or string to format
            
        Returns:
            Formatted markdown string
        """
        if isinstance(data, str):
            import json
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return data
        
        if isinstance(data, list):
            return self.format_list_items(data, "", "plain")
        
        if not isinstance(data, dict):
            return str(data)
        
        # Handle 'result' wrapper
        if "result" in data and len(data) == 1:
            result_data = data["result"]
            # Apply field config for "result" field
            field_style = self.get_field_style("result")
            style_type = field_style.get("style", "plain")
            
            if isinstance(result_data, list):
                lines = [f"\n{self.format_field_name('result', 1)}\n"]
                lines.append(self.format_list_items(result_data, "result", style_type))
                return "\n".join(lines)
            elif isinstance(result_data, str):
                return self.format_string_value(result_data, style_type)
            else:
                return self.format(result_data)
        
        return self._format_dict(data, 1)


def format_output_markdown(
    data: Union[Dict, List, str],
    display_config: Optional[Dict] = None,
    field_config: Optional[Dict] = None
) -> str:
    """
    Format data as markdown with bold titles and hierarchical display.
    """
    formatter = OutputFormatter(display_config, field_config)
    return formatter.format(data)


def format_duration(ms: float) -> str:
    """Format duration with bold label and italic time."""
    seconds = ms / 1000
    if seconds < 1:
        return f"**⏱️ Duration**: *{ms:.0f}ms*"
    elif seconds < 60:
        return f"**⏱️ Duration**: *{seconds:.2f}s*"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"**⏱️ Duration**: *{minutes}m {secs:.1f}s*"


def format_running_time(seconds: float) -> str:
    """Format running time for live display."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    else:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"


def format_step_header(step_name: str, icon: str, status: str = "running") -> str:
    """Format step header for display."""
    status_icons = {
        "running": "🔄",
        "completed": "✅", 
        "failed": "❌",
        "pending": "⏳",
        "skipped": "⏭️"
    }
    status_icon = status_icons.get(status, "❓")
    return f"{icon} {step_name} {status_icon}"


def format_error(error_message: str) -> str:
    """Format error message."""
    return f"❌ **Error**\n\n```\n{error_message}\n```"


def format_completion_message(session_id: str, total_steps: int, total_time_ms: float) -> str:
    """Format a prominent completion message."""
    from core.i18n import t
    
    total_seconds = total_time_ms / 1000
    if total_seconds < 60:
        time_str = f"{total_seconds:.1f}s"
    else:
        minutes = int(total_seconds // 60)
        secs = total_seconds % 60
        time_str = f"{minutes}m {secs:.1f}s"
    
    return (
        f"\n---\n\n"
        f"## ✨ **{t('workflow.complete')}**\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| 📊 {t('completion.total_steps')} | **{total_steps}** |\n"
        f"| ⏱️ {t('completion.total_time')} | **{time_str}** |\n"
        f"| 📁 {t('completion.session')} | `{session_id}` |\n\n"
        f"---\n"
    )
