"""
MindBrush Chainlit Application
==============================
Main entry point for the MindBrush agent with Chainlit UI.

Supports:
- Text input
- Image input
- Text + Image combined input
- Real-time step-by-step progress display with live timer
- Configurable image visualization per field
- Rich output formatting with field-specific styles
- Lazy session creation (only when user sends message)
"""

import os
import json
import yaml
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any

import chainlit as cl

from services.agent_service import MindBrushAgent, StepResult, StepStatus, AgentResult
from core.config_loader import get_settings
from core.session_manager import get_session_manager
from core.formatters import OutputFormatter, format_duration, format_running_time, format_completion_message
from core.i18n import t, setup_chainlit_md


# ==============================================================================
# Configuration
# ==============================================================================

settings = get_settings()
session_manager = get_session_manager()

# Setup localized chainlit.md on startup
setup_chainlit_md()


# Load step configuration
STEP_CONFIG_PATH = Path("./configs/step_config.yaml")
STEP_CONFIG: Dict[str, Any] = {}
TOOLS_CONFIG: Dict[str, Any] = {}
STATUS_ICONS: Dict[str, str] = {}
DISPLAY_CONFIG: Dict[str, Any] = {}
DEFAULT_CONFIG: Dict[str, Any] = {"icon": "⚙️", "show_images": False, "field_display": {}}

try:
    with open(STEP_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        TOOLS_CONFIG = config.get("tools", {})
        STATUS_ICONS = config.get("status_icons", {})
        DEFAULT_CONFIG = config.get("defaults", DEFAULT_CONFIG)
        DISPLAY_CONFIG = config.get("display", {})
        
        # Build step config by display_name for easy lookup
        for tool_id, tool_config in TOOLS_CONFIG.items():
            display_name = tool_config.get("display_name", tool_id)
            STEP_CONFIG[display_name] = tool_config
except Exception as e:
    print(f"Warning: Could not load step config: {e}")
    STATUS_ICONS = {
        "pending": "⏳",
        "running": "🔄",
        "completed": "✅",
        "failed": "❌",
        "skipped": "⏭️",
    }


# ==============================================================================
# Helper Functions
# ==============================================================================

def get_step_config(step_name: str) -> Dict[str, Any]:
    """Get configuration for a step."""
    return STEP_CONFIG.get(step_name, DEFAULT_CONFIG)


def get_step_icon(step_name: str) -> str:
    """Get custom icon for a step."""
    config = get_step_config(step_name)
    return config.get("icon", DEFAULT_CONFIG.get("icon", "⚙️"))


def get_status_icon(status: StepStatus) -> str:
    """Get icon for step status."""
    status_name = status.value if isinstance(status, StepStatus) else str(status)
    return STATUS_ICONS.get(status_name, "❓")


def get_field_display_config(step_name: str) -> Dict[str, Any]:
    """Get field display configuration for a step."""
    config = get_step_config(step_name)
    return config.get("field_display", {})


def get_image_fields(step_name: str) -> List[str]:
    """Get list of fields that contain images."""
    config = get_step_config(step_name)
    return config.get("image_fields", [])


def extract_images_from_output(output_data: Dict[str, Any], image_fields: List[str]) -> List[str]:
    """Extract image paths from output data based on configured fields."""
    if not output_data or not image_fields:
        return []
    
    all_images = []
    
    for field in image_fields:
        data = output_data.get(field, [])
        
        if isinstance(data, list):
            for path in data:
                if isinstance(path, str) and os.path.exists(path):
                    all_images.append(path)
        elif isinstance(data, str) and os.path.exists(data):
            all_images.append(data)
    
    return all_images


async def save_uploaded_file(file: cl.File, session_id: str) -> str:
    """Save uploaded file to session-specific upload directory."""
    upload_dir = session_manager.get_upload_dir()
    
    filename = f"{file.name}"
    file_path = upload_dir / filename
    
    with open(file.path, "rb") as src:
        with open(file_path, "wb") as dst:
            dst.write(src.read())
    
    return str(file_path)


async def extract_image_from_message(message: cl.Message, session_id: str) -> Optional[str]:
    """Extract image path from Chainlit message."""
    if not message.elements:
        return None
    
    for element in message.elements:
        if isinstance(element, cl.Image):
            return await save_uploaded_file(element, session_id)
    
    return None


# ==============================================================================
# Chainlit Event Handlers
# ==============================================================================

@cl.on_chat_start
async def start():
    """Initialize chat session (lazy - no directories created yet)."""
    # Only prepare session ID, do NOT create directories
    session_id = session_manager.prepare_session()
    
    # Store session ID in user session
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("active_steps", {})
    
    await cl.Message(
        content=(
            f"## 🎨 {t('welcome.title')}\n"
            f"--- \n"
            f"{t('welcome.subtitle')}\n\n"
            f"> 🆔 **{t('welcome.session_id')}**\n"
            f"> `{session_id}`\n\n"
            f"### 🚀 **{t('welcome.quick_start')}**\n"
            f"* {t('welcome.text_to_image')}\n"
            f"* {t('welcome.image_to_image')}\n"
            f"* {t('welcome.multi_modal')}\n\n"
            f"--- \n"
            f"✨ *{t('welcome.start_prompt')}*"
        ),
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming user messages."""
    # Get session ID
    session_id = cl.user_session.get("session_id")
    if not session_id:
        await cl.Message(content=f"⚠️ {t('errors.no_session')}").send()
        return
    
    # Parse message content
    text_input = message.content.strip()
    
    # Validate input first (before creating any directories)
    if not text_input and not message.elements:
        await cl.Message(content=f"⚠️ {t('errors.no_input')}").send()
        return
    
    # NOW create session directories (lazy creation)
    session_dir = session_manager.ensure_session_dirs()
    
    # Extract image if provided
    image_input = await extract_image_from_message(message, session_id)
    
    # Create progress message
    await cl.Message(content=f"🚀 {t('workflow.starting')}").send()
    
    # Track active steps for timer updates
    active_steps: Dict[str, Any] = {}
    cl.user_session.set("active_steps", active_steps)
    
    # Initialize agent with session
    agent = MindBrushAgent(
        session_dir=str(session_dir),
        session_manager=session_manager
    )
    
    async def on_step_start(step: StepResult):
        """Callback when a step starts - show step immediately with live timer."""
        step_icon = get_step_icon(step.step_name)
        
        # Create step UI immediately
        ui_step = cl.Step(
            name=f"{step_icon} {step.step_name}",
            type="tool" if "Search" in step.step_name else "llm",
        )
        await ui_step.__aenter__()
        
        # Store start time
        start_time = asyncio.get_event_loop().time()
        
        # Store for later update
        active_steps[step.step_name] = {
            "ui_step": ui_step,
            "start_time": start_time,
            "timer_task": None,
        }
        
        # Start timer update task - shows running time on the right
        async def update_timer():
            try:
                while step.step_name in active_steps:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    timer_display = format_running_time(elapsed)
                    ui_step.output = f"🔄 {t('step.running')} **{timer_display}**"
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                pass
        
        timer_task = asyncio.create_task(update_timer())
        active_steps[step.step_name]["timer_task"] = timer_task
    
    async def on_step_complete(step: StepResult):
        """Callback for step completion - update with formatted output."""
        step_info = active_steps.get(step.step_name)
        
        if step_info:
            # Cancel timer
            if step_info.get("timer_task"):
                step_info["timer_task"].cancel()
                try:
                    await step_info["timer_task"]
                except asyncio.CancelledError:
                    pass
            
            ui_step = step_info["ui_step"]
            
            # Format output
            if step.status == StepStatus.FAILED:
                ui_step.output = f"❌ **Error**\n\n```\n{step.error_message}\n```"
            elif step.status == StepStatus.SKIPPED:
                ui_step.output = "⏭️ Skipped"
            else:
                # Get field display config for this step
                field_config = get_field_display_config(step.step_name)
                
                # Format output with configured styles
                formatter = OutputFormatter(DISPLAY_CONFIG, field_config)
                formatted_output = formatter.format(step.output_data)
                ui_step.output = formatted_output
                
                # Check if we should display images
                image_fields = get_image_fields(step.step_name)
                if image_fields:
                    image_paths = extract_images_from_output(step.output_data, image_fields)
                    
                    if image_paths:
                        image_elements = []
                        for idx, img_path in enumerate(image_paths):
                            image_elements.append(
                                cl.Image(
                                    path=img_path,
                                    name=f"Image {idx + 1}",
                                    display="inline",
                                    size="small" if len(image_paths) > 1 else "medium",
                                )
                            )
                        
                        if image_elements:
                            ui_step.elements = image_elements
                            ui_step.output += f"\n\n📸 {len(image_elements)} {t('step.images_displayed')}"
                
                # Add duration
                ui_step.output += f"\n\n{format_duration(step.duration_ms)}"
            
            # Close step
            await ui_step.__aexit__(None, None, None)
            
            # Remove from active
            del active_steps[step.step_name]
    
    # Run the workflow
    try:
        result = await agent.process(
            text_input=text_input,
            image_input=image_input,
            on_step_start=on_step_start,
            on_step_complete=on_step_complete,
        )
        
        # Display final result
        if result.success:
            elements: List[cl.Element] = []
            
            for idx, img_path in enumerate(result.final_images):
                if os.path.exists(img_path):
                    elements.append(
                        cl.Image(
                            path=img_path,
                            name=f"Generated Image {idx + 1}",
                            display="inline",
                        )
                    )
            
            # Calculate total time
            total_time_ms = sum(s.duration_ms for s in result.steps)
            
            # Format completion message with prominent display
            completion_msg = format_completion_message(
                session_id=session_manager.session_id,
                total_steps=len(result.steps),
                total_time_ms=total_time_ms
            )
            
            # Format final prompt in quote block
            final_content = (
                f"{completion_msg}\n\n"
                f"> **{t('completion.final_prompt')}**\n"
                f"> \n"
                f"> {result.final_prompt}"
            )
            
            await cl.Message(
                content=final_content,
                elements=elements if elements else None,
            ).send()
            
        else:
            await cl.Message(
                content=f"❌ **{t('workflow.failed')}**\n\n"
                        f"```\n{result.error_message}\n```",
            ).send()
            
    except Exception as e:
        await cl.Message(
            content=f"❌ **{t('workflow.unexpected_error')}**\n\n"
                    f"```\n{str(e)}\n```",
        ).send()


# ==============================================================================
# Settings Handler
# ==============================================================================

@cl.on_settings_update
async def setup_agent(settings):
    """Handle settings updates if needed."""
    pass


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    pass