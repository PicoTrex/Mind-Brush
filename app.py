"""
MindBrush Chainlit Application
==============================
Main entry point for the MindBrush agent with Chainlit UI.

Supports:
- Text input
- Image input
- Text + Image combined input
- Real-time step-by-step progress display
- Final image result display
- Session-based storage with timestamps
- Custom step icons
"""

import os
import json
import yaml
import asyncio
from pathlib import Path
from typing import Optional, List, Dict

import chainlit as cl

from services.agent_service import MindBrushAgent, StepResult, StepStatus, AgentResult
from core.config_loader import get_settings
from core.session_manager import get_session_manager

# ==============================================================================
# Configuration
# ==============================================================================

settings = get_settings()
session_manager = get_session_manager()

# Load step icons configuration
STEP_ICONS_CONFIG_PATH = Path("./configs/step_icons.yaml")
STEP_ICONS: Dict[str, str] = {}
STATUS_ICONS: Dict[str, str] = {}

try:
    with open(STEP_ICONS_CONFIG_PATH, "r", encoding="utf-8") as f:
        icons_config = yaml.safe_load(f)
        STEP_ICONS = icons_config.get("step_icons", {})
        STATUS_ICONS = icons_config.get("status_icons", {})
except Exception as e:
    print(f"Warning: Could not load step icons config: {e}")
    # Fallback defaults
    STEP_ICONS = {"default": "⚙️"}
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

def get_step_icon(step_name: str) -> str:
    """Get custom icon for a step."""
    return STEP_ICONS.get(step_name, STEP_ICONS.get("default", "⚙️"))


def get_status_icon(status: StepStatus) -> str:
    """Get icon for step status."""
    status_name = status.value if isinstance(status, StepStatus) else str(status)
    return STATUS_ICONS.get(status_name, "❓")


def format_step_output(step: StepResult) -> str:
    """Format step output for display in Chainlit."""
    if step.status == StepStatus.FAILED:
        return f"❌ Error: {step.error_message}"
    
    if step.status == StepStatus.SKIPPED:
        return "⏭️ Skipped"
    
    output = step.output_data
    
    # Format as JSON if it's a dict
    if isinstance(output, dict):
        # Remove 'result' wrapper if present
        if 'result' in output and len(output) == 1:
            output = output['result']
        
        return f"```json\n{json.dumps(output, indent=2, ensure_ascii=False)}\n```"
    
    # Format as list
    if isinstance(output, list):
        return f"```json\n{json.dumps(output, indent=2, ensure_ascii=False)}\n```"
    
    return str(output)


async def save_uploaded_file(file: cl.File, session_id: str) -> str:
    """
    Save uploaded file to session-specific upload directory.
    
    Args:
        file: Chainlit File object
        session_id: Current session ID
        
    Returns:
        Path to saved file
    """
    upload_dir = session_manager.get_upload_dir()
    
    # Generate unique filename
    filename = f"{file.name}"
    file_path = upload_dir / filename
    
    # Copy file content
    with open(file.path, "rb") as src:
        with open(file_path, "wb") as dst:
            dst.write(src.read())
    
    return str(file_path)


async def extract_image_from_message(message: cl.Message, session_id: str) -> Optional[str]:
    """
    Extract image path from Chainlit message.
    
    Args:
        message: Chainlit Message object
        session_id: Current session ID
        
    Returns:
        Path to saved image file, or None
    """
    if not message.elements:
        return None
    
    for element in message.elements:
        if isinstance(element, cl.Image):
            # Save to session-specific storage
            return await save_uploaded_file(element, session_id)
    
    return None


# ==============================================================================
# Chainlit Event Handlers
# ==============================================================================

@cl.on_chat_start
async def start():
    """Initialize chat session with timestamped storage."""
    # Create new session
    session_id = session_manager.create_session()
    
    # Store session ID in user session
    cl.user_session.set("session_id", session_id)
    
    await cl.Message(
        content=(
            f"## 🎨 Welcome to **MindBrush**\n"
            f"--- \n"
            f"Your creative companion for AI-powered image generation.\n\n"
            f"> 🆔 **Session ID**\n"
            f"> `{session_id}`\n\n"
            f"### 🚀 **Quick Start Guide**\n"
            f"* **Text-to-Image**: Describe your vision (e.g., *'A cyberpunk city in the rain'*).\n"
            f"* **Image-to-Image**: Upload a reference photo to guide the style.\n"
            f"* **Multi-Modal**: Send both text and images for precise control.\n\n"
            f"--- \n"
            f"✨ *Type your prompt below to start creating!*"
        ),
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """
    Handle incoming user messages.
    Orchestrates the complete MindBrush workflow.
    """
    # Get session ID
    session_id = cl.user_session.get("session_id")
    if not session_id:
        await cl.Message(content="⚠️ Session not initialized. Please refresh.").send()
        return
    
    # Parse message content
    text_input = message.content.strip()
    image_input = await extract_image_from_message(message, session_id)
    
    # Validate input
    if not text_input and not image_input:
        await cl.Message(content="⚠️ Please provide text, an image, or both.").send()
        return
    
    # Create step tracking message
    progress_msg = await cl.Message(content="🚀 Starting MindBrush workflow...").send()
    
    # Initialize agent with session directory
    session_dir = session_manager.get_session_dir()
    agent = MindBrushAgent(session_dir=str(session_dir))
    
    # Track completed steps for display
    completed_steps: List[StepResult] = []
    
    async def on_step_complete(step: StepResult):
        """Callback for step completion updates."""
        completed_steps.append(step)
        
        # Get custom icon for this step
        step_icon = get_step_icon(step.step_name)
        status_icon = get_status_icon(step.status)
        
        # Create step display using Chainlit Step
        async with cl.Step(
            name=f"{step_icon} {step.step_name}",
            type="tool" if "Search" in step.step_name or "RAG" in step.step_name else "llm",
        ) as ui_step:
            # Show input
            if step.input_data:
                input_preview = json.dumps(step.input_data, indent=2, ensure_ascii=False)
                if len(input_preview) > 500:
                    input_preview = input_preview[:500] + "..."
                ui_step.input = f"```json\n{input_preview}\n```"
            
            # Show output
            ui_step.output = format_step_output(step)
            
            # Special handling for Image Search - display downloaded images
            if (step.step_name == "Image Search" and step.status == StepStatus.COMPLETED):
                image_paths = []
                
                # Extract image paths from output
                if isinstance(step.output_data, dict) and "result" in step.output_data:
                    image_paths = step.output_data.get("result", [])
                elif isinstance(step.output_data, list):
                    image_paths = step.output_data
                
                # Display images
                if image_paths:
                    image_elements = []
                    for idx, img_path in enumerate(image_paths):
                        if os.path.exists(img_path):
                            image_elements.append(
                                cl.Image(
                                    path=img_path,
                                    name=f"Reference {idx + 1}",
                                    display="inline",
                                    size="small",
                                )
                            )
                    
                    if image_elements:
                        ui_step.elements = image_elements
                        ui_step.output += f"\n\n📸 Downloaded {len(image_elements)} reference images"
            
            # Add duration info
            if step.duration_ms > 0:
                duration_sec = step.duration_ms / 1000
                ui_step.output += f"\n\n⏱️ Duration: {duration_sec:.2f}s"
    
    # Run the workflow
    try:
        result = await agent.process(
            text_input=text_input,
            image_input=image_input,
            on_step_complete=on_step_complete,
        )
        
        # Display final result
        if result.success:
            # Create result message with images
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
            
            # Save result metadata
            results_dir = session_manager.get_results_dir()
            metadata_path = results_dir / "result_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump({
                    "session_id": session_id,
                    "text_input": text_input,
                    "image_input": image_input,
                    "final_prompt": result.final_prompt,
                    "final_images": result.final_images,
                    "steps_count": len(result.steps),
                }, f, indent=2, ensure_ascii=False)
            
            await cl.Message(
                content=f"✨ **Generation Complete!**\n\n"
                        f"**Final Prompt:**\n```\n{result.final_prompt}\n```\n\n"
                        f"📁 Results saved to: `{session_manager.get_session_dir()}`",
                elements=elements if elements else None,
            ).send()
            
        else:
            await cl.Message(
                content=f"❌ **Generation Failed**\n\n"
                        f"Error: {result.error_message}",
            ).send()
            
    except Exception as e:
        await cl.Message(
            content=f"❌ **Unexpected Error**\n\n"
                    f"```\n{str(e)}\n```",
        ).send()


# ==============================================================================
# File Upload Handler
# ==============================================================================

@cl.on_settings_update
async def setup_agent(settings):
    """Handle settings updates if needed."""
    pass


# ==============================================================================
# Entry Point (for development)
# ==============================================================================

if __name__ == "__main__":
    # For development, run with: chainlit run app.py -w
    pass