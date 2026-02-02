"""
MindBrush Agent Service
=======================
Core orchestration service for the MindBrush workflow.
Coordinates tool invocations and manages intermediate state.

Workflow:
1. Intent Analysis -> Determine processing strategy
2. Keyword Generation -> Generate search keywords
3. Text/Image RAG -> Retrieve relevant information
4. Knowledge Reasoning -> Process retrieved data
5. Knowledge Review -> Optimize prompt
6. Image Generation -> Generate final image
"""

import sys
import asyncio
import time
import json

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Awaitable
from enum import Enum
from pathlib import Path
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from core.config_loader import get_settings
from core.session_manager import SessionManager


# ==============================================================================
# Data Models
# ==============================================================================

class StepStatus(str, Enum):
    """Status of a workflow step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepResult:
    """Result of a single workflow step."""
    step_name: str
    status: StepStatus
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    duration_ms: float = 0.0


@dataclass
class AgentResult:
    """Final result of the agent workflow."""
    success: bool
    final_images: List[str] = field(default_factory=list)
    final_prompt: str = ""
    steps: List[StepResult] = field(default_factory=list)
    error_message: Optional[str] = None


# Type alias for step callbacks
StepCallback = Callable[[StepResult], Awaitable[None]]


# ==============================================================================
# MCP Client Wrapper
# ==============================================================================

class MCPToolClient:
    """
    Wrapper for MCP tool invocation via stdio.
    Manages connection lifecycle and tool calls using AsyncExitStack.
    """
    
    def __init__(self, server_name: str, command: str, script_path: str, env: Optional[Dict[str, str]] = None):
        self.server_name = server_name
        self.command = command
        self.script_path = script_path
        self.env = env or {}
        self._session: Optional[ClientSession] = None
        self._exit_stack = AsyncExitStack()
    
    async def connect(self) -> None:
        """Establish connection to MCP server."""
        try:
            import os
            merged_env = os.environ.copy()
            merged_env.update(self.env)
            
            server_params = StdioServerParameters(
                command=self.command,
                args=[self.script_path],
                env=merged_env,
            )
            
            read, write = await self._exit_stack.enter_async_context(stdio_client(server_params))
            self._session = await self._exit_stack.enter_async_context(ClientSession(read, write))
            await self._session.initialize()
        except Exception as e:
            await self.disconnect()
            raise RuntimeError(f"Failed to connect to MCP server {self.server_name}: {str(e)}")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        if not self._session:
            raise RuntimeError(f"Not connected to MCP server: {self.server_name}")
        
        result = await self._session.call_tool(tool_name, arguments)
        
        if hasattr(result, 'content') and result.content:
            first_content = result.content[0]
            if hasattr(first_content, 'text'):
                return first_content.text
        
        return result
    
    async def disconnect(self) -> None:
        """Close connection to MCP server."""
        await self._exit_stack.aclose()
        self._session = None


# ==============================================================================
# Agent Service
# ==============================================================================

class MindBrushAgent:
    """
    Core orchestration agent for MindBrush.
    
    Manages the complete workflow from user input to generated image.
    Supports callbacks for real-time step updates.
    """
    
    def __init__(
        self, 
        session_dir: Optional[str] = None,
        session_manager: Optional[SessionManager] = None
    ):
        self.settings = get_settings()
        self._clients: Dict[str, MCPToolClient] = {}
        self.session_dir = session_dir
        self.session_manager = session_manager
        
    async def _get_client(self, server_name: str) -> MCPToolClient:
        """Get or create an MCP client for a server."""
        if server_name not in self._clients:
            server_config = self.settings.mcp_servers.get(server_name)
            if not server_config:
                raise ValueError(f"Unknown MCP server: {server_name}")
            
            command = server_config.command
            if command == "python":
                command = sys.executable
            
            env = {}
            if self.session_dir:
                env["MINDBRUSH_SESSION_DIR"] = str(self.session_dir)
                
            client = MCPToolClient(
                server_name=server_name,
                command=command,
                script_path=server_config.path,
                env=env,
            )
            await client.connect()
            self._clients[server_name] = client
        
        return self._clients[server_name]
    
    async def _call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """Call a tool and return the result."""
        client = await self._get_client(server_name)
        return await client.call_tool(tool_name, arguments)
    
    def _log_step(self, step_result: StepResult) -> None:
        """Log step result to session logs."""
        if not self.session_manager:
            return
            
        try:
            log_content = {
                "step_name": step_result.step_name,
                "status": step_result.status.value,
                "duration_ms": step_result.duration_ms,
                "input": step_result.input_data,
                "output": step_result.output_data,
                "error": step_result.error_message,
            }
            self.session_manager.write_step_log(step_result.step_name, log_content)
        except Exception as e:
            print(f"Warning: Failed to write step log: {e}")
    
    async def _run_step(
        self,
        step_name: str,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
        on_start: Optional[StepCallback] = None,
        on_complete: Optional[StepCallback] = None
    ) -> StepResult:
        """
        Execute a single workflow step.
        
        Args:
            step_name: Human-readable step name
            server_name: MCP server to use
            tool_name: Tool to invoke
            arguments: Tool arguments
            on_start: Callback when step starts (for real-time UI)
            on_complete: Callback for step completion
            
        Returns:
            StepResult with output or error
        """
        start_time = time.time()
        
        result = StepResult(
            step_name=step_name,
            status=StepStatus.RUNNING,
            input_data=arguments,
        )
        
        # Notify start
        if on_start:
            await on_start(result)
        
        try:
            output = await self._call_tool(server_name, tool_name, arguments)
            
            # Parse output if it's JSON string
            if isinstance(output, str):
                try:
                    output = json.loads(output)
                except json.JSONDecodeError:
                    pass
            
            result.status = StepStatus.COMPLETED
            result.output_data = output if isinstance(output, dict) else {"result": output}
            
        except Exception as e:
            result.status = StepStatus.FAILED
            result.error_message = str(e)
        
        result.duration_ms = (time.time() - start_time) * 1000
        
        # Log to session
        self._log_step(result)
        
        # Invoke completion callback
        if on_complete:
            await on_complete(result)
        
        return result
    
    async def process(
        self,
        text_input: str,
        image_input: Optional[str] = None,
        on_step_start: Optional[StepCallback] = None,
        on_step_complete: Optional[StepCallback] = None
    ) -> AgentResult:
        """
        Process user input through the complete MindBrush workflow.
        
        Args:
            text_input: User's text prompt
            image_input: Optional path to user's input image
            on_step_start: Callback invoked when each step starts
            on_step_complete: Callback invoked after each step completes
            
        Returns:
            AgentResult with final images and step history
        """
        steps: List[StepResult] = []
        
        try:
            # =================================================================
            # Step 1: Intent Analysis
            # =================================================================
            step1 = await self._run_step(
                step_name="Intent Analysis",
                server_name="IntentAnalysis",
                tool_name="intent_analyzer",
                arguments={
                    "user_intent": text_input,
                    "user_image_path": image_input,
                },
                on_start=on_step_start,
                on_complete=on_step_complete,
            )
            steps.append(step1)
            
            if step1.status == StepStatus.FAILED:
                return AgentResult(
                    success=False,
                    steps=steps,
                    error_message=step1.error_message,
                )
            
            intent_result = step1.output_data
            intent_category = intent_result.get("intent_category", "Direct_Generation")
            need_process_problem = intent_result.get("need_process_problem", [])
            
            # =================================================================
            # Step 2: Keyword Generation (if needed)
            # =================================================================
            text_queries = []
            image_queries = []
            
            if need_process_problem:
                step2 = await self._run_step(
                    step_name="Keyword Generation",
                    server_name="KeywordGeneration",
                    tool_name="keyword_generation",
                    arguments={
                        "need_process_problem": need_process_problem,
                    },
                    on_start=on_step_start,
                    on_complete=on_step_complete,
                )
                steps.append(step2)
                
                if step2.status == StepStatus.COMPLETED:
                    text_queries = step2.output_data.get("text_queries", [])
                    image_queries = step2.output_data.get("image_queries", [])
            
            # =================================================================
            # Step 3: Text RAG (if text queries exist)
            # =================================================================
            enriched_prompt = text_input
            
            if text_queries:
                step3 = await self._run_step(
                    step_name="Text Search & Knowledge Injection",
                    server_name="TextRAG",
                    tool_name="text_search_and_knowledge_injection",
                    arguments={
                        "text_queries": text_queries,
                        "user_intent": text_input,
                        "image_queries": image_queries,
                    },
                    on_start=on_step_start,
                    on_complete=on_step_complete,
                )
                steps.append(step3)
                
                if step3.status == StepStatus.COMPLETED:
                    enriched_prompt = step3.output_data.get("prompt", text_input)
                    image_queries = step3.output_data.get("final_image_queries", image_queries)
            
            # =================================================================
            # Step 4: Image RAG (if image queries exist)
            # =================================================================
            downloaded_paths: List[str] = []
            
            if image_queries:
                step4 = await self._run_step(
                    step_name="Image Search",
                    server_name="ImageRAG",
                    tool_name="search_and_download_images_batch",
                    arguments={
                        "image_queries": image_queries,
                    },
                    on_start=on_step_start,
                    on_complete=on_step_complete,
                )
                steps.append(step4)
                
                if step4.status == StepStatus.COMPLETED:
                    result = step4.output_data.get("result", [])
                    if isinstance(result, list):
                        downloaded_paths = result
                    elif isinstance(step4.output_data, list):
                        downloaded_paths = step4.output_data
            
            # =================================================================
            # Step 5: Knowledge Reasoning (if needed)
            # =================================================================
            reasoning_knowledge: List[str] = []
            
            if intent_category in ["Reasoning_Generation", "Reasoning_Search_Generation", "Search_Reasoning_Generation"]:
                step5 = await self._run_step(
                    step_name="Knowledge Reasoning",
                    server_name="KnowledgeReasoning",
                    tool_name="knowledge_reasoning",
                    arguments={
                        "user_intent": enriched_prompt,
                        "need_process_problem": need_process_problem,
                        "intent_category": intent_category,
                        "user_image_path": image_input,
                        "downloaded_paths": downloaded_paths,
                    },
                    on_start=on_step_start,
                    on_complete=on_step_complete,
                )
                steps.append(step5)
                
                if step5.status == StepStatus.COMPLETED:
                    reasoning_knowledge = step5.output_data.get("reasoning_knowledge", [])
            
            # =================================================================
            # Step 6: Knowledge Review
            # =================================================================
            step6 = await self._run_step(
                step_name="Knowledge Review",
                server_name="KnowledgeReview",
                tool_name="knowledge_review",
                arguments={
                    "prompt": enriched_prompt,
                    "need_process_problem": need_process_problem,
                    "reasoning_knowledge": reasoning_knowledge,
                    "downloaded_paths": downloaded_paths,
                    "input_image_path": image_input or "",
                },
                on_start=on_step_start,
                on_complete=on_step_complete,
            )
            steps.append(step6)
            
            final_prompt = enriched_prompt
            reference_images: List[str] = []
            
            if step6.status == StepStatus.COMPLETED:
                final_prompt = step6.output_data.get("final_prompt", enriched_prompt)
                reference_images = step6.output_data.get("reference_image", [])
            
            # =================================================================
            # Step 7: Image Generation
            # =================================================================
            step7 = await self._run_step(
                step_name="Image Generation",
                server_name="ImageGen",
                tool_name="unified_image_generator",
                arguments={
                    "prompt": final_prompt,
                    "reference_images": reference_images,
                },
                on_start=on_step_start,
                on_complete=on_step_complete,
            )
            steps.append(step7)
            
            if step7.status == StepStatus.FAILED:
                return AgentResult(
                    success=False,
                    steps=steps,
                    error_message=step7.error_message,
                    final_prompt=final_prompt,
                )
            
            final_images = step7.output_data.get("result", [])
            if isinstance(step7.output_data, list):
                final_images = step7.output_data
            
            return AgentResult(
                success=True,
                final_images=final_images,
                final_prompt=final_prompt,
                steps=steps,
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                steps=steps,
                error_message=str(e),
            )
        
        finally:
            # Cleanup: disconnect all clients
            for client in self._clients.values():
                try:
                    await client.disconnect()
                except Exception:
                    pass
            self._clients.clear()


# ==============================================================================
# Convenience Functions
# ==============================================================================

async def run_mindbrush(
    text_input: str,
    image_input: Optional[str] = None,
    on_step_start: Optional[StepCallback] = None,
    on_step_complete: Optional[StepCallback] = None
) -> AgentResult:
    """
    Convenience function to run MindBrush agent.
    """
    agent = MindBrushAgent()
    return await agent.process(
        text_input=text_input,
        image_input=image_input,
        on_step_start=on_step_start,
        on_step_complete=on_step_complete,
    )
