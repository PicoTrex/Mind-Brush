"""
Model Provider
==============
Unified interface for LLM API calls.
Abstracts away provider-specific details and provides common utilities.
"""

import json
import re
from typing import List, Dict, Any, Optional, Union
from openai import OpenAI

from .config_loader import get_settings, LLMSettings


class ModelProvider:
    """
    Unified LLM provider interface.
    
    Provides a consistent API for making LLM calls regardless of the
    underlying provider (OpenAI, Claude via OpenAI-compatible API, etc.)
    
    Example:
        >>> provider = ModelProvider()
        >>> response = provider.chat_completion([
        ...     {"role": "system", "content": "You are helpful."},
        ...     {"role": "user", "content": "Hello!"}
        ... ])
    """
    
    def __init__(self, settings: Optional[LLMSettings] = None):
        """
        Initialize the model provider.
        
        Args:
            settings: Optional LLM settings. If not provided, uses global settings.
        """
        if settings is None:
            settings = get_settings().llm
        
        self._settings = settings
        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
        )
    
    @property
    def model_name(self) -> str:
        """Get the configured model name."""
        return self._settings.model_name
    
    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Make a chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response
            **kwargs: Additional arguments passed to the API
            
        Returns:
            The assistant's response content as a string
            
        Raises:
            Exception: If the API call fails
        """
        request_params = {
            "model": self._settings.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens
        
        request_params.update(kwargs)
        
        response = self._client.chat.completions.create(**request_params)
        return response.choices[0].message.content.strip()
    
    def chat_completion_json(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make a chat completion request and parse the response as JSON.
        
        Automatically cleans markdown code blocks from the response.
        
        Args:
            messages: List of message dicts
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional arguments
            
        Returns:
            Parsed JSON response as a dictionary
            
        Raises:
            json.JSONDecodeError: If response cannot be parsed as JSON
        """
        content = self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        cleaned = clean_json_markdown(content)
        return json.loads(cleaned)


def clean_json_markdown(content: str) -> str:
    """
    Clean markdown code block formatting from JSON responses.
    
    LLMs often wrap JSON responses in ```json ... ``` blocks.
    This function removes such formatting to get clean JSON.
    
    Args:
        content: Raw response content that may contain markdown
        
    Returns:
        Cleaned content ready for JSON parsing
        
    Example:
        >>> clean_json_markdown('```json\\n{"key": "value"}\\n```')
        '{"key": "value"}'
    """
    content = content.strip()
    
    # Pattern 1: ```json ... ```
    if content.startswith("```json"):
        content = content[7:]  # Remove ```json
        if content.endswith("```"):
            content = content[:-3]  # Remove trailing ```
        return content.strip()
    
    # Pattern 2: ``` ... ```
    if content.startswith("```"):
        content = content[3:]  # Remove ```
        if content.endswith("```"):
            content = content[:-3]  # Remove trailing ```
        return content.strip()
    
    # Pattern 3: Use regex for more complex cases
    # Matches ```language\n...\n``` pattern
    match = re.match(r'^```\w*\n(.*)\n```$', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return content


def create_multimodal_content(
    text: str,
    images: Optional[List[str]] = None,
    image_detail: str = "auto"
) -> List[Dict[str, Any]]:
    """
    Create multimodal content for vision-enabled LLM calls.
    
    Args:
        text: Text content
        images: List of base64-encoded image data URIs
        image_detail: Image detail level ("auto", "low", "high")
        
    Returns:
        List of content objects suitable for message content
        
    Example:
        >>> content = create_multimodal_content(
        ...     "What's in this image?",
        ...     images=["data:image/jpeg;base64,/9j/4AAQ..."]
        ... )
    """
    content: List[Dict[str, Any]] = []
    
    # Add text content
    if text:
        content.append({"type": "text", "text": text})
    
    # Add image content
    if images:
        for img_data in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": img_data,
                    "detail": image_detail
                }
            })
    
    return content


# ==============================================================================
# Convenience functions for common patterns
# ==============================================================================

def get_model_provider() -> ModelProvider:
    """
    Get a ModelProvider instance with global settings.
    
    Returns:
        ModelProvider: Configured model provider instance
    """
    return ModelProvider()
