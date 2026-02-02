"""
Configuration Loader
====================
Type-safe configuration management using Pydantic models.
Loads settings from YAML files and environment variables.
"""

import os
import yaml
from pathlib import Path
from functools import lru_cache
from typing import Dict, Optional, Any, List
from pydantic import BaseModel, Field


# ==============================================================================
# Configuration Models
# ==============================================================================

class ProxySettings(BaseModel):
    """Proxy configuration for network requests."""
    proxy_on: bool = Field(default=False, description="Enable/disable proxy")
    http_proxy: str = Field(default="http://127.0.0.1:7890", alias="HTTP_PROXY")
    https_proxy: str = Field(default="http://127.0.0.1:7890", alias="HTTPS_PROXY")

    class Config:
        populate_by_name = True


class LLMSettings(BaseModel):
    """LLM API configuration."""
    provider: str = Field(default="openai", alias="PROVIDER")
    api_key: str = Field(default="", alias="OPENAI_API_KEY")
    base_url: str = Field(default="https://api.openai.com/v1", alias="OPENAI_BASE_URL")
    model_name: str = Field(default="gpt-4", alias="OPENAI_MODEL_NAME")

    class Config:
        populate_by_name = True


class ImageGenSettings(BaseModel):
    """Image generation API configuration."""
    api_key: str = Field(default="", alias="IMAGE_API_KEY")
    base_url: str = Field(default="https://api.openai.com/v1", alias="IMAGE_BASE_URL")
    gen_model: str = Field(default="dall-e-3", alias="IMAGE_GEN_MODEL_NAME")
    edit_model: str = Field(default="dall-e-2", alias="IMAGE_EDIT_MODEL_NAME")

    class Config:
        populate_by_name = True


class FluxSettings(BaseModel):
    """FLUX API configuration."""
    api_key: str = Field(default="", alias="FLUX_API_KEY")
    submit_url: str = Field(
        default="https://api.us1.bfl.ai/v1/flux-kontext-pro",
        alias="FLUX_API_URL_SUBMIT"
    )
    result_url: str = Field(
        default="https://api.us1.bfl.ai/v1/get_result",
        alias="FLUX_API_URL_RESULT"
    )

    class Config:
        populate_by_name = True


class SearchSettings(BaseModel):
    """Search API configuration (Google, Serper)."""
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    google_search_engine_id: str = Field(default="", alias="GOOGLE_SEARCH_ENGINE_ID")
    serper_api_key: str = Field(default="", alias="SERPER_API_KEY")

    class Config:
        populate_by_name = True


class ImageSearchSettings(BaseModel):
    """Image search API configuration."""
    provider: str = Field(default="auto", description="Image search provider: auto, google, or serper")
    priority: List[str] = Field(default=["serper", "google"], description="API priority order for auto mode")
    num_images: int = Field(default=5, description="Number of images to fetch per query")


class TempDirSettings(BaseModel):
    """Temporary directory configuration."""
    default: str = Field(default="./temp")
    image_gen: str = Field(default="./temp/image_gen")
    image_rag: str = Field(default="./temp/image_rag")
    text_rag: str = Field(default="./temp/text_rag")


class MCPServerConfig(BaseModel):
    """Individual MCP server configuration."""
    command: str = Field(default="python")
    path: str


class Settings(BaseModel):
    """
    Root configuration model.
    Aggregates all sub-configurations into a single settings object.
    """
    # API Settings
    llm: LLMSettings = Field(default_factory=LLMSettings)
    image_gen: ImageGenSettings = Field(default_factory=ImageGenSettings)
    flux: FluxSettings = Field(default_factory=FluxSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    image_search: ImageSearchSettings = Field(default_factory=ImageSearchSettings)
    
    # Infrastructure Settings
    proxy: ProxySettings = Field(default_factory=ProxySettings)
    temp_dir: TempDirSettings = Field(default_factory=TempDirSettings)
    
    # MCP Servers
    mcp_servers: Dict[str, MCPServerConfig] = Field(default_factory=dict)

    def setup_proxy(self) -> None:
        """Apply proxy settings to environment variables."""
        if self.proxy.proxy_on:
            os.environ["http_proxy"] = self.proxy.http_proxy
            os.environ["https_proxy"] = self.proxy.https_proxy
        else:
            # Clear proxy if disabled
            os.environ.pop("http_proxy", None)
            os.environ.pop("https_proxy", None)

    def get_temp_dir(self, name: str = "default") -> Path:
        """
        Get and create a temporary directory by name.
        
        Args:
            name: Directory name (default, image_gen, image_rag, text_rag)
            
        Returns:
            Absolute path to the temporary directory
        """
        dir_map = {
            "default": self.temp_dir.default,
            "image_gen": self.temp_dir.image_gen,
            "image_rag": self.temp_dir.image_rag,
            "text_rag": self.temp_dir.text_rag,
        }
        path = Path(dir_map.get(name, self.temp_dir.default)).absolute()
        path.mkdir(parents=True, exist_ok=True)
        return path


# ==============================================================================
# Configuration Loading Functions
# ==============================================================================

def _load_yaml_file(file_path: Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dictionary."""
    if not file_path.exists():
        return {}
    
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _find_project_root() -> Path:
    """
    Find the project root directory.
    Walks up from the current file until it finds a marker file.
    """
    current = Path(__file__).resolve().parent
    
    # Walk up to find project root (contains app.py or config.yaml)
    for _ in range(5):  # Max 5 levels up
        if (current / "app.py").exists() or (current / "config.yaml").exists():
            return current
        current = current.parent
    
    # Fallback to current working directory
    return Path.cwd()


def _load_all_configs(project_root: Path) -> Dict[str, Any]:
    """
    Load and merge all configuration files.
    
    Priority (later overrides earlier):
    1. configs/mcp_server.yaml
    2. configs/apis.yaml  
    3. config.yaml (root level, for backward compatibility)
    """
    merged_config: Dict[str, Any] = {}
    
    # Load from configs directory
    configs_dir = project_root / "configs"
    
    if configs_dir.exists():
        # Load MCP server config
        mcp_config = _load_yaml_file(configs_dir / "mcp_server.yaml")
        merged_config.update(mcp_config)
        
        # Load API config
        api_config = _load_yaml_file(configs_dir / "apis.yaml")
        merged_config.update(api_config)
    
    # Load root config.yaml (backward compatibility, highest priority)
    root_config = _load_yaml_file(project_root / "config.yaml")
    merged_config.update(root_config)
    
    return merged_config


def _build_settings(raw_config: Dict[str, Any]) -> Settings:
    """
    Build a Settings object from raw configuration dictionary.
    Handles nested and flat configuration structures.
    """
    # Build LLM settings
    llm_settings = LLMSettings(
        provider=raw_config.get("PROVIDER", "openai"),
        api_key=raw_config.get("OPENAI_API_KEY", ""),
        base_url=raw_config.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        model_name=raw_config.get("OPENAI_MODEL_NAME", "gpt-4"),
    )
    
    # Build Image Gen settings
    image_gen_settings = ImageGenSettings(
        api_key=raw_config.get("IMAGE_API_KEY", raw_config.get("OPENAI_API_KEY", "")),
        base_url=raw_config.get("IMAGE_BASE_URL", raw_config.get("OPENAI_BASE_URL", "")),
        gen_model=raw_config.get("IMAGE_GEN_MODEL_NAME", "dall-e-3"),
        edit_model=raw_config.get("IMAGE_EDIT_MODEL_NAME", "dall-e-2"),
    )
    
    # Build Flux settings
    flux_settings = FluxSettings(
        api_key=raw_config.get("FLUX_API_KEY", ""),
        submit_url=raw_config.get("FLUX_API_URL_SUBMIT", ""),
        result_url=raw_config.get("FLUX_API_URL_RESULT", ""),
    )
    
    # Build Search settings
    search_settings = SearchSettings(
        google_api_key=raw_config.get("GOOGLE_API_KEY", ""),
        google_search_engine_id=raw_config.get("GOOGLE_SEARCH_ENGINE_ID", ""),
        serper_api_key=raw_config.get("SERPER_API_KEY", ""),
    )
    
    # Build Proxy settings
    proxy_config = raw_config.get("proxy", {})
    proxy_settings = ProxySettings(
        proxy_on=proxy_config.get("proxy_on", raw_config.get("proxy_on", False)),
        http_proxy=proxy_config.get("HTTP_PROXY", raw_config.get("HTTP_PROXY", "http://127.0.0.1:7890")),
        https_proxy=proxy_config.get("HTTPS_PROXY", raw_config.get("HTTPS_PROXY", "http://127.0.0.1:7890")),
    )
    
    # Build ImageSearch settings
    image_search_config = raw_config.get("image_search", {})
    if isinstance(image_search_config, dict):
        image_search_settings = ImageSearchSettings(
            provider=image_search_config.get("provider", "auto"),
            priority=image_search_config.get("priority", ["serper", "google"]),
            num_images=image_search_config.get("num_images", 5),
        )
    else:
        image_search_settings = ImageSearchSettings()
    
    # Build TempDir settings
    temp_config = raw_config.get("temp_dir", {})
    if isinstance(temp_config, dict):
        temp_dir_settings = TempDirSettings(
            default=temp_config.get("default", "./temp"),
            image_gen=temp_config.get("image_gen", "./temp/image_gen"),
            image_rag=temp_config.get("image_rag", "./temp/image_rag"),
            text_rag=temp_config.get("text_rag", "./temp/text_rag"),
        )
    else:
        temp_dir_settings = TempDirSettings()
    
    # Build MCP Servers settings
    mcp_servers: Dict[str, MCPServerConfig] = {}
    mcp_config = raw_config.get("mcp_servers", {})
    for name, server_config in mcp_config.items():
        if isinstance(server_config, dict):
            mcp_servers[name] = MCPServerConfig(
                command=server_config.get("command", "python"),
                path=server_config.get("path", ""),
            )
    
    return Settings(
        llm=llm_settings,
        image_gen=image_gen_settings,
        flux=flux_settings,
        search=search_settings,
        image_search=image_search_settings,
        proxy=proxy_settings,
        temp_dir=temp_dir_settings,
        mcp_servers=mcp_servers,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get the global settings instance (singleton).
    
    Uses LRU cache to ensure only one Settings instance is created.
    The settings are loaded from YAML files on first call.
    
    Returns:
        Settings: The global configuration object
        
    Example:
        >>> settings = get_settings()
        >>> settings.setup_proxy()
        >>> print(settings.llm.model_name)
    """
    project_root = _find_project_root()
    raw_config = _load_all_configs(project_root)
    settings = _build_settings(raw_config)
    
    # Auto-setup proxy on load
    settings.setup_proxy()
    
    return settings


def reload_settings() -> Settings:
    """
    Force reload settings from configuration files.
    Clears the LRU cache and reloads all configurations.
    
    Returns:
        Settings: Freshly loaded configuration object
    """
    get_settings.cache_clear()
    return get_settings()


# ==============================================================================
# Prompt Loading Utility
# ==============================================================================

def load_prompt(prompt_name: str) -> str:
    """
    Load a system prompt from the prompts directory.
    
    Args:
        prompt_name: Name of the prompt file (without extension)
        
    Returns:
        The system_prompt content from the YAML file
        
    Example:
        >>> prompt = load_prompt("intent_analysis")
    """
    project_root = _find_project_root()
    prompt_file = project_root / "prompts" / f"{prompt_name}.yaml"
    
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    
    with open(prompt_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data.get("system_prompt", "")


# ==============================================================================
# Module-level convenience
# ==============================================================================

# Pre-load settings when module is imported (optional optimization)
# Uncomment if you want eager loading:
# _settings = get_settings()
