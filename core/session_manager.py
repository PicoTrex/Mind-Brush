"""
Session Manager
===============
Manages timestamped session directories for storing run artifacts.
Supports lazy directory creation and step logging.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class SessionManager:
    """
    Manages session-specific storage directories.
    
    Each session gets a timestamped directory under storage/
    to organize logs, uploads, and intermediate results.
    
    Supports lazy creation - directories are only created when
    the user actually sends a message.
    """
    
    def __init__(self, storage_root: str = "./storage"):
        self.storage_root = Path(storage_root).absolute()
        self.storage_root.mkdir(parents=True, exist_ok=True)
        
        self._session_dir: Optional[Path] = None
        self._session_id: Optional[str] = None
        self._dirs_created: bool = False
    
    def prepare_session(self) -> str:
        """
        Prepare session ID without creating directories.
        Called on chat start.
        
        Returns:
            Session ID (timestamp string)
        """
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._dirs_created = False
        return self._session_id
    
    def ensure_session_dirs(self) -> Path:
        """
        Create session directories if not already created.
        Called lazily when user sends first message.
        
        Returns:
            Session directory path
        """
        if self._dirs_created and self._session_dir:
            return self._session_dir
            
        if not self._session_id:
            self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self._session_dir = self.storage_root / "sessions" / self._session_id
        
        # Create subdirectories (no results folder)
        (self._session_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self._session_dir / "uploads").mkdir(parents=True, exist_ok=True)
        (self._session_dir / "temp" / "image_rag").mkdir(parents=True, exist_ok=True)
        (self._session_dir / "temp" / "image_gen").mkdir(parents=True, exist_ok=True)
        (self._session_dir / "temp" / "text_rag").mkdir(parents=True, exist_ok=True)
        
        self._dirs_created = True
        return self._session_dir
    
    def create_session(self) -> str:
        """
        Legacy method - creates session with directories immediately.
        For backwards compatibility.
        
        Returns:
            Session ID (timestamp string)
        """
        self.prepare_session()
        self.ensure_session_dirs()
        return self._session_id
    
    def get_session_dir(self) -> Path:
        """Get the current session directory."""
        if not self._session_dir:
            raise RuntimeError("No active session. Call ensure_session_dirs() first.")
        return self._session_dir
    
    def get_upload_dir(self) -> Path:
        """Get the uploads directory for current session."""
        return self.get_session_dir() / "uploads"
    
    def get_temp_dir(self, subdir: str = "") -> Path:
        """Get a temp directory for current session."""
        if subdir:
            return self.get_session_dir() / "temp" / subdir
        return self.get_session_dir() / "temp"
    
    def get_log_dir(self) -> Path:
        """Get the logs directory for current session."""
        return self.get_session_dir() / "logs"
    
    def write_step_log(self, step_name: str, content: Dict[str, Any]) -> Path:
        """
        Write step result to log file.
        
        Args:
            step_name: Name of the step
            content: Step data to log (input, output, status, etc.)
            
        Returns:
            Path to the log file
        """
        log_dir = self.get_log_dir()
        
        # Sanitize step name for filename
        safe_name = step_name.replace(" ", "_").replace("&", "and").lower()
        timestamp = datetime.now().strftime("%H%M%S")
        log_file = log_dir / f"{safe_name}_{timestamp}.json"
        
        # Add timestamp to content
        content["logged_at"] = datetime.now().isoformat()
        
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2, ensure_ascii=False, default=str)
        
        return log_file
    
    @property
    def session_id(self) -> str:
        """Get current session ID."""
        if not self._session_id:
            raise RuntimeError("No active session.")
        return self._session_id
    
    @property
    def is_initialized(self) -> bool:
        """Check if session directories have been created."""
        return self._dirs_created


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get or create the global session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
