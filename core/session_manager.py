"""
Session Manager
===============
Manages timestamped session directories for storing run artifacts.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class SessionManager:
    """
    Manages session-specific storage directories.
    
    Each session gets a timestamped directory under storage/
    to organize logs, uploads, and intermediate results.
    """
    
    def __init__(self, storage_root: str = "./storage"):
        self.storage_root = Path(storage_root).absolute()
        self.storage_root.mkdir(parents=True, exist_ok=True)
        
        self._session_dir: Optional[Path] = None
        self._session_id: Optional[str] = None
    
    def create_session(self) -> str:
        """
        Create a new session directory with timestamp.
        
        Returns:
            Session ID (timestamp string)
        """
        # Generate session ID from timestamp: YYYYMMDD_HHMM
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = self.storage_root / "sessions" / self._session_id
        
        # Create subdirectories
        (self._session_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self._session_dir / "uploads").mkdir(parents=True, exist_ok=True)
        (self._session_dir / "temp" / "image_rag").mkdir(parents=True, exist_ok=True)
        (self._session_dir / "temp" / "image_gen").mkdir(parents=True, exist_ok=True)
        (self._session_dir / "temp" / "text_rag").mkdir(parents=True, exist_ok=True)
        (self._session_dir / "results").mkdir(parents=True, exist_ok=True)
        
        return self._session_id
    
    def get_session_dir(self) -> Path:
        """Get the current session directory."""
        if not self._session_dir:
            raise RuntimeError("No active session. Call create_session() first.")
        return self._session_dir
    
    def get_upload_dir(self) -> Path:
        """Get the uploads directory for current session."""
        return self.get_session_dir() / "uploads"
    
    def get_temp_dir(self, subdir: str = "") -> Path:
        """Get a temp directory for current session."""
        if subdir:
            return self.get_session_dir() / "temp" / subdir
        return self.get_session_dir() / "temp"
    
    def get_results_dir(self) -> Path:
        """Get the results directory for current session."""
        return self.get_session_dir() / "results"
    
    def get_log_dir(self) -> Path:
        """Get the logs directory for current session."""
        return self.get_session_dir() / "logs"
    
    @property
    def session_id(self) -> str:
        """Get current session ID."""
        if not self._session_id:
            raise RuntimeError("No active session.")
        return self._session_id


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get or create the global session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
