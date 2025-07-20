"""
Version Manager for Eidolon AI Personal Assistant

Handles version tracking and management.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import subprocess
import sys

from ..utils.logging import get_logger

logger = get_logger(__name__)


class VersionManager:
    """Manages version information and history."""
    
    def __init__(self):
        self.eidolon_home = Path(os.getenv('EIDOLON_HOME', '~/.eidolon')).expanduser()
        self.version_file = self.eidolon_home / '.version'
        self.version_history_file = self.eidolon_home / '.version_history'
        
        # Ensure directory exists
        self.eidolon_home.mkdir(exist_ok=True, parents=True)
        
    def get_current_version(self) -> str:
        """
        Get the currently installed version.
        
        Returns:
            Version string
        """
        try:
            # First try to get from package
            version = self._get_package_version()
            if version:
                return version
                
            # Fall back to version file
            if self.version_file.exists():
                return self.version_file.read_text().strip()
                
            return "unknown"
            
        except Exception as e:
            logger.error(f"Error getting current version: {e}")
            return "unknown"
            
    def set_current_version(self, version: str) -> None:
        """
        Set the current version and update history.
        
        Args:
            version: Version string to set
        """
        try:
            # Update version file
            self.version_file.write_text(version)
            
            # Update version history
            self._add_to_history(version)
            
            logger.info(f"Set current version to {version}")
            
        except Exception as e:
            logger.error(f"Error setting version: {e}")
            
    def get_version_history(self) -> List[Dict]:
        """
        Get version installation history.
        
        Returns:
            List of version history entries
        """
        try:
            if not self.version_history_file.exists():
                return []
                
            history_data = json.loads(self.version_history_file.read_text())
            return history_data.get('versions', [])
            
        except Exception as e:
            logger.error(f"Error reading version history: {e}")
            return []
            
    def get_version_info(self, version: str) -> Optional[Dict]:
        """
        Get detailed information about a specific version.
        
        Args:
            version: Version to get info for
            
        Returns:
            Version information dictionary or None
        """
        history = self.get_version_history()
        for entry in history:
            if entry.get('version') == version:
                return entry
        return None
        
    def is_version_installed(self, version: str) -> bool:
        """
        Check if a specific version was ever installed.
        
        Args:
            version: Version to check
            
        Returns:
            True if version was installed
        """
        return self.get_version_info(version) is not None
        
    def get_latest_installed_version(self) -> Optional[str]:
        """
        Get the most recently installed version from history.
        
        Returns:
            Latest version string or None
        """
        history = self.get_version_history()
        if history:
            # History is sorted by install date, latest first
            return history[0].get('version')
        return None
        
    def get_previous_version(self) -> Optional[str]:
        """
        Get the version that was installed before the current one.
        
        Returns:
            Previous version string or None
        """
        history = self.get_version_history()
        if len(history) >= 2:
            return history[1].get('version')
        return None
        
    def cleanup_version_history(self, keep_count: int = 10) -> None:
        """
        Clean up old version history entries.
        
        Args:
            keep_count: Number of recent entries to keep
        """
        try:
            history = self.get_version_history()
            if len(history) > keep_count:
                # Keep only the most recent entries
                cleaned_history = history[:keep_count]
                
                history_data = {
                    'versions': cleaned_history,
                    'last_cleanup': datetime.now().isoformat()
                }
                
                self.version_history_file.write_text(json.dumps(history_data, indent=2))
                logger.info(f"Cleaned version history, kept {keep_count} entries")
                
        except Exception as e:
            logger.error(f"Error cleaning version history: {e}")
            
    def export_version_info(self) -> Dict:
        """
        Export complete version information.
        
        Returns:
            Dictionary with all version information
        """
        return {
            'current_version': self.get_current_version(),
            'package_version': self._get_package_version(),
            'history': self.get_version_history(),
            'export_timestamp': datetime.now().isoformat()
        }
        
    def _get_package_version(self) -> Optional[str]:
        """Get version from the installed package."""
        try:
            # Try using importlib.metadata (Python 3.8+)
            try:
                from importlib.metadata import version
                return version('eidolon')
            except ImportError:
                # Fall back to pkg_resources
                import pkg_resources
                return pkg_resources.get_distribution('eidolon').version
                
        except Exception:
            # Try running pip show
            try:
                venv_path = self.eidolon_home / 'venv'
                pip_path = venv_path / 'bin' / 'pip'
                
                if not pip_path.exists():
                    pip_path = venv_path / 'Scripts' / 'pip.exe'  # Windows
                    
                result = subprocess.run(
                    [str(pip_path), 'show', 'eidolon'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if line.startswith('Version:'):
                            return line.split(':', 1)[1].strip()
                            
            except Exception:
                pass
                
            return None
            
    def _add_to_history(self, version: str) -> None:
        """Add a version to the installation history."""
        try:
            # Load existing history
            history_data = {'versions': []}
            if self.version_history_file.exists():
                try:
                    history_data = json.loads(self.version_history_file.read_text())
                except json.JSONDecodeError:
                    logger.warning("Invalid version history file, creating new one")
                    
            versions = history_data.get('versions', [])
            
            # Check if this version is already the latest entry
            if versions and versions[0].get('version') == version:
                return
                
            # Create new entry
            new_entry = {
                'version': version,
                'install_date': datetime.now().isoformat(),
                'install_method': 'update_manager',
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            }
            
            # Add to front of list (most recent first)
            versions.insert(0, new_entry)
            
            # Limit history size
            max_history = 50
            if len(versions) > max_history:
                versions = versions[:max_history]
                
            # Save updated history
            history_data['versions'] = versions
            history_data['last_updated'] = datetime.now().isoformat()
            
            self.version_history_file.write_text(json.dumps(history_data, indent=2))
            
        except Exception as e:
            logger.error(f"Error adding to version history: {e}")


def get_version_info() -> Dict:
    """
    Convenience function to get current version information.
    
    Returns:
        Dictionary with version information
    """
    manager = VersionManager()
    return {
        'current': manager.get_current_version(),
        'previous': manager.get_previous_version(),
        'package': manager._get_package_version()
    }