"""
Update Manager for Eidolon AI Personal Assistant

Provides safe update capabilities with automatic rollback on failure.
"""

import os
import sys
import json
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
import packaging.version
from dataclasses import dataclass

from ..utils.config import get_config
from ..utils.logging import get_logger
from .version_manager import VersionManager
from .backup_manager import BackupManager

logger = get_logger(__name__)


@dataclass
class UpdateInfo:
    """Information about an available update."""
    current_version: str
    latest_version: str
    release_notes: str
    download_url: str
    size_mb: float
    is_critical: bool = False
    requires_restart: bool = True


@dataclass
class UpdateResult:
    """Result of an update operation."""
    success: bool
    version: str
    message: str
    rollback_available: bool = False
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class UpdateManager:
    """Manages Eidolon updates with safe rollback capabilities."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.eidolon_home = Path(os.getenv('EIDOLON_HOME', '~/.eidolon')).expanduser()
        self.version_manager = VersionManager()
        self.backup_manager = BackupManager()
        
        # Update configuration
        self.update_config = self.config.get('updates', {})
        self.auto_update = self.update_config.get('auto_update', False)
        self.check_interval_hours = self.update_config.get('check_interval_hours', 24)
        self.download_timeout = self.update_config.get('download_timeout', 300)
        self.backup_before_update = self.update_config.get('backup_before_update', True)
        
        # Paths
        self.update_dir = self.eidolon_home / 'updates'
        self.update_dir.mkdir(exist_ok=True, parents=True)
        self.last_check_file = self.update_dir / '.last_check'
        self.update_lock_file = self.update_dir / '.update_lock'
        
    def check_for_updates(self, force: bool = False) -> Optional[UpdateInfo]:
        """
        Check for available updates.
        
        Args:
            force: Force check even if recently checked
            
        Returns:
            UpdateInfo if update available, None otherwise
        """
        try:
            # Check if we should skip this check
            if not force and not self._should_check_updates():
                logger.debug("Skipping update check - too recent")
                return None
                
            logger.info("Checking for updates...")
            
            # Get current version
            current_version = self.version_manager.get_current_version()
            
            # Check PyPI for latest version
            latest_info = self._get_latest_version_info()
            if not latest_info:
                logger.warning("Could not retrieve version information")
                return None
                
            latest_version = latest_info['version']
            
            # Update last check time
            self._update_last_check()
            
            # Compare versions
            if packaging.version.parse(latest_version) > packaging.version.parse(current_version):
                logger.info(f"Update available: {current_version} -> {latest_version}")
                
                return UpdateInfo(
                    current_version=current_version,
                    latest_version=latest_version,
                    release_notes=latest_info.get('description', ''),
                    download_url=self._get_download_url(latest_version),
                    size_mb=latest_info.get('size_mb', 0),
                    is_critical=self._is_critical_update(latest_version),
                    requires_restart=True
                )
            else:
                logger.info(f"Already up to date: {current_version}")
                return None
                
        except Exception as e:
            logger.error(f"Error checking for updates: {e}")
            return None
            
    def download_update(self, update_info: UpdateInfo) -> bool:
        """
        Download an update package.
        
        Args:
            update_info: Information about the update
            
        Returns:
            True if download successful
        """
        try:
            logger.info(f"Downloading update {update_info.latest_version}...")
            
            # Create download directory
            download_dir = self.update_dir / update_info.latest_version
            download_dir.mkdir(exist_ok=True, parents=True)
            
            # Download using pip
            cmd = [
                sys.executable, '-m', 'pip', 'download',
                f'eidolon=={update_info.latest_version}',
                '--dest', str(download_dir),
                '--no-deps'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.download_timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to download update: {result.stderr}")
                return False
                
            logger.info(f"Successfully downloaded update {update_info.latest_version}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading update: {e}")
            return False
            
    def install_update(self, update_info: UpdateInfo, backup: bool = None) -> UpdateResult:
        """
        Install an update with automatic rollback on failure.
        
        Args:
            update_info: Information about the update
            backup: Whether to create backup (defaults to config setting)
            
        Returns:
            UpdateResult with operation details
        """
        if backup is None:
            backup = self.backup_before_update
            
        # Check for update lock
        if self.update_lock_file.exists():
            return UpdateResult(
                success=False,
                version=update_info.latest_version,
                message="Another update is already in progress",
                errors=["Update lock file exists"]
            )
            
        backup_id = None
        
        try:
            # Create update lock
            self.update_lock_file.touch()
            
            logger.info(f"Installing update {update_info.latest_version}...")
            
            # Create backup if requested
            if backup:
                logger.info("Creating backup before update...")
                backup_result = self.backup_manager.create_backup(
                    backup_type='pre_update',
                    version=update_info.current_version
                )
                if backup_result.success:
                    backup_id = backup_result.backup_id
                    logger.info(f"Backup created: {backup_id}")
                else:
                    logger.warning(f"Backup failed: {backup_result.message}")
                    
            # Stop Eidolon service
            self._stop_service()
            
            # Install update
            install_result = self._install_package(update_info.latest_version)
            
            if install_result.success:
                # Update version info
                self.version_manager.set_current_version(update_info.latest_version)
                
                # Start service
                self._start_service()
                
                # Verify installation
                if self._verify_installation(update_info.latest_version):
                    logger.info(f"Successfully updated to {update_info.latest_version}")
                    return UpdateResult(
                        success=True,
                        version=update_info.latest_version,
                        message=f"Successfully updated to {update_info.latest_version}",
                        rollback_available=backup_id is not None
                    )
                else:
                    logger.error("Installation verification failed")
                    # Auto-rollback on verification failure
                    if backup_id:
                        rollback_result = self.rollback_update(backup_id)
                        return UpdateResult(
                            success=False,
                            version=update_info.latest_version,
                            message=f"Update failed verification, rolled back: {rollback_result.message}",
                            rollback_available=False,
                            errors=["Installation verification failed"]
                        )
                    else:
                        return UpdateResult(
                            success=False,
                            version=update_info.latest_version,
                            message="Update failed verification and no backup available",
                            rollback_available=False,
                            errors=["Installation verification failed", "No backup available"]
                        )
            else:
                logger.error(f"Installation failed: {install_result.message}")
                return UpdateResult(
                    success=False,
                    version=update_info.latest_version,
                    message=f"Installation failed: {install_result.message}",
                    rollback_available=backup_id is not None,
                    errors=install_result.errors
                )
                
        except Exception as e:
            logger.error(f"Error installing update: {e}")
            
            # Auto-rollback on error
            if backup_id:
                try:
                    rollback_result = self.rollback_update(backup_id)
                    return UpdateResult(
                        success=False,
                        version=update_info.latest_version,
                        message=f"Update failed, rolled back: {rollback_result.message}",
                        rollback_available=False,
                        errors=[str(e)]
                    )
                except Exception as rollback_error:
                    logger.error(f"Rollback also failed: {rollback_error}")
                    
            return UpdateResult(
                success=False,
                version=update_info.latest_version,
                message=f"Update failed: {e}",
                rollback_available=backup_id is not None,
                errors=[str(e)]
            )
            
        finally:
            # Remove update lock
            if self.update_lock_file.exists():
                self.update_lock_file.unlink()
                
    def rollback_update(self, backup_id: str) -> UpdateResult:
        """
        Rollback to a previous version using a backup.
        
        Args:
            backup_id: ID of the backup to restore
            
        Returns:
            UpdateResult with operation details
        """
        try:
            logger.info(f"Rolling back using backup {backup_id}...")
            
            # Stop service
            self._stop_service()
            
            # Restore from backup
            restore_result = self.backup_manager.restore_backup(backup_id)
            
            if restore_result.success:
                # Get version from backup metadata
                backup_info = self.backup_manager.get_backup_info(backup_id)
                if backup_info:
                    restored_version = backup_info.get('version', 'unknown')
                    self.version_manager.set_current_version(restored_version)
                    
                # Start service
                self._start_service()
                
                logger.info(f"Successfully rolled back to backup {backup_id}")
                return UpdateResult(
                    success=True,
                    version=restored_version,
                    message=f"Successfully rolled back to {restored_version}",
                    rollback_available=False
                )
            else:
                logger.error(f"Rollback failed: {restore_result.message}")
                return UpdateResult(
                    success=False,
                    version="unknown",
                    message=f"Rollback failed: {restore_result.message}",
                    rollback_available=False,
                    errors=[restore_result.message]
                )
                
        except Exception as e:
            logger.error(f"Error during rollback: {e}")
            return UpdateResult(
                success=False,
                version="unknown",
                message=f"Rollback error: {e}",
                rollback_available=False,
                errors=[str(e)]
            )
            
    def list_available_rollbacks(self) -> List[Dict]:
        """
        List available rollback points.
        
        Returns:
            List of backup information dictionaries
        """
        return self.backup_manager.list_backups(backup_type='pre_update')
        
    def cleanup_old_updates(self, keep_count: int = 3) -> None:
        """
        Clean up old update files and backups.
        
        Args:
            keep_count: Number of recent items to keep
        """
        try:
            logger.info("Cleaning up old update files...")
            
            # Clean up update downloads
            update_dirs = []
            for item in self.update_dir.iterdir():
                if item.is_dir() and item.name != '.':
                    update_dirs.append(item)
                    
            # Sort by modification time and remove oldest
            update_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            for old_dir in update_dirs[keep_count:]:
                shutil.rmtree(old_dir, ignore_errors=True)
                logger.debug(f"Removed old update directory: {old_dir}")
                
            # Clean up old backups
            self.backup_manager.cleanup_old_backups(
                backup_type='pre_update',
                keep_count=keep_count
            )
            
            logger.info("Update cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            
    def _should_check_updates(self) -> bool:
        """Check if we should check for updates based on last check time."""
        if not self.last_check_file.exists():
            return True
            
        try:
            last_check = datetime.fromtimestamp(self.last_check_file.stat().st_mtime)
            now = datetime.now()
            hours_since_check = (now - last_check).total_seconds() / 3600
            return hours_since_check >= self.check_interval_hours
        except Exception:
            return True
            
    def _update_last_check(self) -> None:
        """Update the last check timestamp."""
        self.last_check_file.touch()
        
    def _get_latest_version_info(self) -> Optional[Dict]:
        """Get latest version information from PyPI."""
        try:
            response = requests.get(
                'https://pypi.org/pypi/eidolon/json',
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            latest_version = data['info']['version']
            
            return {
                'version': latest_version,
                'description': data['info']['summary'],
                'size_mb': 50  # Estimated size
            }
        except Exception as e:
            logger.error(f"Error fetching version info: {e}")
            return None
            
    def _get_download_url(self, version: str) -> str:
        """Get download URL for a specific version."""
        return f"https://pypi.org/project/eidolon/{version}/"
        
    def _is_critical_update(self, version: str) -> bool:
        """Determine if an update is critical (security/stability)."""
        # This could be enhanced to check release notes or security advisories
        return False
        
    def _install_package(self, version: str) -> UpdateResult:
        """Install a specific version of Eidolon."""
        try:
            # Get virtual environment path
            venv_path = self.eidolon_home / 'venv'
            pip_path = venv_path / 'bin' / 'pip'
            
            if not pip_path.exists():
                pip_path = venv_path / 'Scripts' / 'pip.exe'  # Windows
                
            cmd = [
                str(pip_path), 'install', 
                f'eidolon=={version}',
                '--upgrade'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                return UpdateResult(
                    success=True,
                    version=version,
                    message=f"Successfully installed version {version}"
                )
            else:
                return UpdateResult(
                    success=False,
                    version=version,
                    message="Installation failed",
                    errors=[result.stderr]
                )
                
        except Exception as e:
            return UpdateResult(
                success=False,
                version=version,
                message=f"Installation error: {e}",
                errors=[str(e)]
            )
            
    def _verify_installation(self, version: str) -> bool:
        """Verify that the installation was successful."""
        try:
            # Try to import and check version
            venv_path = self.eidolon_home / 'venv'
            python_path = venv_path / 'bin' / 'python'
            
            if not python_path.exists():
                python_path = venv_path / 'Scripts' / 'python.exe'  # Windows
                
            cmd = [
                str(python_path), '-c',
                'import eidolon; print(eidolon.__version__)'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                installed_version = result.stdout.strip()
                return installed_version == version
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error verifying installation: {e}")
            return False
            
    def _stop_service(self) -> bool:
        """Stop the Eidolon service."""
        try:
            manage_script = self.eidolon_home / 'manage-service.sh'
            if manage_script.exists():
                subprocess.run([str(manage_script), 'stop'], timeout=30)
            return True
        except Exception as e:
            logger.warning(f"Could not stop service: {e}")
            return False
            
    def _start_service(self) -> bool:
        """Start the Eidolon service."""
        try:
            manage_script = self.eidolon_home / 'manage-service.sh'
            if manage_script.exists():
                subprocess.run([str(manage_script), 'start'], timeout=30)
            return True
        except Exception as e:
            logger.warning(f"Could not start service: {e}")
            return False