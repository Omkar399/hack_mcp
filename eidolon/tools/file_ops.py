"""
File Operations Tool for Eidolon Tool Orchestration Framework

Provides file system operations with safety controls.
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import hashlib
import tempfile

from .base import BaseTool, ToolMetadata, ToolResult, ToolError
from ..core.safety import RiskLevel, ActionCategory
from ..utils.logging import get_component_logger

logger = get_component_logger("tools.file_ops")


class FileOperationsTool(BaseTool):
    """Tool for safe file system operations."""
    
    METADATA = ToolMetadata(
        name="file_operations",
        description="Perform file system operations with safety controls",
        category="system",
        risk_level=RiskLevel.MEDIUM,
        action_category=ActionCategory.WRITE_FILE,
        requires_approval=True,
        timeout_seconds=30.0,
        input_schema={
            "required": ["operation"],
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read", "write", "copy", "move", "delete", "list", "create_dir", "stat"]
                },
                "path": {"type": "string"},
                "content": {"type": "string"},
                "destination": {"type": "string"},
                "encoding": {"type": "string", "default": "utf-8"},
                "backup": {"type": "boolean", "default": True},
                "recursive": {"type": "boolean", "default": False}
            }
        }
    )
    
    def __init__(self, metadata: Optional[ToolMetadata] = None):
        """Initialize file operations tool."""
        super().__init__(metadata or self.METADATA)
        
        # Safety limits
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.allowed_extensions = {
            '.txt', '.md', '.json', '.yaml', '.yml', '.csv', '.log',
            '.py', '.js', '.html', '.css', '.xml', '.ini', '.conf'
        }
        self.forbidden_paths = {
            '/etc/passwd', '/etc/shadow', '/etc/hosts',
            '/System', '/usr/bin', '/usr/sbin'
        }
        
        # Backup directory
        self.backup_dir = Path(tempfile.gettempdir()) / "eidolon_backups"
        self.backup_dir.mkdir(exist_ok=True)
    
    async def execute(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ToolResult:
        """Execute file operation."""
        try:
            # Validate parameters
            validated_params = await self.validate_parameters(parameters)
            operation = validated_params["operation"]
            
            # Route to specific operation
            if operation == "read":
                return await self._read_file(validated_params)
            elif operation == "write":
                return await self._write_file(validated_params)
            elif operation == "copy":
                return await self._copy_file(validated_params)
            elif operation == "move":
                return await self._move_file(validated_params)
            elif operation == "delete":
                return await self._delete_file(validated_params)
            elif operation == "list":
                return await self._list_directory(validated_params)
            elif operation == "create_dir":
                return await self._create_directory(validated_params)
            elif operation == "stat":
                return await self._get_file_stats(validated_params)
            else:
                raise ToolError(f"Unknown operation: {operation}")
                
        except Exception as e:
            logger.error(f"File operation failed: {e}")
            return ToolResult(
                success=False,
                data={"error": str(e)},
                message=f"File operation failed: {str(e)}"
            )
    
    async def _read_file(self, params: Dict[str, Any]) -> ToolResult:
        """Read file contents."""
        file_path = Path(params["path"])
        encoding = params.get("encoding", "utf-8")
        
        # Safety checks
        if not self._is_safe_path(file_path):
            raise ToolError(f"Path not allowed: {file_path}")
        
        if not file_path.exists():
            raise ToolError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise ToolError(f"Path is not a file: {file_path}")
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > self.max_file_size:
            raise ToolError(f"File too large: {file_size} bytes (max: {self.max_file_size})")
        
        try:
            content = file_path.read_text(encoding=encoding)
            
            return ToolResult(
                success=True,
                data={
                    "content": content,
                    "path": str(file_path),
                    "size": file_size,
                    "encoding": encoding
                },
                message=f"Read {file_size} bytes from {file_path.name}"
            )
            
        except UnicodeDecodeError:
            # Try binary read for non-text files
            content = file_path.read_bytes()
            content_hash = hashlib.md5(content).hexdigest()
            
            return ToolResult(
                success=True,
                data={
                    "content": f"<binary data: {len(content)} bytes, MD5: {content_hash}>",
                    "path": str(file_path),
                    "size": len(content),
                    "encoding": "binary"
                },
                message=f"Read {len(content)} bytes (binary) from {file_path.name}"
            )
    
    async def _write_file(self, params: Dict[str, Any]) -> ToolResult:
        """Write content to file."""
        file_path = Path(params["path"])
        content = params.get("content", "")
        encoding = params.get("encoding", "utf-8")
        backup = params.get("backup", True)
        
        # Safety checks
        if not self._is_safe_path(file_path):
            raise ToolError(f"Path not allowed: {file_path}")
        
        if not self._is_allowed_extension(file_path):
            raise ToolError(f"File extension not allowed: {file_path.suffix}")
        
        # Create parent directories
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Backup existing file
        backup_path = None
        if backup and file_path.exists():
            backup_path = await self._create_backup(file_path)
        
        try:
            # Write content
            file_path.write_text(content, encoding=encoding)
            
            file_size = file_path.stat().st_size
            
            result_data = {
                "path": str(file_path),
                "size": file_size,
                "encoding": encoding
            }
            
            if backup_path:
                result_data["backup_path"] = str(backup_path)
            
            return ToolResult(
                success=True,
                data=result_data,
                message=f"Wrote {file_size} bytes to {file_path.name}",
                side_effects=[f"Modified file: {file_path}"]
            )
            
        except Exception as e:
            # Restore backup if write failed
            if backup_path and backup_path.exists():
                try:
                    shutil.copy2(backup_path, file_path)
                    logger.info(f"Restored backup: {file_path}")
                except Exception as restore_error:
                    logger.error(f"Failed to restore backup: {restore_error}")
            
            raise ToolError(f"Failed to write file: {e}")
    
    async def _copy_file(self, params: Dict[str, Any]) -> ToolResult:
        """Copy file or directory."""
        source_path = Path(params["path"])
        dest_path = Path(params["destination"])
        recursive = params.get("recursive", False)
        
        # Safety checks
        if not self._is_safe_path(source_path) or not self._is_safe_path(dest_path):
            raise ToolError("Source or destination path not allowed")
        
        if not source_path.exists():
            raise ToolError(f"Source not found: {source_path}")
        
        try:
            if source_path.is_file():
                # Copy file
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, dest_path)
                copied_size = dest_path.stat().st_size
                
            elif source_path.is_dir() and recursive:
                # Copy directory
                shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
                copied_size = sum(f.stat().st_size for f in dest_path.rglob('*') if f.is_file())
                
            else:
                raise ToolError(f"Cannot copy directory without recursive=True: {source_path}")
            
            return ToolResult(
                success=True,
                data={
                    "source": str(source_path),
                    "destination": str(dest_path),
                    "size": copied_size,
                    "type": "file" if source_path.is_file() else "directory"
                },
                message=f"Copied {source_path.name} to {dest_path.name}",
                side_effects=[f"Created copy: {dest_path}"]
            )
            
        except Exception as e:
            raise ToolError(f"Failed to copy: {e}")
    
    async def _move_file(self, params: Dict[str, Any]) -> ToolResult:
        """Move file or directory."""
        source_path = Path(params["path"])
        dest_path = Path(params["destination"])
        backup = params.get("backup", True)
        
        # Safety checks
        if not self._is_safe_path(source_path) or not self._is_safe_path(dest_path):
            raise ToolError("Source or destination path not allowed")
        
        if not source_path.exists():
            raise ToolError(f"Source not found: {source_path}")
        
        # Backup source if requested
        backup_path = None
        if backup:
            backup_path = await self._create_backup(source_path)
        
        try:
            # Create destination parent directory
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Move file/directory
            shutil.move(str(source_path), str(dest_path))
            
            result_data = {
                "source": str(source_path),
                "destination": str(dest_path),
                "type": "file" if dest_path.is_file() else "directory"
            }
            
            if backup_path:
                result_data["backup_path"] = str(backup_path)
            
            return ToolResult(
                success=True,
                data=result_data,
                message=f"Moved {source_path.name} to {dest_path.name}",
                side_effects=[f"Moved: {source_path} -> {dest_path}"]
            )
            
        except Exception as e:
            # Restore backup if move failed
            if backup_path and backup_path.exists() and not source_path.exists():
                try:
                    if backup_path.is_file():
                        shutil.copy2(backup_path, source_path)
                    else:
                        shutil.copytree(backup_path, source_path)
                    logger.info(f"Restored backup: {source_path}")
                except Exception as restore_error:
                    logger.error(f"Failed to restore backup: {restore_error}")
            
            raise ToolError(f"Failed to move: {e}")
    
    async def _delete_file(self, params: Dict[str, Any]) -> ToolResult:
        """Delete file or directory."""
        file_path = Path(params["path"])
        recursive = params.get("recursive", False)
        backup = params.get("backup", True)
        
        # Safety checks
        if not self._is_safe_path(file_path):
            raise ToolError(f"Path not allowed: {file_path}")
        
        if not file_path.exists():
            raise ToolError(f"Path not found: {file_path}")
        
        # Create backup before deletion
        backup_path = None
        if backup:
            backup_path = await self._create_backup(file_path)
        
        try:
            if file_path.is_file():
                file_size = file_path.stat().st_size
                file_path.unlink()
                deleted_type = "file"
                
            elif file_path.is_dir():
                if not recursive:
                    raise ToolError("Cannot delete directory without recursive=True")
                
                # Count files for reporting
                file_count = sum(1 for f in file_path.rglob('*') if f.is_file())
                shutil.rmtree(file_path)
                file_size = file_count
                deleted_type = "directory"
                
            else:
                raise ToolError(f"Unknown path type: {file_path}")
            
            result_data = {
                "path": str(file_path),
                "type": deleted_type,
                "size": file_size
            }
            
            if backup_path:
                result_data["backup_path"] = str(backup_path)
            
            return ToolResult(
                success=True,
                data=result_data,
                message=f"Deleted {deleted_type}: {file_path.name}",
                side_effects=[f"Deleted: {file_path}"]
            )
            
        except Exception as e:
            raise ToolError(f"Failed to delete: {e}")
    
    async def _list_directory(self, params: Dict[str, Any]) -> ToolResult:
        """List directory contents."""
        dir_path = Path(params["path"])
        recursive = params.get("recursive", False)
        
        # Safety checks
        if not self._is_safe_path(dir_path):
            raise ToolError(f"Path not allowed: {dir_path}")
        
        if not dir_path.exists():
            raise ToolError(f"Directory not found: {dir_path}")
        
        if not dir_path.is_dir():
            raise ToolError(f"Path is not a directory: {dir_path}")
        
        try:
            entries = []
            
            if recursive:
                paths = dir_path.rglob('*')
            else:
                paths = dir_path.iterdir()
            
            for path in sorted(paths):
                try:
                    stat = path.stat()
                    entry = {
                        "name": path.name,
                        "path": str(path.relative_to(dir_path)),
                        "type": "file" if path.is_file() else "directory",
                        "size": stat.st_size if path.is_file() else None,
                        "modified": stat.st_mtime
                    }
                    entries.append(entry)
                except (OSError, PermissionError):
                    # Skip inaccessible files
                    continue
            
            return ToolResult(
                success=True,
                data={
                    "directory": str(dir_path),
                    "entries": entries,
                    "count": len(entries),
                    "recursive": recursive
                },
                message=f"Listed {len(entries)} entries in {dir_path.name}"
            )
            
        except Exception as e:
            raise ToolError(f"Failed to list directory: {e}")
    
    async def _create_directory(self, params: Dict[str, Any]) -> ToolResult:
        """Create directory."""
        dir_path = Path(params["path"])
        
        # Safety checks
        if not self._is_safe_path(dir_path):
            raise ToolError(f"Path not allowed: {dir_path}")
        
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            
            return ToolResult(
                success=True,
                data={
                    "path": str(dir_path),
                    "created": not dir_path.exists()
                },
                message=f"Created directory: {dir_path.name}",
                side_effects=[f"Created directory: {dir_path}"]
            )
            
        except Exception as e:
            raise ToolError(f"Failed to create directory: {e}")
    
    async def _get_file_stats(self, params: Dict[str, Any]) -> ToolResult:
        """Get file/directory statistics."""
        file_path = Path(params["path"])
        
        # Safety checks
        if not self._is_safe_path(file_path):
            raise ToolError(f"Path not allowed: {file_path}")
        
        if not file_path.exists():
            raise ToolError(f"Path not found: {file_path}")
        
        try:
            stat = file_path.stat()
            
            result_data = {
                "path": str(file_path),
                "name": file_path.name,
                "type": "file" if file_path.is_file() else "directory",
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "created": stat.st_ctime,
                "accessed": stat.st_atime,
                "permissions": oct(stat.st_mode)[-3:]
            }
            
            # Additional info for files
            if file_path.is_file():
                result_data["extension"] = file_path.suffix
                
                # Calculate hash for small files
                if stat.st_size < 1024 * 1024:  # 1MB
                    content = file_path.read_bytes()
                    result_data["md5"] = hashlib.md5(content).hexdigest()
            
            return ToolResult(
                success=True,
                data=result_data,
                message=f"Retrieved stats for: {file_path.name}"
            )
            
        except Exception as e:
            raise ToolError(f"Failed to get file stats: {e}")
    
    def _is_safe_path(self, path: Path) -> bool:
        """Check if path is safe for operations."""
        try:
            # Resolve to absolute path
            abs_path = path.resolve()
            
            # Check against forbidden paths
            for forbidden in self.forbidden_paths:
                if str(abs_path).startswith(forbidden):
                    return False
            
            # Don't allow operations outside reasonable directories
            # This is a basic check - in production, would be more sophisticated
            allowed_roots = [
                Path.home(),
                Path.cwd(),
                Path("/tmp"),
                Path("/var/tmp")
            ]
            
            for root in allowed_roots:
                try:
                    abs_path.relative_to(root.resolve())
                    return True
                except ValueError:
                    continue
            
            return False
            
        except Exception:
            return False
    
    def _is_allowed_extension(self, path: Path) -> bool:
        """Check if file extension is allowed."""
        if not self.allowed_extensions:
            return True
        
        return path.suffix.lower() in self.allowed_extensions
    
    async def _create_backup(self, path: Path) -> Optional[Path]:
        """Create backup of file or directory."""
        try:
            timestamp = str(int(os.time.time()))
            backup_name = f"{path.name}.backup.{timestamp}"
            backup_path = self.backup_dir / backup_name
            
            if path.is_file():
                shutil.copy2(path, backup_path)
            elif path.is_dir():
                shutil.copytree(path, backup_path)
            else:
                return None
            
            logger.info(f"Created backup: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")
            return None