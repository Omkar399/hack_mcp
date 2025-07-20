"""
MCP Server integration for Eidolon AI Personal Assistant

Provides Model Context Protocol (MCP) server functionality with EnrichMCP framework
for screen capture, search, error detection, and context analysis.
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Literal
from pathlib import Path

from enrichmcp import EnrichModel, EnrichMCP, Relationship, EnrichParameter, PageResult
from pydantic import BaseModel, Field

# Import existing Eidolon components
from ..core.observer import Observer, Screenshot
from ..core.analyzer import Analyzer, ExtractedText, ContentAnalysis
from ..storage.metadata_db import MetadataDatabase
from ..models.cloud_api import CloudAPIManager
from ..utils.logging import get_component_logger, log_performance, log_exceptions
from ..utils.config import get_config

# Import Phase 6 components
from ..core.agent import AutonomousAgent, Task, TaskStatus, TaskPriority, TaskType
from ..core.safety import SafetyManager, ActionApproval, RiskLevel
from ..tools.registry import ToolRegistry
from ..assistants.email_assistant import EmailAssistant
from ..assistants.document_assistant import DocumentAssistant
from ..assistants.office_assistant import OfficeAssistant

# Initialize logger
logger = get_component_logger("mcp_server")

# Initialize the EnrichMCP app
app = EnrichMCP(
    title="eidolon-mcp",
    description="Eidolon AI Personal Assistant - MCP Server for screen memory and intelligent analysis"
)

# Global instances
observer: Optional[Observer] = None
analyzer: Optional[Analyzer] = None
database: Optional[MetadataDatabase] = None
cloud_api: Optional[CloudAPIManager] = None

# Phase 6 global instances
autonomous_agent: Optional[AutonomousAgent] = None
safety_manager: Optional[SafetyManager] = None
tool_registry: Optional[ToolRegistry] = None
email_assistant: Optional[EmailAssistant] = None
document_assistant: Optional[DocumentAssistant] = None
office_assistant: Optional[OfficeAssistant] = None


class ScreenEvent(EnrichModel):
    """A captured screen event with OCR text and analysis."""
    
    id: int = Field(description="Unique screenshot ID")
    timestamp: datetime = Field(description="When the screen was captured")
    file_path: str = Field(description="Path to screenshot file")
    hash: str = Field(description="Screenshot hash for deduplication")
    window_title: Optional[str] = Field(description="Active window title")
    app_name: Optional[str] = Field(description="Application name")
    full_text: Optional[str] = Field(description="Extracted text via OCR")
    ocr_confidence: Optional[float] = Field(description="OCR confidence score (0-1)")
    content_type: Optional[str] = Field(description="Analyzed content type")
    description: Optional[str] = Field(description="Content description")
    tags: List[str] = Field(default_factory=list, description="Content tags")
    
    # Relationships will be implemented in future phases
    commands: List["Command"] = Relationship(description="Commands extracted from screen")
    errors: List["ErrorEvent"] = Relationship(description="Errors detected in screen")


class Command(EnrichModel):
    """A command extracted from screen content."""
    
    id: int = Field(description="Command ID")
    screen_event_id: int = Field(description="Parent screen event ID")
    timestamp: datetime = Field(description="Command timestamp")
    command: str = Field(description="The command text")
    arguments: Optional[str] = Field(description="Command arguments")
    exit_code: Optional[int] = Field(description="Exit code if available")
    shell: Optional[str] = Field(description="Shell type")
    working_directory: Optional[str] = Field(description="Working directory")
    
    # Relationship back to screen event
    screen_event: Optional[ScreenEvent] = Relationship(
        description="The screen event containing this command"
    )


class ErrorEvent(EnrichModel):
    """An error event extracted from screen content."""
    
    id: int = Field(description="Error ID")
    screen_event_id: int = Field(description="Parent screen event ID")
    timestamp: datetime = Field(description="Error timestamp")
    error_type: Optional[str] = Field(description="Type of error")
    error_message: str = Field(description="Error message")
    app_name: Optional[str] = Field(description="Application that generated the error")
    severity: Optional[Literal["low", "medium", "high", "critical"]] = Field(
        description="Error severity"
    )
    
    # Relationship back to screen event
    screen_event: Optional[ScreenEvent] = Relationship(
        description="The screen event containing this error"
    )


class CaptureResult(EnrichModel):
    """Result of a screen capture operation."""
    
    screenshot_id: int = Field(description="ID of the created screenshot")
    success: bool = Field(description="Whether capture was successful")
    message: str = Field(description="Status message")
    processing_time: float = Field(description="Time taken to process capture")
    extracted_text: Optional[str] = Field(description="Text extracted from screenshot")
    content_type: Optional[str] = Field(description="Detected content type")


class SearchResult(EnrichModel):
    """Search result containing matching screen events."""
    
    events: List[ScreenEvent] = Field(description="Matching screen events")
    total_results: int = Field(description="Total number of matching events")
    query: str = Field(description="Original search query")
    search_time: float = Field(description="Time taken to search")


class AnalysisResult(EnrichModel):
    """Result of AI-powered screen context analysis."""
    
    answer: str = Field(description="AI-generated answer to the query")
    confidence: float = Field(description="Confidence score (0-1)")
    events_analyzed: int = Field(description="Number of events analyzed")
    query: str = Field(description="Original query")
    search_time: Optional[float] = Field(description="Time taken to search")
    provider: Optional[str] = Field(description="AI provider used")


# Phase 6 Models for Autonomous Actions

class TaskModel(EnrichModel):
    """A task in the autonomous agent system."""
    
    id: str = Field(description="Unique task ID")
    title: str = Field(description="Task title")
    description: str = Field(description="Task description")
    task_type: str = Field(description="Type of task")
    priority: str = Field(description="Task priority")
    status: str = Field(description="Current task status")
    created_at: datetime = Field(description="When task was created")
    started_at: Optional[datetime] = Field(description="When task execution started")
    completed_at: Optional[datetime] = Field(description="When task was completed")
    requires_approval: bool = Field(description="Whether task requires user approval")
    risk_level: str = Field(description="Risk level of task")


class TaskResult(EnrichModel):
    """Result of task execution."""
    
    task_id: str = Field(description="ID of executed task")
    success: bool = Field(description="Whether task completed successfully")
    message: str = Field(description="Result message")
    execution_time: float = Field(description="Time taken to execute")
    data: Dict[str, Any] = Field(description="Task result data")
    side_effects: List[str] = Field(description="Side effects of execution")


class ActionRequest(EnrichModel):
    """Request for autonomous action execution."""
    
    action_type: str = Field(description="Type of action to execute")
    parameters: Dict[str, Any] = Field(description="Action parameters")
    context: Optional[Dict[str, Any]] = Field(description="Execution context")
    requires_approval: bool = Field(default=True, description="Whether action requires approval")
    timeout_seconds: Optional[float] = Field(description="Action timeout")


class ActionResult(EnrichModel):
    """Result of action execution."""
    
    success: bool = Field(description="Whether action completed successfully")
    message: str = Field(description="Result message")
    data: Dict[str, Any] = Field(description="Action result data")
    execution_time: float = Field(description="Time taken to execute")
    risk_assessment: Dict[str, Any] = Field(description="Risk assessment data")
    approval_required: bool = Field(description="Whether approval was required")


class EmailComposition(EnrichModel):
    """Email composition result."""
    
    subject: str = Field(description="Email subject")
    body: str = Field(description="Email body")
    recipients: List[str] = Field(description="Email recipients")
    composition_method: str = Field(description="How email was composed")
    confidence: float = Field(description="Confidence in composition quality")


class DocumentGeneration(EnrichModel):
    """Document generation result."""
    
    content: str = Field(description="Generated document content")
    document_type: str = Field(description="Type of document")
    word_count: int = Field(description="Number of words in document")
    generation_method: str = Field(description="How document was generated")
    file_path: Optional[str] = Field(description="Path where document was saved")


# Relationship resolvers
@app.resolve
async def resolve_screen_event_commands(screen_event: ScreenEvent) -> List[Command]:
    """Resolve commands for a screen event."""
    # This will be implemented when command extraction is added
    # For now, return empty list
    return []


@app.resolve
async def resolve_screen_event_errors(screen_event: ScreenEvent) -> List[ErrorEvent]:
    """Resolve errors for a screen event."""
    # This will be implemented when error extraction is added
    # For now, check if the content contains error patterns
    if not screen_event.full_text:
        return []
    
    errors = []
    error_patterns = [
        ("error", "high"),
        ("exception", "high"),
        ("traceback", "critical"),
        ("failed", "medium"),
        ("warning", "low")
    ]
    
    text_lower = screen_event.full_text.lower()
    for pattern, severity in error_patterns:
        if pattern in text_lower:
            # Create a simple error event
            errors.append(ErrorEvent(
                id=len(errors) + 1,
                screen_event_id=screen_event.id,
                timestamp=screen_event.timestamp,
                error_type=pattern,
                error_message=f"{pattern.capitalize()} detected in screen content",
                app_name=screen_event.app_name,
                severity=severity
            ))
    
    return errors


@app.resolve
async def resolve_command_screen_event(command: Command) -> Optional[ScreenEvent]:
    """Resolve the screen event for a command."""
    # This will query the database to get the screen event
    # For now, return None
    return None


@app.resolve
async def resolve_error_screen_event(error: ErrorEvent) -> Optional[ScreenEvent]:
    """Resolve the screen event for an error."""
    # This will query the database to get the screen event
    # For now, return None
    return None


# MCP Resource endpoints
@app.retrieve
async def capture_screen(
    save_image: bool = EnrichParameter(
        default=True,
        description="Whether to save screenshot image",
        examples=[True, False]
    ),
    analyze_content: bool = EnrichParameter(
        default=True,
        description="Whether to analyze content with AI",
        examples=[True, False]
    ),
    use_cloud_ai: bool = EnrichParameter(
        default=False,
        description="Whether to use cloud AI for enhanced analysis",
        examples=[True, False]
    )
) -> CaptureResult:
    """Capture the current screen and extract text/visual information."""
    global observer, analyzer, database
    
    try:
        start_time = datetime.now()
        
        # Initialize components if needed
        if not observer:
            observer = Observer()
        if not analyzer:
            analyzer = Analyzer()
        if not database:
            database = MetadataDatabase()
        
        # Capture screenshot
        screenshot = observer.capture_screenshot()
        if not screenshot:
            return CaptureResult(
                screenshot_id=-1,
                success=False,
                message="Failed to capture screenshot",
                processing_time=(datetime.now() - start_time).total_seconds()
            )
        
        # Save screenshot if requested
        if save_image:
            timestamp_str = screenshot.timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"screenshot_{timestamp_str}_{screenshot.hash[:8]}.png"
            file_path = Path(observer.storage_path) / filename
            screenshot.save(str(file_path))
            screenshot.file_path = str(file_path)
        else:
            screenshot.file_path = None
        
        # Store in database
        screenshot_id = database.store_screenshot(
            file_path=screenshot.file_path,
            timestamp=screenshot.timestamp,
            hash=screenshot.hash,
            size=screenshot.image.size,
            file_size=file_path.stat().st_size if save_image else 0,
            window_info=screenshot.window_info,
            monitor_info=screenshot.monitor_info
        )
        
        # Extract text and analyze content if requested
        extracted_text = None
        content_type = None
        
        if analyze_content:
            # OCR extraction
            text_result = analyzer.extract_text(screenshot.image)
            if text_result and text_result.text:
                database.store_ocr_result(screenshot_id, text_result)
                extracted_text = text_result.text
            
            # Content analysis
            analysis = analyzer.analyze_content(
                screenshot.image,
                text_result.text if text_result else ""
            )
            database.store_content_analysis(screenshot_id, analysis)
            content_type = analysis.content_type
            
            # Cloud AI analysis if requested
            if use_cloud_ai and cloud_api:
                try:
                    cloud_response = await cloud_api.analyze_image(
                        screenshot.file_path if save_image else screenshot.image,
                        prompt="Analyze this screenshot and describe what you see, including any text, UI elements, and activities."
                    )
                    if cloud_response:
                        # Store cloud analysis as additional metadata
                        logger.debug(f"Cloud AI analysis: {cloud_response.content[:200]}...")
                except Exception as e:
                    logger.warning(f"Cloud AI analysis failed: {e}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return CaptureResult(
            screenshot_id=screenshot_id,
            success=True,
            message="Screen captured successfully",
            processing_time=processing_time,
            extracted_text=extracted_text,
            content_type=content_type
        )
        
    except Exception as e:
        logger.error(f"Capture failed: {e}")
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return CaptureResult(
            screenshot_id=-1,
            success=False,
            message=f"Capture failed: {str(e)}",
            processing_time=processing_time
        )


@app.retrieve
async def search_screens(
    query: str = EnrichParameter(
        description="Search query for finding relevant screens",
        examples=["docker command", "error message", "login form", "git commit"]
    ),
    limit: int = EnrichParameter(
        default=10,
        description="Maximum number of results to return",
        examples=[10, 25, 50]
    ),
    since_hours: Optional[int] = EnrichParameter(
        default=None,
        description="Only search screens from the last N hours",
        examples=[1, 24, 168]
    ),
    content_type: Optional[str] = EnrichParameter(
        default=None,
        description="Filter by content type",
        examples=["code", "terminal", "browser", "document", "chat"]
    )
) -> SearchResult:
    """Search screen events by text content and metadata."""
    global database
    
    try:
        start_time = datetime.now()
        
        if not database:
            database = MetadataDatabase()
        
        # Build search parameters
        content_types = [content_type] if content_type else None
        start_time_filter = None
        end_time_filter = None
        
        if since_hours:
            start_time_filter = datetime.now() - timedelta(hours=since_hours)
            end_time_filter = datetime.now()
        
        # Perform combined search
        results = database.search_combined(
            text_query=query,
            content_types=content_types,
            start_time=start_time_filter,
            end_time=end_time_filter,
            limit=limit
        )
        
        # Convert to ScreenEvent models
        events = []
        for result in results:
            events.append(ScreenEvent(
                id=result['screenshot_id'],
                timestamp=datetime.fromisoformat(result['timestamp']),
                file_path=result['file_path'],
                hash=result['hash'],
                window_title=None,  # Not stored in current schema
                app_name=None,  # Not stored in current schema
                full_text=result.get('text'),
                ocr_confidence=result.get('ocr_confidence'),
                content_type=result.get('content_type'),
                description=result.get('description'),
                tags=result.get('tags', [])
            ))
        
        search_time = (datetime.now() - start_time).total_seconds()
        
        return SearchResult(
            events=events,
            total_results=len(events),
            query=query,
            search_time=search_time
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return SearchResult(
            events=[],
            total_results=0,
            query=query,
            search_time=0.0
        )


@app.retrieve
async def get_recent_screens(
    limit: int = EnrichParameter(
        default=20,
        description="Number of recent screens to retrieve",
        examples=[10, 20, 50]
    ),
    hours: int = EnrichParameter(
        default=24,
        description="Look back this many hours",
        examples=[1, 6, 24, 168]
    )
) -> PageResult[ScreenEvent]:
    """Get recently captured screens."""
    global database
    
    try:
        if not database:
            database = MetadataDatabase()
        
        # Calculate time range
        start_time = datetime.now() - timedelta(hours=hours)
        end_time = datetime.now()
        
        # Get screenshots from database
        results = database.get_screenshots_by_timerange(start_time, end_time)
        
        # Convert to ScreenEvent models
        events = []
        for result in results[:limit]:
            events.append(ScreenEvent(
                id=result['id'],
                timestamp=datetime.fromisoformat(result['timestamp']),
                file_path=result['file_path'],
                hash=result['hash'],
                window_title=None,
                app_name=None,
                full_text=result.get('ocr_text'),
                ocr_confidence=result.get('ocr_confidence'),
                content_type=result.get('content_type'),
                description=result.get('description'),
                tags=[]
            ))
        
        return PageResult.create(
            items=events,
            page=1,
            page_size=limit,
            total_items=len(events)
        )
        
    except Exception as e:
        logger.error(f"Failed to get recent screens: {e}")
        return PageResult.create(
            items=[],
            page=1,
            page_size=limit,
            total_items=0
        )


@app.retrieve
async def find_screens_with_errors(
    severity: Optional[Literal["low", "medium", "high", "critical"]] = EnrichParameter(
        default=None,
        description="Filter by error severity",
        examples=["high", "critical"]
    ),
    since_hours: int = EnrichParameter(
        default=24,
        description="Look back this many hours for errors",
        examples=[1, 6, 24]
    ),
    limit: int = EnrichParameter(
        default=10,
        description="Maximum number of error events to return",
        examples=[5, 10, 20]
    )
) -> List[ErrorEvent]:
    """Find screens containing error messages."""
    global database
    
    try:
        if not database:
            database = MetadataDatabase()
        
        # Search for screens with error keywords
        error_keywords = ["error", "exception", "traceback", "failed", "fatal"]
        query = " OR ".join(error_keywords)
        
        # Calculate time range
        start_time = datetime.now() - timedelta(hours=since_hours) if since_hours else None
        
        # Search for error screens
        results = database.search_combined(
            text_query=query,
            start_time=start_time,
            limit=limit * 2  # Get more to filter by severity
        )
        
        # Convert to ErrorEvent models
        errors = []
        for result in results:
            if not result.get('text'):
                continue
            
            text_lower = result['text'].lower()
            
            # Determine error type and severity
            detected_severity = "low"
            error_type = "unknown"
            
            if "traceback" in text_lower or "fatal" in text_lower:
                detected_severity = "critical"
                error_type = "exception" if "traceback" in text_lower else "fatal"
            elif "exception" in text_lower or "error" in text_lower:
                detected_severity = "high"
                error_type = "exception" if "exception" in text_lower else "error"
            elif "failed" in text_lower:
                detected_severity = "medium"
                error_type = "failure"
            elif "warning" in text_lower:
                detected_severity = "low"
                error_type = "warning"
            
            # Apply severity filter
            if severity and detected_severity != severity:
                continue
            
            # Extract error message (simplified)
            error_message = "Error detected in screen content"
            lines = result['text'].split('\n')
            for line in lines:
                if any(keyword in line.lower() for keyword in error_keywords):
                    error_message = line.strip()[:200]  # Limit length
                    break
            
            errors.append(ErrorEvent(
                id=len(errors) + 1,
                screen_event_id=result['screenshot_id'],
                timestamp=datetime.fromisoformat(result['timestamp']),
                error_type=error_type,
                error_message=error_message,
                app_name=None,
                severity=detected_severity
            ))
            
            if len(errors) >= limit:
                break
        
        return errors
        
    except Exception as e:
        logger.error(f"Failed to find error screens: {e}")
        return []


@app.retrieve
async def analyze_screen_context(
    query: str = EnrichParameter(
        description="Question about screen content or context",
        examples=[
            "What was I working on 2 hours ago?",
            "Show me all the git commands I've run today",
            "Find error messages in Terminal from this morning",
            "What documents did I have open yesterday?"
        ]
    ),
    max_events: int = EnrichParameter(
        default=5,
        description="Maximum events to analyze for context",
        examples=[3, 5, 10]
    ),
    use_cloud_ai: bool = EnrichParameter(
        default=True,
        description="Use cloud AI for better analysis",
        examples=[True, False]
    )
) -> AnalysisResult:
    """Analyze screen context using AI to answer questions about captured content."""
    global database, cloud_api
    
    try:
        # Initialize components if needed
        if not database:
            database = MetadataDatabase()
        if not cloud_api and use_cloud_ai:
            cloud_api = CloudAPIManager()
        
        # Get the enrichMCP context for LLM access
        ctx = app.get_context()
        
        # Search for relevant screens
        search_start = datetime.now()
        search_results = await search_screens(query, limit=max_events)
        search_time = (datetime.now() - search_start).total_seconds()
        
        if not search_results.events:
            return AnalysisResult(
                answer="No relevant screen content found for your query.",
                confidence=0.0,
                events_analyzed=0,
                query=query,
                search_time=search_time
            )
        
        # Prepare context for analysis
        context_text = f"User query: {query}\n\nRelevant screen captures:\n"
        for i, event in enumerate(search_results.events[:max_events], 1):
            context_text += f"\n{i}. [{event.timestamp}]"
            if event.content_type:
                context_text += f" Type: {event.content_type}"
            if event.description:
                context_text += f" - {event.description}"
            if event.full_text:
                # Include relevant portion of text
                text_preview = event.full_text[:500]
                if len(event.full_text) > 500:
                    text_preview += "..."
                context_text += f"\nContent: {text_preview}"
            context_text += "\n"
        
        # Use AI to analyze context
        if use_cloud_ai and cloud_api:
            # Use Eidolon's cloud AI
            prompt = (
                f"{context_text}\n\n"
                "Based on the screen captures above, please answer the user's query. "
                "Be specific and reference the relevant screens by their timestamps."
            )
            
            cloud_response = await cloud_api.analyze_text(
                prompt,
                analysis_type="general"
            )
            
            if cloud_response:
                return AnalysisResult(
                    answer=cloud_response.content,
                    confidence=cloud_response.confidence,
                    events_analyzed=len(search_results.events),
                    query=query,
                    search_time=search_time,
                    provider=cloud_response.provider
                )
        
        # Fallback to EnrichMCP's server-side LLM
        try:
            response = await ctx.ask_llm(
                prompt=(
                    f"{context_text}\n\n"
                    "Based on the screen captures above, please answer the user's query. "
                    "Be specific and reference the relevant screens."
                ),
                max_tokens=500,
                model_preferences=["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet"]
            )
            
            return AnalysisResult(
                answer=response.text,
                confidence=0.8,
                events_analyzed=len(search_results.events),
                query=query,
                search_time=search_time,
                provider="enrichmcp"
            )
        except Exception as llm_error:
            logger.warning(f"LLM analysis failed: {llm_error}")
            
            # Basic fallback analysis without AI
            answer = f"Found {len(search_results.events)} relevant screens for your query:\n"
            for event in search_results.events[:3]:
                answer += f"- {event.timestamp}: {event.description or 'No description'}\n"
            
            return AnalysisResult(
                answer=answer,
                confidence=0.3,
                events_analyzed=len(search_results.events),
                query=query,
                search_time=search_time,
                provider="basic"
            )
        
    except Exception as e:
        logger.error(f"Failed to analyze screen context: {e}")
        return AnalysisResult(
            answer=f"Analysis failed: {str(e)}",
            confidence=0.0,
            events_analyzed=0,
            query=query
        )


@app.retrieve
async def start_monitoring(
    capture_interval: Optional[int] = EnrichParameter(
        default=None,
        description="Seconds between automatic screenshots",
        examples=[10, 30, 60]
    )
) -> Dict[str, Any]:
    """Start the Eidolon screenshot monitoring system."""
    global observer
    
    try:
        # Initialize observer with custom interval if provided
        config_override = {}
        if capture_interval:
            config_override['capture_interval'] = capture_interval
        
        if not observer:
            observer = Observer(config_override=config_override)
        
        # Start monitoring
        observer.start_monitoring()
        
        return {
            "success": True,
            "message": "Monitoring started successfully",
            "capture_interval": capture_interval or observer.config.observer.capture_interval,
            "status": observer.get_status()
        }
        
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")
        return {
            "success": False,
            "message": f"Failed to start monitoring: {str(e)}"
        }


@app.retrieve
async def stop_monitoring() -> Dict[str, Any]:
    """Stop the Eidolon screenshot monitoring system."""
    global observer
    
    try:
        if not observer:
            return {
                "success": False,
                "message": "Monitoring system is not running"
            }
        
        # Get final status before stopping
        final_status = observer.get_status()
        
        # Stop monitoring
        observer.stop_monitoring()
        
        return {
            "success": True,
            "message": "Monitoring stopped successfully",
            "final_status": final_status
        }
        
    except Exception as e:
        logger.error(f"Failed to stop monitoring: {e}")
        return {
            "success": False,
            "message": f"Failed to stop monitoring: {str(e)}"
        }


@app.retrieve
async def get_system_status() -> Dict[str, Any]:
    """Get the current status of the Eidolon system."""
    global observer, database
    
    try:
        status = {
            "monitoring_active": False,
            "database_connected": False,
            "cloud_ai_available": False,
            "statistics": {}
        }
        
        # Check observer status
        if observer:
            observer_status = observer.get_status()
            status["monitoring_active"] = observer_status.get("running", False)
            status["observer_status"] = observer_status
        
        # Check database status
        if not database:
            database = MetadataDatabase()
        
        try:
            stats = database.get_statistics()
            status["database_connected"] = True
            status["statistics"] = stats
        except Exception:
            status["database_connected"] = False
        
        # Check cloud AI availability
        if not cloud_api:
            try:
                test_cloud_api = CloudAPIManager()
                status["cloud_ai_available"] = len(test_cloud_api.get_available_providers()) > 0
                status["available_providers"] = test_cloud_api.get_available_providers()
            except Exception:
                status["cloud_ai_available"] = False
        else:
            status["cloud_ai_available"] = len(cloud_api.get_available_providers()) > 0
            status["available_providers"] = cloud_api.get_available_providers()
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        return {
            "error": str(e),
            "monitoring_active": False,
            "database_connected": False,
            "cloud_ai_available": False
        }


# Phase 6 MCP Endpoints for Autonomous Actions

@app.retrieve
async def create_task(
    title: str = EnrichParameter(
        description="Task title",
        examples=["Send weekly report", "Analyze recent emails", "Generate meeting notes"]
    ),
    description: str = EnrichParameter(
        description="Detailed task description",
        examples=["Compose and send the weekly productivity report to the team"]
    ),
    task_type: str = EnrichParameter(
        default="automation",
        description="Type of task",
        examples=["analysis", "automation", "communication", "file_operation"]
    ),
    priority: str = EnrichParameter(
        default="medium",
        description="Task priority level",
        examples=["low", "medium", "high", "urgent"]
    ),
    context: Optional[Dict[str, Any]] = EnrichParameter(
        default=None,
        description="Additional context for task execution"
    )
) -> TaskModel:
    """Create a new autonomous task."""
    global autonomous_agent
    
    try:
        if not autonomous_agent:
            autonomous_agent = AutonomousAgent()
            await autonomous_agent.initialize()
        
        # Suggest task based on description
        suggested_tasks = await autonomous_agent.suggest_task(description)
        
        if suggested_tasks:
            task = suggested_tasks[0]  # Use first suggestion
            
            return TaskModel(
                id=task.id,
                title=task.title,
                description=task.description,
                task_type=task.task_type.value,
                priority=task.priority.value,
                status=task.status.value,
                created_at=task.created_at,
                started_at=task.started_at,
                completed_at=task.completed_at,
                requires_approval=task.requires_approval,
                risk_level=task.risk_level.value
            )
        else:
            # Create basic task if no suggestions
            from uuid import uuid4
            return TaskModel(
                id=str(uuid4()),
                title=title,
                description=description,
                task_type=task_type,
                priority=priority,
                status="pending",
                created_at=datetime.now(),
                started_at=None,
                completed_at=None,
                requires_approval=True,
                risk_level="medium"
            )
            
    except Exception as e:
        logger.error(f"Task creation failed: {e}")
        raise


@app.retrieve
async def execute_action(
    action_type: str = EnrichParameter(
        description="Type of action to execute",
        examples=["send_email", "create_document", "analyze_data", "web_request"]
    ),
    parameters: Dict[str, Any] = EnrichParameter(
        description="Parameters for the action",
        examples=[{"to": ["user@example.com"], "subject": "Test", "body": "Hello"}]
    ),
    require_approval: bool = EnrichParameter(
        default=True,
        description="Whether to require user approval before execution"
    ),
    timeout_seconds: Optional[float] = EnrichParameter(
        default=60.0,
        description="Maximum time to wait for action completion"
    )
) -> ActionResult:
    """Execute an autonomous action with safety controls."""
    global tool_registry, safety_manager
    
    try:
        # Initialize components if needed
        if not tool_registry:
            tool_registry = ToolRegistry()
        if not safety_manager:
            safety_manager = SafetyManager()
        
        start_time = datetime.now()
        
        # Create action request
        action = {
            "type": action_type,
            "params": parameters
        }
        
        # Risk assessment
        risk_level = await safety_manager.assess_risk([action])
        
        # Validate action
        validation = await safety_manager.validate_action(action)
        
        if not validation.get("approved", False) and require_approval:
            return ActionResult(
                success=False,
                message=f"Action requires approval: {validation.get('reason', 'Unknown')}",
                data={"validation": validation},
                execution_time=0.0,
                risk_assessment={"risk_level": risk_level.value},
                approval_required=True
            )
        
        # Execute action using tool registry
        if action_type in ["send_email", "email_operation"]:
            result = await _execute_email_action(parameters)
        elif action_type in ["create_document", "document_operation"]:
            result = await _execute_document_action(parameters)
        elif action_type in ["web_request", "web_operation"]:
            result = await _execute_web_action(parameters)
        else:
            # Try to execute as generic tool
            try:
                tool_result = await tool_registry.execute_tool(action_type, parameters)
                result = {
                    "success": tool_result.success,
                    "data": tool_result.data,
                    "message": tool_result.message
                }
            except Exception as e:
                result = {
                    "success": False,
                    "data": {"error": str(e)},
                    "message": f"Tool execution failed: {e}"
                }
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Record action for audit
        await safety_manager.record_action(
            action,
            validation.get("risk_assessment"),
            validation.get("approval"),
            result
        )
        
        return ActionResult(
            success=result["success"],
            message=result["message"],
            data=result["data"],
            execution_time=execution_time,
            risk_assessment={"risk_level": risk_level.value},
            approval_required=require_approval
        )
        
    except Exception as e:
        logger.error(f"Action execution failed: {e}")
        return ActionResult(
            success=False,
            message=f"Action execution failed: {str(e)}",
            data={"error": str(e)},
            execution_time=0.0,
            risk_assessment={"risk_level": "critical"},
            approval_required=True
        )


@app.retrieve
async def compose_email(
    request: str = EnrichParameter(
        description="Email composition request",
        examples=["Write a follow-up email to the client about our meeting yesterday"]
    ),
    recipients: Optional[List[str]] = EnrichParameter(
        default=None,
        description="Email recipients",
        examples=[["client@example.com"], ["team@company.com", "manager@company.com"]]
    ),
    template_name: Optional[str] = EnrichParameter(
        default=None,
        description="Email template to use",
        examples=["meeting_request", "follow_up", "thank_you"]
    )
) -> EmailComposition:
    """Compose an email using AI assistance."""
    global email_assistant
    
    try:
        if not email_assistant:
            email_assistant = EmailAssistant()
        
        # Compose email
        result = await email_assistant.compose_email(
            request=request,
            context={"recipients": recipients} if recipients else None,
            template_name=template_name,
            recipients=recipients
        )
        
        if result.get("success", False):
            return EmailComposition(
                subject=result.get("subject", ""),
                body=result.get("body", ""),
                recipients=recipients or [],
                composition_method=result.get("template_used", "ai_generated"),
                confidence=0.8 if result.get("ai_generated", False) else 0.6
            )
        else:
            return EmailComposition(
                subject="",
                body=f"Failed to compose email: {result.get('error', 'Unknown error')}",
                recipients=recipients or [],
                composition_method="error",
                confidence=0.0
            )
            
    except Exception as e:
        logger.error(f"Email composition failed: {e}")
        return EmailComposition(
            subject="",
            body=f"Email composition error: {str(e)}",
            recipients=recipients or [],
            composition_method="error",
            confidence=0.0
        )


@app.retrieve
async def generate_document(
    request: str = EnrichParameter(
        description="Document generation request",
        examples=["Create a project status report for this week's activities"]
    ),
    document_type: str = EnrichParameter(
        default="markdown",
        description="Type of document to generate",
        examples=["markdown", "text", "report", "notes"]
    ),
    template_name: Optional[str] = EnrichParameter(
        default=None,
        description="Document template to use",
        examples=["meeting_notes", "project_report", "technical_spec"]
    ),
    output_path: Optional[str] = EnrichParameter(
        default=None,
        description="Where to save the generated document",
        examples=["/path/to/document.md", "./reports/weekly_status.md"]
    )
) -> DocumentGeneration:
    """Generate a document using AI assistance."""
    global document_assistant
    
    try:
        if not document_assistant:
            document_assistant = DocumentAssistant()
        
        # Generate document
        result = await document_assistant.generate_document(
            request=request,
            document_type=document_type,
            template_name=template_name,
            output_path=output_path
        )
        
        if result.get("success", False):
            content = result.get("content", "")
            return DocumentGeneration(
                content=content,
                document_type=result.get("document_type", document_type),
                word_count=len(content.split()),
                generation_method=result.get("template_used", "ai_generated"),
                file_path=result.get("output_path")
            )
        else:
            return DocumentGeneration(
                content=f"Failed to generate document: {result.get('error', 'Unknown error')}",
                document_type=document_type,
                word_count=0,
                generation_method="error",
                file_path=None
            )
            
    except Exception as e:
        logger.error(f"Document generation failed: {e}")
        return DocumentGeneration(
            content=f"Document generation error: {str(e)}",
            document_type=document_type,
            word_count=0,
            generation_method="error",
            file_path=None
        )


@app.retrieve
async def get_active_tasks(
    status_filter: Optional[str] = EnrichParameter(
        default=None,
        description="Filter tasks by status",
        examples=["pending", "running", "completed", "failed"]
    ),
    limit: int = EnrichParameter(
        default=10,
        description="Maximum number of tasks to return",
        examples=[5, 10, 20]
    )
) -> List[TaskModel]:
    """Get active tasks from the autonomous agent."""
    global autonomous_agent
    
    try:
        if not autonomous_agent:
            autonomous_agent = AutonomousAgent()
            await autonomous_agent.initialize()
        
        # Get active tasks
        active_tasks = await autonomous_agent.get_active_tasks()
        
        # Filter by status if specified
        if status_filter:
            active_tasks = [task for task in active_tasks if task.status.value == status_filter]
        
        # Convert to models and limit
        task_models = []
        for task in active_tasks[:limit]:
            task_models.append(TaskModel(
                id=task.id,
                title=task.title,
                description=task.description,
                task_type=task.task_type.value,
                priority=task.priority.value,
                status=task.status.value,
                created_at=task.created_at,
                started_at=task.started_at,
                completed_at=task.completed_at,
                requires_approval=task.requires_approval,
                risk_level=task.risk_level.value
            ))
        
        return task_models
        
    except Exception as e:
        logger.error(f"Failed to get active tasks: {e}")
        return []


@app.retrieve
async def approve_task(
    task_id: str = EnrichParameter(
        description="ID of task to approve",
        examples=["task_123", "autonomous_task_456"]
    ),
    user_id: str = EnrichParameter(
        default="user",
        description="ID of user providing approval",
        examples=["user", "admin", "john_doe"]
    )
) -> Dict[str, Any]:
    """Approve a task for execution."""
    global autonomous_agent
    
    try:
        if not autonomous_agent:
            autonomous_agent = AutonomousAgent()
            await autonomous_agent.initialize()
        
        # Approve the task
        success = await autonomous_agent.approve_task(task_id, user_id)
        
        if success:
            return {
                "success": True,
                "message": f"Task {task_id} approved by {user_id}",
                "task_id": task_id,
                "approved_by": user_id,
                "approved_at": datetime.now().isoformat()
            }
        else:
            return {
                "success": False,
                "message": f"Failed to approve task {task_id} - task not found",
                "task_id": task_id
            }
            
    except Exception as e:
        logger.error(f"Task approval failed: {e}")
        return {
            "success": False,
            "message": f"Task approval failed: {str(e)}",
            "task_id": task_id
        }


@app.retrieve
async def office_assistant_request(
    request: str = EnrichParameter(
        description="Office automation request",
        examples=["Schedule a meeting for next week", "Analyze my productivity this month"]
    ),
    automation_level: str = EnrichParameter(
        default="semi_auto",
        description="Level of automation to apply",
        examples=["manual", "semi_auto", "full_auto"]
    ),
    context: Optional[Dict[str, Any]] = EnrichParameter(
        default=None,
        description="Additional context for the request"
    )
) -> Dict[str, Any]:
    """Process a general office automation request."""
    global office_assistant
    
    try:
        if not office_assistant:
            office_assistant = OfficeAssistant()
            await office_assistant.initialize()
        
        # Process the request
        result = await office_assistant.process_request(
            request=request,
            context=context,
            automation_level=automation_level
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Office assistant request failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": f"Office assistant request failed: {str(e)}"
        }


# Helper functions for action execution

async def _execute_email_action(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Execute email-related action."""
    global email_assistant
    
    try:
        if not email_assistant:
            email_assistant = EmailAssistant()
        
        # Route to appropriate email operation
        operation = parameters.get("operation", "send")
        
        if operation == "send":
            result = await email_assistant.manage_email_workflow("send", parameters)
        elif operation == "compose":
            result = await email_assistant.compose_email(
                parameters.get("request", ""),
                context=parameters.get("context")
            )
        elif operation == "analyze":
            subject = parameters.get("subject", "")
            body = parameters.get("body", "")
            analysis = await email_assistant.analyze_email(subject, body)
            result = {"success": True, "data": analysis, "message": "Email analyzed"}
        else:
            result = {"success": False, "data": {}, "message": f"Unknown email operation: {operation}"}
        
        return result
        
    except Exception as e:
        return {"success": False, "data": {"error": str(e)}, "message": f"Email action failed: {e}"}


async def _execute_document_action(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Execute document-related action."""
    global document_assistant
    
    try:
        if not document_assistant:
            document_assistant = DocumentAssistant()
        
        # Route to appropriate document operation
        operation = parameters.get("operation", "generate")
        
        if operation == "generate":
            result = await document_assistant.generate_document(
                request=parameters.get("request", ""),
                document_type=parameters.get("document_type", "text"),
                template_name=parameters.get("template_name"),
                output_path=parameters.get("output_path")
            )
        elif operation == "analyze":
            result = await document_assistant.analyze_document(
                file_path=parameters.get("file_path", ""),
                content=parameters.get("content"),
                analysis_type=parameters.get("analysis_type", "comprehensive")
            )
            result = {"success": True, "data": result, "message": "Document analyzed"}
        elif operation == "summarize":
            result = await document_assistant.summarize_document(
                file_path=parameters.get("file_path", ""),
                content=parameters.get("content"),
                summary_length=parameters.get("summary_length", "medium")
            )
        else:
            result = {"success": False, "data": {}, "message": f"Unknown document operation: {operation}"}
        
        return result
        
    except Exception as e:
        return {"success": False, "data": {"error": str(e)}, "message": f"Document action failed: {e}"}


async def _execute_web_action(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Execute web-related action."""
    try:
        # Import web operations tool
        from ..tools.web_ops import WebOperationsTool
        
        web_tool = WebOperationsTool()
        tool_result = await web_tool.execute(parameters)
        
        return {
            "success": tool_result.success,
            "data": tool_result.data,
            "message": tool_result.message
        }
        
    except Exception as e:
        return {"success": False, "data": {"error": str(e)}, "message": f"Web action failed: {e}"}


async def initialize_server():
    """Initialize the MCP server components."""
    global observer, analyzer, database, cloud_api
    global autonomous_agent, safety_manager, tool_registry, email_assistant, document_assistant, office_assistant
    
    logger.info("Initializing Eidolon MCP Server with Phase 6 capabilities...")
    
    try:
        # Get configuration
        config = get_config()
        
        # Initialize database
        database = MetadataDatabase()
        logger.info("Database initialized")
        
        # Initialize analyzer
        analyzer = Analyzer()
        logger.info("Analyzer initialized")
        
        # Initialize observer (but don't start monitoring yet)
        observer = Observer()
        logger.info("Observer initialized")
        
        # Initialize cloud API if keys are available
        try:
            cloud_api = CloudAPIManager()
            if cloud_api.get_available_providers():
                logger.info(f"Cloud API initialized with providers: {cloud_api.get_available_providers()}")
            else:
                logger.warning("No cloud AI providers available")
                cloud_api = None
        except Exception as e:
            logger.warning(f"Cloud API initialization failed: {e}")
            cloud_api = None
        
        # Test database connection
        try:
            stats = database.get_statistics()
            logger.info(f"Database contains {stats.get('total_screenshots', 0)} screenshots")
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
        
        # Initialize Phase 6 components
        try:
            # Initialize safety manager
            safety_manager = SafetyManager()
            logger.info("Safety manager initialized")
            
            # Initialize tool registry
            tool_registry = ToolRegistry(safety_manager)
            logger.info("Tool registry initialized")
            
            # Initialize autonomous agent
            autonomous_agent = AutonomousAgent()
            await autonomous_agent.initialize()
            logger.info("Autonomous agent initialized")
            
            # Initialize assistants
            email_assistant = EmailAssistant()
            document_assistant = DocumentAssistant()
            office_assistant = OfficeAssistant()
            await office_assistant.initialize()
            logger.info("Office assistants initialized")
            
            logger.info("Phase 6 components initialized successfully")
            
        except Exception as e:
            logger.warning(f"Phase 6 component initialization failed: {e}")
            # Continue without Phase 6 components for backwards compatibility
            
    except Exception as e:
        logger.error(f"Server initialization failed: {e}")
        raise


def main():
    """Main entry point for the MCP server."""
    # Initialize and run the server
    asyncio.run(initialize_server())
    
    # Run the EnrichMCP server with stdio transport (standard for local MCP)
    logger.info("Starting Eidolon MCP server with stdio transport...")
    app.run(transport="stdio")


if __name__ == "__main__":
    main()