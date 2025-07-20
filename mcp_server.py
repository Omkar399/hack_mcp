"""
EnrichMCP Server for Screen Memory Assistant

Provides MCP-compatible access to screen captures, search, and analysis.
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Literal
from decimal import Decimal

from enrichmcp import EnrichModel, EnrichMCP, Relationship, EnrichParameter, PageResult
from pydantic import BaseModel, Field

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import our existing components
from database import db
from capture import ScreenCapture
from models import ScreenEventResponse

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the EnrichMCP app
app = EnrichMCP(
    title="screen-memory-assistant",
    description="AI-powered screen memory system with capture, OCR, and semantic search"
)

# Global capture system
capture_system: Optional[ScreenCapture] = None


class ScreenEvent(EnrichModel):
    """A captured screen event with OCR text and visual embeddings."""
    
    id: int = Field(description="Unique event ID")
    timestamp: datetime = Field(description="When the screen was captured")
    window_title: Optional[str] = Field(description="Active window title")
    app_name: Optional[str] = Field(description="Application name")
    full_text: Optional[str] = Field(description="Extracted text via OCR")
    ocr_confidence: Optional[int] = Field(description="OCR confidence score (0-100)")
    image_path: Optional[str] = Field(description="Path to screenshot image")
    scene_hash: Optional[str] = Field(description="Hash for duplicate detection")
    
    # Note: Relationships will be implemented when command/error extraction is added


class Command(EnrichModel):
    """A command extracted from screen content."""
    
    id: int = Field(description="Command ID")
    event_id: int = Field(description="Parent screen event ID")
    timestamp: datetime = Field(description="Command timestamp")
    command: str = Field(description="The command text")
    arguments: Optional[str] = Field(description="Command arguments")
    exit_code: Optional[int] = Field(description="Exit code if available")
    shell: Optional[str] = Field(description="Shell type")
    working_directory: Optional[str] = Field(description="Working directory")


class ErrorEvent(EnrichModel):
    """An error event extracted from screen content."""
    
    id: int = Field(description="Error ID")
    event_id: int = Field(description="Parent screen event ID")
    timestamp: datetime = Field(description="Error timestamp")
    error_type: Optional[str] = Field(description="Type of error")
    error_message: str = Field(description="Error message")
    app_name: Optional[str] = Field(description="Application that generated the error")
    severity: Optional[Literal["low", "medium", "high", "critical"]] = Field(description="Error severity")


class CaptureResult(EnrichModel):
    """Result of a screen capture operation."""
    
    event_id: int = Field(description="ID of the created screen event")
    success: bool = Field(description="Whether capture was successful")
    message: str = Field(description="Status message")
    processing_time: float = Field(description="Time taken to process capture")


class SearchResult(EnrichModel):
    """Search result containing matching screen events."""
    
    events: List[ScreenEvent] = Field(description="Matching screen events")
    total_results: int = Field(description="Total number of matching events")
    query: str = Field(description="Original search query")
    search_time: float = Field(description="Time taken to search")


# Note: Relationship resolvers will be added when command/error extraction is implemented


# MCP Resource endpoints
@app.retrieve
async def capture_screen(
    save_image: bool = EnrichParameter(
        default=True, 
        description="Whether to save screenshot image",
        examples=[True, False]
    ),
    use_vision: bool = EnrichParameter(
        default=False,
        description="Whether to use vision API for better text extraction",
        examples=[True, False]
    )
) -> CaptureResult:
    """Capture the current screen and extract text/visual information."""
    global capture_system
    
    if not capture_system:
        capture_system = ScreenCapture()
    
    try:
        start_time = datetime.now()
        
        # Perform screen capture
        result = await capture_system.capture_screen(
            save_image=save_image,
            use_vision_fallback=use_vision
        )
        
        # Save to database
        event_id = await db.save_screen_event(result)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return CaptureResult(
            event_id=event_id,
            success=True,
            message="Screen captured successfully",
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Capture failed: {e}")
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return CaptureResult(
            event_id=-1,
            success=False,
            message=f"Capture failed: {str(e)}",
            processing_time=processing_time
        )


@app.retrieve
async def search_screens(
    query: str = EnrichParameter(
        description="Search query for finding relevant screens",
        examples=["docker command", "error message", "login form"]
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
    app_filter: Optional[str] = EnrichParameter(
        default=None,
        description="Filter by application name",
        examples=["Chrome", "Terminal", "VS Code"]
    )
) -> SearchResult:
    """Search screen events by text content."""
    try:
        start_time = datetime.now()
        
        # Convert since_hours to minutes for the database query
        since_minutes = since_hours * 60 if since_hours else None
        
        # Perform search
        events_data = await db.search_events(
            query=query,
            limit=limit,
            since_minutes=since_minutes,
            app_name=app_filter
        )
        
        # Convert to ScreenEvent models
        events = []
        for event_data in events_data:
            events.append(ScreenEvent(
                id=event_data.id,
                timestamp=event_data.ts,
                window_title=event_data.window_title,
                app_name=event_data.app_name,
                full_text=event_data.full_text,
                ocr_confidence=event_data.ocr_conf,
                image_path=event_data.image_path,
                scene_hash=event_data.scene_hash
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
    try:
        events_data = await db.get_recent_events(limit=limit, hours=hours)
        
        events = []
        for event_data in events_data:
            events.append(ScreenEvent(
                id=event_data.id,
                timestamp=event_data.ts,
                window_title=event_data.window_title,
                app_name=event_data.app_name,
                full_text=event_data.full_text,
                ocr_confidence=event_data.ocr_conf,
                image_path=event_data.image_path,
                scene_hash=event_data.scene_hash
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
async def get_screen_by_id(
    event_id: int = EnrichParameter(
        description="Screen event ID to retrieve",
        examples=[1, 42, 123]
    )
) -> Optional[ScreenEvent]:
    """Get a specific screen event by ID."""
    try:
        # This would need a new database method to get by ID
        # For now, search recent events and filter
        events_data = await db.get_recent_events(limit=1000, hours=24*7)  # Last week
        
        for event_data in events_data:
            if event_data.id == event_id:
                return ScreenEvent(
                    id=event_data.id,
                    timestamp=event_data.ts,
                    window_title=event_data.window_title,
                    app_name=event_data.app_name,
                    full_text=event_data.full_text,
                    ocr_confidence=event_data.ocr_conf,
                    image_path=event_data.image_path,
                    scene_hash=event_data.scene_hash
                )
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to get screen {event_id}: {e}")
        return None


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
    try:
        # This would use the error_events table once error extraction is implemented
        error_events = await db.get_recent_errors(window_minutes=since_hours*60)
        
        # Convert to ErrorEvent models and apply filters
        results = []
        for error_data in error_events:
            if severity and error_data.severity != severity:
                continue
                
            results.append(ErrorEvent(
                id=error_data.id,
                event_id=error_data.event_id,
                timestamp=error_data.ts,
                error_type=error_data.error_type,
                error_message=error_data.error_msg,
                app_name=error_data.app_name,
                severity=error_data.severity
            ))
            
            if len(results) >= limit:
                break
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to find error screens: {e}")
        return []


@app.retrieve
async def analyze_screen_context(
    query: str = EnrichParameter(
        description="Question about screen content or context",
        examples=[
            "What was I working on 10 minutes ago?",
            "Show me login forms from today",
            "Find error messages in Terminal"
        ]
    ),
    max_events: int = EnrichParameter(
        default=5,
        description="Maximum events to analyze for context",
        examples=[3, 5, 10]
    )
) -> Dict[str, Any]:
    """Analyze screen context using AI to answer questions about captured content."""
    try:
        # This would use server-side LLM sampling via EnrichMCP
        ctx = app.get_context()
        
        # Search for relevant screens
        search_results = await search_screens(query, limit=max_events)
        
        if not search_results.events:
            return {
                "answer": "No relevant screen content found for your query.",
                "confidence": 0.0,
                "events_analyzed": 0,
                "query": query
            }
        
        # Prepare context for LLM
        context_text = f"User query: {query}\n\nRelevant screen captures:\n"
        for i, event in enumerate(search_results.events[:max_events], 1):
            context_text += f"\n{i}. [{event.timestamp}] {event.app_name or 'Unknown App'}"
            if event.window_title:
                context_text += f" - {event.window_title}"
            if event.full_text:
                context_text += f"\nText content: {event.full_text[:500]}..."
            context_text += "\n"
        
        # Use server-side LLM sampling
        response = await ctx.ask_llm(
            prompt=f"{context_text}\n\nBased on the screen captures above, please answer the user's query. Be specific and reference the relevant screens.",
            max_tokens=500,
            model_preferences=["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet"]
        )
        
        return {
            "answer": response.text,
            "confidence": 0.8,  # Could be calculated based on search relevance
            "events_analyzed": len(search_results.events),
            "query": query,
            "search_time": search_results.search_time
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze screen context: {e}")
        return {
            "answer": f"Analysis failed: {str(e)}",
            "confidence": 0.0,
            "events_analyzed": 0,
            "query": query
        }


async def initialize_server():
    """Initialize the MCP server components."""
    global capture_system
    
    logger.info("Initializing Screen Memory MCP Server...")
    
    try:
        # Initialize database
        await db.initialize()
        logger.info("Database initialized")
        
        # Initialize capture system
        capture_system = ScreenCapture()
        logger.info("Capture system initialized")
        
        # Test database connection
        if await db.health_check():
            logger.info("Database health check passed")
        else:
            logger.warning("Database health check failed")
            
    except Exception as e:
        logger.error(f"Server initialization failed: {e}")
        raise


def main():
    """Main entry point for the MCP server."""
    # Set context for OpenRouter API if available
    if os.getenv("OPENROUTER_API_KEY"):
        app.set_context({
            "openrouter_api_key": os.getenv("OPENROUTER_API_KEY"),
            "openrouter_base_url": "https://openrouter.ai/api/v1"
        })
        logger.info("OpenRouter API key configured for server-side LLM sampling")
    
    # Initialize and run the server
    asyncio.run(initialize_server())
    
    # Run the EnrichMCP server with stdio transport (standard for local MCP)
    logger.info("Starting EnrichMCP server with stdio transport...")
    app.run()


if __name__ == "__main__":
    main() 