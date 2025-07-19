"""
FastAPI server for Screen Memory Assistant with MCP-compatible endpoints
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np

from capture import ScreenCapture
from database import db
from models import (
    ScreenEventResponse, CommandResponse, CalendarEntryResponse, ErrorEventResponse,
    CaptureRequest, SearchRequest, SemanticSearchRequest
)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
capture_system: Optional[ScreenCapture] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global capture_system
    
    # Startup
    logger.info("Starting Screen Memory Assistant API")
    
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
            logger.error("Database health check failed")
        
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    
    # Shutdown
    logger.info("Shutting down Screen Memory Assistant API")


# Create FastAPI app
app = FastAPI(
    title="Screen Memory Assistant",
    description="Local screen capture and memory with ML-powered search",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """System health check"""
    db_healthy = await db.health_check()
    capture_healthy = capture_system.health_check() if capture_system else {}
    
    return {
        "status": "healthy" if db_healthy else "unhealthy",
        "database": "connected" if db_healthy else "disconnected",
        "capture_system": capture_healthy,
        "timestamp": datetime.utcnow()
    }


# Background processing using FastAPI BackgroundTasks
import uuid
from typing import Dict, Any

# Store for tracking async captures (in production, use Redis or similar)
capture_status: Dict[str, Dict[str, Any]] = {}

async def process_capture_background(capture_id: str, save_image: bool, force_vision: bool):
    """Process capture in background using async/await"""
    try:
        capture_status[capture_id]["status"] = "processing"
        
        # Capture screen (this is already async)
        capture_data = await capture_system.capture_screen(
            save_image=save_image,
            force_vision=force_vision
        )
        
        # Save to database (this is already async)
        event_id = await db.save_screen_event(capture_data)
        
        # Update status
        capture_status[capture_id].update({
            "status": "completed",
            "event_id": event_id,
            "capture_data": capture_data
        })
        
        logger.info(f"Capture {capture_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Capture {capture_id} failed: {e}")
        capture_status[capture_id].update({
            "status": "failed",
            "error": str(e)
        })

# MCP-compatible endpoints
@app.post("/capture_now")
async def capture_now(background_tasks: BackgroundTasks, request: CaptureRequest = CaptureRequest()):
    """
    MCP Tool: capture_now()
    Take an immediate screenshot and process it asynchronously
    """
    if not capture_system:
        raise HTTPException(status_code=500, detail="Capture system not initialized")
    
    try:
        # Generate unique capture ID
        capture_id = str(uuid.uuid4())[:8]
        
        # Initialize status
        capture_status[capture_id] = {
            "status": "started",
            "timestamp": datetime.utcnow(),
            "save_image": request.save_image,
            "force_vision": request.force_vision
        }
        
        # Add to FastAPI background tasks (better than asyncio.create_task)
        background_tasks.add_task(
            process_capture_background, 
            capture_id, 
            request.save_image, 
            request.force_vision
        )
        
        # Return immediately
        return {
            "capture_id": capture_id,
            "status": "started",
            "message": "Capture initiated - processing in background",
            "timestamp": datetime.utcnow()
        }
            
    except Exception as e:
        logger.error(f"Capture failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/capture_status/{capture_id}")
async def get_capture_status(capture_id: str):
    """Get the status of an async capture"""
    if capture_id not in capture_status:
        raise HTTPException(status_code=404, detail="Capture ID not found")
    
    return capture_status[capture_id]


@app.get("/active_captures")
async def get_active_captures():
    """Get all active/recent captures"""
    return {
        "active_captures": len([c for c in capture_status.values() if c["status"] in ["started", "processing"]]),
        "total_captures": len(capture_status),
        "captures": capture_status
    }


@app.post("/find", response_model=List[ScreenEventResponse])
async def find_events(request: SearchRequest):
    """
    MCP Tool: find(pattern: str, since_min: int = 60)
    Full-text search of screen events
    """
    try:
        results = await db.search_events(
            query=request.query,
            limit=request.limit,
            since_minutes=request.since_minutes,
            app_name=request.app_name
        )
        return results
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search_semantic", response_model=List[ScreenEventResponse])
async def search_semantic(request: SemanticSearchRequest):
    """
    MCP Tool: search_semantic(query: str, k: int = 5)
    Semantic vector search using CLIP embeddings
    """
    if not capture_system or not hasattr(capture_system, 'clip_model'):
        raise HTTPException(status_code=501, detail="CLIP not available for semantic search")
    
    try:
        # Generate query embedding
        import clip
        import torch
        from PIL import Image
        
        # For text queries, we need to use CLIP's text encoder
        text_tokens = clip.tokenize([request.query]).to(capture_system.clip_device)
        
        with torch.no_grad():
            text_features = capture_system.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        query_embedding = text_features.cpu().numpy().flatten()
        
        # Search database
        results = await db.semantic_search(
            query_embedding=query_embedding,
            k=request.k,
            threshold=request.threshold
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recent_errors", response_model=List[ErrorEventResponse])
async def recent_errors(window_min: int = 30):
    """
    MCP Tool: recent_errors(window_min: int = 30)
    Get recent error events
    """
    try:
        errors = await db.get_recent_errors(window_minutes=window_min)
        return errors
        
    except Exception as e:
        logger.error(f"Failed to get recent errors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/last_docker", response_model=Optional[CommandResponse])
async def last_docker():
    """
    MCP Tool: last_docker()
    Get the most recent Docker command
    """
    try:
        command = await db.get_last_command(command_type="docker")
        return command
        
    except Exception as e:
        logger.error(f"Failed to get last docker command: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/calendar_between", response_model=List[CalendarEntryResponse])
async def calendar_between(start_ts: datetime, end_ts: datetime):
    """
    MCP Tool: calendar_between(start_ts, end_ts)
    Get calendar entries in time range
    """
    try:
        entries = await db.get_calendar_entries(start_time=start_ts, end_time=end_ts)
        return entries
        
    except Exception as e:
        logger.error(f"Failed to get calendar entries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Additional utility endpoints
@app.get("/recent", response_model=List[ScreenEventResponse])
async def get_recent_events(limit: int = 20, hours: int = 24):
    """Get recent screen events"""
    try:
        events = await db.get_recent_events(limit=limit, hours=hours)
        return events
        
    except Exception as e:
        logger.error(f"Failed to get recent events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        db_stats = await db.get_stats()
        capture_health = capture_system.health_check() if capture_system else {}
        
        return {
            "database": db_stats,
            "capture_system": capture_health,
            "uptime": "unknown"  # TODO: Track actual uptime
        }
        
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cleanup")
async def cleanup_old_events(days: int = 30):
    """Clean up old events"""
    try:
        count = await db.cleanup_old_events(days=days)
        return {"cleaned_up": count, "older_than_days": days}
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background processing endpoints
@app.post("/start_daemon")
async def start_daemon(interval_seconds: int = 2, background_tasks: BackgroundTasks = None):
    """Start background capture daemon"""
    
    async def capture_daemon():
        """Background task for continuous capture"""
        if not capture_system:
            logger.error("Capture system not initialized")
            return
            
        logger.info(f"Starting capture daemon with {interval_seconds}s interval")
        
        try:
            while True:
                # Capture screen
                capture_data = await capture_system.capture_screen(save_image=True)
                
                # Save to database
                event_id = await db.save_screen_event(capture_data)
                
                # TODO: Process derived events (commands, calendar, errors)
                # This would be done by background parsers
                
                logger.debug(f"Captured event {event_id}")
                
                # Sleep
                await asyncio.sleep(interval_seconds)
                
        except asyncio.CancelledError:
            logger.info("Capture daemon cancelled")
        except Exception as e:
            logger.error(f"Capture daemon error: {e}")
    
    # Add background task
    if background_tasks:
        background_tasks.add_task(capture_daemon)
    
    return {"status": "daemon_started", "interval_seconds": interval_seconds}


# Text processing for derived events (commands, calendar, errors)
async def process_derived_events(event_id: int, full_text: str):
    """Process text to extract commands, calendar entries, and errors"""
    # This is a placeholder - in the full implementation, this would use
    # regex patterns or LLM parsing to extract structured data
    
    # Command extraction (simple regex)
    import re
    
    # Docker commands
    docker_match = re.search(r'(docker\s+\w+.*)', full_text, re.IGNORECASE)
    if docker_match:
        await db.save_command(event_id, {
            'cmd': docker_match.group(1),
            'shell': 'bash'
        })
    
    # Git commands  
    git_match = re.search(r'(git\s+\w+.*)', full_text, re.IGNORECASE)
    if git_match:
        await db.save_command(event_id, {
            'cmd': git_match.group(1),
            'shell': 'bash'
        })
    
    # Error detection
    error_patterns = [
        r'error[:;]?\s*(.*)',
        r'exception[:;]?\s*(.*)',
        r'failed[:;]?\s*(.*)',
        r'cannot\s+(.*)',
    ]
    
    for pattern in error_patterns:
        error_match = re.search(pattern, full_text, re.IGNORECASE)
        if error_match:
            await db.save_error_event(event_id, {
                'error_msg': error_match.group(1),
                'error_type': 'general',
                'severity': 'medium'
            })
            break


# Test endpoint for development
@app.get("/test_capture")
async def test_capture():
    """Test endpoint to verify capture works"""
    if not capture_system:
        raise HTTPException(status_code=500, detail="Capture system not initialized")
    
    try:
        # Test basic capture
        capture_data = await capture_system.capture_screen(save_image=False)
        
        return {
            "status": "success",
            "capture_data": {
                "text_length": len(capture_data.get('full_text', '')),
                "ocr_conf": capture_data.get('ocr_conf'),
                "has_embedding": capture_data.get('clip_vec') is not None,
                "window_title": capture_data.get('window_title'),
                "app_name": capture_data.get('app_name')
            },
            "health": capture_system.health_check()
        }
        
    except Exception as e:
        logger.error(f"Test capture failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5003, reload=True) 