"""
Database operations for Screen Memory Assistant
"""
import os
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload
from sqlalchemy import select, and_, or_, text, desc, func
import numpy as np

# Optional pgvector support
PGVECTOR_AVAILABLE = True
try:
    from pgvector.sqlalchemy import Vector
except ImportError:
    PGVECTOR_AVAILABLE = False

from models import Base, ScreenEvent, Command, CalendarEntry, ErrorEvent
from models import ScreenEventResponse, CommandResponse, CalendarEntryResponse, ErrorEventResponse

# Logging
logger = logging.getLogger(__name__)


class Database:
    """Database operations with async SQLAlchemy"""
    
    def __init__(self, database_url: str = None):
        if database_url is None:
            database_url = os.getenv(
                'DATABASE_URL', 
                'postgresql+asyncpg://hack:hack123@localhost:5432/screenmemory'
            )
        
        self.engine = create_async_engine(
            database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False  # Set to True for SQL debugging
        )
        
        self.async_session = async_sessionmaker(
            self.engine, 
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def initialize(self):
        """Initialize database - create tables if they don't exist"""
        try:
            async with self.engine.begin() as conn:
                # Check if tables exist and create if needed
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check database connectivity"""
        try:
            async with self.async_session() as session:
                result = await session.execute(text("SELECT 1"))
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session with proper cleanup"""
        async with self.async_session() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def save_screen_event(self, capture_data: Dict[str, Any]) -> int:
        """Save a screen capture event and return the ID"""
        async with self.get_session() as session:
            try:
                # Convert clip_vec list back to numpy array if present
                clip_vec = capture_data.get('clip_vec')
                if clip_vec and isinstance(clip_vec, list):
                    clip_vec = np.array(clip_vec)
                
                event = ScreenEvent(
                    ts=datetime.utcnow(),
                    window_title=capture_data.get('window_title'),
                    app_name=capture_data.get('app_name'),
                    full_text=capture_data.get('full_text'),
                    ocr_conf=capture_data.get('ocr_conf'),
                    clip_vec=clip_vec,
                    image_path=capture_data.get('image_path'),
                    scene_hash=capture_data.get('scene_hash')
                )
                
                session.add(event)
                await session.commit()
                await session.refresh(event)
                
                logger.info(f"Saved screen event {event.id}")
                return event.id
                
            except Exception as e:
                logger.error(f"Failed to save screen event: {e}")
                raise
    
    async def search_events(self, 
                          query: str, 
                          limit: int = 10,
                          since_minutes: Optional[int] = None,
                          app_name: Optional[str] = None) -> List[ScreenEventResponse]:
        """Full text search of screen events"""
        async with self.get_session() as session:
            try:
                # Base query
                stmt = select(ScreenEvent)
                
                # Add text search filter
                if query:
                    stmt = stmt.where(
                        or_(
                            ScreenEvent.full_text.ilike(f"%{query}%"),
                            func.to_tsvector('english', ScreenEvent.full_text).match(query)
                        )
                    )
                
                # Add time filter
                if since_minutes:
                    since_time = datetime.utcnow() - timedelta(minutes=since_minutes)
                    stmt = stmt.where(ScreenEvent.ts >= since_time)
                
                # Add app filter
                if app_name:
                    stmt = stmt.where(ScreenEvent.app_name.ilike(f"%{app_name}%"))
                
                # Order by relevance and time
                stmt = stmt.order_by(desc(ScreenEvent.ts)).limit(limit)
                
                result = await session.execute(stmt)
                events = result.scalars().all()
                
                return [ScreenEventResponse.from_orm(event) for event in events]
                
            except Exception as e:
                logger.error(f"Search failed: {e}")
                return []
    
    async def semantic_search(self, 
                            query_embedding: np.ndarray, 
                            k: int = 5,
                            threshold: float = 0.7) -> List[ScreenEventResponse]:
        """Vector similarity search using CLIP embeddings"""
        if not PGVECTOR_AVAILABLE:
            logger.warning("pgvector not available - semantic search disabled")
            return []
            
        async with self.get_session() as session:
            try:
                # Convert numpy array to pgvector format
                query_vec = query_embedding.tolist()
                
                # Use pgvector cosine similarity
                stmt = select(
                    ScreenEvent,
                    (1 - func.cosine_distance(ScreenEvent.clip_vec, query_vec)).label('similarity')
                ).where(
                    ScreenEvent.clip_vec.is_not(None)
                ).order_by(
                    func.cosine_distance(ScreenEvent.clip_vec, query_vec)
                ).limit(k)
                
                result = await session.execute(stmt)
                rows = result.all()
                
                # Filter by similarity threshold
                events = []
                for row in rows:
                    event, similarity = row
                    if similarity >= threshold:
                        events.append(ScreenEventResponse.from_orm(event))
                
                return events
                
            except Exception as e:
                logger.error(f"Semantic search failed: {e}")
                return []
    
    async def get_recent_events(self, 
                              limit: int = 50, 
                              hours: int = 24) -> List[ScreenEventResponse]:
        """Get recent screen events"""
        async with self.get_session() as session:
            try:
                since_time = datetime.utcnow() - timedelta(hours=hours)
                
                stmt = select(ScreenEvent).where(
                    ScreenEvent.ts >= since_time
                ).order_by(desc(ScreenEvent.ts)).limit(limit)
                
                result = await session.execute(stmt)
                events = result.scalars().all()
                
                return [ScreenEventResponse.from_orm(event) for event in events]
                
            except Exception as e:
                logger.error(f"Failed to get recent events: {e}")
                return []
    
    async def get_recent_errors(self, window_minutes: int = 30) -> List[ErrorEventResponse]:
        """Get recent error events"""
        async with self.get_session() as session:
            try:
                since_time = datetime.utcnow() - timedelta(minutes=window_minutes)
                
                stmt = select(ErrorEvent).where(
                    ErrorEvent.ts >= since_time
                ).order_by(desc(ErrorEvent.ts))
                
                result = await session.execute(stmt)
                errors = result.scalars().all()
                
                return [ErrorEventResponse.from_orm(error) for error in errors]
                
            except Exception as e:
                logger.error(f"Failed to get recent errors: {e}")
                return []
    
    async def get_last_command(self, command_type: str = None) -> Optional[CommandResponse]:
        """Get the most recent command, optionally filtered by type"""
        async with self.get_session() as session:
            try:
                stmt = select(Command)
                
                if command_type:
                    stmt = stmt.where(Command.cmd.ilike(f"%{command_type}%"))
                
                stmt = stmt.order_by(desc(Command.ts)).limit(1)
                
                result = await session.execute(stmt)
                command = result.scalar_one_or_none()
                
                return CommandResponse.from_orm(command) if command else None
                
            except Exception as e:
                logger.error(f"Failed to get last command: {e}")
                return None
    
    async def get_calendar_entries(self, 
                                 start_time: datetime, 
                                 end_time: datetime) -> List[CalendarEntryResponse]:
        """Get calendar entries in time range"""
        async with self.get_session() as session:
            try:
                stmt = select(CalendarEntry).where(
                    and_(
                        CalendarEntry.event_time >= start_time,
                        CalendarEntry.event_time <= end_time
                    )
                ).order_by(CalendarEntry.event_time)
                
                result = await session.execute(stmt)
                entries = result.scalars().all()
                
                return [CalendarEntryResponse.from_orm(entry) for entry in entries]
                
            except Exception as e:
                logger.error(f"Failed to get calendar entries: {e}")
                return []
    
    async def save_command(self, event_id: int, cmd_data: Dict[str, Any]) -> int:
        """Save a parsed command"""
        async with self.get_session() as session:
            try:
                command = Command(
                    event_id=event_id,
                    ts=cmd_data.get('ts', datetime.utcnow()),
                    cmd=cmd_data['cmd'],
                    args=cmd_data.get('args'),
                    exit_code=cmd_data.get('exit_code'),
                    shell=cmd_data.get('shell'),
                    working_dir=cmd_data.get('working_dir')
                )
                
                session.add(command)
                await session.commit()
                await session.refresh(command)
                
                return command.id
                
            except Exception as e:
                logger.error(f"Failed to save command: {e}")
                raise
    
    async def save_calendar_entry(self, event_id: int, cal_data: Dict[str, Any]) -> int:
        """Save a parsed calendar entry"""
        async with self.get_session() as session:
            try:
                entry = CalendarEntry(
                    event_id=event_id,
                    ts=cal_data.get('ts', datetime.utcnow()),
                    title=cal_data['title'],
                    event_time=cal_data.get('event_time'),
                    end_time=cal_data.get('end_time'),
                    source_app=cal_data.get('source_app'),
                    location=cal_data.get('location'),
                    attendees=cal_data.get('attendees')
                )
                
                session.add(entry)
                await session.commit()
                await session.refresh(entry)
                
                return entry.id
                
            except Exception as e:
                logger.error(f"Failed to save calendar entry: {e}")
                raise
    
    async def save_error_event(self, event_id: int, error_data: Dict[str, Any]) -> int:
        """Save a parsed error event"""
        async with self.get_session() as session:
            try:
                error = ErrorEvent(
                    event_id=event_id,
                    ts=error_data.get('ts', datetime.utcnow()),
                    error_type=error_data.get('error_type'),
                    error_msg=error_data['error_msg'],
                    app_name=error_data.get('app_name'),
                    severity=error_data.get('severity', 'medium')
                )
                
                session.add(error)
                await session.commit()
                await session.refresh(error)
                
                return error.id
                
            except Exception as e:
                logger.error(f"Failed to save error event: {e}")
                raise
    
    async def cleanup_old_events(self, days: int = 30) -> int:
        """Clean up events older than specified days"""
        async with self.get_session() as session:
            try:
                cutoff_time = datetime.utcnow() - timedelta(days=days)
                
                stmt = select(ScreenEvent).where(ScreenEvent.ts < cutoff_time)
                result = await session.execute(stmt)
                old_events = result.scalars().all()
                
                count = len(old_events)
                
                for event in old_events:
                    await session.delete(event)
                
                await session.commit()
                logger.info(f"Cleaned up {count} old events")
                
                return count
                
            except Exception as e:
                logger.error(f"Cleanup failed: {e}")
                return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        async with self.get_session() as session:
            try:
                # Count events by time periods
                now = datetime.utcnow()
                last_hour = now - timedelta(hours=1)
                last_day = now - timedelta(days=1)
                last_week = now - timedelta(weeks=1)
                
                stats = {}
                
                # Total events
                total_result = await session.execute(select(func.count(ScreenEvent.id)))
                stats['total_events'] = total_result.scalar()
                
                # Events in last hour
                hour_result = await session.execute(
                    select(func.count(ScreenEvent.id)).where(ScreenEvent.ts >= last_hour)
                )
                stats['events_last_hour'] = hour_result.scalar()
                
                # Events in last day
                day_result = await session.execute(
                    select(func.count(ScreenEvent.id)).where(ScreenEvent.ts >= last_day)
                )
                stats['events_last_day'] = day_result.scalar()
                
                # Events in last week
                week_result = await session.execute(
                    select(func.count(ScreenEvent.id)).where(ScreenEvent.ts >= last_week)
                )
                stats['events_last_week'] = week_result.scalar()
                
                # Average OCR confidence
                conf_result = await session.execute(
                    select(func.avg(ScreenEvent.ocr_conf)).where(ScreenEvent.ocr_conf.is_not(None))
                )
                avg_conf = conf_result.scalar()
                stats['avg_ocr_confidence'] = float(avg_conf) if avg_conf else None
                
                # Events with CLIP embeddings
                clip_result = await session.execute(
                    select(func.count(ScreenEvent.id)).where(ScreenEvent.clip_vec.is_not(None))
                )
                stats['events_with_embeddings'] = clip_result.scalar()
                
                return stats
                
            except Exception as e:
                logger.error(f"Failed to get stats: {e}")
                return {}


# Global database instance
db = Database() 