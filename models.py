"""
SQLAlchemy models for Screen Memory Assistant
"""
from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, BigInteger, String, Text, SmallInteger, DateTime, ForeignKey, CheckConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

# Optional pgvector support
try:
    from pgvector.sqlalchemy import Vector as VECTOR
    PGVECTOR_AVAILABLE = True
except ImportError:
    # Fallback if pgvector not available - use Text for vector storage
    from sqlalchemy import Text as VECTOR
    PGVECTOR_AVAILABLE = False
from pydantic import BaseModel, Field
import json

Base = declarative_base()


class ScreenEvent(Base):
    """Core screen events table - captures everything shown on screen"""
    __tablename__ = 'screen_events'
    
    id = Column(BigInteger, primary_key=True)
    ts = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    window_title = Column(Text)
    app_name = Column(Text)
    full_text = Column(Text)
    ocr_conf = Column(SmallInteger, CheckConstraint('ocr_conf BETWEEN 0 AND 100'))
    clip_vec = Column(Text)  # CLIP embeddings stored as text for now
    image_path = Column(Text)
    scene_hash = Column(String(64))  # For duplicate detection
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    # Relationships to derived tables
    commands = relationship("Command", back_populates="event", cascade="all, delete-orphan")
    calendar_entries = relationship("CalendarEntry", back_populates="event", cascade="all, delete-orphan")
    error_events = relationship("ErrorEvent", back_populates="event", cascade="all, delete-orphan")


class Command(Base):
    """Commands extracted from screen events"""
    __tablename__ = 'commands'
    
    id = Column(BigInteger, primary_key=True)
    event_id = Column(BigInteger, ForeignKey('screen_events.id', ondelete='CASCADE'))
    ts = Column(DateTime(timezone=True), nullable=False)
    cmd = Column(Text, nullable=False)
    args = Column(Text)
    exit_code = Column(SmallInteger)
    shell = Column(Text)
    working_dir = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    event = relationship("ScreenEvent", back_populates="commands")


class CalendarEntry(Base):
    """Calendar entries extracted from screen events"""
    __tablename__ = 'calendar_entries'
    
    id = Column(BigInteger, primary_key=True)
    event_id = Column(BigInteger, ForeignKey('screen_events.id', ondelete='CASCADE'))
    ts = Column(DateTime(timezone=True), nullable=False)
    title = Column(Text, nullable=False)
    event_time = Column(DateTime(timezone=True))
    end_time = Column(DateTime(timezone=True))
    source_app = Column(Text)
    location = Column(Text)
    attendees = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    event = relationship("ScreenEvent", back_populates="calendar_entries")


class ErrorEvent(Base):
    """Error events extracted from screen events"""
    __tablename__ = 'error_events'
    
    id = Column(BigInteger, primary_key=True)
    event_id = Column(BigInteger, ForeignKey('screen_events.id', ondelete='CASCADE'))
    ts = Column(DateTime(timezone=True), nullable=False)
    error_type = Column(Text)
    error_msg = Column(Text)
    app_name = Column(Text)
    severity = Column(Text, CheckConstraint("severity IN ('low', 'medium', 'high', 'critical')"))
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    
    event = relationship("ScreenEvent", back_populates="error_events")


# Pydantic models for API serialization
class ScreenEventResponse(BaseModel):
    """API response model for screen events"""
    id: int
    ts: datetime
    window_title: Optional[str] = None
    app_name: Optional[str] = None  
    full_text: Optional[str] = None
    ocr_conf: Optional[int] = None
    image_path: Optional[str] = None
    scene_hash: Optional[str] = None
    
    class Config:
        from_attributes = True


class CommandResponse(BaseModel):
    """API response model for commands"""
    id: int
    ts: datetime
    cmd: str
    args: Optional[str] = None
    exit_code: Optional[int] = None
    shell: Optional[str] = None
    working_dir: Optional[str] = None
    
    class Config:
        from_attributes = True


class CalendarEntryResponse(BaseModel):
    """API response model for calendar entries"""
    id: int
    ts: datetime
    title: str
    event_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    source_app: Optional[str] = None
    location: Optional[str] = None
    attendees: Optional[str] = None
    
    class Config:
        from_attributes = True


class ErrorEventResponse(BaseModel):
    """API response model for error events"""
    id: int
    ts: datetime
    error_type: Optional[str] = None
    error_msg: Optional[str] = None
    app_name: Optional[str] = None
    severity: Optional[str] = None
    
    class Config:
        from_attributes = True


class CaptureRequest(BaseModel):
    """Request model for manual captures"""
    save_image: bool = True
    force_vision: bool = False  # Force GPT-4o Vision even if OCR confidence is high


class SearchRequest(BaseModel):
    """Request model for searches"""
    query: str
    limit: int = Field(default=10, ge=1, le=100)
    since_minutes: Optional[int] = Field(default=None, ge=1)
    app_name: Optional[str] = None


class SemanticSearchRequest(BaseModel):
    """Request model for semantic vector searches"""
    query: str
    k: int = Field(default=5, ge=1, le=20)
    threshold: float = Field(default=0.7, ge=0.0, le=1.0) 