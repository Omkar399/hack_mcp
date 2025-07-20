"""
Unified Processing Pipeline for Eidolon AI Personal Assistant

Orchestrates real-time processing of screenshots through all Phase 1-7 features:
- Screenshot capture and monitoring (Phase 1-2)
- Local vision analysis (Phase 3)
- Cloud AI and semantic memory (Phase 4-5)
- MCP and autonomous agency (Phase 6-7)
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json

from ..utils.logging import get_component_logger, log_performance, log_exceptions
from ..utils.config import get_config
from .observer import Observer, Screenshot
from .analyzer import AnalysisEngine
from .memory import MemorySystem
from .interface import Interface
from .mcp_server import EidolonMCPServer
from ..models.decision_engine import DecisionEngine
from ..autonomous.task_executor import TaskExecutor
from ..autonomous.safety_monitor import SafetyMonitor
from ..planning.task_planner import TaskPlanner
from ..proactive.pattern_recognizer import PatternRecognizer
from ..proactive.predictive_assistant import PredictiveAssistant


class ProcessingMode(Enum):
    """Pipeline processing modes"""
    CONTINUOUS = "continuous"
    TRIGGERED = "triggered"
    BATCH = "batch"
    SMART = "smart"


class ProcessingStage(Enum):
    """Processing pipeline stages"""
    CAPTURE = "capture"
    ANALYSIS = "analysis"
    MEMORY = "memory"
    INTELLIGENCE = "intelligence"
    ACTION = "action"


@dataclass
class ProcessingResult:
    """Result from pipeline processing"""
    screenshot_id: str
    timestamp: datetime
    stages_completed: List[ProcessingStage] = field(default_factory=list)
    processing_time: float = 0.0
    ocr_text: str = ""
    content_analysis: Dict[str, Any] = field(default_factory=dict)
    vision_analysis: Dict[str, Any] = field(default_factory=dict)
    cloud_analysis: Dict[str, Any] = field(default_factory=dict)
    memory_stored: bool = False
    memory_id: Optional[str] = None
    patterns_detected: List[Dict[str, Any]] = field(default_factory=list)
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    autonomous_actions: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class UnifiedProcessingPipeline:
    """Unified pipeline that processes screenshots through all Eidolon features"""
    
    def __init__(self):
        self.logger = get_component_logger("processing_pipeline")
        self.config = get_config()
        
        # Core components initialization
        self.observer = None
        self.analyzer = None
        self.memory = None
        self.interface = None
        self.mcp_server = None
        self.decision_engine = None
        self.task_executor = None
        self.safety_monitor = None
        self.task_planner = None
        self.pattern_recognizer = None
        self.predictive_assistant = None
        
        # Pipeline state
        self.is_running = False
        self.processing_mode = ProcessingMode.SMART
        self.processing_queue = asyncio.Queue()
        self.active_processors = 0
        self.max_concurrent_processors = 3
        
        # Performance metrics
        self.metrics = {
            "screenshots_processed": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "errors_count": 0,
            "memory_items_stored": 0,
            "patterns_detected": 0,
            "actions_executed": 0,
            "last_processing_time": None
        }
        
        # Processing results history
        self.processing_history: List[ProcessingResult] = []
        self.max_history = 1000
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            "screenshot_captured": [],
            "analysis_completed": [],
            "memory_stored": [],
            "pattern_detected": [],
            "action_triggered": [],
            "error_occurred": []
        }
        
        self.logger.info("UnifiedProcessingPipeline initialized")
    
    async def initialize(self):
        """Initialize all pipeline components"""
        self.logger.info("Initializing processing pipeline components...")
        
        try:
            # Initialize core components
            self.observer = Observer()
            self.analyzer = AnalysisEngine()
            self.memory = MemorySystem()
            await self.memory.initialize()
            
            self.interface = Interface()
            self.mcp_server = EidolonMCPServer()
            
            # Initialize AI components
            self.decision_engine = DecisionEngine()
            self.task_executor = TaskExecutor()
            self.safety_monitor = SafetyMonitor()
            self.task_planner = TaskPlanner()
            
            # Initialize Phase 7 components
            self.pattern_recognizer = PatternRecognizer()
            self.predictive_assistant = PredictiveAssistant()
            
            self.logger.info("All pipeline components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline components: {e}")
            raise
    
    async def start_continuous_processing(self):
        """Start continuous background processing"""
        if self.is_running:
            self.logger.warning("Pipeline is already running")
            return
        
        self.logger.info("Starting continuous processing pipeline...")
        self.is_running = True
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._screenshot_capture_loop()),
            asyncio.create_task(self._processing_worker_loop()),
            asyncio.create_task(self._pattern_analysis_loop()),
            asyncio.create_task(self._predictive_analysis_loop()),
            asyncio.create_task(self._autonomous_action_loop()),
        ]
        
        # Start MCP server
        await self.mcp_server.start()
        
        # Emit start event
        await self._emit_event("pipeline_started", {"timestamp": datetime.now().isoformat()})
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
            await self.stop()
    
    async def stop(self):
        """Stop the processing pipeline"""
        self.logger.info("Stopping processing pipeline...")
        self.is_running = False
        
        # Stop observer
        if self.observer:
            self.observer.stop_monitoring()
        
        # Stop MCP server
        if self.mcp_server:
            await self.mcp_server.stop()
        
        await self._emit_event("pipeline_stopped", {"timestamp": datetime.now().isoformat()})
        self.logger.info("Processing pipeline stopped")
    
    async def _screenshot_capture_loop(self):
        """Main screenshot capture and queuing loop"""
        self.logger.info("Starting screenshot capture loop")
        
        # Start observer monitoring
        self.observer.start_monitoring()
        
        # Register screenshot callback
        def on_screenshot_captured(screenshot: Screenshot):
            asyncio.create_task(self._queue_screenshot_for_processing(screenshot))
        
        self.observer.register_callback(on_screenshot_captured)
        
        # Keep the loop alive
        while self.is_running:
            await asyncio.sleep(1)
    
    async def _queue_screenshot_for_processing(self, screenshot: Screenshot):
        """Queue a screenshot for processing"""
        try:
            await self.processing_queue.put(screenshot)
            await self._emit_event("screenshot_captured", {
                "screenshot_id": screenshot.id,
                "timestamp": screenshot.timestamp.isoformat(),
                "path": str(screenshot.path)
            })
        except Exception as e:
            self.logger.error(f"Failed to queue screenshot for processing: {e}")
    
    async def _processing_worker_loop(self):
        """Main processing worker loop"""
        self.logger.info("Starting processing worker loop")
        
        while self.is_running:
            try:
                # Check if we can start a new processor
                if self.active_processors >= self.max_concurrent_processors:
                    await asyncio.sleep(0.1)
                    continue
                
                # Get next screenshot from queue
                try:
                    screenshot = await asyncio.wait_for(
                        self.processing_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process screenshot in background
                self.active_processors += 1
                asyncio.create_task(self._process_screenshot_full_pipeline(screenshot))
                
            except Exception as e:
                self.logger.error(f"Processing worker error: {e}")
                await asyncio.sleep(1)
    
    @log_exceptions()
    @log_performance("screenshot_processing")
    async def _process_screenshot_full_pipeline(self, screenshot: Screenshot):
        """Process screenshot through the complete pipeline"""
        start_time = time.time()
        result = ProcessingResult(
            screenshot_id=screenshot.id,
            timestamp=screenshot.timestamp
        )
        
        try:
            self.logger.debug(f"Processing screenshot {screenshot.id} through full pipeline")
            
            # Stage 1: Content Analysis (Phase 1-2)
            result.stages_completed.append(ProcessingStage.CAPTURE)
            
            # Extract OCR text
            if hasattr(screenshot, 'extracted_text'):
                result.ocr_text = screenshot.extracted_text
            else:
                # Fallback OCR extraction
                ocr_result = await self.analyzer.extract_text_async(screenshot.path)
                result.ocr_text = ocr_result.get('text', '')
            
            # Stage 2: AI Analysis (Phase 3-4)
            result.stages_completed.append(ProcessingStage.ANALYSIS)
            
            # Local vision analysis
            if self.analyzer.vision_analyzer:
                vision_result = await self.analyzer.vision_analyzer.analyze_image_async(screenshot.path)
                result.vision_analysis = vision_result
            
            # Cloud AI analysis (intelligent routing)
            routing_decision = await self.decision_engine.should_use_cloud_analysis(
                screenshot, result.ocr_text, result.vision_analysis
            )
            
            if routing_decision['use_cloud']:
                cloud_result = await self.analyzer.cloud_analyzer.analyze_content_async(
                    screenshot.path, result.ocr_text, context="screenshot_analysis"
                )
                result.cloud_analysis = cloud_result
            
            # Stage 3: Memory Storage (Phase 4-5)
            result.stages_completed.append(ProcessingStage.MEMORY)
            
            # Prepare content for memory storage
            content_data = {
                "screenshot_id": screenshot.id,
                "timestamp": screenshot.timestamp.isoformat(),
                "file_path": str(screenshot.path),
                "ocr_text": result.ocr_text,
                "vision_analysis": result.vision_analysis,
                "cloud_analysis": result.cloud_analysis,
                "content_type": "screenshot"
            }
            
            # Store in vector memory
            memory_result = await self.memory.store_with_embeddings(
                content=result.ocr_text,
                metadata=content_data,
                content_type="screenshot"
            )
            
            if memory_result:
                result.memory_stored = True
                result.memory_id = memory_result
                self.metrics["memory_items_stored"] += 1
            
            # Stage 4: Intelligence Analysis (Phase 7)
            result.stages_completed.append(ProcessingStage.INTELLIGENCE)
            
            # Pattern recognition
            patterns = await self.pattern_recognizer.analyze_screenshot_patterns(
                screenshot, result.ocr_text, result.vision_analysis
            )
            result.patterns_detected = patterns
            self.metrics["patterns_detected"] += len(patterns)
            
            # Predictive analysis
            predictions = await self.predictive_assistant.generate_screenshot_predictions(
                screenshot, result.ocr_text, patterns
            )
            result.predictions = predictions
            
            # Stage 5: Autonomous Actions (Phase 6-7)
            result.stages_completed.append(ProcessingStage.ACTION)
            
            # Safety check for autonomous actions
            safety_assessment = await self.safety_monitor.assess_screenshot_safety(
                screenshot, result.ocr_text, result.vision_analysis
            )
            
            if safety_assessment['safe_for_action']:
                # Generate potential autonomous actions
                actions = await self.task_executor.generate_screenshot_actions(
                    screenshot, result.ocr_text, patterns, predictions
                )
                
                # Execute safe actions
                for action in actions:
                    if action.get('risk_level', 'high') == 'low':
                        action_result = await self.task_executor.execute_action(action)
                        result.autonomous_actions.append({
                            "action": action,
                            "result": action_result,
                            "timestamp": datetime.now().isoformat()
                        })
                        self.metrics["actions_executed"] += 1
            
            # Update processing metrics
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            self.metrics["screenshots_processed"] += 1
            self.metrics["total_processing_time"] += processing_time
            self.metrics["average_processing_time"] = (
                self.metrics["total_processing_time"] / self.metrics["screenshots_processed"]
            )
            self.metrics["last_processing_time"] = datetime.now().isoformat()
            
            # Store result in history
            self.processing_history.append(result)
            if len(self.processing_history) > self.max_history:
                self.processing_history.pop(0)
            
            # Emit completion event
            await self._emit_event("analysis_completed", {
                "screenshot_id": screenshot.id,
                "processing_time": processing_time,
                "stages_completed": [stage.value for stage in result.stages_completed],
                "patterns_detected": len(result.patterns_detected),
                "actions_executed": len(result.autonomous_actions)
            })
            
            self.logger.debug(
                f"Completed processing screenshot {screenshot.id} in {processing_time:.2f}s "
                f"({len(result.stages_completed)} stages)"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing screenshot {screenshot.id}: {e}")
            result.errors.append(str(e))
            self.metrics["errors_count"] += 1
            
            await self._emit_event("error_occurred", {
                "screenshot_id": screenshot.id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })
        
        finally:
            self.active_processors -= 1
    
    async def _pattern_analysis_loop(self):
        """Background pattern analysis loop"""
        self.logger.info("Starting pattern analysis loop")
        
        while self.is_running:
            try:
                # Run pattern analysis every 5 minutes
                await asyncio.sleep(300)
                
                if not self.is_running:
                    break
                
                # Analyze recent patterns
                recent_patterns = await self.pattern_recognizer.get_recent_patterns(hours=1)
                
                if recent_patterns:
                    # Emit pattern detection event
                    await self._emit_event("pattern_detected", {
                        "pattern_count": len(recent_patterns),
                        "timestamp": datetime.now().isoformat(),
                        "analysis_window": "1_hour"
                    })
                
            except Exception as e:
                self.logger.error(f"Pattern analysis loop error: {e}")
                await asyncio.sleep(60)
    
    async def _predictive_analysis_loop(self):
        """Background predictive analysis loop"""
        self.logger.info("Starting predictive analysis loop")
        
        while self.is_running:
            try:
                # Run predictive analysis every 10 minutes
                await asyncio.sleep(600)
                
                if not self.is_running:
                    break
                
                # Generate predictions based on recent activity
                predictions = await self.predictive_assistant.generate_daily_predictions()
                
                if predictions:
                    # Store high-confidence predictions
                    for prediction in predictions:
                        if prediction.get('confidence', 0) > 0.7:
                            await self.memory.store_with_embeddings(
                                content=f"Prediction: {prediction.get('description', '')}",
                                metadata={
                                    "type": "prediction",
                                    "confidence": prediction.get('confidence'),
                                    "prediction_type": prediction.get('type'),
                                    "timestamp": datetime.now().isoformat()
                                },
                                content_type="prediction"
                            )
                
            except Exception as e:
                self.logger.error(f"Predictive analysis loop error: {e}")
                await asyncio.sleep(60)
    
    async def _autonomous_action_loop(self):
        """Background autonomous action evaluation loop"""
        self.logger.info("Starting autonomous action loop")
        
        while self.is_running:
            try:
                # Evaluate potential actions every 15 minutes
                await asyncio.sleep(900)
                
                if not self.is_running:
                    break
                
                # Check for proactive action opportunities
                action_opportunities = await self.task_executor.evaluate_proactive_actions()
                
                for opportunity in action_opportunities:
                    # Safety check
                    safety_result = await self.safety_monitor.assess_action_safety(opportunity)
                    
                    if safety_result['safe'] and safety_result['risk_level'] == 'low':
                        # Execute safe proactive action
                        action_result = await self.task_executor.execute_action(opportunity)
                        
                        await self._emit_event("action_triggered", {
                            "action_type": opportunity.get('type'),
                            "result": action_result,
                            "timestamp": datetime.now().isoformat(),
                            "trigger": "proactive"
                        })
                
            except Exception as e:
                self.logger.error(f"Autonomous action loop error: {e}")
                await asyncio.sleep(60)
    
    async def process_single_screenshot(self, screenshot: Screenshot) -> ProcessingResult:
        """Process a single screenshot through the pipeline (synchronous)"""
        await self._process_screenshot_full_pipeline(screenshot)
        
        # Return the latest result for this screenshot
        for result in reversed(self.processing_history):
            if result.screenshot_id == screenshot.id:
                return result
        
        # Return empty result if not found
        return ProcessingResult(screenshot_id=screenshot.id, timestamp=datetime.now())
    
    async def search_processed_content(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search through processed content using semantic search"""
        if not self.memory:
            return []
        
        return await self.memory.search(query, limit=limit)
    
    async def get_recent_patterns(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent patterns detected by the system"""
        if not self.pattern_recognizer:
            return []
        
        return await self.pattern_recognizer.get_recent_patterns(hours=hours)
    
    async def get_predictions(self, prediction_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get current predictions from the system"""
        if not self.predictive_assistant:
            return []
        
        return await self.predictive_assistant.get_current_predictions(prediction_type)
    
    async def get_processing_metrics(self) -> Dict[str, Any]:
        """Get comprehensive processing metrics"""
        return {
            **self.metrics,
            "is_running": self.is_running,
            "processing_mode": self.processing_mode.value,
            "queue_size": self.processing_queue.qsize(),
            "active_processors": self.active_processors,
            "history_size": len(self.processing_history),
            "uptime": time.time() - getattr(self, '_start_time', time.time())
        }
    
    async def get_recent_activity(self, hours: int = 1) -> List[ProcessingResult]:
        """Get recent processing activity"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            result for result in self.processing_history
            if result.timestamp >= cutoff
        ]
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register an event handler for pipeline events"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event to registered handlers"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event_type, data)
                    else:
                        handler(event_type, data)
                except Exception as e:
                    self.logger.error(f"Event handler error for {event_type}: {e}")
    
    async def set_processing_mode(self, mode: ProcessingMode):
        """Set the processing mode"""
        self.processing_mode = mode
        self.logger.info(f"Processing mode set to: {mode.value}")
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "pipeline": {
                "running": self.is_running,
                "mode": self.processing_mode.value,
                "queue_size": self.processing_queue.qsize(),
                "active_processors": self.active_processors
            },
            "components": {
                "observer": self.observer is not None,
                "analyzer": self.analyzer is not None,
                "memory": self.memory is not None,
                "interface": self.interface is not None,
                "mcp_server": self.mcp_server is not None,
                "decision_engine": self.decision_engine is not None,
                "task_executor": self.task_executor is not None,
                "safety_monitor": self.safety_monitor is not None,
                "pattern_recognizer": self.pattern_recognizer is not None,
                "predictive_assistant": self.predictive_assistant is not None
            },
            "metrics": await self.get_processing_metrics(),
            "last_activity": self.metrics.get("last_processing_time"),
            "uptime": time.time() - getattr(self, '_start_time', time.time())
        }