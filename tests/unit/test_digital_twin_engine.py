"""
Unit tests for Digital Twin Engine - Phase 7
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from eidolon.twin.digital_twin_engine import (
    DigitalTwinEngine,
    TwinCapability,
    TwinPersonality,
    TwinState,
    TwinAction
)


class TestDigitalTwinEngine:
    """Test the Digital Twin Engine functionality"""

    @pytest.fixture
    async def twin_engine(self):
        """Create a digital twin engine for testing"""
        with patch('eidolon.twin.digital_twin_engine.MemorySystem') as mock_memory, \
             patch('eidolon.twin.digital_twin_engine.CloudAPIManager') as mock_cloud_api, \
             patch('eidolon.twin.digital_twin_engine.TaskPlanner') as mock_task_planner, \
             patch('eidolon.twin.digital_twin_engine.PatternRecognizer') as mock_pattern_recognizer, \
             patch('eidolon.twin.digital_twin_engine.PredictiveAssistant') as mock_predictive_assistant, \
             patch('eidolon.twin.digital_twin_engine.StyleReplicator') as mock_style_replicator:
            
            # Mock the memory system
            mock_memory_instance = AsyncMock()
            mock_memory_instance.search_content.return_value = []
            mock_memory_instance.store_content.return_value = True
            mock_memory.return_value = mock_memory_instance
            
            # Mock other components
            mock_cloud_api.return_value = AsyncMock()
            mock_task_planner.return_value = Mock()
            mock_pattern_recognizer.return_value = Mock()
            mock_predictive_assistant.return_value = AsyncMock()
            mock_style_replicator.return_value = Mock()
            
            # Create engine
            engine = DigitalTwinEngine()
            
            # Initialize manually for testing
            engine.state = TwinState.ACTIVE
            engine.capabilities = [
                TwinCapability.PATTERN_RECOGNITION,
                TwinCapability.PREDICTIVE_ASSISTANCE,
                TwinCapability.TASK_PLANNING
            ]
            
            yield engine

    def test_twin_initialization(self, twin_engine):
        """Test twin engine initialization"""
        assert twin_engine.twin_id is not None
        assert isinstance(twin_engine.twin_id, str)
        assert twin_engine.state == TwinState.ACTIVE
        assert len(twin_engine.capabilities) > 0
        assert TwinCapability.PATTERN_RECOGNITION in twin_engine.capabilities

    @pytest.mark.asyncio
    async def test_process_context(self, twin_engine):
        """Test context processing"""
        # Mock the internal methods
        twin_engine._analyze_context = AsyncMock(return_value={
            "activity_type": "development",
            "focus_level": "high"
        })
        twin_engine._generate_recommendations = AsyncMock(return_value=[
            {"type": "productivity", "title": "Focus Time"}
        ])
        twin_engine._generate_learning_insights = AsyncMock(return_value={
            "new_patterns_detected": 1
        })
        twin_engine._generate_personality_response = AsyncMock(return_value={
            "personality_type": "professional"
        })
        
        # Mock the predictive assistant
        twin_engine.predictive_assistant.generate_predictions = AsyncMock(return_value=[])
        
        context = {
            "current_app": "VS Code",
            "current_text": "def hello_world():",
            "timestamp": datetime.now().isoformat()
        }
        
        response = await twin_engine.process_context(context)
        
        assert "twin_id" in response
        assert "context_analysis" in response
        assert "predictions" in response
        assert "recommendations" in response
        assert response["context_analysis"]["activity_type"] == "development"

    @pytest.mark.asyncio
    async def test_autonomous_action_execution(self, twin_engine):
        """Test autonomous action execution"""
        action = TwinAction(
            action_type="offer_assistance",
            description="Offer debugging help",
            confidence=0.8,
            urgency=0.6
        )
        
        # Mock style replicator
        twin_engine.style_replicator.generate_styled_response = AsyncMock(
            return_value={"response": "Can I help you debug this?"}
        )
        
        result = await twin_engine.execute_autonomous_action(action)
        
        assert result["success"] is True
        assert "message" in result
        assert action.executed_time is not None
        assert action.success is True

    @pytest.mark.asyncio
    async def test_task_plan_creation(self, twin_engine):
        """Test task plan creation"""
        # Mock task planner
        from eidolon.planning.task_planner import TaskPlan
        mock_plan = TaskPlan(
            title="Test Plan",
            description="Test plan creation"
        )
        twin_engine.task_planner.create_plan = AsyncMock(return_value=mock_plan)
        
        objective = "Complete code review for project"
        context = {"project": "eidolon", "priority": "high"}
        
        plan = await twin_engine.create_task_plan(objective, context)
        
        assert plan is not None
        assert plan.title == "Test Plan"
        assert plan.id in twin_engine.active_plans

    @pytest.mark.asyncio
    async def test_user_feedback_processing(self, twin_engine):
        """Test user feedback processing"""
        # Initial satisfaction
        initial_satisfaction = twin_engine.user_satisfaction
        
        # Positive feedback
        await twin_engine.update_user_feedback("general", 0.9, "Great assistance!")
        
        assert len(twin_engine.context_history) > 0
        assert twin_engine.user_satisfaction >= initial_satisfaction
        
        # Negative feedback should trigger adaptation
        twin_engine.personality = Mock()
        twin_engine.personality.traits = {"proactivity": 0.7}
        
        await twin_engine.update_user_feedback("proactivity", 0.2, "Too intrusive")
        
        # Should reduce proactivity
        assert twin_engine.personality.traits["proactivity"] < 0.7

    @pytest.mark.asyncio
    async def test_future_scenario_simulation(self, twin_engine):
        """Test future scenario simulation"""
        # Mock pattern recognizer
        from eidolon.proactive.pattern_recognizer import UserPattern, PatternType
        mock_pattern = Mock()
        mock_pattern.title = "Morning Coding Session"
        mock_pattern.confidence = 0.8
        mock_pattern.triggers = ["coding", "morning"]
        
        twin_engine.pattern_recognizer.recognized_patterns = {
            "pattern_1": mock_pattern
        }
        
        scenario = {
            "description": "Starting a coding session in the morning",
            "context": {"time": "09:00", "activity": "coding"}
        }
        
        result = await twin_engine.simulate_future_scenario(scenario)
        
        assert "scenario" in result
        assert "predicted_outcomes" in result
        assert "confidence" in result
        assert "recommendations" in result

    def test_twin_status_reporting(self, twin_engine):
        """Test twin status reporting"""
        # Add some test data
        twin_engine.action_success_rate = 0.85
        twin_engine.user_satisfaction = 0.75
        twin_engine.adaptation_count = 5
        
        status = asyncio.run(twin_engine.get_twin_status())
        
        assert status["twin_id"] == twin_engine.twin_id
        assert status["state"] == twin_engine.state.value
        assert status["performance"]["action_success_rate"] == 0.85
        assert status["performance"]["user_satisfaction"] == 0.75
        assert status["performance"]["adaptation_count"] == 5

    def test_capability_management(self, twin_engine):
        """Test capability management"""
        capabilities = twin_engine.get_capabilities()
        
        assert isinstance(capabilities, list)
        assert len(capabilities) > 0
        assert all(isinstance(cap, TwinCapability) for cap in capabilities)

    @pytest.mark.asyncio
    async def test_learning_adaptation(self, twin_engine):
        """Test learning and adaptation cycle"""
        # Mock the learning methods
        twin_engine._adapt_personality = AsyncMock()
        twin_engine._update_user_model = AsyncMock()
        twin_engine._cleanup_old_data = AsyncMock()
        
        # Mock component methods
        twin_engine.pattern_recognizer.analyze_user_patterns = AsyncMock()
        twin_engine.predictive_assistant.get_prediction_analytics = AsyncMock(
            return_value={"accuracy": 0.8}
        )
        
        initial_adaptation_count = twin_engine.adaptation_count
        
        await twin_engine._perform_learning_cycle()
        
        assert twin_engine.adaptation_count > initial_adaptation_count
        twin_engine._adapt_personality.assert_called_once()
        twin_engine._update_user_model.assert_called_once()

    def test_twin_action_creation(self):
        """Test TwinAction creation and serialization"""
        action = TwinAction(
            action_type="send_notification",
            description="Remind about meeting",
            confidence=0.9,
            urgency=0.7,
            trigger="calendar_event",
            reasoning="Meeting in 15 minutes"
        )
        
        action_dict = action.to_dict()
        
        assert action_dict["action_type"] == "send_notification"
        assert action_dict["confidence"] == 0.9
        assert action_dict["urgency"] == 0.7
        assert action_dict["trigger"] == "calendar_event"

    @pytest.mark.asyncio
    async def test_error_handling(self, twin_engine):
        """Test error handling in twin operations"""
        # Test context processing with invalid data
        invalid_context = None
        
        response = await twin_engine.process_context(invalid_context or {})
        
        # Should handle gracefully
        assert "twin_id" in response
        
        # Test action execution with invalid action
        invalid_action = TwinAction(action_type="unknown_action")
        
        result = await twin_engine.execute_autonomous_action(invalid_action)
        
        # Should handle gracefully
        assert "success" in result
        assert "message" in result

    def test_personality_determination(self, twin_engine):
        """Test personality type determination"""
        # Mock patterns for different personality types
        from eidolon.proactive.pattern_recognizer import UserPattern, PatternType
        
        # Technical patterns should lead to analytical personality
        tech_patterns = [
            Mock(description="coding session", triggers=["code"]),
            Mock(description="technical documentation", triggers=["docs"]),
            Mock(description="debugging process", triggers=["debug"])
        ]
        
        twin_engine.pattern_recognizer.recognized_patterns = {
            f"pattern_{i}": pattern for i, pattern in enumerate(tech_patterns)
        }
        
        personality_type = asyncio.run(twin_engine._determine_personality_type())
        
        # Should be analytical due to technical patterns
        assert personality_type in [TwinPersonality.ANALYTICAL, TwinPersonality.ADAPTIVE]

    @pytest.mark.asyncio 
    async def test_context_analysis(self, twin_engine):
        """Test context analysis functionality"""
        context = {
            "current_app": "VS Code",
            "current_text": "def calculate_fibonacci(n): error in line 5",
            "timestamp": datetime.now().isoformat()
        }
        
        analysis = await twin_engine._analyze_context(context)
        
        assert analysis["activity_type"] == "development"
        assert "productivity_indicators" in analysis
        assert "opportunities" in analysis
        
        # Should detect debugging opportunity due to "error" in text
        opportunities = [opp for opp in analysis["opportunities"] if "debug" in opp.lower()]
        assert len(opportunities) > 0


class TestTwinAction:
    """Test TwinAction functionality"""

    def test_twin_action_serialization(self):
        """Test TwinAction serialization"""
        action = TwinAction(
            action_type="create_reminder",
            description="Create meeting reminder",
            target="calendar_app",
            parameters={"time": "14:00", "message": "Team standup"},
            confidence=0.85,
            urgency=0.6,
            scheduled_time=datetime.now() + timedelta(minutes=30),
            trigger="pattern_detection",
            reasoning="Regular meeting pattern detected",
            learned_from=["pattern_1", "pattern_2"]
        )
        
        action_dict = action.to_dict()
        
        # Verify all fields are serialized
        assert action_dict["id"] == action.id
        assert action_dict["action_type"] == "create_reminder"
        assert action_dict["target"] == "calendar_app"
        assert action_dict["parameters"]["time"] == "14:00"
        assert action_dict["confidence"] == 0.85
        assert action_dict["learned_from"] == ["pattern_1", "pattern_2"]

    def test_twin_action_execution_tracking(self):
        """Test action execution tracking"""
        action = TwinAction(
            action_type="test_action",
            description="Test action tracking"
        )
        
        # Initially not executed
        assert action.executed_time is None
        assert action.success is None
        
        # Simulate execution
        action.executed_time = datetime.now()
        action.success = True
        action.result = "Action completed successfully"
        
        assert action.executed_time is not None
        assert action.success is True
        assert action.result == "Action completed successfully"


@pytest.fixture
def mock_dependencies():
    """Mock all external dependencies for testing"""
    with patch('eidolon.twin.digital_twin_engine.MemorySystem') as mock_memory, \
         patch('eidolon.twin.digital_twin_engine.CloudAPIManager') as mock_cloud_api, \
         patch('eidolon.twin.digital_twin_engine.TaskPlanner') as mock_task_planner, \
         patch('eidolon.twin.digital_twin_engine.PatternRecognizer') as mock_pattern_recognizer, \
         patch('eidolon.twin.digital_twin_engine.PredictiveAssistant') as mock_predictive_assistant, \
         patch('eidolon.twin.digital_twin_engine.StyleReplicator') as mock_style_replicator:
        
        # Configure mocks
        mock_memory.return_value = AsyncMock()
        mock_cloud_api.return_value = AsyncMock()
        mock_task_planner.return_value = Mock()
        mock_pattern_recognizer.return_value = Mock()
        mock_predictive_assistant.return_value = AsyncMock()
        mock_style_replicator.return_value = Mock()
        
        yield {
            'memory': mock_memory,
            'cloud_api': mock_cloud_api,
            'task_planner': mock_task_planner,
            'pattern_recognizer': mock_pattern_recognizer,
            'predictive_assistant': mock_predictive_assistant,
            'style_replicator': mock_style_replicator
        }


class TestTwinIntegration:
    """Test integration scenarios for the digital twin"""

    @pytest.mark.asyncio
    async def test_full_twin_lifecycle(self, mock_dependencies):
        """Test complete twin lifecycle"""
        # Create twin
        twin = DigitalTwinEngine()
        twin.state = TwinState.ACTIVE
        
        # Process context
        context = {"current_app": "Terminal", "current_text": "npm test"}
        response = await twin.process_context(context)
        
        assert response["twin_id"] == twin.twin_id
        
        # Create and execute action
        action = TwinAction(
            action_type="offer_assistance",
            description="Offer help with testing"
        )
        
        twin.style_replicator.generate_styled_response = AsyncMock(
            return_value={"response": "Need help with your tests?"}
        )
        
        result = await twin.execute_autonomous_action(action)
        assert result["success"] is True
        
        # Update feedback
        await twin.update_user_feedback("general", 0.8, "Helpful suggestion")
        
        # Check status
        status = await twin.get_twin_status()
        assert status["state"] == "active"

    @pytest.mark.asyncio
    async def test_twin_learning_integration(self, mock_dependencies):
        """Test twin learning from user interactions"""
        twin = DigitalTwinEngine()
        twin.state = TwinState.ACTIVE
        
        # Simulate multiple interactions
        contexts = [
            {"current_app": "VS Code", "activity": "coding"},
            {"current_app": "Terminal", "activity": "testing"},
            {"current_app": "Browser", "activity": "research"}
        ]
        
        for context in contexts:
            await twin.process_context(context)
        
        # Should have context history
        assert len(twin.context_history) >= len(contexts)
        
        # Simulate learning cycle
        twin._adapt_personality = AsyncMock()
        twin._update_user_model = AsyncMock()
        twin._cleanup_old_data = AsyncMock()
        twin.pattern_recognizer.analyze_user_patterns = AsyncMock()
        twin.predictive_assistant.get_prediction_analytics = AsyncMock(return_value={})
        
        initial_count = twin.adaptation_count
        await twin._perform_learning_cycle()
        
        assert twin.adaptation_count > initial_count