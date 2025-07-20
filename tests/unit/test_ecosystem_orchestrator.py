"""
Unit tests for Ecosystem Orchestrator - Phase 7
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from eidolon.orchestration.ecosystem_orchestrator import (
    EcosystemOrchestrator,
    ApplicationNode,
    OrchestrationFlow,
    OrchestrationEvent,
    OrchestrationCapability,
    OrchestrationState,
    IntegrationType
)


class TestEcosystemOrchestrator:
    """Test the Ecosystem Orchestrator functionality"""

    @pytest.fixture
    async def orchestrator(self):
        """Create an orchestrator for testing"""
        with patch('eidolon.orchestration.ecosystem_orchestrator.MemorySystem') as mock_memory, \
             patch('eidolon.orchestration.ecosystem_orchestrator.CloudAPIManager') as mock_cloud_api, \
             patch('eidolon.orchestration.ecosystem_orchestrator.TaskPlanner') as mock_task_planner:
            
            # Mock the memory system
            mock_memory_instance = AsyncMock()
            mock_memory_instance.search_content.return_value = []
            mock_memory_instance.store_content.return_value = True
            mock_memory.return_value = mock_memory_instance
            
            # Mock other components
            mock_cloud_api.return_value = AsyncMock()
            mock_task_planner.return_value = Mock()
            
            # Create orchestrator without digital twin for simplicity
            orchestrator = EcosystemOrchestrator(digital_twin=None)
            
            # Initialize manually for testing
            orchestrator.state = OrchestrationState.MONITORING
            orchestrator.capabilities = [
                OrchestrationCapability.APP_COORDINATION,
                OrchestrationCapability.WORKFLOW_AUTOMATION
            ]
            
            yield orchestrator

    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization"""
        assert orchestrator.orchestrator_id is not None
        assert isinstance(orchestrator.orchestrator_id, str)
        assert orchestrator.state == OrchestrationState.MONITORING
        assert len(orchestrator.capabilities) > 0
        assert OrchestrationCapability.APP_COORDINATION in orchestrator.capabilities

    def test_application_discovery(self, orchestrator):
        """Test application discovery"""
        assert len(orchestrator.applications) > 0
        
        # Check for common applications
        app_names = [app.name for app in orchestrator.applications.values()]
        assert "Chrome" in app_names
        assert "VS Code" in app_names
        assert "Terminal" in app_names

    @pytest.mark.asyncio
    async def test_flow_creation(self, orchestrator):
        """Test orchestration flow creation"""
        flow_definition = {
            "name": "Test Workflow",
            "description": "A test workflow",
            "trigger": "manual",
            "steps": [
                {
                    "type": "app_action",
                    "app": "Terminal",
                    "action": "execute_command",
                    "parameters": {"command": "echo hello"}
                },
                {
                    "type": "notification",
                    "message": "Command executed",
                    "priority": "low"
                }
            ]
        }
        
        flow = await orchestrator.create_orchestration_flow(flow_definition)
        
        assert flow.name == "Test Workflow"
        assert len(flow.steps) == 2
        assert flow.id in orchestrator.active_flows
        assert flow.status == "pending"

    @pytest.mark.asyncio
    async def test_flow_execution(self, orchestrator):
        """Test orchestration flow execution"""
        # Create a simple flow
        flow = OrchestrationFlow(
            name="Simple Test Flow",
            description="Test flow execution",
            steps=[
                {
                    "type": "delay",
                    "duration": 0.1
                },
                {
                    "type": "notification",
                    "message": "Flow completed",
                    "priority": "medium"
                }
            ]
        )
        
        orchestrator.active_flows[flow.id] = flow
        
        # Mock the step execution methods
        orchestrator._execute_delay_step = AsyncMock(return_value={"success": True})
        orchestrator._execute_notification_step = AsyncMock(return_value={"success": True})
        orchestrator._send_notification = AsyncMock()
        
        result = await orchestrator.execute_orchestration_flow(flow.id)
        
        assert result["success"] is True
        assert result["completed_steps"] == 2
        assert flow.status == "completed"
        assert flow.progress == 1.0

    @pytest.mark.asyncio
    async def test_event_handling(self, orchestrator):
        """Test event handling"""
        event = OrchestrationEvent(
            event_type="app_focus_change",
            source="system",
            data={
                "new_app": "VS Code",
                "previous_app": "Chrome"
            }
        )
        
        # Mock the event handler
        orchestrator._handle_app_focus_change = AsyncMock()
        
        await orchestrator._handle_event(event)
        
        orchestrator._handle_app_focus_change.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_automation_rule_creation(self, orchestrator):
        """Test automation rule creation"""
        rule = {
            "name": "Morning Productivity Rule",
            "trigger_type": "time_based",
            "schedule": {"hour": 9},
            "actions": [
                {
                    "type": "send_notification",
                    "message": "Good morning! Ready to start the day?",
                    "priority": "medium"
                }
            ]
        }
        
        rule_id = await orchestrator.add_automation_rule(rule)
        
        assert rule_id is not None
        assert len(orchestrator.automation_rules) > 0
        assert orchestrator.automation_rules[-1]["name"] == "Morning Productivity Rule"

    def test_application_node_creation(self):
        """Test ApplicationNode creation and serialization"""
        app = ApplicationNode(
            name="Test App",
            type="productivity",
            capabilities=["task_management", "note_taking"],
            integration_methods=[IntegrationType.NATIVE_API, IntegrationType.WEB_AUTOMATION],
            status="active"
        )
        
        app_dict = app.to_dict()
        
        assert app_dict["name"] == "Test App"
        assert app_dict["type"] == "productivity"
        assert "task_management" in app_dict["capabilities"]
        assert "native_api" in app_dict["integration_methods"]

    def test_orchestration_flow_serialization(self):
        """Test OrchestrationFlow serialization"""
        flow = OrchestrationFlow(
            name="Test Flow",
            description="Test serialization",
            trigger="manual",
            steps=[{"type": "test_step"}],
            status="running",
            progress=0.5
        )
        
        flow_dict = flow.to_dict()
        
        assert flow_dict["name"] == "Test Flow"
        assert flow_dict["status"] == "running"
        assert flow_dict["progress"] == 0.5
        assert len(flow_dict["steps"]) == 1

    @pytest.mark.asyncio
    async def test_step_execution_types(self, orchestrator):
        """Test different step execution types"""
        # Test delay step
        delay_step = {"type": "delay", "duration": 0.01}
        result = await orchestrator._execute_delay_step(delay_step, {})
        assert result["success"] is True
        
        # Test notification step
        notification_step = {
            "type": "notification",
            "message": "Test notification",
            "priority": "low"
        }
        orchestrator._send_notification = AsyncMock()
        result = await orchestrator._execute_notification_step(notification_step, {})
        assert result["success"] is True
        
        # Test data transform step
        transform_step = {
            "type": "data_transform",
            "transformation": {
                "source": "input_data",
                "target": "output_data",
                "operation": "uppercase"
            }
        }
        context = {"input_data": "hello world"}
        result = await orchestrator._execute_data_transform_step(transform_step, context)
        assert result["success"] is True
        assert result["output"]["output_data"] == "HELLO WORLD"

    @pytest.mark.asyncio
    async def test_condition_evaluation(self, orchestrator):
        """Test condition evaluation"""
        # Test simple condition
        condition_step = {
            "type": "condition",
            "condition": "context.get('test_value', 0) > 5",
            "true_action": {"type": "notification", "message": "Condition true"},
            "false_action": {"type": "notification", "message": "Condition false"}
        }
        
        orchestrator._send_notification = AsyncMock()
        
        # Test with true condition
        context = {"test_value": 10}
        result = await orchestrator._execute_condition_step(condition_step, context)
        assert result["success"] is True
        
        # Test with false condition
        context = {"test_value": 3}
        result = await orchestrator._execute_condition_step(condition_step, context)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_error_handling_in_flows(self, orchestrator):
        """Test error handling in flow execution"""
        # Create flow with failing step
        flow = OrchestrationFlow(
            name="Error Test Flow",
            steps=[
                {"type": "unknown_step_type"},  # This should fail
                {"type": "delay", "duration": 0.1}  # This should not execute
            ]
        )
        
        orchestrator.active_flows[flow.id] = flow
        
        result = await orchestrator.execute_orchestration_flow(flow.id)
        
        assert result["success"] is False
        assert "error" in result
        assert result["completed_steps"] == 0  # Should stop at first error
        assert flow.status == "failed"

    @pytest.mark.asyncio
    async def test_flow_with_continue_on_failure(self, orchestrator):
        """Test flow execution with continue_on_failure"""
        flow = OrchestrationFlow(
            name="Continue on Failure Test",
            steps=[
                {
                    "type": "unknown_step_type",
                    "continue_on_failure": True  # Should continue despite failure
                },
                {
                    "type": "delay",
                    "duration": 0.01
                }
            ]
        )
        
        orchestrator.active_flows[flow.id] = flow
        
        result = await orchestrator.execute_orchestration_flow(flow.id)
        
        # Should complete despite first step failure
        assert result["success"] is True
        assert result["completed_steps"] == 2

    def test_event_creation_and_serialization(self):
        """Test OrchestrationEvent creation and serialization"""
        event = OrchestrationEvent(
            event_type="test_event",
            source="test_source",
            target="test_target",
            data={"key": "value", "number": 42}
        )
        
        event_dict = event.to_dict()
        
        assert event_dict["event_type"] == "test_event"
        assert event_dict["source"] == "test_source"
        assert event_dict["target"] == "test_target"
        assert event_dict["data"]["key"] == "value"
        assert event_dict["processed"] is False

    @pytest.mark.asyncio
    async def test_orchestrator_status(self, orchestrator):
        """Test orchestrator status reporting"""
        # Add some test data
        orchestrator.performance_metrics = {
            "average_success_rate": 0.85,
            "active_flows": 3
        }
        
        status = await orchestrator.get_orchestration_status()
        
        assert status["orchestrator_id"] == orchestrator.orchestrator_id
        assert status["state"] == orchestrator.state.value
        assert "ecosystem" in status
        assert "orchestration" in status
        assert "recent_activity" in status
        assert status["orchestration"]["performance_metrics"]["average_success_rate"] == 0.85

    @pytest.mark.asyncio
    async def test_app_action_execution(self, orchestrator):
        """Test application action execution"""
        # Mock application action execution methods
        orchestrator._execute_native_api_action = AsyncMock(
            return_value={"status": "completed", "result": "API call successful"}
        )
        orchestrator._execute_web_automation_action = AsyncMock(
            return_value={"status": "completed", "result": "Web automation successful"}
        )
        orchestrator._execute_system_command_action = AsyncMock(
            return_value={"status": "completed", "result": "System command successful"}
        )
        
        # Test native API action
        step = {
            "type": "app_action",
            "app": "Chrome",
            "action": "open_tab",
            "parameters": {"url": "https://example.com"}
        }
        
        result = await orchestrator._execute_app_action_step(step, {})
        
        assert result["success"] is True
        assert result["app"] == "Chrome"
        assert result["action"] == "open_tab"

    @pytest.mark.asyncio
    async def test_api_call_execution(self, orchestrator):
        """Test API call step execution"""
        step = {
            "type": "api_call",
            "endpoint": "https://api.example.com/data",
            "method": "GET",
            "headers": {"Authorization": "Bearer token"}
        }
        
        result = await orchestrator._execute_api_call_step(step, {})
        
        # Mock implementation returns success
        assert result["success"] is True
        assert result["output"]["endpoint"] == "https://api.example.com/data"

    @pytest.mark.asyncio
    async def test_automation_rule_execution(self, orchestrator):
        """Test automation rule execution"""
        rule = {
            "name": "Test Rule",
            "actions": [
                {
                    "type": "send_notification",
                    "message": "Rule executed",
                    "priority": "medium"
                }
            ]
        }
        
        orchestrator._send_notification = AsyncMock()
        
        await orchestrator._execute_automation_rule(rule)
        
        orchestrator._send_notification.assert_called_once_with("Rule executed", "medium")
        assert "last_run" in rule

    def test_application_finding(self, orchestrator):
        """Test finding applications by name"""
        app = orchestrator._find_application_by_name("Chrome")
        assert app is not None
        assert app.name == "Chrome"
        
        # Test case insensitive
        app = orchestrator._find_application_by_name("chrome")
        assert app is not None
        assert app.name == "Chrome"
        
        # Test non-existent app
        app = orchestrator._find_application_by_name("NonExistentApp")
        assert app is None

    @pytest.mark.asyncio
    async def test_flow_validation(self, orchestrator):
        """Test flow step validation"""
        # Valid flow
        valid_flow = OrchestrationFlow(
            name="Valid Flow",
            steps=[
                {
                    "type": "app_action",
                    "app": "Terminal",
                    "action": "execute_command"
                },
                {
                    "type": "api_call",
                    "endpoint": "https://api.example.com"
                }
            ]
        )
        
        # Should not raise exception
        await orchestrator._validate_flow_steps(valid_flow)
        
        # Invalid flow - missing app
        invalid_flow = OrchestrationFlow(
            name="Invalid Flow",
            steps=[
                {
                    "type": "app_action",
                    "app": "NonExistentApp",
                    "action": "some_action"
                }
            ]
        )
        
        # Should raise exception
        with pytest.raises(ValueError, match="Application 'NonExistentApp' not found"):
            await orchestrator._validate_flow_steps(invalid_flow)

    @pytest.mark.asyncio
    async def test_time_trigger_checking(self, orchestrator):
        """Test time-based trigger checking"""
        current_time = datetime.now()
        
        # Test hour-based trigger
        schedule = {"hour": current_time.hour}
        result = await orchestrator._check_time_trigger(schedule, current_time)
        assert result is True
        
        # Test different hour
        schedule = {"hour": (current_time.hour + 1) % 24}
        result = await orchestrator._check_time_trigger(schedule, current_time)
        assert result is False
        
        # Test interval-based trigger
        schedule = {"interval_minutes": 60}  # No last_run
        result = await orchestrator._check_time_trigger(schedule, current_time)
        assert result is True
        
        # Test with recent last_run
        schedule = {
            "interval_minutes": 60,
            "last_run": current_time.isoformat()
        }
        result = await orchestrator._check_time_trigger(schedule, current_time)
        assert result is False


class TestOrchestrationFlow:
    """Test OrchestrationFlow functionality"""

    def test_flow_creation(self):
        """Test flow creation with various parameters"""
        flow = OrchestrationFlow(
            name="Test Flow",
            description="Test flow creation",
            trigger="time_based",
            steps=[
                {"type": "step1", "action": "action1"},
                {"type": "step2", "action": "action2"}
            ],
            dependencies={"step2": ["step1"]},
            error_handling={"retry_count": 3}
        )
        
        assert flow.name == "Test Flow"
        assert len(flow.steps) == 2
        assert flow.dependencies["step2"] == ["step1"]
        assert flow.error_handling["retry_count"] == 3
        assert flow.status == "pending"
        assert flow.progress == 0.0

    def test_flow_progress_tracking(self):
        """Test flow progress tracking"""
        flow = OrchestrationFlow(
            name="Progress Test",
            steps=[{"type": "step1"}, {"type": "step2"}, {"type": "step3"}]
        )
        
        # Simulate step execution
        flow.current_step = 0
        flow.progress = 0 / 3
        assert flow.progress == 0.0
        
        flow.current_step = 1
        flow.progress = 1 / 3
        assert abs(flow.progress - 0.333) < 0.01
        
        flow.current_step = 2
        flow.progress = 2 / 3
        assert abs(flow.progress - 0.667) < 0.01
        
        flow.progress = 1.0
        assert flow.progress == 1.0


class TestApplicationNode:
    """Test ApplicationNode functionality"""

    def test_application_creation(self):
        """Test application node creation"""
        app = ApplicationNode(
            name="Test Application",
            type="productivity",
            capabilities=["task_management", "file_editing"],
            integration_methods=[IntegrationType.NATIVE_API],
            status="active",
            usage_frequency=0.8,
            reliability_score=0.95
        )
        
        assert app.name == "Test Application"
        assert app.type == "productivity"
        assert "task_management" in app.capabilities
        assert IntegrationType.NATIVE_API in app.integration_methods
        assert app.usage_frequency == 0.8
        assert app.reliability_score == 0.95

    def test_application_health_tracking(self):
        """Test application health tracking"""
        app = ApplicationNode(name="Health Test App")
        
        # Initial health should be good
        assert app.health_score == 1.0
        assert app.status == "unknown"
        
        # Update health
        app.health_score = 0.7
        app.status = "degraded"
        app.last_seen = datetime.now()
        
        assert app.health_score == 0.7
        assert app.status == "degraded"
        assert app.last_seen is not None


@pytest.fixture
def mock_orchestrator_dependencies():
    """Mock all external dependencies for orchestrator testing"""
    with patch('eidolon.orchestration.ecosystem_orchestrator.MemorySystem') as mock_memory, \
         patch('eidolon.orchestration.ecosystem_orchestrator.CloudAPIManager') as mock_cloud_api, \
         patch('eidolon.orchestration.ecosystem_orchestrator.TaskPlanner') as mock_task_planner:
        
        # Configure mocks
        mock_memory.return_value = AsyncMock()
        mock_cloud_api.return_value = AsyncMock()
        mock_task_planner.return_value = Mock()
        
        yield {
            'memory': mock_memory,
            'cloud_api': mock_cloud_api,
            'task_planner': mock_task_planner
        }


class TestOrchestrationIntegration:
    """Test integration scenarios for orchestration"""

    @pytest.mark.asyncio
    async def test_complete_orchestration_scenario(self, mock_orchestrator_dependencies):
        """Test a complete orchestration scenario"""
        # Create orchestrator
        orchestrator = EcosystemOrchestrator()
        orchestrator.state = OrchestrationState.MONITORING
        
        # Create and execute a workflow
        flow_definition = {
            "name": "Complete Test Workflow",
            "description": "End-to-end test",
            "steps": [
                {"type": "delay", "duration": 0.01},
                {"type": "notification", "message": "Workflow complete"}
            ]
        }
        
        flow = await orchestrator.create_orchestration_flow(flow_definition)
        assert flow.id in orchestrator.active_flows
        
        # Mock necessary methods
        orchestrator._send_notification = AsyncMock()
        
        # Execute the flow
        result = await orchestrator.execute_orchestration_flow(flow.id)
        assert result["success"] is True
        
        # Check status
        status = await orchestrator.get_orchestration_status()
        assert status["orchestrator_id"] == orchestrator.orchestrator_id

    @pytest.mark.asyncio
    async def test_event_driven_orchestration(self, mock_orchestrator_dependencies):
        """Test event-driven orchestration"""
        orchestrator = EcosystemOrchestrator()
        orchestrator.state = OrchestrationState.MONITORING
        
        # Add automation rule
        rule = {
            "name": "Event Response Rule",
            "trigger_type": "app_change",
            "trigger_app": "VS Code",
            "actions": [
                {
                    "type": "send_notification",
                    "message": "Development mode activated",
                    "priority": "medium"
                }
            ]
        }
        
        await orchestrator.add_automation_rule(rule)
        
        # Trigger event
        event = OrchestrationEvent(
            event_type="app_focus_change",
            source="system",
            data={"new_app": "VS Code", "previous_app": "Chrome"}
        )
        
        # Mock event handling
        orchestrator._execute_automation_rule = AsyncMock()
        
        await orchestrator.trigger_event(event)
        assert len(orchestrator.event_queue) > 0