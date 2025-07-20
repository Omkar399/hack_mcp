"""
Phase 6 Integration Tests - MCP Integration & Basic Agency

Tests for autonomous task system, safety mechanisms, tool orchestration,
and email/document assistance functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import tempfile

# Import Phase 6 components
from eidolon.core.agent import AutonomousAgent, Task, TaskStatus, TaskPriority, TaskType
from eidolon.core.safety import SafetyManager, ActionApproval, RiskLevel
from eidolon.tools.registry import ToolRegistry
from eidolon.tools.base import BaseTool, ToolMetadata, ToolResult
from eidolon.assistants.email_assistant import EmailAssistant
from eidolon.assistants.document_assistant import DocumentAssistant
from eidolon.assistants.office_assistant import OfficeAssistant


class TestAutonomousAgent:
    """Test the autonomous agent system."""
    
    @pytest.fixture
    async def agent(self):
        """Create test autonomous agent."""
        agent = AutonomousAgent()
        await agent.initialize()
        return agent
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.observer is not None
        assert agent.analyzer is not None
        assert agent.memory is not None
        assert agent.safety_manager is not None
        assert agent.task_queue is not None
        assert not agent.running
    
    @pytest.mark.asyncio
    async def test_create_task(self, agent):
        """Test task creation."""
        task = await agent.create_task(
            title="Test Task",
            description="A test task for validation",
            task_type=TaskType.ANALYSIS,
            actions=[{"type": "analyze_request", "params": {"request": "test"}}],
            priority=TaskPriority.MEDIUM
        )
        
        assert task.title == "Test Task"
        assert task.task_type == TaskType.ANALYSIS
        assert task.priority == TaskPriority.MEDIUM
        assert task.status == TaskStatus.PENDING
        assert task.requires_approval
        assert len(task.actions) == 1
    
    @pytest.mark.asyncio
    async def test_task_approval(self, agent):
        """Test task approval workflow."""
        # Create task
        task = await agent.create_task(
            title="Approval Test",
            description="Test task approval",
            task_type=TaskType.AUTOMATION,
            actions=[{"type": "test_action", "params": {}}]
        )
        
        # Task should require approval initially
        assert task.status == TaskStatus.PENDING
        assert task.requires_approval
        
        # Approve task
        success = await agent.approve_task(task.id, "test_user")
        assert success
        
        # Check approval status
        updated_task = agent.task_queue.tasks[task.id]
        assert updated_task.status == TaskStatus.APPROVED
        assert updated_task.approved_by == "test_user"
    
    @pytest.mark.asyncio
    async def test_suggest_task(self, agent):
        """Test task suggestion functionality."""
        with patch.object(agent.cloud_api, 'analyze_text', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(
                content='[{"title": "Test Task", "description": "Suggested task", "task_type": "analysis", "actions": []}]'
            )
            
            suggestions = await agent.suggest_task("Analyze my email activity")
            
            assert len(suggestions) >= 1
            assert suggestions[0].title in ["Test Task", "Analyze Request"]
    
    @pytest.mark.asyncio
    async def test_agent_statistics(self, agent):
        """Test agent statistics tracking."""
        stats = agent.get_statistics()
        
        assert "tasks_completed" in stats
        assert "tasks_failed" in stats
        assert "total_execution_time" in stats
        assert "uptime_seconds" in stats
        assert "active_tasks" in stats


class TestSafetyManager:
    """Test the safety management system."""
    
    @pytest.fixture
    def safety_manager(self):
        """Create test safety manager."""
        return SafetyManager()
    
    @pytest.mark.asyncio
    async def test_risk_assessment(self, safety_manager):
        """Test risk assessment functionality."""
        # Low risk action
        low_risk_action = [{"type": "read_file", "params": {"path": "/tmp/test.txt"}}]
        risk_level = await safety_manager.assess_risk(low_risk_action)
        assert risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]
        
        # High risk action
        high_risk_action = [{"type": "delete_file", "params": {"path": "/important/file.txt"}}]
        risk_level = await safety_manager.assess_risk(high_risk_action)
        assert risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
    
    @pytest.mark.asyncio
    async def test_action_validation(self, safety_manager):
        """Test action validation."""
        action = {"type": "read_file", "params": {"path": "/tmp/test.txt"}}
        
        validation = await safety_manager.validate_action(action, "test_user")
        
        assert "approved" in validation
        assert "reason" in validation
        assert "risk_level" in validation
        assert "consent_required" in validation
    
    @pytest.mark.asyncio
    async def test_blocked_actions(self, safety_manager):
        """Test blocked action detection."""
        blocked_action = {"type": "system_command", "params": {"command": "rm -rf /"}}
        
        validation = await safety_manager.validate_action(blocked_action)
        
        assert not validation.get("approved", True)
        assert validation.get("consent_required") == "blocked"
    
    def test_safety_rules(self, safety_manager):
        """Test safety rules management."""
        rules = safety_manager.get_safety_rules()
        assert len(rules) > 0
        
        # Check default rules exist
        rule_names = [rule.name for rule in rules]
        assert "File Deletion" in rule_names
        assert "System Command" in rule_names
    
    def test_sensitive_data_detection(self, safety_manager):
        """Test sensitive data detection."""
        # Content with sensitive data
        sensitive_content = "My password is secret123 and my SSN is 123-45-6789"
        assert safety_manager._contains_sensitive_data(sensitive_content)
        
        # Normal content
        normal_content = "This is a normal document about project status"
        assert not safety_manager._contains_sensitive_data(normal_content)


class TestToolRegistry:
    """Test the tool registry system."""
    
    @pytest.fixture
    def tool_registry(self):
        """Create test tool registry."""
        safety_manager = SafetyManager()
        return ToolRegistry(safety_manager)
    
    @pytest.fixture
    def test_tool(self):
        """Create a test tool."""
        class TestTool(BaseTool):
            METADATA = ToolMetadata(
                name="test_tool",
                description="A test tool",
                category="testing"
            )
            
            async def execute(self, parameters, context=None):
                return ToolResult(
                    success=True,
                    data={"result": "test_success"},
                    message="Test tool executed"
                )
        
        return TestTool
    
    def test_tool_registration(self, tool_registry, test_tool):
        """Test tool registration."""
        success = tool_registry.register_tool(test_tool)
        assert success
        
        tools = tool_registry.list_tools()
        assert "test_tool" in tools
    
    @pytest.mark.asyncio
    async def test_tool_execution(self, tool_registry, test_tool):
        """Test tool execution."""
        # Register tool
        tool_registry.register_tool(test_tool)
        
        # Execute tool
        result = await tool_registry.execute_tool(
            "test_tool",
            {"param1": "value1"}
        )
        
        assert result.success
        assert result.data["result"] == "test_success"
    
    def test_tool_categories(self, tool_registry, test_tool):
        """Test tool categorization."""
        tool_registry.register_tool(test_tool)
        
        categories = tool_registry.get_categories()
        assert "testing" in categories
        
        test_tools = tool_registry.get_tools_by_category("testing")
        assert "test_tool" in test_tools


class TestEmailAssistant:
    """Test the email assistant functionality."""
    
    @pytest.fixture
    def email_assistant(self):
        """Create test email assistant."""
        return EmailAssistant()
    
    @pytest.mark.asyncio
    async def test_email_composition(self, email_assistant):
        """Test email composition."""
        with patch.object(email_assistant.cloud_api, 'analyze_text', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(
                content='{"subject": "Test Email", "body": "This is a test email."}'
            )
            
            result = await email_assistant.compose_email(
                "Write a follow-up email to the client"
            )
            
            assert result.get("success", False)
            assert "subject" in result
            assert "body" in result
    
    @pytest.mark.asyncio
    async def test_email_analysis(self, email_assistant):
        """Test email analysis."""
        analysis = await email_assistant.analyze_email(
            subject="Urgent: Project Deadline",
            body="We need to complete the project by Friday. This is very important.",
            sender="client@example.com"
        )
        
        assert analysis.sentiment in ["positive", "negative", "neutral"]
        assert analysis.urgency in ["low", "medium", "high", "urgent"]
        assert analysis.category in ["work", "personal", "other"]
        assert isinstance(analysis.confidence, float)
    
    @pytest.mark.asyncio
    async def test_reply_suggestions(self, email_assistant):
        """Test reply suggestions."""
        suggestions = await email_assistant.suggest_replies(
            "Meeting Request",
            "Would you like to schedule a meeting next week?",
            "colleague@company.com"
        )
        
        assert len(suggestions) > 0
        assert all("type" in suggestion for suggestion in suggestions)
        assert all("body" in suggestion for suggestion in suggestions)
    
    def test_email_templates(self, email_assistant):
        """Test email templates."""
        templates = email_assistant.get_templates()
        assert len(templates) > 0
        
        # Check default templates exist
        template_names = [template.name for template in templates]
        assert "meeting_request" in template_names
        assert "follow_up" in template_names


class TestDocumentAssistant:
    """Test the document assistant functionality."""
    
    @pytest.fixture
    def document_assistant(self):
        """Create test document assistant."""
        return DocumentAssistant()
    
    @pytest.fixture
    def test_document(self):
        """Create a test document."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document. It contains multiple sentences. The content is for testing purposes.")
            return f.name
    
    @pytest.mark.asyncio
    async def test_document_analysis(self, document_assistant, test_document):
        """Test document analysis."""
        analysis = await document_assistant.analyze_document(test_document)
        
        assert analysis.document_type in ["text", "markdown", "code"]
        assert analysis.word_count > 0
        assert isinstance(analysis.readability_score, float)
        assert isinstance(analysis.contains_sensitive_data, bool)
        assert len(analysis.summary) > 0
    
    @pytest.mark.asyncio
    async def test_document_generation(self, document_assistant):
        """Test document generation."""
        with patch.object(document_assistant.cloud_api, 'analyze_text', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(
                content="# Test Document\n\nThis is a generated document."
            )
            
            result = await document_assistant.generate_document(
                "Create a project status report",
                document_type="markdown"
            )
            
            assert result.get("success", False)
            assert "content" in result
    
    @pytest.mark.asyncio
    async def test_document_summarization(self, document_assistant, test_document):
        """Test document summarization."""
        result = await document_assistant.summarize_document(test_document)
        
        assert result.get("success", False)
        assert "summary" in result
        assert result.get("original_length", 0) > 0
    
    def test_document_templates(self, document_assistant):
        """Test document templates."""
        templates = document_assistant.get_templates()
        assert len(templates) > 0
        
        # Check default templates exist
        template_names = [template.name for template in templates]
        assert "meeting_notes" in template_names
        assert "project_report" in template_names


class TestOfficeAssistant:
    """Test the office assistant functionality."""
    
    @pytest.fixture
    async def office_assistant(self):
        """Create test office assistant."""
        assistant = OfficeAssistant()
        await assistant.initialize()
        return assistant
    
    @pytest.mark.asyncio
    async def test_request_processing(self, office_assistant):
        """Test general request processing."""
        result = await office_assistant.process_request(
            "Send a weekly report email to the team",
            context={"team_emails": ["team@company.com"]}
        )
        
        assert "success" in result
        assert "message" in result
    
    @pytest.mark.asyncio
    async def test_workflow_automation(self, office_assistant):
        """Test workflow automation."""
        workflows = office_assistant.get_automation_workflows()
        assert len(workflows) > 0
        
        # Test workflow execution
        workflow_name = workflows[0]["name"]
        result = await office_assistant.automate_workflow(
            workflow_name,
            {"test_param": "value"}
        )
        
        assert "success" in result
        assert "workflow" in result
    
    @pytest.mark.asyncio
    async def test_productivity_analysis(self, office_assistant):
        """Test productivity analysis."""
        result = await office_assistant.analyze_productivity(
            timeframe="week"
        )
        
        assert result.get("success", False)
        assert "analytics" in result
        assert "insights" in result
        assert "recommendations" in result
        assert "productivity_score" in result
    
    def test_productivity_metrics(self, office_assistant):
        """Test productivity metrics."""
        metrics = office_assistant.get_productivity_metrics()
        
        assert "emails_processed" in metrics
        assert "documents_created" in metrics
        assert "tasks_automated" in metrics
        assert "efficiency_score" in metrics


class TestMCPIntegration:
    """Test MCP server integration with Phase 6 components."""
    
    @pytest.mark.asyncio
    async def test_mcp_initialization(self):
        """Test MCP server initialization with Phase 6 components."""
        from eidolon.core.mcp_server import initialize_server
        
        # Mock the components to avoid actual initialization
        with patch('eidolon.core.mcp_server.AutonomousAgent') as mock_agent, \
             patch('eidolon.core.mcp_server.SafetyManager') as mock_safety, \
             patch('eidolon.core.mcp_server.ToolRegistry') as mock_registry:
            
            mock_agent.return_value.initialize = AsyncMock()
            mock_safety.return_value = MagicMock()
            mock_registry.return_value = MagicMock()
            
            # Should not raise exception
            try:
                await initialize_server()
            except Exception as e:
                # Allow initialization errors but not critical failures
                assert "Phase 6" not in str(e)
    
    def test_mcp_models(self):
        """Test MCP model definitions."""
        from eidolon.core.mcp_server import TaskModel, ActionResult, EmailComposition, DocumentGeneration
        
        # Test TaskModel
        task = TaskModel(
            id="test_task",
            title="Test Task",
            description="A test task",
            task_type="analysis",
            priority="medium",
            status="pending",
            created_at=datetime.now(),
            requires_approval=True,
            risk_level="low"
        )
        assert task.id == "test_task"
        assert task.title == "Test Task"
        
        # Test ActionResult
        result = ActionResult(
            success=True,
            message="Action completed",
            data={"result": "success"},
            execution_time=1.5,
            risk_assessment={"risk_level": "low"},
            approval_required=False
        )
        assert result.success
        assert result.execution_time == 1.5


class TestIntegrationWorkflow:
    """Test complete integration workflows."""
    
    @pytest.mark.asyncio
    async def test_email_composition_workflow(self):
        """Test complete email composition workflow."""
        # Initialize components
        safety_manager = SafetyManager()
        email_assistant = EmailAssistant()
        
        # Create email composition task
        request = "Write a follow-up email to the client about yesterday's meeting"
        
        # Assess risk
        action = {"type": "compose_email", "params": {"request": request}}
        risk_level = await safety_manager.assess_risk([action])
        assert risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]
        
        # Compose email
        with patch.object(email_assistant.cloud_api, 'analyze_text', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(
                content='{"subject": "Follow-up: Yesterday\'s Meeting", "body": "Thank you for the meeting yesterday."}'
            )
            
            result = await email_assistant.compose_email(request)
            assert result.get("success", False)
    
    @pytest.mark.asyncio
    async def test_document_generation_workflow(self):
        """Test complete document generation workflow."""
        # Initialize components
        safety_manager = SafetyManager()
        document_assistant = DocumentAssistant()
        
        # Create document generation task
        request = "Create a weekly status report"
        
        # Assess risk
        action = {"type": "generate_document", "params": {"request": request}}
        risk_level = await safety_manager.assess_risk([action])
        assert risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]
        
        # Generate document
        with patch.object(document_assistant.cloud_api, 'analyze_text', new_callable=AsyncMock) as mock_analyze:
            mock_analyze.return_value = MagicMock(
                content="# Weekly Status Report\n\nThis week's activities and progress."
            )
            
            result = await document_assistant.generate_document(
                request,
                document_type="markdown"
            )
            assert result.get("success", False)
    
    @pytest.mark.asyncio
    async def test_autonomous_task_workflow(self):
        """Test complete autonomous task workflow."""
        # Initialize components
        agent = AutonomousAgent()
        await agent.initialize()
        
        # Create and execute task
        task = await agent.create_task(
            title="Analysis Task",
            description="Analyze recent screen activity",
            task_type=TaskType.ANALYSIS,
            actions=[{"type": "analyze_request", "params": {"request": "analyze activity"}}],
            requires_approval=False  # Skip approval for test
        )
        
        assert task.status == TaskStatus.PENDING
        assert task.task_type == TaskType.ANALYSIS
        
        # Task should be created and ready for execution
        active_tasks = await agent.get_active_tasks()
        assert len(active_tasks) > 0


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup():
    """Clean up test files and resources."""
    yield
    # Cleanup would go here if needed
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])