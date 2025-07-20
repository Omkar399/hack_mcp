"""
Office Assistant for Eidolon AI Personal Assistant

Provides comprehensive office automation and productivity assistance.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from ..utils.logging import get_component_logger
from ..utils.config import get_config
from ..models.cloud_api import CloudAPIManager
from .email_assistant import EmailAssistant
from .document_assistant import DocumentAssistant
from ..core.agent import AutonomousAgent, Task, TaskType, TaskPriority

logger = get_component_logger("assistants.office")


@dataclass
class OfficeTask:
    """Office automation task."""
    id: str
    title: str
    description: str
    category: str  # email, document, calendar, presentation, etc.
    priority: str  # low, medium, high, urgent
    estimated_duration: int  # minutes
    dependencies: List[str]
    automation_level: str  # manual, semi_auto, full_auto


@dataclass
class ProductivityInsight:
    """Productivity analysis insight."""
    category: str
    insight: str
    recommendation: str
    impact_level: str  # low, medium, high
    confidence: float


class OfficeAssistant:
    """
    Comprehensive office assistant for productivity and automation.
    
    Integrates email, document, calendar, and other office functions
    to provide seamless workflow automation.
    """
    
    def __init__(self, config_override: Optional[Dict] = None):
        """Initialize office assistant."""
        self.config = get_config()
        if config_override:
            self.config.update(config_override)
        
        self.cloud_api = CloudAPIManager()
        self.email_assistant = EmailAssistant()
        self.document_assistant = DocumentAssistant()
        self.autonomous_agent = AutonomousAgent()
        
        # Office automation patterns
        self.automation_workflows = {}
        self._load_default_workflows()
        
        # Productivity tracking
        self.productivity_metrics = {
            "emails_processed": 0,
            "documents_created": 0,
            "tasks_automated": 0,
            "time_saved_minutes": 0
        }
        
        logger.info("Office assistant initialized")
    
    async def initialize(self) -> None:
        """Initialize office assistant components."""
        await self.autonomous_agent.initialize()
        logger.info("Office assistant fully initialized")
    
    async def process_request(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
        automation_level: str = "semi_auto"
    ) -> Dict[str, Any]:
        """
        Process a general office request and route to appropriate assistant.
        
        Args:
            request: User's office request
            context: Additional context
            automation_level: Level of automation (manual, semi_auto, full_auto)
            
        Returns:
            Processing result
        """
        try:
            logger.info(f"Processing office request: {request[:100]}...")
            
            # Analyze request to determine intent and routing
            analysis = await self._analyze_request(request, context)
            
            # Route to appropriate handler
            if analysis["category"] == "email":
                return await self._handle_email_request(request, context, analysis)
            elif analysis["category"] == "document":
                return await self._handle_document_request(request, context, analysis)
            elif analysis["category"] == "workflow":
                return await self._handle_workflow_request(request, context, analysis)
            elif analysis["category"] == "productivity":
                return await self._handle_productivity_request(request, context, analysis)
            elif analysis["category"] == "automation":
                return await self._handle_automation_request(request, context, analysis)
            else:
                return await self._handle_general_request(request, context, analysis)
                
        except Exception as e:
            logger.error(f"Office request processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to process office request"
            }
    
    async def automate_workflow(
        self,
        workflow_name: str,
        parameters: Dict[str, Any],
        schedule: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute or schedule an automated workflow.
        
        Args:
            workflow_name: Name of workflow to execute
            parameters: Workflow parameters
            schedule: Optional schedule for recurring execution
            
        Returns:
            Automation result
        """
        try:
            if workflow_name not in self.automation_workflows:
                raise ValueError(f"Unknown workflow: {workflow_name}")
            
            workflow = self.automation_workflows[workflow_name]
            
            # Create autonomous tasks for workflow steps
            tasks = []
            for step in workflow["steps"]:
                task = await self.autonomous_agent.create_task(
                    title=step["title"],
                    description=step["description"],
                    task_type=TaskType(step["type"]),
                    actions=step["actions"],
                    priority=TaskPriority(step.get("priority", "medium")),
                    requires_approval=step.get("requires_approval", True)
                )
                tasks.append(task)
            
            return {
                "success": True,
                "workflow": workflow_name,
                "tasks_created": len(tasks),
                "task_ids": [task.id for task in tasks],
                "schedule": schedule,
                "message": f"Workflow '{workflow_name}' initiated with {len(tasks)} tasks"
            }
            
        except Exception as e:
            logger.error(f"Workflow automation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to automate workflow '{workflow_name}'"
            }
    
    async def analyze_productivity(
        self,
        timeframe: str = "week",
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze productivity patterns and provide insights.
        
        Args:
            timeframe: Analysis timeframe (day, week, month)
            focus_areas: Specific areas to analyze
            
        Returns:
            Productivity analysis
        """
        try:
            # Get data from various sources
            email_data = await self._get_email_analytics(timeframe)
            document_data = await self._get_document_analytics(timeframe)
            task_data = await self._get_task_analytics(timeframe)
            
            # Analyze patterns
            insights = await self._generate_productivity_insights(
                email_data, document_data, task_data, focus_areas
            )
            
            # Generate recommendations
            recommendations = await self._generate_productivity_recommendations(insights)
            
            return {
                "success": True,
                "timeframe": timeframe,
                "analytics": {
                    "email": email_data,
                    "documents": document_data,
                    "tasks": task_data
                },
                "insights": insights,
                "recommendations": recommendations,
                "productivity_score": self._calculate_productivity_score(insights)
            }
            
        except Exception as e:
            logger.error(f"Productivity analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to analyze productivity"
            }
    
    async def manage_schedule(
        self,
        action: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Manage calendar and scheduling.
        
        Args:
            action: Scheduling action (create, update, delete, find_time)
            parameters: Action parameters
            
        Returns:
            Scheduling result
        """
        try:
            if action == "create_event":
                return await self._create_calendar_event(parameters)
            elif action == "find_meeting_time":
                return await self._find_meeting_time(parameters)
            elif action == "schedule_email":
                return await self._schedule_email(parameters)
            elif action == "schedule_task":
                return await self._schedule_task(parameters)
            else:
                raise ValueError(f"Unknown scheduling action: {action}")
                
        except Exception as e:
            logger.error(f"Schedule management failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to {action}"
            }
    
    async def generate_report(
        self,
        report_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate various types of office reports.
        
        Args:
            report_type: Type of report (productivity, email_summary, task_status)
            parameters: Report parameters
            
        Returns:
            Report generation result
        """
        try:
            if report_type == "productivity":
                return await self._generate_productivity_report(parameters)
            elif report_type == "email_summary":
                return await self._generate_email_summary_report(parameters)
            elif report_type == "task_status":
                return await self._generate_task_status_report(parameters)
            elif report_type == "workflow_analytics":
                return await self._generate_workflow_analytics_report(parameters)
            else:
                raise ValueError(f"Unknown report type: {report_type}")
                
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to generate {report_type} report"
            }
    
    def get_automation_workflows(self) -> List[Dict[str, Any]]:
        """Get available automation workflows."""
        return [
            {
                "name": name,
                "description": workflow["description"],
                "category": workflow["category"],
                "automation_level": workflow.get("automation_level", "semi_auto"),
                "estimated_time": workflow.get("estimated_time", "unknown")
            }
            for name, workflow in self.automation_workflows.items()
        ]
    
    def get_productivity_metrics(self) -> Dict[str, Any]:
        """Get current productivity metrics."""
        return {
            **self.productivity_metrics,
            "efficiency_score": self._calculate_efficiency_score(),
            "automation_usage": self._calculate_automation_usage()
        }
    
    async def _analyze_request(self, request: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze request to determine intent and routing."""
        try:
            if self.cloud_api:
                analysis_prompt = f"""
                Analyze this office request and categorize it:
                
                Request: {request}
                Context: {context or {}}
                
                Return JSON with:
                - category: email/document/workflow/productivity/automation/general
                - intent: specific intent within category
                - urgency: low/medium/high/urgent
                - automation_potential: low/medium/high
                - suggested_actions: list of suggested actions
                """
                
                response = await self.cloud_api.analyze_text(
                    analysis_prompt,
                    analysis_type="office_request_analysis"
                )
                
                if response and response.content:
                    try:
                        return json.loads(response.content)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse AI request analysis")
            
            # Fallback to basic analysis
            return self._basic_request_analysis(request)
            
        except Exception as e:
            logger.warning(f"Request analysis failed: {e}")
            return self._basic_request_analysis(request)
    
    def _basic_request_analysis(self, request: str) -> Dict[str, Any]:
        """Basic request analysis without AI."""
        request_lower = request.lower()
        
        # Simple keyword-based categorization
        if any(word in request_lower for word in ['email', 'send', 'reply', 'message']):
            category = "email"
            intent = "email_operation"
        elif any(word in request_lower for word in ['document', 'write', 'create', 'edit', 'report']):
            category = "document"
            intent = "document_operation"
        elif any(word in request_lower for word in ['schedule', 'meeting', 'calendar', 'appointment']):
            category = "workflow"
            intent = "scheduling"
        elif any(word in request_lower for word in ['automate', 'workflow', 'process']):
            category = "automation"
            intent = "automation_request"
        elif any(word in request_lower for word in ['productivity', 'analysis', 'report', 'metrics']):
            category = "productivity"
            intent = "productivity_analysis"
        else:
            category = "general"
            intent = "general_assistance"
        
        # Determine urgency
        urgency = "high" if any(word in request_lower for word in ['urgent', 'asap', 'immediately']) else "medium"
        
        return {
            "category": category,
            "intent": intent,
            "urgency": urgency,
            "automation_potential": "medium",
            "suggested_actions": [f"Route to {category} handler"]
        }
    
    async def _handle_email_request(self, request: str, context: Optional[Dict[str, Any]], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle email-related requests."""
        intent = analysis.get("intent", "email_operation")
        
        if "compose" in request.lower() or "write" in request.lower():
            result = await self.email_assistant.compose_email(request, context)
        elif "analyze" in request.lower():
            # Would need email content from context
            email_content = context.get("email_content", "") if context else ""
            if email_content:
                analysis_result = await self.email_assistant.analyze_email("", email_content)
                result = {"success": True, "analysis": analysis_result}
            else:
                result = {"success": False, "message": "No email content provided for analysis"}
        elif "reply" in request.lower():
            # Would need original email from context
            original_email = context.get("original_email", {}) if context else {}
            if original_email:
                suggestions = await self.email_assistant.suggest_replies(
                    original_email.get("subject", ""),
                    original_email.get("body", ""),
                    original_email.get("sender", "")
                )
                result = {"success": True, "reply_suggestions": suggestions}
            else:
                result = {"success": False, "message": "No original email provided for reply"}
        else:
            result = {"success": False, "message": f"Unknown email request: {request}"}
        
        # Update metrics
        if result.get("success"):
            self.productivity_metrics["emails_processed"] += 1
        
        return result
    
    async def _handle_document_request(self, request: str, context: Optional[Dict[str, Any]], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle document-related requests."""
        if "create" in request.lower() or "generate" in request.lower():
            doc_type = context.get("document_type", "text") if context else "text"
            result = await self.document_assistant.generate_document(request, doc_type, context=context)
        elif "analyze" in request.lower():
            file_path = context.get("file_path") if context else None
            if file_path:
                analysis_result = await self.document_assistant.analyze_document(file_path)
                result = {"success": True, "analysis": analysis_result}
            else:
                result = {"success": False, "message": "No file path provided for analysis"}
        elif "summarize" in request.lower():
            file_path = context.get("file_path") if context else None
            if file_path:
                summary_result = await self.document_assistant.summarize_document(file_path)
                result = summary_result
            else:
                result = {"success": False, "message": "No file path provided for summarization"}
        else:
            result = {"success": False, "message": f"Unknown document request: {request}"}
        
        # Update metrics
        if result.get("success"):
            self.productivity_metrics["documents_created"] += 1
        
        return result
    
    async def _handle_workflow_request(self, request: str, context: Optional[Dict[str, Any]], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle workflow-related requests."""
        # This would integrate with calendar systems
        return {
            "success": True,
            "message": "Workflow request acknowledged",
            "note": "Calendar integration not implemented in this version"
        }
    
    async def _handle_productivity_request(self, request: str, context: Optional[Dict[str, Any]], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle productivity analysis requests."""
        timeframe = context.get("timeframe", "week") if context else "week"
        return await self.analyze_productivity(timeframe)
    
    async def _handle_automation_request(self, request: str, context: Optional[Dict[str, Any]], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle automation requests."""
        workflow_name = context.get("workflow_name") if context else None
        
        if workflow_name and workflow_name in self.automation_workflows:
            parameters = context.get("parameters", {}) if context else {}
            return await self.automate_workflow(workflow_name, parameters)
        else:
            available_workflows = list(self.automation_workflows.keys())
            return {
                "success": False,
                "message": "Please specify a workflow to automate",
                "available_workflows": available_workflows
            }
    
    async def _handle_general_request(self, request: str, context: Optional[Dict[str, Any]], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general office requests."""
        # Try to route to autonomous agent for general task handling
        try:
            suggested_tasks = await self.autonomous_agent.suggest_task(request)
            
            return {
                "success": True,
                "message": "General request analyzed",
                "suggested_tasks": [
                    {
                        "id": task.id,
                        "title": task.title,
                        "description": task.description,
                        "type": task.task_type,
                        "priority": task.priority
                    }
                    for task in suggested_tasks
                ],
                "note": "Tasks created for autonomous execution"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to process general request"
            }
    
    async def _get_email_analytics(self, timeframe: str) -> Dict[str, Any]:
        """Get email analytics for timeframe."""
        # This would integrate with email systems to get real data
        return {
            "emails_sent": 25,
            "emails_received": 67,
            "response_time_avg_hours": 2.5,
            "unread_count": 12,
            "important_emails": 8,
            "spam_filtered": 23
        }
    
    async def _get_document_analytics(self, timeframe: str) -> Dict[str, Any]:
        """Get document analytics for timeframe."""
        return {
            "documents_created": 8,
            "documents_edited": 15,
            "total_words_written": 12500,
            "avg_document_length": 1562,
            "document_types": {
                "reports": 3,
                "emails": 12,
                "notes": 8
            }
        }
    
    async def _get_task_analytics(self, timeframe: str) -> Dict[str, Any]:
        """Get task analytics for timeframe."""
        return {
            "tasks_completed": 18,
            "tasks_pending": 7,
            "avg_completion_time_hours": 3.2,
            "overdue_tasks": 2,
            "automation_percentage": 35
        }
    
    async def _generate_productivity_insights(
        self,
        email_data: Dict[str, Any],
        document_data: Dict[str, Any],
        task_data: Dict[str, Any],
        focus_areas: Optional[List[str]] = None
    ) -> List[ProductivityInsight]:
        """Generate productivity insights from analytics data."""
        insights = []
        
        # Email insights
        if email_data["response_time_avg_hours"] > 4:
            insights.append(ProductivityInsight(
                category="email",
                insight="Email response time is above optimal range",
                recommendation="Set up email templates and automate routine responses",
                impact_level="medium",
                confidence=0.8
            ))
        
        # Document insights
        if document_data["documents_created"] < 5:
            insights.append(ProductivityInsight(
                category="documents",
                insight="Low document creation rate",
                recommendation="Use document templates and AI assistance for faster creation",
                impact_level="low",
                confidence=0.7
            ))
        
        # Task insights
        if task_data["overdue_tasks"] > 0:
            insights.append(ProductivityInsight(
                category="tasks",
                insight="Multiple overdue tasks detected",
                recommendation="Review task prioritization and deadlines",
                impact_level="high",
                confidence=0.9
            ))
        
        return insights
    
    async def _generate_productivity_recommendations(self, insights: List[ProductivityInsight]) -> List[str]:
        """Generate actionable productivity recommendations."""
        recommendations = []
        
        for insight in insights:
            if insight.impact_level in ["high", "medium"]:
                recommendations.append(insight.recommendation)
        
        # Add general recommendations
        recommendations.extend([
            "Consider automating routine tasks",
            "Use templates for common document types",
            "Set up email filters and rules",
            "Schedule focused work time blocks"
        ])
        
        return recommendations[:5]  # Limit to top 5
    
    def _calculate_productivity_score(self, insights: List[ProductivityInsight]) -> float:
        """Calculate overall productivity score."""
        if not insights:
            return 0.8  # Neutral score
        
        # Simple scoring based on impact levels
        score = 1.0
        for insight in insights:
            if insight.impact_level == "high":
                score -= 0.2
            elif insight.impact_level == "medium":
                score -= 0.1
            elif insight.impact_level == "low":
                score -= 0.05
        
        return max(0.0, min(1.0, score))
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate efficiency score based on metrics."""
        total_actions = (
            self.productivity_metrics["emails_processed"] +
            self.productivity_metrics["documents_created"] +
            self.productivity_metrics["tasks_automated"]
        )
        
        if total_actions == 0:
            return 0.0
        
        # Simple efficiency calculation
        automation_ratio = self.productivity_metrics["tasks_automated"] / total_actions
        return min(1.0, automation_ratio * 2)  # Cap at 1.0
    
    def _calculate_automation_usage(self) -> float:
        """Calculate automation usage percentage."""
        total_tasks = self.productivity_metrics["tasks_automated"] + 10  # Assume some manual tasks
        return self.productivity_metrics["tasks_automated"] / total_tasks
    
    def _load_default_workflows(self) -> None:
        """Load default automation workflows."""
        default_workflows = {
            "daily_email_summary": {
                "description": "Generate daily email summary report",
                "category": "email",
                "automation_level": "full_auto",
                "estimated_time": "15 minutes",
                "steps": [
                    {
                        "title": "Collect Email Data",
                        "description": "Gather email metrics and important messages",
                        "type": "analysis",
                        "actions": [{"type": "analyze_emails", "params": {"timeframe": "day"}}],
                        "priority": "medium",
                        "requires_approval": False
                    },
                    {
                        "title": "Generate Summary",
                        "description": "Create email summary document",
                        "type": "automation",
                        "actions": [{"type": "generate_report", "params": {"type": "email_summary"}}],
                        "priority": "medium",
                        "requires_approval": False
                    }
                ]
            },
            
            "weekly_productivity_report": {
                "description": "Generate weekly productivity analysis",
                "category": "productivity",
                "automation_level": "semi_auto",
                "estimated_time": "30 minutes",
                "steps": [
                    {
                        "title": "Analyze Productivity Metrics",
                        "description": "Collect and analyze weekly productivity data",
                        "type": "analysis",
                        "actions": [{"type": "analyze_productivity", "params": {"timeframe": "week"}}],
                        "priority": "medium",
                        "requires_approval": False
                    },
                    {
                        "title": "Generate Report",
                        "description": "Create comprehensive productivity report",
                        "type": "automation",
                        "actions": [{"type": "generate_document", "params": {"type": "productivity_report"}}],
                        "priority": "medium",
                        "requires_approval": True
                    }
                ]
            },
            
            "meeting_preparation": {
                "description": "Prepare for upcoming meetings",
                "category": "workflow",
                "automation_level": "semi_auto",
                "estimated_time": "20 minutes",
                "steps": [
                    {
                        "title": "Gather Meeting Context",
                        "description": "Collect relevant documents and previous discussions",
                        "type": "analysis",
                        "actions": [{"type": "search_documents", "params": {"query": "meeting_topic"}}],
                        "priority": "high",
                        "requires_approval": False
                    },
                    {
                        "title": "Create Meeting Notes Template",
                        "description": "Generate meeting notes template",
                        "type": "automation",
                        "actions": [{"type": "generate_document", "params": {"template": "meeting_notes"}}],
                        "priority": "medium",
                        "requires_approval": False
                    }
                ]
            }
        }
        
        self.automation_workflows.update(default_workflows)
        logger.info(f"Loaded {len(default_workflows)} default automation workflows")