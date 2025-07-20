"""
Email Assistant for Eidolon AI Personal Assistant

Provides intelligent email composition, analysis, and management capabilities.
"""

import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

from ..utils.logging import get_component_logger
from ..utils.config import get_config
from ..models.cloud_api import CloudAPIManager
from ..tools.communication import CommunicationTool
from ..core.safety import SafetyManager

logger = get_component_logger("assistants.email")


@dataclass
class EmailAnalysis:
    """Analysis result for an email."""
    sentiment: str  # positive, negative, neutral
    urgency: str   # low, medium, high, urgent
    category: str  # work, personal, promotional, etc.
    confidence: float
    summary: str
    key_points: List[str]
    suggested_actions: List[str]
    contains_sensitive_data: bool


@dataclass
class EmailTemplate:
    """Email template structure."""
    name: str
    subject_template: str
    body_template: str
    category: str
    description: str
    variables: List[str]


class EmailAssistant:
    """
    Intelligent email assistant for composition, analysis, and management.
    """
    
    def __init__(self, config_override: Optional[Dict] = None):
        """Initialize email assistant."""
        self.config = get_config()
        if config_override:
            self.config.update(config_override)
        
        self.cloud_api = CloudAPIManager()
        self.communication_tool = CommunicationTool()
        self.safety_manager = SafetyManager()
        
        # Load email templates
        self.templates: Dict[str, EmailTemplate] = {}
        self._load_default_templates()
        
        # Email patterns and rules
        self.urgent_keywords = [
            'urgent', 'asap', 'emergency', 'critical', 'deadline',
            'immediately', 'rush', 'priority', 'time sensitive'
        ]
        
        self.work_indicators = [
            'meeting', 'project', 'deadline', 'report', 'presentation',
            'client', 'budget', 'invoice', 'contract', 'proposal'
        ]
        
        self.personal_indicators = [
            'family', 'friend', 'vacation', 'birthday', 'dinner',
            'weekend', 'party', 'hobby', 'personal'
        ]
        
        logger.info("Email assistant initialized")
    
    async def compose_email(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
        template_name: Optional[str] = None,
        recipients: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compose an email based on user request.
        
        Args:
            request: User's request for email content
            context: Additional context information
            template_name: Optional template to use
            recipients: Optional recipient list
            
        Returns:
            Composed email data
        """
        try:
            logger.info(f"Composing email for request: {request[:100]}...")
            
            # Use template if specified
            if template_name and template_name in self.templates:
                template = self.templates[template_name]
                return await self._compose_from_template(template, request, context)
            
            # AI-powered composition
            return await self._compose_with_ai(request, context, recipients)
            
        except Exception as e:
            logger.error(f"Email composition failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to compose email"
            }
    
    async def analyze_email(
        self,
        subject: str,
        body: str,
        sender: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EmailAnalysis:
        """
        Analyze an email for sentiment, urgency, and category.
        
        Args:
            subject: Email subject
            body: Email body
            sender: Email sender
            metadata: Additional metadata
            
        Returns:
            Email analysis result
        """
        try:
            logger.debug(f"Analyzing email: {subject}")
            
            # Basic analysis
            sentiment = self._analyze_sentiment(body)
            urgency = self._analyze_urgency(subject, body)
            category = self._categorize_email(subject, body, sender)
            
            # Check for sensitive data
            sensitive_data = self.safety_manager._contains_sensitive_data(body)
            
            # Use AI for deeper analysis if available
            if self.cloud_api:
                ai_analysis = await self._ai_analyze_email(subject, body)
                
                return EmailAnalysis(
                    sentiment=ai_analysis.get("sentiment", sentiment),
                    urgency=ai_analysis.get("urgency", urgency),
                    category=ai_analysis.get("category", category),
                    confidence=ai_analysis.get("confidence", 0.7),
                    summary=ai_analysis.get("summary", ""),
                    key_points=ai_analysis.get("key_points", []),
                    suggested_actions=ai_analysis.get("suggested_actions", []),
                    contains_sensitive_data=sensitive_data
                )
            else:
                # Fallback to basic analysis
                return EmailAnalysis(
                    sentiment=sentiment,
                    urgency=urgency,
                    category=category,
                    confidence=0.6,
                    summary=self._extract_summary(body),
                    key_points=self._extract_key_points(body),
                    suggested_actions=self._suggest_actions(subject, body, urgency),
                    contains_sensitive_data=sensitive_data
                )
                
        except Exception as e:
            logger.error(f"Email analysis failed: {e}")
            return EmailAnalysis(
                sentiment="neutral",
                urgency="medium",
                category="unknown",
                confidence=0.0,
                summary="Analysis failed",
                key_points=[],
                suggested_actions=[],
                contains_sensitive_data=False
            )
    
    async def suggest_replies(
        self,
        original_subject: str,
        original_body: str,
        sender: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Suggest reply options for an email.
        
        Args:
            original_subject: Original email subject
            original_body: Original email body
            sender: Original sender
            context: Additional context
            
        Returns:
            List of suggested replies
        """
        try:
            # Analyze the original email
            analysis = await self.analyze_email(original_subject, original_body, sender)
            
            suggestions = []
            
            # Generate different types of replies based on analysis
            if analysis.urgency in ["high", "urgent"]:
                suggestions.append(await self._generate_urgent_reply(original_body, analysis))
            
            if analysis.category == "work":
                suggestions.append(await self._generate_professional_reply(original_body, analysis))
            
            if analysis.category == "personal":
                suggestions.append(await self._generate_casual_reply(original_body, analysis))
            
            # Generic acknowledgment
            suggestions.append(await self._generate_acknowledgment_reply(original_body, analysis))
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Reply suggestion failed: {e}")
            return []
    
    async def manage_email_workflow(
        self,
        action: str,
        email_data: Dict[str, Any],
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Manage email workflow actions.
        
        Args:
            action: Action to perform (send, schedule, draft, etc.)
            email_data: Email data
            parameters: Action parameters
            
        Returns:
            Workflow result
        """
        try:
            if action == "send":
                return await self._send_email(email_data, parameters)
            elif action == "schedule":
                return await self._schedule_email(email_data, parameters)
            elif action == "save_draft":
                return await self._save_draft(email_data, parameters)
            elif action == "validate":
                return await self._validate_email(email_data)
            else:
                raise ValueError(f"Unknown workflow action: {action}")
                
        except Exception as e:
            logger.error(f"Email workflow failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Workflow action '{action}' failed"
            }
    
    def add_template(self, template: EmailTemplate) -> bool:
        """Add a custom email template."""
        try:
            self.templates[template.name] = template
            logger.info(f"Added email template: {template.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add template: {e}")
            return False
    
    def get_templates(self) -> List[EmailTemplate]:
        """Get all available email templates."""
        return list(self.templates.values())
    
    def get_template(self, name: str) -> Optional[EmailTemplate]:
        """Get a specific template by name."""
        return self.templates.get(name)
    
    async def _compose_from_template(
        self,
        template: EmailTemplate,
        request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Compose email from template."""
        try:
            # Extract variables from request using AI
            variables = {}
            if self.cloud_api:
                extraction_prompt = f"""
                Extract variable values for this email template:
                Template: {template.name}
                Variables needed: {template.variables}
                User request: {request}
                Context: {context or {}}
                
                Return a JSON object with variable names and values.
                """
                
                response = await self.cloud_api.analyze_text(
                    extraction_prompt,
                    analysis_type="variable_extraction"
                )
                
                if response and response.content:
                    try:
                        variables = json.loads(response.content)
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse AI variable extraction")
            
            # Apply template
            subject = template.subject_template
            body = template.body_template
            
            for var_name, var_value in variables.items():
                placeholder = f"{{{var_name}}}"
                subject = subject.replace(placeholder, str(var_value))
                body = body.replace(placeholder, str(var_value))
            
            return {
                "success": True,
                "subject": subject,
                "body": body,
                "template_used": template.name,
                "variables": variables
            }
            
        except Exception as e:
            logger.error(f"Template composition failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to compose from template"
            }
    
    async def _compose_with_ai(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None,
        recipients: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compose email using AI."""
        try:
            if not self.cloud_api:
                return await self._compose_basic(request, context)
            
            # Determine email tone and style from request
            tone = self._determine_tone(request)
            
            composition_prompt = f"""
            Compose a professional email based on this request:
            
            Request: {request}
            Tone: {tone}
            Recipients: {recipients or 'Not specified'}
            Context: {context or 'None provided'}
            
            Guidelines:
            - Be clear and concise
            - Use appropriate tone ({tone})
            - Include a clear subject line
            - Structure with proper greeting and closing
            - Avoid sensitive information
            
            Return JSON with 'subject' and 'body' fields.
            """
            
            response = await self.cloud_api.analyze_text(
                composition_prompt,
                analysis_type="email_composition"
            )
            
            if response and response.content:
                try:
                    result = json.loads(response.content)
                    return {
                        "success": True,
                        "subject": result.get("subject", ""),
                        "body": result.get("body", ""),
                        "tone": tone,
                        "ai_generated": True
                    }
                except json.JSONDecodeError:
                    logger.warning("Failed to parse AI composition result")
            
            # Fallback to basic composition
            return await self._compose_basic(request, context)
            
        except Exception as e:
            logger.error(f"AI composition failed: {e}")
            return await self._compose_basic(request, context)
    
    async def _compose_basic(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Basic email composition without AI."""
        # Simple template-based composition
        subject = f"Re: {request[:50]}..."
        
        body = f"""Hello,

{request}

Best regards,
[Your name]
"""
        
        return {
            "success": True,
            "subject": subject,
            "body": body,
            "ai_generated": False,
            "note": "Basic composition - consider using AI for better results"
        }
    
    async def _ai_analyze_email(
        self,
        subject: str,
        body: str
    ) -> Dict[str, Any]:
        """Use AI to analyze email content."""
        try:
            analysis_prompt = f"""
            Analyze this email and provide:
            
            Subject: {subject}
            Body: {body[:1000]}...
            
            Return JSON with:
            - sentiment: positive/negative/neutral
            - urgency: low/medium/high/urgent
            - category: work/personal/promotional/support/other
            - confidence: 0.0-1.0
            - summary: brief summary
            - key_points: list of key points
            - suggested_actions: list of suggested actions
            """
            
            response = await self.cloud_api.analyze_text(
                analysis_prompt,
                analysis_type="email_analysis"
            )
            
            if response and response.content:
                return json.loads(response.content)
            
        except Exception as e:
            logger.warning(f"AI email analysis failed: {e}")
        
        return {}
    
    def _analyze_sentiment(self, text: str) -> str:
        """Basic sentiment analysis."""
        positive_words = ['thank', 'great', 'excellent', 'good', 'please', 'appreciate']
        negative_words = ['problem', 'issue', 'error', 'urgent', 'complaint', 'disappointed']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _analyze_urgency(self, subject: str, body: str) -> str:
        """Analyze email urgency."""
        text = f"{subject} {body}".lower()
        
        urgent_count = sum(1 for keyword in self.urgent_keywords if keyword in text)
        
        if urgent_count >= 3:
            return "urgent"
        elif urgent_count >= 2:
            return "high"
        elif urgent_count >= 1:
            return "medium"
        else:
            return "low"
    
    def _categorize_email(self, subject: str, body: str, sender: Optional[str] = None) -> str:
        """Categorize email content."""
        text = f"{subject} {body}".lower()
        
        work_score = sum(1 for indicator in self.work_indicators if indicator in text)
        personal_score = sum(1 for indicator in self.personal_indicators if indicator in text)
        
        # Check sender domain for additional context
        if sender:
            domain = sender.split('@')[-1].lower()
            if domain in ['gmail.com', 'yahoo.com', 'hotmail.com']:
                personal_score += 1
            else:
                work_score += 1
        
        if work_score > personal_score:
            return "work"
        elif personal_score > 0:
            return "personal"
        else:
            return "other"
    
    def _determine_tone(self, request: str) -> str:
        """Determine appropriate email tone from request."""
        request_lower = request.lower()
        
        if any(word in request_lower for word in ['formal', 'professional', 'business', 'official']):
            return "formal"
        elif any(word in request_lower for word in ['casual', 'friendly', 'informal']):
            return "casual"
        elif any(word in request_lower for word in ['urgent', 'asap', 'important']):
            return "urgent"
        else:
            return "professional"
    
    def _extract_summary(self, body: str) -> str:
        """Extract a brief summary from email body."""
        # Simple extraction - take first sentence or paragraph
        sentences = body.split('. ')
        if sentences:
            return sentences[0][:200] + "..." if len(sentences[0]) > 200 else sentences[0]
        return ""
    
    def _extract_key_points(self, body: str) -> List[str]:
        """Extract key points from email body."""
        # Look for bullet points, numbered lists, or key phrases
        key_points = []
        
        # Find bullet points
        bullet_pattern = r'[•\-\*]\s*(.+)'
        bullets = re.findall(bullet_pattern, body)
        key_points.extend(bullets[:5])  # Limit to 5
        
        # Find numbered items
        number_pattern = r'\d+\.\s*(.+)'
        numbers = re.findall(number_pattern, body)
        key_points.extend(numbers[:3])  # Limit to 3
        
        return key_points[:5]  # Overall limit
    
    def _suggest_actions(self, subject: str, body: str, urgency: str) -> List[str]:
        """Suggest actions based on email content."""
        actions = []
        
        if urgency in ["high", "urgent"]:
            actions.append("Respond promptly")
        
        if "meeting" in body.lower():
            actions.append("Check calendar availability")
        
        if "attachment" in body.lower():
            actions.append("Review attachments")
        
        if "deadline" in body.lower():
            actions.append("Add to task list with deadline")
        
        if "?" in body:
            actions.append("Answer questions raised")
        
        return actions
    
    async def _generate_urgent_reply(self, original_body: str, analysis: EmailAnalysis) -> Dict[str, Any]:
        """Generate urgent reply template."""
        return {
            "type": "urgent",
            "subject": "Re: [Urgent Response]",
            "body": "Thank you for your urgent message. I have received it and will address this immediately.\n\nI will get back to you within [timeframe] with a detailed response.\n\nBest regards,",
            "priority": "high"
        }
    
    async def _generate_professional_reply(self, original_body: str, analysis: EmailAnalysis) -> Dict[str, Any]:
        """Generate professional reply template."""
        return {
            "type": "professional",
            "subject": "Re: Professional Response",
            "body": "Thank you for your email.\n\n[Address key points from original message]\n\nPlease let me know if you need any additional information.\n\nBest regards,",
            "priority": "medium"
        }
    
    async def _generate_casual_reply(self, original_body: str, analysis: EmailAnalysis) -> Dict[str, Any]:
        """Generate casual reply template."""
        return {
            "type": "casual",
            "subject": "Re: [Casual Response]",
            "body": "Hi!\n\nThanks for your email. \n\n[Personal response based on content]\n\nTalk soon!",
            "priority": "low"
        }
    
    async def _generate_acknowledgment_reply(self, original_body: str, analysis: EmailAnalysis) -> Dict[str, Any]:
        """Generate acknowledgment reply template."""
        return {
            "type": "acknowledgment",
            "subject": "Re: Acknowledgment",
            "body": "Thank you for your email. I have received it and will review the details.\n\nI will respond with more information shortly.\n\nBest regards,",
            "priority": "medium"
        }
    
    async def _send_email(self, email_data: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send email using communication tool."""
        try:
            result = await self.communication_tool.execute({
                "operation": "send_email",
                **email_data,
                **(parameters or {})
            })
            
            return {
                "success": result.success,
                "message": result.message,
                "data": result.data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to send email"
            }
    
    async def _schedule_email(self, email_data: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Schedule email for later sending."""
        # This would integrate with a scheduling system
        return {
            "success": True,
            "message": "Email scheduled (scheduling system not implemented)",
            "scheduled_time": parameters.get("send_time") if parameters else None
        }
    
    async def _save_draft(self, email_data: Dict[str, Any], parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Save email as draft."""
        try:
            result = await self.communication_tool.execute({
                "operation": "draft_email",
                **email_data
            })
            
            return {
                "success": result.success,
                "message": result.message,
                "draft": result.data.get("draft") if result.success else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to save draft"
            }
    
    async def _validate_email(self, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate email data."""
        try:
            result = await self.communication_tool.execute({
                "operation": "validate_email",
                "email_addresses": email_data.get("to", []) + email_data.get("cc", []),
                "content": email_data.get("body", "")
            })
            
            return {
                "success": result.success,
                "message": result.message,
                "validation": result.data
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to validate email"
            }
    
    def _load_default_templates(self) -> None:
        """Load default email templates."""
        default_templates = [
            EmailTemplate(
                name="meeting_request",
                subject_template="Meeting Request: {topic}",
                body_template="""Hello {recipient},

I would like to schedule a meeting to discuss {topic}.

Proposed time: {date} at {time}
Duration: {duration}
Location: {location}

Please let me know if this works for your schedule.

Best regards,
{sender}""",
                category="work",
                description="Request a meeting",
                variables=["recipient", "topic", "date", "time", "duration", "location", "sender"]
            ),
            
            EmailTemplate(
                name="follow_up",
                subject_template="Follow-up: {original_subject}",
                body_template="""Hello {recipient},

I wanted to follow up on {topic} that we discussed {timeframe}.

{follow_up_message}

Please let me know if you need any additional information.

Best regards,
{sender}""",
                category="work",
                description="Follow up on previous communication",
                variables=["recipient", "topic", "timeframe", "follow_up_message", "sender"]
            ),
            
            EmailTemplate(
                name="thank_you",
                subject_template="Thank you - {reason}",
                body_template="""Hello {recipient},

Thank you for {reason}. {appreciation_message}

I look forward to {next_steps}.

Best regards,
{sender}""",
                category="personal",
                description="Express gratitude",
                variables=["recipient", "reason", "appreciation_message", "next_steps", "sender"]
            )
        ]
        
        for template in default_templates:
            self.templates[template.name] = template
        
        logger.info(f"Loaded {len(default_templates)} default email templates")