"""
Communication Tool for Eidolon Tool Orchestration Framework

Provides email, messaging, and communication capabilities.
"""

import smtplib
import imaplib
import email
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Dict, Any, Optional, List
from pathlib import Path
import re

from .base import BaseTool, ToolMetadata, ToolResult, ToolError
from ..core.safety import RiskLevel, ActionCategory
from ..utils.logging import get_component_logger

logger = get_component_logger("tools.communication")


class CommunicationTool(BaseTool):
    """Tool for email and messaging operations."""
    
    METADATA = ToolMetadata(
        name="communication",
        description="Send emails and manage communication with safety controls",
        category="communication",
        risk_level=RiskLevel.MEDIUM,
        action_category=ActionCategory.EMAIL_SEND,
        requires_approval=True,
        timeout_seconds=60.0,
        input_schema={
            "required": ["operation"],
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["send_email", "read_emails", "draft_email", "validate_email"]
                },
                "to": {"type": "array", "items": {"type": "string"}},
                "cc": {"type": "array", "items": {"type": "string"}},
                "bcc": {"type": "array", "items": {"type": "string"}},
                "subject": {"type": "string"},
                "body": {"type": "string"},
                "html_body": {"type": "string"},
                "attachments": {"type": "array", "items": {"type": "string"}},
                "smtp_server": {"type": "string"},
                "smtp_port": {"type": "integer"},
                "username": {"type": "string"},
                "password": {"type": "string"},
                "use_tls": {"type": "boolean", "default": True}
            }
        }
    )
    
    def __init__(self, metadata: Optional[ToolMetadata] = None):
        """Initialize communication tool."""
        super().__init__(metadata or self.METADATA)
        
        # Safety controls
        self.max_recipients = 50
        self.max_attachment_size = 25 * 1024 * 1024  # 25MB
        self.max_email_size = 50 * 1024 * 1024  # 50MB
        self.allowed_domains = set()  # Empty means all allowed
        self.blocked_domains = {
            'spam.com', 'tempmail.org', '10minutemail.com'
        }
        
        # Email validation patterns
        self.email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
        
        # Sensitive content patterns
        self.sensitive_patterns = [
            r'password[s]?\s*[:=]\s*[\w\d]+',
            r'api[_-]?key[s]?\s*[:=]\s*[\w\d-]+',
            r'secret[s]?\s*[:=]\s*[\w\d]+',
            r'token[s]?\s*[:=]\s*[\w\d-]+',
            r'\d{3}-\d{2}-\d{4}',  # SSN
            r'\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}',  # Credit card
        ]
    
    async def execute(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> ToolResult:
        """Execute communication operation."""
        try:
            # Validate parameters
            validated_params = await self.validate_parameters(parameters)
            operation = validated_params["operation"]
            
            # Route to specific operation
            if operation == "send_email":
                return await self._send_email(validated_params)
            elif operation == "read_emails":
                return await self._read_emails(validated_params)
            elif operation == "draft_email":
                return await self._draft_email(validated_params)
            elif operation == "validate_email":
                return await self._validate_email(validated_params)
            else:
                raise ToolError(f"Unknown operation: {operation}")
                
        except Exception as e:
            logger.error(f"Communication operation failed: {e}")
            return ToolResult(
                success=False,
                data={"error": str(e)},
                message=f"Communication operation failed: {str(e)}"
            )
    
    async def _send_email(self, params: Dict[str, Any]) -> ToolResult:
        """Send an email."""
        to_addresses = params.get("to", [])
        cc_addresses = params.get("cc", [])
        bcc_addresses = params.get("bcc", [])
        subject = params.get("subject", "")
        body = params.get("body", "")
        html_body = params.get("html_body")
        attachments = params.get("attachments", [])
        
        # SMTP configuration
        smtp_server = params.get("smtp_server")
        smtp_port = params.get("smtp_port", 587)
        username = params.get("username")
        password = params.get("password")
        use_tls = params.get("use_tls", True)
        
        # Validation
        all_recipients = to_addresses + cc_addresses + bcc_addresses
        
        if not to_addresses:
            raise ToolError("At least one recipient is required")
        
        if len(all_recipients) > self.max_recipients:
            raise ToolError(f"Too many recipients: {len(all_recipients)} (max: {self.max_recipients})")
        
        # Validate email addresses
        for email_addr in all_recipients:
            if not self._is_valid_email(email_addr):
                raise ToolError(f"Invalid email address: {email_addr}")
            
            if not self._is_email_allowed(email_addr):
                raise ToolError(f"Email address not allowed: {email_addr}")
        
        # Check for sensitive content
        if self._contains_sensitive_content(body) or self._contains_sensitive_content(subject):
            raise ToolError("Email contains sensitive content")
        
        # Validate attachments
        total_attachment_size = 0
        for attachment_path in attachments:
            path = Path(attachment_path)
            if not path.exists():
                raise ToolError(f"Attachment not found: {attachment_path}")
            
            file_size = path.stat().st_size
            if file_size > self.max_attachment_size:
                raise ToolError(f"Attachment too large: {path.name} ({file_size} bytes)")
            
            total_attachment_size += file_size
        
        if not smtp_server or not username:
            raise ToolError("SMTP server and username are required")
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['From'] = username
            msg['To'] = ', '.join(to_addresses)
            if cc_addresses:
                msg['Cc'] = ', '.join(cc_addresses)
            msg['Subject'] = subject
            
            # Add body
            if html_body:
                # Both plain text and HTML
                text_part = MIMEText(body, 'plain')
                html_part = MIMEText(html_body, 'html')
                msg.attach(text_part)
                msg.attach(html_part)
            else:
                # Plain text only
                text_part = MIMEText(body, 'plain')
                msg.attach(text_part)
            
            # Add attachments
            for attachment_path in attachments:
                path = Path(attachment_path)
                
                with open(path, 'rb') as f:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(f.read())
                
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {path.name}'
                )
                msg.attach(part)
            
            # Check total email size
            email_size = len(msg.as_string())
            if email_size > self.max_email_size:
                raise ToolError(f"Email too large: {email_size} bytes (max: {self.max_email_size})")
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if use_tls:
                    server.starttls()
                
                server.login(username, password)
                
                # Send to all recipients
                recipients = to_addresses + cc_addresses + bcc_addresses
                server.send_message(msg, to_addrs=recipients)
            
            return ToolResult(
                success=True,
                data={
                    "to": to_addresses,
                    "cc": cc_addresses,
                    "bcc": bcc_addresses,
                    "subject": subject,
                    "body_length": len(body),
                    "attachments": len(attachments),
                    "total_size": email_size,
                    "recipients_count": len(recipients)
                },
                message=f"Email sent to {len(recipients)} recipients",
                side_effects=[f"Sent email: {subject}"]
            )
            
        except smtplib.SMTPException as e:
            raise ToolError(f"SMTP error: {e}")
        except Exception as e:
            raise ToolError(f"Failed to send email: {e}")
    
    async def _read_emails(self, params: Dict[str, Any]) -> ToolResult:
        """Read emails from an IMAP server."""
        # IMAP configuration
        imap_server = params.get("imap_server")
        username = params.get("username")
        password = params.get("password")
        mailbox = params.get("mailbox", "INBOX")
        limit = params.get("limit", 10)
        unread_only = params.get("unread_only", False)
        
        if not imap_server or not username or not password:
            raise ToolError("IMAP server, username, and password are required")
        
        if limit > 100:
            limit = 100  # Safety limit
        
        try:
            # Connect to IMAP server
            with imaplib.IMAP4_SSL(imap_server) as server:
                server.login(username, password)
                server.select(mailbox)
                
                # Search for emails
                search_criteria = "UNSEEN" if unread_only else "ALL"
                typ, data = server.search(None, search_criteria)
                
                if typ != 'OK':
                    raise ToolError("Failed to search emails")
                
                email_ids = data[0].split()
                email_ids = email_ids[-limit:]  # Get latest emails
                
                emails = []
                for email_id in email_ids:
                    typ, msg_data = server.fetch(email_id, '(RFC822)')
                    
                    if typ != 'OK':
                        continue
                    
                    # Parse email
                    msg = email.message_from_bytes(msg_data[0][1])
                    
                    # Extract basic info
                    email_info = {
                        "id": email_id.decode(),
                        "subject": msg.get("Subject", ""),
                        "from": msg.get("From", ""),
                        "to": msg.get("To", ""),
                        "date": msg.get("Date", ""),
                        "has_attachments": any(part.get_filename() for part in msg.walk())
                    }
                    
                    # Extract body (basic implementation)
                    body = ""
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode('utf-8', errors='replace')
                            break
                    
                    email_info["body_preview"] = body[:500] if body else ""
                    emails.append(email_info)
                
                return ToolResult(
                    success=True,
                    data={
                        "emails": emails,
                        "count": len(emails),
                        "mailbox": mailbox,
                        "unread_only": unread_only
                    },
                    message=f"Retrieved {len(emails)} emails from {mailbox}"
                )
                
        except imaplib.IMAP4.error as e:
            raise ToolError(f"IMAP error: {e}")
        except Exception as e:
            raise ToolError(f"Failed to read emails: {e}")
    
    async def _draft_email(self, params: Dict[str, Any]) -> ToolResult:
        """Create an email draft without sending."""
        to_addresses = params.get("to", [])
        cc_addresses = params.get("cc", [])
        subject = params.get("subject", "")
        body = params.get("body", "")
        template = params.get("template")
        
        # Validate recipients
        all_recipients = to_addresses + cc_addresses
        for email_addr in all_recipients:
            if not self._is_valid_email(email_addr):
                raise ToolError(f"Invalid email address: {email_addr}")
        
        # Apply template if specified
        if template:
            body = self._apply_email_template(template, params)
        
        # Check for sensitive content
        if self._contains_sensitive_content(body) or self._contains_sensitive_content(subject):
            logger.warning("Draft contains sensitive content")
        
        # Create draft structure
        draft = {
            "to": to_addresses,
            "cc": cc_addresses,
            "subject": subject,
            "body": body,
            "created_at": "now",
            "word_count": len(body.split()),
            "character_count": len(body),
            "estimated_read_time": max(1, len(body.split()) // 200)  # ~200 WPM
        }
        
        return ToolResult(
            success=True,
            data={"draft": draft},
            message=f"Created email draft: {subject}"
        )
    
    async def _validate_email(self, params: Dict[str, Any]) -> ToolResult:
        """Validate email addresses and content."""
        email_addresses = params.get("email_addresses", [])
        content = params.get("content", "")
        
        results = {
            "valid_emails": [],
            "invalid_emails": [],
            "blocked_emails": [],
            "content_issues": []
        }
        
        # Validate email addresses
        for email_addr in email_addresses:
            if not self._is_valid_email(email_addr):
                results["invalid_emails"].append(email_addr)
            elif not self._is_email_allowed(email_addr):
                results["blocked_emails"].append(email_addr)
            else:
                results["valid_emails"].append(email_addr)
        
        # Check content
        if self._contains_sensitive_content(content):
            results["content_issues"].append("Contains sensitive information")
        
        if len(content) > self.max_email_size:
            results["content_issues"].append(f"Content too long: {len(content)} characters")
        
        # Overall validation result
        is_valid = (
            len(results["invalid_emails"]) == 0 and
            len(results["blocked_emails"]) == 0 and
            len(results["content_issues"]) == 0
        )
        
        return ToolResult(
            success=is_valid,
            data=results,
            message=f"Validation {'passed' if is_valid else 'failed'}"
        )
    
    def _is_valid_email(self, email_addr: str) -> bool:
        """Validate email address format."""
        if not email_addr or len(email_addr) > 254:
            return False
        
        return bool(self.email_pattern.match(email_addr))
    
    def _is_email_allowed(self, email_addr: str) -> bool:
        """Check if email domain is allowed."""
        try:
            domain = email_addr.split('@')[1].lower()
            
            # Check against blocked domains
            if domain in self.blocked_domains:
                return False
            
            # Check against allowed domains (if specified)
            if self.allowed_domains and domain not in self.allowed_domains:
                return False
            
            return True
            
        except (IndexError, AttributeError):
            return False
    
    def _contains_sensitive_content(self, content: str) -> bool:
        """Check if content contains sensitive information."""
        for pattern in self.sensitive_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        return False
    
    def _apply_email_template(self, template: str, params: Dict[str, Any]) -> str:
        """Apply email template with parameter substitution."""
        # Basic template engine - replace ${variable} with values
        template_content = template
        
        for key, value in params.items():
            placeholder = f"${{{key}}}"
            if placeholder in template_content:
                template_content = template_content.replace(placeholder, str(value))
        
        return template_content