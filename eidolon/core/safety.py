"""
Safety Mechanisms for Eidolon Autonomous Agent

Provides comprehensive safety controls, risk assessment, action validation,
and user consent mechanisms for autonomous operations.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from ..utils.logging import get_component_logger, log_performance, log_exceptions
from ..utils.config import get_config

# Initialize logger
logger = get_component_logger("safety")


class RiskLevel(str, Enum):
    """Risk levels for actions and operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionCategory(str, Enum):
    """Categories of actions for risk assessment."""
    READ_ONLY = "read_only"
    WRITE_FILE = "write_file"
    DELETE_FILE = "delete_file"
    SYSTEM_COMMAND = "system_command"
    NETWORK_REQUEST = "network_request"
    USER_INTERFACE = "user_interface"
    EMAIL_SEND = "email_send"
    DATA_EXPORT = "data_export"
    AUTOMATION = "automation"
    UNKNOWN = "unknown"


class ConsentType(str, Enum):
    """Types of user consent required."""
    EXPLICIT = "explicit"  # User must explicitly approve
    IMPLIED = "implied"    # User has pre-approved this type
    AUTOMATIC = "automatic"  # No approval needed
    BLOCKED = "blocked"    # Action is not allowed


@dataclass
class ActionApproval:
    """User approval record for an action."""
    approved: bool
    approved_by: str
    approved_at: datetime
    conditions: List[str] = field(default_factory=list)
    expires_at: Optional[datetime] = None
    notes: Optional[str] = None


@dataclass
class RiskAssessment:
    """Risk assessment result for an action."""
    risk_level: RiskLevel
    confidence: float
    factors: List[str] = field(default_factory=list)
    mitigations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class SafetyRule:
    """A safety rule for action validation."""
    id: str
    name: str
    description: str
    pattern: str  # Regex pattern to match against actions
    risk_level: RiskLevel
    action_category: ActionCategory
    consent_type: ConsentType
    enabled: bool = True
    conditions: List[str] = field(default_factory=list)


@dataclass
class ActionRecord:
    """Record of an executed action for audit purposes."""
    id: str
    timestamp: datetime
    action_type: str
    parameters: Dict[str, Any]
    risk_assessment: RiskAssessment
    approval: Optional[ActionApproval]
    result: Optional[Dict[str, Any]]
    side_effects: List[str] = field(default_factory=list)
    user_id: str = "system"


class SafetyManager:
    """
    Central safety manager for autonomous operations.
    
    Provides risk assessment, action validation, user consent management,
    and audit logging for all autonomous actions.
    """
    
    def __init__(self, config_override: Optional[Dict] = None):
        """Initialize the safety manager."""
        self.config = get_config()
        if config_override:
            self.config.update(config_override)
        
        # Safety rules
        self.safety_rules: Dict[str, SafetyRule] = {}
        self._load_default_rules()
        
        # Action history for audit
        self.action_history: List[ActionRecord] = []
        self.max_history_size = 10000
        
        # User preferences and approvals
        self.user_approvals: Dict[str, ActionApproval] = {}
        self.user_preferences: Dict[str, Any] = {}
        
        # Risk assessment cache
        self.risk_cache: Dict[str, RiskAssessment] = {}
        self.cache_ttl = timedelta(minutes=30)
        
        # Blocked patterns (never allow)
        self.blocked_patterns = [
            r"rm\s+-rf\s+/",  # Dangerous file deletions
            r"sudo\s+.*passwd",  # Password changes
            r"chmod\s+777",  # Insecure permissions
            r"curl.*\|\s*bash",  # Pipe to shell execution
            r"dd\s+if=.*of=/dev/",  # Disk operations
        ]
        
        # Sensitive data patterns
        self.sensitive_patterns = [
            r"password[s]?\s*[:=]\s*[\w\d]+",
            r"api[_-]?key[s]?\s*[:=]\s*[\w\d-]+",
            r"secret[s]?\s*[:=]\s*[\w\d]+",
            r"token[s]?\s*[:=]\s*[\w\d-]+",
            r"\d{3}-\d{2}-\d{4}",  # SSN
            r"\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}",  # Credit card
        ]
    
    async def assess_risk(self, actions: List[Dict[str, Any]]) -> RiskLevel:
        """
        Assess the overall risk level for a list of actions.
        
        Args:
            actions: List of action dictionaries
            
        Returns:
            Overall risk level
        """
        try:
            max_risk = RiskLevel.LOW
            all_assessments = []
            
            for action in actions:
                assessment = await self._assess_single_action(action)
                all_assessments.append(assessment)
                
                # Take the highest risk level
                risk_levels = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
                if risk_levels.index(assessment.risk_level) > risk_levels.index(max_risk):
                    max_risk = assessment.risk_level
            
            logger.debug(f"Risk assessment complete: {max_risk} (assessed {len(actions)} actions)")
            return max_risk
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return RiskLevel.CRITICAL  # Fail safe
    
    async def validate_action(
        self, 
        action: Dict[str, Any], 
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """
        Validate an action against safety rules and user preferences.
        
        Args:
            action: Action to validate
            user_id: User requesting the action
            
        Returns:
            Validation result with approval status
        """
        try:
            # Assess risk
            risk_assessment = await self._assess_single_action(action)
            
            # Check if action is blocked
            if self._is_action_blocked(action):
                return {
                    "approved": False,
                    "reason": "Action is blocked by safety rules",
                    "risk_level": RiskLevel.CRITICAL,
                    "consent_required": ConsentType.BLOCKED
                }
            
            # Find matching safety rule
            matching_rule = self._find_matching_rule(action)
            
            if matching_rule:
                consent_type = matching_rule.consent_type
                
                # Check if user has pre-approved this type
                if consent_type == ConsentType.AUTOMATIC:
                    return {
                        "approved": True,
                        "reason": "Automatically approved",
                        "risk_level": risk_assessment.risk_level,
                        "consent_required": ConsentType.AUTOMATIC
                    }
                
                # Check for existing approval
                approval_key = f"{user_id}:{matching_rule.id}"
                if approval_key in self.user_approvals:
                    approval = self.user_approvals[approval_key]
                    if approval.approved and (not approval.expires_at or approval.expires_at > datetime.now()):
                        return {
                            "approved": True,
                            "reason": "Previously approved",
                            "risk_level": risk_assessment.risk_level,
                            "consent_required": ConsentType.IMPLIED,
                            "approval": approval
                        }
                
                return {
                    "approved": False,
                    "reason": "User consent required",
                    "risk_level": risk_assessment.risk_level,
                    "consent_required": consent_type,
                    "rule": matching_rule,
                    "risk_assessment": risk_assessment
                }
            
            # Default: require explicit consent for unknown actions
            return {
                "approved": False,
                "reason": "Unknown action type requires approval",
                "risk_level": risk_assessment.risk_level,
                "consent_required": ConsentType.EXPLICIT,
                "risk_assessment": risk_assessment
            }
            
        except Exception as e:
            logger.error(f"Action validation failed: {e}")
            return {
                "approved": False,
                "reason": f"Validation error: {str(e)}",
                "risk_level": RiskLevel.CRITICAL,
                "consent_required": ConsentType.BLOCKED
            }
    
    async def request_approval(
        self,
        action: Dict[str, Any],
        risk_assessment: RiskAssessment,
        user_id: str = "system",
        timeout_seconds: int = 300
    ) -> ActionApproval:
        """
        Request user approval for an action.
        
        Args:
            action: Action requiring approval
            risk_assessment: Risk assessment for the action
            user_id: User to request approval from
            timeout_seconds: How long to wait for approval
            
        Returns:
            Action approval result
        """
        logger.info(f"Requesting approval for action: {action.get('type', 'unknown')}")
        
        # For now, this is a placeholder that would integrate with UI
        # In a real implementation, this would:
        # 1. Send notification to user
        # 2. Display action details and risk assessment
        # 3. Wait for user response
        # 4. Return approval status
        
        # Simulate approval request (would be replaced with actual UI integration)
        return ActionApproval(
            approved=False,  # Default to not approved
            approved_by=user_id,
            approved_at=datetime.now(),
            notes="Approval system not implemented - would show UI prompt"
        )
    
    async def record_action(
        self,
        action: Dict[str, Any],
        risk_assessment: RiskAssessment,
        approval: Optional[ActionApproval] = None,
        result: Optional[Dict[str, Any]] = None,
        user_id: str = "system"
    ) -> str:
        """
        Record an action in the audit log.
        
        Args:
            action: Action that was executed
            risk_assessment: Risk assessment for the action
            approval: User approval if required
            result: Execution result
            user_id: User who initiated the action
            
        Returns:
            Action record ID
        """
        record_id = f"action_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        record = ActionRecord(
            id=record_id,
            timestamp=datetime.now(),
            action_type=action.get("type", "unknown"),
            parameters=action.get("params", {}),
            risk_assessment=risk_assessment,
            approval=approval,
            result=result,
            user_id=user_id
        )
        
        self.action_history.append(record)
        
        # Limit history size
        if len(self.action_history) > self.max_history_size:
            self.action_history = self.action_history[-self.max_history_size:]
        
        logger.info(f"Action recorded: {record_id}")
        return record_id
    
    def approve_action_type(
        self,
        rule_id: str,
        user_id: str,
        duration_hours: Optional[int] = None,
        conditions: Optional[List[str]] = None
    ) -> bool:
        """
        Pre-approve an action type for a user.
        
        Args:
            rule_id: ID of the safety rule to approve
            user_id: User granting approval
            duration_hours: How long approval lasts (None = permanent)
            conditions: Additional conditions for approval
            
        Returns:
            Success status
        """
        try:
            expires_at = None
            if duration_hours:
                expires_at = datetime.now() + timedelta(hours=duration_hours)
            
            approval = ActionApproval(
                approved=True,
                approved_by=user_id,
                approved_at=datetime.now(),
                conditions=conditions or [],
                expires_at=expires_at
            )
            
            approval_key = f"{user_id}:{rule_id}"
            self.user_approvals[approval_key] = approval
            
            logger.info(f"Action type approved: {rule_id} by {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to approve action type: {e}")
            return False
    
    def revoke_approval(self, rule_id: str, user_id: str) -> bool:
        """Revoke a previously granted approval."""
        try:
            approval_key = f"{user_id}:{rule_id}"
            if approval_key in self.user_approvals:
                del self.user_approvals[approval_key]
                logger.info(f"Approval revoked: {rule_id} by {user_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to revoke approval: {e}")
            return False
    
    def get_audit_log(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        risk_level: Optional[RiskLevel] = None
    ) -> List[ActionRecord]:
        """Get filtered audit log entries."""
        filtered_records = []
        
        for record in self.action_history:
            # Apply filters
            if start_time and record.timestamp < start_time:
                continue
            if end_time and record.timestamp > end_time:
                continue
            if user_id and record.user_id != user_id:
                continue
            if risk_level and record.risk_assessment.risk_level != risk_level:
                continue
            
            filtered_records.append(record)
        
        return filtered_records
    
    def get_safety_rules(self) -> List[SafetyRule]:
        """Get all safety rules."""
        return list(self.safety_rules.values())
    
    def add_safety_rule(self, rule: SafetyRule) -> bool:
        """Add a new safety rule."""
        try:
            self.safety_rules[rule.id] = rule
            logger.info(f"Safety rule added: {rule.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add safety rule: {e}")
            return False
    
    def update_safety_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing safety rule."""
        try:
            if rule_id not in self.safety_rules:
                return False
            
            rule = self.safety_rules[rule_id]
            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
            
            logger.info(f"Safety rule updated: {rule.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to update safety rule: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get safety manager statistics."""
        total_actions = len(self.action_history)
        recent_actions = [r for r in self.action_history 
                         if r.timestamp > datetime.now() - timedelta(hours=24)]
        
        risk_counts = {}
        for record in recent_actions:
            risk_level = record.risk_assessment.risk_level
            risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
        
        return {
            "total_actions": total_actions,
            "recent_actions_24h": len(recent_actions),
            "safety_rules": len(self.safety_rules),
            "active_approvals": len(self.user_approvals),
            "risk_distribution": risk_counts,
            "blocked_patterns": len(self.blocked_patterns)
        }
    
    async def _assess_single_action(self, action: Dict[str, Any]) -> RiskAssessment:
        """Assess risk for a single action."""
        action_str = json.dumps(action, sort_keys=True)
        
        # Check cache
        if action_str in self.risk_cache:
            cached = self.risk_cache[action_str]
            # Simple TTL check (in production, would be more sophisticated)
            return cached
        
        risk_level = RiskLevel.LOW
        factors = []
        mitigations = []
        recommendations = []
        confidence = 0.8
        
        action_type = action.get("type", "").lower()
        action_params = action.get("params", {})
        
        # Check for blocked patterns
        if self._is_action_blocked(action):
            risk_level = RiskLevel.CRITICAL
            factors.append("Action matches blocked pattern")
            confidence = 1.0
        
        # Check for sensitive data
        elif self._contains_sensitive_data(action):
            risk_level = RiskLevel.HIGH
            factors.append("Action contains sensitive data")
            mitigations.append("Redact sensitive information")
        
        # Assess by action type
        elif any(keyword in action_type for keyword in ["delete", "remove", "rm"]):
            risk_level = RiskLevel.HIGH
            factors.append("Destructive action")
            mitigations.append("Create backup before deletion")
            recommendations.append("Confirm file paths before deletion")
        
        elif any(keyword in action_type for keyword in ["system", "command", "execute"]):
            risk_level = RiskLevel.MEDIUM
            factors.append("System command execution")
            mitigations.append("Validate command syntax")
            recommendations.append("Run in sandbox environment")
        
        elif any(keyword in action_type for keyword in ["email", "send", "message"]):
            risk_level = RiskLevel.MEDIUM
            factors.append("Communication action")
            mitigations.append("Review recipients and content")
        
        elif any(keyword in action_type for keyword in ["write", "create", "modify"]):
            risk_level = RiskLevel.LOW
            factors.append("File modification")
            mitigations.append("Create backup before modification")
        
        # Cache result
        assessment = RiskAssessment(
            risk_level=risk_level,
            confidence=confidence,
            factors=factors,
            mitigations=mitigations,
            recommendations=recommendations
        )
        
        self.risk_cache[action_str] = assessment
        
        return assessment
    
    def _is_action_blocked(self, action: Dict[str, Any]) -> bool:
        """Check if action matches any blocked patterns."""
        action_str = json.dumps(action).lower()
        
        for pattern in self.blocked_patterns:
            if re.search(pattern, action_str, re.IGNORECASE):
                return True
        
        return False
    
    def _contains_sensitive_data(self, action: Dict[str, Any]) -> bool:
        """Check if action contains sensitive data patterns."""
        action_str = json.dumps(action)
        
        for pattern in self.sensitive_patterns:
            if re.search(pattern, action_str, re.IGNORECASE):
                return True
        
        return False
    
    def _find_matching_rule(self, action: Dict[str, Any]) -> Optional[SafetyRule]:
        """Find the first safety rule that matches an action."""
        action_str = json.dumps(action)
        
        for rule in self.safety_rules.values():
            if not rule.enabled:
                continue
            
            if re.search(rule.pattern, action_str, re.IGNORECASE):
                return rule
        
        return None
    
    def _load_default_rules(self) -> None:
        """Load default safety rules."""
        default_rules = [
            SafetyRule(
                id="file_deletion",
                name="File Deletion",
                description="Actions that delete files or directories",
                pattern=r"(delete|remove|rm).*file",
                risk_level=RiskLevel.HIGH,
                action_category=ActionCategory.DELETE_FILE,
                consent_type=ConsentType.EXPLICIT
            ),
            SafetyRule(
                id="system_command",
                name="System Command",
                description="Execution of system commands",
                pattern=r"(system|command|execute|run).*",
                risk_level=RiskLevel.MEDIUM,
                action_category=ActionCategory.SYSTEM_COMMAND,
                consent_type=ConsentType.EXPLICIT
            ),
            SafetyRule(
                id="email_send",
                name="Email Sending",
                description="Sending emails or messages",
                pattern=r"(email|send|message).*",
                risk_level=RiskLevel.MEDIUM,
                action_category=ActionCategory.EMAIL_SEND,
                consent_type=ConsentType.EXPLICIT
            ),
            SafetyRule(
                id="file_write",
                name="File Writing",
                description="Creating or modifying files",
                pattern=r"(write|create|modify).*file",
                risk_level=RiskLevel.LOW,
                action_category=ActionCategory.WRITE_FILE,
                consent_type=ConsentType.IMPLIED
            ),
            SafetyRule(
                id="read_only",
                name="Read-Only Operations",
                description="Safe read-only operations",
                pattern=r"(read|view|analyze|search).*",
                risk_level=RiskLevel.LOW,
                action_category=ActionCategory.READ_ONLY,
                consent_type=ConsentType.AUTOMATIC
            ),
            SafetyRule(
                id="network_request",
                name="Network Requests",
                description="Making network requests",
                pattern=r"(http|request|download|upload).*",
                risk_level=RiskLevel.MEDIUM,
                action_category=ActionCategory.NETWORK_REQUEST,
                consent_type=ConsentType.EXPLICIT
            )
        ]
        
        for rule in default_rules:
            self.safety_rules[rule.id] = rule
        
        logger.info(f"Loaded {len(default_rules)} default safety rules")


# Convenience functions for common operations
async def quick_risk_assessment(actions: List[Dict[str, Any]]) -> RiskLevel:
    """Quick risk assessment for a list of actions."""
    safety_manager = SafetyManager()
    return await safety_manager.assess_risk(actions)


async def validate_single_action(action: Dict[str, Any], user_id: str = "system") -> Dict[str, Any]:
    """Validate a single action."""
    safety_manager = SafetyManager()
    return await safety_manager.validate_action(action, user_id)


def create_approval(user_id: str, notes: str = "") -> ActionApproval:
    """Create an approval record."""
    return ActionApproval(
        approved=True,
        approved_by=user_id,
        approved_at=datetime.now(),
        notes=notes
    )