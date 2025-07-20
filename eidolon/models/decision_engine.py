"""
Decision Engine for routing between local and cloud AI models

Provides intelligent routing logic to decide when to use local models
vs cloud APIs based on content complexity, cost constraints, and quality requirements.
"""

import os
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import json

from ..utils.logging import get_component_logger, log_performance, log_exceptions
from ..utils.config import get_config


class AnalysisRequest:
    """Represents a request for content analysis."""
    
    def __init__(
        self,
        content_type: str,
        content_path: Optional[str] = None,
        text_content: str = "",
        image_size: Tuple[int, int] = (0, 0),
        metadata: Optional[Dict[str, Any]] = None,
        user_preferences: Optional[Dict[str, Any]] = None
    ):
        self.content_type = content_type  # "image", "text", "mixed"
        self.content_path = content_path
        self.text_content = text_content
        self.image_size = image_size
        self.metadata = metadata or {}
        self.user_preferences = user_preferences or {}
        self.timestamp = datetime.now()


class RoutingDecision:
    """Represents a routing decision for analysis."""
    
    def __init__(
        self,
        use_cloud: bool,
        provider: Optional[str] = None,
        reasoning: str = "",
        confidence: float = 0.0,
        estimated_cost: float = 0.0,
        estimated_quality: float = 0.0
    ):
        self.use_cloud = use_cloud
        self.provider = provider
        self.reasoning = reasoning
        self.confidence = confidence
        self.estimated_cost = estimated_cost
        self.estimated_quality = estimated_quality
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "use_cloud": self.use_cloud,
            "provider": self.provider,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "estimated_cost": self.estimated_cost,
            "estimated_quality": self.estimated_quality,
            "timestamp": self.timestamp.isoformat()
        }


class DecisionEngine:
    """Intelligent routing engine for local vs cloud AI decisions."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_component_logger("models.decision_engine")
        
        # Load configuration
        self.routing_config = self.config.analysis.routing
        self.importance_threshold = self.routing_config.get("importance_threshold", 0.7)
        self.cost_limit_daily = self.routing_config.get("cost_limit_daily", 10.0)
        self.local_first = self.routing_config.get("local_first", True)
        
        # Track daily usage and costs
        self.daily_stats = {
            "cloud_requests": 0,
            "local_requests": 0,
            "total_cost": 0.0,
            "last_reset": datetime.now().date()
        }
        
        # Quality scoring weights
        self.quality_weights = {
            "content_complexity": 0.3,
            "user_importance": 0.25,
            "image_quality": 0.2,
            "text_richness": 0.15,
            "historical_performance": 0.1
        }
        
        # Cost estimates per provider (USD per 1K tokens)
        self.cost_estimates = {
            "google": 0.0015,      # Gemini Flash
            "anthropic": 0.003,    # Claude 3.5 Sonnet
            "openai": 0.005,       # GPT-4o
            "local": 0.0001        # Estimated electricity cost
        }
        
        self.logger.info("Decision engine initialized")
        self.logger.info(f"Config: importance_threshold={self.importance_threshold}, cost_limit=${self.cost_limit_daily}")
    
    def _reset_daily_stats_if_needed(self):
        """Reset daily statistics if it's a new day."""
        today = datetime.now().date()
        if self.daily_stats["last_reset"] != today:
            self.daily_stats = {
                "cloud_requests": 0,
                "local_requests": 0,
                "total_cost": 0.0,
                "last_reset": today
            }
            self.logger.info("Daily statistics reset for new day")
    
    def _calculate_content_complexity(self, request: AnalysisRequest) -> float:
        """Calculate content complexity score (0.0 - 1.0)."""
        complexity = 0.0
        
        # Image complexity
        if request.content_type in ["image", "mixed"]:
            width, height = request.image_size
            total_pixels = width * height
            
            # Higher resolution = higher complexity
            if total_pixels > 2000000:  # > 2MP
                complexity += 0.4
            elif total_pixels > 500000:  # > 0.5MP
                complexity += 0.2
            else:
                complexity += 0.1
        
        # Text complexity
        if request.content_type in ["text", "mixed"]:
            text_length = len(request.text_content)
            word_count = len(request.text_content.split())
            
            # Longer text = higher complexity
            if word_count > 200:
                complexity += 0.3
            elif word_count > 50:
                complexity += 0.2
            else:
                complexity += 0.1
            
            # Technical content indicators
            technical_keywords = [
                "code", "function", "class", "import", "def", "if", "for", "while",
                "error", "exception", "traceback", "debug", "api", "json", "xml"
            ]
            
            text_lower = request.text_content.lower()
            technical_matches = sum(1 for keyword in technical_keywords if keyword in text_lower)
            if technical_matches > 3:
                complexity += 0.2
        
        # Metadata complexity indicators
        if request.metadata:
            if request.metadata.get("has_ui_elements", False):
                complexity += 0.1
            if request.metadata.get("has_multiple_windows", False):
                complexity += 0.1
            if request.metadata.get("content_type") in ["error", "debug", "technical"]:
                complexity += 0.2
        
        return min(complexity, 1.0)
    
    def _calculate_user_importance(self, request: AnalysisRequest) -> float:
        """Calculate user-indicated importance (0.0 - 1.0)."""
        preferences = request.user_preferences
        
        # Explicit user importance setting
        if "importance" in preferences:
            return float(preferences["importance"])
        
        # Infer importance from user behavior patterns
        importance = 0.5  # Default medium importance
        
        # Time-based importance (work hours = higher importance)
        hour = request.timestamp.hour
        if 9 <= hour <= 17:  # Work hours
            importance += 0.2
        elif 18 <= hour <= 22:  # Evening hours
            importance += 0.1
        
        # Content type importance
        if request.metadata.get("content_type") in ["error", "debugging", "code"]:
            importance += 0.3  # Development work is often important
        elif request.metadata.get("content_type") in ["email", "communication"]:
            importance += 0.2  # Communication is moderately important
        
        return min(importance, 1.0)
    
    def _calculate_image_quality_score(self, request: AnalysisRequest) -> float:
        """Calculate image quality score affecting analysis needs (0.0 - 1.0)."""
        if request.content_type not in ["image", "mixed"]:
            return 0.5  # Not applicable
        
        width, height = request.image_size
        total_pixels = width * height
        
        # Higher resolution generally means better quality and more detail
        if total_pixels > 3000000:  # > 3MP
            return 0.9
        elif total_pixels > 2000000:  # > 2MP
            return 0.8
        elif total_pixels > 1000000:  # > 1MP
            return 0.7
        elif total_pixels > 500000:   # > 0.5MP
            return 0.6
        else:
            return 0.4
    
    def _calculate_text_richness(self, request: AnalysisRequest) -> float:
        """Calculate text richness score (0.0 - 1.0)."""
        if request.content_type not in ["text", "mixed"]:
            return 0.5  # Not applicable
        
        text = request.text_content
        if not text:
            return 0.0
        
        word_count = len(text.split())
        char_count = len(text)
        
        richness = 0.0
        
        # Word count richness
        if word_count > 100:
            richness += 0.3
        elif word_count > 25:
            richness += 0.2
        elif word_count > 5:
            richness += 0.1
        
        # Character diversity
        unique_chars = len(set(text.lower()))
        if unique_chars > 20:
            richness += 0.2
        elif unique_chars > 10:
            richness += 0.1
        
        # Sentence structure
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        if sentence_count > 3:
            richness += 0.2
        elif sentence_count > 1:
            richness += 0.1
        
        # Special content indicators
        if any(indicator in text.lower() for indicator in ["error", "warning", "critical", "debug"]):
            richness += 0.2  # Error messages often need careful analysis
        
        return min(richness, 1.0)
    
    def _get_historical_performance(self, provider: str) -> float:
        """Get historical performance score for a provider (0.0 - 1.0)."""
        # This would ideally load from a historical performance database
        # For now, return estimated performance based on provider capabilities
        
        performance_estimates = {
            "local": 0.7,      # Good for basic tasks
            "google": 0.8,     # Good general performance
            "anthropic": 0.9,  # High quality analysis
            "openai": 0.85     # Good balance of quality and speed
        }
        
        return performance_estimates.get(provider, 0.5)
    
    def _calculate_quality_score(self, request: AnalysisRequest, provider: str) -> float:
        """Calculate expected quality score for a provider (0.0 - 1.0)."""
        scores = {
            "content_complexity": self._calculate_content_complexity(request),
            "user_importance": self._calculate_user_importance(request),
            "image_quality": self._calculate_image_quality_score(request),
            "text_richness": self._calculate_text_richness(request),
            "historical_performance": self._get_historical_performance(provider)
        }
        
        # Weighted average
        quality_score = sum(
            scores[factor] * weight
            for factor, weight in self.quality_weights.items()
        )
        
        return quality_score
    
    def _estimate_cost(self, request: AnalysisRequest, provider: str) -> float:
        """Estimate cost for analysis with given provider."""
        if provider == "local":
            return self.cost_estimates["local"]
        
        # Estimate tokens based on content
        estimated_tokens = 0
        
        if request.content_type in ["text", "mixed"]:
            # Text tokens (rough estimate: 1 token ≈ 0.75 words)
            word_count = len(request.text_content.split())
            estimated_tokens += word_count * 1.33
        
        if request.content_type in ["image", "mixed"]:
            # Image tokens (varies by model, rough estimate)
            width, height = request.image_size
            total_pixels = width * height
            # Rough estimate: larger images need more tokens
            estimated_tokens += min(max(total_pixels / 10000, 100), 1000)
        
        # Add prompt overhead
        estimated_tokens += 100  # Base prompt tokens
        
        cost_per_token = self.cost_estimates.get(provider, 0.005) / 1000
        return estimated_tokens * cost_per_token
    
    @log_performance
    def make_routing_decision(
        self,
        request: AnalysisRequest,
        available_providers: List[str]
    ) -> RoutingDecision:
        """
        Make an intelligent routing decision for content analysis.
        
        Args:
            request: Analysis request details
            available_providers: List of available cloud providers
            
        Returns:
            RoutingDecision with routing choice and reasoning
        """
        self._reset_daily_stats_if_needed()
        
        # Calculate quality scores for each option
        local_quality = self._calculate_quality_score(request, "local")
        
        cloud_options = []
        for provider in available_providers:
            quality = self._calculate_quality_score(request, provider)
            cost = self._estimate_cost(request, provider)
            cloud_options.append({
                "provider": provider,
                "quality": quality,
                "cost": cost
            })
        
        # Sort cloud options by quality (descending)
        cloud_options.sort(key=lambda x: x["quality"], reverse=True)
        
        # Decision logic
        decision_factors = []
        
        # 1. Check cost limits
        if self.daily_stats["total_cost"] >= self.cost_limit_daily:
            decision_factors.append("Daily cost limit reached")
            return RoutingDecision(
                use_cloud=False,
                reasoning="Daily cost limit reached, using local analysis",
                confidence=1.0,
                estimated_cost=self.cost_estimates["local"],
                estimated_quality=local_quality
            )
        
        # 2. Check local-first preference
        if self.local_first and local_quality >= self.importance_threshold:
            decision_factors.append("Local model meets quality threshold")
            return RoutingDecision(
                use_cloud=False,
                reasoning="Local analysis sufficient for quality requirements",
                confidence=0.8,
                estimated_cost=self.cost_estimates["local"],
                estimated_quality=local_quality
            )
        
        # 3. If no cloud providers available
        if not cloud_options:
            return RoutingDecision(
                use_cloud=False,
                reasoning="No cloud providers available",
                confidence=0.6,
                estimated_cost=self.cost_estimates["local"],
                estimated_quality=local_quality
            )
        
        # 4. Choose best cloud option if it significantly outperforms local
        best_cloud = cloud_options[0]
        quality_improvement = best_cloud["quality"] - local_quality
        
        # Use cloud if:
        # - Quality improvement is significant (> 0.2)
        # - Content complexity is high
        # - User importance is high
        # - Within cost constraints
        
        complexity = self._calculate_content_complexity(request)
        importance = self._calculate_user_importance(request)
        
        should_use_cloud = (
            quality_improvement > 0.2 or
            complexity > 0.7 or
            importance > 0.8 or
            (quality_improvement > 0.1 and complexity > 0.5)
        )
        
        remaining_budget = self.cost_limit_daily - self.daily_stats["total_cost"]
        if should_use_cloud and best_cloud["cost"] <= remaining_budget:
            decision_factors.extend([
                f"Quality improvement: {quality_improvement:.2f}",
                f"Complexity: {complexity:.2f}",
                f"Importance: {importance:.2f}",
                f"Cost: ${best_cloud['cost']:.4f}"
            ])
            
            return RoutingDecision(
                use_cloud=True,
                provider=best_cloud["provider"],
                reasoning=f"Cloud analysis recommended: {', '.join(decision_factors)}",
                confidence=0.9,
                estimated_cost=best_cloud["cost"],
                estimated_quality=best_cloud["quality"]
            )
        else:
            # Use local analysis
            if not should_use_cloud:
                reason = "Local analysis sufficient"
            else:
                reason = f"Cloud analysis desired but cost ${best_cloud['cost']:.4f} exceeds remaining budget ${remaining_budget:.4f}"
            
            return RoutingDecision(
                use_cloud=False,
                reasoning=reason,
                confidence=0.7,
                estimated_cost=self.cost_estimates["local"],
                estimated_quality=local_quality
            )
    
    def record_analysis_result(
        self,
        decision: RoutingDecision,
        actual_cost: float = 0.0,
        success: bool = True,
        quality_feedback: Optional[float] = None
    ):
        """Record the result of an analysis for performance tracking."""
        self._reset_daily_stats_if_needed()
        
        if decision.use_cloud:
            self.daily_stats["cloud_requests"] += 1
            self.daily_stats["total_cost"] += actual_cost
        else:
            self.daily_stats["local_requests"] += 1
        
        # Log performance data
        result_data = {
            "decision": decision.to_dict(),
            "actual_cost": actual_cost,
            "success": success,
            "quality_feedback": quality_feedback
        }
        
        self.logger.debug(f"Analysis result recorded: {result_data}")
    
    def get_decision_stats(self) -> Dict[str, Any]:
        """Get decision engine statistics."""
        self._reset_daily_stats_if_needed()
        
        total_requests = self.daily_stats["cloud_requests"] + self.daily_stats["local_requests"]
        
        return {
            "daily_stats": self.daily_stats.copy(),
            "total_requests_today": total_requests,
            "cloud_usage_percent": (
                (self.daily_stats["cloud_requests"] / max(total_requests, 1)) * 100
            ),
            "cost_utilization_percent": (
                (self.daily_stats["total_cost"] / self.cost_limit_daily) * 100
            ),
            "config": {
                "importance_threshold": self.importance_threshold,
                "cost_limit_daily": self.cost_limit_daily,
                "local_first": self.local_first
            }
        }
    
    def update_configuration(self, **kwargs):
        """Update decision engine configuration."""
        if "importance_threshold" in kwargs:
            self.importance_threshold = float(kwargs["importance_threshold"])
        
        if "cost_limit_daily" in kwargs:
            self.cost_limit_daily = float(kwargs["cost_limit_daily"])
        
        if "local_first" in kwargs:
            self.local_first = bool(kwargs["local_first"])
        
        self.logger.info(f"Configuration updated: threshold={self.importance_threshold}, cost_limit=${self.cost_limit_daily}")
    
    def handle_cloud_failure(
        self,
        request: AnalysisRequest,
        failed_provider: str,
        error: Exception,
        available_providers: List[str]
    ) -> RoutingDecision:
        """
        Handle cloud API failure with intelligent fallback.
        
        Args:
            request: Original analysis request
            failed_provider: Provider that failed
            error: Exception that occurred
            available_providers: Other providers to try
            
        Returns:
            RoutingDecision for fallback approach
        """
        self.logger.warning(f"Cloud provider {failed_provider} failed: {error}")
        
        # Remove failed provider from available options
        remaining_providers = [p for p in available_providers if p != failed_provider]
        
        # Try another cloud provider first
        if remaining_providers:
            best_fallback = None
            best_quality = 0
            
            for provider in remaining_providers:
                quality = self._calculate_quality_score(request, provider)
                cost = self._estimate_cost(request, provider)
                
                # Check if we can afford this provider
                if self.daily_stats["total_cost"] + cost <= self.cost_limit_daily:
                    if quality > best_quality:
                        best_quality = quality
                        best_fallback = provider
            
            if best_fallback:
                return RoutingDecision(
                    use_cloud=True,
                    provider=best_fallback,
                    reasoning=f"Fallback to {best_fallback} after {failed_provider} failure",
                    confidence=0.7,
                    estimated_cost=self._estimate_cost(request, best_fallback),
                    estimated_quality=best_quality
                )
        
        # Fallback to local analysis
        local_quality = self._calculate_quality_score(request, "local")
        
        return RoutingDecision(
            use_cloud=False,
            reasoning=f"Fallback to local analysis after cloud failures",
            confidence=0.8,
            estimated_cost=self.cost_estimates.get("local", 0.0),
            estimated_quality=local_quality
        )
    
    def get_provider_priority_list(self, request: AnalysisRequest) -> List[str]:
        """
        Get prioritized list of providers for a request.
        
        Args:
            request: Analysis request
            
        Returns:
            List of providers in priority order
        """
        # Get all available providers (this should be passed from cloud API manager)
        available_providers = ["gemini", "claude", "openai"]
        
        provider_scores = []
        for provider in available_providers:
            quality = self._calculate_quality_score(request, provider)
            cost = self._estimate_cost(request, provider)
            
            # Composite score (quality / cost ratio)
            score = quality / max(cost, 0.01)  # Avoid division by zero
            
            provider_scores.append({
                "provider": provider,
                "score": score,
                "quality": quality,
                "cost": cost
            })
        
        # Sort by score (descending)
        provider_scores.sort(key=lambda x: x["score"], reverse=True)
        
        return [p["provider"] for p in provider_scores]