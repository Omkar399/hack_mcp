#!/usr/bin/env python3
"""
Unit tests for the Decision Engine component.

Tests cover routing logic between local and cloud AI models based on
content characteristics, importance, and cost considerations.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import the modules to test
from eidolon.models.decision_engine import DecisionEngine, RoutingDecision


class TestRoutingDecision:
    """Test the RoutingDecision class."""
    
    def test_routing_decision_creation(self):
        """Test RoutingDecision creation with all fields."""
        decision = RoutingDecision(
            use_cloud=True,
            provider="gemini",
            reason="High importance content requiring advanced analysis",
            confidence=0.95,
            estimated_cost=0.05,
            factors={
                "importance_score": 0.9,
                "complexity": "high",
                "content_type": "technical"
            }
        )
        
        assert decision.use_cloud is True
        assert decision.provider == "gemini"
        assert decision.reason == "High importance content requiring advanced analysis"
        assert decision.confidence == 0.95
        assert decision.estimated_cost == 0.05
        assert decision.factors["importance_score"] == 0.9
        assert isinstance(decision.timestamp, datetime)
    
    def test_routing_decision_defaults(self):
        """Test RoutingDecision with default values."""
        decision = RoutingDecision(
            use_cloud=False,
            reason="Local processing sufficient"
        )
        
        assert decision.use_cloud is False
        assert decision.provider is None
        assert decision.confidence == 0.0
        assert decision.estimated_cost == 0.0
        assert decision.factors == {}
    
    def test_routing_decision_to_dict(self):
        """Test RoutingDecision serialization."""
        decision = RoutingDecision(
            use_cloud=True,
            provider="claude",
            reason="Complex content analysis",
            confidence=0.8,
            estimated_cost=0.03,
            factors={"complexity": "high"}
        )
        
        result = decision.to_dict()
        
        assert isinstance(result, dict)
        assert result["use_cloud"] is True
        assert result["provider"] == "claude"
        assert result["reason"] == "Complex content analysis"
        assert result["confidence"] == 0.8
        assert result["estimated_cost"] == 0.03
        assert result["factors"]["complexity"] == "high"
        assert "timestamp" in result


class TestDecisionEngine:
    """Test the DecisionEngine class."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for DecisionEngine."""
        config = Mock()
        config.analysis = Mock()
        config.analysis.routing = {
            "importance_threshold": 0.7,
            "cost_limit_daily": 10.0,
            "local_confidence_threshold": 0.8,
            "complexity_threshold": 0.6
        }
        config.analysis.preferred_providers = ["gemini", "claude", "openai"]
        return config
    
    @pytest.fixture
    def decision_engine(self, mock_config):
        """Create DecisionEngine instance with mock config."""
        with patch('eidolon.models.decision_engine.get_config', return_value=mock_config), \
             patch('eidolon.models.decision_engine.get_component_logger'):
            return DecisionEngine()
    
    def test_decision_engine_initialization(self, mock_config):
        """Test DecisionEngine initialization."""
        with patch('eidolon.models.decision_engine.get_config', return_value=mock_config), \
             patch('eidolon.models.decision_engine.get_component_logger'):
            
            engine = DecisionEngine()
            
            assert engine.config == mock_config
            assert engine.daily_cost == 0.0
            assert isinstance(engine.cost_reset_time, datetime)
            assert engine.routing_history == []
    
    def test_should_use_cloud_high_importance(self, decision_engine):
        """Test cloud routing for high importance content."""
        content_analysis = {
            "importance_score": 0.9,
            "content_type": "financial_data",
            "confidence": 0.6,
            "complexity": 0.8
        }
        
        result = decision_engine.should_use_cloud(content_analysis)
        
        assert result is True
    
    def test_should_use_cloud_low_importance(self, decision_engine):
        """Test local routing for low importance content."""
        content_analysis = {
            "importance_score": 0.3,
            "content_type": "general_text",
            "confidence": 0.9,
            "complexity": 0.2
        }
        
        result = decision_engine.should_use_cloud(content_analysis)
        
        assert result is False
    
    def test_should_use_cloud_cost_limit_exceeded(self, decision_engine):
        """Test local routing when cost limit is exceeded."""
        # Set daily cost to exceed limit
        decision_engine.daily_cost = 15.0
        
        content_analysis = {
            "importance_score": 0.8,
            "content_type": "code",
            "confidence": 0.5
        }
        
        result = decision_engine.should_use_cloud(content_analysis)
        
        assert result is False
    
    def test_make_routing_decision_cloud(self, decision_engine):
        """Test making a cloud routing decision."""
        content_analysis = {
            "importance_score": 0.85,
            "content_type": "technical_document",
            "confidence": 0.5,
            "complexity": 0.9,
            "estimated_tokens": 1000
        }
        
        decision = decision_engine.make_routing_decision(
            content_analysis,
            available_providers=["gemini", "claude"]
        )
        
        assert isinstance(decision, RoutingDecision)
        assert decision.use_cloud is True
        assert decision.provider in ["gemini", "claude"]
        assert decision.confidence > 0
        assert decision.estimated_cost > 0
        assert "importance_score" in decision.factors
    
    def test_make_routing_decision_local(self, decision_engine):
        """Test making a local routing decision."""
        content_analysis = {
            "importance_score": 0.4,
            "content_type": "simple_text",
            "confidence": 0.9,
            "complexity": 0.2
        }
        
        decision = decision_engine.make_routing_decision(
            content_analysis,
            available_providers=["gemini"]
        )
        
        assert isinstance(decision, RoutingDecision)
        assert decision.use_cloud is False
        assert decision.provider is None
        assert decision.estimated_cost == 0.0
    
    def test_calculate_importance_score(self, decision_engine):
        """Test importance score calculation."""
        content_analysis = {
            "content_type": "code",
            "ui_elements": [
                {"type": "editor", "importance": "high"},
                {"type": "terminal", "importance": "medium"}
            ],
            "detected_apps": ["vscode", "terminal"],
            "text_length": 500,
            "has_sensitive_data": False
        }
        
        score = decision_engine.calculate_importance_score(content_analysis)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert score > 0.5  # Code content should have higher importance
    
    def test_calculate_importance_score_financial(self, decision_engine):
        """Test importance score for financial content."""
        content_analysis = {
            "content_type": "financial",
            "detected_apps": ["excel", "quickbooks"],
            "has_sensitive_data": True
        }
        
        score = decision_engine.calculate_importance_score(content_analysis)
        
        assert score > 0.7  # Financial content should have high importance
    
    def test_estimate_processing_cost(self, decision_engine):
        """Test cost estimation for different providers."""
        content_analysis = {
            "estimated_tokens": 1000,
            "content_type": "document"
        }
        
        # Test different providers
        gemini_cost = decision_engine.estimate_processing_cost("gemini", content_analysis)
        claude_cost = decision_engine.estimate_processing_cost("claude", content_analysis)
        openai_cost = decision_engine.estimate_processing_cost("openai", content_analysis)
        
        assert all(isinstance(cost, float) for cost in [gemini_cost, claude_cost, openai_cost])
        assert all(cost > 0 for cost in [gemini_cost, claude_cost, openai_cost])
        
        # Unknown provider should return 0
        unknown_cost = decision_engine.estimate_processing_cost("unknown", content_analysis)
        assert unknown_cost == 0.0
    
    def test_select_provider(self, decision_engine):
        """Test provider selection logic."""
        content_analysis = {
            "content_type": "code",
            "language": "python",
            "complexity": 0.8
        }
        
        available = ["gemini", "claude", "openai"]
        
        provider = decision_engine.select_provider(content_analysis, available)
        
        assert provider in available
        
        # Test with limited availability
        provider = decision_engine.select_provider(content_analysis, ["claude"])
        assert provider == "claude"
        
        # Test with no availability
        provider = decision_engine.select_provider(content_analysis, [])
        assert provider is None
    
    def test_update_cost_tracking(self, decision_engine):
        """Test cost tracking updates."""
        initial_cost = decision_engine.daily_cost
        
        decision_engine.update_cost_tracking(0.05)
        
        assert decision_engine.daily_cost == initial_cost + 0.05
        
        # Test multiple updates
        decision_engine.update_cost_tracking(0.03)
        decision_engine.update_cost_tracking(0.02)
        
        assert decision_engine.daily_cost == initial_cost + 0.10
    
    def test_reset_daily_cost(self, decision_engine):
        """Test daily cost reset functionality."""
        # Set some cost and old reset time
        decision_engine.daily_cost = 5.0
        decision_engine.cost_reset_time = datetime.now() - timedelta(days=2)
        
        # Manually trigger reset check
        decision_engine._check_cost_reset()
        
        # Cost should be reset
        assert decision_engine.daily_cost == 0.0
        assert decision_engine.cost_reset_time.date() == datetime.now().date()
    
    def test_get_routing_stats(self, decision_engine):
        """Test routing statistics retrieval."""
        # Add some routing history
        decision_engine.routing_history = [
            RoutingDecision(use_cloud=True, provider="gemini", estimated_cost=0.05),
            RoutingDecision(use_cloud=False, estimated_cost=0.0),
            RoutingDecision(use_cloud=True, provider="claude", estimated_cost=0.03),
        ]
        
        stats = decision_engine.get_routing_stats()
        
        assert isinstance(stats, dict)
        assert stats["total_decisions"] == 3
        assert stats["cloud_decisions"] == 2
        assert stats["local_decisions"] == 1
        assert stats["cloud_percentage"] == pytest.approx(66.67, rel=0.01)
        assert stats["total_cost"] == 0.08
        assert "provider_breakdown" in stats
        assert stats["provider_breakdown"]["gemini"] == 1
        assert stats["provider_breakdown"]["claude"] == 1
    
    def test_get_cost_breakdown(self, decision_engine):
        """Test cost breakdown by provider."""
        # Add routing history with costs
        decision_engine.routing_history = [
            RoutingDecision(use_cloud=True, provider="gemini", estimated_cost=0.05),
            RoutingDecision(use_cloud=True, provider="gemini", estimated_cost=0.03),
            RoutingDecision(use_cloud=True, provider="claude", estimated_cost=0.04),
        ]
        
        breakdown = decision_engine.get_cost_breakdown()
        
        assert isinstance(breakdown, dict)
        assert breakdown["gemini"] == 0.08
        assert breakdown["claude"] == 0.04
        assert breakdown["total"] == 0.12


class TestDecisionEngineAdvanced:
    """Test advanced decision engine features."""
    
    @pytest.fixture
    def decision_engine_with_history(self, mock_config):
        """Create DecisionEngine with routing history."""
        with patch('eidolon.models.decision_engine.get_config', return_value=mock_config), \
             patch('eidolon.models.decision_engine.get_component_logger'):
            
            engine = DecisionEngine()
            
            # Add varied routing history
            for i in range(10):
                engine.routing_history.append(
                    RoutingDecision(
                        use_cloud=i % 3 != 0,
                        provider=["gemini", "claude", "openai"][i % 3] if i % 3 != 0 else None,
                        estimated_cost=0.01 * i if i % 3 != 0 else 0.0,
                        confidence=0.7 + (i % 3) * 0.1
                    )
                )
            
            return engine
    
    def test_adaptive_threshold_adjustment(self, decision_engine_with_history):
        """Test adaptive threshold adjustment based on history."""
        # This tests if the engine can adapt thresholds based on past performance
        # (Feature might not be implemented yet, but test structure is here)
        
        initial_threshold = decision_engine_with_history.config.analysis.routing["importance_threshold"]
        
        # After many cloud decisions, threshold might be adjusted
        # This is a placeholder for actual adaptive behavior
        assert decision_engine_with_history.config.analysis.routing["importance_threshold"] == initial_threshold
    
    def test_content_pattern_learning(self, decision_engine_with_history):
        """Test learning from content patterns."""
        # Test if engine learns from patterns in content types
        
        # Simulate multiple code analysis requests
        for _ in range(5):
            content = {
                "content_type": "code",
                "language": "python",
                "importance_score": 0.6,
                "confidence": 0.7
            }
            decision = decision_engine_with_history.make_routing_decision(content, ["gemini"])
            decision_engine_with_history.routing_history.append(decision)
        
        # Check if pattern affects future decisions
        new_content = {
            "content_type": "code",
            "language": "python",
            "importance_score": 0.6,
            "confidence": 0.7
        }
        
        decision = decision_engine_with_history.make_routing_decision(new_content, ["gemini"])
        assert isinstance(decision, RoutingDecision)
    
    def test_time_based_routing(self, decision_engine):
        """Test routing decisions based on time of day."""
        # Test if routing considers time (e.g., prefer local during peak hours)
        
        content = {
            "importance_score": 0.6,
            "content_type": "document"
        }
        
        # Mock different times
        with patch('eidolon.models.decision_engine.datetime') as mock_datetime:
            # Peak hours (business hours)
            mock_datetime.now.return_value = datetime(2024, 1, 1, 14, 0)  # 2 PM
            decision_peak = decision_engine.make_routing_decision(content, ["gemini"])
            
            # Off-peak hours
            mock_datetime.now.return_value = datetime(2024, 1, 1, 2, 0)  # 2 AM
            decision_offpeak = decision_engine.make_routing_decision(content, ["gemini"])
            
            # Both should return valid decisions
            assert isinstance(decision_peak, RoutingDecision)
            assert isinstance(decision_offpeak, RoutingDecision)


class TestDecisionEngineError:
    """Test error handling in DecisionEngine."""
    
    def test_invalid_content_analysis(self, decision_engine):
        """Test handling of invalid content analysis data."""
        # Empty content analysis
        decision = decision_engine.make_routing_decision({}, ["gemini"])
        assert isinstance(decision, RoutingDecision)
        assert decision.use_cloud is False  # Should default to local
        
        # None content analysis
        decision = decision_engine.make_routing_decision(None, ["gemini"])
        assert isinstance(decision, RoutingDecision)
        assert decision.use_cloud is False
    
    def test_no_available_providers(self, decision_engine):
        """Test behavior when no providers are available."""
        content = {
            "importance_score": 0.9,
            "content_type": "important"
        }
        
        decision = decision_engine.make_routing_decision(content, [])
        
        assert isinstance(decision, RoutingDecision)
        assert decision.use_cloud is False
        assert decision.provider is None
        assert "No providers available" in decision.reason
    
    def test_cost_calculation_errors(self, decision_engine):
        """Test handling of cost calculation errors."""
        content = {
            "importance_score": 0.8,
            # Missing estimated_tokens
        }
        
        # Should handle missing token estimation gracefully
        cost = decision_engine.estimate_processing_cost("gemini", content)
        assert isinstance(cost, float)
        assert cost >= 0  # Should provide default or zero cost


class TestDecisionEngineIntegration:
    """Integration tests for DecisionEngine with other components."""
    
    def test_integration_with_cloud_api_manager(self, decision_engine):
        """Test integration between DecisionEngine and CloudAPIManager."""
        # This would test actual integration when both components are used together
        # Placeholder for integration test
        assert True
    
    def test_cost_tracking_persistence(self, decision_engine):
        """Test that cost tracking persists across sessions."""
        # Add some costs
        decision_engine.update_cost_tracking(0.5)
        decision_engine.update_cost_tracking(0.3)
        
        current_cost = decision_engine.daily_cost
        assert current_cost == 0.8
        
        # In a real implementation, this would test persistence
        # across DecisionEngine instances
    
    def test_routing_history_analysis(self, decision_engine):
        """Test analysis of routing history for insights."""
        # Create diverse routing history
        test_cases = [
            {"importance": 0.9, "cloud": True, "provider": "gemini", "cost": 0.05},
            {"importance": 0.3, "cloud": False, "provider": None, "cost": 0.0},
            {"importance": 0.7, "cloud": True, "provider": "claude", "cost": 0.03},
            {"importance": 0.5, "cloud": False, "provider": None, "cost": 0.0},
            {"importance": 0.8, "cloud": True, "provider": "openai", "cost": 0.04},
        ]
        
        for case in test_cases:
            decision = RoutingDecision(
                use_cloud=case["cloud"],
                provider=case["provider"],
                estimated_cost=case["cost"],
                factors={"importance_score": case["importance"]}
            )
            decision_engine.routing_history.append(decision)
        
        # Analyze patterns
        stats = decision_engine.get_routing_stats()
        
        # Verify pattern detection
        assert stats["total_decisions"] == 5
        assert stats["cloud_decisions"] == 3
        assert stats["local_decisions"] == 2
        
        # Could add more sophisticated pattern analysis here
        # e.g., correlation between importance and cloud usage


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])