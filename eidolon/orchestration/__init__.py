"""
Orchestration module for Eidolon AI Personal Assistant - Phase 7

Provides ecosystem orchestration for multi-application coordination,
cross-platform automation, API integration, and workflow synchronization.
"""

from .ecosystem_orchestrator import EcosystemOrchestrator, OrchestrationCapability
from .app_coordinator import AppCoordinator, AppIntegration, CrossAppWorkflow
from .workflow_synchronizer import WorkflowSynchronizer, SyncState, SyncRule
from .api_integration_manager import APIIntegrationManager, APIEndpoint, IntegrationFlow

__all__ = [
    'EcosystemOrchestrator',
    'OrchestrationCapability',
    'AppCoordinator',
    'AppIntegration',
    'CrossAppWorkflow',
    'WorkflowSynchronizer',
    'SyncState',
    'SyncRule',
    'APIIntegrationManager',
    'APIEndpoint',
    'IntegrationFlow'
]