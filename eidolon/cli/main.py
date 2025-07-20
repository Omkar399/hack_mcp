"""
Command Line Interface for Eidolon AI Personal Assistant

Provides commands for starting/stopping monitoring, searching content,
and managing the system.
"""

import sys
import json
import time
import asyncio
from pathlib import Path
from typing import Optional
import click

from ..utils.config import get_config, load_config
from ..utils.logging import setup_logging, get_logger
from ..core.observer import Observer
from .chat import chat as chat_command

# Phase 7 imports
from ..twin.digital_twin_engine import DigitalTwinEngine
from ..orchestration.ecosystem_orchestrator import EcosystemOrchestrator
from ..planning.task_planner import TaskPlanner
from ..proactive.pattern_recognizer import PatternRecognizer
from ..personality.style_replicator import StyleReplicator, ResponseType


# Global observer instance
_observer: Optional[Observer] = None


def get_observer() -> Observer:
    """Get or create the global observer instance."""
    global _observer
    if _observer is None:
        _observer = Observer()
    return _observer


@click.group()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file"
)
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    help="Set log level"
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output"
)
@click.pass_context
def cli(ctx, config, log_level, verbose):
    """Eidolon AI Personal Assistant - Your digital twin."""
    
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Set up logging
    if verbose and not log_level:
        log_level = "DEBUG"
    
    setup_logging(log_level=log_level)
    
    # Load configuration
    if config:
        ctx.obj['config'] = load_config(config)
    else:
        ctx.obj['config'] = get_config()
    
    ctx.obj['verbose'] = verbose


@cli.command()
@click.option(
    "--interval",
    "-i", 
    type=int,
    help="Screenshot capture interval in seconds"
)
@click.option(
    "--background",
    "-b",
    is_flag=True,
    help="Run in background (daemon mode)"
)
@click.pass_context
def capture(ctx, interval, background):
    """Start screenshot capture and monitoring."""
    logger = get_logger(__name__)
    
    try:
        # Get observer
        observer = get_observer()
        
        # Override interval if specified
        if interval:
            observer.config.observer.capture_interval = interval
            click.echo(f"Using capture interval: {interval} seconds")
        
        # Start monitoring
        click.echo("Starting screenshot monitoring...")
        observer.start_monitoring()
        
        if background:
            click.echo("Running in background mode. Use 'eidolon stop' to stop monitoring.")
            # In a real implementation, this would daemonize the process
            try:
                while True:
                    time.sleep(config.observer.sleep_interval_short)
            except KeyboardInterrupt:
                pass
        else:
            click.echo("Press Ctrl+C to stop monitoring...")
            try:
                while True:
                    # Show status based on configured interval
                    time.sleep(config.observer.sleep_interval_status)
                    status = observer.get_status()
                    if ctx.obj.get('verbose'):
                        click.echo(f"Captured: {status['capture_count']} screenshots")
            except KeyboardInterrupt:
                click.echo("\nStopping monitoring...")
        
        observer.stop_monitoring()
        click.echo("Screenshot monitoring stopped.")
        
    except Exception as e:
        logger.error(f"Error starting capture: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context  
def stop(ctx):
    """Stop screenshot monitoring."""
    logger = get_logger(__name__)
    
    try:
        observer = get_observer()
        observer.stop_monitoring()
        click.echo("Screenshot monitoring stopped.")
        
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("query")
@click.option(
    "--limit",
    "-l",
    type=int,
    default=10,
    help="Maximum number of results"
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json", "simple"]),
    default="table",
    help="Output format"
)
@click.option(
    "--content-type",
    type=str,
    help="Filter by content type"
)
@click.pass_context
def search(ctx, query, limit, output_format, content_type):
    """Search captured content using OCR text and metadata."""
    logger = get_logger(__name__)
    
    try:
        from ..storage.metadata_db import MetadataDatabase
        
        click.echo(f"Searching for: '{query}'")
        if content_type:
            click.echo(f"Content type filter: {content_type}")
        
        # Initialize database
        db = MetadataDatabase()
        
        # Search using database
        if content_type:
            # Filter by content type first, then search within results
            content_results = db.get_content_by_type(content_type, limit * 2)
            results = []
            for result in content_results:
                # Simple text search within the results
                searchable_text = f"{result.get('description', '')} {' '.join(result.get('tags', []))}"
                if query.lower() in searchable_text.lower():
                    results.append(result)
                    if len(results) >= limit:
                        break
        else:
            # Use full-text search on OCR text
            results = db.search_text(query, limit)
        
        # Display results
        if not results:
            click.echo("No results found.")
            return
        
        if output_format == "json":
            click.echo(json.dumps(results, indent=2, default=str))
        elif output_format == "simple":
            for result in results:
                timestamp = result.get('timestamp', 'unknown')
                if isinstance(timestamp, str) and len(timestamp) > 19:
                    timestamp = timestamp[:19]
                click.echo(f"{timestamp} - {result.get('content_type', 'unknown')}")
        else:  # table format
            click.echo(f"Found {len(results)} results:")
            click.echo("-" * 80)
            click.echo(f"{'Timestamp':<19} | {'Type':<12} | {'Confidence':<10} | {'Description'}")
            click.echo("-" * 80)
            
            for result in results:
                timestamp = result.get('timestamp', 'unknown')
                if isinstance(timestamp, str) and len(timestamp) > 19:
                    timestamp = timestamp[:19]
                
                content_type = result.get('content_type', 'unknown')[:12]
                confidence = result.get('ocr_confidence', result.get('confidence', 0))
                description = result.get('description', '')[:40]
                
                click.echo(f"{timestamp:<19} | {content_type:<12} | {confidence:<10.2f} | {description}")
        
        # Show additional stats
        if ctx.obj.get('verbose'):
            stats = db.get_statistics()
            click.echo(f"\nDatabase stats:")
            click.echo(f"Total screenshots: {stats['total_screenshots']}")
            click.echo(f"With text: {stats['screenshots_with_text']}")
            click.echo(f"Analyzed: {stats['analyzed_screenshots']}")
        
    except Exception as e:
        logger.error(f"Error searching: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output as JSON"
)
@click.pass_context
def status(ctx, as_json):
    """Show system status and statistics."""
    logger = get_logger(__name__)
    
    try:
        observer = get_observer()
        status_info = observer.get_status()
        
        if as_json:
            click.echo(json.dumps(status_info, indent=2, default=str))
        else:
            click.echo("Eidolon System Status")
            click.echo("=" * 40)
            click.echo(f"Running: {status_info['running']}")
            click.echo(f"Captures: {status_info['capture_count']}")
            click.echo(f"Start time: {status_info['start_time'] or 'Not started'}")
            click.echo(f"Storage path: {status_info['storage_path']}")
            
            if status_info['performance_metrics']:
                click.echo("\nPerformance Metrics:")
                click.echo("-" * 20)
                metrics = status_info['performance_metrics']
                click.echo(f"Captures/min: {metrics.get('captures_per_minute', 0):.1f}")
                click.echo(f"Memory usage: {metrics.get('memory_usage_mb', 0):.1f} MB")
                click.echo(f"CPU usage: {metrics.get('cpu_usage_percent', 0):.1f}%")
                click.echo(f"Duplicates filtered: {metrics.get('duplicates_filtered', 0)}")
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--days",
    type=int,
    help="Number of days of data to keep"
)
@click.option(
    "--confirm",
    is_flag=True,
    help="Skip confirmation prompt"
)
@click.pass_context
def cleanup(ctx, days, confirm):
    """Clean up old screenshots and data."""
    logger = get_logger(__name__)
    
    try:
        config = ctx.obj['config']
        if days is None:
            days = config.privacy.data_retention_days
        
        if not confirm:
            click.confirm(
                f"This will delete screenshots older than {days} days. Continue?",
                abort=True
            )
        
        observer = get_observer()
        deleted_count = observer.cleanup_old_screenshots(days)
        
        click.echo(f"Cleaned up {deleted_count} old screenshots.")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--path",
    type=click.Path(),
    help="Export path (default: ./eidolon_export.json)"
)
@click.pass_context
def export(ctx, path):
    """Export captured data."""
    logger = get_logger(__name__)
    
    if path is None:
        path = "./eidolon_export.json"
    
    try:
        config = ctx.obj['config']
        storage_path = Path(config.observer.storage_path)
        
        # Collect all metadata
        export_data = {
            "export_timestamp": time.time(),
            "config": config.dict(),
            "screenshots": []
        }
        
        if storage_path.exists():
            for file_path in storage_path.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        export_data["screenshots"].append(metadata)
                except Exception:
                    continue
        
        # Save export
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        click.echo(f"Data exported to: {path}")
        click.echo(f"Exported {len(export_data['screenshots'])} screenshots")
        
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def version(ctx):
    """Show version information."""
    from .. import __version__, __description__
    
    click.echo(f"Eidolon AI Personal Assistant v{__version__}")
    click.echo(__description__)
    click.echo()
    click.echo("Components:")
    click.echo("- Observer: Screenshot capture and monitoring")
    click.echo("- Analyzer: AI-powered content analysis")  
    click.echo("- Memory: Knowledge base and semantic search")
    click.echo("- Interface: Natural language queries")
    click.echo("- Chat: LLM-powered conversational assistant")


# Add the chat command to the CLI group
cli.add_command(chat_command)


# Phase 7 Commands - Digital Twin and Advanced Features

@cli.group()
@click.pass_context
def twin(ctx):
    """Digital Twin commands for advanced AI assistance."""
    pass


@twin.command()
@click.option(
    "--personality",
    type=click.Choice(["professional", "creative", "analytical", "collaborative", "personal", "adaptive"]),
    default="adaptive",
    help="Twin personality type"
)
@click.option(
    "--auto-learn",
    is_flag=True,
    help="Enable automatic learning from interactions"
)
@click.pass_context
def init(ctx, personality, auto_learn):
    """Initialize the digital twin."""
    click.echo("Initializing Eidolon Digital Twin...")
    
    async def init_twin():
        try:
            # Create digital twin engine
            twin_engine = DigitalTwinEngine()
            
            # Wait for initialization
            await asyncio.sleep(1)
            
            status = await twin_engine.get_twin_status()
            
            click.echo(f"Digital Twin initialized successfully!")
            click.echo(f"Twin ID: {status['twin_id']}")
            click.echo(f"Personality: {status['personality']['type']}")
            click.echo(f"Capabilities: {len(status['capabilities'])}")
            click.echo(f"State: {status['state']}")
            
            return twin_engine
            
        except Exception as e:
            click.echo(f"Error initializing twin: {e}", err=True)
            return None
    
    twin_engine = asyncio.run(init_twin())
    if twin_engine:
        # Store reference for other commands
        ctx.obj = ctx.obj or {}
        ctx.obj['twin_engine'] = twin_engine


@twin.command()
@click.option(
    "--format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format"
)
@click.pass_context
def status(ctx, format):
    """Show digital twin status."""
    
    async def get_status():
        try:
            twin_engine = DigitalTwinEngine()
            status = await twin_engine.get_twin_status()
            
            if format == "json":
                click.echo(json.dumps(status, indent=2, default=str))
            else:
                click.echo("Digital Twin Status")
                click.echo("=" * 40)
                click.echo(f"Twin ID: {status['twin_id']}")
                click.echo(f"State: {status['state']}")
                click.echo(f"Personality: {status['personality']['type']}")
                click.echo(f"Capabilities: {len(status['capabilities'])}")
                click.echo(f"Action Success Rate: {status['performance']['action_success_rate']:.2%}")
                click.echo(f"User Satisfaction: {status['performance']['user_satisfaction']:.2%}")
                click.echo(f"Patterns Learned: {status['knowledge']['patterns_learned']}")
                click.echo(f"Active Plans: {status['knowledge']['active_plans']}")
                
        except Exception as e:
            click.echo(f"Error getting twin status: {e}", err=True)
    
    asyncio.run(get_status())


@twin.command()
@click.argument("context_data")
@click.option(
    "--format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format"
)
@click.pass_context
def interact(ctx, context_data, format):
    """Interact with the digital twin using context data."""
    
    async def process_interaction():
        try:
            # Parse context data
            if context_data.startswith("{"):
                context = json.loads(context_data)
            else:
                context = {"user_input": context_data}
            
            twin_engine = DigitalTwinEngine()
            response = await twin_engine.process_context(context)
            
            if format == "json":
                click.echo(json.dumps(response, indent=2, default=str))
            else:
                click.echo("Digital Twin Response")
                click.echo("=" * 40)
                
                if response.get("predictions"):
                    click.echo(f"Predictions: {len(response['predictions'])}")
                    for pred in response["predictions"][:3]:
                        click.echo(f"  - {pred['title']} (confidence: {pred['confidence']:.2f})")
                
                if response.get("recommendations"):
                    click.echo(f"Recommendations: {len(response['recommendations'])}")
                    for rec in response["recommendations"][:3]:
                        click.echo(f"  - {rec['title']}")
                
                if response.get("planned_actions"):
                    click.echo(f"Planned Actions: {len(response['planned_actions'])}")
                    for action in response["planned_actions"][:3]:
                        click.echo(f"  - {action['action_type']}: {action['description']}")
                
        except Exception as e:
            click.echo(f"Error interacting with twin: {e}", err=True)
    
    asyncio.run(process_interaction())


@twin.command()
@click.argument("objective")
@click.option(
    "--context",
    help="Additional context as JSON string"
)
@click.pass_context
def plan(ctx, objective, context):
    """Create a task plan using the digital twin."""
    
    async def create_plan():
        try:
            context_dict = {}
            if context:
                context_dict = json.loads(context)
            
            twin_engine = DigitalTwinEngine()
            plan = await twin_engine.create_task_plan(objective, context_dict)
            
            click.echo("Task Plan Created")
            click.echo("=" * 40)
            click.echo(f"Plan ID: {plan.id}")
            click.echo(f"Title: {plan.title}")
            click.echo(f"Description: {plan.description}")
            click.echo(f"Tasks: {len(plan.tasks)}")
            click.echo(f"Estimated Duration: {plan.estimated_total_duration}")
            
            if plan.tasks:
                click.echo("\nTasks:")
                for task_id, task in list(plan.tasks.items())[:5]:
                    click.echo(f"  - {task.title} ({task.priority.value})")
                
                if len(plan.tasks) > 5:
                    click.echo(f"  ... and {len(plan.tasks) - 5} more tasks")
            
        except Exception as e:
            click.echo(f"Error creating plan: {e}", err=True)
    
    asyncio.run(create_plan())


@cli.group()
@click.pass_context
def orchestrate(ctx):
    """Ecosystem orchestration commands."""
    pass


@orchestrate.command()
@click.option(
    "--format",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format"
)
@click.pass_context
def status(ctx, format):
    """Show orchestration system status."""
    
    async def get_orchestration_status():
        try:
            orchestrator = EcosystemOrchestrator()
            status = await orchestrator.get_orchestration_status()
            
            if format == "json":
                click.echo(json.dumps(status, indent=2, default=str))
            else:
                click.echo("Ecosystem Orchestration Status")
                click.echo("=" * 40)
                click.echo(f"Orchestrator ID: {status['orchestrator_id']}")
                click.echo(f"State: {status['state']}")
                click.echo(f"Applications: {status['ecosystem']['applications']}")
                click.echo(f"Active Apps: {status['ecosystem']['active_apps']}")
                click.echo(f"Active Flows: {status['orchestration']['active_flows']}")
                click.echo(f"Automation Rules: {status['orchestration']['automation_rules']}")
                click.echo(f"Pending Events: {status['orchestration']['pending_events']}")
                
        except Exception as e:
            click.echo(f"Error getting orchestration status: {e}", err=True)
    
    asyncio.run(get_orchestration_status())


@orchestrate.command()
@click.argument("flow_definition")
@click.pass_context
def create_flow(ctx, flow_definition):
    """Create an orchestration flow from JSON definition."""
    
    async def create_flow():
        try:
            if flow_definition.startswith("{"):
                flow_def = json.loads(flow_definition)
            else:
                # Load from file
                with open(flow_definition, 'r') as f:
                    flow_def = json.load(f)
            
            orchestrator = EcosystemOrchestrator()
            flow = await orchestrator.create_orchestration_flow(flow_def)
            
            click.echo("Orchestration Flow Created")
            click.echo("=" * 40)
            click.echo(f"Flow ID: {flow.id}")
            click.echo(f"Name: {flow.name}")
            click.echo(f"Description: {flow.description}")
            click.echo(f"Steps: {len(flow.steps)}")
            click.echo(f"Status: {flow.status}")
            
        except Exception as e:
            click.echo(f"Error creating flow: {e}", err=True)
    
    asyncio.run(create_flow())


@orchestrate.command()
@click.argument("flow_id")
@click.pass_context
def execute_flow(ctx, flow_id):
    """Execute an orchestration flow."""
    
    async def execute():
        try:
            orchestrator = EcosystemOrchestrator()
            result = await orchestrator.execute_orchestration_flow(flow_id)
            
            click.echo("Flow Execution Result")
            click.echo("=" * 40)
            click.echo(f"Success: {result['success']}")
            click.echo(f"Completed Steps: {result['completed_steps']}")
            
            if not result['success'] and 'error' in result:
                click.echo(f"Error: {result['error']}")
            
        except Exception as e:
            click.echo(f"Error executing flow: {e}", err=True)
    
    asyncio.run(execute())


@cli.group()
@click.pass_context
def patterns(ctx):
    """Pattern recognition and analysis commands."""
    pass


@patterns.command()
@click.option(
    "--days",
    type=int,
    default=7,
    help="Number of days to analyze"
)
@click.option(
    "--type",
    "pattern_type",
    type=click.Choice(["temporal", "workflow", "application", "communication", "content", "productivity", "break", "error"]),
    help="Specific pattern type to analyze"
)
@click.pass_context
def analyze(ctx, days, pattern_type):
    """Analyze user patterns."""
    
    async def analyze_patterns():
        try:
            recognizer = PatternRecognizer()
            
            # Set analysis window
            end_date = None
            start_date = None
            if days:
                from datetime import datetime, timedelta
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
            
            # Filter pattern types
            pattern_types = None
            if pattern_type:
                from ..proactive.pattern_recognizer import PatternType
                pattern_types = [PatternType(pattern_type)]
            
            patterns = await recognizer.analyze_user_patterns(start_date, end_date, pattern_types)
            
            click.echo(f"Pattern Analysis Results ({days} days)")
            click.echo("=" * 40)
            click.echo(f"Patterns found: {len(patterns)}")
            
            for pattern in patterns[:10]:
                click.echo(f"\n{pattern.title}")
                click.echo(f"  Type: {pattern.pattern_type.value}")
                click.echo(f"  Strength: {pattern.strength:.2f}")
                click.echo(f"  Confidence: {pattern.confidence:.2f}")
                click.echo(f"  Occurrences: {len(pattern.occurrences)}")
                click.echo(f"  Description: {pattern.description}")
            
            if len(patterns) > 10:
                click.echo(f"\n... and {len(patterns) - 10} more patterns")
            
        except Exception as e:
            click.echo(f"Error analyzing patterns: {e}", err=True)
    
    asyncio.run(analyze_patterns())


@patterns.command()
@click.pass_context
def insights(ctx):
    """Get pattern insights and recommendations."""
    
    async def get_insights():
        try:
            recognizer = PatternRecognizer()
            insights = await recognizer.get_pattern_insights()
            
            click.echo("Pattern Insights")
            click.echo("=" * 40)
            click.echo(f"Total patterns: {insights['summary']['total_patterns']}")
            
            if insights['summary']['by_type']:
                click.echo("\nPatterns by type:")
                for ptype, count in insights['summary']['by_type'].items():
                    click.echo(f"  {ptype}: {count}")
            
            if insights['summary']['most_reliable']:
                click.echo("\nMost reliable patterns:")
                for pattern in insights['summary']['most_reliable']:
                    click.echo(f"  - {pattern['title']} ({pattern['type']})")
            
            if insights['recommendations']:
                click.echo("\nRecommendations:")
                for rec in insights['recommendations']:
                    click.echo(f"  - {rec}")
            
        except Exception as e:
            click.echo(f"Error getting insights: {e}", err=True)
    
    asyncio.run(get_insights())


@cli.group()
@click.pass_context
def style(ctx):
    """Communication style analysis and replication commands."""
    pass


@style.command()
@click.argument("text_samples", nargs=-1)
@click.option(
    "--file",
    type=click.Path(exists=True),
    help="Load text samples from file (one per line)"
)
@click.pass_context
def analyze(ctx, text_samples, file):
    """Analyze communication style from text samples."""
    
    async def analyze_style():
        try:
            replicator = StyleReplicator()
            
            # Prepare samples
            samples = []
            
            if file:
                with open(file, 'r') as f:
                    for line in f:
                        if line.strip():
                            samples.append({"text": line.strip(), "type": "text"})
            
            for sample in text_samples:
                samples.append({"text": sample, "type": "text"})
            
            if not samples:
                click.echo("No text samples provided. Use --file or provide samples as arguments.")
                return
            
            style_model = await replicator.analyze_communication_samples(samples)
            
            click.echo("Style Analysis Results")
            click.echo("=" * 40)
            click.echo(f"Formality Score: {style_model.formality_score:.2f}")
            click.echo(f"Verbosity Score: {style_model.verbosity_score:.2f}")
            click.echo(f"Technical Score: {style_model.technical_score:.2f}")
            click.echo(f"Emotion Score: {style_model.emotion_score:.2f}")
            click.echo(f"Structure Score: {style_model.structure_score:.2f}")
            click.echo(f"Avg Sentence Length: {style_model.avg_sentence_length:.1f} words")
            click.echo(f"Sample Count: {style_model.sample_count}")
            click.echo(f"Confidence: {style_model.confidence:.2f}")
            
            if style_model.common_openings:
                click.echo(f"\nCommon Openings: {', '.join(style_model.common_openings[:3])}")
            
            if style_model.common_closings:
                click.echo(f"Common Closings: {', '.join(style_model.common_closings[:3])}")
            
        except Exception as e:
            click.echo(f"Error analyzing style: {e}", err=True)
    
    asyncio.run(analyze_style())


@style.command()
@click.argument("text")
@click.option(
    "--type",
    "response_type",
    type=click.Choice(["email", "message", "document", "code_comment", "social_post"]),
    default="message",
    help="Type of response to generate"
)
@click.pass_context
def generate(ctx, text, response_type):
    """Generate text in the user's communication style."""
    
    async def generate_styled():
        try:
            replicator = StyleReplicator()
            
            # Convert string to enum
            from ..personality.style_replicator import ResponseType
            rtype = ResponseType(response_type)
            
            result = await replicator.generate_styled_response(text, rtype)
            
            click.echo("Generated Response")
            click.echo("=" * 40)
            click.echo(result["response"])
            click.echo()
            click.echo(f"Styled: {result['styled']}")
            click.echo(f"Confidence: {result['confidence']:.2f}")
            
            if result.get('adjustments'):
                click.echo(f"Adjustments made: {len(result['adjustments'])}")
                for adj in result['adjustments'][:3]:
                    click.echo(f"  - {adj}")
            
        except Exception as e:
            click.echo(f"Error generating styled response: {e}", err=True)
    
    asyncio.run(generate_styled())


@cli.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "http", "websocket"]),
    default="stdio",
    help="MCP transport method"
)
@click.option(
    "--port",
    type=int,
    default=8080,
    help="Port for HTTP/WebSocket transport"
)
@click.option(
    "--host",
    type=str,
    default="localhost",
    help="Host for HTTP/WebSocket transport"
)
@click.option(
    "--auto-monitor",
    is_flag=True,
    help="Automatically start screenshot monitoring"
)
@click.pass_context
def mcp(ctx, transport, port, host, auto_monitor):
    """Start the MCP (Model Context Protocol) server."""
    logger = get_logger(__name__)
    
    try:
        from ..core.mcp_server import main as mcp_main
        
        click.echo("Starting Eidolon MCP Server...")
        click.echo(f"Transport: {transport}")
        
        if transport != "stdio":
            click.echo(f"Host: {host}")
            click.echo(f"Port: {port}")
        
        if auto_monitor:
            click.echo("Auto-monitoring enabled")
        
        # Set environment variables for MCP server configuration
        import os
        os.environ['EIDOLON_MCP_TRANSPORT'] = transport
        os.environ['EIDOLON_MCP_HOST'] = host
        os.environ['EIDOLON_MCP_PORT'] = str(port)
        os.environ['EIDOLON_MCP_AUTO_MONITOR'] = str(auto_monitor)
        
        # Start the MCP server
        click.echo("MCP server starting... (use Ctrl+C to stop)")
        mcp_main()
        
    except KeyboardInterrupt:
        click.echo("\nMCP server stopped.")
    except Exception as e:
        logger.error(f"Error starting MCP server: {e}")
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()