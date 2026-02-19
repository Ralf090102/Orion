#!/usr/bin/env python3
"""
Orion - Local RAG Assistant CLI

Main command-line interface for ingesting documents and querying your
personal knowledge base.
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
import json
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

from src.core.ingest import (
    DocumentIngestor,
    clear_knowledge_base,
    get_supported_formats,
    ingest_documents,
    ingest_with_watchdog,
)
from src.generation.generate import AnswerGenerator
from src.generation.session_manager import get_session_manager
from src.retrieval.retriever import OrionRetriever
from src.retrieval.vector_store import create_vector_store
from src.utilities.config import get_config
from src.utilities.utils import log_error, log_info, log_warning, log_success

app = typer.Typer(
    name="orion",
    help="Orion - Local RAG Assistant for your personal knowledge base",
    add_completion=False,
)

console = Console()


# ========== HELPER FUNCTIONS ==========
def print_banner():
    """Print Orion banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║      ██████╗ ██████╗ ██╗ ██████╗ ███╗   ██╗                   ║
║     ██╔═══██╗██╔══██╗██║██╔═══██╗████╗  ██║                   ║
║     ██║   ██║██████╔╝██║██║   ██║██╔██╗ ██║                   ║
║     ██║   ██║██╔══██╗██║██║   ██║██║╚██╗██║                   ║
║     ╚██████╔╝██║  ██║██║╚██████╔╝██║ ╚████║                   ║
║      ╚═════╝ ╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝                   ║
║                                                               ║
║         Local RAG Assistant                                   ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


def check_gpu_status():
    """Check and display GPU status"""
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            console.print(
                f"🚀 GPU: {gpu_name} ({gpu_memory:.1f} GB)",
                style="bold green",
            )
            return True
        else:
            console.print("💻 Running on CPU", style="yellow")
            return False
    except ImportError:
        console.print("⚠️  PyTorch not installed", style="yellow")
        return False


def print_config_summary():
    """Print current configuration summary"""
    config = get_config()

    table = Table(title="Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    # Embedding
    table.add_row("Embedding Model", config.rag.embedding.model)
    table.add_row("Batch Size", str(config.rag.embedding.batch_size))

    # Chunking
    table.add_row("Chunk Size", str(config.rag.chunking.chunk_size))
    table.add_row("Chunk Overlap", str(config.rag.chunking.chunk_overlap))

    # Retrieval
    table.add_row("Default K", str(config.rag.retrieval.default_k))
    table.add_row("Hybrid Search", "✓" if config.rag.retrieval.enable_hybrid_search else "✗")
    table.add_row("MMR", "✓" if config.rag.retrieval.enable_mmr else "✗")
    table.add_row("Reranking", "✓" if config.rag.retrieval.enable_reranking else "✗")

    # GPU
    table.add_row("GPU Enabled", "✓" if config.gpu.enabled else "✗")

    console.print(table)
    
# ========== SOURCES COMMAND ==========
@app.command()
def sources(
    source_file: Optional[str] = typer.Option(None, "--file", "-f", help="Show chunks from specific source file"),
    json_output: bool = typer.Option(False, "--json", help="Output results as JSON"),
    config_env: bool = typer.Option(False, "--env", help="Load configuration from environment variables"),
):
    """List all sources in the knowledge base or show chunks from a specific source."""
    print_banner()

    config = get_config(from_env=config_env)

    try:
        vector_store = create_vector_store(config=config)

        if source_file:
            chunks = vector_store.get_chunks_by_source(source_file)

            if not chunks:
                log_info(f"No chunks found for source: {source_file}", config=config)
                return

            if json_output:
                console.print(json.dumps(chunks, indent=2, ensure_ascii=False))
            else:
                log_success(f"Found {len(chunks)} chunks from {source_file}", config=config)
                for i, chunk in enumerate(chunks, 1):
                    console.print(f"\n[bold cyan]Chunk {i}:[/bold cyan]")
                    content = chunk.get("content") or chunk.get("document") or chunk.get("text") or ""
                    console.print(content[:200] + "..." if len(content) > 200 else content)
                    console.print(f"Metadata: {chunk.get('metadata', {})}")

        else:
            sources = vector_store.list_all_sources()

            if not sources:
                log_info("No documents found in knowledge base", config=config)
                return

            if json_output:
                console.print(json.dumps(sources, indent=2, ensure_ascii=False))
            else:
                total_files = len(sources)
                total_chunks = sum(s.get("chunk_count", 0) for s in sources)

                table = Table(title=f"Knowledge Base Sources ({total_files} files)", show_header=True, header_style="bold magenta")
                table.add_column("Source File", style="cyan")
                table.add_column("File Name", style="white")
                table.add_column("Type", style="green")
                table.add_column("Chunks", style="green", justify="right")

                for s in sorted(sources, key=lambda x: x.get("file_name", "")):
                    table.add_row(s.get("source_file", ""), s.get("file_name", ""), s.get("file_type", ""), str(s.get("chunk_count", 0)))

                console.print(table)
                log_success(f"Total: {total_files} source files, {total_chunks} chunks", config=config)

    except Exception as e:
        log_error(f"Failed to retrieve sources: {e}", config=config)
        raise typer.Exit(1)


# ========== INGEST COMMAND ==========
@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to file or directory to ingest"),
    clear: bool = typer.Option(False, "--clear", "-c", help="Clear existing knowledge base first"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch for file changes and auto-ingest"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r/-R", help="Recursively scan directories"),
    skip_existing: bool = typer.Option(True, "--skip-existing", help="Skip files already present in the vector store"),
    gpu: bool = typer.Option(None, "--gpu/--no-gpu", help="Enable/disable GPU acceleration"),
):
    """
    Ingest documents into the knowledge base.

    Examples:
        orion ingest /path/to/documents
        orion ingest /path/to/file.pdf --clear
        orion ingest /path/to/kb --watch
    """
    print_banner()

    # Update GPU setting if specified
    config = get_config()
    if gpu is not None:
        config.gpu.enabled = gpu

    # Check GPU
    check_gpu_status()
    console.print()

    path_obj = Path(path)
    if not path_obj.exists():
        console.print(f"❌ Error: Path not found: {path}", style="bold red")
        raise typer.Exit(1)

    # Watch mode
    if watch:
        console.print(f"👀 Starting file watcher for: {path}", style="bold cyan")
        console.print("   Press Ctrl+C to stop\n", style="dim")

        try:
            ingestor, watcher = ingest_with_watchdog([str(path_obj)], config=config)

            # Show initial stats
            stats = ingestor.get_ingestion_summary()
            console.print(f"📊 Current knowledge base: {stats.get('total_documents', 0)} chunks", style="green")

            # Keep running
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            console.print("\n\n⏹️  Stopping watcher...", style="yellow")
            watcher.stop()
            console.print("✅ Watcher stopped", style="green")
            return

    # Regular ingestion
    with console.status("[bold green]Ingesting documents...") as status:
        start_time = time.time()

        try:
            stats = ingest_documents(
                path,
                config=config,
                clear_existing=clear,
                skip_existing=skip_existing,
                recursive=recursive,
            )

            elapsed = time.time() - start_time

            # Display results
            console.print()
            result_table = Table(
                title="✅ Ingestion Complete",
                show_header=True,
                header_style="bold green",
            )
            result_table.add_column("Metric", style="cyan")
            result_table.add_column("Value", style="green", justify="right")

            result_table.add_row("Total Files", str(stats.total_files))
            result_table.add_row("Successful", str(stats.successful_files))
            result_table.add_row("Failed", str(stats.failed_files))
            result_table.add_row("Success Rate", f"{stats.success_rate:.1f}%")
            result_table.add_row("Total Chunks", str(stats.total_chunks))
            result_table.add_row("Processing Time", f"{elapsed:.2f}s")

            console.print(result_table)

            # Show errors if any
            if stats.errors:
                console.print("\n⚠️  Errors:", style="bold yellow")
                for error in stats.errors[:5]:  # Show first 5 errors
                    console.print(f"  • {error}", style="yellow")
                if len(stats.errors) > 5:
                    console.print(f"  ... and {len(stats.errors) - 5} more errors", style="dim")

        except Exception as e:
            console.print(f"\n❌ Ingestion failed: {e}", style="bold red")
            raise typer.Exit(1)

# ========== QUERY COMMAND ==========
@app.command()
def query(
    question: str = typer.Argument(..., help="Your question to search the knowledge base"),
    k: int = typer.Option(5, "--top-k", "-k", help="Number of results to return"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed results"),
    gpu: bool = typer.Option(None, "--gpu/--no-gpu", help="Enable/disable GPU acceleration"),
):
    """
    Query your knowledge base.

    Examples:
        orion query "What is RAG?"
        orion query "Explain embeddings" --top-k 10
        orion query "Search query" --verbose
    """
    print_banner()

    # Update GPU setting if specified
    config = get_config()
    if gpu is not None:
        config.gpu.enabled = gpu

    # Check GPU
    check_gpu_status()
    console.print()

    console.print(f"🔍 Query: [bold cyan]{question}[/bold cyan]\n")

    with console.status("[bold green]Searching knowledge base...") as status:
        try:
            start_time = time.time()

            # Initialize retriever
            retriever = OrionRetriever(config=config)

            # Perform search (formatted=False to get list of SearchResult objects)
            results = retriever.query(question, k=k, formatted=False)

            elapsed = time.time() - start_time

            # Display results
            console.print(f"⚡ Found {len(results)} results in {elapsed:.2f}s\n", style="bold green")

            if not results:
                console.print("No results found.", style="yellow")
                return

            for i, result in enumerate(results, 1):
                # Create panel for each result
                content = result.content[:500] + "..." if len(result.content) > 500 else result.content

                metadata = result.metadata
                source = metadata.get("source", metadata.get("file_path", "Unknown"))
                file_name = metadata.get("file_name", Path(source).name if source != "Unknown" else "Unknown")

                panel_title = f"[{i}] Score: {result.score:.4f} | Source: {file_name}"

                if verbose:
                    # Detailed view
                    details = f"""[bold]Content:[/bold]
{content}

[bold]Metadata:[/bold]
  • Source: {source}
  • File Type: {metadata.get('file_type', 'Unknown')}
  • Chunk: {metadata.get('chunk_index', 'N/A')} / {metadata.get('chunk_count', 'N/A')}
"""
                    console.print(Panel(details, title=panel_title, border_style="cyan"))
                else:
                    # Compact view
                    console.print(Panel(content, title=panel_title, border_style="cyan"))

                console.print()

        except Exception as e:
            console.print(f"\n❌ Query failed: {e}", style="bold red")
            import traceback

            traceback.print_exc()
            raise typer.Exit(1)


# ========== ASK COMMAND (RAG MODE WITH LLM) ==========
@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question"),
    k: int = typer.Option(5, "--top-k", "-k", help="Number of contexts to use"),
    show_sources: bool = typer.Option(True, "--sources/--no-sources", help="Show source citations"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information"),
    gpu: bool = typer.Option(None, "--gpu/--no-gpu", help="Enable/disable GPU acceleration"),
):
    """
    Ask a question and get an AI-generated answer with citations (RAG mode).

    This uses the full RAG pipeline: retrieval → context preparation → 
    prompt building → LLM generation.

    Examples:
        orion ask "What is machine learning?"
        orion ask "Explain RAG" --top-k 10
        orion ask "How does retrieval work?" --no-sources
    """
    print_banner()

    # Update GPU setting if specified
    config = get_config()
    if gpu is not None:
        config.gpu.enabled = gpu

    # Check GPU
    check_gpu_status()
    console.print()

    console.print(f"💬 Question: [bold cyan]{question}[/bold cyan]\n")

    with console.status("[bold green]Generating answer...") as status:
        try:
            start_time = time.time()

            # Initialize generator
            generator = AnswerGenerator(config=config)

            # Generate answer
            result = generator.generate_rag_response(
                query=question, k=k, include_sources=show_sources
            )

            elapsed = time.time() - start_time

            # Display answer
            console.print()
            answer_panel = Panel(
                result.answer,
                title=f"✨ Answer ({result.query_type} query)",
                border_style="green",
                padding=(1, 2),
            )
            console.print(answer_panel)
            console.print()

            # Display sources if requested
            if show_sources and result.sources:
                console.print(f"📚 [bold cyan]Sources ({len(result.sources)}):[/bold cyan]\n")

                for source in result.sources:
                    # Format source info
                    source_title = f"[{source['index']}] Score: {source['score']:.4f}"

                    if source.get('citation'):
                        source_title += f" | {source['citation']}"
                    elif source.get('source_file'):
                        source_title += f" | {source['source_file']}"

                    if verbose:
                        # Detailed view with preview
                        source_text = source.get('text', '')
                        console.print(
                            Panel(
                                source_text,
                                title=source_title,
                                border_style="cyan",
                            )
                        )
                    else:
                        # Compact view
                        console.print(f"  {source_title}")

                    console.print()

            # Show metadata if verbose
            if verbose:
                meta_table = Table(
                    title="Metadata", show_header=False, box=None, padding=(0, 1)
                )
                meta_table.add_column("Key", style="dim")
                meta_table.add_column("Value", style="green")

                meta_table.add_row("Query Type", result.metadata.get("query_type", "Unknown"))
                meta_table.add_row("Contexts Used", str(result.metadata.get("num_contexts_used", 0)))
                meta_table.add_row("Total Tokens", str(result.metadata.get("total_tokens", 0)))
                meta_table.add_row("LLM Model", result.metadata.get("llm_model", "Unknown"))
                meta_table.add_row("Processing Time", f"{elapsed:.2f}s")

                console.print(meta_table)
                console.print()
                
                # Display timing breakdown if available
                if result.timing:
                    timing_summary = result.timing.format_timing_summary()
                    console.print(Panel(
                        timing_summary,
                        title="⏱️  Performance Breakdown",
                        border_style="blue",
                        padding=(1, 2)
                    ))
                    console.print()

        except Exception as e:
            console.print(f"\n❌ Failed to generate answer: {e}", style="bold red")
            import traceback

            traceback.print_exc()
            raise typer.Exit(1)


# ========== STATUS COMMAND ==========
@app.command()
def status():
    """
    Show knowledge base status and statistics.
    """
    print_banner()

    config = get_config()

    # GPU Status
    check_gpu_status()
    console.print()

    # Vector store stats
    with console.status("[bold green]Loading statistics..."):
        try:
            vector_store = create_vector_store(config=config)
            stats = vector_store.get_collection_stats()

            # Stats table
            stats_table = Table(
                title="📊 Knowledge Base Status",
                show_header=True,
                header_style="bold green",
            )
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="green", justify="right")

            stats_table.add_row("Total Chunks", str(stats.get("total_chunks", 0)))
            stats_table.add_row("Unique Files", str(stats.get("unique_files", 0)))
            stats_table.add_row("Collection", stats.get("collection_name", "Unknown"))
            stats_table.add_row("Storage Path", stats.get("persist_directory", "Unknown"))

            # File type distribution
            file_types = stats.get("file_type_distribution", {})
            if file_types:
                stats_table.add_row("", "")  # Separator
                stats_table.add_row("[bold]File Types[/bold]", "")
                for file_type, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
                    stats_table.add_row(f"  {file_type}", str(count))

            console.print(stats_table)

        except Exception as e:
            console.print(f"❌ Failed to get status: {e}", style="bold red")
            raise typer.Exit(1)

    console.print()

    # Configuration
    print_config_summary()


# ========== CLEAR COMMAND ==========
@app.command()
def clear(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """
    Clear the entire knowledge base.

    WARNING: This will delete all ingested documents and embeddings.
    """
    if not confirm:
        console.print(
            "⚠️  [bold yellow]WARNING:[/bold yellow] This will delete all ingested documents and embeddings!",
        )
        response = typer.confirm("Are you sure you want to continue?")
        if not response:
            console.print("Cancelled.", style="yellow")
            raise typer.Exit(0)

    with console.status("[bold yellow]Clearing knowledge base..."):
        try:
            config = get_config()
            success = clear_knowledge_base(config=config)

            if success:
                console.print("\n✅ Knowledge base cleared successfully", style="bold green")
            else:
                console.print("\n❌ Failed to clear knowledge base", style="bold red")
                raise typer.Exit(1)

        except Exception as e:
            console.print(f"\n❌ Error: {e}", style="bold red")
            raise typer.Exit(1)


# ========== FORMATS COMMAND ==========
@app.command()
def formats():
    """
    Show supported file formats.
    """
    print_banner()

    supported = get_supported_formats()

    # Create tree structure
    tree = Tree("📄 Supported File Formats", guide_style="cyan")

    # Group by category
    categories = {
        "Documents": [".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".txt", ".md", ".rtf"],
        "Data": [".csv", ".json", ".xml", ".yaml", ".yml"],
        "Code": [
            ".py",
            ".js",
            ".ts",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".hpp",
            ".cs",
            ".go",
            ".rs",
            ".rb",
            ".php",
            ".swift",
            ".kt",
        ],
        "Web": [".html", ".css", ".scss", ".jsx", ".tsx", ".vue"],
        "Config": [".ini", ".conf", ".toml"],
    }

    for category, extensions in categories.items():
        category_branch = tree.add(f"[bold cyan]{category}[/bold cyan]")
        for ext in extensions:
            if ext in supported:
                category_branch.add(f"[green]{ext}[/green] - {supported[ext]}")

    console.print(tree)
    console.print(f"\n[bold]Total:[/bold] {len(supported)} file formats supported", style="green")


# ========== CONFIG COMMAND ==========
@app.command()
def config(
    show_all: bool = typer.Option(False, "--all", "-a", help="Show all configuration options"),
):
    """
    Show current configuration.
    """
    print_banner()

    if show_all:
        # Show full config as JSON
        from rich.syntax import Syntax
        import json

        config = get_config()
        config_dict = config.model_dump()

        json_str = json.dumps(config_dict, indent=2)
        syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)

        console.print(Panel(syntax, title="Full Configuration", border_style="cyan"))
    else:
        # Show summary
        check_gpu_status()
        console.print()
        print_config_summary()


# ========== SESSIONS COMMAND ==========
@app.command()
def sessions(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed session info"),
):
    """
    List all saved chat sessions.

    Shows session IDs, message counts, and timestamps.
    Use the session ID with 'chat --session <id>' to resume.

    Examples:
        orion sessions
        orion sessions --verbose
    """
    print_banner()

    # Initialize session manager with persistence
    session_manager = get_session_manager(persist_to_disk=True)

    sessions_list = session_manager.list_sessions()

    if not sessions_list:
        console.print("📭 No saved sessions found.", style="yellow")
        console.print("\nUse [cyan]orion chat --persist[/cyan] to save conversations.", style="dim")
        return

    # Sort by updated_at (most recent first)
    sessions_list = sorted(
        sessions_list,
        key=lambda s: datetime.fromisoformat(s["updated_at"]),
        reverse=True
    )

    console.print(f"📋 [bold cyan]Saved Chat Sessions ({len(sessions_list)}):[/bold cyan]\n")

    # Create table
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=3)
    table.add_column("Session ID", style="cyan", no_wrap=True)
    table.add_column("Messages", justify="right", style="green", width=10)
    table.add_column("Last Updated", style="yellow", width=14)
    
    if verbose:
        table.add_column("Created", style="dim", width=16)

    for i, session in enumerate(sessions_list, 1):
        session_id = session["session_id"]
        
        # Parse timestamp
        updated_at = datetime.fromisoformat(session["updated_at"])
        time_ago = _format_time_ago(updated_at)
        
        row = [
            str(i),
            session_id,
            str(session["message_count"]),
            time_ago
        ]
        
        if verbose:
            created_at = datetime.fromisoformat(session["created_at"])
            row.append(created_at.strftime("%Y-%m-%d %H:%M"))
        
        table.add_row(*row)

    console.print(table)
    
    # Show most recent session info
    if sessions_list:
        most_recent = sessions_list[0]
        console.print(f"\n💡 [dim]To resume most recent session:[/dim]")
        console.print(f"   [cyan]python run.py chat --persist[/cyan]  (auto-resumes)")
        console.print(f"   [cyan]python run.py chat --session {most_recent['session_id']}[/cyan]")


def _format_time_ago(dt: datetime) -> str:
    """Format datetime as human-readable time ago."""
    from datetime import datetime, timedelta
    
    now = datetime.now()
    diff = now - dt
    
    if diff < timedelta(minutes=1):
        return "just now"
    elif diff < timedelta(hours=1):
        mins = int(diff.total_seconds() / 60)
        return f"{mins}m ago"
    elif diff < timedelta(days=1):
        hours = int(diff.total_seconds() / 3600)
        return f"{hours}h ago"
    elif diff < timedelta(days=7):
        days = diff.days
        return f"{days}d ago"
    else:
        return dt.strftime("%Y-%m-%d")


# ========== DELETE-SESSION COMMAND ==========
@app.command("delete-session")
def delete_session(
    session_id: str = typer.Argument(..., help="Session ID to delete"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """
    Delete a specific chat session.

    This will permanently delete the session and all its messages.

    Examples:
        orion delete-session abc123-def456-...
        orion delete-session <session-id> --yes
    """
    print_banner()

    # Initialize session manager with persistence
    session_manager = get_session_manager(persist_to_disk=True)

    # Check if session exists
    session = session_manager.get_session(session_id)
    if not session:
        console.print(f"❌ Session not found: [cyan]{session_id}[/cyan]", style="bold red")
        console.print("\nUse [cyan]orion sessions[/cyan] to see available sessions.", style="dim")
        raise typer.Exit(1)

    # Show session info
    console.print(f"📝 Session: [cyan]{session_id}[/cyan]")
    console.print(f"   Messages: {len(session.messages)}")
    console.print(f"   Created: {session.created_at}")
    console.print()

    # Confirm deletion
    if not confirm:
        console.print("⚠️  [bold yellow]WARNING:[/bold yellow] This will permanently delete this session!")
        response = typer.confirm("Are you sure you want to continue?")
        if not response:
            console.print("Cancelled.", style="yellow")
            raise typer.Exit(0)

    # Delete session
    success = session_manager.delete_session(session_id)

    if success:
        console.print(f"\n✅ Deleted session: [cyan]{session_id}[/cyan]", style="bold green")
    else:
        console.print(f"\n❌ Failed to delete session", style="bold red")
        raise typer.Exit(1)


# ========== CLEAR-SESSIONS COMMAND ==========
@app.command("clear-sessions")
def clear_sessions(
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
):
    """
    Delete all chat sessions.

    WARNING: This will permanently delete all saved conversations.

    Examples:
        orion clear-sessions
        orion clear-sessions --yes
    """
    print_banner()

    # Initialize session manager with persistence
    session_manager = get_session_manager(persist_to_disk=True)

    sessions_list = session_manager.list_sessions()

    if not sessions_list:
        console.print("📭 No sessions found.", style="yellow")
        return

    # Show count
    console.print(f"📋 Found [cyan]{len(sessions_list)}[/cyan] session(s)\n")

    # Confirm deletion
    if not confirm:
        console.print(
            "⚠️  [bold yellow]WARNING:[/bold yellow] This will permanently delete ALL chat sessions and messages!"
        )
        response = typer.confirm("Are you sure you want to continue?")
        if not response:
            console.print("Cancelled.", style="yellow")
            raise typer.Exit(0)

    # Delete all sessions
    with console.status("[bold yellow]Deleting sessions..."):
        count = session_manager.delete_all_sessions()

    console.print(f"\n✅ Deleted {count} session(s)", style="bold green")


# ========== CHAT COMMAND (CONVERSATIONAL MODE) ==========
@app.command()
def chat(
    persist: bool = typer.Option(False, "--persist", "-p", help="Save conversation to disk"),
    session_id: str = typer.Option(None, "--session", "-s", help="Resume existing session"),
    show_sources: bool = typer.Option(False, "--sources", help="Show sources when RAG is triggered"),
    rag_mode: str = typer.Option(None, "--rag-mode", help="RAG trigger mode: always/auto/manual/never"),
    gpu: bool = typer.Option(None, "--gpu/--no-gpu", help="Enable/disable GPU acceleration"),
):
    """
    Start interactive chat mode with conversation memory.

    This mode maintains conversation history and uses RAG intelligently:
    - Auto mode: Retrieves context when detecting questions
    - Always mode: Always retrieves context
    - Manual mode: Use 'search: your query' to trigger RAG
    - Never mode: Pure chat without retrieval

    Examples:
        orion chat
        orion chat --persist --rag-mode auto
        orion chat --session abc123 --sources
    """
    print_banner()

    # Update config
    config = get_config()
    if gpu is not None:
        config.gpu.enabled = gpu
    if rag_mode:
        config.rag.generation.rag_trigger_mode = rag_mode

    # Set mode to chat
    config.rag.generation.mode = "chat"

    check_gpu_status()
    console.print()

    # Initialize session manager
    session_manager = get_session_manager(persist_to_disk=persist)

    # Create or resume session
    if session_id:
        # User explicitly provided a session ID
        session = session_manager.get_session(session_id)
        if session:
            console.print(f"📝 Resumed session: [cyan]{session_id}[/cyan]", style="green")
            console.print(f"   Messages: {len(session.messages)}", style="dim")
        else:
            console.print(f"⚠️  Session not found: {session_id}", style="yellow")
            session_id = session_manager.create_session()
            console.print(f"📝 Created new session: [cyan]{session_id}[/cyan]", style="green")
    else:
        # No session ID provided - try to auto-resume most recent session if persist is enabled
        if persist:
            recent_session = session_manager.get_most_recent_session()
            if recent_session and len(recent_session.messages) > 0:
                session_id = recent_session.session_id
                console.print(f"📝 Auto-resumed recent session: [cyan]{session_id[:8]}...[/cyan]", style="green")
                console.print(f"   Messages: {len(recent_session.messages)}", style="dim")
                console.print(f"   [dim]Tip: Use --session {session_id} to explicitly resume this session[/dim]\n")
            else:
                session_id = session_manager.create_session()
                console.print(f"📝 New session: [cyan]{session_id[:8]}...[/cyan]", style="green")
                console.print(f"   [dim]Tip: Use --session {session_id} to resume later[/dim]\n")
        else:
            # In-memory session
            session_id = session_manager.create_session()
            console.print(f"📝 Session: [cyan]{session_id[:8]}...[/cyan] (in-memory)", style="green")

    # Display mode info
    console.print(
        f"🤖 Chat Mode | RAG: [cyan]{config.rag.generation.rag_trigger_mode}[/cyan]",
        style="bold",
    )

    console.print("\n[bold cyan]Commands:[/bold cyan]")
    console.print("  • Type your message and press Enter")
    console.print("  • 'clear' - Clear conversation history")
    console.print("  • 'history' - Show conversation summary")
    console.print("  • 'sources' - Toggle source display")
    console.print("  • 'exit' or 'quit' - Exit chat mode")
    console.print()

    # Initialize generator
    with console.status("[bold green]Initializing chat..."):
        try:
            generator = AnswerGenerator(config=config)

            # Restore conversation history from session
            messages = session_manager.get_messages(session_id)
            generator.prompt_builder.conversation_history = messages

            console.print("✅ Ready to chat!\n", style="green")
        except Exception as e:
            console.print(f"❌ Failed to initialize: {e}", style="bold red")
            raise typer.Exit(1)

    # Chat loop
    while True:
        try:
            # Get user input
            message = console.input("[bold cyan]You:[/bold cyan] ")

            if not message.strip():
                continue

            message_lower = message.lower().strip()

            # Handle commands
            if message_lower in ["exit", "quit", "q"]:
                console.print("\n👋 Goodbye!", style="bold cyan")
                break

            elif message_lower == "clear":
                generator.clear_conversation()
                session_manager.clear_session_messages(session_id)
                console.print("✅ Conversation cleared\n", style="green")
                continue

            elif message_lower == "history":
                summary = generator.get_conversation_summary()
                console.print("\n📊 Conversation Summary:", style="bold cyan")
                console.print(f"  • Total messages: {summary['total_messages']}")
                console.print(f"  • User messages: {summary['user_messages']}")
                console.print(f"  • Assistant messages: {summary['assistant_messages']}")
                console.print(f"  • Total tokens: {summary['total_tokens']}\n")
                continue

            elif message_lower == "sources":
                show_sources = not show_sources
                status = "ON" if show_sources else "OFF"
                console.print(f"✅ Source display: {status}\n", style="green")
                continue

            # Generate response
            console.print()
            with console.status("[bold green]Thinking..."):
                start_time = time.time()

                result = generator.generate_chat_response(
                    message=message, include_sources=show_sources
                )

                elapsed = time.time() - start_time

            # Save to session
            session_manager.add_message(session_id, "user", message)
            session_manager.add_message(session_id, "assistant", result.answer)

            # Display response
            answer_style = "green" if result.metadata.get("rag_retrieval_triggered") else "white"
            rag_indicator = "🔍 " if result.metadata.get("rag_retrieval_triggered") else ""

            console.print(f"[bold cyan]Orion:[/bold cyan] {rag_indicator}", end="")
            console.print(result.answer, style=answer_style)
            console.print()

            # Show sources if requested and RAG was used
            if show_sources and result.sources:
                console.print(f"[dim]Sources ({len(result.sources)}):[/dim]")
                for source in result.sources[:3]:  # Show top 3
                    citation = source.get("citation", source.get("source_file", "Unknown"))
                    console.print(f"  • [{source['index']}] {citation}", style="dim")
                console.print()

            # Show timing in verbose mode
            if config.logging.verbose:
                console.print(f"[dim]⏱️  {elapsed:.2f}s[/dim]")
                console.print()

        except KeyboardInterrupt:
            console.print("\n\n👋 Goodbye!", style="bold cyan")
            break
        except Exception as e:
            console.print(f"\n❌ Error: {e}\n", style="bold red")
            import traceback

            traceback.print_exc()


# ========== INTERACTIVE MODE ==========
@app.command()
def interactive(
    gpu: bool = typer.Option(None, "--gpu/--no-gpu", help="Enable/disable GPU acceleration"),
):
    """
    Start interactive query mode.

    Type your questions and get instant answers from your knowledge base.
    Type 'exit' or 'quit' to exit.
    """
    print_banner()

    # Update GPU setting if specified
    config = get_config()
    if gpu is not None:
        config.gpu.enabled = gpu

    check_gpu_status()
    console.print()

    console.print("💬 [bold cyan]Interactive Mode[/bold cyan]", style="bold")
    console.print("   Type your questions below. Type 'exit' or 'quit' to exit.\n", style="dim")

    # Initialize retriever once
    with console.status("[bold green]Initializing retriever..."):
        try:
            retriever = OrionRetriever(config=config)
            console.print("✅ Ready!\n", style="green")
        except Exception as e:
            console.print(f"❌ Failed to initialize: {e}", style="bold red")
            raise typer.Exit(1)

    while True:
        try:
            # Get query
            query_text = console.input("[bold cyan]❯[/bold cyan] ")

            if not query_text.strip():
                continue

            if query_text.lower() in ["exit", "quit", "q"]:
                console.print("\n👋 Goodbye!", style="bold cyan")
                break

            # Search
            console.print()
            with console.status("[bold green]Searching..."):
                start_time = time.time()
                results = retriever.query(query_text, k=3, formatted=False)
                elapsed = time.time() - start_time

            if results:
                console.print(f"⚡ Found {len(results)} results in {elapsed:.2f}s\n", style="bold green")

                for i, result in enumerate(results, 1):
                    content = result.content[:300] + "..." if len(result.content) > 300 else result.content
                    source = result.metadata.get("file_name", "Unknown")

                    console.print(
                        f"[bold cyan][{i}][/bold cyan] [dim]Score: {result.score:.3f} | {source}[/dim]",
                    )
                    console.print(f"   {content}\n")
            else:
                console.print("No results found.\n", style="yellow")

        except KeyboardInterrupt:
            console.print("\n\n👋 Goodbye!", style="bold cyan")
            break
        except Exception as e:
            console.print(f"\n❌ Error: {e}\n", style="bold red")


# ========== MAIN ==========
@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-V", help="Show version"),
):
    """
    Orion - Local RAG Assistant

    Your personal knowledge base, powered by AI.
    """
    if version:
        config = get_config()
        console.print(f"Orion v{config.version}", style="bold cyan")
        raise typer.Exit(0)

    if ctx.invoked_subcommand is None:
        # Show help if no command
        print_banner()
        console.print("Use --help to see available commands\n", style="dim")


if __name__ == "__main__":
    app()
