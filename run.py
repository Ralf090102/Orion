#!/usr/bin/env python3
"""
Orion - Local RAG Assistant CLI

Main command-line interface for ingesting documents and querying your
personal knowledge base.
"""

import sys
import time
from pathlib import Path
from typing import Optional

import typer
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
from src.retrieval.retriever import OrionRetriever
from src.retrieval.vector_store import create_vector_store
from src.utilities.config import get_config
from src.utilities.utils import log_error, log_info

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


# ========== INGEST COMMAND ==========
@app.command()
def ingest(
    path: str = typer.Argument(..., help="Path to file or directory to ingest"),
    clear: bool = typer.Option(False, "--clear", "-c", help="Clear existing knowledge base first"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Watch for file changes and auto-ingest"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", "-r/-R", help="Recursively scan directories"),
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
            stats = ingest_documents(path, config=config, clear_existing=clear)

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
