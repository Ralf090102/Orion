"""
Orion Project Runner - Local Mode
Run ingestion or querying from the command line.
"""

import argparse
import sys
from app.utils import log_info, log_success, log_error, log_warning
from app.utils import validate_path_exists, validate_nonempty_string
from app.ingest import build_vectorstore
from app.query import query_knowledgebase
from app.llm import get_available_models, check_ollama_connection
from app.config import Config


def validate_args(args):
    """
    Validate command line arguments for different commands.

    Args:
        args: Parsed command line arguments from argparse

    Returns:
        True if arguments are valid, False otherwise
    """
    if args.command == "ingest":
        import os

        if not os.path.exists(args.path):
            log_error(f"Document folder not found: {args.path}")
            return False

    elif args.command == "query":
        if not args.question.strip():
            log_error("Question cannot be empty")
            return False

    return True


def handle_ingest(args: argparse.Namespace, cfg: Config) -> bool:
    """
    Handle document ingestion command.

    Args:
        args: Parsed command line arguments containing ingestion parameters
        cfg: Configuration object with default settings

    Returns:
        True if ingestion was successful, False otherwise
    """
    log_info("Starting document ingestion...")

    # Check Ollama connection before starting
    if not check_ollama_connection():
        log_error("Ollama service is not running. Please start it with 'ollama serve'")
        return False

    try:
        # Use user-specific persist path
        user_persist_path = args.persist or cfg.user_persist_path

        vectorstore = build_vectorstore(
            folder_path=args.path,
            persist_path=user_persist_path,
            chunk_size=args.chunk_size or cfg.chunk_size,
            chunk_overlap=args.chunk_overlap or cfg.chunk_overlap,
        )

        if vectorstore:
            log_success(
                f"Document ingestion completed successfully for user '{cfg.user_id}'!"
            )
            log_info(f"Vectorstore saved to: {user_persist_path}")
            return True
        else:
            log_error("Document ingestion failed")
            return False

    except Exception as e:
        log_error(f"Ingestion failed with error: {e}")
        return False


def handle_query(args: argparse.Namespace, cfg: Config) -> bool:
    """
    Handle query command.

    Args:
        args: Parsed command line arguments containing query parameters
        cfg: Configuration object with default settings

    Returns:
        True if query was successful, False otherwise
    """
    log_info(
        f"Querying knowledgebase for user '{cfg.user_id}' with question: {args.question}"
    )

    try:
        # Use user-specific persist path
        user_persist_path = args.persist or cfg.user_persist_path

        result = query_knowledgebase(
            query=args.question,
            persist_path=user_persist_path,
            model=args.model or cfg.llm_model,
            k=args.k or cfg.retrieval_k,
            use_query_enhancement=not getattr(args, "no_enhance", False),
        )

        # Handle both old string format and new dict format
        if isinstance(result, dict):
            answer = result.get("answer", "")
            sources = result.get("sources", [])
        else:
            answer = result
            sources = []

        if answer.startswith("[Error:") or answer.startswith("[Warning:"):
            log_error(f"Query failed: {answer}")
            return False
        else:
            log_success("Enhanced query processing completed successfully!")
            print(f"\n{'='*60}")
            print(f"QUESTION: {args.question}")
            print(f"{'='*60}")
            print(f"ANSWER:\n{answer}")

            # Show sources if available
            if sources:
                print(f"{'='*60}")
                print("SOURCES:")
                for i, src in enumerate(sources, 1):
                    source_path = src.get("source", "Unknown")
                    page = src.get("page")
                    page_info = f" (page {page})" if page else ""
                    print(f"{i}. {source_path}{page_info}")
            print(f"{'='*60}\n")
            return True

    except Exception as e:
        log_error(f"Query failed with error: {e}")
        return False


def list_models():
    """
    List available Ollama models and display them to the user.
    Fetches models from Ollama service and prints them in a user-friendly format.
    """
    models = get_available_models()
    if models:
        log_info("Available Ollama models:")
        for model in models:
            print(f"  • {model}")
    else:
        log_warning("No models found or Ollama not accessible")


def main() -> None:
    cfg = Config.from_env()

    parser = argparse.ArgumentParser(
        description="Orion - Local RAG Pipeline",
        epilog="Examples:\n"
        "  %(prog)s ingest --path ./documents --user alice\n"
        "  %(prog)s query --question 'What is the main topic?' --user alice\n"
        "  %(prog)s models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global arguments
    parser.add_argument(
        "--user",
        type=str,
        help="User ID for workspace isolation (default: from config/env)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest documents into vectorstore"
    )
    ingest_parser.add_argument(
        "--path", type=str, required=True, help="Path to folder containing documents"
    )
    ingest_parser.add_argument("--persist", type=str, help="Path to save vectorstore")
    ingest_parser.add_argument(
        "--chunk-size", type=int, help="Text chunk size for splitting"
    )
    ingest_parser.add_argument(
        "--chunk-overlap", type=int, help="Overlap between chunks"
    )

    # Query command
    query_parser = subparsers.add_parser("query", help="Query vectorstore using LLM")
    query_parser.add_argument(
        "--question", type=str, required=True, help="Your question"
    )
    query_parser.add_argument("--persist", type=str, help="Path to load vectorstore")
    query_parser.add_argument("--model", type=str, help="Ollama model to use")
    query_parser.add_argument(
        "-k", type=int, help="Number of relevant documents to retrieve"
    )
    query_parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Disable query enhancement (use basic retrieval only)",
    )

    # Models command
    subparsers.add_parser("models", help="List available Ollama models")

    # Parse arguments
    args = parser.parse_args()

    # Update config with user argument if provided
    if hasattr(args, "user") and args.user:
        cfg.user_id = args.user
        log_info(f"Using user workspace: {cfg.user_id}")

    # Handle no command
    if not args.command:
        parser.print_help()
        log_error("No command provided. Use --help for usage details.")
        sys.exit(1)

    # Validation
    if args.command == "ingest" and not validate_path_exists(args.path):
        sys.exit(1)
    if args.command == "query" and not validate_nonempty_string(
        args.question, "Question cannot be empty"
    ):
        sys.exit(1)

    # Execute commands
    success = (
        handle_ingest(args, cfg)
        if args.command == "ingest"
        else (
            handle_query(args, cfg)
            if args.command == "query"
            else list_models() or True if args.command == "models" else False
        )
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
