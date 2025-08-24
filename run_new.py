"""
Enhanced interactive runner for Orion RAG with improved UX.
Now using the new modular architecture!
"""

import sys
import subprocess
import asyncio

# Import temp cleanup early to prevent exit errors

from core.utils.config import Config
from core.utils.orion_utils import (
    log_info,
    log_success,
    log_error,
    log_warning,
)
from core.rag.llm import check_ollama_connection


def print_banner():
    """Display the Orion banner with new architecture info"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    ██████╗ ██████╗ ██╗ ██████╗ ███╗   ██╗    ██████╗  █████╗  ██████╗      ║
║   ██╔═══██╗██╔══██╗██║██╔═══██╗████╗  ██║    ██╔══██╗██╔══██╗██╔════╝      ║
║   ██║   ██║██████╔╝██║██║   ██║██╔██╗ ██║    ██████╔╝███████║██║  ███╗     ║
║   ██║   ██║██╔══██╗██║██║   ██║██║╚██╗██║    ██╔══██╗██╔══██║██║   ██║     ║
║   ╚██████╔╝██║  ██║██║╚██████╔╝██║ ╚████║    ██║  ██║██║  ██║╚██████╔╝     ║
║    ╚═════╝ ╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝    ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝      ║
║                                                                              ║
║                       🌟 Personal Knowledge RAG 🌟                          ║
║                     Now with Modular Architecture!                          ║
║                                                                              ║
║  📁 Core: RAG Engine    🌐 Backend: FastAPI    🎨 Frontend: Svelte          ║
║  🖥️  Desktop: Tauri     🔔 System Tray        📊 Performance Optimized      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_architecture_status():
    """Show the current architecture status"""
    print("\n🏗️ Architecture Status:")
    print("  ✅ Core Engine (RAG Pipeline)")
    print("  🚧 FastAPI Backend (In Development)")
    print("  🚧 Svelte Frontend (Planned)")
    print("  🚧 System Tray Service (Ready)")
    print("  🚧 Tauri Desktop App (Planned)")
    print()


def main_menu():
    """Enhanced main menu with architecture options"""
    print("\n📋 Main Menu:")
    print("  [1] 📚 Ingest Documents (Async)")
    print("  [2] 🔄 Incremental Update (Smart)")
    print("  [3] 🔍 Query Knowledge Base")
    print("  [4] 💬 Interactive Chat Mode")
    print("  [5] 📊 Performance Demo")
    print()
    print("🏗️ New Architecture:")
    print("  [6] 🚀 Start FastAPI Backend")
    print("  [7] 🔔 Start System Tray Service")
    print("  [8] 🌐 Open API Documentation")
    print()
    print("⚙️ System:")
    print("  [9] 🧠 List Available Models")
    print("  [10] 📈 Cache Statistics")
    print("  [11] 🔧 Clear Cache")
    print("  [12] ⚙️ Settings")
    print("  [0] 👋 Exit")


async def start_backend_server():
    """Start the FastAPI backend server"""
    try:
        log_info("🚀 Starting FastAPI backend server...")

        # Check if uvicorn is available
        try:
            import importlib.util

            if importlib.util.find_spec("uvicorn") is None:
                raise ImportError
        except ImportError:
            log_error("FastAPI/Uvicorn not installed. Please install with:")
            log_info("pip install fastapi uvicorn[standard]")
            return

        # Start the server
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "backend.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--reload",
            ]
        )

        log_success("✅ Backend server started!")
        log_info("📖 API Documentation: http://localhost:8000/docs")
        log_info("🏥 Health Check: http://localhost:8000/health")
        log_info("Press Ctrl+C to stop the server")

        # Wait for process
        try:
            process.wait()
        except KeyboardInterrupt:
            log_info("🛑 Stopping backend server...")
            process.terminate()
            process.wait()
            log_success("✅ Backend server stopped")

    except Exception as e:
        log_error(f"Failed to start backend: {e}")


def start_system_tray():
    """Start the system tray service"""
    try:
        log_info("🔔 Starting system tray service...")

        # Check if pystray is available
        try:
            import importlib.util

            if importlib.util.find_spec("pystray") is None:
                raise ImportError
        except ImportError:
            log_error("pystray not installed. Please install with:")
            log_info("pip install pystray")
            return

        # Import and start the service
        from system_tray.service import OrionTrayService

        log_info("✨ System tray service starting...")
        service = OrionTrayService()
        service.run()

    except Exception as e:
        log_error(f"Failed to start system tray: {e}")


def open_api_docs():
    """Open the FastAPI documentation"""
    import webbrowser

    webbrowser.open("http://localhost:8000/docs")
    log_info("📖 Opening API documentation in browser...")


async def main():
    """Enhanced main function with new architecture support"""
    config = Config()

    print_banner()
    print_architecture_status()

    # Check Ollama connection
    if not await check_ollama_connection():
        log_warning("⚠️ Ollama is not running or not accessible")
        log_info("Please start Ollama and make sure it's running on the default port")

        choice = input("\nContinue anyway? (y/N): ").lower().strip()
        if choice != "y":
            log_info("👋 Goodbye!")
            return

    while True:
        try:
            main_menu()
            choice = input("\n🎯 Enter your choice: ").strip()

            if choice == "0":
                log_success("👋 Thank you for using Orion!")
                break

            elif choice == "1":
                # Async ingestion (existing functionality)
                await handle_async_ingestion()

            elif choice == "2":
                # Incremental update (existing functionality)
                await handle_incremental_update()

            elif choice == "3":
                # Query knowledgebase (existing functionality)
                await handle_query()

            elif choice == "4":
                # Chat mode (existing functionality)
                await handle_chat_mode()

            elif choice == "5":
                # Performance demo (existing functionality)
                await handle_performance_demo()

            elif choice == "6":
                # Start FastAPI backend
                await start_backend_server()

            elif choice == "7":
                # Start system tray service
                start_system_tray()

            elif choice == "8":
                # Open API docs
                open_api_docs()

            elif choice == "9":
                # List models (existing functionality)
                await handle_list_models()

            elif choice == "10":
                # Cache stats (existing functionality)
                handle_cache_stats()

            elif choice == "11":
                # Clear cache (existing functionality)
                handle_clear_cache()

            elif choice == "12":
                # Settings (existing functionality)
                handle_settings(config)

            else:
                log_warning("⚠️ Invalid choice. Please try again.")

        except KeyboardInterrupt:
            log_info("\n🛑 Operation cancelled by user")
        except Exception as e:
            log_error(f"❌ An error occurred: {e}")


# TODO: Import existing handler functions from old run.py
# For now, these are placeholders


async def handle_async_ingestion():
    log_info("🚧 Async ingestion - migrating to new architecture...")


async def handle_incremental_update():
    log_info("🚧 Incremental update - migrating to new architecture...")


async def handle_query():
    log_info("🚧 Query handler - migrating to new architecture...")


async def handle_chat_mode():
    log_info("🚧 Chat mode - migrating to new architecture...")


async def handle_performance_demo():
    log_info("🚧 Performance demo - migrating to new architecture...")


async def handle_list_models():
    log_info("🚧 List models - migrating to new architecture...")


def handle_cache_stats():
    log_info("🚧 Cache stats - migrating to new architecture...")


def handle_clear_cache():
    log_info("🚧 Clear cache - migrating to new architecture...")


def handle_settings(config):
    log_info("🚧 Settings - migrating to new architecture...")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log_info("\n👋 Goodbye!")
    except Exception as e:
        log_error(f"💥 Fatal error: {e}")
        sys.exit(1)
