#!/usr/bin/env python3
"""
Manual FastAPI Server Test - Start server and test endpoints
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    """Start FastAPI server manually for testing"""
    print("🚀 Starting Orion FastAPI Server")
    print("=" * 40)

    try:
        from backend.main import run_server

        print("Starting server on port 8003...")
        print("Visit: http://localhost:8003")
        print("Health: http://localhost:8003/health")
        print("Config: http://localhost:8003/api/system/config")
        print("\nPress Ctrl+C to stop\n")

        # Start server (this will block)
        run_server(host="127.0.0.1", port=8003, reload=False)

    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
