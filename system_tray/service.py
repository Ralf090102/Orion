"""
System Tray Service for Orion RAG
Provides background service functionality
"""

import pystray
import webbrowser
import subprocess
import sys
from PIL import Image, ImageDraw

from core.utils.orion_utils import log_info, log_error


class OrionTrayService:
    def __init__(self):
        self.icon = None
        self.api_process = None
        self.frontend_process = None

    def create_icon_image(self):
        """Create a simple icon for the system tray"""
        # Create a simple colored circle as icon
        image = Image.new("RGB", (64, 64), color="blue")
        draw = ImageDraw.Draw(image)
        draw.ellipse([8, 8, 56, 56], fill="white", outline="blue", width=2)
        draw.text((20, 25), "O", fill="blue")
        return image

    def create_menu(self):
        """Create the system tray menu"""
        return pystray.Menu(
            pystray.MenuItem("Open Orion", self.open_app),
            pystray.MenuItem("Start Services", self.start_services),
            pystray.MenuItem("Stop Services", self.stop_services),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Settings", self.open_settings),
            pystray.MenuItem("Logs", self.open_logs),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self.quit_app),
        )

    def start_services(self, icon=None, item=None):
        """Start the backend and frontend services"""
        try:
            log_info("Starting Orion services...")

            # Start FastAPI backend
            if not self.api_process or self.api_process.poll() is not None:
                self.api_process = subprocess.Popen(
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
                log_info("Backend API started on port 8000")

            # TODO: Start Svelte frontend
            # self.frontend_process = subprocess.Popen([...])

            if self.icon:
                self.icon.title = "Orion RAG (Running)"

        except Exception as e:
            log_error(f"Failed to start services: {e}")

    def stop_services(self, icon=None, item=None):
        """Stop the backend and frontend services"""
        try:
            log_info("Stopping Orion services...")

            if self.api_process and self.api_process.poll() is None:
                self.api_process.terminate()
                self.api_process = None
                log_info("Backend API stopped")

            if self.frontend_process and self.frontend_process.poll() is None:
                self.frontend_process.terminate()
                self.frontend_process = None
                log_info("Frontend stopped")

            if self.icon:
                self.icon.title = "Orion RAG (Stopped)"

        except Exception as e:
            log_error(f"Failed to stop services: {e}")

    def open_app(self, icon=None, item=None):
        """Open the main application"""
        # Start services if not running
        if not self.api_process or self.api_process.poll() is not None:
            self.start_services()

        # Open web interface
        webbrowser.open("http://localhost:3000")  # Will be Svelte app

    def open_settings(self, icon=None, item=None):
        """Open settings page"""
        webbrowser.open("http://localhost:8000/docs")  # FastAPI docs for now

    def open_logs(self, icon=None, item=None):
        """Open logs viewer"""
        # TODO: Implement proper logs viewer
        webbrowser.open("http://localhost:8000/api/system/logs")

    def quit_app(self, icon=None, item=None):
        """Quit the application"""
        log_info("Shutting down Orion...")
        self.stop_services()
        if self.icon:
            self.icon.stop()

    def run(self):
        """Start the system tray service"""
        try:
            log_info("Starting Orion system tray service...")

            # Create icon
            image = self.create_icon_image()
            menu = self.create_menu()

            self.icon = pystray.Icon("Orion RAG", image, "Orion RAG", menu)

            # Start services automatically
            self.start_services()

            # Run the icon (blocking)
            self.icon.run()

        except Exception as e:
            log_error(f"System tray service failed: {e}")
            raise


def main():
    """Main entry point for system tray service"""
    service = OrionTrayService()
    service.run()


if __name__ == "__main__":
    main()
