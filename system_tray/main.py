"""
System tray service for Orion RAG Assistant.
Provides system tray integration with profile switching and server management.
"""

import sys
import threading
import webbrowser
import logging
from typing import TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    import pystray

try:
    import pystray
    from pystray import MenuItem as Item
    from PIL import Image
except ImportError:
    print("Error: Required packages not installed.")
    print("Please install: pip install pystray pillow")
    sys.exit(1)

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.services import get_config_service, create_port_manager, get_gpu_manager
from backend.models.system import ProfileInfo

logger = logging.getLogger(__name__)


class OrionSystemTray:
    """System tray service for Orion RAG Assistant"""

    def __init__(self):
        self.config_service = get_config_service()
        self.port_manager = create_port_manager()
        self.gpu_manager = get_gpu_manager()

        # Server state
        self.server_thread = None
        self.server_port = None
        self.is_server_running = False

        # System tray
        self.icon = None
        self.is_running = False

        # GPU capabilities
        self.gpu_capabilities = None
        self._detect_gpu_capabilities()

        # Create system tray icon
        self._create_icon()

    def _detect_gpu_capabilities(self) -> None:
        """Detect GPU capabilities on startup"""
        try:
            self.gpu_capabilities = self.gpu_manager.detect_gpu_capabilities()
            if self.gpu_capabilities.has_cuda:
                logger.info(f"GPU acceleration available: {len(self.gpu_capabilities.gpus)} CUDA devices")
            else:
                logger.info("GPU acceleration not available (CPU-only mode)")
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
            self.gpu_capabilities = None

    def _create_icon(self) -> None:
        """Create system tray icon and menu"""
        # Create a simple icon (you can replace this with a proper .ico file)
        icon_image = self._create_default_icon()

        # Build menu
        menu = pystray.Menu(
            Item("Orion RAG Assistant", self._on_click_title, enabled=False),
            pystray.Menu.SEPARATOR,
            Item("Open Web Interface", self._open_web_interface, enabled=lambda _: self.is_server_running),
            pystray.Menu.SEPARATOR,
            self._create_profiles_submenu(),
            pystray.Menu.SEPARATOR,
            self._create_system_info_submenu(),
            pystray.Menu.SEPARATOR,
            Item("Start Server", self._start_server, enabled=lambda _: not self.is_server_running),
            Item("Stop Server", self._stop_server, enabled=lambda _: self.is_server_running),
            Item("Restart Server", self._restart_server, enabled=lambda _: self.is_server_running),
            pystray.Menu.SEPARATOR,
            Item("Settings", self._open_settings),
            Item("About", self._show_about),
            pystray.Menu.SEPARATOR,
            Item("Exit", self._exit_application),
        )

        # Create system tray icon
        self.icon = pystray.Icon(name="Orion RAG Assistant", icon=icon_image, title="Orion RAG Assistant", menu=menu)

    def _create_default_icon(self) -> Image.Image:
        """Create a default icon for the system tray"""
        # Create a simple 32x32 icon with "O" text
        from PIL import Image, ImageDraw, ImageFont

        # Create image
        img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Draw circle background
        draw.ellipse([2, 2, 30, 30], fill=(54, 69, 79), outline=(255, 255, 255))

        # Draw "O" text
        try:
            # Try to use default font
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            # Fallback to default font
            font = ImageFont.load_default()

        # Get text size and center it
        bbox = draw.textbbox((0, 0), "O", font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x = (32 - text_width) // 2
        y = (32 - text_height) // 2 - 2

        draw.text((x, y), "O", fill=(255, 255, 255), font=font)

        return img

    def _create_profiles_submenu(self) -> Item:
        """Create profiles submenu"""
        profiles = self.config_service.list_profiles()
        active_profile = self.config_service.get_active_profile()

        profile_items = []
        for profile in profiles:
            is_active = active_profile and profile.name == active_profile.name
            profile_items.append(
                Item(
                    f"{'✓ ' if is_active else ''}{profile.display_name}",
                    lambda _, p=profile: self._switch_profile(p),
                    enabled=not is_active,
                )
            )

        if not profile_items:
            profile_items.append(Item("No profiles available", lambda _: None, enabled=False))

        return Item("Switch Profile", pystray.Menu(*profile_items))

    def _create_system_info_submenu(self) -> Item:
        """Create system information submenu"""
        system_config = self.config_service.get_system_config()

        info_items = [
            Item(f"CPU Limit: {system_config.max_cpu_usage_percent}%", lambda _: None, enabled=False),
            Item(f"Memory Limit: {system_config.max_memory_usage_mb}MB", lambda _: None, enabled=False),
        ]

        # GPU information
        if self.gpu_capabilities and self.gpu_capabilities.has_cuda:
            gpu_count = len(self.gpu_capabilities.gpus)
            total_memory = self.gpu_capabilities.total_gpu_memory_mb
            info_items.extend(
                [
                    pystray.Menu.SEPARATOR,
                    Item(f"GPU: {gpu_count} CUDA device(s)", lambda _: None, enabled=False),
                    Item(f"GPU Memory: {total_memory}MB total", lambda _: None, enabled=False),
                    Item(
                        f'GPU Acceleration: {"Enabled" if system_config.enable_gpu_acceleration else "Disabled"}',
                        lambda _: None,
                        enabled=False,
                    ),
                ]
            )
        else:
            info_items.extend([pystray.Menu.SEPARATOR, Item("GPU: Not available (CPU-only)", lambda _: None, enabled=False)])

        return Item("System Info", pystray.Menu(*info_items))

    def _on_click_title(self, icon, item) -> None:
        """Handle title click (disabled)"""
        pass

    def _open_web_interface(self, icon, item) -> None:
        """Open web interface in default browser"""
        if self.server_port:
            url = f"http://localhost:{self.server_port}"
            webbrowser.open(url)
            logger.info(f"Opened web interface: {url}")
        else:
            logger.warning("Server not running, cannot open web interface")

    def _switch_profile(self, profile: ProfileInfo) -> None:
        """Switch to a different profile"""
        try:
            self.config_service.activate_profile(profile.name)
            logger.info(f"Switched to profile: {profile.name}")

            # Update menu to reflect new active profile
            self._update_menu()

            # TODO: Notify web interface of profile change
            # This could be done via WebSocket or server-sent events

        except Exception as e:
            logger.error(f"Failed to switch profile: {e}")

    def _start_server(self, icon, item) -> None:
        """Start the FastAPI server"""
        if self.is_server_running:
            logger.warning("Server is already running")
            return

        try:
            # Get system configuration
            system_config = self.config_service.get_system_config()
            
            # Find available port
            self.server_port = self.port_manager.find_available_port(system_config.port)
            
            if self.server_port != system_config.port:
                logger.info(f"Requested port {system_config.port} not available, using {self.server_port}")

            # Start server in background thread
            self.server_thread = threading.Thread(target=self._run_server, args=(self.server_port,), daemon=True)
            self.server_thread.start()

            # Give server a moment to start
            import time
            time.sleep(2)
            
            # Verify server is running
            if self._check_server_health():
                self.is_server_running = True
                self._update_menu()
                logger.info(f"✅ Server started successfully on port {self.server_port}")
            else:
                logger.error("❌ Server failed to start properly")
                self.is_server_running = False
                self.server_port = None

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            self.is_server_running = False
            self.server_port = None

    def _stop_server(self, icon, item) -> None:
        """Stop the FastAPI server"""
        if not self.is_server_running:
            logger.warning("Server is not running")
            return

        try:
            # TODO: Implement graceful server shutdown
            # This would require server instance management

            self.is_server_running = False
            self.server_port = None
            self.server_thread = None
            self._update_menu()

            logger.info("Server stopped")

        except Exception as e:
            logger.error(f"Failed to stop server: {e}")

    def _restart_server(self, icon, item) -> None:
        """Restart the FastAPI server"""
        logger.info("Restarting server...")
        self._stop_server(icon, item)

        # Small delay to ensure clean shutdown
        import time

        time.sleep(1)

        self._start_server(icon, item)

    def _run_server(self, port: int) -> None:
        """Run FastAPI server in background thread"""
        try:
            import uvicorn
            from backend.main import create_app
            
            logger.info(f"Starting FastAPI server on port {port}")
            
            # Create FastAPI app
            app = create_app()
            
            # Run server with uvicorn
            uvicorn.run(
                app,
                host="127.0.0.1",
                port=port,
                log_level="warning",  # Reduce log noise in system tray
                access_log=False      # Disable access logging for system tray
            )
            
        except Exception as e:
            logger.error(f"Server error: {e}")
            self.is_server_running = False
    
    def _check_server_health(self) -> bool:
        """Check if server is responding to health checks"""
        if not self.server_port:
            return False
            
        try:
            import requests
            response = requests.get(f"http://127.0.0.1:{self.server_port}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False

    def _open_settings(self, icon, item) -> None:
        """Open settings dialog or web interface settings page"""
        if self.is_server_running and self.server_port:
            url = f"http://localhost:{self.server_port}/settings"
            webbrowser.open(url)
        else:
            logger.info("Settings: Server not running, starting server first...")
            self._start_server(icon, item)

            # Wait a moment for server to start
            import time

            time.sleep(2)

            if self.is_server_running and self.server_port:
                url = f"http://localhost:{self.server_port}/settings"
                webbrowser.open(url)

    def _show_about(self, icon, item) -> None:
        """Show about dialog"""
        # TODO: Implement proper about dialog
        # For now, just log info
        logger.info("Orion RAG Assistant - Personal AI Assistant")
        logger.info("Version: 1.0.0-alpha")

    def _exit_application(self, icon, item) -> None:
        """Exit the application"""
        logger.info("Exiting Orion system tray...")

        # Stop server if running
        if self.is_server_running:
            self._stop_server(icon, item)

        # Stop system tray
        self.is_running = False
        if self.icon:
            self.icon.stop()

    def _update_menu(self) -> None:
        """Update system tray menu (recreate menu with current state)"""
        if self.icon:
            # Update menu by recreating it
            self._create_icon()
            # Note: pystray doesn't support dynamic menu updates well
            # A full restart might be needed for complex menu changes

    def run(self) -> None:
        """Start the system tray service"""
        if not self.icon:
            logger.error("System tray icon not created")
            return

        logger.info("Starting Orion system tray service...")
        self.is_running = True

        # Check if auto-start is enabled
        system_config = self.config_service.get_system_config()
        if system_config.auto_start:
            logger.info("Auto-start enabled, starting server...")
            self._start_server(None, None)

        # Run system tray (this blocks until icon.stop() is called)
        try:
            self.icon.run()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        finally:
            self.is_running = False
            logger.info("System tray service stopped")


def main():
    """Main entry point for system tray service"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("orion_tray.log"), logging.StreamHandler(sys.stdout)],
    )

    logger.info("Initializing Orion System Tray Service")

    try:
        # Create and run system tray
        tray_service = OrionSystemTray()
        tray_service.run()

    except Exception as e:
        logger.error(f"System tray service error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
