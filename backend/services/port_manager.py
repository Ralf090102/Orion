"""
Port management service for Orion FastAPI backend.
Handles port conflicts and finds available ports automatically.
"""

import socket
import logging
from typing import Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PortStatus:
    """Port availability status"""

    port: int
    available: bool
    error: Optional[str] = None


class PortManager:
    """Manages port allocation and conflict resolution for Orion backend"""

    DEFAULT_PORT = 8000
    PORT_RANGE_START = 8000
    PORT_RANGE_END = 8100
    RESERVED_PORTS = {8080, 8443, 8888}  # Common ports to avoid

    def __init__(self, preferred_port: int = DEFAULT_PORT):
        self.preferred_port = preferred_port
        self._tested_ports: List[PortStatus] = []

    def find_available_port(self, start_port: Optional[int] = None) -> int:
        """
        Find the first available port starting from start_port.

        Args:
            start_port: Port to start searching from (defaults to preferred_port)

        Returns:
            First available port number

        Raises:
            RuntimeError: If no available ports found in range
        """
        start_port = start_port or self.preferred_port

        # Clear previous test results
        self._tested_ports.clear()

        # First try the preferred port
        if start_port == self.preferred_port:
            port_status = self._test_port(self.preferred_port)
            if port_status.available:
                logger.info(f"Using preferred port {self.preferred_port}")
                return self.preferred_port
            else:
                logger.warning(f"Preferred port {self.preferred_port} not available: {port_status.error}")

        # Search for available port in range
        for port in range(start_port, self.PORT_RANGE_END):
            # Skip reserved ports
            if port in self.RESERVED_PORTS:
                continue

            port_status = self._test_port(port)
            if port_status.available:
                logger.info(f"Found available port {port}")
                return port

        # Log all tested ports for debugging
        available_ports = [p.port for p in self._tested_ports if p.available]
        unavailable_ports = [f"{p.port}({p.error})" for p in self._tested_ports if not p.available]

        logger.error(f"No available ports found in range {start_port}-{self.PORT_RANGE_END}")
        logger.debug(f"Available ports tested: {available_ports}")
        logger.debug(f"Unavailable ports: {unavailable_ports}")

        raise RuntimeError(
            f"No available ports found in range {start_port}-{self.PORT_RANGE_END}. "
            f"Please free up some ports or expand the port range."
        )

    def _test_port(self, port: int) -> PortStatus:
        """
        Test if a specific port is available for binding.

        Args:
            port: Port number to test

        Returns:
            PortStatus indicating availability and any error
        """
        try:
            # Test TCP binding
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("localhost", port))
                sock.listen(1)

                # Port is available
                status = PortStatus(port=port, available=True)
                self._tested_ports.append(status)
                return status

        except OSError as e:
            # Port is not available
            error_msg = str(e)
            if "Address already in use" in error_msg:
                error_msg = "already in use"
            elif "Permission denied" in error_msg:
                error_msg = "permission denied"

            status = PortStatus(port=port, available=False, error=error_msg)
            self._tested_ports.append(status)
            return status
        except Exception as e:
            # Unexpected error
            status = PortStatus(port=port, available=False, error=f"unexpected error: {e}")
            self._tested_ports.append(status)
            return status

    def is_port_available(self, port: int) -> bool:
        """
        Check if a specific port is available.

        Args:
            port: Port number to check

        Returns:
            True if port is available, False otherwise
        """
        return self._test_port(port).available

    def get_port_status(self, port: int) -> PortStatus:
        """
        Get detailed status for a specific port.

        Args:
            port: Port number to check

        Returns:
            PortStatus with availability and error details
        """
        return self._test_port(port)

    def get_tested_ports(self) -> List[PortStatus]:
        """
        Get list of all ports tested in the last find_available_port call.

        Returns:
            List of PortStatus objects from last search
        """
        return self._tested_ports.copy()

    def suggest_alternative_ports(self, count: int = 5) -> List[int]:
        """
        Suggest alternative ports that might be available.

        Args:
            count: Number of alternative ports to suggest

        Returns:
            List of suggested port numbers
        """
        suggestions = []

        # Start from preferred port + 1
        start_port = self.preferred_port + 1

        for port in range(start_port, self.PORT_RANGE_END):
            if port in self.RESERVED_PORTS:
                continue

            if self.is_port_available(port):
                suggestions.append(port)

            if len(suggestions) >= count:
                break

        return suggestions


def create_port_manager(preferred_port: int = PortManager.DEFAULT_PORT) -> PortManager:
    """
    Factory function to create a PortManager instance.

    Args:
        preferred_port: Preferred port number to use

    Returns:
        Configured PortManager instance
    """
    return PortManager(preferred_port=preferred_port)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    # Test port manager
    port_manager = create_port_manager()

    try:
        # Find an available port
        available_port = port_manager.find_available_port()
        print(f"Available port found: {available_port}")

        # Test specific port
        status = port_manager.get_port_status(8000)
        print(f"Port 8000 status: {'available' if status.available else f'unavailable ({status.error})'}")

        # Get alternative suggestions
        alternatives = port_manager.suggest_alternative_ports(3)
        print(f"Alternative ports: {alternatives}")

    except RuntimeError as e:
        print(f"Error: {e}")
