"""
Configuration management service for Orion.
Handles system configuration, profile settings, and auto-indexing preferences.
"""

import json
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import logging

from backend.models.system import SystemConfig, AutoIndexingConfig, AutoIndexMode, ProfileInfo, SystemConfigUpdateRequest

logger = logging.getLogger(__name__)


class ConfigurationService:
    """Manages Orion system configuration and persistence"""

    DEFAULT_CONFIG_DIR = Path("./orion-config")
    DEFAULT_DATA_DIR = Path("./orion-data")

    CONFIG_FILES = {
        "system": "system_config.json",
        "profiles": "profiles_config.json",
        "auto_indexing": "auto_indexing_config.json",
    }

    def __init__(self, config_dir: Optional[Path] = None, data_dir: Optional[Path] = None):
        """
        Initialize configuration service.

        Args:
            config_dir: Directory for configuration files
            data_dir: Directory for data storage
        """
        self.config_dir = Path(config_dir) if config_dir else self.DEFAULT_CONFIG_DIR
        self.data_dir = Path(data_dir) if data_dir else self.DEFAULT_DATA_DIR

        # Ensure directories exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize configuration cache
        self._system_config: Optional[SystemConfig] = None
        self._profiles_config: Dict[str, ProfileInfo] = {}

        # Load existing configurations
        self._load_configurations()

    def get_system_config(self) -> SystemConfig:
        """
        Get current system configuration.

        Returns:
            Current SystemConfig instance
        """
        if self._system_config is None:
            self._system_config = self._create_default_system_config()
            self.save_system_config(self._system_config)

        return self._system_config

    def update_system_config(self, update_request: SystemConfigUpdateRequest) -> SystemConfig:
        """
        Update system configuration with provided changes.

        Args:
            update_request: Configuration update request

        Returns:
            Updated SystemConfig instance
        """
        current_config = self.get_system_config()

        # Create updated config dict
        config_dict = current_config.model_dump()

        # Apply updates
        update_dict = update_request.model_dump(exclude_unset=True)
        config_dict.update(update_dict)

        # Create new config instance
        updated_config = SystemConfig(**config_dict)

        # Save and cache
        self.save_system_config(updated_config)
        self._system_config = updated_config

        logger.info(f"System configuration updated: {list(update_dict.keys())}")
        return updated_config

    def save_system_config(self, config: SystemConfig) -> None:
        """
        Save system configuration to disk.

        Args:
            config: SystemConfig to save
        """
        config_file = self.config_dir / self.CONFIG_FILES["system"]

        try:
            # Convert to dict and add metadata
            config_data = {"config": config.model_dump(), "last_updated": datetime.now().isoformat(), "version": "1.0"}

            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=2, default=str)

            logger.debug(f"System configuration saved to {config_file}")

        except Exception as e:
            logger.error(f"Failed to save system configuration: {e}")
            raise

    def get_profile_info(self, profile_name: str) -> Optional[ProfileInfo]:
        """
        Get information about a specific profile.

        Args:
            profile_name: Name of the profile

        Returns:
            ProfileInfo if found, None otherwise
        """
        return self._profiles_config.get(profile_name)

    def list_profiles(self) -> List[ProfileInfo]:
        """
        List all available profiles.

        Returns:
            List of ProfileInfo objects
        """
        return list(self._profiles_config.values())

    def create_profile(self, name: str, display_name: str) -> ProfileInfo:
        """
        Create a new user profile.

        Args:
            name: Profile identifier name
            display_name: Human-readable profile name

        Returns:
            Created ProfileInfo instance
        """
        if name in self._profiles_config:
            raise ValueError(f"Profile '{name}' already exists")

        # Create profile directory structure
        profile_dir = self.data_dir / "profiles" / name
        profile_dir.mkdir(parents=True, exist_ok=True)

        vectorstore_dir = profile_dir / "vectorstore"
        documents_dir = profile_dir / "documents"

        vectorstore_dir.mkdir(exist_ok=True)
        documents_dir.mkdir(exist_ok=True)

        # Create profile info
        profile_info = ProfileInfo(
            name=name,
            display_name=display_name,
            vectorstore_path=str(vectorstore_dir),
            document_count=0,
            total_size_mb=0.0,
            created_at=datetime.now(),
            is_active=len(self._profiles_config) == 0,  # First profile is active
        )

        # Add to cache and save
        self._profiles_config[name] = profile_info
        self._save_profiles_config()

        logger.info(f"Created profile '{name}' ({display_name})")
        return profile_info

    def activate_profile(self, profile_name: str) -> ProfileInfo:
        """
        Activate a specific profile.

        Args:
            profile_name: Name of profile to activate

        Returns:
            Activated ProfileInfo instance

        Raises:
            ValueError: If profile doesn't exist
        """
        if profile_name not in self._profiles_config:
            raise ValueError(f"Profile '{profile_name}' does not exist")

        # Deactivate all profiles
        for profile in self._profiles_config.values():
            profile.is_active = False

        # Activate target profile
        target_profile = self._profiles_config[profile_name]
        target_profile.is_active = True

        # Update system config
        system_config = self.get_system_config()
        system_config.active_profile = profile_name
        self.save_system_config(system_config)

        # Save profiles config
        self._save_profiles_config()

        logger.info(f"Activated profile '{profile_name}'")
        return target_profile

    def get_active_profile(self) -> Optional[ProfileInfo]:
        """
        Get currently active profile.

        Returns:
            Active ProfileInfo if found, None otherwise
        """
        for profile in self._profiles_config.values():
            if profile.is_active:
                return profile
        return None

    def update_profile_stats(self, profile_name: str, document_count: int, total_size_mb: float) -> None:
        """
        Update profile statistics.

        Args:
            profile_name: Name of profile to update
            document_count: Number of documents
            total_size_mb: Total size in MB
        """
        if profile_name in self._profiles_config:
            profile = self._profiles_config[profile_name]
            profile.document_count = document_count
            profile.total_size_mb = total_size_mb
            profile.last_indexed = datetime.now()

            self._save_profiles_config()
            logger.debug(f"Updated stats for profile '{profile_name}': {document_count} docs, {total_size_mb:.1f}MB")

    def get_auto_indexing_config(self) -> AutoIndexingConfig:
        """
        Get auto-indexing configuration.

        Returns:
            AutoIndexingConfig instance
        """
        system_config = self.get_system_config()
        return system_config.auto_indexing

    def update_auto_indexing_config(self, config: AutoIndexingConfig) -> AutoIndexingConfig:
        """
        Update auto-indexing configuration.

        Args:
            config: New AutoIndexingConfig

        Returns:
            Updated AutoIndexingConfig
        """
        system_config = self.get_system_config()
        system_config.auto_indexing = config
        self.save_system_config(system_config)

        logger.info("Auto-indexing configuration updated")
        return config

    def _load_configurations(self) -> None:
        """Load all configuration files from disk."""
        # Load system config
        self._load_system_config()

        # Load profiles config
        self._load_profiles_config()

    def _load_system_config(self) -> None:
        """Load system configuration from disk."""
        config_file = self.config_dir / self.CONFIG_FILES["system"]

        if not config_file.exists():
            logger.info("System configuration file not found, will create default")
            return

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            config_data = data.get("config", {})
            self._system_config = SystemConfig(**config_data)

            logger.debug(f"Loaded system configuration from {config_file}")

        except Exception as e:
            logger.error(f"Failed to load system configuration: {e}")
            logger.warning("Using default system configuration")

    def _load_profiles_config(self) -> None:
        """Load profiles configuration from disk."""
        config_file = self.config_dir / self.CONFIG_FILES["profiles"]

        if not config_file.exists():
            logger.info("Profiles configuration file not found, will create default profile")
            self._create_default_profile()
            return

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            profiles_data = data.get("profiles", {})
            for name, profile_data in profiles_data.items():
                # Convert datetime strings back to datetime objects
                if "created_at" in profile_data:
                    profile_data["created_at"] = datetime.fromisoformat(profile_data["created_at"])
                if "last_indexed" in profile_data and profile_data["last_indexed"]:
                    profile_data["last_indexed"] = datetime.fromisoformat(profile_data["last_indexed"])

                self._profiles_config[name] = ProfileInfo(**profile_data)

            logger.debug(f"Loaded {len(self._profiles_config)} profiles from {config_file}")

        except Exception as e:
            logger.error(f"Failed to load profiles configuration: {e}")
            logger.warning("Creating default profile")
            self._create_default_profile()

    def _save_profiles_config(self) -> None:
        """Save profiles configuration to disk."""
        config_file = self.config_dir / self.CONFIG_FILES["profiles"]

        try:
            # Convert profiles to dict
            profiles_data = {}
            for name, profile in self._profiles_config.items():
                profiles_data[name] = profile.model_dump()

            data = {"profiles": profiles_data, "last_updated": datetime.now().isoformat(), "version": "1.0"}

            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"Saved profiles configuration to {config_file}")

        except Exception as e:
            logger.error(f"Failed to save profiles configuration: {e}")
            raise

    def _create_default_system_config(self) -> SystemConfig:
        """Create default system configuration."""
        # Create default auto-indexing config
        auto_indexing = AutoIndexingConfig(
            mode=AutoIndexMode.SMART_DEFAULTS,
            max_file_size_mb=10,
            excluded_extensions=[".tmp", ".log", ".cache", ".lock", ".db", ".sqlite", ".flac", ".mp3"],
            included_extensions=[".txt", ".md", ".pdf", ".docx", ".py", ".js", ".ts"],
            schedule_enabled=False,
            schedule_time=None,
        )

        # Detect GPU capabilities for defaults
        gpu_memory_limit = None
        enable_gpu = True
        preferred_gpu = "auto"

        try:
            from .gpu_manager import get_gpu_manager

            gpu_manager = get_gpu_manager()
            capabilities = gpu_manager.detect_gpu_capabilities()

            if capabilities.has_cuda and capabilities.gpus:
                gpu_memory_limit = gpu_manager.get_optimal_memory_limit()
                preferred_gpu = capabilities.recommended_device or "auto"
                logger.info(f"GPU acceleration enabled: {len(capabilities.gpus)} CUDA devices found")
            else:
                enable_gpu = False
                logger.info("GPU acceleration disabled: No CUDA devices found")

        except Exception as e:
            enable_gpu = False
            logger.warning(f"GPU detection failed, disabling GPU acceleration: {e}")

        # Create system config with hardware-appropriate defaults
        config = SystemConfig(
            port=8000,
            auto_start=False,
            max_memory_usage_mb=4096,  # ~25% of 16GB
            max_cpu_usage_percent=75,  # ~75% of 6 cores (4.5 cores)
            active_profile="default",
            load_multiple_vectorstores=False,
            auto_indexing=auto_indexing,
            data_directory=str(self.data_dir),
            config_directory=str(self.config_dir),
            require_ollama_setup=True,
            enable_gpu_acceleration=enable_gpu,
            gpu_memory_limit_mb=gpu_memory_limit,
            preferred_gpu_device=preferred_gpu,
        )

        logger.info("Created default system configuration")
        return config

    def _create_default_profile(self) -> None:
        """Create default user profile."""
        try:
            default_profile = self.create_profile(name="default", display_name="Default Profile")
            logger.info("Created default profile")

        except Exception as e:
            logger.error(f"Failed to create default profile: {e}")


# Global configuration service instance
_config_service: Optional[ConfigurationService] = None


def get_config_service() -> ConfigurationService:
    """
    Get global configuration service instance.

    Returns:
        ConfigurationService singleton instance
    """
    global _config_service

    if _config_service is None:
        _config_service = ConfigurationService()

    return _config_service


def initialize_config_service(config_dir: Optional[Path] = None, data_dir: Optional[Path] = None) -> ConfigurationService:
    """
    Initialize global configuration service with custom directories.

    Args:
        config_dir: Custom configuration directory
        data_dir: Custom data directory

    Returns:
        ConfigurationService instance
    """
    global _config_service

    _config_service = ConfigurationService(config_dir=config_dir, data_dir=data_dir)
    return _config_service


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    # Initialize config service
    config_service = get_config_service()

    # Test system config
    system_config = config_service.get_system_config()
    print(f"System config - Port: {system_config.port}, Memory limit: {system_config.max_memory_usage_mb}MB")

    # Test profile management
    profiles = config_service.list_profiles()
    print(f"Available profiles: {[p.name for p in profiles]}")

    active_profile = config_service.get_active_profile()
    if active_profile:
        print(f"Active profile: {active_profile.name} ({active_profile.display_name})")

    # Test auto-indexing config
    auto_config = config_service.get_auto_indexing_config()
    print(f"Auto-indexing mode: {auto_config.mode}, Max file size: {auto_config.max_file_size_mb}MB")
