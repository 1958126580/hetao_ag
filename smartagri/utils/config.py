"""
Configuration Management for SmartAgri

Provides centralized configuration management for the library,
supporting environment variables, configuration files, and
runtime settings.

Features:
    - Hierarchical configuration
    - Environment variable override
    - Type-safe access
    - Default values
    - Configuration validation
"""

import os
import json
from typing import Any, Optional, Dict, Union, TypeVar
from pathlib import Path
from dataclasses import dataclass, field, asdict
import threading
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class GPUConfig:
    """GPU acceleration configuration."""
    enabled: bool = True
    device_id: int = 0
    memory_fraction: float = 0.9
    allow_growth: bool = True


@dataclass
class ParallelConfig:
    """Parallel computing configuration."""
    enabled: bool = True
    n_workers: Optional[int] = None  # None = auto-detect
    backend: str = "process"  # 'process' or 'thread'
    chunk_size: int = 1000


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None


@dataclass
class DataConfig:
    """Data handling configuration."""
    cache_dir: str = ".smartagri_cache"
    temp_dir: str = "/tmp/smartagri"
    max_cache_size_gb: float = 10.0
    enable_caching: bool = True


@dataclass
class Config:
    """
    Master configuration for SmartAgri library.

    Centralizes all configuration settings with sensible defaults.
    Can be loaded from file or environment variables.

    Example:
        >>> config = Config()
        >>> config.gpu.enabled = True
        >>> config.parallel.n_workers = 8
        >>> save_config(config, "config.json")
    """
    gpu: GPUConfig = field(default_factory=GPUConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Custom settings dictionary for extensions
    custom: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Apply environment variable overrides."""
        self._apply_env_overrides()

    def _apply_env_overrides(self):
        """Override settings from environment variables."""
        env_mappings = {
            "SMARTAGRI_GPU_ENABLED": ("gpu", "enabled", bool),
            "SMARTAGRI_GPU_DEVICE": ("gpu", "device_id", int),
            "SMARTAGRI_PARALLEL_ENABLED": ("parallel", "enabled", bool),
            "SMARTAGRI_PARALLEL_WORKERS": ("parallel", "n_workers", int),
            "SMARTAGRI_LOG_LEVEL": ("logging", "level", str),
            "SMARTAGRI_CACHE_DIR": ("data", "cache_dir", str),
        }

        for env_var, (section, key, dtype) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                section_obj = getattr(self, section)
                if dtype == bool:
                    value = value.lower() in ("true", "1", "yes")
                elif dtype == int:
                    value = int(value)
                setattr(section_obj, key, value)
                logger.debug(f"Applied env override: {env_var}={value}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "gpu": asdict(self.gpu),
            "parallel": asdict(self.parallel),
            "logging": asdict(self.logging),
            "data": asdict(self.data),
            "custom": self.custom,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        config = cls()

        if "gpu" in data:
            config.gpu = GPUConfig(**data["gpu"])
        if "parallel" in data:
            config.parallel = ParallelConfig(**data["parallel"])
        if "logging" in data:
            config.logging = LoggingConfig(**data["logging"])
        if "data" in data:
            config.data = DataConfig(**data["data"])
        if "custom" in data:
            config.custom = data["custom"]

        return config

    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        filepath = Path(filepath)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved configuration to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "Config":
        """Load configuration from JSON file."""
        filepath = Path(filepath)
        with open(filepath, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded configuration from {filepath}")
        return cls.from_dict(data)


class ConfigManager:
    """
    Thread-safe global configuration manager.

    Provides singleton access to library configuration with
    support for multiple named configurations.

    Example:
        >>> manager = ConfigManager()
        >>> manager.set("gpu.enabled", True)
        >>> gpu_enabled = manager.get("gpu.enabled")
    """

    _instance: Optional["ConfigManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ConfigManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._configs = {"default": Config()}
                cls._instance._active = "default"
        return cls._instance

    @property
    def config(self) -> Config:
        """Get active configuration."""
        return self._configs[self._active]

    def register(self, name: str, config: Config) -> None:
        """Register a named configuration."""
        self._configs[name] = config

    def activate(self, name: str) -> None:
        """Activate a named configuration."""
        if name not in self._configs:
            raise ValueError(f"Configuration '{name}' not registered")
        self._active = name

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'gpu.enabled')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        parts = key.split(".")
        obj = self.config

        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return default

        return obj

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'gpu.enabled')
            value: Value to set
        """
        parts = key.split(".")
        obj = self.config

        for part in parts[:-1]:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict):
                if part not in obj:
                    obj[part] = {}
                obj = obj[part]

        final_key = parts[-1]
        if hasattr(obj, final_key):
            setattr(obj, final_key, value)
        elif isinstance(obj, dict):
            obj[final_key] = value

    def reset(self) -> None:
        """Reset to default configuration."""
        self._configs = {"default": Config()}
        self._active = "default"


# Global configuration instance
_config_manager = ConfigManager()


def get_config() -> Config:
    """
    Get current global configuration.

    Returns:
        Active Config object

    Example:
        >>> config = get_config()
        >>> if config.gpu.enabled:
        ...     print("GPU acceleration is enabled")
    """
    return _config_manager.config


def set_config(key: str, value: Any) -> None:
    """
    Set global configuration value.

    Args:
        key: Configuration key using dot notation
        value: Value to set

    Example:
        >>> set_config("gpu.enabled", False)
        >>> set_config("parallel.n_workers", 4)
    """
    _config_manager.set(key, value)


def load_config(filepath: Union[str, Path]) -> Config:
    """
    Load configuration from file and activate it.

    Args:
        filepath: Path to configuration file

    Returns:
        Loaded Config object
    """
    config = Config.load(filepath)
    _config_manager.register("loaded", config)
    _config_manager.activate("loaded")
    return config


def save_config(filepath: Union[str, Path]) -> None:
    """
    Save current configuration to file.

    Args:
        filepath: Output file path
    """
    get_config().save(filepath)


def configure_logging() -> None:
    """Configure Python logging based on library settings."""
    config = get_config()
    log_config = config.logging

    level = getattr(logging, log_config.level.upper(), logging.INFO)

    handlers = [logging.StreamHandler()]
    if log_config.file:
        handlers.append(logging.FileHandler(log_config.file))

    logging.basicConfig(
        level=level,
        format=log_config.format,
        handlers=handlers,
    )

    logger.info("Logging configured")
