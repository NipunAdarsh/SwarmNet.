"""
SwarmNet Backend Configuration.

Centralized configuration management using Pydantic BaseSettings.
All environment variables are defined and validated here.
"""

import os
from typing import List, Optional, Set
from functools import lru_cache

from pydantic import Field, field_validator

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings  # Fallback for older pydantic versions


class ServerConfig(BaseSettings):
    """Server configuration."""

    HOST: str = Field(default="0.0.0.0", description="Server bind host")
    PORT: int = Field(default=8000, ge=1, le=65535, description="Server bind port")
    DEBUG: bool = Field(default=False, description="Enable debug mode")

    model_config = {"env_prefix": ""}


class CORSConfig(BaseSettings):
    """CORS configuration."""

    ORIGINS: str = Field(
        default="http://localhost:8000,https://localhost:8000",
        description="Comma-separated list of allowed origins",
    )
    ALLOW_CREDENTIALS: bool = Field(default=True, description="Allow credentials in CORS")

    @property
    def origins_list(self) -> List[str]:
        """Parse origins string into list."""
        return [origin.strip() for origin in self.ORIGINS.split(",") if origin.strip()]

    model_config = {"env_prefix": "CORS_"}


class SecurityConfig(BaseSettings):
    """Security configuration."""

    SSL_CERT_FILE: Optional[str] = Field(default=None, description="Path to SSL certificate file")
    SSL_KEY_FILE: Optional[str] = Field(default=None, description="Path to SSL private key file")
    ADMIN_SECRET: str = Field(default="", description="Admin secret key (min 16 chars for production)")
    WS_AUTH_REQUIRED: bool = Field(default=False, description="Require WebSocket authentication")
    MAX_BODY_BYTES: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        ge=1024,
        description="Maximum request body size in bytes",
    )

    @field_validator("ADMIN_SECRET")
    @classmethod
    def validate_admin_secret(cls, v):
        """Warn if ADMIN_SECRET is too short in production."""
        if v and len(v) < 16:
            import logging
            logging.getLogger(__name__).warning(
                "ADMIN_SECRET is shorter than 16 characters. Use a stronger secret in production."
            )
        return v

    model_config = {"env_prefix": ""}


class ImageConfig(BaseSettings):
    """Image validation configuration."""

    MAX_SIZE_MB: int = Field(default=10, ge=1, le=100, description="Maximum image size in MB")
    ALLOWED_FORMATS: str = Field(
        default="jpg,jpeg,png,webp,gif,bmp",
        description="Comma-separated list of allowed image formats",
    )
    MAX_WIDTH: int = Field(default=4096, ge=1, le=16384, description="Maximum image width")
    MAX_HEIGHT: int = Field(default=4096, ge=1, le=16384, description="Maximum image height")

    @property
    def max_size_bytes(self) -> int:
        """Get max size in bytes."""
        return self.MAX_SIZE_MB * 1024 * 1024

    @property
    def formats_set(self) -> Set[str]:
        """Parse formats string into set."""
        return {fmt.strip().lower() for fmt in self.ALLOWED_FORMATS.split(",") if fmt.strip()}

    model_config = {"env_prefix": "IMAGE_"}


class ESConfig(BaseSettings):
    """Evolutionary Strategies training configuration."""

    POPULATION_SIZE: int = Field(default=50, ge=2, le=500, description="Population size (must be even)")
    SIGMA: float = Field(default=0.02, gt=0, lt=1, description="Mutation noise standard deviation")
    LEARNING_RATE: float = Field(default=0.03, gt=0, lt=1, description="Learning rate")
    WEIGHT_DECAY: float = Field(default=0.005, ge=0, lt=1, description="Weight decay regularization")
    MOMENTUM: float = Field(default=0.9, ge=0, lt=1, description="Momentum coefficient")
    MAX_GENERATIONS: int = Field(default=100, ge=1, le=10000, description="Maximum training generations")

    @field_validator("POPULATION_SIZE")
    @classmethod
    def validate_pop_size_even(cls, v):
        """Ensure population size is even for antithetic sampling."""
        if v % 2 != 0:
            return v + 1
        return v

    model_config = {"env_prefix": "ES_"}


class ModelConfig(BaseSettings):
    """Model paths and NPU configuration."""

    MODEL_PATH: str = Field(default="model/mobilenetv2-12.onnx", description="Path to main ONNX model")
    ES_MODEL_PATH: str = Field(default="model/es_trained.onnx", description="Path to ES-trained model")
    LABELS_PATH: str = Field(default="model/imagenet_labels.json", description="Path to labels JSON")

    SIMULATE_NPU: bool = Field(default=True, description="Simulate NPU if hardware not available")
    SIMULATED_NPU_PROVIDER: str = Field(
        default="QNNExecutionProvider",
        description="Simulated NPU provider name",
    )

    model_config = {"env_prefix": ""}


class SwarmConfig(BaseSettings):
    """Swarm discovery configuration."""

    MULTICAST_GROUP: str = Field(default="224.1.1.1", description="Multicast group for discovery")
    MULTICAST_PORT: int = Field(default=5007, ge=1, le=65535, description="Multicast port")
    NODE_HEARTBEAT_INTERVAL: int = Field(default=5, ge=1, le=60, description="Heartbeat interval in seconds")

    model_config = {"env_prefix": "SWARM_"}


class RateLimitConfig(BaseSettings):
    """Rate limiting configuration."""

    REQUESTS_PER_MINUTE: int = Field(default=120, ge=1, le=10000, description="Requests per minute per IP")

    model_config = {"env_prefix": "RATE_LIMIT_"}


class AppConfig(BaseSettings):
    """
    Main application configuration.

    Combines all sub-configurations into a single config object.
    """

    server: ServerConfig = Field(default_factory=ServerConfig)
    cors: CORSConfig = Field(default_factory=CORSConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    es: ESConfig = Field(default_factory=ESConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    swarm: SwarmConfig = Field(default_factory=SwarmConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache()
def get_config() -> AppConfig:
    """
    Get the application configuration singleton.

    Uses LRU cache to ensure config is only loaded once.
    """
    return AppConfig()


# Convenience access to config sections
config = get_config()


def reload_config() -> AppConfig:
    """
    Reload configuration from environment.

    Clears the cached config and reloads from environment variables.
    Useful for testing or dynamic configuration updates.
    """
    get_config.cache_clear()
    return get_config()
