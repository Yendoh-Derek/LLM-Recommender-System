"""
Centralized configuration management for the LLM Recommender API.
All paths, settings, and environment variables are defined here.
"""
import os
from pathlib import Path
from typing import Optional, Any
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Settings
    API_TITLE: str = "LLM Recommender API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Reinforcement Learning-based LLM Recommendation System"
    DEBUG: bool = False
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
    MODELS_DIR: Path = ARTIFACTS_DIR / "models"
    EMBEDDINGS_DIR: Path = ARTIFACTS_DIR / "embeddings"
    PREPROCESSING_DIR: Path = ARTIFACTS_DIR / "preprocessing"
    FEATURES_DIR: Path = ARTIFACTS_DIR / "features"
    CONFIG_DIR: Path = ARTIFACTS_DIR / "config"
    
    # Model Settings (Placeholders - will be replaced when artifacts are available)
    MODEL_NAME: Optional[str] = None  # SAC agent model name
    EMBEDDING_MODEL_NAME: Optional[str] = None  # Embedding model name
    PREPROCESSING_CONFIG_PATH: Optional[Path] = None  # Preprocessing config path
    
    # Recommendation Settings
    DEFAULT_TOP_K: int = 10  # Default number of recommendations
    MAX_TOP_K: int = 50  # Maximum number of recommendations
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # "json" or "text"
    LOG_FILE: Optional[Path] = None  # If None, logs to console only
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()


def ensure_directories():
    """Ensure all required directories exist."""
    settings.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    settings.EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    settings.PREPROCESSING_DIR.mkdir(parents=True, exist_ok=True)
    settings.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    settings.CONFIG_DIR.mkdir(parents=True, exist_ok=True)

