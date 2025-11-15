"""
Configuration settings for the Text Generator API
Uses pydantic-settings for environment variable management
"""

from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """
    Application settings
    Can be overridden by environment variables
    """
    
    # Project Info
    PROJECT_NAME: str = "Text Generator API"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"  # development, staging, production
    
    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    
    # Default Model Settings
    DEFAULT_MODEL_NAME: str = "gpt2"  # Default model to load on startup
    DEVICE: str = "auto"  # "auto", "cpu", "cuda", "cuda:0", etc.
    MAX_LENGTH: int = 2048
    
    # Model Cache Settings
    MAX_CACHED_MODELS: int = 2  # Maximum number of models to keep in memory
    
    # Generation Limits
    MAX_NEW_TOKENS_LIMIT: int = 2048
    MAX_PROMPT_LENGTH: int = 5000
    
    # CORS Settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:8501",  # Streamlit default port
        "http://localhost:3000",  # React default port
        "http://127.0.0.1:8501",
        "http://127.0.0.1:3000",
        "*"  # Remove in production, specify exact origins
    ]
    
    # Rate Limiting (if implemented)
    RATE_LIMIT_PER_MINUTE: int = 10
    
    # Logging
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Create global settings instance
settings = Settings()


# Optional: Print settings on import (for debugging)
if __name__ == "__main__":
    print("Current Settings:")
    print(f"  Project: {settings.PROJECT_NAME}")
    print(f"  Version: {settings.VERSION}")
    print(f"  Environment: {settings.ENVIRONMENT}")
    print(f"  API Prefix: {settings.API_V1_PREFIX}")
    print(f"  Default Model: {settings.DEFAULT_MODEL_NAME}")
    print(f"  Device: {settings.DEVICE}")
    print(f"  Max Cached Models: {settings.MAX_CACHED_MODELS}")