"""
ForecastIQ Backend Configuration Settings
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    PROJECT_NAME: str = "ForecastIQ Backend"
    API_V1_STR: str = "/api"
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]
    MAX_UPLOAD_SIZE_BYTES: int = 150 * 1024 * 1024  # 150 MB
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DATABASE: str = "forecastiq"
    MONGODB_ENABLED: bool = True

    model_config = SettingsConfigDict(case_sensitive=True, env_file=".env", extra="ignore")


settings = Settings()
