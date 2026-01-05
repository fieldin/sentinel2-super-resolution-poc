"""
Application settings using Pydantic for environment variable loading.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # UP42 Credentials (username/password auth per docs.up42.com)
    # Made optional so POC can run with pre-loaded tiles
    up42_username: str = Field(default="", description="UP42 account email (optional for POC)")
    up42_password: str = Field(default="", description="UP42 account password (optional for POC)")
    up42_project_id: str = Field(
        default="", description="UP42 Project/Account ID (optional)"
    )

    # Imagery Search Parameters
    days_lookback: int = Field(
        default=30, description="Number of days to search back for imagery"
    )
    max_cloud_pct: float = Field(
        default=10.0, description="Maximum cloud coverage percentage"
    )

    # Tiling Parameters
    tile_min_zoom: int = Field(default=10, description="Minimum zoom level for tiles")
    tile_max_zoom: int = Field(default=16, description="Maximum zoom level for tiles")

    # Mapbox (for client configuration)
    mapbox_access_token: str = Field(..., description="Mapbox GL JS access token")

    # Paths
    aoi_path: str = Field(
        default="/app/config/aoi.geojson", description="Path to AOI GeoJSON file"
    )
    data_dir: str = Field(
        default="/app/data", description="Base directory for output data"
    )

    # Server
    server_host: str = Field(default="0.0.0.0", description="Server host")
    server_port: int = Field(default=8080, description="Server port")

    # UP42 API Configuration
    up42_auth_url: str = Field(
        default="https://auth.up42.com/realms/public/protocol/openid-connect/token",
        description="UP42 OAuth2 token endpoint",
    )
    up42_api_base: str = Field(
        default="https://api.up42.com/v2", description="UP42 API v2 base URL"
    )
    up42_catalog_url: str = Field(
        default="https://api.up42.com/catalog/stac/search",
        description="UP42 STAC catalog search endpoint",
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
