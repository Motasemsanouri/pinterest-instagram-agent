from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─── Gemini (Phase 2 — image analysis, free tier) ─────────────────────────
    gemini_api_key: str = ""
    vision_model: str = "gemini-2.0-flash-lite"
    image_analysis_concurrency: int = 5
    # ─── Pinterest credentials (optional) ────────────────────────────────────
    pinterest_email: str = ""
    pinterest_password: str = ""

    # ─── Scraping ─────────────────────────────────────────────────────────────
    max_pins_per_query: int = 50
    headless_browser: bool = True
    browser_timeout: int = 30000
    scroll_delay_min: float = 1.0
    scroll_delay_max: float = 3.0

    # ─── Retry ────────────────────────────────────────────────────────────────
    # ─── Phase 5 — Instagram Publishing ───────────────────────────────────────
    instagram_dry_run: bool = True
    instagram_access_token: str = ""
    instagram_account_id: str = ""
    max_posts_per_run: int = 3
    base_hashtags: str = "homedecor,interiordesign,homeinspo,decor"

    # ─── Phase 4 — Quality Evaluation ─────────────────────────────────────────
    quality_threshold: float = 0.70
    min_images_to_publish: int = 1

    max_retries: int = 3
    retry_delay: float = 2.0

    # ─── Logging ──────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: str = "logs/pinterest_agent.log"

    # ─── Output ───────────────────────────────────────────────────────────────
    output_dir: str = "data"
    save_results: bool = True


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings