from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Paths ---
    resume_dir: Path = Path("data/sample_resumes")
    output_dir: Path = Path("data/output")
    json_dir: Path = Path("data/output/json")
    failed_dir: Path = Path("data/output/failed")
    manifest_path: Path = Path("data/output/manifest.json")

    # --- Supported file types ---
    supported_extensions: set[str] = {
        ".pdf", ".docx", ".doc",
        ".jpg", ".jpeg", ".png", ".webp",
    }

    # --- LLM settings ---
    groq_model: str = "llama-3.3-70b-versatile"
    openai_vision_model: str = "gpt-4o-mini"
    llm_model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 4096
    image_dpi: int = 150

    # --- Processing ---
    batch_size: int = 100
    max_retries: int = 3
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB

    # --- Database ---
    database_url: str = "postgresql+asyncpg://user:password@localhost/resume_matcher"

    # --- Redis / Celery ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Storage ---
    storage_backend: str = "local"          # "local" or "s3"
    local_upload_dir: Path = Path("data/uploads")
    s3_bucket: str = "resume-matcher-files"
    aws_region: str = "us-east-1"

    def ensure_dirs(self):
        for d in [self.output_dir, self.json_dir, self.failed_dir, self.local_upload_dir]:
            d.mkdir(parents=True, exist_ok=True)


config = AppConfig()
