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

    # --- API keys ---
    groq_api_key: str = ""
    openai_api_key: str = ""

    # --- LLM settings ---
    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    openai_vision_model: str = "gpt-4o-mini"
    llm_model: str = "claude-haiku-4-5-20251001"
    max_tokens: int = 4096
    image_dpi: int = 150

    # --- Processing ---
    batch_size: int = 100
    max_retries: int = 1
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MB

    # --- Database ---
    database_url: str = "postgresql+asyncpg://user:password@localhost/resume_matcher"

    # --- Redis / Celery ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Qdrant ---
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "resume_chunks"

    # --- Storage ---
    storage_backend: str = "local"          # "local" or "s3"
    local_upload_dir: Path = Path("data/uploads")
    s3_bucket: str = "resume-matcher-files"
    aws_region: str = "us-east-1"

    @property
    def sync_database_url(self) -> str:
        return self.database_url.replace("+asyncpg", "")

    def ensure_dirs(self):
        for d in [self.output_dir, self.json_dir, self.failed_dir, self.local_upload_dir]:
            d.mkdir(parents=True, exist_ok=True)


config = AppConfig()
