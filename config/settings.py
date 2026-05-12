"""
Central configuration for the resume matcher.

WHY Pydantic BaseModel?
  - Type validation: catches typos like image_dpi="high" (should be int)
  - Easy to extend: add new settings without rewriting anything
  - Serializable: can dump to JSON for logging/debugging
  - FastAPI uses Pydantic too, so when you add the API layer,
    your config and request/response models use the same pattern

WHY a global instance?
  Every module does: from config.settings import config
  This gives you one source of truth. No passing config
  objects through 5 layers of function calls.
"""

from pathlib import Path
from pydantic import BaseModel


class AppConfig(BaseModel):
    """All settings for the resume matcher pipeline."""

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
    max_tokens: int = 4096
    image_dpi: int = 150       # 150 DPI ≈ 1,600 tokens/page (vision fallback)

    # --- Processing ---
    batch_size: int = 100      # files per processing batch
    max_retries: int = 3       # retry failed extractions

    def ensure_dirs(self):
        """Create all output directories if they don't exist."""
        for d in [self.output_dir, self.json_dir, self.failed_dir]:
            d.mkdir(parents=True, exist_ok=True)


# Single global instance — import this everywhere
config = AppConfig()