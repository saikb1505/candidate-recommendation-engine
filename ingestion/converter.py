"""
Document → Image converter.

Every resume gets converted to page images, regardless of format.
These images are sent to Claude's vision API for extraction.

Pipeline per format:
  PDF  → pdf2image renders each page → list of base64 JPEG strings
  DOCX → libreoffice converts to PDF → then same as PDF path
  IMG  → Pillow loads the file → single base64 JPEG string

WHY convert everything to images?
  - One extraction path instead of three
  - Claude sees the page exactly as a human would
  - No garbled text from multi-column layouts or tables
  - Creative resume designs, charts, logos all handled

WHY base64 output?
  The Anthropic API accepts images as base64-encoded strings.
  Converting here means the extraction module can pass them
  directly to the API without any file I/O.

WHY JPEG instead of PNG?
  JPEG files are 3-5x smaller than PNG at comparable quality.
  Smaller images = faster API uploads. At 10K resumes × 3 pages
  = 30K images, this adds up. Quality loss is negligible for
  text-heavy documents like resumes.
"""

import base64
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from dataclasses import dataclass

from PIL import Image
from pdf2image import convert_from_path

from ingestion.scanner import FileType
from config.settings import config
import shutil
import platform


@dataclass
class ConversionResult:
    """
    Output of converting a document to images.

    images: list of base64-encoded JPEG strings (one per page)
    page_count: number of pages detected
    error: error message if conversion failed, empty string if ok

    WHY a result object instead of just returning the list?
    You need to know the page count (for cost estimation) and
    whether conversion failed (for error handling). Returning
    a tuple of (images, count, error) gets messy fast.
    """
    images: list[str]       # base64-encoded JPEG strings
    page_count: int = 0
    error: str = ""

    @property
    def success(self) -> bool:
        return len(self.images) > 0 and not self.error


def convert_to_images(file_path: str, file_type: str) -> ConversionResult:
    """
    Convert any supported document to base64 page images.

    This is the only function other modules call.
    It routes to the right converter based on file type.

    Args:
        file_path: absolute or relative path to the resume file
        file_type: FileType enum value ("pdf", "docx", "image")

    Returns:
        ConversionResult with base64 images and metadata
    """
    try:
        if file_type == FileType.PDF:
            return _convert_pdf(file_path)
        elif file_type == FileType.DOCX:
            return _convert_docx(file_path)
        elif file_type == FileType.IMAGE:
            return _convert_image(file_path)
        else:
            return ConversionResult(
                images=[],
                error=f"Unsupported file type: {file_type}",
            )
    except Exception as e:
        return ConversionResult(
            images=[],
            error=f"Conversion failed: {str(e)}",
        )


# ──────────────────────────────────────────────
#  FORMAT-SPECIFIC CONVERTERS (private)
# ──────────────────────────────────────────────

def _convert_pdf(file_path: str) -> ConversionResult:
    """
    Render each PDF page as a JPEG image.

    pdf2image uses poppler under the hood — a battle-tested
    PDF rendering library. It handles:
    - Multi-column layouts
    - Embedded fonts
    - Vector graphics and charts
    - Scanned pages (they're already images)

    DPI setting (from config):
      150 DPI → ~1,600 tokens/page, good balance
      200 DPI → ~2,800 tokens/page, better for small text
      100 DPI → ~1,000 tokens/page, cheaper but may miss details
    """
    pil_images = convert_from_path(
        file_path,
        dpi=config.image_dpi,
        fmt="jpeg",
    )

    base64_images = [_pil_to_base64(img) for img in pil_images]

    return ConversionResult(
        images=base64_images,
        page_count=len(base64_images),
    )

def _get_libreoffice_path() -> str:
    """
    Find libreoffice binary across any OS.

    Strategy:
      1. Check PATH first (works if user installed it properly)
      2. Fall back to known installation paths per OS
      3. Raise clear error if not found

    WHY shutil.which()?
      It does what 'which' does in terminal — searches PATH
      for an executable. Works on Mac, Linux, and Windows.
      OS-independent by design.
    """
    # 1. Check PATH — works on any OS if properly installed
    for cmd in ("libreoffice", "soffice"):
        path = shutil.which(cmd)
        if path:
            return path

    # 2. Check known installation paths
    known_paths = []

    system = platform.system()
    if system == "Darwin":  # macOS
        known_paths = [
            "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        ]
    elif system == "Windows":
        known_paths = [
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
        ]
    # Linux: usually in PATH, no known_paths needed

    for path in known_paths:
        if Path(path).exists():
            return path

    raise FileNotFoundError(
        "LibreOffice not found. Install it:\n"
        "  Mac:     brew install --cask libreoffice\n"
        "  Linux:   sudo apt install libreoffice-core\n"
        "  Windows: https://www.libreoffice.org/download"
    )

def _convert_docx(file_path: str) -> ConversionResult:
    """
    Convert DOCX to images via intermediate PDF.

    DOCX → (libreoffice) → PDF → (pdf2image) → JPEG images

    WHY libreoffice?
    python-docx can read text but can't render the visual layout.
    libreoffice is a full document renderer — it preserves
    formatting, tables, columns, fonts, everything. The PDF
    it produces looks exactly like the original DOCX.

    The --headless flag runs libreoffice without a GUI window.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Step 1: DOCX → PDF using libreoffice
        subprocess.run(
            [
                _get_libreoffice_path(),
                "--headless",           # no GUI
                "--convert-to", "pdf",  # output format
                "--outdir", tmp_dir,    # where to save
                file_path,              # input file
            ],
            capture_output=True,
            timeout=30,  # kill if it hangs
        )

        # Find the generated PDF
        pdf_files = list(Path(tmp_dir).glob("*.pdf"))
        if not pdf_files:
            return ConversionResult(
                images=[],
                error="libreoffice failed to convert DOCX to PDF",
            )

        # Step 2: PDF → images (reuse PDF converter)
        return _convert_pdf(str(pdf_files[0]))


def _convert_image(file_path: str) -> ConversionResult:
    """
    Load an image file and convert to base64 JPEG.

    Image-based resumes are already visual — just load,
    optionally resize if too large, and encode.

    WHY resize?
    Some scanned resumes are 3000x4000+ pixels.
    Claude's vision API works best with images under
    1568px on the longest side. Larger images get
    downscaled by the API anyway, so we do it ourselves
    to reduce upload size and cost.
    """
    img = Image.open(file_path)

    # Convert to RGB if necessary (handles RGBA, palette mode)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize if too large (max 1568px on longest side)
    max_dimension = 1568
    if max(img.size) > max_dimension:
        img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

    base64_str = _pil_to_base64(img)

    return ConversionResult(
        images=[base64_str],
        page_count=1,
    )


# ──────────────────────────────────────────────
#  UTILITY
# ──────────────────────────────────────────────

def _pil_to_base64(img: Image.Image, quality: int = 85) -> str:
    """
    Convert a PIL Image to a base64-encoded JPEG string.

    WHY quality=85?
    JPEG quality 85 is the sweet spot:
    - Visually identical to quality 95 for text documents
    - File size is ~40% smaller than quality 95
    - At 10K resumes × 3 pages, this saves significant
      upload bandwidth and API processing time
    """
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")