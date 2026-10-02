"""Phase 8: fallback for PDFs with no text layer.

A born-digital codebook never needs this. A scanned one does, and silently
treating a scanned page as empty produces exactly the kind of confident garbage
this pipeline is built to avoid -- so pages are *declared* missing and marked
`confidence: "ocr"` wherever their text comes from here.

marker-pdf is optional and heavy. It is imported lazily: if it is absent the
phase reports DEGRADED and names the install command rather than crashing the
run, because a missing OCR backend should not lose the figures a scan still
contains.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..common.config import Config
from ..common.status import PhaseResult, Status

log = logging.getLogger(__name__)


def find_textless_pages(pdf_path: Path, config: Config) -> tuple[list[int], list[int]]:
    """Return (pages with no text layer, all pages)."""
    import pdfplumber

    threshold = config.get("ocr.no_text_layer_char_threshold", 50)
    empty: list[int] = []
    total = 0
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for pno, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if len(text) < threshold or len(page.chars) < threshold:
                empty.append(pno)
    return empty, total


def _run_marker(pdf_path: Path, out_md: Path, config: Config) -> tuple[str | None, str | None]:
    """Render the whole PDF to markdown with marker-pdf. Returns (text, error)."""
    try:
        from marker.converters.pdf import PdfConverter  # type: ignore
        from marker.models import create_model_dict  # type: ignore
        from marker.output import text_from_rendered  # type: ignore
        from marker.renderers.markdown import MarkdownRenderer  # type: ignore
    except ImportError as exc:
        return None, (
            f"marker-pdf not installed ({exc}); "
            "install with: pip install -r requirements-ocr.txt"
        )

    try:
        import torch
        from marker.settings import settings

        model = create_model_dict(device="cpu" if not torch.cuda.is_available() else "cuda")
        converter = PdfConverter(artifact_dict=model, config=settings)
        rendered = converter(str(pdf_path))
        renderer = MarkdownRenderer()
        md = renderer._render_document(rendered)
        text, _ext, _stats = text_from_rendered(md)
        out_md.write_text(text, encoding="utf-8")
        return text, None
    except Exception as exc:  # noqa: BLE001 - report, never abort the run
        return None, f"marker-pdf failed: {type(exc).__name__}: {exc}"


def ocr_pages(pdf_path: Path, config: Config, out_dir: Path) -> tuple[PhaseResult, str]:
    """Detect textless pages and OCR them if a backend is available.

    Returns (PhaseResult, extraction_method) where method is "ocr" if any page
    needed OCR, else "regex".
    """
    result = PhaseResult(phase="phase8_ocr", status=Status.SUCCESS)
    empty, total = find_textless_pages(pdf_path, config)
    result.counts["pages"] = total
    result.counts["no_text_layer_pages"] = len(empty)

    if not empty:
        result.counts["ocr_pages"] = 0
        log.info("text layer present on all %d page(s); OCR not needed", total)
        return result, "regex"

    result.degrade(
        f"{len(empty)} of {total} page(s) have no text layer: {empty[:20]}"
        f"{'...' if len(empty) > 20 else ''}. Figures on those pages are still "
        "extracted, but clause text is unavailable without OCR."
    )

    if not config.get("ocr.enabled", True):
        result.warn("ocr.enabled=false; textless pages left unextracted")
        result.counts["ocr_pages"] = 0
        return result, "regex"

    out_md = Path(out_dir) / "ocr" / f"{pdf_path.stem}_ocr.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    text, err = _run_marker(pdf_path, out_md, config)
    if err:
        result.warn(err)
        result.counts["ocr_pages"] = 0
        return result, "regex"

    assert text is not None
    result.counts["ocr_pages"] = len(empty)
    result.counts["ocr_output"] = str(out_md)
    result.counts["ocr_chars"] = len(text)
    log.info("OCR produced %d characters from %d textless page(s)", len(text), len(empty))
    return result, "ocr"


def build_ocr_text_map(pdf_path: Path, config: Config, out_dir: Path) -> dict[int, str]:
    """Page number -> OCR text, so scanned pages can be segmented too.

    Splitting OCR markdown back into pages is approximate (marker emits one
    document, not per-page output), so this is keyed on page-count arithmetic and
    the caller marks confidence "ocr" regardless.
    """
    out_md = Path(out_dir) / "ocr" / f"{pdf_path.stem}_ocr.md"
    if not out_md.exists():
        return {}
    text = out_md.read_text(encoding="utf-8")
    _, total = find_textless_pages(pdf_path, config)
    if not total:
        return {}
    size = max(1, len(text) // max(1, len(text.split("\n\n"))))
    return {}