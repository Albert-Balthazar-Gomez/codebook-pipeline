"""Phase 8: local OCR for pages with no usable text layer.

Tesseract only (`--oem 1 --psm 6`: LSTM engine, single uniform block). No cloud
OCR, no model download beyond tesseract's own language data.

Two things this module is careful about:

1.  **Cross-checking.** A page can carry a sparse text layer *and* be missing its
    body text (a scanned page with only a stamped header). Blindly preferring OCR
    would then overwrite good text with guesses, so agreement between the two is
    measured and a disagreement flags the page instead of silently choosing.

2.  **Declaring failure.** If tesseract is not installed the phase reports
    DEGRADED and names the binary, rather than returning empty strings that would
    look like a clean extraction of nothing.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Any

from ..common.config import Config
from ..common.status import PhaseResult, Status

log = logging.getLogger(__name__)

_WORD = re.compile(r"[A-Za-z0-9]+")


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def tesseract_version() -> str | None:
    exe = shutil.which("tesseract")
    if not exe:
        return None
    import subprocess

    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=20)
        return (out.stdout or out.stderr).splitlines()[0].strip()
    except Exception:
        return None


def find_textless_pages(pdf_path: Path, config: Config) -> tuple[list[int], int]:
    """Return (pages with no usable text layer, total pages)."""
    import pdfplumber

    threshold = config.get("ocr.no_text_layer_char_threshold", 50)
    empty: list[int] = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for pno, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if len(text) < threshold or len(page.chars) < threshold:
                empty.append(pno)
    return empty, total


def _ocr_page(pdf_path: Path, page_no: int, config: Config, out_dir: Path) -> tuple[str, str | None]:
    """OCR one page at high DPI. Returns (text, error)."""
    try:
        import pytesseract
    except ImportError as exc:
        return "", f"pytesseract not installed: {exc}"
    if not tesseract_available():
        return "", "tesseract binary not found on PATH"

    dpi = config.get("ocr.dpi", 600)
    oem = config.get("ocr.oem", 1)
    psm = config.get("ocr.psm", 6)
    lang = config.get("ocr.lang", "eng")

    try:
        import pymupdf
    except ImportError as exc:
        return "", f"PyMuPDF not installed: {exc}"

    try:
        doc = pymupdf.open(pdf_path)
        try:
            page = doc[page_no - 1]
            pix = page.get_pixmap(dpi=dpi)
            png = out_dir / f"p{page_no:04d}.png"
            png.parent.mkdir(parents=True, exist_ok=True)
            pix.save(str(png))
        finally:
            doc.close()
        cfg = f"--oem {oem} --psm {psm}"
        text = pytesseract.image_to_string(str(png), lang=lang, config=cfg)
        return text, None
    except Exception as exc:  # noqa: BLE001
        return "", f"{type(exc).__name__}: {str(exc)[:180]}"


def _token_overlap(a: str, b: str) -> float:
    """Fraction of a's tokens that also appear in b."""
    ta = {w.lower() for w in _WORD.findall(a or "")}
    tb = {w.lower() for w in _WORD.findall(b or "")}
    if not ta:
        return 0.0
    return len(ta & tb) / len(ta)


def ocr_pages(
    pdf_path: Path, config: Config, out_dir: Path
) -> tuple[PhaseResult, str, dict[int, dict[str, Any]]]:
    """OCR textless pages.

    Returns (PhaseResult, extraction_method, per-page detail) where detail carries
    the cross-check verdict for each OCR'd page.
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    result = PhaseResult(phase="phase8_ocr", status=Status.SUCCESS)
    empty, total = find_textless_pages(pdf_path, config)
    detail: dict[int, dict[str, Any]] = {}

    result.counts["pages"] = total
    result.counts["no_text_layer_pages"] = len(empty)

    if not empty:
        result.counts["ocr_pages"] = 0
        log.info("text layer present on all %d page(s); OCR not needed", total)
        return result, "text_layer", detail

    result.degrade(
        f"{len(empty)} of {total} page(s) have no usable text layer: {empty[:20]}"
        f"{'...' if len(empty) > 20 else ''}. Figures on those pages are still "
        "extracted, but clause text is unavailable without OCR."
    )

    if not config.get("ocr.enabled", True):
        result.warn("ocr.enabled=false; textless pages left unextracted")
        result.counts["ocr_pages"] = 0
        return result, "text_layer", detail

    version = tesseract_version()
    result.counts["tesseract"] = version or "missing"
    if not version:
        result.warn(
            "tesseract is not installed, so OCR'd pages produce no text. "
            "Install it with:  sudo apt install tesseract-ocr"
        )
        result.counts["ocr_pages"] = 0
        result.counts["ocr_failures"] = len(empty)
        return result, "text_layer", detail

    import pdfplumber

    min_overlap = config.get("ocr.min_crosscheck_overlap", 0.35)
    ocr_dir = out_dir / "ocr"
    failures: list[str] = []
    disagreements = 0

    with pdfplumber.open(pdf_path) as pdf:
        for pno in empty:
            embedded = (pdf.pages[pno - 1].extract_text() or "").strip()
            text, err = _ocr_page(pdf_path, pno, config, ocr_dir)
            if err:
                failures.append(f"p{pno}: {err}")
                detail[pno] = {"status": "failed", "error": err}
                continue
            text = text.strip()
            if not text:
                failures.append(f"p{pno}: OCR returned no text")
                detail[pno] = {"status": "empty"}
                continue

            # Cross-check against whatever text layer exists. An empty layer means
            # there is nothing to disagree with.
            overlap = _token_overlap(embedded, text) if embedded else None
            if overlap is None:
                verdict = "no_text_layer"
            elif overlap >= min_overlap:
                verdict = "agrees_with_text_layer"
            else:
                verdict = "disagrees_with_text_layer"
                disagreements += 1

            detail[pno] = {
                "status": "ok",
                "verdict": verdict,
                "overlap": None if overlap is None else round(overlap, 3),
                "chars": len(text),
                "text": text,
                "embedded_chars": len(embedded),
            }

    ok = sum(1 for d in detail.values() if d["status"] == "ok")
    result.counts["ocr_pages"] = ok
    result.counts["ocr_failures"] = len(failures)
    result.counts["ocr_disagreements"] = disagreements
    result.counts["ocr_output"] = str(ocr_dir)

    if failures:
        result.warn(
            f"{len(failures)} page(s) failed OCR: {failures[:5]}"
            f"{'...' if len(failures) > 5 else ''}"
        )
    if disagreements:
        result.degrade(
            f"{disagreements} OCR'd page(s) disagreed with the embedded text layer "
            f"(overlap below {min_overlap}). One of the two is wrong; the page is "
            "flagged in the per-page OCR detail rather than silently merged"
        )
    log.info("OCR produced text for %d of %d textless page(s)", ok, len(empty))
    return result, ("ocr" if ok else "text_layer"), detail


def load_ocr_detail(out_dir: Path, stem: str) -> dict[int, dict[str, Any]]:
    """Per-page OCR detail written by run.py, for auditing."""
    from ..common.jsonl import read_json

    data = read_json(Path(out_dir) / "ocr" / f"{stem}_pages.json")
    return {int(k): v for k, v in (data or {}).items()}