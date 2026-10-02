"""Optional high-DPI page renders.

Off by default (`page_rendering.enabled: false`). A 200 DPI A4 PNG is roughly
1-2 MB, so rendering a 400-page codebook costs hundreds of megabytes for output
that only matters when a human needs to eyeball a suspicious extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pymupdf

from ..common.config import Config
from ..common.status import PhaseResult, Status

log = logging.getLogger(__name__)


def render_pages(
    pdf_path: Path, config: Config, out_dir: Path, pages: tuple[int, int] | None = None
) -> PhaseResult:
    result = PhaseResult(phase="page_renderer", status=Status.SUCCESS)
    dpi = config.get("page_rendering.dpi", 200)
    out = Path(out_dir) / "pages"
    out.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    try:
        total = doc.page_count
        result.counts["pages"] = total
        written = 0
        for pno in range(total):
            if pages and not (pages[0] <= pno + 1 <= pages[1]):
                continue
            page = doc[pno]
            pix = page.get_pixmap(dpi=dpi)
            pix.save(str(out / f"p{pno + 1:04d}.png"))
            written += 1
        result.counts["rendered"] = written
        result.counts["dpi"] = dpi
        result.counts["output"] = str(out)
    finally:
        doc.close()
    return result