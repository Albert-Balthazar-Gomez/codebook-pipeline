"""Tests for end-to-end non-OCR (scanned) PDF processing.

Verifies that a 100% scanned PDF with no digital text layer, containing:
  - Technical clauses and specifications
  - Technical drawings, diagrams, and schematics
  - Tables
is successfully converted into:
  - Structured clauses and hierarchy
  - High-resolution technical drawing PNGs
  - Markdown (.md) and Plain Text (.txt) complete documents
  - Embedded Qdrant vector database with hybrid search retrieval
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pymupdf
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.common.config import Config, load_config
from src.embedding.qdrant_indexer import HybridRetriever, index_chunks
from src.extraction.figure_extractor import extract_figures
from src.extraction.ocr_fallback import find_textless_pages, ocr_pages, tesseract_available
from src.extraction.table_extractor import extract_tables
from src.extraction.text_extractor import extract_clauses
from src.pipeline.run import run_pipeline
from src.processing.chunker import build_chunks
from src.processing.exporter import export_markdown, export_plaintext
from src.processing.linker import link_items


@pytest.fixture(scope="module")
def non_ocr_pdf(tmp_path_factory) -> Path:
    """Create a 100% raster scanned PDF with technical drawings and specifications."""
    pdf_dir = tmp_path_factory.mktemp("non_ocr_test")
    pdf_path = pdf_dir / "test_scanned_tech_drawings.pdf"

    doc = pymupdf.open()
    dpi = 150  # 150 DPI for test speed and clean OCR rendering
    pw, ph = int(8.5 * dpi), int(11 * dpi)  # Letter size

    # Page 1: Scanned technical clauses
    img1 = Image.new("RGB", (pw, ph), color="white")
    d1 = ImageDraw.Draw(img1)
    d1.text((80, 80), "SECTION 1.0 GENERAL SPECIFICATIONS", fill="black")
    d1.text((80, 140), "1.1 Scope and Objective", fill="black")
    d1.text(
        (80, 180),
        "This specification sets forth the structural design requirements for reinforced\n"
        "concrete elements, structural foundations, and engineering layout drawings.",
        fill="black",
    )
    d1.text((80, 260), "1.2 Design Methodology", fill="black")
    d1.text(
        (80, 300),
        "All structures shall be designed in accordance with the limit state method\n"
        "ensuring adequate safety, serviceability, and durability under load.",
        fill="black",
    )

    # Page 2: Scanned technical drawing with figure caption
    img2 = Image.new("RGB", (pw, ph), color="white")
    d2 = ImageDraw.Draw(img2)
    d2.text((80, 80), "2.1 Reinforcement Details", fill="black")
    d2.text(
        (80, 120),
        "Longitudinal reinforcement shall be anchored in accordance with Fig. 1 below.",
        fill="black",
    )
    # Draw a technical drawing schematic (beam cross section with rebars and dimensions)
    # Beam outline
    d2.rectangle([200, 220, 700, 500], outline="black", width=4)
    # Stirrup
    d2.rectangle([230, 250, 670, 470], outline="black", width=2)
    # Longitudinal rebar circles
    for cx, cy in [(250, 270), (450, 270), (650, 270), (250, 450), (450, 450), (650, 450)]:
        d2.ellipse([cx - 15, cy - 15, cx + 15, cy + 15], fill="black", outline="black")
    # Dimension lines and annotations
    d2.line([180, 220, 180, 500], fill="black", width=2)
    d2.text((120, 350), "D = 600 mm", fill="black")
    d2.line([200, 520, 700, 520], fill="black", width=2)
    d2.text((420, 535), "B = 300 mm", fill="black")
    # Figure caption
    d2.text((250, 620), "Fig. 1 Typical Reinforcement Detail and Beam Cross Section", fill="black")

    # Page 3: Scanned drawing sheet / blueprint with title block
    img3 = Image.new("RGB", (pw, ph), color="white")
    d3 = ImageDraw.Draw(img3)
    # Outer blueprint border
    d3.rectangle([50, 50, pw - 50, ph - 50], outline="black", width=3)
    # Structural schematic grid
    for x in range(150, pw - 100, 200):
        d3.line([x, 100, x, ph - 250], fill="black", width=2)
    for y in range(150, ph - 250, 150):
        d3.line([100, y, pw - 100, y], fill="black", width=2)
    # Foundation pads at intersections
    for x in range(150, pw - 100, 200):
        for y in range(150, ph - 250, 150):
            d3.rectangle([x - 30, y - 30, x + 30, y + 30], outline="black", width=3)
    # Title block in bottom right
    d3.rectangle([pw - 400, ph - 200, pw - 60, ph - 60], outline="black", width=2)
    d3.text((pw - 380, ph - 180), "DWG-001: STRUCTURAL FOUNDATION LAYOUT", fill="black")
    d3.text((pw - 380, ph - 150), "PROJECT: CIVIL ENGINEERING CODEBOOK", fill="black")
    d3.text((pw - 380, ph - 120), "SCALE: 1:100 | DATE: 2026-11-08", fill="black")

    # Page 4: Scanned table
    img4 = Image.new("RGB", (pw, ph), color="white")
    d4 = ImageDraw.Draw(img4)
    d4.text((80, 80), "3.1 Material Properties", fill="black")
    d4.text((80, 120), "Table 1 Minimum Reinforcement and Concrete Grades", fill="black")
    # Table grid
    d4.rectangle([80, 160, 800, 360], outline="black", width=2)
    d4.line([80, 210, 800, 210], fill="black", width=2)
    d4.line([80, 260, 800, 260], fill="black", width=1)
    d4.line([80, 310, 800, 310], fill="black", width=1)
    d4.line([300, 160, 300, 360], fill="black", width=2)
    d4.line([550, 160, 550, 360], fill="black", width=2)
    # Headers
    d4.text((100, 180), "Member Type", fill="black")
    d4.text((320, 180), "Grade", fill="black")
    d4.text((580, 180), "Min Reinforcement (%)", fill="black")
    # Row 1
    d4.text((100, 230), "Beams", fill="black")
    d4.text((320, 230), "M25", fill="black")
    d4.text((580, 230), "0.24", fill="black")
    # Row 2
    d4.text((100, 280), "Columns", fill="black")
    d4.text((320, 280), "M30", fill="black")
    d4.text((580, 280), "0.80", fill="black")
    # Row 3
    d4.text((100, 330), "Slabs", fill="black")
    d4.text((320, 330), "M20", fill="black")
    d4.text((580, 330), "0.15", fill="black")

    # Insert pure raster images as PDF pages (no digital text layer)
    for img in [img1, img2, img3, img4]:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        img_bytes = buf.read()

        # Create page with exact point dimensions matching image aspect ratio
        rect = pymupdf.Rect(0, 0, pw * 72.0 / dpi, ph * 72.0 / dpi)
        p = doc.new_page(width=rect.width, height=rect.height)
        p.insert_image(rect, stream=img_bytes)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_tesseract_is_operational():
    assert tesseract_available(), "Tesseract OCR binary must be available for non-OCR pipeline"


def test_find_textless_pages_detects_all_pages(non_ocr_pdf):
    config = load_config()
    empty, total = find_textless_pages(non_ocr_pdf, config)
    assert total == 4
    assert len(empty) == 4, f"All 4 pages must be flagged as textless, got {empty}"


def test_ocr_extracts_text_and_lines(non_ocr_pdf, tmp_path):
    config = load_config({"ocr": {"dpi": 200}})
    result, method, detail = ocr_pages(non_ocr_pdf, config, tmp_path)
    assert result.counts["ocr_pages"] >= 3
    assert method == "ocr"
    # Page 1 should contain Scope and Objective
    p1 = detail[1]
    assert "Scope" in p1["text"]
    assert len(p1["lines"]) > 0
    # Coordinates must be in 72-DPI points
    for l in p1["lines"]:
        assert len(l["bbox"]) == 4
        assert l["bbox"][0] >= 0 and l["bbox"][1] >= 0


def test_text_extractor_recovers_clauses_from_non_ocr_pdf(non_ocr_pdf, tmp_path):
    config = load_config({"ocr": {"dpi": 200}})
    res_ocr, _, ocr_detail = ocr_pages(non_ocr_pdf, config, tmp_path)
    res_text, clauses = extract_clauses(non_ocr_pdf, config, tmp_path, ocr_detail=ocr_detail)

    assert len(clauses) >= 2, f"Must recover at least 2 clauses, got {len(clauses)}"
    clause_ids = [c["clause_id"] for c in clauses]
    # Check that generic/technical clauses were recovered
    assert any("1." in cid or "1" in cid for cid in clause_ids)


def test_figure_extractor_extracts_technical_drawings_from_scanned_pdf(non_ocr_pdf, tmp_path):
    config = load_config({"ocr": {"dpi": 200}})
    res_ocr, _, ocr_detail = ocr_pages(non_ocr_pdf, config, tmp_path)
    res_fig, figures = extract_figures(non_ocr_pdf, config, tmp_path, ocr_detail=ocr_detail)

    assert len(figures) >= 1, "Must extract at least one technical drawing from scanned pages"
    # Check technical drawing attributes
    for f in figures:
        assert f["is_technical_drawing"] is True
        assert f["status"] == "OK"
        png_path = tmp_path / f["png_path"]
        assert png_path.exists(), f"Figure PNG {png_path} must exist on disk"
        # Verify valid non-empty PNG
        img = Image.open(png_path)
        assert img.width > 50 and img.height > 50


def test_complete_markdown_and_plaintext_exports(non_ocr_pdf, tmp_path):
    config = load_config({"ocr": {"dpi": 200}})
    res_ocr, _, ocr_detail = ocr_pages(non_ocr_pdf, config, tmp_path)
    _, clauses = extract_clauses(non_ocr_pdf, config, tmp_path, ocr_detail=ocr_detail)
    _, tables, _ = extract_tables(non_ocr_pdf, config, tmp_path, ocr_detail=ocr_detail)
    _, figures = extract_figures(non_ocr_pdf, config, tmp_path, ocr_detail=ocr_detail)

    md_path = export_markdown(clauses, tables, figures, "scanned_doc", tmp_path, standard_name="Scanned Test Standard")
    txt_path = export_plaintext(clauses, tables, figures, "scanned_doc", tmp_path, standard_name="Scanned Test Standard")

    assert md_path.exists()
    assert txt_path.exists()

    md_content = md_path.read_text(encoding="utf-8")
    assert "# Scanned Test Standard" in md_content
    assert "## Table of Contents" in md_content
    # Figures and drawings must be embedded
    if figures:
        assert "![" in md_content

    txt_content = txt_path.read_text(encoding="utf-8")
    assert "SCANNED TEST STANDARD" in txt_content
    assert "CLAUSE" in txt_content


def test_end_to_end_pipeline_and_vector_db_indexing(non_ocr_pdf, tmp_path):
    config = load_config({
        "ocr": {"dpi": 200},
        "dataset_generation": {"enabled": False},
        "embedding": {"device": "cpu"},
    })

    # Run full pipeline phases: OCR (8), Text (2), Tables (3), Figures (4), Chunking/Linking (5), Vector DB (6)
    exit_code = run_pipeline(
        non_ocr_pdf,
        config,
        phases=("2", "3", "4", "5", "6"),
        out_root=tmp_path / "extracted",
    )
    assert exit_code in (0, 1), f"Pipeline run must complete cleanly, got {exit_code}"

    # Verify generated markdown and txt files exist in processed dir
    codebook = "test_scanned_tech_drawings"
    processed_dir = config.out_dirs(codebook)["processed"]
    md_file = processed_dir / f"{codebook}.md"
    txt_file = processed_dir / f"{codebook}.txt"
    assert md_file.exists(), f"Markdown file {md_file} must be generated"
    assert txt_file.exists(), f"Plain text file {txt_file} must be generated"

    # Verify Vector DB has indexed chunks including text and technical drawings
    vdb_dir = config.out_dirs(codebook)["vector_db"]
    assert vdb_dir.exists(), "Vector DB directory must exist"

    retriever = HybridRetriever(
        config=config,
        db_dir=vdb_dir,
        codebook=codebook,
    )
    try:
        # Search for technical drawing content
        hits = retriever.search("reinforcement detail and beam cross section", top_k=5)
        assert len(hits) > 0, "Vector hybrid search must return results for technical drawing query"

        # Check payload carries technical drawing attributes and png_path
        fig_hits = [h for h in hits if h.get("source_type") == "figure"]
        if fig_hits:
            assert fig_hits[0].get("png_path") is not None
    finally:
        retriever.close()
