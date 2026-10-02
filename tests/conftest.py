"""Shared pytest fixtures.

Builds the synthetic fixture once per session so the tests assert against a
document whose correct answer is known exactly (33 clauses, 6 tables, 4 figures).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "test_is456_excerpt.pdf"

sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def fixture_pdf() -> Path:
    if not FIXTURE.exists():
        subprocess.run(
            [sys.executable, str(ROOT / "fixtures" / "make_fixture.py")], check=True, cwd=ROOT
        )
    return FIXTURE


@pytest.fixture(scope="session")
def config():
    from src.common.config import load_config

    return load_config()


@pytest.fixture(scope="session")
def extracted(fixture_pdf, config, tmp_path_factory):
    """Run phases 2-5 once and share the result across tests."""
    from src.extraction.figure_extractor import extract_figures
    from src.extraction.table_extractor import extract_tables
    from src.extraction.text_extractor import extract_clauses
    from src.processing.chunker import build_chunks
    from src.processing.linker import link_items

    out = tmp_path_factory.mktemp("extracted")
    r2, clauses = extract_clauses(fixture_pdf, config, out)
    r3, tables, boxes = extract_tables(fixture_pdf, config, out)
    r4, figures = extract_figures(fixture_pdf, config, out, boxes)
    r5, figures, tables = link_items(clauses, figures, tables)
    processed = tmp_path_factory.mktemp("processed")
    r6, chunks = build_chunks(clauses, figures, tables, config, processed)
    return {
        "dir": out,
        "processed": processed,
        "clauses": clauses,
        "tables": tables,
        "figures": figures,
        "chunks": chunks,
        "results": {"p2": r2, "p3": r3, "p4": r4, "p5_link": r5, "p5_chunk": r6},
    }