"""Single entry point for the pipeline.

    python -m src.pipeline.run --pdf data/raw/is456.pdf --standard is456
    python -m src.pipeline.run --pdf <pdf> --phases 2 3 4 5
    python -m src.pipeline.run --all          # everything found in data/raw/

Every phase returns SUCCESS / DEGRADED / FAILED and the run exits non-zero if any
phase FAILED. A DEGRADED run still produces complete artifacts; the reasons are in
pipeline_report.json and on stderr.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from ..common.config import load_config, slugify
from ..common.jsonl import read_jsonl, write_json
from ..common.status import PhaseReport, PhaseResult, Status
from ..extraction.figure_extractor import extract_figures
from ..extraction.ocr_fallback import ocr_pages
from ..extraction.page_renderer import render_pages
from ..extraction.table_extractor import extract_tables
from ..extraction.text_extractor import extract_clauses
from ..processing.chunker import build_chunks
from ..processing.linker import link_items, spot_check_samples

log = logging.getLogger("pipeline")

ALL_PHASES = ("2", "3", "4", "5", "6", "7")


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def slug_for(pdf: Path, config) -> str:
    explicit = config.get("codebook.name")
    if explicit:
        return slugify(explicit)
    return slugify(pdf.stem)


def run_pipeline(
    pdf_path: Path,
    config,
    phases: tuple[str, ...],
    out_root: Path | None = None,
) -> int:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        log.error("PDF not found: %s", pdf_path)
        return 2

    codebook = slug_for(pdf_path, config)
    dirs = config.out_dirs(codebook)
    extracted = Path(out_root) if out_root else dirs["extracted"]
    processed = dirs["processed"]
    extracted.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)

    log.info("codebook=%s pdf=%s", codebook, pdf_path.name)
    report = PhaseReport(codebook, processed)

    # ---- Phase 8 preflight: OCR pages that have no text layer -------------- #
    ocr_result = None
    ocr_detail: dict = {}
    if config.get("ocr.enabled", True):
        ocr_result, method, ocr_detail = ocr_pages(pdf_path, config, extracted)
        report.add(ocr_result)
        if ocr_result.counts.get("ocr_pages"):
            write_json(extracted / "ocr" / f"{pdf_path.stem}_pages.json", {
                str(k): {kk: vv for kk, vv in v.items()} for k, v in ocr_detail.items()
            })
            log.warning(
                "OCR produced %d page(s); those pages are marked confidence 'ocr'",
                ocr_result.counts["ocr_pages"],
            )

    # ---- Phase 2: text ------------------------------------------------------ #
    clauses: list[dict[str, Any]] = []
    if "2" in phases:
        r, clauses = extract_clauses(pdf_path, config, extracted, method=method)
        report.add(r)
    else:
        clauses = read_jsonl(extracted / "text" / "clauses.jsonl")

    # ---- Phase 3: tables ---------------------------------------------------- #
    tables: list[dict[str, Any]] = []
    table_boxes: dict[int, list[tuple[float, float, float, float]]] = {}
    if "3" in phases:
        r, tables, table_boxes = extract_tables(pdf_path, config, extracted)
        report.add(r)
    else:
        tables = read_jsonl(extracted / "tables" / "tables.jsonl")
        table_boxes = {}
        from ..extraction.text_extractor import find_ruled_grid_bboxes
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            all_boxes = find_ruled_grid_bboxes(pdf)
        table_boxes = {
            p: [tuple(t["bbox"]) for t in tables if t.get("page") == p and t.get("bbox")]
            for p in range(1, len(pdf.pages) + 1)
        } if tables else all_boxes

    # ---- Phase 4: figures --------------------------------------------------- #
    figures: list[dict[str, Any]] = []
    if "4" in phases:
        r, figures = extract_figures(pdf_path, config, extracted, table_boxes)
        report.add(r)
    else:
        figures = read_jsonl(extracted / "figures" / "figures_metadata.jsonl")

    # ---- Phase 5: link + chunk ---------------------------------------------- #
    if "5" in phases:
        from ..common.jsonl import write_jsonl

        r_link, figures, tables = link_items(clauses, figures, tables)
        report.add(r_link)
        # Persist the link columns back into the phase 3/4 artifacts.
        write_jsonl(extracted / "tables" / "tables.jsonl", tables)
        write_jsonl(extracted / "figures" / "figures_metadata.jsonl", figures)

        r_chunk, chunks = build_chunks(clauses, figures, tables, config, processed)
        report.add(r_chunk)

        samples = spot_check_samples(figures, clauses, n=10)
        write_json(processed / "figure_link_samples.json", samples)
        log.info("wrote %d figure-link samples for manual spot-check", len(samples))
    else:
        chunks = read_jsonl(processed / "multimodal_chunks.jsonl")

    # ---- optional: page renders -------------------------------------------- #
    if config.get("page_rendering.enabled", False):
        report.add(render_pages(pdf_path, config, extracted))

    # ---- Phase 6: vector db ------------------------------------------------- #
    if "6" in phases and config.get("dataset_generation.enabled", True) is not None:
        from ..embedding.qdrant_indexer import index_chunks

        r = index_chunks(chunks, figures, config, dirs["vector_db"], codebook)
        report.add(r)

    # ---- Phase 7: instruction dataset -------------------------------------- #
    if "7" in phases:
        from ..dataset_generation.instruction_generator import generate_dataset

        r = generate_dataset(chunks, config, processed)
        report.add(r)

    report.write(extra={"pdf": str(pdf_path), "phases_run": list(phases)})

    # ---- validation report ------------------------------------------------- #
    try:
        from ..validation.validation_report import write_validation_report

        vpath = write_validation_report(
            config, codebook, processed, extracted, report, clauses, tables, figures, chunks
        )
        log.info("wrote %s", vpath)
    except Exception as exc:  # noqa: BLE001
        log.warning("validation report failed: %s", exc)

    log.info("=== overall: %s ===", report.overall.value)
    for r in report.results:
        log.info("  %s", r.summary_line())
    return report.overall.exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="codebook-pipeline",
        description="Universal codebook -> multimodal AI pipeline",
    )
    parser.add_argument("--pdf", type=Path, help="Path to the codebook PDF")
    parser.add_argument("--all", action="store_true", help="Run every PDF in data/raw/")
    parser.add_argument("--standard", default=None, help="Override codebook.standard (is456|eurocode|aci318|bs8110)")
    parser.add_argument("--name", default=None, help="Override the codebook slug")
    parser.add_argument(
        "--phases",
        nargs="+",
        default=list(ALL_PHASES[:4]),
        choices=list(ALL_PHASES),
        help="Phases to run (default: 2 3 4 5)",
    )
    parser.add_argument("--config", type=Path, default=None, help="Extra config YAML to merge")
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)

    overrides: dict[str, Any] = {}
    if args.standard:
        overrides.setdefault("codebook", {})["standard"] = args.standard
    if args.name:
        overrides.setdefault("codebook", {})["name"] = args.name
    if args.config:
        import yaml

        overrides = _deep_merge(overrides, yaml.safe_load(args.config.read_text()) or {})

    config = load_config(overrides)
    setup_logging(args.log_level or config.get("logging.level", "INFO"))

    if not args.pdf and not args.all:
        parser.error("pass --pdf <file> or --all")

    if args.all:
        raw = config.path("raw")
        pdfs = sorted(p for p in raw.glob("*.pdf"))
        if not pdfs:
            log.error("no PDFs in %s", raw)
            return 2
        log.info("running %d PDF(s) from %s", len(pdfs), raw)
        worst = 0
        for pdf in pdfs:
            log.info("=" * 70)
            worst = max(worst, run_pipeline(pdf, config, tuple(args.phases)))
        return worst

    return run_pipeline(args.pdf, config, tuple(args.phases))


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    sys.exit(main())