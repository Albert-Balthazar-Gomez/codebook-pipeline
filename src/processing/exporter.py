"""Complete document exporters for Markdown (.md) and Plain Text (.txt).

Generates end-to-end representations of processed codebooks, specifications,
and technical drawing sets, embedding text clauses, markdown tables, and
high-resolution technical drawing PNG figures.
"""

from __future__ import annotations

import datetime
import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def _slugify_anchor(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "-", s)


def export_markdown(
    clauses: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    figures: list[dict[str, Any]],
    codebook_slug: str,
    out_dir: Path | str,
    standard_name: str = "Technical Standard / Codebook",
) -> Path:
    """Generate a comprehensive Markdown (.md) export of the codebook.

    Embeds clause hierarchy, body text, GitHub-flavored markdown tables,
    and technical drawing PNGs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{codebook_slug}.md"

    # Group tables and figures by linked clause
    tables_by_clause: dict[str, list[dict[str, Any]]] = {}
    for t in tables:
        ref = t.get("clause_ref")
        if ref:
            tables_by_clause.setdefault(ref, []).append(t)

    figures_by_clause: dict[str, list[dict[str, Any]]] = {}
    for f in figures:
        ref = f.get("related_clause")
        if ref:
            figures_by_clause.setdefault(ref, []).append(f)

    linked_table_ids = {t["table_id"] for t_list in tables_by_clause.values() for t in t_list}
    linked_figure_ids = {f["figure_id"] for f_list in figures_by_clause.values() for f in f_list}
    unlinked_tables = [t for t in tables if t["table_id"] not in linked_table_ids]
    unlinked_figures = [f for f in figures if f["figure_id"] not in linked_figure_ids]

    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines: list[str] = [
        f"# {standard_name}",
        "",
        f"> **Document**: `{codebook_slug}`  ",
        f"> **Generated**: {now_iso}  ",
        f"> **Clauses**: {len(clauses)} | **Tables**: {len(tables)} | **Figures / Drawings**: {len(figures)}  ",
        "",
        "---",
        "",
        "## Table of Contents",
        "",
    ]

    for c in clauses:
        cid = c.get("clause_id", "")
        title = c.get("title") or "General"
        depth = max(1, int(c.get("depth", 1)))
        indent = "  " * (depth - 1)
        anchor = _slugify_anchor(f"clause-{cid}-{title}")
        lines.append(f"{indent}- [{cid} {title}](#{anchor})")

    if unlinked_tables or unlinked_figures:
        lines.append("- [Additional Figures & Tables](#additional-figures--tables)")

    lines.extend(["", "---", "", "## Document Content", ""])

    for c in clauses:
        cid = c.get("clause_id", "")
        title = c.get("title") or ""
        depth = max(1, int(c.get("depth", 1)))
        h_prefix = "#" * min(6, depth + 2)
        anchor = _slugify_anchor(f"clause-{cid}-{title}")
        page = c.get("page", 1)

        lines.append(f'<a id="{anchor}"></a>')
        lines.append(f"{h_prefix} {cid} {title}")
        lines.append(f"*Page: {page} | Confidence: {c.get('confidence', 'standard')}*")
        lines.append("")

        body = (c.get("body") or "").strip()
        if body:
            lines.append(body)
            lines.append("")

        # Embedded tables in this clause
        clause_tables = tables_by_clause.get(cid, [])
        for t in clause_tables:
            caption = t.get("caption") or f"Table {t.get('table_id')}"
            md_table = (t.get("markdown") or "").strip()
            lines.append(f"**{caption}** (Page {t.get('page')})")
            lines.append("")
            if md_table:
                lines.append(md_table)
                lines.append("")

        # Embedded figures / technical drawings in this clause
        clause_figs = figures_by_clause.get(cid, [])
        for f in clause_figs:
            caption = f.get("caption") or f"Figure {f.get('figure_id')}"
            png_path = f.get("png_path") or ""
            is_dwg = f.get("is_technical_drawing", False)
            tag = "Technical Drawing" if is_dwg else "Figure"
            lines.append(f"**{tag}: {caption}** (Page {f.get('page')})")
            lines.append("")
            if png_path:
                lines.append(f"![{caption}]({png_path})")
                lines.append("")

    if unlinked_tables or unlinked_figures:
        lines.extend([
            "---",
            "",
            "## Additional Figures & Tables",
            "",
        ])
        for t in unlinked_tables:
            caption = t.get("caption") or f"Table {t.get('table_id')}"
            md_table = (t.get("markdown") or "").strip()
            lines.append(f"### {caption} (Page {t.get('page')})")
            lines.append("")
            if md_table:
                lines.append(md_table)
                lines.append("")
        for f in unlinked_figures:
            caption = f.get("caption") or f"Figure {f.get('figure_id')}"
            png_path = f.get("png_path") or ""
            is_dwg = f.get("is_technical_drawing", False)
            tag = "Technical Drawing" if is_dwg else "Figure"
            lines.append(f"### {tag}: {caption} (Page {f.get('page')})")
            lines.append("")
            if png_path:
                lines.append(f"![{caption}]({png_path})")
                lines.append("")

    content = "\n".join(lines).strip() + "\n"
    md_path.write_text(content, encoding="utf-8")
    log.info("wrote complete Markdown export to %s (%d bytes)", md_path, len(content))
    return md_path


def export_plaintext(
    clauses: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    figures: list[dict[str, Any]],
    codebook_slug: str,
    out_dir: Path | str,
    standard_name: str = "Technical Standard / Codebook",
) -> Path:
    """Generate a clean Plain Text (.txt) export of the codebook."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"{codebook_slug}.txt"

    tables_by_clause: dict[str, list[dict[str, Any]]] = {}
    for t in tables:
        ref = t.get("clause_ref")
        if ref:
            tables_by_clause.setdefault(ref, []).append(t)

    figures_by_clause: dict[str, list[dict[str, Any]]] = {}
    for f in figures:
        ref = f.get("related_clause")
        if ref:
            figures_by_clause.setdefault(ref, []).append(f)

    now_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    divider = "=" * 80
    subdivider = "-" * 80

    lines: list[str] = [
        divider,
        f"{standard_name.upper()}",
        f"Document: {codebook_slug}",
        f"Generated: {now_iso}",
        f"Clauses: {len(clauses)} | Tables: {len(tables)} | Figures/Drawings: {len(figures)}",
        divider,
        "",
    ]

    for c in clauses:
        cid = c.get("clause_id", "")
        title = c.get("title") or ""
        page = c.get("page", 1)
        lines.append(subdivider)
        lines.append(f"CLAUSE {cid}: {title} (Page {page})")
        lines.append(subdivider)

        body = (c.get("body") or "").strip()
        if body:
            lines.append(body)
            lines.append("")

        for t in tables_by_clause.get(cid, []):
            caption = t.get("caption") or f"Table {t.get('table_id')}"
            lines.append(f"[TABLE: {caption} | Page {t.get('page')}]")
            md_table = (t.get("markdown") or "").strip()
            if md_table:
                lines.append(md_table)
            lines.append("")

        for f in figures_by_clause.get(cid, []):
            caption = f.get("caption") or f"Figure {f.get('figure_id')}"
            png_path = f.get("png_path") or ""
            is_dwg = f.get("is_technical_drawing", False)
            tag = "TECHNICAL DRAWING" if is_dwg else "FIGURE"
            lines.append(f"[{tag}: {caption} | Image: {png_path} | Page {f.get('page')}]")
            lines.append("")

    content = "\n".join(lines).strip() + "\n"
    txt_path.write_text(content, encoding="utf-8")
    log.info("wrote complete Plain Text export to %s (%d bytes)", txt_path, len(content))
    return txt_path
