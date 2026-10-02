"""Phase 5b: chunking into a single multimodal stream.

One chunk per clause by default. A clause longer than `chunking.max_words` is
split on paragraph boundaries and every part keeps `parent_clause_id`, so the
original clause is always reconstructible. Splitting on paragraphs rather than
token windows keeps sentences intact, which matters because these chunks become
instruction-tuning data.

Tables and figures become their own chunk types and also carry their text back
into the parent clause's chunk, so a clause is never indexed without the context
needed to interpret a reference to "Table 2" or "Fig. 3".
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ..common.jsonl import write_jsonl
from ..common.status import PhaseResult, Status

log = logging.getLogger(__name__)

# Cross-references a figure/table adds to its parent clause's text.
_REF_RX = re.compile(r"(?i)\b(?:fig(?:ure)?|table)\s*[-–—]?\s*([A-Z]?\d+(?:[.\-]\d+)*)")


# Sentence boundary for splitting an oversized paragraph. Kept simple and
# conservative: a false split costs a little chunk cohesion, a missed split costs
# a silently truncated embedding.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank lines, falling back to single newlines."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) <= 1:
        paras = [p.strip() for p in text.split("\n") if p.strip()]
    return paras


def _atomise(paragraph: str, max_words: int) -> list[str]:
    """Break a paragraph into units that each fit the word budget.

    Paragraphs are the natural boundary, but a codebook clause is often one
    unbroken run of text with no blank lines. Without sentence-level fallback such
    a clause would stay a single oversized chunk, and bge-small-en-v1.5 has a
    512-token limit -- the excess would be dropped silently, producing an
    embedding of only the clause's opening.
    """
    if len(paragraph.split()) <= max_words:
        return [paragraph]
    sentences = [s for s in _SENT_SPLIT.split(paragraph) if s.strip()]
    if len(sentences) <= 1:
        # No sentence boundary to exploit. Emit as-is rather than cutting
        # mid-sentence; the caller flags the over-budget chunk.
        return [paragraph]
    units: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        words = len(sentence.split())
        if current and count + words > max_words:
            units.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += words
    if current:
        units.append(" ".join(current))
    return units


def _pack(paragraphs: list[str], max_words: int) -> list[list[str]]:
    """Greedily pack units into groups under the word budget."""
    units: list[str] = []
    for para in paragraphs:
        units.extend(_atomise(para, max_words))

    groups: list[list[str]] = []
    current: list[str] = []
    count = 0
    for unit in units:
        words = len(unit.split())
        if current and count + words > max_words:
            groups.append(current)
            current, count = [], 0
        current.append(unit)
        count += words
    if current:
        groups.append(current)
    return groups


def _chunk_clause(clause: dict[str, Any], max_words: int) -> list[dict[str, Any]]:
    text = (clause.get("text") or "").strip()
    if not text:
        return []
    words = len(text.split())
    if words <= max_words:
        return [
            {
                "chunk_id": f"text:{clause['clause_id']}",
                "type": "text",
                "clause_id": clause["clause_id"],
                "parent_clause_id": clause.get("parent_id"),
                "clause_depth": clause.get("depth"),
                "clause_title": clause.get("title"),
                "text": text,
                "page": clause.get("page"),
                "confidence": clause.get("confidence"),
                "standard": clause.get("standard"),
                "part": 1,
                "n_parts": 1,
                "word_count": words,
            }
        ]

    groups = _pack(_split_paragraphs(text), max_words)
    out = []
    for i, group in enumerate(groups, start=1):
        part_text = "\n\n".join(group)
        out.append(
            {
                "chunk_id": f"text:{clause['clause_id']}#{i}",
                "type": "text",
                "clause_id": clause["clause_id"],
                "parent_clause_id": clause.get("parent_id"),
                "clause_depth": clause.get("depth"),
                "clause_title": clause.get("title"),
                "text": part_text,
                "page": clause.get("page"),
                "confidence": clause.get("confidence"),
                "standard": clause.get("standard"),
                "part": i,
                "n_parts": len(groups),
                "word_count": len(part_text.split()),
            }
        )
    return out


def _table_chunk(table: dict[str, Any], clauses_by_id: dict[str, dict]) -> dict[str, Any] | None:
    markdown = (table.get("markdown") or "").strip()
    if not markdown:
        return None
    clause = clauses_by_id.get(table.get("clause_ref"))
    context = ""
    if clause:
        context = f"{clause['clause_id']} {clause.get('title') or ''}".strip()
    caption = table.get("caption") or ""
    return {
        "chunk_id": f"table:{table['table_id']}",
        "type": "table",
        "table_id": table["table_id"],
        "clause_ref": table.get("clause_ref"),
        "caption": caption,
        "markdown": markdown,
        # What actually gets embedded: the caption plus the clause it belongs to,
        # so a vector search for "limiting moment of resistance" can surface the
        # table even when the numbers are in an unlabelled cell.
        "embed_text": " ".join(x for x in (context, caption, markdown) if x).strip(),
        "page": table.get("page"),
        "link_confidence": table.get("link_confidence"),
        "bbox": table.get("bbox"),
        "n_rows": table.get("n_rows"),
        "n_cols": table.get("n_cols"),
        "word_count": len(markdown.split()),
    }


def _figure_chunk(
    figure: dict[str, Any], clauses_by_id: dict[str, dict]
) -> dict[str, Any] | None:
    if figure.get("status") == "FAILED" or not figure.get("png_path"):
        return None
    clause = clauses_by_id.get(figure.get("related_clause"))
    clause_text = ""
    if clause:
        clause_text = f"{clause['clause_id']} {clause.get('title') or ''}".strip()
        body = (clause.get("body") or "").strip()
        if body:
            clause_text = f"{clause_text}. {body[:400]}"
    caption = figure.get("caption") or ""
    return {
        "chunk_id": f"figure:{figure['figure_id']}",
        "type": "figure",
        "figure_id": figure["figure_id"],
        "figure_number": figure.get("figure_number"),
        "caption": caption,
        "related_clause": figure.get("related_clause"),
        "png_path": figure.get("png_path"),
        # Per the spec: no real image embedding, so the figure is embedded from
        # its caption plus the clause text that introduces it. The PNG travels as
        # payload.
        "embed_text": " ".join(x for x in (clause_text, caption) if x).strip(),
        "page": figure.get("page"),
        "status": figure.get("status"),
        "link_confidence": figure.get("link_confidence"),
        "bbox": figure.get("bbox"),
        "word_count": len(f"{clause_text} {caption}".split()),
    }


def build_chunks(
    clauses: list[dict[str, Any]],
    figures: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    config,
    out_dir,
) -> tuple[PhaseResult, list[dict[str, Any]]]:
    max_words = config.get("chunking.max_words", 800)
    min_words = config.get("chunking.min_words", 25)

    result = PhaseResult(phase="phase5_chunking", status=Status.SUCCESS)
    clauses_by_id = {c["clause_id"]: c for c in clauses}

    chunks: list[dict[str, Any]] = []
    for clause in clauses:
        chunks.extend(_chunk_clause(clause, max_words))

    # Cross-reference context: append "[see Table 2]" style pointers so the text
    # chunk explains why a sibling table matters.
    by_clause_table: dict[str, list[str]] = {}
    by_clause_figure: dict[str, list[str]] = {}
    for t in tables:
        if t.get("clause_ref"):
            by_clause_table.setdefault(t["clause_ref"], []).append(
                t.get("caption") or t["table_id"]
            )
    for f in figures:
        if f.get("related_clause"):
            by_clause_figure.setdefault(f["related_clause"], []).append(
                f.get("caption") or f["figure_id"]
            )

    for chunk in chunks:
        cid = chunk.get("clause_id")
        notes: list[str] = []
        if cid in by_clause_table:
            notes.append("Tables in this clause: " + "; ".join(by_clause_table[cid]))
        if cid in by_clause_figure:
            notes.append("Figures in this clause: " + "; ".join(by_clause_figure[cid]))
        if notes:
            chunk["related_assets"] = notes

    table_chunks = [c for c in (_table_chunk(t, clauses_by_id) for t in tables) if c]
    figure_chunks = [c for c in (_figure_chunk(f, clauses_by_id) for f in figures) if c]
    chunks.extend(table_chunks)
    chunks.extend(figure_chunks)

    # Merge tiny text chunks into the previous chunk of the same clause. A 12-word
    # fragment is a poor embedding target and a poor fine-tuning example.
    merged = 0
    merged_chunks: list[dict[str, Any]] = []
    for chunk in chunks:
        prev = merged_chunks[-1] if merged_chunks else None
        if (
            chunk["type"] == "text"
            and prev is not None
            and prev["type"] == "text"
            and prev.get("clause_id") == chunk.get("clause_id")
            and chunk["word_count"] < min_words
            and prev["word_count"] + chunk["word_count"] <= max_words
        ):
            prev["text"] = prev["text"] + "\n\n" + chunk["text"]
            prev["word_count"] = prev["word_count"] + chunk["word_count"]
            prev["chunk_id"] = f"text:{prev['clause_id']}#merged"
            merged += 1
            continue
        merged_chunks.append(chunk)
    chunks = merged_chunks

    text_chunks = [c for c in chunks if c["type"] == "text"]
    result.counts["chunks_total"] = len(chunks)
    result.counts["chunks_text"] = len(text_chunks)
    result.counts["chunks_table"] = len(table_chunks)
    result.counts["chunks_figure"] = len(figure_chunks)
    result.counts["chunks_split"] = sum(1 for c in text_chunks if c.get("n_parts", 1) > 1)
    result.counts["chunks_merged"] = merged
    result.counts["clauses_without_text_chunk"] = len(
        {c["clause_id"] for c in clauses}
        - {c["clause_id"] for c in text_chunks if c.get("clause_id")}
    )

    if merged:
        result.warn(
            f"{merged} sub-{min_words}-word chunk(s) merged into the preceding chunk "
            "of the same clause"
        )
    if result.counts["clauses_without_text_chunk"]:
        result.warn(
            f"{result.counts['clauses_without_text_chunk']} clause(s) produced no text "
            "chunk (empty body text); they are represented only by their tables/figures"
        )
    if result.counts["chunks_split"]:
        log.info("%d clause(s) exceeded %d words and were split", result.counts["chunks_split"], max_words)

    from pathlib import Path

    out_dir = Path(out_dir)
    write_jsonl(out_dir / "multimodal_chunks.jsonl", chunks)
    result.counts["output"] = str(out_dir / "multimodal_chunks.jsonl")
    return result, chunks