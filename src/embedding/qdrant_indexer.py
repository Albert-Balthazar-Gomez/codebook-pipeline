"""Phase 6: Qdrant vector index with hybrid retrieval.

Runs Qdrant in embedded mode (`QdrantClient(path=...)`) rather than a Docker
container, so the index needs no daemon. Verified working with no Docker present.

Embeddings are `BAAI/bge-small-en-v1.5` (384-dim, 512 max tokens) as specified.
The dimension is asserted against config instead of assumed: a model swap that
silently changes dimensionality would otherwise either fail at write time with an
opaque error or, worse, succeed against a stale collection.

Per the spec, figures get no image embedding. A figure chunk is embedded from its
caption plus the clause text that introduces it, and the PNG travels as payload.

Retrieval is hybrid. Dense vectors alone are bad at the queries codebooks are
actually asked -- "Cl. 26.5.1.1", "Table 2" -- because a clause number is an
arbitrary token with no semantic neighbourhood. BM25 catches those exactly, so
both rankings are fused with Reciprocal Rank Fusion.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from ..common.config import Config
from ..common.status import PhaseResult, Status

log = logging.getLogger(__name__)

SOURCE_TEXT = "text"
SOURCE_TABLE = "table"
SOURCE_FIGURE = "figure"


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
class Embedder:
    """Lazy wrapper so Phases 2-5 never import torch."""

    def __init__(self, model_name: str, expected_dim: int, device: str = "cpu") -> None:
        self.model_name = model_name
        self.expected_dim = expected_dim
        self.device = device
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(self.model_name, device=self.device)
        # Renamed in sentence-transformers 6; keep the old name working.
        getter = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension"
        )
        actual = getter()
        if actual != self.expected_dim:
            raise ValueError(
                f"{self.model_name} produces {actual}-dim vectors but config expects "
                f"{self.expected_dim}. Update embedding.expected_dim or pick a different "
                "model; reusing a collection built at the old dimension would corrupt it."
            )
        log.info("loaded %s (%d-dim) on %s", self.model_name, actual, self.device)

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        self.load()
        vectors = self._model.encode(  # type: ignore[union-attr]
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 200,
        )
        return [list(map(float, v)) for v in vectors]


def _chunk_embed_text(chunk: dict[str, Any]) -> str:
    """The text actually embedded for a chunk."""
    ctype = chunk.get("type")
    if ctype == "table":
        return chunk.get("embed_text") or chunk.get("markdown") or ""
    if ctype == "figure":
        return chunk.get("embed_text") or chunk.get("caption") or ""
    return chunk.get("text") or ""


# --------------------------------------------------------------------------- #
# Indexing
# --------------------------------------------------------------------------- #
def index_chunks(
    chunks: list[dict[str, Any]],
    figures: list[dict[str, Any]],
    config: Config,
    db_dir: Path,
    codebook: str,
) -> PhaseResult:
    result = PhaseResult(phase="phase6_vector_db", status=Status.SUCCESS)

    if not chunks:
        result.status = Status.FAILED
        result.error = "no chunks to index"
        return result

    model_name = config.get("embedding.model", "BAAI/bge-small-en-v1.5")
    expected_dim = config.get("embedding.expected_dim", 384)
    batch_size = config.get("embedding.batch_size", 32)
    collection = f"{config.get('embedding.collection_prefix', 'codebook')}_{codebook}"

    # Import late so a missing torch degrades Phase 6 only.
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:
        result.status = Status.FAILED
        result.error = f"qdrant-client not installed: {exc}"
        return result

    db_dir = Path(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)

    # Figures referenced by payload id, so a query hit can return the PNG path.
    fig_by_id = {f["figure_id"]: f for f in figures}

    indexable = [c for c in chunks if _chunk_embed_text(c).strip()]
    result.counts["chunks_total"] = len(chunks)
    result.counts["chunks_indexed"] = len(indexable)
    result.counts["chunks_skipped_empty"] = len(chunks) - len(indexable)

    embedder = Embedder(model_name, expected_dim)
    try:
        vectors = embedder.encode([_chunk_embed_text(c) for c in indexable], batch_size)
    except ImportError as exc:
        result.status = Status.FAILED
        result.error = f"sentence-transformers not installed: {exc}"
        return result
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001
        result.status = Status.FAILED
        result.error = f"embedding failed: {type(exc).__name__}: {exc}"
        return result

    client = QdrantClient(path=str(db_dir))
    try:
        # Recreate from scratch: a stale collection at a different dimension is a
        # silent-corruption trap, and reindexing is cheap relative to a codebook.
        if client.collection_exists(collection):
            client.delete_collection(collection)
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=expected_dim, distance=models.Distance.COSINE),
        )

        points = []
        for i, (chunk, vec) in enumerate(zip(indexable, vectors)):
            ctype = chunk.get("type")
            payload: dict[str, Any] = {
                "source_type": ctype,
                "chunk_id": chunk.get("chunk_id"),
                "page": chunk.get("page"),
                "codebook": codebook,
                "text": _chunk_embed_text(chunk),
                "word_count": chunk.get("word_count"),
                "confidence": chunk.get("confidence"),
                "link_confidence": chunk.get("link_confidence"),
            }
            if ctype == SOURCE_TEXT:
                payload["clause_id"] = chunk.get("clause_id")
                payload["parent_clause_id"] = chunk.get("parent_clause_id")
                payload["clause_title"] = chunk.get("clause_title")
                payload["clause_depth"] = chunk.get("clause_depth")
            elif ctype == SOURCE_TABLE:
                payload["table_id"] = chunk.get("table_id")
                payload["clause_ref"] = chunk.get("clause_ref")
                payload["caption"] = chunk.get("caption")
                payload["markdown"] = chunk.get("markdown")
            elif ctype == SOURCE_FIGURE:
                payload["figure_id"] = chunk.get("figure_id")
                payload["related_clause"] = chunk.get("related_clause")
                payload["caption"] = chunk.get("caption")
                fig = fig_by_id.get(chunk.get("figure_id"), {})
                payload["png_path"] = fig.get("png_path")
                payload["figure_status"] = fig.get("status")
            points.append(models.PointStruct(id=i, vector=vec, payload=payload))

        client.upsert(collection_name=collection, points=points, wait=True)
        result.counts["collection"] = collection
        result.counts["points"] = len(points)
        result.counts["db_path"] = str(db_dir)
        result.counts["by_source"] = {
            t: sum(1 for c in indexable if c.get("type") == t)
            for t in (SOURCE_TEXT, SOURCE_TABLE, SOURCE_FIGURE)
        }
    finally:
        client.close()

    if result.counts.get("chunks_skipped_empty"):
        result.warn(
            f"{result.counts['chunks_skipped_empty']} chunk(s) had no embeddable text "
            "and were not indexed"
        )
    return result


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
_TOKEN = re.compile(r"[A-Za-z]+(?:[.\-]\d+)*|\d+(?:[.\-]\d+)+|\d+")


def _tokens(text: str) -> list[str]:
    """Tokenise, preserving dotted clause numbers as single tokens.

    Splitting "26.5.1.1" into four tokens lets BM25 match a query for "26.5"
    against it, which is wrong for an exact-identifier search.
    """
    out: list[str] = []
    for m in _TOKEN.finditer(text or ""):
        tok = m.group(0)
        out.append(tok)
        out.append(tok.lower())
    return out


class HybridRetriever:
    """Dense + BM25 fused with Reciprocal Rank Fusion."""

    def __init__(self, config: Config, db_dir: Path, codebook: str) -> None:
        self.config = config
        self.db_dir = Path(db_dir)
        self.collection = (
            f"{config.get('embedding.collection_prefix', 'codebook')}_{codebook}"
        )
        self._client = None
        self._embedder = None
        self._bm25 = None
        self._corpus: list[dict[str, Any]] = []

    # -- lazy handles -------------------------------------------------------- #
    def _get_client(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(path=str(self.db_dir))
        return self._client

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(
                self.config.get("embedding.model", "BAAI/bge-small-en-v1.5"),
                self.config.get("embedding.expected_dim", 384),
            )
        return self._embedder

    def _load_corpus(self) -> list[dict[str, Any]]:
        """BM25 is scored in-process over the stored payloads.

        Clause-level corpora are thousands of rows, not millions, so an exact
        scan costs milliseconds and needs no separate sparse index to keep in
        sync with the dense one.
        """
        if self._corpus:
            return self._corpus
        client = self._get_client()
        records, _ = client.scroll(
            collection_name=self.collection,
            limit=100000,
            with_payload=True,
            with_vectors=False,
        )
        # scroll() yields pydantic Records, not dicts. Normalise to plain dicts
        # with a hoisted payload so both the BM25 corpus and the result assembly
        # can treat them uniformly.
        self._corpus = [
            {
                "id": getattr(r, "id", None),
                "payload": dict(getattr(r, "payload", None) or {}),
            }
            for r in records
        ]
        return self._corpus

    def _bm25_ranked(self, query: str, k: int) -> list[int]:
        from rank_bm25 import BM25Okapi  # type: ignore

        corpus = self._load_corpus()
        if not corpus:
            return []
        # Text lives inside the normalised record's payload, not on the record.
        tokens = [
            _tokens((p.get("payload") or {}).get("text", "")) or ["<empty>"] for p in corpus
        ]
        if self._bm25 is None or len(self._bm25.doc_freqs) != len(tokens):
            self._bm25 = BM25Okapi(tokens)
        scores = self._bm25.get_scores(_tokens(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [i for i in order[:k] if scores[i] > 0]

    # -- search -------------------------------------------------------------- #
    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        config = self.config
        final_k = top_k or config.get("retrieval.final_top_k", 10)
        vec_k = config.get("retrieval.vector_top_k", 20)
        bm_k = config.get("retrieval.bm25_top_k", 20)
        rrf_k = config.get("retrieval.rrf_k", 60)
        prefix = config.get("embedding.query_prefix", "")

        corpus = self._load_corpus()
        client = self._get_client()
        embedder = self._get_embedder()

        vector_hits: list[int] = []
        try:
            qvec = embedder.encode([prefix + query])[0]
            res = client.query_points(
                collection_name=self.collection, query=qvec, limit=vec_k, with_payload=True
            )
            id_to_idx = {p["id"]: i for i, p in enumerate(corpus)}
            vector_hits = [id_to_idx[p.id] for p in res.points if p.id in id_to_idx]
        except Exception as exc:  # noqa: BLE001 - degrade to keyword-only
            log.warning("vector search failed, falling back to BM25 only: %s", exc)

        try:
            keyword_hits = self._bm25_ranked(query, bm_k)
        except Exception as exc:  # noqa: BLE001
            log.warning("BM25 failed: %s", exc)
            keyword_hits = []

        # Reciprocal Rank Fusion: rank-based, so the two score scales never mix.
        fused: dict[int, float] = {}
        for ranking in (vector_hits, keyword_hits):
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)

        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:final_k]
        out = []
        for idx, score in ordered:
            payload = dict(corpus[idx].get("payload") or {})
            out.append(
                {
                    "score": round(score, 6),
                    "vector_rank": (
                        vector_hits.index(idx) + 1 if idx in vector_hits else None
                    ),
                    "bm25_rank": (
                        keyword_hits.index(idx) + 1 if idx in keyword_hits else None
                    ),
                    **payload,
                }
            )
        return out

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None