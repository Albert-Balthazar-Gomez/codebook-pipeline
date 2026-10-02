"""Query the vector database.

    # one-shot
    python -m src.pipeline.query "what is the limiting moment of resistance for M25"
    python -m src.pipeline.query "Cl. 26.5.1.1" --top-k 5

    # interactive
    python -m src.pipeline.query --interactive

    # other codebooks
    python -m src.pipeline.query "shear" --codebook is456 --codebook eurocode

Retrieval is hybrid (dense + BM25, fused with RRF). `--bm25-only` and
`--vector-only` exist because the two disagree in an informative way: if BM25
finds a clause by its exact number but the vector search does not, the clause is
correctly located and just not semantically distinctive.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..common.config import load_config, slugify
from ..common.status import Status


def _fmt(hit: dict, show_text: bool = True) -> str:
    stype = hit.get("source_type")
    ident = hit.get("clause_id") or hit.get("table_id") or hit.get("figure_id")
    page = hit.get("page")
    title = hit.get("clause_title") or hit.get("caption") or ""
    title = (title or "").strip()
    line = (
        f"  [{stype:<6}] {str(ident):<14} p{str(page):<4} "
        f"score={hit.get('score', 0):.5f}  vec={hit.get('vector_rank')} bm25={hit.get('bm25_rank')}"
    )
    if hit.get("link_confidence"):
        line += f"  link={hit['link_confidence']}"
    out = [line]
    if title:
        out.append(f"      {title[:100]}")
    if show_text:
        text = (hit.get("text") or "").replace("\n", " ")
        out.append(f"      {text[:220]}")
    if stype == "figure" and hit.get("png_path"):
        out.append(f"      image: {hit['png_path']}")
    return "\n".join(out)


def build_retriever(config, codebook: str):
    from ..embedding.qdrant_indexer import HybridRetriever

    db_dir = config.out_dirs(codebook)["vector_db"]
    if not db_dir.exists():
        raise SystemExit(
            f"no vector db for {codebook} at {db_dir}\n"
            f"run:  python -m src.pipeline.run --pdf <pdf> --phases 2 3 4 5 6"
        )
    return HybridRetriever(config, db_dir, codebook)


def search_once(r, query: str, top_k: int, mode: str, show_text: bool) -> list[dict]:
    if mode == "bm25-only":
        idx = r._bm25_ranked(query, top_k)
        corpus = r._load_corpus()
        return [
            {"score": 0.0, "vector_rank": None, "bm25_rank": k + 1, **dict(corpus[i]["payload"])}
            for k, i in enumerate(idx)
        ]
    if mode == "vector-only":
        from qdrant_client import QdrantClient  # noqa: F401  (import check only)

        corpus = r._load_corpus()
        id_to_idx = {p["id"]: i for i, p in enumerate(corpus)}
        client = r._get_client()
        qvec = r._get_embedder().encode(
            [config_prefix(r) + query]
        )[0]
        res = client.query_points(
            collection_name=r.collection, query=qvec, limit=top_k, with_payload=True
        )
        out = []
        for rank, p in enumerate(res.points):
            if p.id not in id_to_idx:
                continue
            out.append(
                {
                    "score": 0.0,
                    "vector_rank": rank + 1,
                    "bm25_rank": None,
                    **dict(corpus[id_to_idx[p.id]]["payload"]),
                }
            )
        return out
    return r.search(query, top_k=top_k)


def config_prefix(r) -> str:
    return r.config.get("embedding.query_prefix", "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="codebook-query", description="Query the codebook vector db")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--codebook", action="append", default=None, help="Codebook slug (repeatable). Default: all found.")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--mode", choices=("hybrid", "bm25-only", "vector-only"), default="hybrid")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--no-text", action="store_true", help="Hide the chunk text")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    config = load_config()
    if args.config:
        import yaml

        from ..common.config import deep_merge

        config.data = deep_merge(config.data, yaml.safe_load(args.config.read_text()) or {})

    codebooks = args.codebook
    if not codebooks:
        root = config.path("vector_db")
        codebooks = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []
    if not codebooks:
        raise SystemExit(f"no indexed codebooks under {config.path('vector_db')}")

    retrievers = []
    for cb in codebooks:
        try:
            retrievers.append((cb, build_retriever(config, cb)))
        except SystemExit as exc:
            print(f"skipping {cb}: {exc}")
    if not retrievers:
        raise SystemExit("nothing to query")

    top_k = args.top_k or config.get("retrieval.final_top_k", 10)

    def show(query: str) -> None:
        for cb, r in retrievers:
            print(f"\n=== {query!r}  [{cb}] ===")
            hits = search_once(r, query, top_k, args.mode, not args.no_text)
            if not hits:
                print("  (no hits)")
            for hit in hits:
                print(_fmt(hit, not args.no_text))

    if args.interactive:
        print("type a query, or 'quit'")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if line.lower() in ("quit", "exit", "q", ""):
                break
            show(line)
        for _, r in retrievers:
            r.close()
        return 0

    if not args.query:
        parser.error("pass a query, or --interactive")
    show(args.query)
    for _, r in retrievers:
        r.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())