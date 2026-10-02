"""JSONL helpers.

Writes are atomic (temp file + rename) so an interrupted run cannot leave a
half-written clauses.jsonl that later phases silently treat as complete.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator


def write_jsonl(path: Path | str, records: Iterable[dict[str, Any]]) -> int:
    """Write records as JSONL. Returns the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, default=str))
            fh.write("\n")
            n += 1
    os.replace(tmp, path)
    return n


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Read JSONL, skipping unparseable lines.

    A corrupt line yields a logged warning rather than an exception: losing one
    clause is recoverable, aborting a 20-minute run over it is not.
    """
    path = Path(path)
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                import logging

                logging.getLogger(__name__).warning(
                    "%s:%d skipping unparseable line: %s", path, lineno, exc
                )
    return out


def iter_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    yield from read_jsonl(path)


def write_json(path: Path | str, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_json(path: Path | str) -> Any:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))