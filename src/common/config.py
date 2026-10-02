"""Configuration loading.

configs/default.yaml holds the pipeline thresholds; configs/standards/<name>.yaml
holds the per-standard clause grammar. Both are deep-merged so a partial
override file only needs the keys it changes.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# src/common/config.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_LOCATOR = PROJECT_ROOT / ".tools_locator.json"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


@dataclass
class Config:
    data: dict[str, Any]
    root: Path = PROJECT_ROOT

    # -- access ---------------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        """Fetch `a.b.c`, returning `default` if any level is absent."""
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, dotted: str) -> Any:
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise KeyError(dotted)
        return value

    # -- paths ----------------------------------------------------------------
    def path(self, key: str) -> Path:
        """Resolve a configured path against the project root."""
        raw = self.get(f"paths.{key}")
        if raw is None:
            raise KeyError(f"paths.{key} not configured")
        p = Path(raw)
        return p if p.is_absolute() else (self.root / p)

    def tools_dir(self) -> Path:
        """Where sbt / JDK / figure-extractor were installed.

        scripts/setup_deps.sh relocates these out of the repo when the repo path
        contains whitespace (sbt word-splits its options), and records the real
        location in .tools_locator.json.
        """
        if _LOCATOR.exists():
            import json

            try:
                return Path(json.loads(_LOCATOR.read_text())["tools_dir"])
            except Exception:
                pass
        return self.root / self.get("paths.tools", "tools")

    def out_dirs(self, codebook: str) -> dict[str, Path]:
        return {
            "extracted": self.path("extracted") / codebook,
            "processed": self.path("processed") / codebook,
            "vector_db": self.path("vector_db") / codebook,
        }

    # -- standard grammar -----------------------------------------------------
    def standard_cfg(self, name: str | None = None) -> dict[str, Any]:
        name = name or self.get("codebook.standard", "is456")
        path = PROJECT_ROOT / "configs" / "standards" / f"{name}.yaml"
        return _load_yaml(path)

    def clause_regexes(self, name: str | None = None) -> dict[str, re.Pattern[str]]:
        """Compile the standard's clause grammar once, with clear failures."""
        cfg = self.standard_cfg(name)
        out: dict[str, re.Pattern[str]] = {}
        for key in ("pattern", "pattern_number_only", "reference_pattern"):
            raw = cfg.get("clause", {}).get(key)
            if raw:
                try:
                    out[key] = re.compile(raw)
                except re.error as exc:
                    raise ValueError(
                        f"configs/standards/{name}.yaml: bad regex for clause.{key}: {exc}"
                    ) from exc
        annex = cfg.get("annex", {}).get("pattern")
        if annex:
            out["annex"] = re.compile(annex)
        for key in ("figure_pattern", "table_pattern"):
            raw = cfg.get("caption", {}).get(key)
            if raw:
                out[key] = re.compile(raw)
        return out

    def standard_display_name(self, name: str | None = None) -> str:
        return self.standard_cfg(name).get("standard", name or "unknown")


def load_config(overrides: dict[str, Any] | None = None, config_path: Path | None = None) -> Config:
    base = _load_yaml(config_path or (PROJECT_ROOT / "configs" / "default.yaml"))
    if overrides:
        base = deep_merge(base, overrides)
    return Config(base)


def slugify(name: str) -> str:
    """Filesystem-safe slug that stays readable."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return re.sub(r"_+", "_", slug) or "codebook"