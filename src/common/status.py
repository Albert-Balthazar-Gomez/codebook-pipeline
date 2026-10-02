"""Status reporting.

The project's core rule is that every phase declares SUCCESS / DEGRADED /
FAILED and never silently produces garbage. This module is the only place that
status is defined, so the rule has exactly one implementation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class Status(str, Enum):
    """Outcome of a pipeline phase."""

    SUCCESS = "SUCCESS"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"

    @property
    def exit_code(self) -> int:
        return {"SUCCESS": 0, "DEGRADED": 0, "FAILED": 1}[self.value]


@dataclass
class PhaseResult:
    """Result of one phase.

    `warnings` are recoverable problems worth surfacing (a rejected table
    candidate, a figure with no caption). Their presence forces DEGRADED at the
    latest: a phase that noticed something wrong and still claims SUCCESS is
    exactly the failure mode this project is built to avoid.
    """

    phase: str
    status: Status
    counts: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    # Reasons the output is incomplete but still usable. Distinct from
    # warnings: these describe known capability limits rather than surprises.
    degraded_reasons: list[str] = field(default_factory=list)
    error: str | None = None

    def __post_init__(self) -> None:
        # Self-correct: any recorded problem downgrades an optimistic status.
        if self.status is Status.SUCCESS and (self.warnings or self.degraded_reasons):
            self.status = Status.DEGRADED
        if self.status is Status.FAILED and not self.error:
            self.error = f"{self.phase} failed"

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        if self.status is Status.SUCCESS:
            self.status = Status.DEGRADED

    def degrade(self, reason: str) -> None:
        """Record a known capability limit (e.g. an unsupported table style)."""
        if reason not in self.degraded_reasons:
            self.degraded_reasons.append(reason)
        if self.status is Status.SUCCESS:
            self.status = Status.DEGRADED

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "status": self.status.value,
            "counts": self.counts,
            "warnings": self.warnings,
            "degraded_reasons": self.degraded_reasons,
            "error": self.error,
        }

    def summary_line(self) -> str:
        bits = [f"{self.phase}={self.status.value}"]
        for k, v in self.counts.items():
            bits.append(f"{k}={v}")
        if self.warnings:
            bits.append(f"warnings={len(self.warnings)}")
        if self.degraded_reasons:
            bits.append(f"degraded_reasons={len(self.degraded_reasons)}")
        return "  ".join(bits)


class PhaseReport:
    """Collects every phase result and writes pipeline_report.json."""

    def __init__(self, codebook: str, out_dir: Path) -> None:
        self.codebook = codebook
        self.out_dir = Path(out_dir)
        self.results: list[PhaseResult] = []

    def add(self, result: PhaseResult) -> PhaseResult:
        self.results.append(result)
        level = {
            Status.SUCCESS: logging.INFO,
            Status.DEGRADED: logging.WARNING,
            Status.FAILED: logging.ERROR,
        }[result.status]
        log.log(level, result.summary_line())
        for reason in result.degraded_reasons:
            log.log(level, "    limit: %s", reason)
        for warning in result.warnings[:10]:
            log.log(level, "    warn: %s", warning)
        if len(result.warnings) > 10:
            log.log(level, "    ... and %d more warnings", len(result.warnings) - 10)
        if result.error:
            log.log(level, "    error: %s", result.error)
        return result

    @property
    def overall(self) -> Status:
        if any(r.status is Status.FAILED for r in self.results):
            return Status.FAILED
        if any(r.status is Status.DEGRADED for r in self.results):
            return Status.DEGRADED
        return Status.SUCCESS

    def write(self, extra: dict[str, Any] | None = None) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "pipeline_report.json"
        phases = [r.to_dict() for r in self.results]

        # Carry over phases this invocation did not run, so re-running (say)
        # phase 7 alone does not erase the audit trail for phases 2-6. Without
        # this, a report is only ever accurate for whichever phases ran last,
        # which is exactly when you most want the full picture.
        ran = {r["phase"] for r in phases}
        for previous in self._previous_phases():
            if previous["phase"] not in ran:
                previous = dict(previous)
                previous["carried_over"] = True
                phases.append(previous)

        payload = {
            "codebook": self.codebook,
            "overall_status": self.overall_status_for(phases).value,
            "phases": phases,
        }
        if extra:
            payload.update(extra)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)
        log.info("wrote %s (overall=%s)", path, payload["overall_status"])
        return path

    @staticmethod
    def overall_status_for(phases: list[dict[str, Any]]) -> Status:
        statuses = {p.get("status") for p in phases}
        if Status.FAILED.value in statuses:
            return Status.FAILED
        if Status.DEGRADED.value in statuses:
            return Status.DEGRADED
        return Status.SUCCESS

    def _previous_phases(self) -> list[dict[str, Any]]:
        path = self.out_dir / "pipeline_report.json"
        if not path.exists():
            return []
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("phases", [])
        except Exception:
            return []