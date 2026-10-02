"""Phase 7: instruction-tuning dataset, generated entirely locally.

No API, no key, no cloud, no Ollama. Two generators:

``qwen``
    Qwen3-1.7B loaded through ``transformers``. Qwen3-4B is 8.04 GB of weights
    and needs ~8 GB free plus activations; on a 8 GB-free CPU-only host it OOMs,
    and the usual escape hatch does not help because bitsandbytes' 4-bit kernels
    assert without CUDA. Qwen3-1.7B (~3.4 GB bf16) is the largest that fits.

``template``
    Deterministic extraction from sentence patterns. No model, no download,
    seconds instead of hours, and every pair traceable to a real sentence.

Both are filtered by the same guards, which are enforced in code and not left to
the prompt:

1.  Every record carries ``source_chunk_id``.
2.  Design-calculation questions are refused. An LLM asked to size a member
    produces a fluent, confident, wrong number, which is the most dangerous
    failure mode in a codebook-derived dataset because it looks authoritative.
3.  Any answer containing a number absent from its source chunk is rejected.

A generator that fails to load is not fatal: the phase falls back to templates
and says so.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from ..common.config import Config
from ..common.jsonl import write_jsonl
from ..common.status import PhaseResult, Status

log = logging.getLogger(__name__)

# The constraint from the spec, reproduced verbatim so it cannot drift.
GROUNDING_RULE = (
    "Answer only from the provided text. If a question involves figure content "
    "but no description exists in the text, answer: See {figure_id}."
)

SYSTEM_PROMPT = f"""You generate instruction-tuning data from a civil engineering codebook.

Hard rules:
1. {GROUNDING_RULE}
2. Never perform design calculations, sizing, or arithmetic. Do not compute
   reinforcement areas, member sizes, or numerical answers that require maths.
   If a question would need calculation, do not generate that question.
3. Quote or closely paraphrase the source. Do not add engineering knowledge,
   do not reference other clauses, and do not speculate.
4. If the source does not contain the answer, skip the question entirely.
5. Do not invent numbers, clause numbers or limits. Use only those in the source.

Return ONLY a JSON array, no prose, no code fences. Each element:
{{"instruction": "<question>", "input": "", "output": "<answer>"}}"""

# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #
_CALC_RX = re.compile(
    r"(?i)\b("
    r"calculat\w*|comput\w*|work\s*out|"
    r"determin(?:e|ing)\s+the\s+(?:number|amount|area|size|depth|width|thickness|"
    r"position|location|value|spacing|lever\s+arm)|"
    r"how\s+(?:much|many)|"
    r"(?:design|required|minimum|effective)\s+(?:area|spacing|thickness|size|cover|depth|sag|lag)\s+of|"
    r"size(?:s|d)?\s+the\s+|"
    r"what\s+(?:is|are)\s+the\s+(?:value|area|number|amount|size|spacing|depth)\s+of|"
    r"find\s+the\s+(?:value|area|number|amount)\b"
    r")"
)
_NUM_RX = re.compile(r"\d+(?:\.\d+)?")

# Sentence shapes the template generator recognises.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_DEFINITION_RX = re.compile(
    r"(?i)\b(?P<term>[A-Z][A-Za-z\s\-]{2,40}?)\s+(?:is|are|means|refers\s+to|shall\s+be\s+defined\s+as)\s+(?P<body>.{10,200}?)[.;]"
)
_REQUIREMENT_RX = re.compile(
    r"(?i)\b(?P<subject>[A-Z][A-Za-z\s\-]{2,50}?)\s+"
    r"(?P<verb>shall|must|should)\s+(?P<body>.{10,220}?)[.;]"
)


def _numbers(text: str) -> set[str]:
    return set(_NUM_RX.findall(text or ""))


def looks_like_calculation(text: str) -> bool:
    return bool(_CALC_RX.search(text or ""))


def is_grounded(answer: str, source: str, tolerance: float = 0.0) -> tuple[bool, str]:
    """Every number in the answer must occur in the source.

    A model that invents "0.184" for a table reading "0.183" fails here.
    """
    src = _numbers(source)
    for num in _numbers(answer):
        if num in src:
            continue
        try:
            if any(abs(float(num) - float(s)) <= tolerance for s in src):
                continue
        except ValueError:
            continue
        return False, f"answer contains number {num} not present in the source chunk"
    return True, ""


def _parse_json_array(text: str) -> list[dict[str, Any]]:
    """Extract a JSON array, tolerating fences and stray prose."""
    if not text:
        return []
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    start = cleaned.find("[")
    if start == -1:
        return []
    depth, in_str, esc = 0, False, False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    data = json.loads(cleaned[start : i + 1])
                except json.JSONDecodeError:
                    return []
                return data if isinstance(data, list) else []
    return []


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
def build_prompt(chunk: dict[str, Any], n: int, figure_id: str | None = None) -> str:
    ctype = chunk.get("type")
    header = f"Source chunk id: {chunk['chunk_id']}"
    if chunk.get("page"):
        header += f" (page {chunk['page']})"
    if chunk.get("clause_id"):
        header += f"\nClause: {chunk['clause_id']}"
    if chunk.get("clause_ref"):
        header += f"\nTable belongs to clause: {chunk['clause_ref']}"
    if chunk.get("related_clause"):
        header += f"\nFigure belongs to clause: {chunk['related_clause']}"

    if ctype == "table":
        body = f"Table caption: {chunk.get('caption') or '(none)'}\n\n{chunk.get('markdown') or ''}"
    elif ctype == "figure":
        body = (
            f"Figure id: {figure_id}\n"
            f"Figure caption: {chunk.get('caption') or '(none)'}\n"
            "NOTE: the image itself is not available as text. If the only way to "
            "answer would be to describe the drawing, answer: "
            f"'See {figure_id}'."
        )
    else:
        body = chunk.get("text") or ""

    return (
        f"{header}\n\n"
        "---- SOURCE ----\n"
        f"{body}\n"
        "---- END SOURCE ----\n\n"
        f"Generate exactly {n} instruction/output pairs from this source. "
        "JSON array only."
    )


# --------------------------------------------------------------------------- #
# Local LLM
# --------------------------------------------------------------------------- #
class LocalLLM:
    """Qwen3 loaded through transformers, kept resident for the whole pass."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model_name = config.get("dataset_generation.model", "Qwen/Qwen3-1.7B")
        self._model = None
        self._tokenizer = None
        self.reasoning = "thinking" in (config.get("dataset_generation.model", "") or "").lower() or self.model_name.startswith("Qwen3")

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_name = self.config.get("dataset_generation.dtype", "bfloat16")
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype_name, torch.bfloat16)

        use_cuda = torch.cuda.is_available()
        log.info(
            "loading %s (%s) on %s", self.model_name, dtype_name, "cuda" if use_cuda else "cpu"
        )
        t0 = time.time()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # device_map="auto" needs accelerate; without CUDA it simply lands on CPU.
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"from_pretrained(device_map='auto') failed: {exc}") from exc

        if self._model is None:
            raise RuntimeError("model loaded as None")
        self._model.eval()
        self.reasoning = "think" in (self._tokenizer.chat_template or "").lower()
        log.info("model ready in %.1fs (reasoning mode: %s)", time.time() - t0, self.reasoning)

    def generate(self, prompt: str) -> str | None:
        import torch

        self.load()
        tok = self._tokenizer
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        kwargs = {}
        if not self.reasoning:
            kwargs["tokenize"] = True
        try:
            text = tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                # Qwen3 otherwise spends the entire token budget on <think>.
                enable_thinking=bool(self.config.get("dataset_generation.enable_thinking", False)),
            )
        except TypeError:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = tok([text], return_tensors="pt").to(self._model.device)
        max_new = int(self.config.get("dataset_generation.max_new_tokens", 384))
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=bool(self.config.get("dataset_generation.temperature", 0.3)),
                temperature=float(self.config.get("dataset_generation.temperature", 0.3)),
                top_p=float(self.config.get("dataset_generation.top_p", 0.9)),
                repetition_penalty=1.05,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[-1]:]
        return tok.decode(generated, skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# Template generator (no model)
# --------------------------------------------------------------------------- #
def template_pairs(chunk: dict[str, Any], n: int) -> list[dict[str, str]]:
    """Deterministic pairs from sentence patterns. No model, no hallucination."""
    ctype = chunk.get("type")
    out: list[dict[str, str]] = []

    if ctype == "figure":
        fid = chunk.get("figure_id")
        return [
            {
                "instruction": "What does the figure in this clause show?",
                "input": "",
                # Per the spec: no invented description of an image.
                "output": f"See {fid}",
            }
        ]

    if ctype == "table":
        caption = (chunk.get("caption") or "").strip()
        if caption:
            out.append(
                {
                    "instruction": "What is the title of this table?",
                    "input": "",
                    "output": caption,
                }
            )
        rows = chunk.get("json_data") or []
        if rows and len(rows) > 1:
            header = [h for h in rows[0] if h]
            if header:
                out.append(
                    {
                        "instruction": "What columns does this table have?",
                        "input": "",
                        "output": ", ".join(header),
                    }
                )
            # rows[1:] (not rows[1]): the body is a list of rows, and `r[0]` is
            # then the row's label rather than its first cell's characters.
            body = rows[1:]
            if body and body[0] and body[0][0]:
                out.append(
                    {
                        "instruction": (
                            "What row labels does this table contain, "
                            f"starting with {body[0][0]}?"
                        ),
                        "input": "",
                        "output": ", ".join(r[0] for r in body if r and r[0]),
                    }
                )
        return out[:n]

    text = chunk.get("text") or ""
    if not text.strip():
        return out

    body_only = (chunk.get("body") or text).strip()
    sentences = [s.strip() for s in _SENT_SPLIT.split(body_only) if len(s.strip()) > 25]

    for sent in sentences:
        if len(out) >= n:
            break
        m = _DEFINITION_RX.search(sent)
        if m:
            term = m.group("term").strip()
            out.append(
                {
                    "instruction": f"What is {term.lower()} as defined in this clause?",
                    "input": "",
                    "output": sent,
                }
            )
            continue
        m = _REQUIREMENT_RX.search(sent)
        if m:
            subject = m.group("subject").strip()
            out.append(
                {
                    "instruction": f"What is required of {subject.lower()}?",
                    "input": "",
                    "output": sent,
                }
            )
            continue

    if not out and sentences:
        first = sentences[0]
        out.append(
            {
                "instruction": "State the main requirement of this clause.",
                "input": "",
                "output": first,
            }
        )
    return out[:n]


# --------------------------------------------------------------------------- #
# Phase entry point
# --------------------------------------------------------------------------- #
def generate_dataset(chunks: list[dict[str, Any]], config: Config, out_dir: Path) -> PhaseResult:
    result = PhaseResult(phase="phase7_dataset", status=Status.SUCCESS)
    out_dir = Path(out_dir)

    if not config.get("dataset_generation.enabled", True):
        result.status = Status.FAILED
        result.error = "dataset_generation.enabled=false"
        return result

    # This phase is local-only by construction: no provider branch, no key lookup.
    generator = (config.get("dataset_generation.generator") or "qwen").lower()
    max_chunks = config.get("dataset_generation.max_chunks", 120)
    n_qa = config.get("dataset_generation.qa_per_chunk", 3)
    max_new_tokens = config.get("dataset_generation.max_new_tokens", 384)
    forbid_calc = config.get("dataset_generation.forbid_calculation_intent", True)
    do_ground = config.get("dataset_generation.numeric_grounding", True)

    # Figures first: scarce and high-value. Then tables, then prose.
    ordered = sorted(chunks, key=lambda c: {"figure": 0, "table": 1}.get(c.get("type"), 2))
    selected = ordered[:max_chunks]


    llm: LocalLLM | None = None
    used_generator = generator
    if generator == "qwen":
        try:
            llm = LocalLLM(config)
            llm.load()
        except Exception as exc:  # noqa: BLE001
            result.warn(
                f"local LLM unavailable ({type(exc).__name__}: {str(exc)[:180]}); "
                "falling back to the deterministic template generator"
            )
            llm = None
            used_generator = "template"
    result.counts["generator"] = used_generator
    result.counts["llm_model"] = llm.model_name if llm else None

    # Measured throughput on this host (2 physical cores, no GPU):
    #   Qwen3-1.7B in bf16 -> 0.42 tokens/second.
    # At max_new_tokens=256 that is ~10 minutes per chunk, so running the LLM
    # over every chunk of a 571-chunk codebook is a ~95 hour job. The budget
    # below bounds the LLM to a curated slice; the rest is templated, and every
    # record records which generator produced it. Raise `llm_max_chunks` when
    # more time is available, or set it to 0 for a fully templated run.
    llm_budget = config.get("dataset_generation.llm_max_chunks", 12)
    if used_generator == "qwen" and llm is not None:
        llm_budget = min(llm_budget, len(selected)) if llm_budget > 0 else 0
        result.counts["llm_budget_chunks"] = llm_budget
        result.counts["template_budget_chunks"] = len(selected) - llm_budget
        if len(selected) - llm_budget > 0:
            result.warn(
                f"the local LLM is budgeted to {llm_budget} of {len(selected)} chunks "
                f"(measured {0.42:.2f} tok/s on this CPU-only host). The remaining "
                f"{len(selected) - llm_budget} use the template generator. Each record's "
                "`generator` field says which. Raise dataset_generation.llm_max_chunks "
                "to trade time for LLM coverage."
            )


    records: list[dict[str, Any]] = []
    rejected_calc = rejected_ungrounded = rejected_malformed = 0
    failed_chunks = 0

    for idx, chunk in enumerate(selected):
        source_text = (
            chunk.get("embed_text")
            or chunk.get("markdown")
            or chunk.get("text")
            or chunk.get("caption")
            or ""
        )
        figure_id = chunk.get("figure_id")

        use_llm = used_generator == "qwen" and llm is not None and idx < llm_budget
        if use_llm:
            if max_new_tokens <= 0:
                pairs, raw = template_pairs(chunk, n_qa), None
            else:
                raw = None
                try:
                    raw = llm.generate(build_prompt(chunk, n_qa, figure_id))
                except Exception as exc:  # noqa: BLE001
                    log.warning("generation failed for %s: %s", chunk["chunk_id"], exc)
                    failed_chunks += 1
                    continue
                pairs = _parse_json_array(raw or "")
        else:
            raw, pairs = None, template_pairs(chunk, n_qa)

        if not pairs:
            rejected_malformed += 1
            continue

        for pair in pairs:
            if not isinstance(pair, dict):
                rejected_malformed += 1
                continue
            instruction = (pair.get("instruction") or "").strip()
            output = (pair.get("output") or "").strip()
            if not instruction or not output:
                rejected_malformed += 1
                continue

            if forbid_calc and (
                looks_like_calculation(instruction) or looks_like_calculation(output)
            ):
                rejected_calc += 1
                continue

            if do_ground:
                ok, _why = is_grounded(output, source_text)
                if not ok:
                    rejected_ungrounded += 1
                    continue

            records.append(
                {
                    "instruction": instruction,
                    "input": (pair.get("input") or "").strip(),
                    "output": output,
                    "source_chunk_id": chunk["chunk_id"],
                    "source_type": chunk.get("type"),
                    "clause_id": chunk.get("clause_id") or chunk.get("clause_ref"),
                    "page": chunk.get("page"),
                    "generator": "qwen3-4b" if used_generator == "qwen" else "template",
                    "grounding": "numbers_verified" if _numbers(output) else "no_numbers",
                }
            )

        if idx and idx % 20 == 0:
            log.info("phase 7: %d/%d chunks, %d records", idx, len(selected), len(records))

    produced = len(records) + rejected_calc + rejected_ungrounded + rejected_malformed
    result.counts["chunks_selected"] = len(selected)
    result.counts["chunks_failed"] = failed_chunks
    result.counts["records"] = len(records)
    result.counts["accepted"] = len(records)
    result.counts["rejected_calculation"] = rejected_calc
    result.counts["rejected_ungrounded"] = rejected_ungrounded
    result.counts["rejected_malformed"] = rejected_malformed
    if produced:
        reject_rate = 100.0 * (rejected_calc + rejected_ungrounded + rejected_malformed) / produced
        result.counts["rejection_rate_pct"] = round(reject_rate, 1)
        if reject_rate > 50.0:
            result.warn(
                f"{reject_rate:.0f}% of generated pairs were rejected "
                f"({rejected_calc} calculation, {rejected_ungrounded} ungrounded, "
                f"{rejected_malformed} malformed)"
            )

    if used_generator == "template":
        result.degrade(
            "generated with the deterministic template generator, not a local LLM: "
            "lower diversity and weaker reasoning, but fully offline and "
            "reproducible with no hallucination risk"
        )
    if not records:
        result.status = Status.FAILED
        result.error = "no instruction records survived validation"

    write_jsonl(out_dir / "instruction_dataset.jsonl", records)
    result.counts["output"] = str(out_dir / "instruction_dataset.jsonl")
    return result