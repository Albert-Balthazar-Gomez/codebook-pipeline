"""Phase 7: instruction-tuning dataset generation.

The hard constraints, all enforced in code rather than trusted to the prompt:

1.  Every record carries `source_chunk_id`, so any generated claim is traceable
    back to the exact clause, table or figure it came from.
2.  No design-calculation questions. An LLM asked to size a member will produce
    a fluent, confidently wrong number, and that number is the most dangerous
    thing in a codebook-derived dataset because it looks authoritative.
    `forbid_calculation_intent` rejects such items after generation, and a
    numeric-grounding check rejects answers whose numbers do not appear in the
    source text.
3.  A question whose answer lives only inside an image is answered with a
    pointer to the figure rather than an invented description.

The model key is read from GEMINI_API_KEY in the environment and is never
written to disk.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
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
5. Cover different aspects: definitions, requirements, limits, cross-references.

Return ONLY a JSON array, no prose, no code fences. Each element:
{{"instruction": "<question>", "input": "", "output": "<answer>"}}"""

_CALC_RX = re.compile(
    r"(?i)\b("
    r"calculat\w*|comput\w*|work\s*out|"
    # "determine the depth/position/value..." always implies deriving a number.
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


def _numbers(text: str) -> set[str]:
    return set(_NUM_RX.findall(text or ""))


def looks_like_calculation(text: str) -> bool:
    return bool(_CALC_RX.search(text or ""))


def is_grounded(answer: str, source: str, tolerance: float = 0.0) -> tuple[bool, str]:
    """Every number in the answer must occur in the source.

    A model that invents "0.184" for a table reading "0.183" fails here. The
    check is deliberately blunt: it can miss a hallucinated number that happens to
    appear elsewhere in the source, but it catches the common case cheaply.
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


def build_prompt(chunk: dict[str, Any], n: int) -> str:
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
        figure_id = chunk.get("figure_id")
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


def _parse_json_array(text: str) -> list[dict[str, Any]]:
    """Extract the JSON array, tolerating fences and stray prose."""
    if not text:
        return []
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE)
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


# Error classes. Transient ones are retried; the rest abort the run, because
# retrying a quota-exhausted key just burns wall-clock and produces nothing.
TRANSIENT = "transient"
QUOTA = "quota"
AUTH = "auth"
CONFIG = "config"


def classify_api_error(exc: Exception) -> tuple[str, str]:
    """Bucket an SDK exception so the caller knows whether retrying can help."""
    from google.genai import errors as genai_errors  # type: ignore

    if isinstance(exc, genai_errors.ClientError):
        code = getattr(getattr(exc, "code", None), "value", None) or getattr(exc, "code", None)
        status = int(code) if isinstance(code, int) else 0
        text = str(exc)
        if status == 429 or "RESOURCE_EXHAUSTED" in text:
            if "quota" in text.lower() or "billing" in text.lower():
                return QUOTA, "API quota exhausted (429 RESOURCE_EXHAUSTED)"
            return TRANSIENT, "rate limited (429)"
        if status in (401, 403) or "API key" in text or "permission" in text.lower():
            return AUTH, f"authentication/permission error: {text[:140]}"
        if status == 404 or "not found" in text.lower() or "no longer available" in text.lower():
            return CONFIG, f"model unavailable: {text[:140]}"
        return CONFIG, f"client error {status}: {text[:140]}"
    if isinstance(exc, genai_errors.ServerError):
        return TRANSIENT, f"server error {getattr(exc, 'code', '?')}: {str(exc)[:120]}"
    if isinstance(exc, (genai_errors.APIError, TimeoutError, ConnectionError)):
        return TRANSIENT, f"{type(exc).__name__}: {str(exc)[:120]}"
    return TRANSIENT, f"{type(exc).__name__}: {str(exc)[:140]}"


def _call_gemini(
    prompt: str, config: Config, attempt: int = 0
) -> tuple[str | None, str | None, str]:
    """One generation attempt.

    Returns (text, error_kind, error_message). `error_kind` is None on success and
    otherwise one of TRANSIENT / QUOTA / AUTH / CONFIG.

    API errors are caught and returned rather than raised: a single transient 503
    must not abort a run that has already paid for every chunk before it.
    """
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except ImportError as exc:
        return None, CONFIG, f"google-genai not installed ({exc}); pip install -r requirements-embeddings.txt"

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None, AUTH, "GEMINI_API_KEY is not set"

    model = config.get("dataset_generation.model", "gemini-3.8-flash")
    try:
        client = genai.Client(api_key=api_key)
        cfg = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=config.get("dataset_generation.temperature", 0.4),
            max_output_tokens=config.get("dataset_generation.max_output_tokens", 2048),
            response_mime_type="application/json",
        )
        resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
        return resp.text, None, ""
    except Exception as exc:  # noqa: BLE001
        kind, message = classify_api_error(exc)
        return None, kind, message


def _call_openai_compatible(
    prompt: str, config: Config, attempt: int
) -> tuple[str | None, str | None, str]:
    """OpenAI-compatible backend (DeepSeek, Groq, Together, vLLM, LM Studio...).

    Uses urllib rather than the `openai` SDK so the pipeline gains a second
    provider without a new dependency, and so pointing at a self-hosted
    OpenAI-compatible server needs nothing but a base URL.
    """
    key_env = config.get("dataset_generation.api_key_env", "LLM_API_KEY")
    api_key = os.environ.get(key_env) or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None, AUTH, (
            f"${key_env} is not set (an OpenAI-compatible key is required for this provider)"
        )

    base = (config.get("dataset_generation.base_url") or "https://api.deepseek.com/v1").rstrip("/")
    model = config.get("dataset_generation.model") or "deepseek-chat"
    max_tokens = config.get("dataset_generation.max_output_tokens", 2048)
    temperature = config.get("dataset_generation.temperature", 0.4)

    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        # JSON mode keeps the response parseable; the parser also tolerates a
        # fenced array in case the server ignores this.
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=config.get("dataset_generation.timeout_seconds", 120)) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"], None, ""
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode()[:300]
        except Exception:
            pass
        if exc.code in (401, 403):
            return None, AUTH, f"authentication error {exc.code}: {body}"
        if exc.code == 402:
            # DeepSeek and several others report an exhausted balance as 402.
            return None, QUOTA, f"payment required (402): account balance exhausted. {body}"
        if exc.code == 429:
            if "balance" in body.lower() or "quota" in body.lower() or "billing" in body.lower():
                return None, QUOTA, f"quota/billing exhausted (429): {body}"
            return None, TRANSIENT, f"rate limited (429): {body}"
        if 500 <= exc.code < 600:
            return None, TRANSIENT, f"server error {exc.code}: {body}"
        return None, CONFIG, f"unexpected HTTP {exc.code}: {body}"
    except urllib.error.URLError as exc:
        return None, TRANSIENT, f"network error: {str(exc)[:140]}"
    except Exception as exc:  # noqa: BLE001
        return None, TRANSIENT, f"{type(exc).__name__}: {str(exc)[:140]}"


def _call_model(
    prompt: str, config: Config, attempt: int = 0
) -> tuple[str | None, str | None, str]:
    """Dispatch to the configured provider. See _call_gemini / _call_openai_compatible."""
    provider = (config.get("dataset_generation.provider") or "gemini").lower()
    if provider == "gemini":
        return _call_gemini(prompt, config, attempt)
    if provider in ("openai_compatible", "openai", "deepseek"):
        return _call_openai_compatible(prompt, config, attempt)
    return None, CONFIG, f"unknown dataset_generation.provider={provider!r}"


def generate_dataset(
    chunks: list[dict[str, Any]], config: Config, out_dir: Path
) -> PhaseResult:
    result = PhaseResult(phase="phase7_dataset", status=Status.SUCCESS)
    out_dir = Path(out_dir)

    if not config.get("dataset_generation.enabled", True):
        result.status = Status.FAILED
        result.error = "dataset_generation.enabled=false"
        return result

    provider = (config.get("dataset_generation.provider") or "gemini").lower()
    if provider == "gemini":
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            result.status = Status.FAILED
            result.error = (
                "GEMINI_API_KEY is not set. Export it and re-run, e.g. "
                "`export GEMINI_API_KEY=...`. The key is read from the environment "
                "only and is never written to disk."
            )
            return result
    else:
        key_env = config.get("dataset_generation.api_key_env", "LLM_API_KEY")
        if not (
            os.environ.get(key_env)
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        ):
            result.status = Status.FAILED
            result.error = (
                f"no API key found for provider {provider!r}. Set ${key_env} (or "
                "DEEPSEEK_API_KEY / OPENAI_API_KEY) in the environment."
            )
            return result

    max_chunks = config.get("dataset_generation.max_chunks", 500)
    n_qa = config.get("dataset_generation.qa_per_chunk", 3)
    rpm = config.get("dataset_generation.requests_per_minute", 10)
    retries = config.get("dataset_generation.max_retries", 3)
    forbid_calc = config.get("dataset_generation.forbid_calculation_intent", True)

    # Figures first: they are the scarce, high-value chunks.
    ordered = sorted(chunks, key=lambda c: {"figure": 0, "table": 1}.get(c.get("type"), 2))
    selected = ordered[:max_chunks]

    records: list[dict[str, Any]] = []
    rejected_calc = 0
    rejected_ungrounded = 0
    rejected_malformed = 0
    failed_chunks = 0

    interval = 60.0 / max(1, rpm)
    aborted_reason: str | None = None
    aborted_at: int | None = None
    chunks_done = 0
    for idx, chunk in enumerate(selected):
        prompt = build_prompt(chunk, n_qa)
        raw = None
        for attempt in range(retries):
            if idx or attempt:
                # Rate limit plus exponential backoff: overload errors need more
                # than the flat interval, and a hot loop just re-triggers them.
                time.sleep(interval * (2**attempt))
            raw, kind, message = _call_model(prompt, config, attempt)
            if raw:
                break
            if kind != TRANSIENT:
                # Quota exhausted, bad key, or an unavailable model: retrying
                # cannot help and every further call just stalls the run.
                aborted_reason = f"{kind}: {message}"
                aborted_at = idx
                log.error("aborting Phase 7 at chunk %d/%d -- %s", idx + 1, len(selected), aborted_reason)
                break
            log.warning("retrying chunk %d (%d/%d): %s", idx + 1, attempt + 1, retries, message)
        if aborted_reason:
            break
        if not raw:
            failed_chunks += 1
            continue
        chunks_done += 1

        pairs = _parse_json_array(raw)
        if not pairs:
            rejected_malformed += 1
            continue

        source_text = (
            chunk.get("embed_text")
            or chunk.get("markdown")
            or chunk.get("text")
            or chunk.get("caption")
            or ""
        )
        figure_id = chunk.get("figure_id")

        for pair in pairs:
            if not isinstance(pair, dict):
                rejected_malformed += 1
                continue
            instruction = (pair.get("instruction") or "").strip()
            output = (pair.get("output") or "").strip()
            if not instruction or not output:
                rejected_malformed += 1
                continue

            if forbid_calc and (looks_like_calculation(instruction) or looks_like_calculation(output)):
                rejected_calc += 1
                continue

            ok, reason = is_grounded(output, source_text)
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
                    "grounding": "numbers_verified" if _numbers(output) else "no_numbers",
                }
            )

    result.counts["chunks_selected"] = len(selected)
    result.counts["chunks_completed"] = chunks_done
    result.counts["chunks_failed"] = failed_chunks
    result.counts["records"] = len(records)
    result.counts["rejected_calculation"] = rejected_calc
    result.counts["rejected_ungrounded"] = rejected_ungrounded
    result.counts["rejected_malformed"] = rejected_malformed
    if aborted_reason:
        result.counts["aborted_at_chunk"] = aborted_at + 1
        result.counts["chunks_not_attempted"] = len(selected) - (aborted_at or 0)

    # Expected filtering is not a fault: the guards working is the point, and a
    # run that kept every candidate while inventing numbers would be worse. So
    # rejections are counted and logged, and only an implausibly high rejection
    # rate -- evidence the model is ignoring the prompt -- counts as a warning
    # that degrades the phase.
    produced = len(records) + rejected_calc + rejected_ungrounded + rejected_malformed
    if produced:
        reject_rate = 100.0 * (rejected_calc + rejected_ungrounded + rejected_malformed) / produced
        result.counts["rejection_rate_pct"] = round(reject_rate, 1)
    if rejected_calc:
        log.info(
            "%d candidate Q/A pair(s) rejected as design-calculation requests; an "
            "LLM sizing a member invents plausible numbers",
            rejected_calc,
        )
    if rejected_ungrounded:
        log.info(
            "%d candidate Q/A pair(s) rejected: the answer contained numbers absent "
            "from the source chunk",
            rejected_ungrounded,
        )
    if rejected_malformed:
        log.info("%d candidate Q/A pair(s) were malformed and discarded", rejected_malformed)
    if produced and reject_rate > 50.0:
        result.warn(
            f"{reject_rate:.0f}% of generated Q/A pairs were rejected "
            f"({rejected_calc} calculation, {rejected_ungrounded} ungrounded, "
            f"{rejected_malformed} malformed). The model may be ignoring the "
            "grounding rules; check the generated records before using them."
        )
    if failed_chunks:
        result.warn(f"{failed_chunks} chunk(s) produced no usable response after {retries} attempts")
    if aborted_reason:
        # Partial dataset is still written: whatever was generated before the
        # abort is valid and traceable. Callers get DEGRADED, not a fake SUCCESS.
        result.error = aborted_reason
        result.status = Status.DEGRADED if records else Status.FAILED
        result.degrade(
            f"generation stopped early: {aborted_reason}. "
            f"{len(records)} record(s) were produced from the chunks processed before "
            "the abort; the rest were never attempted."
        )
    if not records and result.status is not Status.FAILED:
        result.status = Status.FAILED
        result.error = "no instruction records survived validation"

    write_jsonl(out_dir / "instruction_dataset.jsonl", records)
    result.counts["output"] = str(out_dir / "instruction_dataset.jsonl")
    return result