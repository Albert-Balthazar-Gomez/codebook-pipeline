# Universal Codebook → Multimodal AI Pipeline

Turns a born-digital technical codebook PDF (IS 456, Eurocode, ACI 318, BS 8110, …)
into hierarchical clause text, extracted figures and tables, a Qdrant vector
database, and an instruction-tuning dataset.

**Core rule: every phase declares `SUCCESS` / `DEGRADED` / `FAILED`.** Nothing
reports success it did not achieve. `pipeline_report.json` is the source of truth
and the process exits non-zero if any phase FAILED.

---

## Quick start

```bash
# 1. Python environment (no sudo needed)
bash scripts/setup_env.sh

# 2. Phase 4 dependency: Java 11 JDK + sbt + figure-extractor + pdffigures2.jar
bash scripts/setup_deps.sh

# 3. Run phases 2-5 on a PDF
cp /path/to/is456_excerpt.pdf data/raw/
.venv/bin/python -m src.pipeline.run --pdf data/raw/is456_excerpt.pdf --standard is456

# 4. Query it
.venv/bin/python -m src.pipeline.query "what is the limiting moment of resistance"
```

---

## Adding a new codebook

1. **Drop the PDF** into `data/raw/`.
2. **Pick the grammar.** Look for `configs/standards/<name>.yaml`, or copy
   `is456.yaml` and edit the regexes:

   ```bash
   cp configs/standards/is456.yaml configs/standards/mycode.yaml
   # edit standard:, clause.pattern, clause.max_depth, caption.*
   ```
3. **Run it:**
   ```bash
   .venv/bin/python -m src.pipeline.run \
       --pdf data/raw/mycode.pdf --standard mycode --name mycode
   ```
4. **Check the report.** If Phase 2 says `FAILED: no clause headings matched`, the
   regex is wrong — tune `clause.pattern` against the real document. If it finds
   depth-1 clauses only, `max_depth` or the pattern needs loosening.

Everything downstream (phases 3–7) reads from the clause file, so a wrong Phase 2
grammar invalidates the whole run. Fix it there first.

To process everything in `data/raw/` at once:

```bash
.venv/bin/python -m src.pipeline.run --all
```

---

## Running the pipeline

```bash
python -m src.pipeline.run --pdf <pdf> [--phases 2 3 4 5 6 7] \
                           [--standard is456|eurocode|aci318|bs8110] \
                           [--name <slug>] [--config extra.yaml] [--log-level DEBUG]
```

| Phase | What it does | Output |
|---|---|---|
| 2 | Clause tree: running heads stripped, anchored regex, hierarchy from number depth | `data/extracted/<cb>/text/clauses.jsonl` |
| 3 | Tables: `find_tables` → `is_real_table` gate → markdown | `data/extracted/<cb>/tables/tables.jsonl` |
| 4 | Figures: caption-anchored detection + PDFFigures 2.0, merged | `data/extracted/<cb>/figures/*.png`, `figures_metadata.jsonl` |
| 5 | Link clause↔figure↔table, then chunk | `data/processed/<cb>/multimodal_chunks.jsonl` |
| 6 | Qdrant embedded index, `bge-small-en-v1.5` | `data/vector_db/<cb>/` |
| 7 | Gemini instruction dataset | `data/processed/<cb>/instruction_dataset.jsonl` |
| 8 | OCR fallback (auto-detected, on by default) | `data/extracted/<cb>/ocr/` |

Phases are independently runnable and read prior artifacts from disk, so you can
re-run just phase 3 without repeating phase 2.

### Output contract

```jsonc
// clauses.jsonl
{"clause_id":"26.5.1.1","parent_id":"26.5.1","number":"26.5.1.1","depth":4,
 "title":"","text":"...","page":2,"confidence":"regex",
 "confidence_reason":"anchored regex; bold/size confirms heading","bbox":[...]}

// tables.jsonl
{"table_id":"T2","table_number":"2","clause_ref":"26.5.1.2","caption":"...",
 "markdown":"| a | b |\n| --- | --- |\n| 1 | 2 |","json_data":[["a","b"],["1","2"]],
 "page":3,"bbox":[...],"link_confidence":"high"}

// figures_metadata.jsonl
{"figure_id":"p2_f1","figure_number":"1","png_path":"figures/p2_f1.png",
 "caption":"Fig. 1 Typical Arrangement...","page":2,"bbox":[...],
 "related_clause":"26.5.1.1","is_inside_table":false,"confidence":"caption_anchored",
 "detector":"pycaption","status":"OK"}

// instruction_dataset.jsonl
{"instruction":"...","input":"","output":"...","source_chunk_id":"text:26.5.1.1",
 "source_type":"text","clause_id":"26.5.1.1","page":2,"grounding":"numbers_verified"}
```

**`confidence` values (Phase 2):** `regex` — clause number read off the page and
confirmed by bold/font size. `heuristic` — regex matched but the heading was not
typographically emphatic, or the parent was inferred. `ocr` — from Phase 8.
`preamble` — unnumbered front matter.

**`status` values (Phase 4):** `OK`, `SUSPECT` (no caption or implausible
geometry — verify before training on it), `FAILED` (no usable image).

---

## Querying the vector database

```bash
.venv/bin/python -m src.pipeline.query "what shear stress applies to M25" --top-k 5
.venv/bin/python -m src.pipeline.query "Cl. 26.5.1.1" --mode bm25-only
.venv/bin/python -m src.pipeline.query -i          # interactive
.venv/bin/python -m src.pipeline.query "shear" --codebook is456 --codebook eurocode
```

Retrieval is **hybrid**: dense vectors fused with BM25 via Reciprocal Rank Fusion.
This is not redundant. Clause numbers are arbitrary tokens with no semantic
neighbourhood, so dense retrieval is poor at exactly the queries a codebook gets.
Measured on the fixture, for `"Cl. 27.3.1.1"`:

| mode | top hit |
|---|---|
| vector only | `31.1 General` ✗ |
| BM25 only | `27.3.1.1` ✓ |
| hybrid (RRF) | `27.3.1.1` ✓ |

From Python:

```python
from pathlib import Path
from src.common.config import load_config
from src.embedding.qdrant_indexer import HybridRetriever

config = load_config()
cb = "is456_excerpt"
r = HybridRetriever(config, config.out_dirs(cb)["vector_db"], cb)
for hit in r.search("minimum water cement ratio", top_k=5):
    print(hit["source_type"], hit.get("clause_id"), hit["score"], hit["text"][:80])
r.close()
```

---

## Using the dataset for fine-tuning

`instruction_dataset.jsonl` is one JSON object per line, ready for any
Alpaca/ShareGPT-style loader.

**OpenAI-style (`instruction` / `input` / `output`):**

```python
import json
rows = [json.loads(l) for l in open("data/processed/is456_excerpt/instruction_dataset.jsonl")]
train = [{"instruction": r["instruction"], "input": r["input"], "output": r["output"]} for r in rows]
json.dump(train, open("train.json", "w"), indent=2)
```

**Verl / TRL:**

```python
dataset = load_dataset("json", data_files="instruction_dataset.jsonl", split="train")
```

### What is deliberately absent, and why

Phase 7 **refuses design-calculation questions**. An LLM asked to size a member
produces a fluent, confident, wrong number, and in a codebook-derived dataset that
is the most dangerous failure mode — it looks authoritative. Two filters run after
generation:

- `forbid_calculation_intent` drops any pair whose question or answer reads as a
  computation.
- A numeric grounding check drops any answer containing a number that does not
  appear in its source chunk.

Rejections are counted in `pipeline_report.json`
(`rejected_calculation`, `rejected_ungrounded`), so you can see how much was
discarded rather than trusting that nothing was.

Every record carries `source_chunk_id`, so any generated claim is traceable to the
exact clause, table or figure. Drop a record whose chunk has a low-confidence
parent if you need a stricter dataset:

```bash
# chunks whose clause is only heuristically linked
jq -c 'select(.confidence=="heuristic")' data/processed/<cb>/multimodal_chunks.jsonl
```

### Multimodal fine-tuning

Figure chunks carry `png_path`; the PNGs are in
`data/extracted/<cb>/figures/`. Join on `figure_id` to build a VLM dataset:

```bash
jq -c 'select(.type=="figure")' data/processed/<cb>/multimodal_chunks.jsonl
```

Figures have **no image embeddings** — they are embedded from `caption` +
`related_clause` text, and the image travels as payload. For VLM training the
image itself is used directly.

---

## Configuration

`configs/default.yaml` holds every threshold. Common overrides:

```yaml
# extra.yaml
text_extraction:
  running_head_min_page_fraction: 0.4   # looser header detection
chunking:
  max_words: 400                        # smaller chunks
embedding:
  model: BAAI/bge-base-en-v1.5
  expected_dim: 768                     # must match the model
dataset_generation:
  requests_per_minute: 4
page_rendering:
  enabled: true                         # also write page PNGs
```

> If you change the embedding model, change `expected_dim` too. The pipeline
> asserts the model's real dimension and refuses to index on a mismatch rather
> than writing vectors into a collection built at the old size.

---

## Known limits

These are reported at runtime as `degraded_reasons`, not hidden.

### Measured on IS 456:2000 (114 pages)

That PDF is the project's real-world test. It is **not born-digital**: its
producer is `itextpdf`, an OCR tool, and the text layer it produced has lost
word spacing on 2.1% of tokens and mangles digits elsewhere (`27.1l` for
`27.11`, `bUNII.d`, `01relll;on`). Phase 2 measures this and reports it rather
than passing a damaged file off as clean — `text_damage` in the report, with
examples.

| Phase | Result | Note |
|---|---|---|
| 2 | 537 clauses, 0 orphans, depths 1–4 | 108 `regex` / 428 `heuristic`; 33 duplicate numbers from OCR splitting them (`26.1` read as `26` + `1`) |
| 3 | 30 tables vs 31 `Table N` captions | see below |
| 4 | 7 figures, all `OK` | 13 caption lines, so some captions have no adjacent graphic |
| 5 | 571 chunks | 6 table numbers reused by footnotes/cross-references, flagged |
| 6 | 571 chunks indexed | hybrid retrieval working |

Specific limits, each measured rather than assumed:

- **Table detection needs the caption to locate borderless tables.** IS 456's
  tables carry no ruling at all — page 28 has two horizontal rules (header and
  footer) and zero vertical rules — so `find_tables(vertical_strategy="lines")`
  scores **0 candidates** and finds 4 tables against 31 captions (~13% recall).
  `is_real_table` cannot close this gap: a borderless table and two-column prose
  are both "rows with two filled columns". The pipeline therefore crops the page
  to the band each `Table N` caption opens and runs the text strategy only
  inside it, which lifts recall to 30/31. Run that pass globally instead and it
  invents page-sized pseudo-tables containing the running head — which is why it
  is caption-anchored.
- **Two-column pages need per-column re-extraction.** pdfplumber groups lines by
  vertical position across the full page width, so a two-column page's text has
  already fused by the time `extract_text_lines` returns it; sorting afterwards
  cannot separate it. Gutter detection uses 2pt bins with a *non-zero* coverage
  threshold — coarse bins miss a 10–18pt gutter, while exact interval merging
  fails the opposite way because any full-width caption bridges the gap — and
  then the page is filtered per column. Detection runs per page and is
  reconciled against a document-level consensus, so pages carrying a
  full-width figure inherit the layout the rest of the book agrees on.
- **Tables of contents parse as clauses** unless excluded. IS 456's contents
  pages look exactly like a clause list; a trailing page number is what
  distinguishes them. Left in, they add ~200 fake depth-1 clauses.
- **Clause numbers need a plausible range.** IS 456 has sections 0–23, so
  `max_number: 40` rejects `415`, `900` and `IS 2116` — standard references and
  margin values that otherwise dominate the tree as depth-1 clauses.
- **Figure captions must be on the same page as the graphic**, within
  `max_clause_gap_pt`. A caption in a running list of figures is not captured.
- **PDFFigures 2.0 is weak on codebooks.** Measured: 0% figure recall (0 of 5
  captioned figures) and 29% table recall at 50% precision. It runs as a
  supplement for *uncaptioned* figures only; caption-anchored detection is the
  primary path and its contributions carry lower confidence.
- **Clause detection is regex-dependent.** Body prose that opens with a clause
  number is treated as a heading and marked `heuristic`.
- **OCR needs marker-pdf**, which is heavy and separate:
  `pip install -r requirements-ocr.txt`. Without it, scanned pages are reported
  as having no text layer rather than silently producing empty clauses. Note
  that for IS 456 a fresh OCR would likely be *better* than the damaged text
  layer already present.

### Phase 7 LLM provider

Phase 7 supports two backends, selected with `dataset_generation.provider`:

```yaml
# Gemini (default)
dataset_generation:
  provider: gemini
  model: gemini-3.8-flash
```

```yaml
# OpenAI-compatible: DeepSeek, Groq, Together, vLLM, LM Studio, ...
dataset_generation:
  provider: openai_compatible
  base_url: https://api.deepseek.com/v1
  model: deepseek-chat          # DeepSeek exposes deepseek-flash / deepseek-v4-pro
  api_key_env: DEEPSEEK_API_KEY
```

```bash
export DEEPSEEK_API_KEY=...     # or GEMINI_API_KEY, LLM_API_KEY, OPENAI_API_KEY
python -m src.pipeline.run --pdf data/raw/<pdf>.pdf --phases 7
```

The key is read from the environment only, never written to disk. The
OpenAI-compatible backend uses `urllib` rather than the `openai` SDK, so adding a
provider needs no new dependency — just a `base_url`.

### Phase 7 needs working quota

Phase 7 reports `FAILED` and aborts fast when the key cannot generate:

| Symptom | Meaning |
|---|---|
| `quota: API quota exhausted (429 RESOURCE_EXHAUSTED)` | Gemini free-tier generation quota spent. `models.list()` still succeeds, so the key is valid. |
| `quota: payment required (402): account balance exhausted` | DeepSeek and similar: no credit on the account. |

Both are account limits, not code faults. Whatever was generated before the
abort is written and marked `DEGRADED`; nothing is silently padded. Fix billing
or quota, then re-run `--phases 7`.

---

## Tests

```bash
.venv/bin/python -m pytest tests/ -q     # 58 tests, no network, no API key
```

Assertions are against the synthetic fixture in `fixtures/`, whose correct answer
is known exactly: 33 clauses, 6 tables (plus 1 layout frame that must be
rejected), 4 figures, and 4 figure→clause links. Regenerate with
`python fixtures/make_fixture.py`.

---

## Layout

```
configs/default.yaml              thresholds
configs/standards/*.yaml          per-standard clause + caption grammar
src/common/                       status, config, jsonl
src/extraction/                   text, table, figure, page, ocr extractors
src/processing/                   chunker, linker
src/embedding/qdrant_indexer.py   index + hybrid retriever
src/dataset_generation/           Gemini instruction generator
src/pipeline/run.py               single entry point
src/pipeline/query.py             vector db CLI
scripts/setup_env.sh              sudo-free venv + pip
scripts/setup_deps.sh             Java 11 + sbt + pdffigures2
```

### Installation notes

Both setup scripts avoid `sudo` entirely, because this class of machine usually
requires a password for it:

- **pip** — the system Python has neither `pip` nor `ensurepip`, so `get-pip.py`
  bootstraps it inside `.venv`.
- **Java** — many systems ship a **JRE** only (no `javac`), which cannot compile
  Scala. `setup_deps.sh` detects that and downloads a Temurin JDK 11 into the
  tools directory.
- **sbt** — unpacked from the official tarball. Pinned to 1.x: sbt 2.x removed
  `bintrayPackage`, which pdffigures2's `build.sbt` still uses.
- **Build tooling location** — if the repo path contains whitespace, sbt word-splits
  its options and the build fails with a corrupted class name. `setup_deps.sh`
  detects this and relocates the tools to `$XDG_CACHE_HOME/codebook-pipeline/tools`,
  recording the location in `.tools_locator.json`.
- **Qdrant** runs embedded (`QdrantClient(path=...)`); no Docker required.

### Security

`GEMINI_API_KEY` is read from the environment only and is never written to disk.
`.env` is gitignored; `.env.example` is committed. Treat any key pasted into a
chat or a log as compromised and rotate it.