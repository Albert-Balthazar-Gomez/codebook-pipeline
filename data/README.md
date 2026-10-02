# data/

Everything here is generated. Nothing in this directory is committed except the
placeholder files that preserve the structure.

| Path | Written by | Contents |
|---|---|---|
| `raw/` | you | Source PDFs. **Not committed** -- see below. |
| `extracted/<codebook>/` | phases 2-4 | `text/clauses.jsonl`, `tables/tables.jsonl`, `figures/*.png`, `figures/figures_metadata.jsonl` |
| `processed/<codebook>/` | phases 5-7 | `multimodal_chunks.jsonl`, `instruction_dataset.jsonl`, `pipeline_report.json`, `figure_link_samples.json` |
| `vector_db/<codebook>/` | phase 6 | Embedded Qdrant storage |

## Why the PDFs and outputs are not committed

The extracted JSONL reproduces the text of the source codebook verbatim. IS 456
is a Bureau of Indian Standards publication under copyright, and the extracted
clause text is a derivative copy of it. Committing either the PDF or its
extraction would redistribute that material, so both are excluded.

Drop your PDF into `raw/` and run the pipeline; everything regenerates:

```bash
python -m src.pipeline.run --pdf data/raw/<file>.pdf --standard is456
```
