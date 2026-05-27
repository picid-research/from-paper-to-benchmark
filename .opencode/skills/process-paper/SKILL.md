---
name: process-paper
description: Convert a paper PDF to structured markdown using marker and build a deterministic section index. Run this at the start of the paper-validation pipeline whenever a PDF is present. Produces the markdown and document_index.json that downstream agents (chunk-indexer, conceptual-analysis, algorithmic-spec) read directly.
---

# /process-paper

Preprocess a research paper so the rest of the validate-paper pipeline can read it.

This skill is a thin wrapper around two deterministic steps:

1. Convert the PDF to markdown + metadata with `marker_single`.
2. Build a section index (`document_index.json`) from the marker output using the committed script `.opencode/tools/index_paper.py`.

No MCP server, no semantic classification heuristics, no document-type detection. Marker gives clean markdown with proper ATX headers and a `table_of_contents` in its `_meta.json` — the indexer script just walks those.

## Input contract

The orchestrator must provide:

- **paper_dir**: directory containing the paper (e.g. `vault/paper`). Expected to contain exactly one `.pdf` file, OR already-processed marker output under `{paper_dir}/processed_paper/{pdf_stem}/`.

## Procedure

### Step 1 — Locate the PDF (or confirm existing marker output)

```bash
ls {paper_dir}/*.pdf 2>/dev/null
ls {paper_dir}/processed_paper/ 2>/dev/null
```

Determine `pdf_stem` (PDF filename without extension). If `{paper_dir}/processed_paper/{pdf_stem}/{pdf_stem}.md` already exists, skip step 2.

### Step 2 — Run marker

```bash
uv run marker_single {paper_dir}/{pdf_stem}.pdf --output_dir {paper_dir}/processed_paper
```

Marker writes:

```
{paper_dir}/processed_paper/{pdf_stem}/
    {pdf_stem}.md
    {pdf_stem}_meta.json
    _page_N_<Figure|Picture>_N.jpeg ...
```

If the command fails, stop and report the error — do not try to work around it. Common causes: marker not installed, PDF corrupt, output dir not writable.

### Step 3 — Build the section index

```bash
uv run python .opencode/tools/index_paper.py {paper_dir}/processed_paper/{pdf_stem}/
```

This writes `{paper_dir}/processed_paper/{pdf_stem}/document_index.json` with schema:

```json
{
  "title": "<paper title>",
  "markdown_path": "<abs path to {pdf_stem}.md>",
  "meta_path": "<abs path to {pdf_stem}_meta.json>",
  "sections": [
    {
      "id": "s0",
      "title": "I. INTRODUCTION",
      "content_type": "introduction",
      "page_id": 0,
      "char_start": 1234,
      "char_end": 5678,
      "content": "<full section markdown>"
    }
  ]
}
```

`content_type` is one of: `abstract | introduction | related_work | methodology | experiments | conclusion | references | general`.

### Step 4 — Verify and report

Confirm the index is well-formed:

```bash
test -f {paper_dir}/processed_paper/{pdf_stem}/document_index.json && \
  uv run python -c "import json; d=json.load(open('{paper_dir}/processed_paper/{pdf_stem}/document_index.json')); print(f'OK: {len(d[\"sections\"])} sections, title={d[\"title\"][:80]}')"
```

Report to the caller:
- `pdf_stem`
- `markdown_path`: `{paper_dir}/processed_paper/{pdf_stem}/{pdf_stem}.md`
- `document_index_path`: `{paper_dir}/processed_paper/{pdf_stem}/document_index.json`
- section count and top-level content types found

## Idempotency

All steps are idempotent. If marker output already exists, step 2 is skipped. Running the indexer always overwrites `document_index.json` with fresh output — this is intentional so upstream edits to the markdown propagate.

## What this skill does NOT do

- Does not segment semantically beyond the markdown's own ATX headers.
- Does not filter, rank, or retrieve content by query. Downstream agents read the index and markdown directly.
- Does not run any Python daemon or MCP server.
