"""Build a deterministic document_index.json from marker_single output.

Usage:
    uv run python .opencode/tools/index_paper.py <processed_paper_subdir>

The subdir is expected to be the marker output directory for a single paper,
e.g. ``vault/paper/processed_paper/MyPaper/``, containing:
    - ``<stem>.md``
    - ``<stem>_meta.json``
    - image assets (ignored here)

The script parses the markdown's ATX headers, pairs each header with its
entry in the meta.json ``table_of_contents`` when possible, infers a
``content_type`` from the title, and writes ``document_index.json`` to the
same directory.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

HEADER_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

CONTENT_TYPE_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("abstract", ("abstract",)),
    ("introduction", ("introduction",)),
    ("related_work", ("related work", "background", "literature", "prior work")),
    ("methodology", ("method", "approach", "proposed", "framework", "architecture", "model")),
    ("experiments", ("experiment", "evaluation", "result", "performance", "ablation")),
    ("conclusion", ("conclusion", "future work", "summary", "discussion")),
    ("references", ("reference", "bibliography", "acknowledgment", "acknowledgement")),
]


@dataclass
class Section:
    id: str
    title: str
    content_type: str
    page_id: int | None
    char_start: int
    char_end: int
    content: str


def infer_content_type(title: str) -> str:
    t = title.lower()
    for label, keywords in CONTENT_TYPE_KEYWORDS:
        if any(k in t for k in keywords):
            return label
    return "general"


def normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip().lower()


def find_single_markdown(paper_dir: Path) -> Path:
    mds = [p for p in paper_dir.glob("*.md") if not p.name.startswith("document_index")]
    if len(mds) != 1:
        raise SystemExit(
            f"Expected exactly one markdown file in {paper_dir}, found {len(mds)}: {mds}"
        )
    return mds[0]


def load_toc(meta_path: Path) -> list[dict]:
    if not meta_path.is_file():
        return []
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    toc = data.get("table_of_contents") or []
    return toc if isinstance(toc, list) else []


def match_toc_page(title: str, toc: list[dict], used: set[int]) -> int | None:
    target = normalize_title(title)
    for i, entry in enumerate(toc):
        if i in used:
            continue
        toc_title = normalize_title(str(entry.get("title", "")))
        if toc_title == target or target in toc_title or toc_title in target:
            used.add(i)
            page_id = entry.get("page_id")
            return int(page_id) if isinstance(page_id, int) else None
    return None


def extract_sections(md_text: str, toc: list[dict]) -> list[Section]:
    lines = md_text.splitlines(keepends=True)
    boundaries: list[tuple[int, str]] = []  # (char_start, title)
    offset = 0
    for line in lines:
        m = HEADER_RE.match(line)
        if m:
            boundaries.append((offset, m.group(2).strip()))
        offset += len(line)

    if not boundaries:
        return [
            Section(
                id="s0",
                title="(document)",
                content_type="general",
                page_id=None,
                char_start=0,
                char_end=len(md_text),
                content=md_text,
            )
        ]

    used_toc: set[int] = set()
    sections: list[Section] = []
    for i, (start, title) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(md_text)
        sections.append(
            Section(
                id=f"s{i}",
                title=title,
                content_type=infer_content_type(title),
                page_id=match_toc_page(title, toc, used_toc),
                char_start=start,
                char_end=end,
                content=md_text[start:end],
            )
        )
    return sections


def build_index(paper_dir: Path) -> dict:
    md_path = find_single_markdown(paper_dir)
    stem = md_path.stem
    meta_path = paper_dir / f"{stem}_meta.json"

    md_text = md_path.read_text(encoding="utf-8")
    toc = load_toc(meta_path)

    title = str(toc[0]["title"]).strip() if toc else stem
    # Normalize multi-line TOC titles (marker sometimes has embedded \n)
    title = re.sub(r"\s+", " ", title)

    sections = extract_sections(md_text, toc)

    return {
        "title": title,
        "markdown_path": str(md_path.resolve()),
        "meta_path": str(meta_path.resolve()) if meta_path.is_file() else None,
        "sections": [asdict(s) for s in sections],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paper_dir", type=Path, help="marker output dir for a single paper")
    args = parser.parse_args(argv)

    paper_dir: Path = args.paper_dir
    if not paper_dir.is_dir():
        print(f"Not a directory: {paper_dir}", file=sys.stderr)
        return 1

    index = build_index(paper_dir)
    out_path = paper_dir / "document_index.json"
    out_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {out_path} ({len(index['sections'])} sections)")
    for s in index["sections"]:
        print(f"  {s['id']:>4}  p{s['page_id']}  {s['content_type']:<14}  {s['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
