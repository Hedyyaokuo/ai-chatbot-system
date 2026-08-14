from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from pypdf import PdfReader


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, size: int = 1200, overlap: int = 180) -> list[str]:
    text = clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def report_records(path: Path) -> list[dict]:
    records = []
    reader = PdfReader(path)
    for page_number, page in enumerate(reader.pages, start=1):
        for chunk_number, content in enumerate(chunk_text(page.extract_text() or ""), start=1):
            records.append({
                "title": f"AI 聊天系统报告，第 {page_number} 页",
                "source": "S5004312_YixinZhang.pdf",
                "family": "architecture",
                "section": f"report-page-{page_number}",
                "modality": "report_text",
                "content": content,
                "chunk": chunk_number,
            })
    return records


def markdown_records(path: Path, title: str, family: str) -> list[dict]:
    return [
        {
            "title": title,
            "source": path.name,
            "family": family,
            "section": "project-documentation",
            "modality": "text",
            "content": content,
            "chunk": index,
        }
        for index, content in enumerate(chunk_text(path.read_text(encoding="utf-8")), start=1)
    ]


def csv_records(path: Path, title: str) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [{
        "title": title,
        "source": path.name,
        "family": "evaluation",
        "section": "evaluation-results",
        "modality": "table",
        "content": json.dumps(row, ensure_ascii=False),
        "chunk": index,
    } for index, row in enumerate(rows, start=1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--project-readme", type=Path, required=True)
    parser.add_argument("--site-readme", type=Path, required=True)
    parser.add_argument("--retrieval-csv", type=Path, required=True)
    parser.add_argument("--judge-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = []
    records.extend(report_records(args.report))
    records.extend(markdown_records(args.project_readme, "项目 README", "architecture"))
    records.extend(markdown_records(args.site_readme, "展示网站 README", "architecture"))
    records.extend(csv_records(args.retrieval_csv, "多模态检索评估结果"))
    records.extend(csv_records(args.judge_csv, "LLM-as-Judge 评估结果"))
    args.output.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(records)} knowledge records to {args.output}")


if __name__ == "__main__":
    main()

