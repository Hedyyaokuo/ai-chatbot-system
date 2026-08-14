from __future__ import annotations

import argparse
import gzip
import json
import sqlite3
from pathlib import Path


TEXT_COLLECTION = "personalised_multimodal_knowledge_base"
IMAGE_COLLECTION = "eventnow_true_image_index"


def _metadata_value(row: sqlite3.Row):
    for key in ("string_value", "int_value", "float_value", "bool_value"):
        if row[key] is not None:
            return row[key]
    return None


def _read_collection(database_path: Path, collection_name: str) -> list[dict]:
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT e.embedding_id, m.*
        FROM embeddings e
        JOIN segments s ON s.id = e.segment_id
        JOIN collections c ON c.id = s.collection
        JOIN embedding_metadata m ON m.id = e.id
        WHERE c.name = ?
        ORDER BY e.id, m.key
        """,
        (collection_name,),
    ).fetchall()
    connection.close()

    records: dict[str, dict] = {}
    for row in rows:
        record = records.setdefault(
            row["embedding_id"], {"original_embedding_id": row["embedding_id"]}
        )
        record[row["key"]] = _metadata_value(row)
    return list(records.values())


def export_knowledge(source_root: Path, output_path: Path) -> dict:
    text_records = _read_collection(
        source_root / "vector_db" / "chroma.sqlite3", TEXT_COLLECTION
    )
    true_image_records = _read_collection(
        source_root / "image_vector_db" / "chroma.sqlite3", IMAGE_COLLECTION
    )
    true_image_files = {
        str(record.get("source_file", "")).lower() for record in true_image_records
    }

    exported = []
    for record in text_records:
        source_file = str(record.get("source_file", "unknown"))
        exported.append({
            "id": record["original_embedding_id"],
            "content": str(record.get("chroma:document", "")),
            "source_file": source_file,
            "source_path": str(record.get("source_path", "")),
            "modality": str(record.get("modality", "text")),
            "section": str(record.get("section", "general")),
            "document_family": str(record.get("document_family", "general")),
            "knowledge_type": str(record.get("knowledge_type", "unknown")),
            "visual_type": str(record.get("visual_type", "")),
            "tags": str(record.get("tags", "")),
            "page_label": str(record.get("page_label", "")),
            "chunk_id": record.get("chunk_id"),
            "has_true_image_embedding": source_file.lower() in true_image_files,
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(exported, handle, ensure_ascii=False, separators=(",", ":"))

    return {
        "records": len(exported),
        "text_chunks": sum(item["modality"] == "text" for item in exported),
        "image_captions": sum(
            item["modality"] == "image_caption" for item in exported
        ),
        "true_image_links": sum(
            item["has_true_image_embedding"] for item in exported
        ),
        "source_files": len({item["source_file"] for item in exported}),
        "chunk_size": 650,
        "chunk_overlap": 100,
        "source_collection": TEXT_COLLECTION,
        "true_image_collection": IMAGE_COLLECTION,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从原始 Chroma 数据库导出云端可读取的知识块。"
    )
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path)
    arguments = parser.parse_args()

    manifest = export_knowledge(arguments.source_root.resolve(), arguments.output.resolve())
    if arguments.manifest:
        arguments.manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
