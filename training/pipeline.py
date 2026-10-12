"""Validate supervised conversations and split source groups without leakage."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


class DatasetError(ValueError):
    pass


def fingerprint(value) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_record(record: dict, *, allow_synthetic: bool = False) -> dict:
    from config.prompts import CHAT_SYSTEM_PROMPT, RAG_USER_TEMPLATE

    if not isinstance(record, dict):
        raise DatasetError("Each JSONL row must be an object.")
    if any(key in record for key in ("image", "images", "video", "videos")):
        raise DatasetError("This pipeline only accepts text, not multimodal training inputs.")
    for key in ("id", "source_id"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise DatasetError(f"A nonempty {key} is required.")
    synthetic = record.get("synthetic", False)
    if not isinstance(synthetic, bool):
        raise DatasetError("synthetic must be a boolean.")
    if synthetic and not allow_synthetic:
        raise DatasetError("Synthetic examples require --allow-synthetic; they are not a real corpus.")
    if "messages" in record:
        messages = record["messages"]
    else:
        question, answer = record.get("question"), record.get("answer")
        if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
            raise DatasetError("Provide messages or a nonempty question/answer pair.")
        context = record.get("context", "")
        if not isinstance(context, str):
            raise DatasetError("context must be a string.")
        content = RAG_USER_TEMPLATE.format(context=context, query=question) if context.strip() else question
        messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT},
                    {"role": "user", "content": content}, {"role": "assistant", "content": answer}]
    if not isinstance(messages, list) or len(messages) < 2:
        raise DatasetError("A conversation needs at least a user turn and an assistant answer.")
    cleaned = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant"}:
            raise DatasetError("Only system/user/assistant text messages are supported.")
        if set(message) != {"role", "content"} or not isinstance(message["content"], str) or not message["content"].strip():
            raise DatasetError("Every message needs only role and nonblank text content.")
        if message["role"] == "system" and index != 0:
            raise DatasetError("The system message may only appear first.")
        if message["role"] == "assistant" and any(tag in message["content"].lower() for tag in ("<think>", "</think>")):
            raise DatasetError("Assistant labels must contain the final answer, not private reasoning tags.")
        cleaned.append(dict(message))
    if cleaned[0]["role"] != "system":
        cleaned.insert(0, {"role": "system", "content": CHAT_SYSTEM_PROMPT})
    for index, message in enumerate(cleaned[1:]):
        if message["role"] != ("user" if index % 2 == 0 else "assistant"):
            raise DatasetError("Conversation roles must alternate user and assistant.")
    if cleaned[-1]["role"] != "assistant":
        raise DatasetError("The last turn must be the supervised assistant answer.")
    return {"id": record["id"].strip(), "source_id": record["source_id"].strip(), "synthetic": synthetic,
            "messages": cleaned}


def read_records(path: Path, *, allow_synthetic: bool = False) -> list[dict]:
    records, ids, inputs = [], set(), {}
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = normalize_record(json.loads(line), allow_synthetic=allow_synthetic)
            except (json.JSONDecodeError, DatasetError) as exc:
                raise DatasetError(f"{path.name}, line {line_number}: {exc}") from exc
            if row["id"] in ids:
                raise DatasetError(f"Duplicate sample id at line {line_number}.")
            key = fingerprint(row["messages"][:-1])
            if key in inputs:
                raise DatasetError(f"Duplicate or conflicting supervised input at line {line_number}.")
            ids.add(row["id"])
            inputs[key] = row["messages"][-1]["content"]
            records.append(row)
    if not records:
        raise DatasetError("The dataset is empty.")
    return records


def split_records(records: list[dict], *, validation_ratio: float = 0.1,
                  test_ratio: float = 0.1, seed: int = 42) -> dict[str, list[dict]]:
    if not 0 < validation_ratio < 1 or not 0 < test_ratio < 1 or validation_ratio + test_ratio >= 1:
        raise DatasetError("Validation/test ratios must leave a positive training portion.")
    groups = sorted({row["source_id"] for row in records})
    if len(groups) < 3:
        raise DatasetError("At least three independent source groups are needed for train/validation/test.")
    random.Random(seed).shuffle(groups)
    n_val, n_test = max(1, round(len(groups) * validation_ratio)), max(1, round(len(groups) * test_ratio))
    if n_val + n_test >= len(groups):
        raise DatasetError("The requested split leaves no training source group.")
    val, test = set(groups[:n_val]), set(groups[n_val:n_val + n_test])
    result = {"train": [], "validation": [], "test": []}
    for row in records:
        name = "validation" if row["source_id"] in val else "test" if row["source_id"] in test else "train"
        result[name].append(row)
    return result


def write_splits(records: list[dict], output_dir: Path, **kwargs) -> dict:
    partitions = split_records(records, **kwargs)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"format": "supervised-conversations-v1", "split_by": "source_id",
                "seed": kwargs.get("seed", 42), "synthetic": any(row["synthetic"] for row in records),
                "files": {}}
    for name, rows in partitions.items():
        path = output_dir / f"{name}.jsonl"
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        path.write_bytes(payload.encode("utf-8"))
        manifest["files"][name] = {"rows": len(rows), "source_groups": len({r["source_id"] for r in rows}),
                                    "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_splits(directory: Path, *, allow_synthetic: bool = False) -> dict[str, list[dict]]:
    result = {name: read_records(directory / f"{name}.jsonl", allow_synthetic=allow_synthetic)
              for name in ("train", "validation", "test")}
    seen_sources, seen_inputs, seen_ids = set(), set(), set()
    for name, rows in result.items():
        sources = {row["source_id"] for row in rows}
        if seen_sources & sources:
            raise DatasetError(f"Source leakage into {name} split.")
        inputs = {fingerprint(row["messages"][:-1]) for row in rows}
        ids = {row["id"] for row in rows}
        if seen_inputs & inputs or seen_ids & ids:
            raise DatasetError(f"Duplicate supervised examples across the {name} split.")
        seen_sources |= sources
        seen_inputs |= inputs
        seen_ids |= ids
    return result
