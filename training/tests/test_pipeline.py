"""Dataset integrity tests which do not load model weights or training libraries."""
import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from training.pipeline import DatasetError, load_splits, normalize_record, read_records, split_records, write_splits


def sample(index, source=None):
    return {"id": f"q{index}", "source_id": source or f"source-{index}",
            "question": f"What fact is recorded in source {index}?", "context": f"Source {index} records a fictional fact.",
            "answer": f"The recorded fictional fact is {index}."}


class DatasetPipelineTests(unittest.TestCase):
    def test_raw_pair_becomes_a_supervised_conversation(self):
        row = normalize_record(sample(1))
        self.assertEqual([m["role"] for m in row["messages"]], ["system", "user", "assistant"])
        self.assertIn("Source 1", row["messages"][1]["content"])
        self.assertEqual(row["messages"][-1]["content"], sample(1)["answer"])

    def test_unlabeled_predicate_data_cannot_be_used_as_sft(self):
        with self.assertRaises(DatasetError):
            normalize_record({"id": "golden-1", "source_id": "catalogue", "question": "Where?", "must_satisfy": []})

    def test_multimodal_inputs_are_not_silently_dropped(self):
        with self.assertRaises(DatasetError):
            normalize_record({**sample(1), "images": ["photo.png"]})

    def test_synthetic_data_is_explicitly_gated(self):
        row = {**sample(1), "synthetic": True}
        with self.assertRaises(DatasetError):
            normalize_record(row)
        self.assertTrue(normalize_record(row, allow_synthetic=True)["synthetic"])

    def test_rejects_missing_answers_and_private_reasoning_labels(self):
        for answer in ("", "<think>private reasoning</think>answer"):
            with self.subTest(answer=answer), self.assertRaises(DatasetError):
                normalize_record({**sample(1), "answer": answer})

    def test_source_group_never_crosses_partitions(self):
        rows = [normalize_record(sample(i, f"group-{i // 2}")) for i in range(12)]
        splits = split_records(rows, seed=7)
        groups = [{row["source_id"] for row in part} for part in splits.values()]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertEqual(splits, split_records(rows, seed=7))

    def test_write_and_load_detects_leakage_in_edited_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [normalize_record(sample(i)) for i in range(6)]
            manifest = write_splits(rows, root)
            for name, details in manifest["files"].items():
                self.assertEqual(details["sha256"], hashlib.sha256((root / f"{name}.jsonl").read_bytes()).hexdigest())
            self.assertEqual(sum(len(rows) for rows in load_splits(root).values()), 6)
            train = (root / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
            (root / "test.jsonl").write_text(train + "\n", encoding="utf-8")
            with self.assertRaises(DatasetError):
                load_splits(root)

    def test_duplicate_input_is_rejected_even_with_different_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.jsonl"
            first = sample(1)
            second = {**first, "id": "another-id", "source_id": "another-source"}
            path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8")
            with self.assertRaises(DatasetError):
                read_records(path)


if __name__ == "__main__":
    unittest.main()
