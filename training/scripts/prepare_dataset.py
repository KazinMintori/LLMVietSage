"""Validate JSONL and create source-separated supervised splits."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.pipeline import DatasetError, read_records, write_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "training/data/prepared")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()
    try:
        records = read_records(args.input, allow_synthetic=args.allow_synthetic)
        manifest = write_splits(records, args.output_dir, validation_ratio=args.validation_ratio,
                                test_ratio=args.test_ratio, seed=args.seed)
    except (DatasetError, OSError) as exc:
        parser.exit(1, str(exc) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
