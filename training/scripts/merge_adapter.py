"""Merge a trained adapter into its local base checkpoint for later conversion."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "training/artifacts/merged")
    args = parser.parse_args()
    if not (args.base_model / "config.json").is_file() or not (args.adapter / "adapter_config.json").is_file():
        parser.error("Provide a local base checkpoint and a trained PEFT adapter.")
    try:
        import torch
        import transformers
        from peft import PeftModel
    except ImportError:
        parser.exit(1, "Install training/requirements.txt in the separate training environment.\n")
    config = transformers.AutoConfig.from_pretrained(str(args.base_model), local_files_only=True, trust_remote_code=False)
    architectures = config.architectures or []
    model_class = getattr(transformers, architectures[0], None) if architectures else None
    if model_class is None:
        parser.error("The installed Transformers does not support this checkpoint architecture.")
    base = model_class.from_pretrained(str(args.base_model), dtype=torch.float16,
                                       device_map={"": "cpu"}, low_cpu_mem_usage=True,
                                       local_files_only=True, trust_remote_code=False)
    adapted = PeftModel.from_pretrained(base, str(args.adapter), local_files_only=True)
    merged = adapted.merge_and_unload()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.output_dir), safe_serialization=True, max_shard_size="2GB")
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(args.base_model), local_files_only=True, trust_remote_code=False)
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Merged checkpoint saved to {args.output_dir}; GGUF conversion is a separate step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
