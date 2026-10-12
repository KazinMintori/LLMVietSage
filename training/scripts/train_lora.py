"""Offline supervised LoRA/QLoRA training from a local Hugging Face checkpoint."""
import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training.pipeline import DatasetError, load_splits, fingerprint


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def run(config: dict, *, dry_run: bool = False):
    partitions = load_splits(resolve_path(config["dataset_dir"]), allow_synthetic=config.get("allow_synthetic", False))
    for key in ("max_length", "batch_size", "gradient_accumulation_steps", "lora_r", "lora_alpha"):
        if isinstance(config.get(key), bool) or not isinstance(config.get(key), int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if config["learning_rate"] <= 0 or config["num_train_epochs"] <= 0:
        raise ValueError("Learning rate and epochs must be positive.")
    if not 0 <= config["lora_dropout"] < 1:
        raise ValueError("LoRA dropout must be in [0, 1).")
    if not isinstance(config["target_modules"], list) or not config["target_modules"]:
        raise ValueError("Select LoRA target modules for the chosen model architecture.")
    plan = {"mode": "dry_run" if dry_run else "training", "rows": {k: len(v) for k, v in partitions.items()},
            "base_model_configured": bool(config.get("base_model_dir")), "output_dir": str(resolve_path(config["output_dir"]))}
    if dry_run:
        return plan
    if not config.get("base_model_dir"):
        raise ValueError("Set base_model_dir to a local Hugging Face checkpoint; an Ollama model tag is not a checkpoint path.")
    base = resolve_path(config["base_model_dir"])
    if not (base / "config.json").is_file():
        raise ValueError("The local base model needs config.json and pretrained model/tokenizer files.")
    try:
        import torch
        from datasets import Dataset
        import transformers
        from transformers import AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("Install the separate training/requirements.txt in a training environment first.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("This low-memory QLoRA profile requires a CUDA GPU.")
    tokenizer = AutoTokenizer.from_pretrained(str(base), local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError("The selected model needs a supported conversation chat template.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    for name, rows in partitions.items():
        for row in rows:
            tokens = tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False)
            if len(tokens) > config["max_length"]:
                raise DatasetError(f"A {name} sample exceeds max_length; trim its context explicitly to avoid losing the supervised answer.")
    datasets = {name: Dataset.from_list([{"prompt": row["messages"][:-1], "completion": [row["messages"][-1]]} for row in rows])
                for name, rows in partitions.items()}
    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype) if config["load_in_4bit"] else None
    model_config = transformers.AutoConfig.from_pretrained(str(base), local_files_only=True, trust_remote_code=False)
    architectures = model_config.architectures or []
    if getattr(model_config, "vision_config", None) is not None:
        raise ValueError("This starter trainer supports text-only checkpoints. A vision-language checkpoint needs its model-specific processor/collator before training.")
    model_class = getattr(transformers, architectures[0], None) if architectures else None
    if model_class is None:
        raise ValueError("The installed Transformers does not support this checkpoint architecture.")
    model = model_class.from_pretrained(str(base), local_files_only=True, trust_remote_code=False,
                                        dtype=dtype, device_map={"": 0}, quantization_config=quantization)
    model.config.use_cache = False
    if config["load_in_4bit"]:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config["gradient_checkpointing"])
    model = get_peft_model(model, LoraConfig(r=config["lora_r"], lora_alpha=config["lora_alpha"],
                                            lora_dropout=config["lora_dropout"], target_modules=config["target_modules"],
                                            bias="none", task_type="CAUSAL_LM"))
    output = resolve_path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    kwargs = dict(output_dir=str(output), max_length=config["max_length"],
                  per_device_train_batch_size=config["batch_size"], per_device_eval_batch_size=1,
                  gradient_accumulation_steps=config["gradient_accumulation_steps"],
                  learning_rate=config["learning_rate"], num_train_epochs=config["num_train_epochs"],
                  gradient_checkpointing=config["gradient_checkpointing"],
                  bf16=use_bf16, fp16=not use_bf16, packing=False,
                  completion_only_loss=True, eval_strategy="epoch", save_strategy="epoch",
                  save_total_limit=2, report_to="none", push_to_hub=False, seed=config["seed"])
    args = SFTConfig(**kwargs)
    trainer = SFTTrainer(model=model, args=args, processing_class=tokenizer,
                         train_dataset=datasets["train"], eval_dataset=datasets["validation"])
    trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    metrics = trainer.evaluate(datasets["test"], metric_key_prefix="held_out_test")
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "trl", "peft", "datasets", "bitsandbytes")}
    metadata = {**plan, "metrics": metrics, "config": config, "versions": versions,
                "dataset_hashes": {name: fingerprint(rows) for name, rows in partitions.items()},
                "gpu": torch.cuda.get_device_name(0)}
    (output / "training_run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "training/configs/qlora.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        result = run(json.loads(args.config.read_text(encoding="utf-8")), dry_run=args.dry_run)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, str(exc) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
