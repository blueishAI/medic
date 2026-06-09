import os

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MedicConfig


def main() -> None:
    cfg = MedicConfig()
    os.makedirs(cfg.prepared_base_dir, exist_ok=True)

    marker = os.path.join(cfg.prepared_base_dir, "config.json")
    if os.path.isfile(marker):
        print(f"[prepare] using existing Anant base -> {cfg.prepared_base_dir}")
        return

    print(f"[prepare] base={cfg.base_model_id}")
    print(f"[prepare] anant={cfg.anant_repo_id}/{cfg.anant_adapter_subdir}")

    repo_dir = snapshot_download(
        cfg.anant_repo_id,
        allow_patterns=[f"{cfg.anant_adapter_subdir}/*"],
    )
    adapter_dir = os.path.join(repo_dir, cfg.anant_adapter_subdir)

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.merge_and_unload()

    model.save_pretrained(cfg.prepared_base_dir, safe_serialization=False)
    tokenizer.save_pretrained(cfg.prepared_base_dir)
    print(f"[prepare] saved Anant-merged base -> {cfg.prepared_base_dir}")


if __name__ == "__main__":
    main()
