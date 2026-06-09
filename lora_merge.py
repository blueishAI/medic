import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MedicConfig


def main() -> None:
    cfg = MedicConfig()
    if not os.path.isdir(cfg.adapter_dir):
        raise RuntimeError(f"Missing LoRA adapter directory: {cfg.adapter_dir}")

    base = AutoModelForCausalLM.from_pretrained(
        cfg.prepared_base_dir,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, cfg.adapter_dir)
    model = model.merge_and_unload()

    tok = AutoTokenizer.from_pretrained(cfg.adapter_dir, use_fast=True)
    os.makedirs(cfg.merged_dir, exist_ok=True)
    model.save_pretrained(cfg.merged_dir, safe_serialization=False)
    tok.save_pretrained(cfg.merged_dir)
    print(f"[merge] merged model saved -> {cfg.merged_dir}")


if __name__ == "__main__":
    main()
