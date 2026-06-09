import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MedicConfig
from lora_train import SYSTEM_PROMPT


PROMPTS = [
    "Someone cut their hand while cooking and it is bleeding. What should I do first?",
    "A person may be choking and cannot speak. Give first aid steps.",
    "What should I do for a small burn from hot water?",
    "A child drank household cleaner. What should I do?",
    "Someone has chest pain and shortness of breath. What now?",
]


def _looks_broken(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 20:
        return True
    if not re.search(r"[A-Za-z0-9]", stripped):
        return True
    if "call" not in stripped.lower() and "emergency" not in stripped.lower() and "seek" not in stripped.lower():
        return True
    return False


def main() -> None:
    cfg = MedicConfig()
    if not os.path.isdir(cfg.merged_dir):
        raise RuntimeError(f"Missing merged model directory: {cfg.merged_dir}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.merged_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.merged_dir, torch_dtype=torch.float32, attn_implementation="sdpa")
    model.eval()

    failures = []
    os.makedirs(os.path.dirname(cfg.validation_log_path), exist_ok=True)
    with open(cfg.validation_log_path, "w", encoding="utf-8") as log:
        for prompt in PROMPTS:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=96,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(output_ids[0, input_ids.shape[1] :], skip_special_tokens=True)
            log.write(f"PROMPT: {prompt}\nRESPONSE: {response.strip()}\n\n")
            print(f"[validate] prompt: {prompt}")
            print(response.strip())
            print()
            if _looks_broken(response):
                failures.append(prompt)

    if failures:
        raise RuntimeError(f"Validation weak for {len(failures)} prompt(s). See {cfg.validation_log_path}")
    print(f"[validate] completed -> {cfg.validation_log_path}")


if __name__ == "__main__":
    main()
