import gc
import ast
import os
from pathlib import Path
from itertools import islice
from typing import Dict, List

import bitsandbytes as bnb
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import MedicConfig


SYSTEM_PROMPT = (
    "You are Medic, a tiny first-aid assistant. Give fast, practical, safe first-aid steps. "
    "Tell the user when to call emergency services. Do not diagnose. Do not replace a clinician."
)


def _cuda_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 and torch.cuda.is_bf16_supported() else torch.float16


def _memory_snapshot() -> str:
    if not torch.cuda.is_available():
        return "cuda=unavailable"
    parts = []
    for idx in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(idx)
        parts.append(f"cuda:{idx} free={free / 1024**3:.1f}GiB total={total / 1024**3:.1f}GiB")
    return "; ".join(parts)


def _qlora_device_map(local_rank: int):
    requested = os.getenv("MEDIC_DEVICE_MAP", "single").strip().lower()
    if requested == "auto":
        return "auto"
    return {"": local_rank}


def _qlora_max_memory():
    if os.getenv("MEDIC_DEVICE_MAP", "single").strip().lower() != "auto" or not torch.cuda.is_available():
        return None
    gpu_limit = os.getenv("MEDIC_MAX_MEMORY_GPU", "13GiB")
    cpu_limit = os.getenv("MEDIC_MAX_MEMORY_CPU", "24GiB")
    memory = {idx: gpu_limit for idx in range(torch.cuda.device_count())}
    memory["cpu"] = cpu_limit
    return memory


def _model_input_device(model, fallback: str):
    if hasattr(model, "module"):
        model = model.module
    embeddings = model.get_input_embeddings()
    if embeddings is not None:
        return embeddings.weight.device
    return next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device(fallback)


def _clean_messages(messages: List[Dict]) -> List[Dict[str, str]]:
    cleaned = []
    for message in messages:
        role = str(message.get("role", "user")).strip().lower()
        content = str(message.get("content", "")).strip()
        if role not in {"system", "assistant"}:
            role = "user"
        if content:
            cleaned.append({"role": role, "content": content})
    return cleaned


def _first_text(example: Dict, names: List[str]) -> str:
    for name in names:
        value = example.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _example_messages(example: Dict, cfg: MedicConfig) -> List[Dict[str, str]]:
    if cfg.messages_column in example and example[cfg.messages_column]:
        messages = _clean_messages(example[cfg.messages_column])
        return [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    if "conversations" in example and example["conversations"]:
        mapped = []
        for item in example["conversations"]:
            role = item.get("from", item.get("role", "user"))
            role = "assistant" if str(role).lower() in {"assistant", "gpt"} else "user"
            mapped.append({"role": role, "content": item.get("value", item.get("content", ""))})
        return [{"role": "system", "content": SYSTEM_PROMPT}] + _clean_messages(mapped)

    if "patterns" in example and "responses" in example:
        try:
            patterns = ast.literal_eval(str(example["patterns"]))
            responses = ast.literal_eval(str(example["responses"]))
            question = str(patterns[0]).strip() if patterns else ""
            answer = str(responses[0]).strip() if responses else ""
            if question and answer:
                return [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
        except (SyntaxError, ValueError):
            pass

    question = _first_text(
        example,
        [cfg.prompt_column, "instruction", "question", "input", "Query", "Question", "Patient", "Description"],
    )
    answer = _first_text(
        example,
        [
            cfg.response_column,
            "output",
            "answer",
            "response",
            "Answer",
            "Response",
            "Doctor",
            "long_answer",
            "response (content)",
        ],
    )

    if example.get("input") and example.get("instruction") and example.get("output"):
        question = f"{example['instruction']}\n\n{example['input']}".strip()
        answer = str(example["output"]).strip()

    if "reasoning (reasoning_content)" in example and answer:
        reasoning = str(example.get("reasoning (reasoning_content)", "")).strip()
        if reasoning:
            answer = f"{reasoning}\n\nFinal answer: {answer}"

    if "Complex_CoT" in example and answer:
        reasoning = str(example.get("Complex_CoT", "")).strip()
        if reasoning:
            answer = f"{reasoning}\n\nFinal answer: {answer}"

    if not question or not answer or len(answer.strip()) < 8:
        raise KeyError(f"Cannot map dataset row with columns: {sorted(example.keys())}")

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def _tokenize_chat(tokenizer, messages: List[Dict], max_length: int) -> Dict[str, List[int]]:
    messages = _clean_messages(messages)
    if not messages:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        max_length=max_length,
    )
    labels = [-100] * len(input_ids)

    prefix: List[Dict[str, str]] = []
    prev_len = 0
    for message in messages:
        current = prefix + [message]
        current_ids = tokenizer.apply_chat_template(
            current,
            tokenize=True,
            add_generation_prompt=False,
            truncation=True,
            max_length=max_length,
        )
        current_len = min(len(current_ids), len(input_ids))
        if message["role"] == "assistant" and current_len > prev_len:
            labels[prev_len:current_len] = input_ids[prev_len:current_len]
        prefix = current
        prev_len = current_len
        if prev_len >= len(input_ids):
            break

    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def _tokenize_example(tokenizer, example: Dict, cfg: MedicConfig) -> Dict[str, List[int]]:
    try:
        return _tokenize_chat(tokenizer, _example_messages(example, cfg), cfg.seq_len)
    except (KeyError, TypeError, ValueError, SyntaxError):
        return {"input_ids": [], "attention_mask": [], "labels": []}


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",")]


def _load_training_dataset(cfg: MedicConfig, tokenizer):
    dataset_ids = [item for item in _split_csv(cfg.dataset_id) if item]
    configs = _split_csv(cfg.dataset_config)
    splits = [item for item in _split_csv(cfg.dataset_split) if item]
    if len(configs) == 1 and len(dataset_ids) > 1:
        configs = configs * len(dataset_ids)
    if len(splits) == 1 and len(dataset_ids) > 1:
        splits = splits * len(dataset_ids)
    if len(configs) < len(dataset_ids):
        configs += [""] * (len(dataset_ids) - len(configs))
    if len(dataset_ids) != len(splits):
        raise ValueError(f"Dataset/split mismatch: {dataset_ids} vs {splits}")

    datasets = []
    per_dataset_max = cfg.max_samples // len(dataset_ids) if cfg.max_samples > 0 and len(dataset_ids) > 1 else cfg.max_samples
    for dataset_id, config_name, split in zip(dataset_ids, configs, splits):
        local_path = Path(dataset_id)
        if per_dataset_max > 0:
            if local_path.exists():
                stream = load_dataset("json", data_files=str(local_path), split=split, streaming=True)
            else:
                stream = load_dataset(dataset_id, config_name or None, split=split, streaming=True)
            ds = Dataset.from_list(list(islice(stream, per_dataset_max)))
        else:
            if local_path.exists():
                ds = load_dataset("json", data_files=str(local_path), split=split)
            else:
                ds = load_dataset(dataset_id, config_name or None, split=split)
        ds = ds.map(
            lambda example: _tokenize_example(tokenizer, example, cfg),
            remove_columns=ds.column_names,
        )
        datasets.append(ds)
    return concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]


def _collate_batch(tokenizer, examples: List[Dict]) -> Dict[str, torch.Tensor]:
    labels = [example["labels"] for example in examples]
    features = [{"input_ids": example["input_ids"], "attention_mask": example["attention_mask"]} for example in examples]
    batch = tokenizer.pad(features, padding=True, return_tensors="pt")
    max_len = batch["input_ids"].shape[1]
    batch["labels"] = torch.tensor([row + [-100] * (max_len - len(row)) for row in labels], dtype=torch.long)
    return batch


def main() -> None:
    cfg = MedicConfig()
    os.makedirs(cfg.adapter_dir, exist_ok=True)

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    if rank == 0:
        print(f"[lora] artifact={cfg.artifact_name}")
        print(f"[lora] prepared_base={cfg.prepared_base_dir}")
        print(f"[lora] dataset={cfg.dataset_id} split={cfg.dataset_split}")
        print(f"[lora] world_size={world_size}")
        print(f"[lora] memory_before_load={_memory_snapshot()}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.prepared_base_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_qlora = os.getenv("MEDIC_QLORA", "1") == "1"
    quantization_config = None
    if use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=_cuda_dtype(),
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.prepared_base_dir,
        torch_dtype=_cuda_dtype(),
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        device_map=_qlora_device_map(local_rank) if use_qlora and device == "cuda" else None,
        max_memory=_qlora_max_memory() if use_qlora and device == "cuda" else None,
    )
    model.config.use_cache = False
    if use_qlora:
        model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    if not use_qlora:
        model.to(device)
    if world_size > 1 and os.getenv("MEDIC_DEVICE_MAP", "single").strip().lower() != "auto":
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    model.train()

    ds = _load_training_dataset(cfg, tokenizer)
    ds = ds.filter(lambda row: any(label != -100 for label in row["labels"]) and len(row["input_ids"]) > 0)
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        ds,
        batch_size=cfg.micro_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
        collate_fn=lambda examples: _collate_batch(tokenizer, examples),
    )

    trainable_params = (p for p in model.parameters() if p.requires_grad)
    optimizer = bnb.optim.PagedAdamW8bit(trainable_params, lr=cfg.lora_lr, weight_decay=0.0)

    step = 0
    optimizer.zero_grad(set_to_none=True)
    while step < cfg.lora_steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for i, batch in enumerate(loader, start=1):
            step += 1
            input_device = _model_input_device(model, device)
            input_ids = batch["input_ids"].to(input_device)
            attention_mask = batch["attention_mask"].to(input_device)
            labels = batch["labels"].to(input_device)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            shift_logits = out.logits[:, :-1, :].contiguous().float()
            shift_labels = labels[:, 1:].contiguous().to(shift_logits.device)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite LoRA loss at step {step}: {float(loss.detach().cpu())}")
            (loss / cfg.grad_accum_steps).backward()

            if step % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % cfg.log_every == 0 and rank == 0:
                print(f"[lora] step={step}/{cfg.lora_steps} batch={i}/{len(loader)} loss={float(loss.detach().cpu()):.4f}")
            if step >= cfg.lora_steps:
                break

    if rank == 0:
        saver = model.module if hasattr(model, "module") else model
        saver.save_pretrained(cfg.adapter_dir)
        tokenizer.save_pretrained(cfg.adapter_dir)
        print(f"[lora] training finished -> {cfg.adapter_dir}")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
