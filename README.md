# Medic Training

Kaggle T4 x2 QLoRA pipeline for Medic, trained on top of Anant.

Run:

```bash
bash train.sh
```

Outputs:

```text
/kaggle/working/output_cuda/medic-1.5b-firstaid-Q4_K_M.gguf
/kaggle/working/output_cuda/medic-1.5b-firstaid-Q8_0.gguf
/kaggle/working/output_cuda/adaptors-medic-firstaid/
/kaggle/working/output_cuda/medic_cuda_artifacts.tar.gz
```

For maximum speed, ship `Q4_K_M` first. It is the main release file. `Q8_0` is slower and only for better quality.

The modelfile caps context and answer length:

```bash
ollama create medic -f modelfile-medic-q4
```

Fast llama.cpp CPU run:

```bash
bash speed_run.sh ./medic-1.5b-firstaid-Q4_K_M.gguf "Someone is choking. What do I do?"
```

Default base path:

1. Merge `Qwen/Qwen2.5-1.5B-Instruct` with `Bluish-AI/anant/adaptors-anant-base`.
2. Train Medic QLoRA on first-aid/medical instruction data.
3. Merge Medic adapter.
4. Convert to GGUF.
5. Quantize to `Q4_K_M` and `Q8_0`.

Default data mix is first-aid only. Do not add broad medical QA or reasoning datasets for this model; they teach diagnosis-style answers and rambling.

```text
first_aid_seed.jsonl repeated 5x
i-am-mushfiq/FirstAidQA
```

`nuhmanpk/firstaid-treatment-instruct` is intentionally not in the default mix. It is large and first-aid-labeled, but many rows are passage-summary/chunk tasks; use it only with the training quality filter enabled.

Training uses QLoRA by default with `MEDIC_DEVICE_MAP=auto`, so one 4-bit model is spread across the two T4 GPUs instead of loading duplicate copies.

Override data:

```bash
export MEDIC_DATASET="your/dataset"
export MEDIC_DATASET_SPLIT="train"
export MEDIC_DATASET_CONFIG=""
bash train.sh
```

For the bigger Anant reason base, edit the final `run_variant` line in `train.sh`:

```bash
run_variant 3b firstaid "Qwen/Qwen2.5-3B-Instruct" "adaptors-anant-reason" ...
```
