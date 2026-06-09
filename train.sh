#!/usr/bin/env bash
set -euo pipefail

if [[ "${MEDIC_VERBOSE:-0}" == "1" ]]; then
  set -x
  export MEDIC_LOG_EVERY="${MEDIC_LOG_EVERY:-1}"
fi

cd "$(dirname "$0")"

export MEDIC_WORK_DIR="/kaggle/working"
export MEDIC_OUTPUT_DIR="/kaggle/working/output_cuda"
export HF_HOME="/kaggle/temp/hf_cache"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_TORCHVISION=1
export USE_TF=0
export CUDA_VISIBLE_DEVICES="0,1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export LD_LIBRARY_PATH="/kaggle/temp/llama.cpp/build/bin:${LD_LIBRARY_PATH:-}"

mkdir -p "${MEDIC_OUTPUT_DIR}/logs" "${HF_HOME}" /kaggle/temp
: > "${MEDIC_OUTPUT_DIR}/logs/train.log"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Use Kaggle GPU T4 x2." >&2
  exit 1
fi

GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed 's/^ *//;s/ *$//')"
GPU_COUNT="$(printf '%s\n' "${GPU_NAMES}" | sed '/^$/d' | wc -l)"
echo "[setup] detected GPUs:"
printf '%s\n' "${GPU_NAMES}" | sed 's/^/[setup] - /'

if [[ "${GPU_COUNT}" -lt 2 ]] || ! printf '%s\n' "${GPU_NAMES}" | grep -q "T4"; then
  echo "Expected Kaggle accelerator GPU T4 x2." >&2
  exit 1
fi

bash setup.sh

quantize_bin() {
  if [[ -x /kaggle/temp/llama.cpp/build/bin/llama-quantize ]]; then
    echo /kaggle/temp/llama.cpp/build/bin/llama-quantize
  elif [[ -x /kaggle/temp/llama.cpp/build/bin/quantize ]]; then
    echo /kaggle/temp/llama.cpp/build/bin/quantize
  else
    echo "Missing llama.cpp quantize binary" >&2
    exit 1
  fi
}

convert_script() {
  if [[ -f /kaggle/temp/llama.cpp-src/convert_hf_to_gguf.py ]]; then
    echo /kaggle/temp/llama.cpp-src/convert_hf_to_gguf.py
  elif [[ -f /kaggle/temp/llama.cpp/convert_hf_to_gguf.py ]]; then
    echo /kaggle/temp/llama.cpp/convert_hf_to_gguf.py
  else
    echo "Missing llama.cpp HF-to-GGUF converter" >&2
    exit 1
  fi
}

llama_cli_bin() {
  if [[ -x /kaggle/temp/llama.cpp/build/bin/llama-cli ]]; then
    echo /kaggle/temp/llama.cpp/build/bin/llama-cli
  elif [[ -x /kaggle/temp/llama.cpp/build/bin/main ]]; then
    echo /kaggle/temp/llama.cpp/build/bin/main
  else
    echo "Missing llama.cpp CLI binary" >&2
    exit 1
  fi
}

smoke_test_gguf() {
  local artifact="$1"
  local quant="$2"
  local gguf="${MEDIC_OUTPUT_DIR}/${artifact}-${quant}.gguf"
  local log="${MEDIC_OUTPUT_DIR}/logs/${artifact}-${quant}-smoke.log"
  local prompt="A person cut their palm and it is bleeding. Give first aid steps."

  echo "[smoke] ${artifact}-${quant}"
  {
    echo "MODEL: ${artifact}-${quant}"
    echo "PROMPT: ${prompt}"
    timeout 180 "$(llama_cli_bin)" \
      -m "${gguf}" \
      -ngl 0 \
      -c 512 \
      -n 96 \
      --temp 0.2 \
      --top-p 0.9 \
      --no-conversation \
      -p "<|im_start|>system
You are Medic, a tiny first-aid assistant. Give fast, practical, safe first-aid steps. Tell the user when to call emergency services. Do not diagnose. Do not replace a clinician.<|im_end|>
<|im_start|>user
${prompt}<|im_end|>
<|im_start|>assistant
"
  } 2>&1 | tee "${log}" || {
    status="$?"
    echo "[smoke] ${artifact}-${quant} exited ${status}; continuing." | tee -a "${log}"
  }
}

run_variant() {
  local param_label="$1"
  local variant="$2"
  local base_model="$3"
  local anant_adapter="$4"
  local steps="$5"
  local seq_len="$6"
  local micro_batch="$7"
  local grad_accum="$8"
  local lr="$9"
  local lora_r="${10}"
  local lora_alpha="${11}"
  local max_samples="${12}"

  export MEDIC_PARAM_LABEL="${param_label}"
  export MEDIC_VARIANT="${variant}"
  export MEDIC_BASE_MODEL="${base_model}"
  export MEDIC_ANANT_REPO="${MEDIC_ANANT_REPO:-Bluish-AI/anant}"
  export MEDIC_ANANT_ADAPTER_SUBDIR="${anant_adapter}"
  export MEDIC_LORA_STEPS="${steps}"
  export MEDIC_SEQ_LEN="${seq_len}"
  export MEDIC_MICRO_BATCH="${micro_batch}"
  export MEDIC_GRAD_ACCUM="${grad_accum}"
  export MEDIC_LORA_LR="${lr}"
  export MEDIC_LORA_R="${lora_r}"
  export MEDIC_LORA_ALPHA="${lora_alpha}"
  export MEDIC_LORA_DROPOUT="0.03"
  export MEDIC_MAX_SAMPLES="${max_samples}"
  export MEDIC_STRICT_VALIDATE="${MEDIC_STRICT_VALIDATE:-0}"
  export MEDIC_LOG_EVERY="20"
  export MEDIC_QLORA="1"
  export MEDIC_DEVICE_MAP="auto"
  export MEDIC_MAX_MEMORY_GPU="13GiB"
  export MEDIC_MAX_MEMORY_CPU="24GiB"

  local artifact="medic-${param_label}-${variant}"
  local merged_dir="${MEDIC_OUTPUT_DIR}/merged/${artifact}-F16"
  local gguf_f16="${MEDIC_OUTPUT_DIR}/${artifact}-F16.gguf"
  local gguf_q4="${MEDIC_OUTPUT_DIR}/${artifact}-Q4_K_M.gguf"
  local gguf_q8="${MEDIC_OUTPUT_DIR}/${artifact}-Q8_0.gguf"

  echo "[prepare] ${artifact}"
  python prepare_anant_base.py

  echo "[train] ${artifact}"
  python lora_train.py 2>&1 | tee -a "${MEDIC_OUTPUT_DIR}/logs/train.log"

  echo "[merge] ${artifact}"
  python lora_merge.py

  echo "[validate] ${artifact}"
  python lora_validate.py

  echo "[gguf] ${artifact} F16"
  python "$(convert_script)" "${merged_dir}" --outfile "${gguf_f16}" --outtype f16

  echo "[gguf] ${artifact} Q4_K_M"
  "$(quantize_bin)" "${gguf_f16}" "${gguf_q4}" Q4_K_M

  echo "[gguf] ${artifact} Q8_0"
  "$(quantize_bin)" "${gguf_f16}" "${gguf_q8}" Q8_0

  rm -f "${gguf_f16}"
  smoke_test_gguf "${artifact}" Q4_K_M
  smoke_test_gguf "${artifact}" Q8_0
}

run_variant 1.5b firstaid "Qwen/Qwen2.5-1.5B-Instruct" "adaptors-anant-base" \
  4000 192 1 16 8e-5 8 16 500000

echo "[final] Package artifacts"
cp modelfile-medic-q4 modelfile-medic-q8 speed_run.sh "${MEDIC_OUTPUT_DIR}/"
tar -C "${MEDIC_OUTPUT_DIR}" -czf "${MEDIC_OUTPUT_DIR}/medic_cuda_artifacts.tar.gz" \
  "adaptors-medic-firstaid" \
  "medic-1.5b-firstaid-Q4_K_M.gguf" \
  "medic-1.5b-firstaid-Q8_0.gguf" \
  "modelfile-medic-q4" \
  "modelfile-medic-q8" \
  "speed_run.sh" \
  "logs"

echo "Medic CUDA QLoRA pipeline complete."
ls -lh "${MEDIC_OUTPUT_DIR}" | sed -n '1,200p'
