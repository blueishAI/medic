#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-./medic-1.5b-firstaid-Q4_K_M.gguf}"
PROMPT="${2:-Someone cut their hand and it is bleeding. What do I do?}"
CTX="${MEDIC_CTX:-512}"
TOKENS="${MEDIC_TOKENS:-96}"
THREADS="${MEDIC_THREADS:-}"

if [[ -z "${THREADS}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    THREADS="$(nproc)"
  else
    THREADS="4"
  fi
fi

if command -v llama-cli >/dev/null 2>&1; then
  LLAMA_BIN="llama-cli"
elif [[ -x /kaggle/temp/llama.cpp/build/bin/llama-cli ]]; then
  LLAMA_BIN="/kaggle/temp/llama.cpp/build/bin/llama-cli"
else
  echo "Missing llama-cli. Install llama.cpp or run setup.sh on Kaggle." >&2
  exit 1
fi

"${LLAMA_BIN}" \
  -m "${MODEL}" \
  -t "${THREADS}" \
  -c "${CTX}" \
  -n "${TOKENS}" \
  --temp 0.2 \
  --top-p 0.9 \
  --repeat-penalty 1.08 \
  --no-conversation \
  -p "<|im_start|>system
You are Medic, a tiny first-aid assistant. Give fast, practical, safe first-aid steps. Tell the user when to call emergency services. Do not diagnose. Do not replace a clinician.<|im_end|>
<|im_start|>user
${PROMPT}<|im_end|>
<|im_start|>assistant
"
