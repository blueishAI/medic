import os
from dataclasses import dataclass


@dataclass
class MedicConfig:
    # Artifact identity
    param_label: str = os.getenv("MEDIC_PARAM_LABEL", "1.5b")
    variant: str = os.getenv("MEDIC_VARIANT", "firstaid")

    # Anant source. Default is the fastest Anant path.
    base_model_id: str = os.getenv("MEDIC_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    anant_repo_id: str = os.getenv("MEDIC_ANANT_REPO", "Bluish-AI/anant")
    anant_adapter_subdir: str = os.getenv("MEDIC_ANANT_ADAPTER_SUBDIR", "adaptors-anant-base")
    prepared_base_dir_env: str = os.getenv("MEDIC_PREPARED_BASE_DIR", "")

    # LoRA training schedule tuned for Kaggle T4 x2 and tiny-device output.
    seq_len: int = int(os.getenv("MEDIC_SEQ_LEN", "192"))
    micro_batch_size: int = int(os.getenv("MEDIC_MICRO_BATCH", "1"))
    grad_accum_steps: int = int(os.getenv("MEDIC_GRAD_ACCUM", "16"))
    log_every: int = int(os.getenv("MEDIC_LOG_EVERY", "20"))
    save_every: int = int(os.getenv("MEDIC_SAVE_EVERY", "200"))
    lora_r: int = int(os.getenv("MEDIC_LORA_R", "8"))
    lora_alpha: int = int(os.getenv("MEDIC_LORA_ALPHA", "16"))
    lora_dropout: float = float(os.getenv("MEDIC_LORA_DROPOUT", "0.03"))
    lora_lr: float = float(os.getenv("MEDIC_LORA_LR", "1e-4"))
    lora_steps: int = int(os.getenv("MEDIC_LORA_STEPS", "1200"))

    # Data. Multiple datasets/splits are comma-separated.
    dataset_id: str = os.getenv(
        "MEDIC_DATASET",
        "first_aid_seed.jsonl,i-am-mushfiq/FirstAidQA,nuhmanpk/firstaid-treatment-instruct,lextale/FirstAidInstructionsDataset,belvisk/First-Aid-Dataset,badri55/First_aid__dataset,lavita/medical-qa-datasets,ruslanmv/ai-medical-chatbot,FreedomIntelligence/Medical-R1-Distill-Data,FreedomIntelligence/medical-o1-reasoning-SFT,medalpaca/medical_meadow_medqa,medalpaca/medical_meadow_wikidoc,medalpaca/medical_meadow_medical_flashcards,keivalya/MedQuad-MedicalQnADataset,qiaojin/PubMedQA",
    )
    dataset_config: str = os.getenv("MEDIC_DATASET_CONFIG", ",,,,,,all-processed,,,en,,,,,pqa_labeled")
    dataset_split: str = os.getenv(
        "MEDIC_DATASET_SPLIT",
        "train,train,train,train,train,train,train,train,train,train,train,train,train,train,train",
    )
    max_samples: int = int(os.getenv("MEDIC_MAX_SAMPLES", "500000"))
    messages_column: str = os.getenv("MEDIC_MESSAGES_COLUMN", "messages")
    prompt_column: str = os.getenv("MEDIC_PROMPT_COLUMN", "prompt")
    response_column: str = os.getenv("MEDIC_RESPONSE_COLUMN", "response")

    # Paths
    work_dir: str = os.getenv("MEDIC_WORK_DIR", "/kaggle/working")
    output_dir: str = os.getenv("MEDIC_OUTPUT_DIR", "/kaggle/working/output_cuda")
    hf_cache: str = os.getenv("HF_HOME", "/kaggle/temp/hf_cache")

    @property
    def artifact_name(self) -> str:
        return f"medic-{self.param_label}-{self.variant}"

    @property
    def prepared_base_dir(self) -> str:
        if self.prepared_base_dir_env:
            return self.prepared_base_dir_env
        return os.path.join(self.output_dir, "prepared", f"anant-{self.param_label}")

    @property
    def adapter_dir(self) -> str:
        return os.path.join(self.output_dir, f"adaptors-medic-{self.variant}")

    @property
    def merged_dir(self) -> str:
        return os.path.join(self.output_dir, "merged", f"{self.artifact_name}-F16")

    @property
    def gguf_dir(self) -> str:
        return os.path.join(self.output_dir, "gguf")

    @property
    def validation_log_path(self) -> str:
        return os.path.join(self.output_dir, "logs", f"{self.artifact_name}-validation.log")
