import os
import dataclasses
from typing import Optional, Tuple

@dataclasses.dataclass
class ModelConfig:
    audio_model_id: str = "openai/whisper-medium"
    text_model_id: str = "sarvamai/sarvam-m"
    hidden_size: int = 2048
    projector_act: str = "gelu"
    stack_factor: int = 8

    def to_dict(self):
        return dataclasses.asdict(self)

@dataclasses.dataclass
class TrainConfig:
    # --- Batch & GPU (tuned for A100 80GB) ---
    batch_size: int = 32          # per-device; try 64 if no OOM
    accum_steps: int = 2          # effective batch = 32*2=64; reduce if OOM
    use_bf16: bool = True         # A100 native bf16: faster + less VRAM
    gradient_checkpointing: bool = True  # set True if OOM to trade compute for memory
    dataloader_num_workers: int = 8
    dataloader_pin_memory: bool = True

    learning_rate: float = 1e-4
    num_epochs: int = 1
    max_steps: int = 10000 # Use either epochs or steps

    # Paths
    output_dir: str = "./checkpoints"
    # data_path: str = "./data/train.jsonl" # REMOVED
    dataset_name: str = "fixie-ai/common_voice_17_0"
    dataset_subset: str = "hi" # Hindi
    dataset_split: str = "train"
    val_dataset_split: str = "validation"
    
    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Hub
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None  # e.g. "username/model-name"
    hub_token: Optional[str] = None
    hub_private_repo: bool = True

    # WandB
    wandb_project: str = os.getenv("WANDB_PROJECT", "audio-language-model")
    wandb_entity: Optional[str] = os.getenv("WANDB_ENTITY", None)
    wandb_run_name: Optional[str] = None
    wandb_watch: str = "false" # "gradients", "all", "false"
    wandb_log_model: str = "false" # "true", "false"

    # Misc
    seed: int = 42
    log_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    sample_pred_every_steps: int = 100  # print ground-truth vs predicted transcript every N steps
