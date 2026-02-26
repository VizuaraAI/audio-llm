import os
import dataclasses
from typing import Optional, Tuple, List

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
    batch_size: int = 8           # Reduced for fine-tuning stability/memory if full model loaded
    accum_steps: int = 4          # effective batch = 32
    use_bf16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 8
    dataloader_pin_memory: bool = True

    learning_rate: float = 5e-5   # Lower LR for fine-tuning
    num_epochs: int = 3
    max_steps: int = -1           # Use epochs

    # Paths
    output_dir: str = "./finetune_checkpoints"
    
    # Dataset
    # Can be a list of dataset names or a single string
    dataset_names: List[str] = dataclasses.field(default_factory=lambda: ["Mayank022/Audio_question_answer_dataset_with_audio"])
    dataset_subsets: List[Optional[str]] = dataclasses.field(default_factory=lambda: [None])
    dataset_splits: List[str] = dataclasses.field(default_factory=lambda: ["train"])
    
    # Model Loading
    load_from_checkpoint: bool = True
    checkpoint_repo: str = "Mayank022/Audio-Language-Model"
    checkpoint_subfolder: str = "latest_checkpoint"
    checkpoint_filename: str = "model.safetensors"
    
    # Prompting
    instruction_template: str = "User: {instruction}\nAssistant:" # Simple chat format
    # default_prompts removed as we now use empty instruction for missing prompts

    # LoRA
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = dataclasses.field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    # Hub
    push_to_hub: bool = False
    hub_model_id: Optional[str] = "Mayank022/Audio-Language-Model-Finetuned"
    hub_token: Optional[str] = os.getenv("HF_TOKEN", None)
    hub_private_repo: bool = True

    # WandB
    wandb_project: str = os.getenv("WANDB_PROJECT", "audio-language-model-finetune")
    wandb_entity: Optional[str] = os.getenv("WANDB_ENTITY", None)
    wandb_run_name: Optional[str] = None
    wandb_watch: str = "false" 
    wandb_log_model: str = "false"

    # Misc
    seed: int = 42
    log_steps: int = 10
    eval_steps: int = 200
    save_steps: int = 200
    sample_pred_every_steps: int = 100
