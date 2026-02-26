import os
import dataclasses
import torch
import transformers
from transformers import Trainer, TrainingArguments, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import HfApi, login
import wandb
from dotenv import 
load_dotenv()
from config import TrainConfig, ModelConfig
from model import MultiModalModel
from data import AudioTextDataset, DataCollator


class SamplePredictionCallback(TrainerCallback):
    """Every N steps, print ground-truth vs model-predicted transcript for a few samples."""

    def __init__(self, tokenizer, data_collator, train_dataset, sample_every_n_steps: int = 100, num_samples: int = 2, prompt: str = "Transcribe the following audio:"):
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.sample_every_n_steps = sample_every_n_steps
        self.num_samples = num_samples
        self.prompt = prompt
    def on_log(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 or state.global_step % self.sample_every_n_steps != 0:
            return
        if model is None:
            return
        model.eval()
        device = next(model.parameters()).device
        try:
            indices = [i % len(self.train_dataset) for i in range(self.num_samples)]
            batch = self.data_collator([self.train_dataset[i] for i in indices])
            audio_values = batch["audio_values"].to(device)
            labels_batch = batch["labels"]
            continuations = batch.get("continuation", [""] * audio_values.size(0))
            prompt_ids = self.tokenizer(self.prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
            prompt_ids = prompt_ids.expand(audio_values.size(0), -1)
            with torch.no_grad():
                gen_ids = model.generate(
                    input_ids=prompt_ids,
                    audio_values=audio_values,
                    max_new_tokens=120,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            prompt_len = prompt_ids.size(1)
            
            # Create a wandb Table
            columns = ["Step", "Audio Index", "Ground Truth", "Prediction", "Continuation"]
            table = wandb.Table(columns=columns)
            
            print(f"\n[WandB] Logging sample predictions at step {state.global_step}")
            
            for i in range(audio_values.size(0)):
                gt_tokens = [t for t in labels_batch[i].tolist() if t != -100]
                gt_text = self.tokenizer.decode(gt_tokens, skip_special_tokens=True).strip()
                pred_text = self.tokenizer.decode(gen_ids[i][prompt_len:], skip_special_tokens=True).strip()
                
                cont_ref = continuations[i] if i < len(continuations) else ""
                
                # Add row to table
                table.add_data(state.global_step, i, gt_text, pred_text, cont_ref)
                
            # Log the table to wandb
            if wandb.run is not None:
                wandb.log({"sample_predictions": table}, step=state.global_step)
            else:
                print("Warning: WandB run not active, skipping logging.")

        except Exception as e:
            print(f"[SamplePredictionCallback] Error: {e}\n")
        finally:
            model.train()


def train():
    # Load environment variables
    load_dotenv()

    # Load Configs
    train_config = TrainConfig()
    model_config = ModelConfig()
    
    # Initialize WandB
    wandb.init(
        project=train_config.wandb_project,
        entity=train_config.wandb_entity,
        name=train_config.wandb_run_name,
        config=dataclasses.asdict(train_config),
    )

    
    # Initialize Tokenizer & Processor
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_config.text_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    processor = transformers.AutoProcessor.from_pretrained(model_config.audio_model_id)
    
    # Initialize Model
    model = MultiModalModel(model_config)
    
    # Apply LoRA if requested
    if train_config.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=train_config.lora_r, 
            lora_alpha=train_config.lora_alpha, 
            lora_dropout=train_config.lora_dropout,
            target_modules=["q_proj", "v_proj"]
        )
        model.llm = get_peft_model(model.llm, peft_config)
        model.llm.print_trainable_parameters()
        
    # Dataset
    train_dataset = AudioTextDataset(train_config, processor, model_config, tokenizer)
    data_collator = DataCollator(processor, tokenizer)
    
    # Training Arguments (tuned for A100 80GB: bf16, larger batch, fast dataloader)
    training_args = TrainingArguments(
        output_dir=train_config.output_dir,
        per_device_train_batch_size=train_config.batch_size,
        gradient_accumulation_steps=train_config.accum_steps,
        learning_rate=train_config.learning_rate,
        num_train_epochs=train_config.num_epochs,
        max_steps=train_config.max_steps,
        bf16=train_config.use_bf16,
        gradient_checkpointing=train_config.gradient_checkpointing,
        dataloader_num_workers=train_config.dataloader_num_workers,
        dataloader_pin_memory=train_config.dataloader_pin_memory,
        logging_steps=train_config.log_steps,
        logging_first_step=True,
        logging_nan_inf_filter=True,
        save_steps=train_config.save_steps,
        eval_strategy="no",  # change if val set provided
        remove_unused_columns=False,  # Important because we have custom forward signature
        report_to="wandb",
        log_level="info",
        log_level_replica="info",
    )

    sample_callback = SamplePredictionCallback(
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        sample_every_n_steps=train_config.sample_pred_every_steps,
        num_samples=2,
        prompt="Transcribe the following audio:",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[sample_callback],
    )

    total_steps = train_config.max_steps
    print(f"\n>>> Training: max_steps={total_steps}, batch_size={train_config.batch_size}, "
          f"grad_accum={train_config.accum_steps} (effective batch={train_config.batch_size * train_config.accum_steps})")
    print(f">>> Sample predictions (GT vs predicted transcript) every {train_config.sample_pred_every_steps} steps.\n")

    trainer.train()
    
    # Save
    trainer.save_model(train_config.output_dir)
    tokenizer.save_pretrained(train_config.output_dir)
    processor.save_pretrained(train_config.output_dir)

    # Push to Hub
    if train_config.push_to_hub:
        print(f"\n>>> Pushing model to Hugging Face Hub: {train_config.hub_model_id}")
        if train_config.hub_token:
            login(token=train_config.hub_token)
        
        api = HfApi()
        
        # Create repo if needed
        # private=True by default for safety, user can adjust
        try:
            api.create_repo(repo_id=train_config.hub_model_id, private=train_config.hub_private_repo, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create repo {train_config.hub_model_id}. Error: {e}")
        
        # Upload model folder
        try:
            api.upload_folder(
                folder_path=train_config.output_dir,
                repo_id=train_config.hub_model_id,
                repo_type="model",
            )
            
            # Upload code files to ensure custom model works
            for file in ["model.py", "config.py", "data.py", "inference.py"]:
                 if os.path.exists(file):
                      api.upload_file(
                           path_or_fileobj=file,
                           path_in_repo=file,
                           repo_id=train_config.hub_model_id,
                           repo_type="model",
                      )

            print(f">>> Successfully pushed to {train_config.hub_model_id}")
        except Exception as e:
            print(f"Error pushing to hub: {e}")

if __name__ == "__main__":
    train()
