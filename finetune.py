import os
import dataclasses
import torch
import transformers
from transformers import Trainer, TrainingArguments, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from huggingface_hub import HfApi, login, hf_hub_download
import safetensors.torch
import wandb
import numpy as np
from dotenv import load_dotenv
try:
    import evaluate
except ImportError:
    print("Evaluate library not found. WER metric will be disabled. Install with `pip install evaluate jiwer`")
    evaluate = None

# Import our new modules
from finetune_config import TrainConfig, ModelConfig
from model import MultiModalModel
from finetune_data import AudioTextFinetuneDataset, DataCollator

class SamplePredictionCallback(TrainerCallback):
    """Every N steps, print ground-truth vs model-predicted transcript for a few samples."""

    def __init__(self, tokenizer, data_collator, train_dataset, sample_every_n_steps: int = 100, num_samples: int = 2, prompt_template: str = None):
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.sample_every_n_steps = sample_every_n_steps
        self.num_samples = num_samples
        self.prompt_template = prompt_template or "User: Transcribe the following audio.\nAssistant:"
        
    def on_log(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 or state.global_step % self.sample_every_n_steps != 0:
            return
        if model is None:
            return
            
        # Ensure model is in eval mode
        model.eval()
        device = next(model.parameters()).device
        
        try:
            # Pick random samples
            indices = [torch.randint(0, len(self.train_dataset), (1,)).item() for _ in range(self.num_samples)]
            # Use collator to prepare batch
            raw_samples = [self.train_dataset[i] for i in indices]
            batch = self.data_collator(raw_samples)
            
            # Move to device
            audio_values = batch["audio_values"].to(device)
            # We construct a prompt for generation. 
            # Note: For generation, we want to feed [Audio] + [Instruction] and let it generate [Answer]
            # The raw_samples have 'instruction' field from our custom dataset
            
            instructions = [s.get("instruction", "Transcribe the following audio.") for s in raw_samples]
            ground_truths = [s.get("answer", "") for s in raw_samples]
            
            # Create input ids for the prompt only
            prompts = [self.prompt_template.format(instruction=inst) for inst in instructions]
            
            prompt_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            prompt_ids = prompt_inputs.input_ids
            
            with torch.no_grad():
                # Generate
                gen_ids = model.generate(
                    input_ids=prompt_ids,
                    audio_values=audio_values,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )
            
            # Decode
            prompt_len = prompt_ids.shape[1]
            decoded_preds = self.tokenizer.batch_decode(gen_ids[:, prompt_len:], skip_special_tokens=True)
            
            # Print to console and log to W&B
            print(f"\n[Step {state.global_step}] Sample Predictions:")
            
            columns = ["Step", "Instruction", "Ground Truth", "Prediction"]
            data = []
            
            for inst, gt, pred in zip(instructions, ground_truths, decoded_preds):
                print(f"Inst: {inst[:50]}...")
                print(f"GT  : {gt[:100]}...")  # Show more of GT
                print(f"Pred: {pred[:100]}...") # Show more of Pred
                print("-" * 20)
                data.append([state.global_step, inst, gt, pred])
                
            if wandb.run is not None:
                wandb.log({"finetune_samples": wandb.Table(data=data, columns=columns)}, step=state.global_step)

        except Exception as e:
            print(f"[SamplePredictionCallback] Error: {e}")
        finally:
            model.train()

def compute_metrics(eval_pred, tokenizer):
    if evaluate is None:
        return {}
        
    metric = evaluate.load("wer")
    predictions, labels = eval_pred
    
    # Decode
    # Replace -100 in labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # WER
    wer = metric.compute(predictions=decoded_preds, references=decoded_labels)
    
    return {"wer": wer}


def train():
    load_dotenv()
    
    # 1. Configs
    train_config = TrainConfig()
    model_config = ModelConfig()
    
    print("--- Finetuning Configuration ---")
    print(train_config)
    print("------------------------------")

    # 2. WandB
    if train_config.wandb_project:
        wandb.init(
            project=train_config.wandb_project,
            entity=train_config.wandb_entity,
            name=train_config.wandb_run_name,
            config=dataclasses.asdict(train_config)
        )

    # 3. Tokenizer & Processor
    print(f"Loading tokenizer: {model_config.text_model_id}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_config.text_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"Loading processor: {model_config.audio_model_id}")
    processor = transformers.AutoProcessor.from_pretrained(model_config.audio_model_id)

    # 4. Model Initialization
    print("Initializing MultiModalModel...")
    model = MultiModalModel(model_config)
    
    # 5. Load Pre-trained Checkpoint (Projector weights, etc.)
    if train_config.load_from_checkpoint:
        print(f"Loading checkpoint from Hub: {train_config.checkpoint_repo} / {train_config.checkpoint_subfolder}")
        try:
            checkpoint_path = hf_hub_download(
                repo_id=train_config.checkpoint_repo,
                filename=train_config.checkpoint_filename,
                subfolder=train_config.checkpoint_subfolder
            )
            print(f"Downloaded checkpoint to: {checkpoint_path}")
            
            # Load state dict
            state_dict = safetensors.torch.load_file(checkpoint_path)
            
            # Load into model
            # strict=False because we might have extra keys or missing keys if architecture slightly drifted
            # or if the checkpoint only contains partial weights.
            # However, typically it should contain full model or at least projector.
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            
            print(f"Checkpoint loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            if len(missing) > 0:
                print(f"First 5 missing: {missing[:5]}")
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Proceeding with initialized weights (WARNING: This might be random initialization if not intended!)")

    # 6. LoRA / PEFT Setup
    # Freeze everything first (implicit in some workflows, but good to be explicit or let LoRA handle it)
    # The MultiModalModel init already freezes audio_encoder:
    # self.audio_encoder.parameters() -> requires_grad = False (in model.py)
    
    # We want to train Projector + LLM (via LoRA).
    # Check projector grad
    for param in model.projector.parameters():
        param.requires_grad = True # Enable projector training
        
    print("Projector weights set to require_grad = True")

    if train_config.use_lora:
        print("Applying LoRA to LLM...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=train_config.lora_r,
            lora_alpha=train_config.lora_alpha,
            lora_dropout=train_config.lora_dropout,
            target_modules=train_config.lora_target_modules
        )
        
        # We need to wrap ONLY the LLM part if we use get_peft_model on the sub-module?
        # Or wrap the whole model? 
        # get_peft_model usually wraps the base transformer.
        # model.llm is the AutoModelForCausalLM.
        
        model.llm = get_peft_model(model.llm, peft_config)
        model.llm.print_trainable_parameters()
        
    # Verify trainable parameters
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.4f}")

    # 7. Dataset
    dataset_list = []
    # If config supports multiple datasets, logic should be in finetune_data.py
    # But here we instantiate the wrapper
    
    train_dataset = AudioTextFinetuneDataset(
        dataset_name=train_config.dataset_names[0], # Using first for now, can extend to loop
        subset=train_config.dataset_subsets[0],
        split=train_config.dataset_splits[0],
        train_config=train_config,
        processor=processor,
        model_config=model_config,
        tokenizer=tokenizer
    )
    
    data_collator = DataCollator(processor, tokenizer)
    
    # 8. Trainer
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
        save_steps=train_config.save_steps,
        report_to="wandb" if train_config.wandb_project else "none",
        remove_unused_columns=False,
        save_total_limit=2,
        eval_strategy="steps" if train_config.eval_steps > 0 else "no",
        eval_steps=train_config.eval_steps,
    )
    
    sample_callback = SamplePredictionCallback(
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        sample_every_n_steps=train_config.sample_pred_every_steps,
        prompt_template=train_config.instruction_template,
    )
    
    # Helper for metrics
    def compute_metrics_wrapper(eval_pred):
        return compute_metrics(eval_pred, tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset if train_config.eval_steps > 0 else None, # Use train as eval for simplicity or split?
        # Ideally we valid on validation set, but for now user didn't specific separate valid set in config easily.
        # finetune_config has 'dataset_splits', if we wanted we could parse it.
        # For now, let's just not pass eval_dataset if we don't have explicit valid split loaded.
        # Or better, just pass train_dataset to verify overfitting/learning.
        data_collator=data_collator,
        callbacks=[sample_callback],
        compute_metrics=compute_metrics_wrapper if (evaluate is not None and train_config.eval_steps > 0) else None,
    )
    
    print("Starting Training...")
    trainer.train()
    
    # 9. Save
    print(f"Saving model to {train_config.output_dir}")
    trainer.save_model(train_config.output_dir)
    tokenizer.save_pretrained(train_config.output_dir)
    processor.save_pretrained(train_config.output_dir)
    
    # 10. Push to Hub
    if train_config.push_to_hub and train_config.hub_model_id:
        print(f"Pushing to Hub: {train_config.hub_model_id}")
        try:
             # Create API client
             if train_config.hub_token:
                 login(token=train_config.hub_token)
             api = HfApi()
             
             # Create repo explicitly (Trainer does this often but good to be safe)
             try:
                 api.create_repo(repo_id=train_config.hub_model_id, private=train_config.hub_private_repo, exist_ok=True)
             except Exception as e:
                 print(f"Repo creation warning: {e}")

             # Use Trainer to push weights/tokenizer
             kwargs = {"private": train_config.hub_private_repo} if train_config.hub_private_repo else {}
             trainer.push_to_hub(repo_id=train_config.hub_model_id, **kwargs)
             
             # Explicitly push code files for reproducibility/usage
             files_to_push = ["model.py", "finetune_config.py", "finetune_data.py", "finetune.py", "inference.py"]
             print(f"Pushing code files: {files_to_push}")
             
             for file in files_to_push:
                  if os.path.exists(file):
                       try:
                           # We push finetune_config as config.py for inference compatibility if needed?
                           # Actually, inference.py uses 'config.py', so we better push our fine-tune config as 'config.py' 
                           # so inference works out of the box with the new params?
                           # Or kept as is. Let's push as is + maybe 'config.py' -> 'original_config.py'
                           
                           target_path = file
                           # Special handling: if we want inference.py to work, it imports 'config', which is 'config.py'.
                           # 'finetune_config.py' has the same ModelConfig class structure mostly.
                           # Let's just push them as is.
                           
                           api.upload_file(
                                path_or_fileobj=file,
                                path_in_repo=target_path,
                                repo_id=train_config.hub_model_id,
                                repo_type="model",
                           )
                       except Exception as e:
                           print(f"Failed to push {file}: {e}")
                           
             print("Push success.")
        except Exception as e:
             print(f"Push failed: {e}")

if __name__ == "__main__":
    train()
