import torch
from torch.utils.data import Dataset, ConcatDataset
import transformers
import datasets
from typing import List, Dict, Any, Optional
import dataclasses
import random
from finetune_config import ModelConfig, TrainConfig

class AudioTextFinetuneDataset(Dataset):
    def __init__(self, dataset_name: str, subset: Optional[str], split: str, 
                 train_config: TrainConfig, 
                 processor: transformers.AutoProcessor, 
                 model_config: ModelConfig, 
                 tokenizer: transformers.PreTrainedTokenizer):
        
        self.sampling_rate = 16000
        self.config = train_config
        self.processor = processor
        self.tokenizer = tokenizer
        
        print(f"Loading dataset: {dataset_name} ({subset}) split={split}")
        # Handle cases where subset is None
        if subset:
            self.dataset = datasets.load_dataset(dataset_name, subset, split=split, trust_remote_code=True)
        else:
            self.dataset = datasets.load_dataset(dataset_name, split=split, trust_remote_code=True)
            
        self.dataset = self.dataset.cast_column("audio", datasets.Audio(sampling_rate=self.sampling_rate))
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # --- Audio Processing ---
        audio_array = item["audio"]["array"]
        sampling_rate = item["audio"]["sampling_rate"]
        
        audio = torch.from_numpy(audio_array).float()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        elif audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
            
        audio_inputs = self.processor(
            audio.squeeze().numpy(), 
            sampling_rate=sampling_rate or self.sampling_rate, 
            return_tensors="pt"
        )
        audio_values = audio_inputs.input_features.squeeze(0)
        
        # --- Text Processing (Instruction + Answer) ---
        # Determine instruction
        
        # Try different column aliases for instruction
        instruction = item.get("instruction_prompt") or item.get("instruction") or item.get("prompt")
        
        if instruction is None:
            # User Feedback: "there should be no prompts, in those kind of dataset"
            # We treat missing instruction as empty (text continuation scenario)
            instruction = ""
            
        # Try different column aliases for answer/target
        # Prioritize 'generated_answer' as per user dataset, then 'answer', 'text', 'sentence'
        answer = item.get("generated_answer") or item.get("answer") or item.get("text") or item.get("sentence") or ""
        
        # Format: "User: <instruction>\nAssistant: <answer>"
        # Note: Audio is prepended by the model, so we just format the text part.
        
        # Construct the full text for input_ids
        # We assume the model prepends audio features.
        # Template: "User: {instruction}\nAssistant:"
        prompt_str = self.config.instruction_template.format(instruction=instruction)
        full_text = prompt_str + " " + str(answer) + self.tokenizer.eos_token
        
        # Tokenize
        # We need to compute where the answer starts to mask the loss for the instruction
        
        # 1. Tokenize prompt only
        prompt_ids = self.tokenizer(prompt_str, add_special_tokens=True, return_tensors="pt").input_ids.squeeze(0)
        
        # 2. Tokenize full text
        # Note: simple concatenation of ids might not work if tokenizer merges tokens across boundaries, better to tokenize full string
        full_ids = self.tokenizer(full_text, add_special_tokens=True, return_tensors="pt").input_ids.squeeze(0)
        
        # Calculate mask
        # We want labels to be -100 for propert_ids, and actual ids for answer
        # Heuristic: Match the length. 
        # Caution: Tokenizer might tokenize "User:" differently at start vs middle.
        # Usually it's safer to use the length of prompt_ids as the mask length, 
        # provided the prompt is at the start.
        
        labels = full_ids.clone()
        prompt_len = prompt_ids.shape[0]
        
        # If full_ids is shorter than prompt (shouldn't happen unless answer is empty and weird tokenization), clamp
        if prompt_len > labels.shape[0]:
             prompt_len = labels.shape[0]
             
        # Mask the prompt
        labels[:prompt_len] = -100
        
        input_ids = full_ids
        
        return {
            "audio_values": audio_values,
            "input_ids": input_ids,
            "labels": labels,
            "instruction": instruction,
            "answer": answer
        }

def get_dataset(train_config: TrainConfig, processor, model_config, tokenizer):
    datasets_list = []
    
    for name, subset, split in zip(train_config.dataset_names, train_config.dataset_subsets, train_config.dataset_splits):
        ds = AudioTextFinetuneDataset(
            dataset_name=name,
            subset=subset,
            split=split,
            train_config=train_config,
            processor=processor,
            model_config=model_config,
            tokenizer=tokenizer
        )
        datasets_list.append(ds)
        
    if len(datasets_list) > 1:
        return ConcatDataset(datasets_list)
    return datasets_list[0]

@dataclasses.dataclass
class DataCollator:
    processor: transformers.AutoProcessor
    tokenizer: transformers.PreTrainedTokenizer
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        audio_values = [f["audio_values"] for f in features]
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        
        # Pad Audio
        # Check shapes
        if audio_values[0].shape[-1] == 3000:
             # Fixed size (Whisper standard often is 3000 frames for 30s)
             audio_batch = torch.stack(audio_values)
        else:
             # Variable length, pad
             audio_values_T = [a.T for a in audio_values]
             audio_batch_T = torch.nn.utils.rnn.pad_sequence(audio_values_T, batch_first=True)
             audio_batch = audio_batch_T.transpose(1, 2)

        # Pad Text
        input_ids_batch = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_batch = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        
        return {
            "audio_values": audio_batch,
            "input_ids": input_ids_batch,
            "labels": labels_batch,
            "attention_mask": (input_ids_batch != self.tokenizer.pad_token_id).long(),
        }
