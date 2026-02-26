import json
import torch
from torch.utils.data import Dataset

import transformers
import datasets
from typing import List, Dict, Any, Optional
import dataclasses
from config import ModelConfig, TrainConfig

class AudioTextDataset(Dataset):
    def __init__(self, train_config: TrainConfig, processor: transformers.AutoProcessor, model_config: ModelConfig, tokenizer: transformers.PreTrainedTokenizer):
        self.sampling_rate = 16000
        print(f"Loading dataset: {train_config.dataset_name} ({train_config.dataset_subset}) split={train_config.dataset_split}")
        self.dataset = datasets.load_dataset(
            train_config.dataset_name,
            train_config.dataset_subset,
            split=train_config.dataset_split,
            verification_mode="no_checks",  # avoid NonMatchingSplitsSizesError when Hub metadata differs from cached
        )
        # Audio(sampling_rate=...) decodes and resamples via TorchCodec; requires system FFmpeg (apt install ffmpeg)
        self.dataset = self.dataset.cast_column("audio", datasets.Audio(sampling_rate=self.sampling_rate))
        
        self.processor = processor
        self.tokenizer = tokenizer
        self.model_config = model_config
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        # HF Audio returns {'audio': {'array': ..., 'sampling_rate': ...}, 'sentence': ...}
        audio_array = item["audio"]["array"]
        sampling_rate = item["audio"]["sampling_rate"]
        text = item.get("sentence", item.get("text", ""))
        continuation = item.get("continuation", item.get("continuation_text", ""))

        audio = torch.from_numpy(audio_array).float()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)  # (1, T)
        elif audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)  # mono

        audio_inputs = self.processor(audio.squeeze().numpy(), sampling_rate=sampling_rate or self.sampling_rate, return_tensors="pt")
        audio_values = audio_inputs.input_features.squeeze(0)
        
        text_inputs = self.tokenizer(text, return_tensors="pt", padding=False, truncation=True)
        input_ids = text_inputs.input_ids.squeeze(0)
        labels = input_ids.clone()
        
        return {
            "audio_values": audio_values,
            "input_ids": input_ids,
            "labels": labels,
            "continuation": continuation,
        }

@dataclasses.dataclass
class DataCollator:
    processor: transformers.AutoProcessor
    tokenizer: transformers.PreTrainedTokenizer
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        audio_values = [f["audio_values"] for f in features]
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]
        continuations = [f.get("continuation", "") for f in features]

        if audio_values[0].shape[-1] == 3000:
             audio_batch = torch.stack(audio_values)
        else:
             audio_values_T = [a.T for a in audio_values]
             audio_batch_T = torch.nn.utils.rnn.pad_sequence(audio_values_T, batch_first=True)
             audio_batch = audio_batch_T.transpose(1, 2)

        
        input_ids_batch = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels_batch = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        
        return {
            "audio_values": audio_batch,
            "input_ids": input_ids_batch,
            "labels": labels_batch,
            "attention_mask": (input_ids_batch != self.tokenizer.pad_token_id).long(),
            "continuation": continuations,
        }
