
import torch
import torch.nn as nn
import transformers
from typing import Optional, Tuple, Union, List
from config import ModelConfig

class ModelProjector(nn.Module):
    def __init__(self, config: ModelConfig, audio_hidden_size: int):
        super().__init__()
        self.stack_factor = config.stack_factor
        input_dim = audio_hidden_size * self.stack_factor
        
        self.linear1 = nn.Linear(input_dim, config.hidden_size)
        self.act = nn.GELU() if config.projector_act == 'gelu' else nn.ReLU()
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size)
    
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        if audio_features.dim() == 3 and audio_features.shape[1] < audio_features.shape[2]:
             audio_features = audio_features.transpose(1, 2)
        
        B, T, C = audio_features.shape
        
        if T % self.stack_factor != 0:
            pad_len = self.stack_factor - (T % self.stack_factor)
            audio_features = torch.nn.functional.pad(audio_features, (0, 0, 0, pad_len))
            T = T + pad_len
            
        audio_features = audio_features.view(B, T // self.stack_factor, C * self.stack_factor)
        
        x = self.linear1(audio_features)
        x = self.act(x)
        x = self.linear2(x)
        x = self.norm(x)
        return x

class MultiModalModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.audio_encoder = transformers.AutoModel.from_pretrained(config.audio_model_id).encoder
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
            
        audio_hidden_size = self.audio_encoder.config.hidden_size
        
        self.llm = transformers.AutoModelForCausalLM.from_pretrained(config.text_model_id, trust_remote_code=True)
        self.llm_hidden_size = self.llm.config.hidden_size
        
        self.projector = ModelProjector(config, audio_hidden_size)
        if config.hidden_size != self.llm_hidden_size:
             self.projector.linear2 = nn.Linear(config.hidden_size, self.llm_hidden_size)
             self.projector.norm = nn.LayerNorm(self.llm_hidden_size)

        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        audio_values: Optional[torch.Tensor] = None, 
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        if audio_values is not None:
             audio_outputs = self.audio_encoder(audio_values)
             audio_features = audio_outputs.last_hidden_state
             
             audio_projected = self.projector(audio_features)
             
             inputs_embeds = torch.cat([audio_projected, inputs_embeds], dim=1)
             
             if labels is not None:
                  audio_labels = torch.full((audio_projected.shape[0], audio_projected.shape[1]), -100, device=labels.device, dtype=labels.dtype)
                  labels = torch.cat([audio_labels, labels], dim=1)
                  
             if "attention_mask" in kwargs:
                  audio_mask = torch.ones((audio_projected.shape[0], audio_projected.shape[1]), device=inputs_embeds.device, dtype=kwargs["attention_mask"].dtype)
                  kwargs["attention_mask"] = torch.cat([audio_mask, kwargs["attention_mask"]], dim=1)

        # Match LLM dtype (e.g. bfloat16) to avoid "float != bfloat16" in linear layers
        llm_dtype = next(self.llm.parameters()).dtype
        inputs_embeds = inputs_embeds.to(llm_dtype)
        if labels is not None:
            labels = labels.to(llm_dtype) if labels.dtype.is_floating_point else labels

        # Drop non-tensor keys (e.g. continuation) so LLM forward doesn't receive them
        kwargs = {k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)}
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            labels=labels,
            **kwargs
        )
        
        return outputs
    
    def generate(self, input_ids, audio_values=None, **kwargs):
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        if audio_values is not None:
             audio_outputs = self.audio_encoder(audio_values)
             audio_features = audio_outputs.last_hidden_state
             audio_projected = self.projector(audio_features)
             inputs_embeds = torch.cat([audio_projected, inputs_embeds], dim=1)
             
             if "attention_mask" in kwargs:
                  audio_mask = torch.ones((audio_projected.shape[0], audio_projected.shape[1]), device=inputs_embeds.device, dtype=kwargs["attention_mask"].dtype)
                  kwargs["attention_mask"] = torch.cat([audio_mask, kwargs["attention_mask"]], dim=1)
             inputs_embeds = inputs_embeds.to(next(self.llm.parameters()).dtype)

        return self.llm.generate(inputs_embeds=inputs_embeds, **kwargs)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
