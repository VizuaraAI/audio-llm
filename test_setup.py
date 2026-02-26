
import sys
import os
import torch

# Ensure we can import the module
sys.path.append(os.getcwd())

from config import ModelConfig
from model import MultiModalModel

def test_instantiation():
    print("Testing model instantiation...")
    config = ModelConfig()
    # Use CPU to avoid CUDA errors if not available, and smaller text model if possible for speed
    # But for verification of code we can just use what config says.
    # Note: Sarvam-2B might be large to download on the fly. 
    # Can we mock? Or just check if classes initialize.
    # If the user doesn't have the weights, this will fail.
    # To be safe/fast, let's try to mock or assume user has internet.
    # Ideally we should use a tiny model for test.
    
    config.text_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # Smaller model for quick test
    config.audio_model_id = "openai/whisper-tiny"
    
    try:
        model = MultiModalModel(config)
        print("Model instantiated successfully.")
    except Exception as e:
        print(f"Failed to instantiate model: {e}")
        return False
        
    print("Testing forward pass with dummy data...")
    try:
        # Dummy inputs
        vocab_size = model.llm.config.vocab_size
        input_ids = torch.randint(0, vocab_size, (1, 10))
        audio_values = torch.randn(1, 80, 3000) # [B, C, T] ? Whisper expects different sometimes but let's try
        
        # Whisper tiny: [B, 80, T]
        
        outputs = model(input_ids=input_ids, audio_values=audio_values)
        print("Forward pass successful.")
        print("Output shape:", outputs.logits.shape)
    except Exception as e:
        print(f"Forward pass failed: {e}")
        return False
        
    return True

if __name__ == "__main__":
    if test_instantiation():
        print("Verification Passed")
    else:
        print("Verification Failed")
