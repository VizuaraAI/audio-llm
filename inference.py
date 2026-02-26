
import torch
import torchaudio
import transformers
from config import ModelConfig
from model import MultiModalModel

def run_inference(audio_path: str, model_path: str = None):
    # Load Config & Model
    config = ModelConfig()
    
    
    model = MultiModalModel(config)
    
    if model_path:
        state_dict = torch.load(f"{model_path}/pytorch_model.bin", map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    
    # Process Audio
    processor = transformers.AutoProcessor.from_pretrained(config.audio_model_id)
    audio, sr = torchaudio.load(audio_path)
    if sr != 16000:
        audio = torchaudio.functional.resample(audio, sr, 16000)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
        
    audio_inputs = processor(audio.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    audio_values = audio_inputs.input_features
    
    # Create Input Text
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.text_model_id)
    text = "Transcribe the following audio:"
    text_inputs = tokenizer(text, return_tensors="pt")
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=text_inputs.input_ids,
            audio_values=audio_values,
            max_new_tokens=200
        )
    
    transcription = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("Transcription:", transcription)
    return transcription

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_inference(sys.argv[1])
    else:
        print("Usage: python -m audio_lm.inference path/to/audio.wav")
