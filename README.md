
# Audio LM Implementation

<img width="822" height="864" alt="Joint_embedding_model_Sarvam_with_Whisper" src="https://github.com/user-attachments/assets/b780509b-72cc-4503-8be6-7f1e8254539a" />


A modular, simplified implementation of a multimodal LLM inspired by Audio Language Modeling. This codebase fuses an Audio Encoder (Whisper) with an LLM (Sarvam-M) using a projector.

[Huggingface](https://huggingface.co/teamvizuara/Vocal-LLM)

## Directory Structure

- `config.py`: Configuration for Model and Training.
- `data.py`: Dataset and DataCollator implementation.
- `model.py`: Core model architecture (AudioEncoder + Projector + LLM).
- `train.py`: Training script using Hugging Face Trainer.
- `inference.py`: Inference script for testing the model.

## Setup

1. **System: Install FFmpeg** (required for audio decoding with Common Voice / MP3):
   ```bash
   # Ubuntu/Debian
   sudo apt-get update && sudo apt-get install -y ffmpeg
   ```
   Without FFmpeg, loading audio will fail with `Could not load libtorchcodec` / `libavutil.so: cannot open shared object file`.

2. **Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Or: `pip install torch torchaudio transformers peft accelerate datasets`

3. **Dataset:**
   The project is configured to use Hugging Face datasets (default: `fixie-ai/common_voice_17_0`).
   You can change the dataset settings in `config.py`.

4. **Environment Configuration (.env):**
   Create a `.env` file in the root directory to configure Weights & Biases (and optionally Hugging Face).
   
   ```bash
   touch .env
   ```
   
   Add the following variables to `.env`:
   
   ```ini
   # Required for WandB tracking
   WANDB_API_KEY=your_wandb_api_key_here
   WANDB_PROJECT=audio-language-model
   WANDB_ENTITY=your_username_or_team_name  # Optional
   
   # Optional: Hugging Face Token (if pushing to Hub)
   HF_TOKEN=your_hf_token_here
   ```

## Usage

### Training

1. Configure `config.py` if needed (batch size, learning rate, paths).
2. Run training:
   ```bash
   python train.py
   ```

### Inference

Run inference on an audio file:
```bash
python inference.py /path/to/audio.wav
```

### Fine-tuning

To fine-tune the model (e.g., for instruction following or specific domains) using the pre-trained projector:

1.  **Configuration**: 
    -   Check `finetune_config.py` for settings.
    -   Key parameters: `dataset_names`, `load_from_checkpoint`, `lora_r`.

2.  **Run Fine-tuning**:
    ```bash
    # Uses finetune.py, finetune_config.py, and finetune_data.py
    bash launch_finetune.sh
    # OR
    python finetune.py
    ```

    **Note**: The fine-tuning script:
    -   Automatically downloads the pre-trained `model.safetensors` from Hugging Face (`Mayank022/Audio-Language-Model`).
    -   Freezes the Audio Encoder.
    -   Loads the Projector weights.
    -   Applies **LoRA** to the LLM (Sarvam-M) for efficient training.
    -   Handles mixed datasets:
        -   **With Instructions**: Uses the `instruction_prompt` column.
        -   **Without Instructions**: Randomly samples default prompts (e.g., "Transcribe this audio").

## Model Architecture

The `AudioLM` consists of:
- **Audio Encoder**: `openai/whisper-small` (frozen).
- **Projector**: A 2-layer MLP (`Linear` -> `GELU` -> `Linear`) maps audio features to LLM embedding space.
- **LLM**: `sarvamai/sarvam-2b-v0.5` with LoRA (fine-tuned).


## Dataset Guide

To train a model capable of real-time voice interaction (especially for Indic languages using Sarvam), you need a mix of datasets served in two main stages:

### Stage 1: Alignment (ASR & Continuation)
The goal is to teach the projector to map audio to the LLM's embedding space.
*   **Data Type**: `<Audio> -> <Transcription>`
*   **Recommended Datasets**:
    *   **Indic/Multilingual**: [CommonVoice 17.0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) (Hindi, Tamil, etc.), [IndicVoices](https://huggingface.co/datasets/AI4Bharat/IndicVoices), [Kathbath](https://huggingface.co/datasets/AI4Bharat/Kathbath).
    *   **English**: [LibriSpeech](https://huggingface.co/datasets/librispeech_asr), [GigaSpeech](https://huggingface.co/datasets/speechcolab/gigaspeech).
*   **Task**: The model is given audio and forced to output the exact transcription.

### Stage 2: Instruction Tuning (Speech-to-Text Chat)
The goal is to teach the model to *understand* speech and *respond* intelligently.
*   **Data Type**: `<Audio Instruction> -> <Text Response>`
*   **Strategy**:
    1.  **Synthetic**: Take a text-only instruction dataset (e.g., Alpaca, OpenAssist). Use a TTS (Text-to-Speech) model to generate audio for the "User Instruction". Train the model to output the "Assistant Response".
    2.  **Continuation**: Split a spoken sentence in half. Feed the first half as Audio. Train LLM to complete the text of the second half.
    3.  **Cross-Modal**: Use mixed datasets like [CoVoST 2](https://huggingface.co/datasets/covost2) (Speech Translation).

### Formatting for this Codebase
The codebase uses `datasets.load_dataset`. See `data.py` for details on how different datasets are mapped to `audio` and `text` fields.
