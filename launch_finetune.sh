#!/bin/bash

# Install requirements if needed
# pip install -r requirements.txt
# pip install peft

# Run Finetuning
# Ensure you are in the correct directory
# python finetune.py

# To run with specific GPUs
# CUDA_VISIBLE_DEVICES=0 python finetune.py

# Accelerated launch (Multi-GPU)
# accelerate launch finetune.py

python finetune.py
