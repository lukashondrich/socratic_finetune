#!/usr/bin/env python3
"""
Chat demo for fine‑tuned TinyLlama with LoRA adapters.
"""

import argparse
import logging
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from peft import PeftModel, PeftConfig

# -----------------------------------------------------------------------------
# Config & Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Interactive chat with a LoRA‑fine‑tuned TinyLlama model"
    )
    p.add_argument(
        "--model_dir",
        type=Path,
        default=Path("tinyllama-socratic-guanaco"),
        help="Path to the directory containing the fine‑tuned model and tokenizer",
    )
    p.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Optional: HuggingFace ID or path of the base model (if not stored in PEFT config)",
    )
    return p.parse_args()

# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    # Detect device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Resolve base model
    try:
        peft_conf = PeftConfig.from_pretrained(args.model_dir)
        base_model_name = peft_conf.base_model_name_or_path
        logger.info(f"Detected base model from PEFT config: {base_model_name}")
    except Exception:
        base_model_name = args.base_model or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        logger.info(f"Using provided/default base model: {base_model_name}")

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    logger.info(f"Loading base model '{base_model_name}'")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map=device
    )

    # Load and merge LoRA adapter
    logger.info(f"Applying LoRA weights from '{args.model_dir}'")
    lora_model = PeftModel.from_pretrained(
        base_model,
        args.model_dir,
        is_trainable=False
    )
    model = lora_model.merge_and_unload()
    model.to(device)
    model.eval()

    # Set up streaming
    streamer = TextStreamer(tokenizer)

    logger.info("Model loaded. Type your questions (or 'exit' to quit).")
    while True:
        q = input("\nYou: ")
        if q.strip().lower() == "exit":
            break

        prompt = f"You are SocraticTutor.\nUser: {q}\nAssistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Generate with sampling
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                temperature=0.7,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer,
            )

if __name__ == "__main__":
    main()
