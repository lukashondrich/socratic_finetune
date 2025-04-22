#!/usr/bin/env python3
"""
TinyLlama LoRA fine‑tuning with Guanaco format
Adapts your data to the format TinyLlama expects
"""

import os
import argparse
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TextStreamer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel

# define script & project roots
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer'])
parser.add_argument('--data_path', type=str, default=str(ROOT / 'data' / 'tutor_v1_sft.jsonl'))
parser.add_argument('--output_dir', type=str, default=str(ROOT / 'tinyllama-socratic-guanaco'))
parser.add_argument('--base_model', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
parser.add_argument('--max_length', type=int, default=512)
args = parser.parse_args()

# Device setup
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# ------------------------------------------------------------------ Training Mode
def train():
    """Fine‑tune TinyLlama with LoRA using Guanaco format"""
    print(f"Loading base model: {args.base_model}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False, legacy=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    MAX_LENGTH = args.max_length

    def format_guanaco_and_tokenize(example):
        system = example["system"]
        user_query = example["user"]
        assistant_response = example["assistant"]

        # Guanaco format:
        txt = (
            "<s>[INST] <<SYS>> "
            f"{system}"
            " <</SYS>> "
            f"{user_query}"
            " [/INST] "
            f"{assistant_response}"
            " </s>"
        )
        enc = tokenizer(txt, truncation=True, max_length=MAX_LENGTH)
        input_ids = enc.input_ids
        attention_mask = enc.attention_mask

        # **Manual pad / truncate** to MAX_LENGTH
        if len(input_ids) < MAX_LENGTH:
            pad_len = MAX_LENGTH - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
        else:
            input_ids = input_ids[:MAX_LENGTH]
            attention_mask = attention_mask[:MAX_LENGTH]

        # locate end of user part
        inst_token_ids = tokenizer.encode(" [/INST]", add_special_tokens=False)
        inst_end = None
        for i in range(len(input_ids) - len(inst_token_ids) + 1):
            if input_ids[i : i + len(inst_token_ids)] == inst_token_ids:
                inst_end = i + len(inst_token_ids)
                break

        # build labels
        if inst_end is not None:
            labels = [-100] * inst_end + input_ids[inst_end:]
            if len(labels) < MAX_LENGTH:
                labels = labels + [-100] * (MAX_LENGTH - len(labels))
            else:
                labels = labels[:MAX_LENGTH]
        else:
            labels = [-100] * MAX_LENGTH

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # Load & tokenize dataset
    print(f"Loading dataset from {args.data_path}")
    ds = load_dataset("json", data_files=args.data_path, split="train")
    tokenized_ds = ds.map(format_guanaco_and_tokenize, remove_columns=ds.column_names)

    # Check for real <unk> in labels
    unk_id = tokenizer.unk_token_id
    def has_unk(ex):
        return unk_id in [i for i in ex["labels"] if i != -100]
    unk_examples = tokenized_ds.filter(has_unk)
    print(f"Examples with <unk> in labels: {len(unk_examples)}")

    # Show a few examples
    print("\nSample training examples:")
    for i in range(min(3, len(tokenized_ds))):
        full = tokenizer.decode([i for i in tokenized_ds[i]["input_ids"] if i != tokenizer.pad_token_id])
        lab  = tokenizer.decode([i for i in tokenized_ds[i]["labels"] if i != -100 and i != tokenizer.pad_token_id])
        print(f"\nExample {i+1}:")
        print(f"  Full text: {full[:100]}...")
        print(f"  Label    : {lab[:100]}...")
        print(f"  Input len: {len(tokenized_ds[i]['input_ids'])}")
        print(f"  Label len: {len(tokenized_ds[i]['labels'])}")
        print("-" * 50)

    # Load model & apply LoRA
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.float32, device_map=device
    )
    print("Applying LoRA configuration...")
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        fp16=False,
        remove_unused_columns=False,
        group_by_length=False,
        lr_scheduler_type="cosine",
        seed=42,
        max_grad_norm=0.3,
    )

    # Log a sample
    ex = next(iter(tokenized_ds))
    print("Model dtype:", model.dtype)
    print("Sample ids:", ex["input_ids"][:5], "dtype:", type(ex["input_ids"]))

    # Train
    print("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )
    trainer.train()

    # Save info + model
    os.makedirs(args.output_dir, exist_ok=True)
    with open(Path(args.output_dir) / "base_model_info.txt", "w") as f:
        f.write(f"base_model: {args.base_model}\n")
    print(f"Saving adapter + tokenizer to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete!")

# ------------------------------------------------------------------ Inference Mode
def infer():
    """Run interactive inference with the fine‑tuned Guanaco model."""
    print("Loading tokenizer & model...")
    base_model_path = args.base_model
    info_file = Path(args.output_dir) / "base_model_info.txt"
    if info_file.exists():
        with open(info_file) as f:
            for line in f:
                if line.startswith("base_model:"):
                    base_model_path = line.split(":", 1)[1].strip()
                    break

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False, legacy=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, device_map=device
    )
    model = PeftModel.from_pretrained(base_model, args.output_dir, is_trainable=False)

    streamer = TextStreamer(tokenizer)
    print("\nModel loaded! Type 'exit' to quit.")

    while True:
        query = input("\nYou: ")
        if query.strip().lower() == "exit":
            break
        prompt = f"<s>[INST] <<SYS>> You are SocraticTutor. <</SYS>> {query} [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                top_k=40,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer,
            )

# ------------------------------------------------------------------ Main
if __name__ == "__main__":
    if args.mode == "train":
        train()
    else:
        infer()
