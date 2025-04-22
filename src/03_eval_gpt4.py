#!/usr/bin/env python3
"""
Evaluate base vs. fine‑tuned TinyLlama on unseen topics using GPT‑4o scoring.
Produces CSVs, a bar plot (eval_plot.png), and detailed topic comparisons.
"""
import argparse
import json
import logging
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from openai import OpenAI
from peft import PeftModel, PeftConfig
from scipy.stats import ttest_rel
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------------------------------------------------------------
# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate base vs. tuned TinyLlama models")
    p.add_argument(
        "--base_model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace ID or path of the base (untuned) model",
    )
    p.add_argument(
        "--tuned_dir",
        type=Path,
        default=Path("tinyllama-socratic-guanaco"),
        help="Directory containing the fine‑tuned LoRA adapter",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("."),
        help="Where to write eval_scores.csv, eval_plot.png, etc.",
    )
    return p.parse_args()

# -----------------------------------------------------------------------------
def load_models(base_model_id: str, tuned_dir: Path, device: str):
    # Base tokenizer & model
    logger.info(f"Loading base model/tokenizer '{base_model_id}'")
    tok_base = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    m_base = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=torch.float16, device_map=device
    )

    # Determine actual base for the adapter
    try:
        peft_conf = PeftConfig.from_pretrained(tuned_dir)
        adapter_base = peft_conf.base_model_name_or_path
        logger.info(f"Adapter built on '{adapter_base}'")
    except Exception:
        adapter_base = base_model_id

    # Load base for tuned
    logger.info(f"Loading base model for adapter: '{adapter_base}'")
    base_for_tuned = AutoModelForCausalLM.from_pretrained(
        adapter_base, torch_dtype=torch.float16, device_map=device
    )
    # Load and attach LoRA weights
    m_tuned = PeftModel.from_pretrained(
        base_for_tuned, tuned_dir, is_trainable=False
    )
    return tok_base, m_base, tok_base, m_tuned

# -----------------------------------------------------------------------------
def ask(model, tokenizer, question: str, tuned: bool, device: str):
    if tuned:
        # Guanaco-style prompt
        prompt = f"<s>[INST] <<SYS>> You are SocraticTutor. <</SYS>> {question} [/INST]"
    else:
        prompt = f"You are SocraticTutor.\nUser: {question}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=1.0,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if tuned:
        m = re.search(r"\[\/INST\](.*)", text, re.DOTALL)
        return m.group(1).strip() if m else text.strip()
    else:
        return text.split("Assistant:")[-1].strip()

# -----------------------------------------------------------------------------
def score_with_openai(client, prompt: str, reply: str):
    rubric = (
        "You are an evaluator.\n"
        "Rate the assistant's reply on two scales 0-1:\n"
        "A) Socratic-ness (guiding questions, no direct answer),\n"
        "B) Helpfulness (clarity, encouragement, correctness).\n"
        "Reply as JSON: {\"socratic\":<float>,\"helpful\":<float>} only."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": rubric},
                {"role": "user", "content": f"Question: {prompt}\nAssistant reply: {reply}"},
            ],
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        logger.error(f"OpenAI scoring error: {e}")
        return {"socratic": 0.5, "helpful": 0.5}

# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    args.output_dir.mkdir(exist_ok=True)

    # Load env and OpenAI client
    load_dotenv()
    client = OpenAI()

    # Device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load models
    tok_base, m_base, tok_tuned, m_tuned = load_models(
        args.base_model, args.tuned_dir, device
    )

    # Define fresh topics
    TOPICS = [
        "Faraday's law", "gene expression", "principle of least action", "hash collisions",
        "inflationary cosmology", "Hamming distance", "latent heat of fusion",
        "Newton's cradle", "Garbage Collection in Java", "K‑means clustering",
        "photosynthetic Calvin cycle", "Shor's algorithm", "Lyapunov stability",
        "net present value", "Kaplan‑Meier curves", "HTTP 3‑way handshake",
        "Lorenz attractor", "PID control", "Poisson distribution", "CRISPR‑Cas immunity"
    ]

    rows = []
    for idx, topic in enumerate(TOPICS, 1):
        logger.info(f"Evaluating topic {idx}/{len(TOPICS)}: {topic}")
        q = f"Explain {topic}."

        # Base
        base_reply = ask(m_base, tok_base, q, tuned=False, device=device)
        # Tuned
        tuned_reply = ask(m_tuned, tok_tuned, q, tuned=True, device=device)

        # Score
        base_score = score_with_openai(client, q, base_reply)
        time.sleep(1)
        tuned_score = score_with_openai(client, q, tuned_reply)
        time.sleep(1)

        rows.append({"model": "base", "topic": topic, **base_score})
        rows.append({"model": "tuned", "topic": topic, **tuned_score})

    df = pd.DataFrame(rows)
    csv_path = args.output_dir / "eval_scores.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved scores to {csv_path}")

    # Paired t-test
    logger.info("Running paired t-tests")
    for metric in ["socratic", "helpful"]:
        a = df[df.model == "base"][metric].values
        b = df[df.model == "tuned"][metric].values
        t_stat, p_val = ttest_rel(a, b)
        logger.info(f"{metric}: base={a.mean():.2f}, tuned={b.mean():.2f}, p={p_val:.4f}")

    # Plot
    means = df.groupby("model")[["socratic", "helpful"]].mean()
    sems = df.groupby("model")[["socratic", "helpful"]].sem()

    ax = means.plot(
        kind="bar", y=["socratic", "helpful"], yerr=sems, capsize=4, figsize=(8, 6)
    )
    ax.set_ylabel("Score (0–1)")
    ax.set_ylim(0, 1)
    ax.set_title("Socratic-ness & Helpfulness: Base vs. Fine-tuned")
    plot_path = args.output_dir / "eval_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    logger.info(f"Saved plot to {plot_path}")

    # Detailed pivot
    pivot = df.pivot_table(index="topic", columns="model", values=["socratic", "helpful"])
    pivot_path = args.output_dir / "topic_comparison.csv"
    pivot.to_csv(pivot_path)
    logger.info(f"Saved detailed comparison to {pivot_path}")

if __name__ == "__main__":
    main()
