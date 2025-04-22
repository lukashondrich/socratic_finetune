#!/usr/bin/env python3
"""
00_gen_data_openai.py
Generate Socratic tutor dataset via OpenAI API.
"""
import os
import json
import random
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# -- Setup logging
logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", level=logging.WARNING)
logger = logging.getLogger(__name__)

# -- Paths
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
ENV_PATH = ROOT / ".env"

# -- Load environment
load_dotenv(dotenv_path=ENV_PATH)
client = OpenAI()

# -- Configuration
N = 50  # target examples
PRINCIPLES = [
    "Ask guiding questions first.",
    "Do not reveal the full solution until the student has tried.",
    "Be concise and encouraging."
]
SYSTEM_MSG = (
    "You are DataGenBot generating JSON for SocraticTutor finetuning.\n"
    "The tutoring PRINCIPLES are:\n- " + "\n- ".join(PRINCIPLES) + "\n\n"
    "Return only valid JSON objects, one per assistant message, "
    "with keys: \"user\", \"assistant\"."
)
TOPICS = [
    "Ohm's law", "DNA replication", "Pythagorean theorem", "capital of Japan",
    # ... (list of 50 topics) ...
    "game-theory Nash equilibrium", "neural back-propagation"
]
SEED_PROMPTS = [
    "What's the derivative of sin(x)?",
    "Explain photosynthesis.",
    "How does binary search work?",
    "Why is the sky blue?",
]

# -- Functions
def call_gpt(user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=256,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


# -- Main
def main():
    out = []
    for i in range(N):
        # choose or invent a question
        if i < len(SEED_PROMPTS):
            q = SEED_PROMPTS[i]
        else:
            q = f"Explain {random.choice(TOPICS)}."
        reply_json = call_gpt(f'Compose a Socratic reply to: "{q}"')
        try:
            pair = json.loads(reply_json)
            out.append({"system": "You are SocraticTutor.", **pair})
        except json.JSONDecodeError as e:
            logger.warning(f"JSONDecodeError at idx {i}: {e}")

        if i % 20 == 0:
            logger.info(f"{i}/{N} done…")
            time.sleep(10)

    # write file
    outfile = DATA_DIR / "tutor_v1_sft.jsonl"
    with open(outfile, "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(out)} examples to {outfile}")

if __name__ == "__main__":
    main()
