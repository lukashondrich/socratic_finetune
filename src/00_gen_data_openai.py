import os, json, pathlib, random, time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()                     # reads .env
client = OpenAI()                 # key auto‑picked from env

N = 500                           # target examples
PRINCIPLES = [
    "Ask guiding questions first.",
    "Do not reveal the full solution until the student has tried.",
    "Be concise and encouraging."
]
SYSTEM_MSG = (
    "You are DataGenBot, producing JSON for SocraticTutor finetuning.\n"
    f"The tutoring PRINCIPLES are:\n- " + "\n- ".join(PRINCIPLES) + "\n\n"
    "Return ONLY valid JSON objects, one per assistant message, "
    'with keys: "user", "assistant".'
)
TOPICS = [
    "Ohm's law", "DNA replication", "Pythagorean theorem", "capital of Japan",
    "photosynthesis", "binary search", "Mendelian inheritance", "thermodynamic entropy",
    "Newton's second law", "Heisenberg uncertainty principle", "Bayes' theorem",
    "supply‑and‑demand curve", "mitosis vs. meiosis", "Fourier transform",
    "black‑hole event horizon", "Shannon information", "RNA translation",
    "Hubble's law", "SQL JOINs", "Big‑O notation", "Euler's formula (e^{iπ}+1=0)",
    "chemical bonding types", "prime‑factorization", "Boolean logic gates",
    "blockchain hashing", "solar eclipse", "continental drift theory",
    "standard deviation", "diffusion osmosis", "CAP theorem", "photosynthetic light reactions",
    "Gaussian distribution", "CRISPR gene editing", "UDP vs. TCP", "quantum superposition",
    "plate tectonics", "acid–base titration", "graph breadth‑first search",
    "momentum conservation", "genetic algorithms", "Doppler effect", "monopolistic competition",
    "Kepler’s laws", "cytokine storm", "hash tables", "hydrogen bonding",
    "Pascal’s triangle", "game‑theory Nash equilibrium", "neural back‑propagation"
]
seed_prompts = [
    "What's the derivative of sin(x)?",
    "Explain photosynthesis.",
    "How does binary search work?",
    "Why is the sky blue?",
]

def call_gpt(user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=256,
        messages=[
            {"role":"system","content":SYSTEM_MSG},
            {"role":"user","content":user_prompt}
        ]
    )
    return resp.choices[0].message.content.strip()

out = []
for i in range(N):
    # choose or invent a question
    q = random.choice(seed_prompts) if i < len(seed_prompts) else \
        f"Explain {random.choice(TOPICS)}."
    reply_json = call_gpt(f'Compose a Socratic reply to: "{q}"')
    try:
        pair = json.loads(reply_json)
        out.append({"system":"You are SocraticTutor.", **pair})
    except Exception as e:
        print("JSON error, skipping:", e)

    # throttle to stay under 60 RPM
    if i % 20 == 0:
        print(f"{i}/{N} done …")
        time.sleep(10)

pathlib.Path("data").mkdir(exist_ok=True)
with open("data/tutor_v1_sft.jsonl","w") as f:
    for row in out:
        f.write(json.dumps(row, ensure_ascii=False)+"\n")

print(f"Saved {len(out)} examples to data/tutor_v1_sft.jsonl")