import json, random, pathlib
PRINCIPLES = [
    "Ask guiding questions first.",
    "Do not reveal the full solution until the student has tried.",
    "Be concise and encouraging."
]
SEED_QA = [
    ("What's the derivative of sin(x)?", 
     ["What do you know about how sine changes?", 
      "Great! How might that relate to cosine?"]),
    ("Explain photosynthesis.", 
     ["Let's break it down: what two main ingredients do plants need?", 
      "Right. Given light and CO₂, what compound do they produce?"])
]

out = []
for q, hints in SEED_QA:
    # good reply is first hint
    out.append({"system":"You are SocraticTutor.", "user":q, "assistant":hints[0]})

# synth‑expand with simple permutations
topics = ["capital of France", "Ohm's law", "Newton's 2nd law", "binary search",
          "Why is the sky blue?", "DNA replication", "Pythagorean theorem"]
for t in topics:
    q = f"Can you explain {t}?"
    a = f"What comes to mind when you think about {t.split()[0]}?"
    out.append({"system":"You are SocraticTutor.", "user":q, "assistant":a})

random.shuffle(out)
pathlib.Path("data").mkdir(exist_ok=True)
with open("data/tutor_v1_sft.jsonl","w") as f:
    for row in out:
        f.write(json.dumps(row)+"\n")
print(len(out),"examples saved.")
