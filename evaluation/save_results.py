from pathlib import Path
import torch
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path("model.pt") if Path("model.pt").exists() else ROOT_DIR / "model.pt"
OUTPUT_PATH = Path(__file__).resolve().parent / "smoke_test_results.txt"

model = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
model.eval()
tok = AutoTokenizer.from_pretrained("stanford-crfm/BioMedLM", use_fast=True)
EOS = tok.eos_token_id

def generate(prompt, n=60):
    ids = torch.tensor([tok.encode(prompt, add_special_tokens=False)])
    for _ in range(n):
        with torch.no_grad():
            logits = model(ids)
        nxt = logits[0, -1].argmax().view(1, 1)
        if nxt.item() == EOS:
            break
        ids = torch.cat([ids, nxt], dim=1)
    return tok.decode(ids[0])

def generate_sample(prompt, n=60, temperature=0.8, top_k=50):
    ids = torch.tensor([tok.encode(prompt, add_special_tokens=False)])
    for _ in range(n):
        with torch.no_grad():
            logits = model(ids)
        logits = logits[0, -1] / temperature
        top_vals, top_idx = torch.topk(logits, top_k)
        probs = torch.softmax(top_vals, dim=-1)
        choice = torch.multinomial(probs, 1)
        nxt = top_idx[choice].view(1, 1)
        if nxt.item() == EOS:
            break
        ids = torch.cat([ids, nxt], dim=1)
    return tok.decode(ids[0])

prompts = [
    "Metformin is a first-line treatment for",
    "The mechanism of action of warfarin involves",
    "Patients with chronic kidney disease should avoid",
    "In this randomized controlled trial, participants were",
    "Common adverse reactions of ibuprofen include",
    "The recommended dose of acetaminophen for adults is",
    "Hypertension is diagnosed when blood pressure",
    "Insulin resistance is associated with",
    "Contraindications to beta blockers include",
]

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write("First test (greedy decoding, looped):\n")
    f.write(generate("Acute myocardial infarction is characterized by"))
    f.write("\n\n" + "="*60 + "\n\n")
    f.write("Sampled outputs:\n\n")
    for p in prompts:
        f.write("PROMPT: " + p + "\n")
        f.write("OUTPUT: " + generate_sample(p) + "\n")
        f.write("-"*60 + "\n")

print(f"Saved to {OUTPUT_PATH}")