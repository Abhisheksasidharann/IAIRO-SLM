import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from olm.models.meta.llama3 import Llama3Model


ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "model.pt" if (ROOT_DIR / "model.pt").exists() else Path("./model.pt")
TOKENIZER_PATH = ROOT_DIR / "tokenizer" if (ROOT_DIR / "tokenizer").exists() else Path("./tokenizer")
DATA_PATH = ROOT_DIR / "results" / "eval_data.jsonl" if (ROOT_DIR / "results" / "eval_data.jsonl").exists() else Path("./results/eval_data.jsonl")
OUTPUT_PATH = ROOT_DIR / "results" / "perplexity.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 1024


def load_tokens():
    tokens = []

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            tokens.extend(record["input_ids"])

    return tokens


def main():
    print("Device:", DEVICE)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    print("Tokenizer length:", len(tokenizer))
    print("EOS token:", repr(tokenizer.eos_token))
    print("EOS ID:", tokenizer.eos_token_id)

    print("\nLoading model...")
    model = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model.eval()

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print("Model loaded")
    print("Parameters:", f"{parameter_count:,}")

    print("\nLoading evaluation tokens...")
    tokens = load_tokens()

    print("Total tokens loaded:", f"{len(tokens):,}")

    if len(tokens) < 2:
        raise ValueError("Not enough tokens for evaluation.")

    total_loss = 0.0
    total_tokens = 0

    start_time = time.time()

    print("\nRunning evaluation...")

    with torch.no_grad():

        for start in range(
            0,
            len(tokens) - 1,
            MAX_LENGTH,
        ):

            chunk = tokens[
                start : start + MAX_LENGTH + 1
            ]

            if len(chunk) < 2:
                continue

            input_ids = torch.tensor(
                [chunk[:-1]],
                dtype=torch.long,
                device=DEVICE,
            )

            target_ids = torch.tensor(
                [chunk[1:]],
                dtype=torch.long,
                device=DEVICE,
            )

            logits = model(input_ids)

            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                reduction="sum",
            )

            total_loss += loss.item()
            total_tokens += target_ids.numel()

            if total_tokens // 500_000 > (
                (total_tokens - target_ids.numel()) // 500_000
            ):
                print(
                    f"Evaluated {total_tokens:,} tokens"
                )

    elapsed = time.time() - start_time

    mean_loss = total_loss / total_tokens
    perplexity = math.exp(mean_loss)

    result = {
        "model": "252M biomedical SLM",
        "model_parameters": parameter_count,
        "tokenizer_length": len(tokenizer),
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "dataset": "almanach/Biomed-Enriched",
        "dataset_config": "default",
        "dataset_split": "commercial",
        "evaluation_type": "independent 5M-token evaluation",
        "tokens_evaluated": total_tokens,
        "loss": mean_loss,
        "perplexity": perplexity,
        "device": DEVICE,
        "runtime_seconds": elapsed,
    }

    Path(OUTPUT_PATH).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print("\n=== Evaluation complete ===")
    print("Tokens:", f"{total_tokens:,}")
    print("Loss:", mean_loss)
    print("Perplexity:", perplexity)
    print("Runtime:", elapsed, "seconds")
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()