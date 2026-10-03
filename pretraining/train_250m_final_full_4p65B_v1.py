#!/usr/bin/env python3
import gzip, json, math, os, random, time, gc
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from olm.models.meta.llama3 import Llama3Model

# ============================================================
# PATHS / VERSION
# ============================================================
ROOT = Path.home() / "dev_data" / "development_4B"
TRAIN_DIR = ROOT / "train"
VAL_DIR = ROOT / "validation"
RUN_NAME = os.environ.get("RUN_NAME", "dev250M_final_full_4p65B_v1")
DIAGNOSTIC_ONLY = os.environ.get("DIAGNOSTIC_ONLY", "1").lower() not in {"0", "false", "no"}
RUN_ROOT = ROOT / "output" / RUN_NAME
CKPT_DIR = RUN_ROOT / "checkpoints"
MODEL_DIR = RUN_ROOT / "model"
BEST_DIR = RUN_ROOT / "best_model"
for p in (CKPT_DIR, MODEL_DIR, BEST_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
SEED = 42
TOKENIZER_NAME = "stanford-crfm/BioMedLM"

# 250M pilot model — controlled scale-up from the validated 100M baseline.
# Keep context/tokenizer/optimizer/data protocol unchanged so the experiment
# isolates model-capacity effects as cleanly as possible.
N_LAYERS, D_MODEL = 26, 768
N_Q_HEADS, N_KV_HEADS = 12, 6
FFN_DIM = 3072
CONTEXT = 1024
DROPOUT = 0.0
ROPE_THETA = 10_000.0
TIE_WEIGHTS = True
# OLM's default torch Embedding initialization is std=1.0. With tied input/output
# embeddings this produces enormous logits at initialization (~20-30+) and the
# observed CE ~440. Use a standard small LM embedding initialization instead.
EMBED_INIT_STD = 0.02

# Full development run: 1.8B training tokens.
# 3434 updates × 524,288 tokens/update = 1,800,404,992 tokens (~1.8004B).
TARGET_TOKENS = 4_650_000_000
MICRO_BATCH = 32
GRAD_ACCUM = 16
TOKENS_PER_UPDATE = MICRO_BATCH * CONTEXT * GRAD_ACCUM
TOTAL_STEPS = math.ceil(TARGET_TOKENS / TOKENS_PER_UPDATE)
WARMUP_STEPS = max(1, int(TOTAL_STEPS * 0.02))

LR = 3e-4
MIN_LR = 3e-5
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
EPS = 1e-5
CLIP = 1.0

NUM_WORKERS = 8
EVAL_EVERY = 500
VAL_TOKENS = 5_000_000
SAVE_EVERY = 500
KEEP_LAST = 5
LOG_EVERY = 10

# Optional: set MAX_STEPS=N for a controlled smoke/resume test; omit for full run.
_max_steps_env = os.environ.get("MAX_STEPS", "")
MAX_STEPS = int(_max_steps_env) if _max_steps_env else None
# Set RESUME=0 to deliberately start from scratch.
RESUME = os.environ.get("RESUME", "1").lower() not in {"0", "false", "no"}

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required.")
DEVICE = torch.device("cuda")
GPU = torch.cuda.get_device_name(0)
if "H100" not in GPU.upper():
    print("WARNING: script is tuned for H100; detected:", GPU)
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("H100/BF16 support required.")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
VOCAB_SIZE = len(tokenizer)
EOS_ID = tokenizer.eos_token_id
if EOS_ID is None:
    raise RuntimeError("Tokenizer has no EOS token.")

print("=" * 80)
print("Biomedical SLM | 250M FULL DEVELOPMENT | H100 | version:", RUN_NAME)
print("=" * 80)
print("GPU:", GPU)
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("Tokenizer:", TOKENIZER_NAME, "len:", VOCAB_SIZE, "EOS:", EOS_ID)
print("Tokens/update:", f"{TOKENS_PER_UPDATE:,}")
print("Total steps:", TOTAL_STEPS, "Warmup:", WARMUP_STEPS)
print("Max steps:", MAX_STEPS if MAX_STEPS is not None else "FULL RUN")
print("Resume:", RESUME)
print("Diagnostic only:", DIAGNOSTIC_ONLY)
print("Embedding init std:", EMBED_INIT_STD)
print("Training corpus target tokens:", f"{TARGET_TOKENS:,}")
print("Approx optimizer steps:", TOTAL_STEPS)
print("Fast validation tokens:", f"{VAL_TOKENS:,}")
print("NOTE: Run full 200M-token validation separately after training.")

# ============================================================
# STREAMING PACKED DATASET
# ============================================================
class BiomedicalPackedDataset(torch.utils.data.IterableDataset):
    def __init__(self, folder, tokenizer, context_length=1024,
                 seed=42, shuffle=True, max_tokens=None,
                 start_sequence=0):
        super().__init__()
        self.folder = Path(folder)
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.seed = seed
        self.shuffle = shuffle
        self.max_tokens = max_tokens
        self.start_sequence = start_sequence

    def _shards(self):
        shards = sorted(self.folder.glob("shard_*.jsonl.gz"))
        if not shards:
            raise FileNotFoundError(f"No shard_*.jsonl.gz in {self.folder}")
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(shards)
        wi = torch.utils.data.get_worker_info()
        if wi is not None:
            shards = shards[wi.id::wi.num_workers]
        return shards

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        worker_id = wi.id if wi else 0
        nworkers = wi.num_workers if wi else 1

        buf = []
        local_seq = 0
        tokens = 0

        for shard in self._shards():
            with gzip.open(shard, "rt", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    text = rec.get("text")
                    if not text:
                        continue
                    buf.extend(self.tokenizer.encode(
                        text, add_special_tokens=False))
                    buf.append(EOS_ID)

                    while len(buf) >= self.context_length + 1:
                        chunk = buf[:self.context_length + 1]
                        del buf[:self.context_length]

                        # Global sequence number for exact resume.
                        global_seq = worker_id + local_seq * nworkers
                        local_seq += 1

                        if global_seq < self.start_sequence:
                            continue

                        x = torch.tensor(chunk[:-1], dtype=torch.long)
                        y = torch.tensor(chunk[1:], dtype=torch.long)
                        tokens += self.context_length
                        yield x, y

                        if self.max_tokens and tokens >= self.max_tokens:
                            return

# ============================================================
# MODEL
# ============================================================
model = Llama3Model(
    vocab_size=VOCAB_SIZE,
    embed_dim=D_MODEL,
    intermediate_size=FFN_DIM,
    num_layers=N_LAYERS,
    num_heads=N_Q_HEADS,
    num_kv_heads=N_KV_HEADS,
    max_seq_len=CONTEXT,
    rope_theta=ROPE_THETA,
    dropout=DROPOUT,
    tie_weights=TIE_WEIGHTS,
).to(DEVICE, dtype=torch.float32)

# ------------------------------------------------------------------
# Critical initialization fix for tied embeddings
# ------------------------------------------------------------------
# OLM's Embedding module inherits PyTorch's default Embedding reset, which
# initializes the embedding matrix at std=1.0. Because tie_weights=True, the
# same matrix is also the LM output projection. That makes the initial logits
# far too large and saturates softmax. Reinitialize the tied embedding/output
# matrix to a standard LM scale before constructing the optimizer.
embedding_params = [
    (name, p) for name, p in model.named_parameters()
    if name.endswith("embedding.weight") or name.endswith("embed_tokens.weight")
]
if len(embedding_params) != 1:
    raise RuntimeError(f"Expected exactly one tied embedding parameter; found: {[n for n, _ in embedding_params]}")
embedding_name, embedding_weight = embedding_params[0]
with torch.no_grad():
    torch.nn.init.normal_(embedding_weight, mean=0.0, std=EMBED_INIT_STD)
print(f"Embedding init: {embedding_name} ~ N(0, {EMBED_INIT_STD})")
print("Embedding post-init std:", f"{embedding_weight.detach().float().std().item():.6f}")

PARAMS = sum(p.numel() for p in model.parameters())
assert 220_000_000 < PARAMS < 280_000_000, PARAMS
assert hasattr(model, "save")
print("Architecture:", f"layers={N_LAYERS}, d_model={D_MODEL}, q_heads={N_Q_HEADS}, kv_heads={N_KV_HEADS}, ffn={FFN_DIM}, context={CONTEXT}")
print("Parameters:", f"{PARAMS:,}")
print("Master weight dtype:", next(model.parameters()).dtype)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, betas=BETAS, eps=EPS,
    weight_decay=WEIGHT_DECAY
)

def lr_at(step):
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    p = min(1.0, (step - WARMUP_STEPS) /
            max(1, TOTAL_STEPS - WARMUP_STEPS))
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * p))

def set_lr(step):
    lr = lr_at(step)
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr

# ============================================================
# CHECKPOINT / DATA RESUME
# ============================================================
STATE = CKPT_DIR / "latest.pt"

def save_ckpt(step, tokens_seen, best_val):
    payload = {
        "step": step,
        "tokens_seen": tokens_seen,
        "best_val": best_val,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    tmp = CKPT_DIR / "latest.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, STATE)
    snap = CKPT_DIR / f"step_{step:07d}.pt"
    torch.save(payload, snap)
    snaps = sorted(CKPT_DIR.glob("step_*.pt"))
    while len(snaps) > KEEP_LAST:
        snaps.pop(0).unlink()
    print(f"[checkpoint] step {step:,}")

def load_ckpt():
    if not (RESUME and STATE.exists()):
        return 0, 0, None
    c = torch.load(STATE, map_location="cpu")
    model.load_state_dict(c["model"])
    optimizer.load_state_dict(c["optimizer"])
    for s in optimizer.state.values():
        for k, v in s.items():
            if torch.is_tensor(v):
                s[k] = v.to(DEVICE)
    step = int(c["step"])
    tokens = int(c["tokens_seen"])
    print(f"[resume] step={step:,}, tokens={tokens:,}")
    return step, tokens, c.get("best_val")

# ============================================================
# PREFLIGHT + DEEP DIAGNOSTICS
# ============================================================
step0, tokens_seen, best_val = load_ckpt()
start_sequence = step0 * GRAD_ACCUM * MICRO_BATCH

train_ds = BiomedicalPackedDataset(
    TRAIN_DIR, tokenizer, CONTEXT, SEED, True,
    start_sequence=start_sequence
)
val_ds = BiomedicalPackedDataset(
    VAL_DIR, tokenizer, CONTEXT, 123, False,
    max_tokens=VAL_TOKENS
)

train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=MICRO_BATCH, num_workers=NUM_WORKERS,
    pin_memory=True, persistent_workers=NUM_WORKERS > 0,
    prefetch_factor=4 if NUM_WORKERS else None
)
val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=MICRO_BATCH, num_workers=NUM_WORKERS,
    pin_memory=True, persistent_workers=NUM_WORKERS > 0,
    prefetch_factor=4 if NUM_WORKERS else None
)

x, y = next(iter(train_loader))
print("Preflight batch:", tuple(x.shape))
print("Token range:", int(x.min()), int(x.max()))
print("Label range:", int(y.min()), int(y.max()))
assert int(x.min()) >= 0 and int(x.max()) < VOCAB_SIZE
assert int(y.min()) >= 0 and int(y.max()) < VOCAB_SIZE
assert x.shape == y.shape == (MICRO_BATCH, CONTEXT)

# Basic parameter diagnostics. These are especially important because the initial loss should remain near the expected random-model scale
# (~ln(vocab_size) = 10.27).
print("Expected uniform-vocabulary CE:", f"{math.log(VOCAB_SIZE):.4f}")
print("Parameter diagnostics:")
with torch.no_grad():
    p_abs_max = 0.0
    p_rms_acc = 0.0
    p_count = 0
    for name, p in model.named_parameters():
        pf = p.detach().float()
        p_abs_max = max(p_abs_max, float(pf.abs().max()))
        p_rms_acc += float(pf.pow(2).sum())
        p_count += pf.numel()
    print("  global param RMS:", f"{math.sqrt(p_rms_acc / p_count):.6g}")
    print("  global param abs max:", f"{p_abs_max:.6g}")
    for name, p in list(model.named_parameters())[:12]:
        pf = p.detach().float()
        print(f"  {name:55s} shape={tuple(p.shape)!s:22s} mean={pf.mean():+.3e} std={pf.std():.3e} max={pf.abs().max():.3e}")

x = x.to(DEVICE, non_blocking=True)
y = y.to(DEVICE, non_blocking=True)
model.train()
optimizer.zero_grad(set_to_none=True)

# Compare FP32 and BF16-autocast forward passes on exactly the same batch.
# If both have huge logits/loss, this is a model initialization/forward issue,
# not an H100 BF16 numerical issue.
with torch.no_grad():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits_bf16 = model(x)
    logits_bf16_f = logits_bf16.float()
    loss_bf16 = F.cross_entropy(
        logits_bf16_f.reshape(-1, logits_bf16_f.size(-1)), y.reshape(-1)
    )

    logits_fp32 = model(x.float()) if False else None

print("Logits diagnostics (BF16 autocast):")
print("  shape:", tuple(logits_bf16.shape))
print("  dtype:", logits_bf16.dtype)
print("  mean:", f"{logits_bf16_f.mean().item():+.6f}")
print("  std:", f"{logits_bf16_f.std().item():.6f}")
print("  min:", f"{logits_bf16_f.min().item():+.6f}")
print("  max:", f"{logits_bf16_f.max().item():+.6f}")
print("  abs max:", f"{logits_bf16_f.abs().max().item():.6f}")
print("  BF16 CE:", f"{loss_bf16.item():.6f}")
print("  BF16 PPL:", "inf" if loss_bf16.item() >= 80 else f"{math.exp(loss_bf16.item()):.3f}")

# Inspect target-token logits and entropy. A healthy random initialization should
# not have enormous logit magnitudes or an almost-deterministic softmax.
with torch.no_grad():
    flat_logits = logits_bf16_f.reshape(-1, VOCAB_SIZE)
    flat_y = y.reshape(-1)
    target_logits = flat_logits.gather(1, flat_y.unsqueeze(1)).squeeze(1)
    log_probs = F.log_softmax(flat_logits, dim=-1)
    mean_nll = -log_probs.gather(1, flat_y.unsqueeze(1)).mean()
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    top1 = flat_logits.argmax(dim=-1).eq(flat_y).float().mean()
print("Target/softmax diagnostics:")
print("  target-logit mean:", f"{target_logits.mean().item():+.6f}")
print("  target-logit std:", f"{target_logits.std().item():.6f}")
print("  mean NLL independently:", f"{mean_nll.item():.6f}")
print("  mean softmax entropy:", f"{entropy.item():.6f}")
print("  max possible entropy:", f"{math.log(VOCAB_SIZE):.6f}")
print("  teacher-forced top-1 accuracy:", f"{100*top1.item():.4f}%")

# Backward diagnostic without changing weights. We explicitly use float logits
# for the loss to avoid making BF16 cross-entropy itself the source of ambiguity.
logits = None
with torch.autocast("cuda", dtype=torch.bfloat16):
    logits = model(x)
loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
print("Preflight loss:", float(loss.detach()))
assert torch.isfinite(loss)
loss.backward()
gn_raw = torch.linalg.vector_norm(torch.stack([
    p.grad.detach().float().norm() for p in model.parameters() if p.grad is not None
]))
gn_clipped = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
print("Preflight grad norm (raw):", float(gn_raw))
print("Preflight grad norm (clip return):", float(gn_clipped))
assert torch.isfinite(gn_raw)
optimizer.zero_grad(set_to_none=True)

del x, y, logits, logits_bf16, logits_bf16_f, loss, loss_bf16
if 'flat_logits' in locals():
    del flat_logits, flat_y, target_logits, log_probs, probs
if 'mean_nll' in locals():
    del mean_nll, entropy, top1
gc.collect()
torch.cuda.empty_cache()
print("✓ H100 diagnostic preflight passed")

if DIAGNOSTIC_ONLY:
    print("DIAGNOSTIC_ONLY=1: stopping before optimizer updates.")
    print("If logits/loss are healthy, rerun with DIAGNOSTIC_ONLY=0 for training.")
    raise SystemExit(0)

@torch.no_grad()
def evaluate():
    model.eval()
    total_loss = 0.0
    total_n = 0
    for x, y in val_loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model(x)
            l = F.cross_entropy(z.reshape(-1, z.size(-1)),
                                y.reshape(-1))
        if not torch.isfinite(l):
            raise FloatingPointError("Non-finite validation loss")
        n = y.numel()
        total_loss += l.item() * n
        total_n += n
    model.train()
    mean = total_loss / total_n
    return mean, math.exp(mean) if mean < 80 else float("inf")

# ============================================================
# TRAIN
# ============================================================
target_step = min(TOTAL_STEPS, MAX_STEPS) if MAX_STEPS else TOTAL_STEPS
train_iter = iter(train_loader)
model.train()
optimizer.zero_grad(set_to_none=True)
run_t0 = time.time()
last_t = run_t0
last_tok = tokens_seen
loss_sum = 0.0
loss_count = 0

while step0 < target_step:
    accum_loss = 0.0

    for _ in range(GRAD_ACCUM):
        x, y = next(train_iter)
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = model(x)
            l = F.cross_entropy(z.reshape(-1, z.size(-1)),
                                y.reshape(-1))
            scaled_l = l / GRAD_ACCUM

        if not torch.isfinite(l):
            raise FloatingPointError(
                f"Non-finite training loss at step {step0}: {l.item()}"
            )
        scaled_l.backward()
        accum_loss += l.item()
        tokens_seen += MICRO_BATCH * CONTEXT
        del z, scaled_l

    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    if not torch.isfinite(gn):
        raise FloatingPointError("Non-finite gradient norm")

    lr = set_lr(step0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    step0 += 1

    mean_l = accum_loss / GRAD_ACCUM
    loss_sum += mean_l
    loss_count += 1

    if step0 % LOG_EVERY == 0 or step0 == 1:
        now = time.time()
        tps = (tokens_seen - last_tok) / max(now - last_t, 1e-9)
        avg = loss_sum / max(loss_count, 1)
        ppl = math.exp(avg) if avg < 80 else float("inf")
        print(
            f"step={step0:5d}/{target_step} | "
            f"tokens={tokens_seen/1e9:.3f}B | "
            f"loss={avg:.4f} | ppl={ppl:.3f} | "
            f"tok/s={tps:,.0f} | lr={lr:.3e} | "
            f"grad={float(gn):.3f} | "
            f"time={(now-run_t0)/3600:.2f}h",
            flush=True
        )
        last_t, last_tok = now, tokens_seen
        loss_sum, loss_count = 0.0, 0

    if step0 % EVAL_EVERY == 0:
        vl, vp = evaluate()
        print(f"[validation] step={step0} loss={vl:.4f} ppl={vp:.3f}", flush=True)
        if best_val is None or vl < best_val:
            best_val = vl
            model.save(str(BEST_DIR))
            tokenizer.save_pretrained(str(BEST_DIR))
            print("[validation] new best model + tokenizer saved")

    if step0 % SAVE_EVERY == 0:
        save_ckpt(step0, tokens_seen, best_val)

# ============================================================
# FINAL OUTPUT
# ============================================================
model.save(str(MODEL_DIR))
tokenizer.save_pretrained(str(MODEL_DIR))
save_ckpt(step0, tokens_seen, best_val)

print("=" * 80)
print("250M FULL DEVELOPMENT TRAINING COMPLETE")
print("Final model:", MODEL_DIR)
print("Best model:", BEST_DIR)
print("Checkpoints:", CKPT_DIR)
print("Steps:", step0, "Tokens:", f"{tokens_seen:,}")
print("=" * 80)
