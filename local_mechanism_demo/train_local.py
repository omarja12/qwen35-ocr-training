"""Train the tiny stand-in on the local Hugging Face samples: real pages, held-out validation.

run_local_demo.py proves the mechanism on ONE page repeated. This runs the same
model, masking and loss over many different pages - 900 train / 100 validation,
one pass - and measures loss on pages the model never trained on.

Same limit as the demo: the stand-in's vision tower is broken, so image
embeddings are a fixed random stand-in. The model learns the Arabic text, not
how to read an image. This proves the pipeline; the real Qwen3.5-9B needs the GPUs.

    python fetch_hf_samples.py --n 1000            # once; needs HF_TOKEN (gated dataset)
    python train_local.py                          # full run
"""

import argparse
import datetime
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoProcessor

from run_local_demo import (
    MODEL_ID,
    SEED,
    build_batch,
    cross_entropy_loss,
    load_fresh_model_and_tokenizer,
    load_real_sample,
)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
LOGS = HERE / "logs"
N_VAL = 100
EVAL_EVERY = 100
MAX_TOKENS = 300  # ponytail: RAM cap for an 8 GB laptop (logits scale with the 248K vocab); raise on a real machine

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=1, help="passes over the training pages")
args = parser.parse_args()

stems = sorted(p.with_suffix("") for p in DATA.glob("*.json"))
random.Random(SEED).shuffle(stems)
val_stems, train_stems = stems[:N_VAL], stems[N_VAL:]
assert val_stems and train_stems, f"no samples in {DATA} - run fetch_hf_samples.py"

torch.manual_seed(SEED)
model, tok = load_fresh_model_and_tokenizer()
processor = AutoProcessor.from_pretrained(MODEL_ID)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


def batch(stem):
    embeds, labels, stats = build_batch(model, tok, processor, load_real_sample(stem))
    return (embeds, labels) if stats["total_tokens"] <= MAX_TOKENS else None


def loss_on(embeds, labels):
    mask = torch.ones(embeds.shape[:2], dtype=torch.long)
    return cross_entropy_loss(model(inputs_embeds=embeds, attention_mask=mask).logits, labels)


def val_loss():
    model.eval()
    with torch.no_grad():
        losses = [loss_on(*b).item() for b in map(batch, val_stems) if b]
    model.train()
    return sum(losses) / len(losses)


LOGS.mkdir(exist_ok=True)
stamp = datetime.date.today().isoformat()
if args.epochs > 1:
    stamp += f"_{args.epochs}ep"
log_path = LOGS / f"train_local_{stamp}.log"
log = log_path.open("w", encoding="utf-8")


def say(line):
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()


say(f"model {MODEL_ID} | {len(train_stems)} train / {len(val_stems)} val pages | "
    f"{args.epochs} epoch(s) | AdamW lr 1e-3, batch 1, plain CE, max {MAX_TOKENS} tokens")
train_curve, val_curve, skipped, started = [], [], 0, time.time()
val_curve.append((0, val_loss()))
say(f"step    0  val {val_curve[-1][1]:.4f}")

# Epoch 1 keeps the original order, so a 1-epoch run is unchanged; later epochs reshuffle.
order = list(train_stems)
for e in range(1, args.epochs):
    shuffled = list(train_stems)
    random.Random(SEED + e).shuffle(shuffled)
    order += shuffled
total_steps = len(order)

for step, stem in enumerate(order, start=1):
    b = batch(stem)
    if b is None:
        skipped += 1
        continue
    optimizer.zero_grad()
    loss = loss_on(*b)
    loss.backward()
    optimizer.step()
    train_curve.append((step, loss.item()))
    if step % 10 == 0:
        say(f"step {step:4d}  train {loss.item():.4f}  ({time.time() - started:.0f}s)")
    if step % EVAL_EVERY == 0 or step == total_steps:
        val_curve.append((step, val_loss()))
        say(f"step {step:4d}  val {val_curve[-1][1]:.4f}")

say(f"done in {time.time() - started:.0f}s | {len(train_curve)} training steps, "
    f"skipped {skipped} over {MAX_TOKENS} tokens | val {val_curve[0][1]:.4f} -> {val_curve[-1][1]:.4f}")
log.close()

# Chart: train loss smoothed over 25 steps, validation loss at each evaluation.
window = 25
steps = [s for s, _ in train_curve]
raw = [v for _, v in train_curve]
smooth = [sum(raw[max(0, i - window + 1): i + 1]) / len(raw[max(0, i - window + 1): i + 1]) for i in range(len(raw))]
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
ax.plot(steps, smooth, color="#2a78d6", linewidth=2, label=f"train (avg of {window} steps)")
for e in range(1, args.epochs):  # epoch boundaries
    ax.axvline(e * len(train_stems), color="#c9c9c4", linewidth=1, linestyle="--")
    ax.text(e * len(train_stems), 0.98, f" epoch {e + 1}", transform=ax.get_xaxis_transform(),
            va="top", fontsize=8, color="#5f5f5a")
vs, vv = zip(*val_curve)
ax.plot(vs, vv, color="#eb6834", linewidth=2, marker="o", markersize=5, label="validation (100 unseen pages)")
for x, y, name in ((steps[-1], smooth[-1], "train"), (vs[-1], vv[-1], "validation")):
    ax.annotate(f"{name} {y:.2f}", (x, y), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=9, color="#3d3d3a")
ax.set_title(f"Tiny Qwen3.5 stand-in, {len(train_stems)} real OCR pages × {args.epochs} epoch(s), CPU",
             loc="left", fontsize=11, color="#1f1f1d")
ax.set_xlabel("training step (one page each)", color="#5f5f5a")
ax.set_ylabel("cross-entropy loss", color="#5f5f5a")
ax.grid(axis="y", color="#e6e6e3", linewidth=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#c9c9c4")
ax.tick_params(colors="#5f5f5a")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
chart_path = LOGS / f"train_local_{stamp}.png"
fig.savefig(chart_path)
print(f"log:   {log_path}\nchart: {chart_path}")
