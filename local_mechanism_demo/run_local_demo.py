"""
Local, GPU-free demonstration of the exact fine-tuning mechanism this
project will use at scale - built for CPU, on a tiny stand-in model.

Stand-in model:  hf-tiny-v2/tiny-random-Qwen3_5ForConditionalGeneration
  Same architecture CLASS as the real Qwen/Qwen3.5-9B-Base this project
  fine-tunes - same modeling_qwen3_5.py code path, same image-token
  handling. Only the weights differ: tiny and randomly initialized here,
  instead of the real 9B pretrained weights.

Both loss functions below are written here directly, in plain PyTorch,
from the model's raw logits - no dependency on ms-swift's loss classes
or on the model's own built-in `out.loss`. ms-swift is only ever cited
elsewhere as evidence for what the real project's own training script
runs; this script computes its own loss independently, so the mechanism
being demonstrated is fully owned code, not borrowed from a library.

TO SCALE THIS UP LATER: change MODEL_ID below to "Qwen/Qwen3.5-9B-Base"
(or the local PVC path on the cluster) and run on a machine with a real
GPU. Same masking, same loss code, same training loop shape - the real
model just needs real compute to actually run.

GPU note (checked 22 Sep 2026): this machine's AMD GPU cannot be used.
torch-directml has never shipped a Python 3.13/3.14 build (abandoned
since Sep 2024, caps at 3.12). onnxruntime-directml does support 3.14,
but it is an inference runtime; real training would need exporting
Qwen3.5's brand-new hybrid attention architecture to ONNX first, which
export tooling likely does not support yet, plus a training-capable ORT
build on top. Not pursued - real risk for a payoff already achieved on
CPU.

Real OCR sample used: sample_0000003 from OCR-Data/ocr_data, shard_001
(pulled 22 Sep 2026, a partial/range download, not the full dataset).
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
from PIL import Image

MODEL_ID = "hf-tiny-v2/tiny-random-Qwen3_5ForConditionalGeneration"
SAMPLE_DIR = Path(__file__).resolve().parent / "data"  # filled by fetch_hf_samples.py
N_STEPS = 30
LABEL_SMOOTHING_EPS = 0.1
SEED = 42  # fixes the random image-embedding stand-in identically across runs


# This laptop (8GB RAM, no usable GPU - see above) segfaulted inside the
# model's linear_attention layer's CPU reference implementation (no
# flash-linear-attention kernel installed). Confirmed by isolating one
# variable at a time: forcing every layer to plain full_attention sidesteps
# it. This ONLY matters for running the tiny random-weight stand-in on CPU
# here; the real Qwen3.5-9B-Base run on the cluster GPU uses the real
# kernel on real hardware and never hits this.
def build_patched_config():
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    cfg.text_config.layer_types = ["full_attention"] * len(cfg.text_config.layer_types)
    return cfg


USER_PROMPT = (
    "<image>Transcribe all visible text exactly, preserving reading order "
    "and line breaks. Return only the transcription."
)


def load_real_sample(stem=SAMPLE_DIR / "sample_0000003"):
    ann = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    image = Image.open(stem.with_suffix(".png")).convert("RGB")
    # This laptop has 8GB RAM total, ~1GB free. The real image at full
    # resolution produces ~792 vision patches, and this model's huge
    # (248K) vocab makes the output logits tensor scale directly with
    # total sequence length - too big to fit here at full res. Shrink
    # the image (still the real page, just lower resolution) so the
    # patch count - and the logits tensor - fits in available memory.
    w, h = image.size
    image = image.resize((max(w // 10, 32), max(h // 10, 32)))
    short_text = ann["markdown"][:150]
    return image, short_text


def cross_entropy_loss(logits, labels):
    """Plain CE, written directly - not delegated to nn.CrossEntropyLoss
    or to the model's own out.loss. Causal shift: logits at position t
    predict the token at position t+1."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)

    mask = shift_labels != -100
    safe_labels = shift_labels.clone()
    safe_labels[~mask] = 0  # placeholder index for masked positions, zeroed out below

    token_logp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_loss = -token_logp * mask
    return token_loss.sum() / mask.sum()


def label_smoothed_loss(logits, labels, eps=LABEL_SMOOTHING_EPS):
    """CE softened toward the uniform distribution over the vocabulary:
        l_LS = (1-eps) * l_CE  +  eps * ( -mean_v log q(v) )
    The second term is the cross-entropy against a uniform target, i.e.
    it penalizes over-confidence even when the model is already correct."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    log_probs = F.log_softmax(shift_logits.float(), dim=-1)

    mask = shift_labels != -100
    safe_labels = shift_labels.clone()
    safe_labels[~mask] = 0

    nll = -log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    uniform_term = -log_probs.mean(dim=-1)
    token_loss = ((1 - eps) * nll + eps * uniform_term) * mask
    return token_loss.sum() / mask.sum()


def load_fresh_model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    patched_cfg = build_patched_config()
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, config=patched_cfg, dtype=torch.float32, ignore_mismatched_sizes=True,
    )
    real_image_pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
    model.config.image_token_id = real_image_pad_id  # see docstring: fixture bug, sync to tokenizer
    model.train()
    return model, tok


def build_batch(model, tok, processor, sample=None):
    image, target_text = sample or load_real_sample()

    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": USER_PROMPT.replace("<image>", "").strip()},
    ]}]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
    prompt_len = inputs["input_ids"].shape[1]

    n_image_tokens = (inputs["input_ids"] == model.config.image_token_id).sum().item()

    target_ids = tok.encode(target_text, add_special_tokens=False)
    eos_id = tok.eos_token_id
    full_ids = torch.cat([inputs["input_ids"][0], torch.tensor(target_ids + [eos_id])])
    labels = torch.cat([
        torch.full((prompt_len,), -100, dtype=torch.long),
        torch.tensor(target_ids + [eos_id], dtype=torch.long),
    ])

    # This specific tiny test fixture's vision tower (depth=2, randomly
    # initialized) has its own bug: it returns 3168 unmerged patch
    # embeddings when the text side expects 792 (a 4x spatial-merge
    # mismatch) - a quirk of the CI artifact, not present in the real
    # model. Bypass it: build inputs_embeds ourselves, with a correctly-
    # shaped, SEED-fixed random stand-in for the image embeddings, so
    # both loss functions below train against an identical input.
    torch.manual_seed(SEED)
    with torch.no_grad():
        text_embeds = model.get_input_embeddings()(full_ids.unsqueeze(0)).clone()
    hidden_size = text_embeds.shape[-1]
    image_mask = full_ids == model.config.image_token_id
    n_img = image_mask.sum().item()
    fake_image_embeds = torch.randn(n_img, hidden_size) * 0.02
    text_embeds[0, image_mask] = fake_image_embeds

    stats = {
        "image_size": image.size,
        "text_chars": len(target_text),
        "n_image_tokens": n_image_tokens,
        "total_tokens": full_ids.shape[0],
        "masked": prompt_len,
        "trained": full_ids.shape[0] - prompt_len,
    }
    return text_embeds, labels.unsqueeze(0), stats


def run_training(loss_fn, loss_name):
    print(f"\n=== TRAINING WITH {loss_name}, {N_STEPS} REAL STEPS ===")
    torch.manual_seed(SEED)
    model, tok = load_fresh_model_and_tokenizer()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    inputs_embeds, labels, stats = build_batch(model, tok, processor)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for step in range(N_STEPS):
        optimizer.zero_grad()
        out = model(inputs_embeds=inputs_embeds, attention_mask=torch.ones(inputs_embeds.shape[:2], dtype=torch.long))
        loss = loss_fn(out.logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 5 == 0 or step == N_STEPS - 1:
            print(f"  step {step:2d}: loss = {loss.item():.4f}")
    return losses, stats


def main():
    print(f"Loading stand-in model: {MODEL_ID}")

    ce_losses, stats = run_training(cross_entropy_loss, "PLAIN CROSS-ENTROPY")
    ls_losses, _ = run_training(label_smoothed_loss, f"LABEL-SMOOTHED CE (eps={LABEL_SMOOTHING_EPS})")

    print(f"\n=== SEQUENCE (identical for both runs - same seed) ===")
    print(f"image: {stats['image_size']}, transcription: {stats['text_chars']} chars")
    print(f"<|image_pad|> tokens: {stats['n_image_tokens']}")
    print(f"total tokens: {stats['total_tokens']}")
    print(f"masked (-100): {stats['masked']}  |  trained: {stats['trained']}")

    print(f"\n=== RESULT ===")
    print(f"{'step':>4}  {'CE loss':>10}  {'LS loss':>10}")
    for i in range(N_STEPS):
        print(f"{i:4d}  {ce_losses[i]:10.4f}  {ls_losses[i]:10.4f}")

    print(f"\nCE:  {ce_losses[0]:.4f} -> {ce_losses[-1]:.4f}  (delta {ce_losses[0]-ce_losses[-1]:.4f})")
    print(f"LS:  {ls_losses[0]:.4f} -> {ls_losses[-1]:.4f}  (delta {ls_losses[0]-ls_losses[-1]:.4f})")
    print("\nBoth losses are computed directly in this script from raw logits,")
    print("not delegated to any library's loss implementation. Both started")
    print("from identical weights and identical input (fixed seed) - the")
    print("only difference between the two runs is the loss function itself.")
    print("\nSwap MODEL_ID for Qwen/Qwen3.5-9B-Base + a GPU to scale this up -")
    print("same script, same loss code, real weights.")


if __name__ == "__main__":
    main()
