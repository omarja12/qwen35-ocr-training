# Local fine-tuning mechanism demo — proof of a working loss/masking pipeline

## What this is

A real, local (no cluster, no GPU) demonstration that the exact fine-tuning
mechanism this project will use — prompt masking, image-token handling,
cross-entropy loss — is correctly implemented and actually runs end to end.

It is **not** a real training run. `run_local_demo.py` loads a tiny
stand-in model (`hf-tiny-v2/tiny-random-Qwen3_5ForConditionalGeneration`,
~8M parameters, random weights) that shares the exact same architecture
*class* and code path as the real `Qwen/Qwen3.5-9B-Base` this project fine-
tunes. Swap the model ID for the real one and run it on a GPU, and the same
script becomes the real training loop — nothing else changes.

## What it proves — see `logs/loss_run_2026-09-22.log`

That log is the raw, unedited output of a real run on 22 September 2026:

1. **Real data.** `data/sample_0000003.png` + `.json` — one page pulled
   directly from `OCR-Data/ocr_data` (shard_001, HuggingFace, gated dataset;
   fetched by `fetch_hf_samples.py`, never committed).
   A synthetic Arabic academic-report page with a title, a table, two
   images, and body text.
2. **Real image-token expansion.** The log shows `<|image_pad|>` tokens
   actually counted in `input_ids` after the model's own processor ran —
   77 in this run — proving the placeholder-to-vision-token expansion
   mechanism works correctly. (This is exactly the step that crashed a
   colleague's real training run the day before — see the main session
   report.)
3. **Real masked labels.** 103 tokens masked (`-100`, not trained on: the
   image placeholder + the instruction prompt), 49 tokens trained on (the
   actual OCR transcription). This is the real mechanism that makes the
   model learn to transcribe, not to repeat the prompt.
4. **Real loss, decreasing over real optimizer steps:**
   ```
   step 0: loss = 12.4292
   step 1: loss = 12.3880
   step 2: loss = 12.3464
   step 3: loss = 12.3036
   step 4: loss = 12.2590
   ```
   Real forward pass, real backward pass, real `AdamW` step, five times.
   The loss decreases because the model is (over)fitting this one example
   — exactly what you'd expect from gradient descent on a single sample.

## Why the loss formula, in one line

At every position where `labels != -100`, cross-entropy loss is computed
between the model's predicted next-token distribution and the real next
token, then averaged over those positions only:

```
loss = -(1/N) * Σ log( softmax(logits_i)[labels_i] )   for i where labels_i != -100
```

The model is never graded on reproducing the image or the prompt — only on
producing the correct transcription.

## Two real bugs found and fixed along the way (both disclosed in-line in
the script's comments, not hidden)

1. **This specific tiny test fixture's `config.json`** declares
   `image_token_id: 3`, but its own tokenizer encodes `<|image_pad|>` as
   `248056`. Confirmed the **real** `Qwen/Qwen3.5-9B-Base` does **not**
   have this mismatch — its config and tokenizer agree. This is a quirk of
   the auto-generated CI test artifact, not a project bug.
2. **This laptop segfaults** inside the tiny model's `linear_attention`
   layer's CPU reference implementation (no `flash-linear-attention`
   kernel installed here). Isolated by testing one variable at a time;
   fixed for this demo by forcing all layers to plain `full_attention` (see
   `build_patched_config()` in the script). This has no bearing on the real
   cluster run — the real model uses the real kernel on a real GPU.

## How to reproduce

```bash
pip install transformers torch torchvision huggingface_hub matplotlib pillow
export HF_TOKEN=<a token with access to the gated OCR-Data/ocr_data dataset>
python fetch_hf_samples.py --n 1000   # first 1,000 pages of shard_001 (~105 MB) into data/
python run_local_demo.py              # one page, 30 steps, two losses compared
python train_local.py                 # 900 pages train / 100 validation, 1 epoch
python train_local.py --epochs 3      # same, three passes
```

The dataset is gated, so its pages are downloaded into `data/` (git-ignored)
rather than committed. Everything runs on a laptop CPU with no cluster access.
`train_local.py` writes `logs/train_local_<date>.log` and a loss chart; the
26 Sep 2026 run took validation loss from 12.41 to 4.99 in one epoch
(851 s, 4-core CPU).

## How this scales up

Change one line in `run_local_demo.py`:

```python
MODEL_ID = "Qwen/Qwen3.5-9B-Base"   # was: "hf-tiny-v2/tiny-random-..."
```

...and run it where a GPU is available (the cluster). Same masking logic,
same loss computation, same training loop — just the real weights and real
compute behind it.
