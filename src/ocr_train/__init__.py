"""Training-side tooling for the Qwen3.5-9B Arabic OCR fine-tune.

Three jobs, each runnable on its own:

  - `ocr_train.prepare`  OCR-Data/ocr_data tar shards -> ms-swift JSONL + ocr-eval manifests
  - `scripts/train.sh`   ms-swift full fine-tune on 4 GPUs (the only step that needs them)
  - `ocr_train.predict`  a checkpoint's transcriptions, in the form `ocr-eval score` reads

`records` holds the pieces the three must agree on — the prompt, the page_id
scheme, the `[UNREADABLE]` target — so a drift between training and evaluation
is a code change in one file, not a silent mismatch between two.
"""
