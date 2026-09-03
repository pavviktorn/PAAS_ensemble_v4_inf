# PAAS_ensemble_v4_inf — inference only

Face-liveness / deepfake scoring. This tree contains **only what serving needs**: no trainers, no
experiment tooling, no run artifacts. It is a pruned copy of `PAAS_ensemble_v4` (built 2026-08-14).

## What it is

Ensemble of four detectors, fused by plain mean of `{ffaa, A2_9c, gsd, selop}`:

| member | what it is |
|---|---|
| **ffaa** | Qwen3.5-4B MLLM (merged, in-process vLLM) + MIDS 4-class head (CLIP-L/14-336 + T5-base) |
| **A2_9c** | 9-class SVD+GenD head, MLLM-free — 3 fixed claim anchors, `label = 3*true + claim` |
| **gsd** | dual-stream semantic decoupling; the semantic basis (`anchor_U`) is embedded in the checkpoint |
| **selop** | per-layer low-rank orthogonal subspace removal (LROR) |

All four preprocess the **whole frame** (letterbox: pad to square, resize, CLIP-normalise —
`paas/preprocess.py`). No center-crop: for presentation attacks the evidence lives at the border
(screen edges, bezels, hands, moiré).

## Run

Everything uses the one venv (`transformers>=5` is enforced at import; it will not run on tf4):

```bash
VENV=/datasets/work/vLLM/temp/PAAS_qwen3vl/venv/bin/python

# HTTP server
bash run_server.sh                 # uvicorn, single worker; stop with stop_server.sh

# single image / directory
PYTHONNOUSERSITE=1 $VENV inference.py --input <image-or-dir>

# batch video/images
PYTHONNOUSERSITE=1 $VENV test_video_image_batch.py --input <dir>

# score a dataset -> results_paas.txt (the file threshold calibration reads)
PYTHONNOUSERSITE=1 $VENV scripts/run_dataset.py --config config/experiments/paas4_qwen.json \
    --input-dir <dir> --out-dir <out> --frame-stride 10
```

`PYTHONNOUSERSITE=1` matters: `~/.local` still carries a complete legacy stack (transformers 4.37.2,
peft 0.7.1, tokenizers 0.15.2). It is inert only because the venv sorts earlier on `sys.path`, so the
guard in `paas/env.py` aborts if any non-venv `site-packages` appears.

## Layout

```
paas/           pipeline, config, fusion, decision, preprocess, per-model wrappers
ffaa/           MIDS 4-class arch + selector; yolo11 rotation detector (ONNX, every request)
ensemble9/      mids9lib — 9-class model + production scoring
gsd/  selop/    the two CLIP detectors (model/config only; no trainers)
config/         paas4_qwen.json (default) + ensemble9.json + experiment variants
weights/        detector checkpoints 1.7G + qwen35_4b_merged 8.5G
base_models/    clip-vit-large-patch14-336 1.6G, t5-base 853M
```

**STANDALONE — 13 GB, zero symlinks.** Every weight is a real file inside this tree; nothing is
shared with `PAAS_ensemble_v4` or `PAAS_qwen3vl`, so the tree can be moved or archived as-is and a
retrain promoting new weights into the training project cannot change what this one serves. Verify
at any time with `find . -type l` (must print nothing).

Only ONE weight format per model is shipped. The upstream dirs carry the same tensors up to five
times (t5-base had safetensors + .bin + tf + rust + flax at 851 MB each; clip had .bin + tf at
1.6 GB each) — 4.9 GB of pure duplication. Every load site is a plain `from_pretrained(path)` with
no `use_safetensors` override, so transformers takes safetensors when present and `pytorch_model.bin`
otherwise; we keep exactly the file each model will actually open:

| dir | kept | dropped |
|---|---|---|
| `base_models/t5-base` | `model.safetensors` | `.bin`, `tf_model.h5`, `rust_model.ot`, `flax_model.msgpack` |
| `base_models/clip-…-336` | `pytorch_model.bin` (no safetensors upstream) | `tf_model.h5` |
| `weights/qwen35_4b_merged` | `model.safetensors` | — (single format) |

Verified by loading each one after pruning: CLIPVisionModel 303.5M params (hidden 1024, 24 layers),
T5EncoderModel 109.6M (d_model 768), T5Tokenizer(slow) vocab 32100, Qwen3_5Config + 9.1 GB weights.
`CLIPVisionModel` reports `UNEXPECTED text_model.*` keys — expected, it discards CLIP's text tower.

## Scoring, and the one number you must re-fit

Production forgery score for the 9-class member is the **true-class marginal**, not the
winning-answer selector score:

```
Pt = softmax.view(n, 3, 3).sum(dim=2)      # (answers, 3 true-classes)
fake = mean_a(1 - Pt[:, 0])
```

The fused threshold in `config/experiments/paas4_qwen.json` is **specific to these weights**
(fitted at a real-recall floor). After any retrain it must be re-fitted — otherwise fused accuracy
drops even when every individual head improved.

## Not included (training project only)

`train/`, `qwen/` (gen + merge + eval), `run_finetuning.sh`, `mine_hard.py`, `smoke_v4_full.py`,
`status_report.sh`, `runs/`, `docs/`, `images/`. The dropped Python dependencies are listed at the
top of `requirements.txt`.
