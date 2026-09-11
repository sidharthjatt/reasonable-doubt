---
title: Reasonable Doubt
emoji: ⚖️
colorFrom: yellow
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
license: cc-by-4.0
models:
  - sidharthjatt/reasonable-doubt-deberta-ledgar
datasets:
  - coastalcph/lex_glue
short_description: Classify a contract clause into 100 LEDGAR provision types
---

# Reasonable Doubt

Paste a contract clause; get a **LEDGAR provision type** (100 classes), the **top 3** with
scores, and a **`needs_review` flag** when the model is not confident.

Serves [`sidharthjatt/reasonable-doubt-deberta-ledgar`](https://huggingface.co/sidharthjatt/reasonable-doubt-deberta-ledgar)
— DeBERTa-v3-base fine-tuned on LEDGAR (LexGLUE), ONNX FP32, **macro-F1 0.8091** on a
3,000-row test sample, **single seed**.

**Research artefact. Not legal advice.**

## Endpoints

| | |
|---|---|
| `GET /` | the demo page |
| `POST /classify` | `{"text": "..."}` → label, `top_3`, `needs_review`, `margin`, cost |
| `GET /health` | canary result, CPU ISA flags, router provenance |
| `GET /docs` | OpenAPI |

Input is capped at 20,000 characters.

## What the flag means

`needs_review: true` means **low confidence — human review recommended**: the margin
between the top two classes fell below a threshold calibrated on a held-out dev split
(never on test). On exactly those rows the model measures **~0.35 accuracy** against
~0.87 overall. It is a reliable signal that the answer is doubtful — not a second opinion.

Escalating those rows to a frontier LLM **was measured and did not help**
(+0.0008 macro-F1, 95% CI [−0.0060, +0.0072]), so this Space flags instead of escalating.
It makes **no API calls** and holds **no secrets**.

## Two startup refusals

1. **Weights are verified by sha256** at build time against the published model repo. A
   mismatch fails the build rather than serving unrecorded weights.
2. **A canary classifies 200 fixed rows before the server binds** and refuses to start
   below an accuracy floor. This matters here: an INT8 quantisation of these weights
   collapses to chance on CPUs without AVX-512 VNNI — silently, raising nothing. FP32 is
   served for that reason, and the canary is what proves it on this host.

## Attribution

Derivative of **LexGLUE** (CC-BY-4.0) and **microsoft/deberta-v3-base** (MIT). Full
citations, licence notices and limitations are on the
[model card](https://huggingface.co/sidharthjatt/reasonable-doubt-deberta-ledgar).
