# Reasonable Doubt

Classify a contract clause into one of LEDGAR's 100 provision types, on a CPU, for
$0.00088 per thousand. When the model is unsure it says so instead of guessing quietly.

LEDGAR is `coastalcph/lex_glue`, config `ledgar`: about 80,000 provisions from SEC EDGAR
Exhibit-10 filings, 100 single-label classes, split chronologically rather than at random.

I set out to build a three-tier cost cascade: a local encoder, a small fine-tuned decoder,
then the Claude API for the hard rows. I measured each tier before trusting it, and two of
the three turned out to be unnecessary.

| | |
|---|---|
| **macro-F1, `test_3000`** | **0.8091** (ONNX FP32, arm64, single seed) |
| Sonnet 5 zero-shot, same 3,000 rows | 0.6148 at **496×** the cost per clause |
| Architecture | one tier, plus a `needs_review` flag |

![Accuracy versus cost](docs/figures/accuracy_vs_cost.png)

The fine-tuned encoder beats both frontier models on this task and costs about 1/500th as
much. Escalating its low-confidence rows to Sonnet 5 was measured and did not help:
+0.000822 macro-F1, 95% CI [−0.0060, +0.0072]. So I turned escalation off and kept the
router, which flags those rows instead. [REPORT.md](REPORT.md) has the numbers, the
limitations, and the bugs that produced plausible wrong answers along the way.

## There is no live demo link

Hugging Face gates compute-backed Spaces behind a PRO subscription, so the hosted demo is
not up. The service is a self-contained Docker image and runs anywhere with a CPU. Its
startup canary passes at 191/200 on macOS arm64 and on Linux aarch64, and at 191/200 on a
Linux x86 build that was emulated on Apple Silicon rather than run on real x86 hardware. The
Space deployment is staged in `deploy/space/` and needs only an account that can host it.

## What it does, on real input

A confident clause. This is actual `/classify` output, with the `cost_detail` and `router`
blocks trimmed.

```json
{
  "label": "Governing Laws",
  "margin": 0.9867302775382996,
  "needs_review": false,
  "needs_review_meaning": null,
  "top_3": [
    {"label": "Governing Laws",  "score": 0.9931608497753377},
    {"label": "Applicable Laws", "score": 0.006430555948988789},
    {"label": "Construction",    "score": 0.00014505338475388416}
  ],
  "tier_used": "tier0",
  "estimated_cost_usd": 8.788115222574201e-07
}
```

Input: *"This Agreement shall be governed by and construed in accordance with the laws of
the State of New York, without regard to its conflict of laws principles."*

Now one the model should not be confident about. Input: *"This Sixth Amendment shall not be
valid and binding on Landlord and Tenant unless and until it has been completely executed
by and delivered to both parties."*

```json
{
  "label": "Effectiveness",
  "margin": 0.1398802399635315,
  "needs_review": true,
  "needs_review_meaning": "low confidence — human review recommended",
  "top_3": [
    {"label": "Effectiveness",   "score": 0.5495100799149649},
    {"label": "Binding Effects", "score": 0.40962984841772326},
    {"label": "Assigns",         "score": 0.01274250195615738}
  ],
  "tier_used": "tier0",
  "estimated_cost_usd": 8.788115222574201e-07
}
```

That clause is genuinely two things at once, and the model splits 0.55 to 0.41 between two
defensible labels. The flag fires because the top-two margin falls below a threshold
calibrated on a held-out split. On the rows it flags, the encoder is right about 35% of the
time against 87% overall, so the flag is a reliable signal that the answer is doubtful. It
is not a second opinion, and nothing downstream acts on it yet.

## Quickstart

```bash
docker build -t reasonable-doubt:local . && docker run --rm -p 8000:8000 -v "$PWD/models:/models:ro" -e TIER0_MODEL_DIR=/models/onnx_ce10ep_1_fp32 reasonable-doubt:local
```

Weights are on the Hub:
<https://huggingface.co/sidharthjatt/reasonable-doubt-deberta-ledgar>
(`model.onnx`, sha256 `e70fb095ee8ddd82c9449d6da0006b4fa1c7a92bb8547d5538f869c445010200`).

The container classifies 200 fixed rows before uvicorn binds and exits non-zero if accuracy
falls below 0.80. That check exists because the INT8 build of these same weights scores
0.000166 on a CPU without AVX-512 VNNI, which is chance, and raises no error while doing it.

`/classify` takes `{"text": "..."}` up to 20,000 characters. `/health` reports the canary
result, the CPU ISA flags and the router's provenance. `/` is a one-page demo.

## Develop

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev,mlx]" && pytest -q
```

Kaggle training dependencies live in `requirements-kaggle.txt` and install inside the Kaggle
runtime only. `bitsandbytes` is CUDA-only and does not work on Apple Silicon, so it must not
be installed locally. `CLAUDE.md` carries the binding project constraints.

## Repository

```
configs/     costs.yaml (every price), serve.yaml, frozen manifests, router threshold
src/data/    LEDGAR loading, splits, label space
src/api/     response cache, batch client, token counting, cost from usage fields
src/train/   Kaggle training scripts (encoder, QLoRA)
src/serve/   ONNX encoder, router, FastAPI app, startup canary, local benchmarks
src/router/  routing signals and threshold calibration
src/eval/    metrics, bootstrap CIs, break-even curve
scripts/     CLI entrypoints, including the report's figure generation
deploy/      model upload, and the staged Docker Space
results/     committed result summaries and the spend ledger
```

## Reading further

[REPORT.md](REPORT.md) is the writeup: what is running, what the cascade premise turned
into, the findings with their limitations attached, and a section on the four bugs that
produced believable wrong numbers.

[PREREGISTRATION.md](PREREGISTRATION.md) is the appendix and the working record. Every
accept rule was written before its run, and rejected experiments stay in it. It is long and
it is not a summary of the report; it is what the report is accountable to.

## Licence and attribution

**The code in this repository is MIT ([LICENSE](LICENSE)). The trained weights are
CC-BY-4.0, which is a different licence with an attribution requirement.** The weights carry
it because they derive from LEDGAR as distributed in LexGLUE (CC-BY-4.0) and from
`microsoft/deberta-v3-base` (MIT). Citations and full licence notices are on the
[model card](https://huggingface.co/sidharthjatt/reasonable-doubt-deberta-ledgar).

Research artefact, trained on 2016–2017 US SEC EDGAR filings and tested on 2019. Not legal
advice.
