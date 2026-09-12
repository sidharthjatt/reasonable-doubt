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
| Live | <https://reasonable-doubt-111680840326.asia-south1.run.app> |

![Accuracy versus cost](docs/figures/accuracy_vs_cost.png)

The fine-tuned encoder beats both frontier models on this task and costs about 1/500th as
much. Escalating its low-confidence rows to Sonnet 5 was measured and did not help:
+0.000822 macro-F1, 95% CI [−0.0060, +0.0072]. So I turned escalation off and kept the
router, which flags those rows instead. [REPORT.md](REPORT.md) has the numbers, the
limitations, and the bugs that produced plausible wrong answers along the way.

## The live service, and its cold start

It runs on Cloud Run in Mumbai, scaled to zero. **The first request after an idle period
takes about 150 seconds.** Warm requests take 0.2 to 0.3 seconds.

That is not a bug and I have not tuned it away. With `min-instances 0` nothing is running
between requests, so a cold request pays for the container starting, the 715 MB ONNX
session opening, and a 200-row canary finishing before the port opens. Keeping an instance
warm would fix it and would also burn free-tier quota around the clock, which is a bad
trade for a demo. Load the page, wait, then it is fast.

**It needs 4 GiB, not 2.** A 2 GiB revision failed to start: `Memory limit of 2048 MiB
exceeded with 2087 MiB used`, a 2% overshoot, during the canary. Worth knowing before
sizing a box for this.

A Hugging Face Space is staged in `deploy/space/` but not deployed, because HF gates
compute-backed Spaces behind a PRO subscription.

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

Now something that is not a contract clause at all. Input: *"Please preheat the oven to 180
degrees Celsius and bake the cake for 35 minutes until golden brown."*

```json
{
  "label": "Terms",
  "margin": 0.4477890729904175,
  "needs_review": true,
  "needs_review_meaning": "low confidence — human review recommended",
  "top_3": [
    {"label": "Terms",           "score": 0.5904529792940023},
    {"label": "Interpretations", "score": 0.14266393628137297},
    {"label": "General",         "score": 0.07784326163120221}
  ],
  "tier_used": "tier0",
  "estimated_cost_usd": 8.788115222574201e-07
}
```

A 100-class classifier has no "none of these" option, so it must return a label, and it
returns `Terms`. What it does not do is return it confidently: 59% against a top-two margin
of 0.45, under the threshold, flagged. That is the behaviour worth having. The failure mode
to fear from a model like this is not a wrong label, it is a wrong label at 99%.

It fires on genuinely ambiguous contract text too. *"This Sixth Amendment shall not be valid
and binding on Landlord and Tenant unless and until it has been completely executed and
delivered"* splits 0.55 `Effectiveness` against 0.41 `Binding Effects`, margin 0.14, flagged.
Both labels are defensible there.

The threshold comes from a held-out split, never from test. On the rows it flags, the
encoder is right about 35% of the time against 87% overall, so the flag is a reliable signal
that the answer is doubtful. It is not a second opinion, and nothing downstream acts on it.

## Quickstart

Try the live one, or run it yourself:

```bash
docker build -t reasonable-doubt:local . && docker run --rm -p 8000:8000 -v "$PWD/models:/models:ro" -e TIER0_MODEL_DIR=/models/onnx_ce10ep_1_fp32 reasonable-doubt:local
```

Weights are on the Hub:
<https://huggingface.co/sidharthjatt/reasonable-doubt-deberta-ledgar>
(`model.onnx`, sha256 `e70fb095ee8ddd82c9449d6da0006b4fa1c7a92bb8547d5538f869c445010200`).

The container classifies 200 fixed rows before uvicorn binds and exits non-zero if accuracy
falls below 0.80. That check exists because the INT8 build of these same weights scores
0.000166 on a CPU without AVX-512 VNNI, which is chance, and raises no error while doing it.
On Cloud Run it passes at 191/200, the same as everywhere else. **That machine has
AVX-512 VNNI**, so it does not test the hardware class INT8 broke on; see
[REPORT.md](REPORT.md).

To deploy your own copy, `deploy/cloudrun/` has the image (model baked in, sha256 verified
at build) and the exact `gcloud run deploy` flags.

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
