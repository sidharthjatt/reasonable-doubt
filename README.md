# Reasonable Doubt

A 3-tier cascade for legal contract clause classification on **LEDGAR**
(`coastalcph/lex_glue`, config `ledgar` — ~80k SEC EDGAR Exhibit-10 provisions,
100 single-label classes), routing each request to the cheapest tier that can handle
it, and measuring whether that actually saves money without losing accuracy.

| Tier | Model | Where |
|------|-------|-------|
| 0 | DeBERTa-v3-base, fine-tuned, ONNX INT8 | local, milliseconds |
| 1 | Qwen2.5-1.5B-Instruct + Llama-3.2-3B-Instruct, QLoRA | local (Apple Silicon, MLX) |
| 2 | Claude Sonnet 5 / Haiku 4.5 | API, Batch + prompt caching |

A router decides per request whether the current tier's answer is trustworthy or
whether to escalate. **The deliverable is a cost-vs-accuracy Pareto frontier, not a
single accuracy number.**

Primary metric is **macro-F1** (the class distribution is heavily long-tailed, so
accuracy is misleading). Accuracy is reported alongside it, never instead of it.

## Constraints

`CLAUDE.md` holds the binding constraints — hardware, Batch API behaviour, hard rules,
budget. Read it before writing code. The ones that bite most often:

- Router thresholds calibrate on **dev only**. Test is touched once.
- Results are **mean ± std over ≥ 3 seeds**.
- API responses are **cached to disk by prompt hash before use**; never re-call.
- **No hardcoded prices** — everything derives from `configs/costs.yaml`.
- Every experiment is preregistered in `PREREGISTRATION.md` with its accept rule
  written before the run.
- Spending scripts print an estimated cost and require `--confirm`.

## Layout

    configs/         costs.yaml, model configs, prompt templates
    src/data/        LEDGAR loading, splits, label handling
    src/api/         cache layer, batch client, token counting, cost calculation
    src/train/       Kaggle training scripts (Tier 0 encoder, Tier 1 QLoRA)
    src/serve/       local serving, FastAPI app
    src/router/      routing strategies, threshold calibration
    src/eval/        metrics, bootstrap CIs, Pareto frontier
    scripts/         thin CLI entrypoints
    cache/           gitignored, on-disk API response cache
    results/         gitignored except .gitkeep
    tests/

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,mlx]"
cp .env.example .env   # then fill in keys
```

Kaggle training deps are in `requirements-kaggle.txt` and are installed inside the
Kaggle runtime only — `bitsandbytes` is CUDA-only and must never be installed locally.

## Status

Scaffold only. No data, training, or API code yet.
