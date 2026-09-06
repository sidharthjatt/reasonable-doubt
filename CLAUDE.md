# Reasonable Doubt — project constraints

A 3-tier cascade for legal contract clause classification (LEDGAR, 100 classes) that
routes each request to the cheapest tier that can handle it, and MEASURES whether that
actually saves money without losing accuracy.

  Tier 0: fine-tuned DeBERTa-v3-base encoder, ONNX INT8, runs locally (milliseconds)
  Tier 1: Qwen2.5-1.5B-Instruct + Llama-3.2-3B-Instruct, QLoRA fine-tuned, served
          locally on Apple Silicon
  Tier 2: Claude API (Sonnet 5 / Haiku 4.5), batch + prompt caching

A router decides, per request, whether the current tier's answer is trustworthy or
whether to escalate. The deliverable is a cost-vs-accuracy Pareto frontier, not a
single accuracy number.

## Dataset (verified — do not substitute)

- HuggingFace: `coastalcph/lex_glue`, config `ledgar`
- ~80k contract provisions from SEC EDGAR Exhibit-10 filings, 100 single-label classes
- Split is CHRONOLOGICAL, not random: train 60k (2016-2017), dev 10k (2018),
  test 10k (2019). PRESERVE THIS ORDER. Never reshuffle across splits.
- Heavy long-tail class imbalance.
- PRIMARY METRIC IS MACRO-F1, not accuracy. Accuracy is misleading on this
  distribution. Report both, lead with macro-F1.

## Hardware (verified — do not suggest otherwise)

- Local machine: Apple Silicon Mac Mini, 10-core CPU / 10-core GPU.
- `bitsandbytes` is CUDA-only. It does NOT work on macOS/MPS. PyTorch MPS has no
  QLoRA. Axolotl depends on bitsandbytes. Never propose these for local training.
- Local training path, if any, is MLX (`mlx-lm`): supports LoRA/DoRA/QLoRA. QLoRA
  activates automatically when `--model` points at an already-quantized model.
- Main training happens on Kaggle (free CUDA GPU) with PyTorch + transformers + peft
  + bitsandbytes. Scripts must be written to run as a standalone Kaggle notebook/script.
- Local serving uses `mlx_lm.server` (OpenAI-compatible) and/or llama.cpp GGUF.

## Claude Batch API facts (verified — build to these)

- 50% discount on all usage. Most batches finish under 1 hour; bound is 24 hours.
- Max 10,000 requests per batch. Results available 29 days.
- Results return IN ANY ORDER. Join on `custom_id`. NEVER on list position.
- Prompt caching works with batches but cache hits are BEST-EFFORT because requests
  process concurrently and out of order. Therefore: set cache TTL to "1h", not the
  5-minute default, and send a small warm-up batch first to write the cache.
- Token counts must come from the API's `count_tokens` endpoint for the target model.
  tiktoken undercounts Claude tokens — never use it for budgeting.
- Read `usage.cache_read_input_tokens` off every response so we know actual cache
  behaviour rather than assuming it.
- Claude 4.7 and later use a NEWER TOKENIZER producing roughly 30% more tokens for
  the same text; Sonnet 4.6 and earlier use the previous one. Sonnet 5 and Haiku 4.5
  therefore report DIFFERENT token counts for an identical clause. The whole cost
  comparison depends on respecting this — see hard rule 9.
- Pricing modifiers STACK: a cache-read token inside a batch bills at
  0.5 * 0.1 * base_input. Cache writes cost 1.25x (5m) / 2.0x (1h) of base input;
  caching pays off after one cache read at 5m, or two at 1h.

## Hard rules — never violate these

1. Router thresholds are calibrated on the DEV split only. Never on test.
2. Every model result reports mean ± std over >= 3 seeds. Never a single best run.
3. Every API response is cached to disk BEFORE use, keyed by (model, prompt_hash) —
   never by prompt_hash alone, see rule 9. Never re-call the API for a
   (model, prompt) pair we have already sent.
4. All offline evaluation runs use the Batch API with prompt caching enabled.
5. All costs are derived from `configs/costs.yaml`. No hardcoded prices anywhere in
   the codebase.
6. Every experiment must exist in `PREREGISTRATION.md` with its accept rule written
   BEFORE the run is executed.
7. Negative and rejected results stay in the report. Never delete a failed experiment.
8. Every API-spending script must print an estimated cost and require an explicit
   `--confirm` flag before it sends anything. It must also REFUSE to run if
   cumulative recorded spend + this run's estimate exceeds `budget.hard_stop_usd`
   in `configs/costs.yaml`.
9. Token counts are per-model. Never compute a token count once and reuse it across
   models. Every model's token count comes from that model's own count_tokens call,
   or from the `usage` block of that model's own response. Cache token counts keyed
   by (model, prompt_hash), never by prompt_hash alone.
10. Cost is computed from the three usage fields separately: `input_tokens` (uncached),
   `cache_creation_input_tokens` (write), `cache_read_input_tokens` (read). Never sum
   them into a single input figure. Any function that ingests a usage block must assert
   that all three fields were read, and must fail loudly if a field is missing rather
   than defaulting it to zero.

## Budget

Total Claude API budget: $15, spent in stages with a gate between them.
Development iterations go to free-tier providers (Gemini / Groq), never to Claude.
Only final locked runs use Claude.

## Repository layout

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
