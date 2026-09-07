# Preregistration

> Hard rule 6: every experiment must appear in this file, **with its accept rule
> written, before the run is executed**. Hard rule 7: rejected results are never
> deleted — they move to the Rejected experiments section and stay there.

Status vocabulary: `planned` → `running` → `accepted` | `rejected` | `abandoned`.

---

## 1. Primary metric

- **Primary:** _(TODO — fill in)_
- **Secondary:** _(TODO)_
- **Cost metric:** _(TODO — how a dollar figure is attached to a result)_
- **Uncertainty:** _(TODO — seeds, bootstrap procedure, what "±" means)_
- **Splits:** _(TODO — what is calibrated on dev, what is only ever touched on test)_

## 2. Hypotheses

| id | hypothesis | rationale | how it could be wrong |
|----|-----------|-----------|-----------------------|
| H1 | _(TODO)_ | _(TODO)_ | _(TODO)_ |
| H2 | _(TODO)_ | _(TODO)_ | _(TODO)_ |

## 3. Planned experiments

Fill in the accept rule **before** running. An experiment with a blank accept rule
must not be executed.

| id | description | accept rule | status |
|----|-------------|-------------|--------|
| E1 | _(TODO)_ | _(TODO)_ | planned |
| E2 | _(TODO)_ | _(TODO)_ | planned |
| E3 | _(TODO)_ | _(TODO)_ | planned |
| C1 | Calibration-set robustness check (see 3a) | _(TODO — fill in before running)_ | planned |
| C2 | Few-shot confidence anchoring probe (see 3b) | _(TODO — fill in before running)_ | planned |
| C4 | Exemplar class over-prediction probe (see 3d) | _(TODO — fill in before running)_ | planned |

### 3a. C1 — Calibration-set robustness check

`dev_2000` covers 99 of 100 classes: `Books` is too rare to earn a proportional seat.
This was **accepted deliberately**, not patched. The set exists to calibrate a scalar
threshold, whose precision depends on total N rather than per-class coverage, and a
≥1-per-class floor would distort the class distribution away from realistic traffic and
thereby bias the calibrated threshold. Proportional sampling is correct here because
deployment traffic is proportional.

The check: for the single canonical Tier 1 config, calibrate the router threshold on
`dev_2000` **and** independently on the full 10k validation split, and report whether
the two thresholds agree. Agreement vindicates `dev_2000`; divergence is itself a
finding. Costs no API spend — local Tier 1 inference time only.

- **Accept rule:** _(TODO — state the tolerance within which the two thresholds count
  as agreeing, before running)_
- **Reporting caveat:** macro-F1 computed on `dev_2000` is an average over **99**
  classes, not 100. State this wherever such a number appears.

### 3b. C2 — Few-shot confidence anchoring probe

The few-shot exemplars carry a **fixed** `"confidence": 0.9`. This is a deliberate
anchoring probe, not an oversight. A constant is the cleanest probe: varying the
demonstrated values would fabricate a calibration curve.

Because the anchor is only present in the few-shot prompt, the two runs we are already
doing form the comparison:

- **zero-shot** — no exemplars, therefore no anchoring. Router **R2 is evaluated
  primarily here**, cleanly.
- **few-shot** — exemplars fixed at 0.9, therefore anchored.

Comparing the distribution of returned `confidence` values across the two runs measures
the anchoring effect directly, at no extra cost.

- **Predicted direction, recorded BEFORE the run:** few-shot confidences cluster near
  0.9 more tightly than zero-shot confidences — i.e. lower variance and a mode at or
  near 0.9.
- **Accept rule:** _(TODO — state the statistic and threshold, before running)_

### 3d. C4 — Exemplar class over-prediction probe

The frozen `exemplars_8` set is a uniform draw from the train split, i.e. a genuine
draw from the class prior. A consequence of that draw is that it covers **6 distinct
classes over 8 exemplars, with one class appearing 3 times**. This is honest but not
neutral: it may bias the few-shot run toward that class.

As with C2, both runs happen anyway, so the check is free.

The check: compare the predicted-label distribution between the zero-shot and few-shot
runs. Specifically, test whether the class appearing 3× in the exemplar set is
over-predicted in few-shot relative to zero-shot.

- **Predicted direction, recorded BEFORE the run:** yes — it will be over-predicted.
- **Accept rule:** _(TODO — state the statistic and threshold, before running)_

### 3e. Recorded failure class — library defaults that are wrong for this project

Recorded **before** any paid run, so it is on the record that these were found by
audit rather than discovered in a run we had paid for.

All three bugs found so far share one shape: **a library or convention default that is
reasonable in general and wrong for us, which degrades to a plausible number instead of
raising.** None would have thrown an error. Each would have produced a figure that
looked ordinary in a results table.

| # | Default | Reasonable in general because | Wrong for us because | Found |
|---|---------|-------------------------------|----------------------|-------|
| 1 | `chars/4` token estimate as a tokenizer fallback | a rough token count is usually fine for capacity planning | it was ~30% off — the same magnitude as the Claude 4.7+ tokenizer effect this project exists to measure | Block 2 EDA |
| 2 | `re.compile(r"\{.*\}", DOTALL)` to extract JSON from model output | greedy brace matching works when there is exactly one object | reasoning models emit DRAFT objects inside `<think>`; first-object parsing returns a confidently wrong label rather than an error | Rung 0, qwen3.6-27b |
| 3 | `sklearn` `f1_score(zero_division=0)` | scoring an undefined class as 0.0 avoids a crash | it silently deflates macro-F1 in proportion to absent classes — worst on the long tail, which is exactly where the cascade must be measured | Block 4 audit |

Consequence: **hard rule 11** (no silent degradation) was written after #1 and has since
caught #2 and #3. The standing mitigation is that any library default touching a
measured or billed quantity must be explicitly reviewed, not inherited — and where a
degraded value is legitimately wanted, the caller must ask for it by name
(`allow_absent_classes=True`, `assume_cache_hits=False`).

### 3c. Standing methodological limitations

- **Exemplar selection policy was not ablated due to budget constraints.** Few-shot
  results are conditional on proportional (uniform-over-rows) sampling from the train
  split at seed **20260907**, frozen in `configs/manifests/exemplars_8.json`
  (sha256 `ac7e7be88613…`). One-per-distinct-class was considered and rejected: at
  N=8 it covers 8% of the label space, which does not achieve label coverage and so
  cannot justify distorting the class prior. The full label list in the system prompt
  already conveys the taxonomy; the exemplars' job is to demonstrate output format.
- **Sampling temperature is not settable.** `anthropic` SDK 1.4.0 removes
  `temperature` and `top_p` from `Messages.create` entirely — they are absent from the
  signature, not defaulted. Runs therefore cannot be pinned to `temperature=0`, and
  run-to-run variance must be MEASURED across seeds (hard rule 2) rather than assumed
  away. Discovered at Rung 1, 2026-09-07.
- **Structured output was available and deliberately not used.** SDK 1.4.0 exposes
  `output_config.format` (JSON schema enforcement). Adopting it would drive the format
  failure rate to ~0 by construction — but format failure rate is a *measured* quantity
  in this project, and enforcing it away would delete the finding. Revisit only as an
  explicit, preregistered ablation.
- A 4-exemplar run uses the **first 4** of the frozen 8, so it is a strict subset of
  the 8-exemplar run and the two remain comparable.

## 4. Results log

Append-only. One entry per executed run, written after the run, referencing its
experiment id. Never edit a past entry — add a correcting entry instead.

### _(template — copy per run)_

- **Experiment id:** _(TODO)_
- **Date:** _(TODO)_
- **Commit:** _(TODO)_
- **Config / command:** _(TODO)_
- **Seeds:** _(TODO)_
- **Result:** _(TODO — mean ± std, primary metric first)_
- **Cost:** _(TODO — actual, from configs/costs.yaml)_
- **FX (only if any figure is INR-derived):** _(state the usd_per_inr_rate, its
  verification date, and its source — every INR-derived number in the report must
  carry the rate and date it was converted at)_
- **Manifest:** _(name + text_sha256, from configs/manifests/)_
- **Accept rule met?** _(yes / no)_
- **Notes:** _(TODO)_

## 5. Rejected experiments

Experiments whose accept rule was not met, or that were abandoned. These stay in the
final report.

| id | description | what happened | why rejected |
|----|-------------|---------------|--------------|
| _(none yet)_ | | | |
