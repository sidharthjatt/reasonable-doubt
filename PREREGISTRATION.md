# Preregistration

> Hard rule 6: every experiment must appear in this file, **with its accept rule
> written, before the run is executed**. Hard rule 7: rejected results are never
> deleted — they move to the Rejected experiments section and stay there.

Status vocabulary: `planned` → `running` → `accepted` | `rejected` | `abandoned`.

---

## 1. Primary metric

> **DRAFT — first pass by Claude, 2026-09-07. Every number below is a proposal with
> its reasoning shown; the human owner edits and owns the final version.**

- **Primary:** **macro-F1 over the 100 LEDGAR classes**, measured on `test_3000`.
  Averaged over classes present in gold; where a class is absent, `allow_absent_classes`
  must be set explicitly and the count of averaged classes reported alongside the score
  (see hard rule 11 and the `sklearn zero_division` bug in §3e).
- **Secondary:** accuracy; **format failure rate** (outputs not mapping to a canonical
  label, counted as wrong, never dropped); **truncation rate** reported separately from
  classification error; **escalation rate** per tier.
- **Cost metric:** **USD per 1,000 classified clauses**, all rates from
  `configs/costs.yaml` (hard rule 5). API tiers from the four usage fields separately
  (hard rule 10); local tiers from amortised hardware cost. **BLOCKED — see §1a.**
- **Uncertainty:**
  - Every model result is **mean ± std over ≥ 3 seeds** (hard rule 2).
  - **Absolute** macro-F1 carries the row-draw noise floor below.
  - **Model-vs-model on identical rows** uses a **paired bootstrap**: resample the rows
    with replacement, recompute both models' macro-F1 on each resample, report the
    distribution of the *difference*. The row draw is common to both models and largely
    cancels, so applying the absolute floor to a paired comparison overstates its
    uncertainty.
- **Splits — four disjoint roles, enforced in code by `src/train/splits.py`:**

  | split | role | may be used to |
  |-------|------|----------------|
  | `train` minus holdout (57k) | fitting | update weights |
  | `train_holdout_3000` | selection | pick epoch, hyperparameters, loss arm, signal, `max_length` |
  | `dev_2000` | calibration | fit router thresholds ONLY (hard rule 1) |
  | `test_3000` | reporting | be touched once, for the reported number |

### 1a. Measured noise floors — an accept rule inside these is not an accept rule

Monte Carlo over per-class binomial sampling at the observed performance level:

| split | classes | macro-F1 sd | 95% interval | unpaired diff needed |
|-------|---------|-------------|--------------|----------------------|
| `test_3000` | 100 | **±0.0145** | ±0.028 | **≈0.040** |
| `dev_2000` | 99 | ±0.0154 | ±0.030 | ≈0.043 |

The floor is driven by the long tail: `test_3000` has classes with 1, 2 and 4 rows,
each contributing 1% of macro-F1 from that many draws.

**Consequences that constrain every accept rule below:**

- An **unpaired** macro-F1 difference smaller than **~0.04** on `test_3000` is not
  distinguishable from row-draw noise. Rules asserting less than that must be paired,
  or they are unfalsifiable as written.
- A **paired** difference can be much smaller, but its threshold cannot be set until
  the paired bootstrap has been run once — see **C3**.
- **Seed variance is not yet measured.** Hard rule 2 gives mean ± std over ≥3 seeds,
  but we do not know the std for this model on this data. **Any accept rule with a
  margin under ~0.02 is provisional until C3 reports it.** This is flagged, not hidden.

### 1b. BLOCKED — the cost axis cannot currently be computed

`configs/costs.yaml` has `local_hardware.device_cost_usd: null`, so amortised cost per
request for Tier 0 and Tier 1 is **uncomputable**, and with it the x-axis of the Pareto
frontier. Also null: `measured_throughput_rps`, every `per_tier_throughput` entry, and
`fx.usd_per_inr_rate` is set but `device_cost_usd` is not derived from it.

**E6 cannot be evaluated until these are filled and throughput is measured.** No
latency-measurement harness exists yet either. Recorded here so it is not discovered
at reporting time.

## 2. Hypotheses

> **DRAFT.**

| id | hypothesis | rationale | how it could be wrong |
|----|-----------|-----------|-----------------------|
| H1 | A 3-tier cascade reaches macro-F1 within 0.04 of Sonnet-5-alone on `test_3000` at **under half** the USD/1k-clause cost. | Tier 0 answers head-class clauses in milliseconds at near-zero marginal cost; the top 10 classes are 31.5% of the corpus. | Escalation is driven by rare classes, which are also where Tier 0 is weakest, so the router escalates most of the tail and saves little. Or Tier 0's errors are confident, so the router does not catch them. |
| H2 | Tier 0's softmax **margin** is a better routing signal than LLM **verbalized confidence**, measured by AUROC for predicting own-correctness on `dev_2000`. | Verbalized confidence is severely compressed in every model measured: sd 0.031 (Gemini, non-reasoning), 0.083 (gpt-oss), 0.174 (Haiku), 0.184 (Sonnet), all with means 0.86–0.97 against accuracies 0.65–0.84. Haiku emitted **5 distinct values in 20 answers**. A continuous margin has no such plateaus. | The encoder is also badly calibrated, or its margin is compressed in a different way; or verbalized confidence, despite discretization, still ranks correctness well enough that AUROC is comparable. |
| H3 | Class weighting raises macro-F1 over unweighted CE at 137.7x train imbalance. | Unweighted CE optimises accuracy, which the head dominates; macro-F1 weights all 100 classes equally. | Weighting destabilises training or trades so much head accuracy that macro-F1 does not move beyond the noise floor. At `inv_freq` the weight ratio is 137x and may simply not converge. |

## 3. Planned experiments

> **DRAFT — proposed accept rules with reasoning. Edit and own before running.**
> An experiment with a blank accept rule must not be executed (hard rule 6).

| id | description | accept rule | status |
|----|-------------|-------------|--------|
| C3 | Measure seed-variance and paired-bootstrap floors | descriptive — no accept rule; **gates E2, E3, E7** | planned |
| E1 | Tier 0: DeBERTa-v3-base, CE baseline, 3 seeds | **macro-F1 ≥ 0.80 on `test_3000`, measured on the ONNX-INT8 artefact**, FP32 reported alongside. Anchored to LexGLUE Table 3 (DeBERTa m-F1 83.1) | planned |
| E2 | Tier 0 loss arms vs E1 (sqrt-inv-freq, effective-number, inv-freq) | best arm beats E1 by **≥ √2·1.96·seed_sd** (paired, same rows) — **NOT YET SETTABLE** | **[C3-gated]** |
| E3 | INT8 vs FP32 at the deployed precision | **\|INT8 − FP32\| ≤ 0.01** macro-F1, paired. FP32-as-headline prohibited | planned **[C3-gated]** |
| E4 | Tier 1: Qwen2.5-1.5B-Instruct LoRA, 3 seeds | macro-F1 **≥ E1 + 0.04**; else Tier 1 not justified → E4b | planned |
| E4b | Two-tier `Tier 0 → Claude` fallback | E6's rule with Tier 1 removed. **Registered before E4 runs** | registered |
| E5 | Routing signal: margin vs max-softmax vs entropy | best **AUROC ≥ 0.75** AND **≥ 0.05** above worst, on `dev_2000` only | planned |
| E6 | **Cost-vs-volume break-even curve** — the headline | (1) within **0.04** macro-F1 of Sonnet-5-alone; (2) finite crossover **V\* ≤ V_max**; (3) asymptote **< 50%** of Sonnet-5. **V\* is reported, not tested** | **BLOCKED by §1b** |
| E7 | `max_length` 256 vs 512 ablation | loses **< 0.02** macro-F1 paired AND **≥ 1.5×** faster; per-class deltas required | planned **[C3-gated]** |
| C1 | Calibration-set robustness (§3a) | thresholds agree within **1 decile** AND escalation within **3pp** | planned |
| C2 | Few-shot confidence anchoring (§3b) | **Sonnet 5 ONLY**: few-shot sd **≤ 0.5×** zero-shot sd AND mode within 0.02 of 0.9 | planned |
| C4 | Exemplar class over-prediction (§3d) | 3× class predicted at **≥ 2×** its zero-shot rate AND above every 1× class; **+2pp fallback if zero-shot rate < 1%** | planned |

### C3 — Noise-floor measurement *(no accept rule; this is instrumentation)*

- **Hypothesis:** none. This measures what the other rules need in order to be
  falsifiable at all.
- **Metric / split:** (i) macro-F1 std across ≥3 training seeds of E1 on
  `train_holdout_3000`; (ii) the paired-bootstrap distribution width for two models on
  identical rows.
- **Why it comes first:** §1a shows the *row-draw* floor is ±0.0145 on `test_3000`, but
  **seed variance is unmeasured**. If seed std turns out to be ±0.03, then E2's 0.04
  margin is barely one std and E3's 0.01 tolerance is meaningless. Numbers below marked
  **[C3-gated]** must be revisited once this reports.
- **Falsification:** n/a. If seed std exceeds ~0.03, several rules below are
  unfalsifiable as written and must be loosened or moved to paired comparisons.

### E1 — Tier 0 encoder baseline

- **Hypothesis:** A fine-tuned DeBERTa-v3-base classifies LEDGAR clauses well enough to
  serve as the cascade's first tier.
- **Metric / split:** macro-F1 on `test_3000`, **both precisions reported, INT8
  headline**; epoch chosen on `train_holdout_3000`. Mean ± std over seeds 1, 2, 3.
- **Accept rule:** **macro-F1 ≥ 0.80 on `test_3000`, measured on the ONNX-INT8
  artefact**, with the FP32 figure reported alongside it.
- **Why INT8 is the number the rule attaches to:** INT8 is what deploys. An FP32 accept
  would certify a system that is never served (see E3, and hard rule 11). If INT8 clears
  0.80 and FP32 does not, that is impossible in practice and indicates a measurement
  bug; if FP32 clears and INT8 does not, **E1 is NOT met** — the deployed tier failed,
  and E3's delta explains why.
- **Provenance of the number — ANCHORED, and a correction.** An earlier draft proposed
  0.70 and justified it with a vague appeal to "published baselines in the high 0.7s".
  **That was picked, not anchored, and it was wrong by ~13 points.** The actual figures,
  from LexGLUE Table 3 (Chalkidis et al., ACL 2022, `aclanthology.org/2022.acl-long.297`,
  test set, LEDGAR column), are:

  | model | µ-F1 (micro) | **m-F1 (macro)** |
  |-------|--------------|------------------|
  | TFIDF+SVM | 87.2 | 82.4 |
  | BERT | 87.6 | 81.8 |
  | RoBERTa | 87.9 | 82.3 |
  | **DeBERTa** | **88.2** | **83.1** |
  | Legal-BERT | 88.2 | 83.0 |
  | CaseLaw-BERT | 88.3 | 83.0 |

  Macro-F1 on LEDGAR is **~0.83**, not "materially below micro" — the gap is ~5 points,
  not ~15. **An accept rule at 0.70 would have been passed by a substantially broken
  model**, including one that loses the entire tail; even TFIDF+SVM scores 0.824.

  0.80 is therefore set at **~2 noise floors below the published DeBERTa result**
  (0.831 − 2×0.0145 ≈ 0.802), which allows for: `test_3000` being a 3,000-row
  proportional subset rather than the full 10k test set; our fitting on 57k rows rather
  than 60k after the selection holdout; and ordinary implementation variance. It tests
  **"we reproduced a known baseline"**, which is falsifiable, rather than a usefulness
  floor I invented.
  - Note in our favour: LexGLUE used `*-base` DeBERTa; we use **DeBERTa-v3**-base, which
    is generally stronger. If we land materially below 0.80 the likely cause is our
    training setup, not the task.
- **Falsification:** mean macro-F1 **< 0.80** across 3 seeds ⇒ Tier 0 as specified does
  not reproduce the published DeBERTa baseline on LEDGAR; report it and investigate the
  training setup before changing the encoder or dropping the tier. **Negative result
  stays in the report (hard rule 7).**

### E2 — Tier 0 class-imbalance arms

- **Hypothesis (H3):** Class weighting raises macro-F1 over unweighted CE.
- **Metric / split:** macro-F1 on `test_3000`, 3 seeds per arm. Arm *selection* happens
  on `train_holdout_3000`; only the selected arm is reported on test.
- **Accept rule: NOT YET SETTABLE. [C3-gated — blocking]** The margin is
  **√2 × 1.96 × seed_sd**, where `seed_sd` is the across-seed macro-F1 std measured by
  C3. It will be filled in once C3 reports, and before E2 runs.
- **Reasoning — corrected.** An earlier draft set 0.04, the *unpaired* floor. **That was
  the wrong variance component.** E1 and E2 are evaluated on the **same `test_3000`
  rows**, so row-draw variance is common to both and cancels — the same paired-vs-
  unpaired distinction applied to the free-tier comparison. The residual variance is
  across-seed, not across-rows.
  - Using 0.040 would preregister a **Type II error against the single largest lever on
    the primary metric**: a real 0.03 macro-F1 gain from class weighting — plausible at
    137.7× imbalance — would be declared not-accepted.
  - For scale: if `seed_sd` is 0.005, the margin is ~0.014; if it is 0.015, ~0.042. The
    rule cannot be honestly written before that number exists.
- **Falsification:** no arm clears the margin ⇒ H3 is not supported at this scale; report
  unweighted CE as the Tier 0 configuration and record that weighting did not help.
- **Note:** `inv_freq` at 137× may fail to converge. That is a result, not a bug, and is
  reported as such.

### E3 — INT8 vs FP32 *(non-negotiable: INT8 is what deploys)*

- **Hypothesis:** INT8 dynamic quantisation does not materially degrade Tier 0.
- **Metric / split:** macro-F1 of both precisions on `test_3000`, **paired on identical
  rows** — same model, same rows, so the row draw cancels entirely.
- **Accept rule:** **|INT8 − FP32| ≤ 0.01 macro-F1**, paired. **[C3-gated]**
- **Reasoning for the number:** this comparison is paired *and* same-model, so the
  ±0.0145 absolute floor does not apply; the only variance is quantisation-induced
  prediction flips. 0.01 is chosen as the point at which the deployed system's accuracy
  would need reporting as a materially different number.
- **Falsification:** delta > 0.01 ⇒ the deployed system is not the measured system.
  Report the INT8 number as the headline regardless, with the delta stated. **Reporting
  the FP32 figure as the system's accuracy is prohibited** — it describes something that
  does not exist.

### E4 — Tier 1 QLoRA

- **Hypothesis:** A QLoRA-tuned small decoder is enough better than Tier 0 to justify a
  middle tier.
- **Metric / split:** macro-F1 on `test_3000`, 3 seeds. Primary model
  **Qwen2.5-1.5B-Instruct**; `meta-llama/Llama-3.2-3B-Instruct` if access lands,
  otherwise **Qwen2.5-3B-Instruct** — whichever is used is recorded here before the run.
- **Accept rule:** **macro-F1 ≥ E1 + 0.04.**
- **Reasoning for the number:** a middle tier only earns its complexity and its serving
  cost if it is *detectably* better than the tier below. 0.04 is the unpaired floor from
  §1a — below it we cannot tell Tier 1 from Tier 0 on this test set.
- **Falsification:** Tier 1 within 0.04 of Tier 0 ⇒ **the middle tier is not justified.**
  Proceed to **E4b**, which is registered NOW so that reporting a two-tier cascade cannot
  read as a post-hoc rescue.

### E4b — Two-tier fallback architecture *(registered in advance of E4's outcome)*

- **Trigger:** E4 not met. Registered **before** E4 runs, so both outcomes are on the
  record before either is observed.
- **Hypothesis:** if Tier 1 does not clear Tier 0, the primary architecture is the
  **two-tier `Tier 0 → Claude` cascade**, and it still beats Claude-alone on cost at
  comparable accuracy.
- **Why this is the expected branch, not a remote one:** a discriminatively fine-tuned
  encoder with 57k in-domain labelled rows beating a 1.5B decoder given a LoRA adapter
  and the same data is a **common** outcome, not a surprising one. The encoder is
  trained directly on the objective; the decoder is adapted to emit label strings.
- **Metric / split:** macro-F1 and USD/1,000 clauses on `test_3000`, thresholds from
  `dev_2000`.
- **Accept rule:** the two-tier cascade reaches macro-F1 **within 0.04** of
  Sonnet-5-alone at **< 50%** of its cost — i.e. E6's rule with Tier 1 removed.
- **Falsification:** if the two-tier cascade also fails that bar, **the cascade premise
  fails** and the honest report is that routing did not pay for itself on this task.
  That result stays in the report (hard rule 7) and is the paper's finding.
- **Reporting requirement:** whichever branch is taken, **both E4 and E4b appear in the
  final report**, with E4's measured result shown even when it triggers E4b.

### E5 — Routing signal selection

- **Hypothesis (H2):** margin > max-softmax ≈ entropy as a router input.
- **Metric / split:** **AUROC for predicting own-correctness**, on `dev_2000` only
  (hard rule 1). All three signals from the same logits — one model, no extra compute.
- **Accept rule:** best signal **AUROC ≥ 0.75** AND **≥ 0.05 above the worst** of the
  three.
- **Reasoning for the numbers:** AUROC 0.5 is a coin flip and ~0.70 is the usual
  threshold for a signal being decision-useful; 0.75 demands the router actually
  separates its own errors. The 0.05 spread requirement exists because all three come
  from the same logits and may be near-equivalent — if they are, "margin is best" is not
  a finding and we should say so rather than pick a winner by noise.
- **Falsification:** all three below 0.75 ⇒ Tier 0 cannot self-assess and the router
  needs a different mechanism. Spread < 0.05 ⇒ signal choice does not matter; report
  that and use margin for its temperature-invariance property alone.

### E6 — Cascade cost-vs-volume break-even curve *(the headline)*

- **Hypothesis (H1):** the cascade reaches macro-F1 within 0.04 of Sonnet-5-alone, and
  there exists a **finite, realistic clause volume** beyond which it is cheaper.

- **The deliverable is a CURVE, not a point.** An earlier draft asked for a single
  Pareto point at one amortisation window. **That hides the assumption that decides the
  answer:** amortise a 59,900 INR Mac Mini over 1,000 clauses and local inference looks
  absurd; over 10,000,000 it is free. Same model, same accuracy, opposite conclusion.
  Leaving the window unstated means whoever picks it later picks the headline result.

  **Preregistered deliverable:** clause volume *V* on the x-axis (log scale), **USD per
  1,000 clauses** on the y-axis, **one curve per tier and one for the cascade**, with
  **no single amortisation window privileged.** The report states the crossover volume
  explicitly.

- **Curve definitions:**
  - API tiers are **flat in V**: cost/1k is a constant from the four usage fields.
  - Local tiers are **hyperbolic plus a floor**:
    `cost_per_1k(V) = 1000·device_cost_usd/V + 1000·energy_cost_per_clause`
  - **The energy term is what makes this honest.** Without it the local curve tends to
    zero and local always wins at sufficient volume, which is false. With it the curve
    **asymptotes to the per-clause energy cost**, and whether that asymptote sits above
    or below the API line decides whether local *ever* wins at any volume. That is the
    actual question.
  - The local curve is **bounded in V** by what the hardware can process in its life:
    `V_max = throughput_rps × 3600 × 24 × 365 × lifetime_years × duty_cycle`. Beyond
    `V_max` the curve **steps** — a second device — rather than continuing to fall.

- **Accept rule — attaches to the CROSSOVER, not to a point on the curve:**
  1. **Accuracy:** cascade macro-F1 within **0.04** of Sonnet-5-alone on `test_3000`
     (the unpaired floor from §1a — the strongest claim the test set supports); **and**
  2. **Crossover exists and is reachable:** a finite crossover volume *V\** exists at
     which the cascade becomes cheaper than Sonnet-5-alone, **and V\* ≤ V_max** — the
     device can physically process that many clauses within its life; **and**
  3. **Asymptote:** the cascade's high-volume cost/1k is **< 50%** of
     Sonnet-5-alone's — i.e. the saving survives after the hardware is fully amortised
     and only energy remains.
- **V\* IS THE REPORTED RESULT, NOT A TEST.** An earlier draft required *V\** ≤ 1,000,000
  clauses. **That number was picked, not anchored** — the same defect as E1's original
  0.70 — and no source fixes what volume a "realistic deployment" reaches; it depends
  entirely on the reader's situation. So the crossover volume is **reported as the
  finding** ("local overtakes Sonnet-5 at V\* = N clauses"), and the reader judges it
  against their own volume. Only conditions 1 and 3, which are anchored — the §1a noise
  floor and a materiality threshold — function as accept rules.
- **Reasoning for the anchored numbers:** 0.04 is the measured unpaired floor from §1a,
  the strongest accuracy claim `test_3000` supports. `V_max` is measured, not chosen: a
  crossover beyond it means **the device dies before it breaks even**, a clean negative.
  The 50% asymptote condition prevents an accept driven purely by amortisation
  arithmetic rather than by the cascade doing anything useful.
- **Falsification:** no finite crossover at or below `V_max`, or an asymptote above
  50% ⇒ the cascade does not pay for itself. **Report the full curve and V\* regardless** — a negative break-even result is exactly what measuring was for.

- **⚠ STILL BLOCKED, see §1b.** `device_cost_usd` is null (59,900 INR and an FX rate are
  recorded but the USD figure is not derived), `measured_throughput_rps` and every
  `per_tier_throughput` entry are null, and **`power_draw_watts` /
  `electricity_cost_usd_per_kwh` are null**.

- **What the curve changes about the measurement harness — answering the question
  directly:**
  1. **`power_draw_watts` is promoted from optional to MANDATORY.** In the old
     single-point framing energy was a rounding error. In the curve framing it *is* the
     asymptote, and the asymptote decides condition 3. A curve without it is wrong in
     the exact region the hypothesis is about.
  2. **Throughput is needed for a second reason.** Previously it was only an input to
     amortisation; now it also sets `V_max`, the point where the local curve stops
     falling and steps. A crossover past `V_max` is a negative result the harness must
     be able to detect.
  3. **Per-tier, not aggregate.** Tier 0 (ONNX INT8) and Tier 1 (MLX) have different
     throughput and different power draw, and the cascade's curve is a routing-weighted
     mixture of them. The harness must measure each tier separately.
  4. **Latency (p50/p95) is now secondary for E6** — it does not enter the cost curve at
     all. It remains a reported quality of Tier 0 ("milliseconds") and is what E7 tests,
     but it is no longer on the critical path for the headline result. **Throughput and
     power are.**

### E7 — `max_length` latency ablation

- **Hypothesis:** 256 tokens is enough for Tier 0, at materially lower latency.
- **Metric / split:** macro-F1 on `test_3000` (paired, same rows) and measured p50/p95
  latency locally.
- **Accept rule:** 256 loses **< 0.02 macro-F1 paired** AND is **≥ 1.5× faster** at p50.
- **Reasoning for the numbers:** only **1.2–1.6%** of clauses exceed 512 tokens
  (measured, `results/data_report.md`), and the p50 clause is ~100 tokens, so most rows
  are unaffected either way; 0.02 allows for the long-clause tail being systematically
  hurt. 1.5× is the smallest speedup that would change a deployment decision.
- **Falsification:** loss ≥ 0.02 or speedup < 1.5× ⇒ keep 512.
- **Note:** truncation at 256 will hit long clauses, which may concentrate in particular
  classes. **Report per-class deltas, not just the aggregate** — a uniform-looking 0.02
  could be one class collapsing.

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

- **Accept rule (DRAFT):** the two thresholds agree within **1 decile of the routing
  signal's distribution** on the full 10k validation split, AND the resulting escalation
  rates differ by **≤ 3 percentage points**.
- **Reasoning for the numbers:** a threshold is only meaningful through the escalation
  rate it produces, so the rule is stated in both units — a threshold difference that
  moves escalation by <3pp is operationally the same threshold. One decile is chosen
  because the signal is calibrated by rank, not by absolute value, so a rank-based
  tolerance survives any monotone rescaling (including temperature).
- **Falsification:** thresholds differ by more than a decile, or escalation by more than
  3pp ⇒ `dev_2000` is too small to calibrate on and the full 10k validation split should
  be used instead. That is a finding about our own method and stays in the report.
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
- **Accept rule (DRAFT):** few-shot verbalized-confidence **sd ≤ 0.5× zero-shot sd**
  for the same model on the same rows, AND the few-shot **mode within 0.02 of 0.9**
  (the anchored value).
- **Reasoning for the numbers:** confidence is already severely compressed with no
  anchor at all — measured zero-shot sd is 0.031 (Gemini), 0.083 (gpt-oss), 0.174
  (Haiku), 0.184 (Sonnet). A halving is chosen as the smallest effect that is clearly
  not the run-to-run wobble of an already-narrow distribution. The mode condition tests
  the anchor *directly* rather than inferring it from spread.
- **SCOPE: this accept rule applies to Sonnet 5 ONLY.** Sonnet 5's zero-shot sd of
  0.184 is the widest measured and leaves room for a halving to be visible. The other
  models are reported descriptively and are **not** subject to the rule.
- **Why the others are excluded, stated in advance:** for a model whose zero-shot sd is
  already 0.031 (Gemini 3.5-flash-lite), a halving to 0.015 is below the granularity of
  the values these models actually emit — Haiku used **5 distinct values in 20 answers**,
  Gemini **6 in 32**. On such models the mode-near-0.9 test **measures emission
  granularity, not anchoring**: a model that emits {0.88, 0.92, 0.95, 0.98, 0.99, 1.0}
  cannot place a mode at 0.9 whether it is anchored or not. **A null result on the
  compressed models is therefore NOT evidence against anchoring** and must not be
  reported as such.
- **Falsification:** sd ratio > 0.5 and mode away from 0.9 ⇒ the fixed exemplar
  confidence did not anchor, and verbalized confidence is compressed for reasons
  intrinsic to the models rather than to our prompt.

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
- **Accept rule (DRAFT):** the class appearing 3× in `exemplars_8` is predicted in
  few-shot at **≥ 2× its zero-shot prediction rate** on the same rows, AND its shift is
  larger than that of any class appearing exactly once in the exemplar set.
- **Reasoning for the numbers:** prediction rates for a mid-frequency class over 3,000
  rows have a standard error near 1 percentage point, so a doubling sits far outside
  noise while a 20–30% shift would not. The second clause guards against a general
  few-shot drift being misread as exemplar-specific anchoring — if every exemplar class
  rises equally, the cause is few-shot prompting, not repetition.
- **⚠ Rate fallback, registered in advance:** if the 3× class's **zero-shot** prediction
  rate is under **1%**, a ratio to a near-zero baseline is unstable and the rule switches
  to an absolute form: **+2 percentage points** over its zero-shot rate. The switch is
  decided by the measured zero-shot rate and **must be recorded before the few-shot run
  is scored**, never after seeing the few-shot number.
- **Falsification:** shift < 2× (or < 2pp under the fallback), or not exceeding the
  single-exemplar classes ⇒ exemplar repetition did not bias predictions, and the
  uniform-draw exemplar policy carries less risk than §3c records.

### 3e. Recorded failure class — failures that present as normal operation

Recorded **before** any paid run, so it is on the record that these were found by
audit rather than discovered in a run we had paid for.

All five bugs found so far share one shape: **a failure that does not surface as a
failure.** None threw an error. Three degraded a measured quantity to a plausible
number; the fourth degraded a control-flow state to a plausible status. In every case
the system continued to look like it was working.

The first three are library or convention defaults that are reasonable in general and
wrong for this project. The fourth is our own control flow, which shows the class is
broader than "bad defaults" — any construct that can only report success, or that
treats an unrecognised state as a normal one, belongs here.

| # | Default | Reasonable in general because | Wrong for us because | Found |
|---|---------|-------------------------------|----------------------|-------|
| 1 | `chars/4` token estimate as a tokenizer fallback | a rough token count is usually fine for capacity planning | it was ~30% off — the same magnitude as the Claude 4.7+ tokenizer effect this project exists to measure | Block 2 EDA |
| 2 | `re.compile(r"\{.*\}", DOTALL)` to extract JSON from model output | greedy brace matching works when there is exactly one object | reasoning models emit DRAFT objects inside `<think>`; first-object parsing returns a confidently wrong label rather than an error | Rung 0, qwen3.6-27b |
| 3 | `sklearn` `f1_score(zero_division=0)` | scoring an undefined class as 0.0 avoids a crash | it silently deflates macro-F1 in proportion to absent classes — worst on the long tail, which is exactly where the cascade must be measured | Block 4 audit |
| 4 | `until grep -q "GATE" log; do sleep; done` — a wait loop with a success condition and no failure condition | polling for a completion marker is the obvious way to wait | the job it watched crashed with a traceback and never wrote `GATE`, so a dead job displayed as **Running for 2.5 hours**. A stale job showing Running is worse than no indicator: it hides the next real stall | Block 4, qwen run |

| 5 | `str.replace(old, new)` in an editing script, with no check that `old` matched | a replace that finds nothing is normally harmless | it silently did nothing, so **C4's accept rule was left holding C2's text** — a preregistered rule for the wrong experiment, which would have been signed off as correct. Found only by reading the diff | Block 4, drafting this file |

Instance 5 is the same defect as instance 4 in a different costume, and was committed
*while writing the section that describes the class*: an operation with a success path
and no failure path. The mitigation is identical — `assert old in s` before every
replace, so a non-matching edit raises instead of passing. **Read the diff, not the
exit code.**

Instance 4 also generalises the mitigation: a wait must have a failure condition as
well as a success condition — process liveness, a timeout, or an error marker — or it
cannot distinguish "still working" from "died". This is the same defect as
`batch.poll()` treating an unrecognised `processing_status` as "still running", found
in the same audit.

Consequence: **hard rule 11** (no silent degradation) was written after #1 and has since
caught #2, #3, #4 and #5.

**Practice note — two readings before declaring a limit.** Two claims have been
retracted, and both had the same shape: *a structural fact inferred from a single
observation.* (i) The `anthropic` SDK signature lacking `temperature` was taken to mean
the API rejects it; the API in fact validates and honours it. (ii) One reading of
`x-ratelimit-remaining-requests` was taken to mean a hard daily cap; a second reading
showed the counter refilling, i.e. a rolling window. Standing practice: **before
declaring any limit, quota, or capability, take at least two readings separated in time
and show the delta** — and include a control that is known to behave the opposite way
(the `definitely_not_a_real_param` → HTTP 400 probe is what made the temperature result
conclusive rather than merely suggestive). One sample describes a moment, not a rule. The standing mitigation is that any library default touching a
measured or billed quantity must be explicitly reviewed, not inherited — and where a
degraded value is legitimately wanted, the caller must ask for it by name
(`allow_absent_classes=True`, `assume_cache_hits=False`).

### 3f. Amendment — Stage 1 per-model output budgets (2026-09-07)

**Decision: per-model output budgets, Haiku 128 / Sonnet 48. `max_usd` NOT amended.**
Uncached gating figure $6.7981, against `stage_1.max_usd` of $7.00.

Measured output at Rung 1, 20 rows per model:

| model | mean out | max out | truncations |
|-------|----------|---------|-------------|
| Haiku 4.5 | 25.2 | 64 | 1 |
| Sonnet 5 | 22.1 | 25 | 0 |

**The reasoning, stated carefully.** An earlier draft of this entry justified raising
Haiku on the width of the interval around 1/20 while justifying cutting Sonnet on 0/20.
That is an asymmetry: by the rule of three, 0/20 carries a 95% upper bound near 15%, so
it is the same weak evidence pointed the other way. It cannot support both moves.

The decision rests instead on **the character of the observed failure, not its rate.**
Haiku's single truncation was not budget starvation at the margin. The answer itself
needs ~25 tokens. What happened was a self-correction excursion into prose: Haiku
emitted a fenced answer carrying an invented label, then began reasoning aloud about
its own mistake. An excursion of that kind overruns 48 and 64 alike; only a budget of
128 or more gives it room to land on an answer.

Two things follow. First, the money is worth spending where the excursion actually
happens — Haiku — because that is the only place a larger budget changes an outcome.
Second, and for the same reason, the 48-vs-64 choice on Sonnet is close to irrelevant
to this failure mode: neither value would contain such an excursion, so the difference
between them buys nothing against it. Sonnet's budget was therefore set by what frees
the headroom Haiku needs while staying clear of Sonnet's observed output length.

| configuration | uncached gating | vs $7.00 |
|---------------|-----------------|----------|
| Haiku 64, Sonnet 64 (previous) | $6.5581 | clears |
| Haiku 128, Sonnet 64 | $7.0381 | breaches by $0.04 |
| **Haiku 128, Sonnet 48 (adopted)** | **$6.7981** | **clears, $0.20 spare** |
| Haiku 128, Sonnet 32 | $6.5581 | clears |

**Residual risk, stated plainly.** If Sonnet ever produces an answer requiring 48–64
output tokens, that row is truncated and lost. **Nothing in 20 rows rules this out** —
0/20 truncations at a 64-token budget is compatible with a true overrun rate as high as
~15%, and the observed maximum of 25 tokens is a maximum over 20 draws, not a ceiling.
This is a risk accepted to fund Haiku's budget, not a risk measured away. Stage 1 must
report Sonnet's truncation count at 48; if it is non-zero, this amendment was wrong and
the entry stays in the record either way (hard rule 7).

**Also rejected:** raising `max_usd` — loosening a budget guard to fit a change is the
wrong direction, and proved unnecessary. Haiku at some value between 64 and 128 — no
measurement supports any particular intermediate.

### 3g. Retired — `test_stratified_764` (2026-09-07)

**Status: retired before it was built. Not rejected on its merits.**

A class-stratified subset of `test_3000` (~8 rows/class, all 100 classes, 764 rows)
was designed to enable a free-tier-vs-Claude comparison **on identical rows**, where
the row draw is common to both models and the class-mix bias cancels in the difference.
Its two justifications were measured and stand:

1. **Class coverage.** Proportional sampling at 400 rows drops ~30 classes to zero,
   making macro-F1 over 100 classes undefined rather than merely noisy.
2. **Mix bias is negligible.** Reweighting the confusion matrix from 230 cached
   predictions gave a proportional-vs-stratified macro-F1 difference of **-0.0010**.
   Recall is mix-invariant by construction; the whole effect is precision, and it is
   ~10^-3.

Neither justification referenced provider quota. It is retired **because the
free-tier Pareto point it enabled has itself been dropped** — both free providers
throttled too hard to produce a baseline with usable n (Groq ~6-7 min/row after ~200
rows; Gemini 118/150 requests rate-limited in a sustained probe).

**If a free-tier baseline is ever revived, rebuild this manifest.** The reasons above
are why, and they will still hold: the comparison needs identical rows and full class
coverage, and the mix bias is small enough to ignore.

### 3c. Standing methodological limitations

- **Exemplar selection policy was not ablated due to budget constraints.** Few-shot
  results are conditional on proportional (uniform-over-rows) sampling from the train
  split at seed **20260907**, frozen in `configs/manifests/exemplars_8.json`
  (sha256 `ac7e7be88613…`). One-per-distinct-class was considered and rejected: at
  N=8 it covers 8% of the label space, which does not achieve label coverage and so
  cannot justify distorting the class prior. The full label list in the system prompt
  already conveys the taxonomy; the exemplars' job is to demonstrate output format.
- **Sampling temperature IS pinnable, via `extra_body`.** *(Corrected 2026-09-07; an
  earlier version of this entry claimed the opposite and was wrong.)* `anthropic` SDK
  1.4.0 removes `temperature` from the `Messages.create` signature, so passing it as a
  kwarg raises `TypeError` **client-side, before any HTTP request**. That is an SDK
  fact, not an API fact. The API still accepts and validates it:

  | request | result |
  |---------|--------|
  | `temperature=0.0` | HTTP 200 |
  | `temperature=2.0` | HTTP 400 — `temperature: range: 0..1` |
  | `temperature=-1.0` | HTTP 400 — `` `temperature` cannot be set to -1 for this model`` |
  | `temperature="hot"` | HTTP 400 — `temperature: Input should be a valid number` |
  | `definitely_not_a_real_param=123` (control) | HTTP 400 — `Extra inputs are not permitted` |

  The control line is what makes this conclusive: unknown fields are rejected, so a
  200 on `temperature=0.0` plus range- and type-validation means the parameter is
  parsed and honoured, not silently discarded. All runs therefore pin
  `temperature=0.0` through `extra_body`, and `temperature` stays in the cache key.
  Seed-based variance measurement under hard rule 2 remains required regardless —
  temperature 0 is not a determinism guarantee.
- **Structured output was available and deliberately not used.** SDK 1.4.0 exposes
  `output_config.format` (JSON schema enforcement). Adopting it would drive the format
  failure rate to ~0 by construction — but format failure rate is a *measured* quantity
  in this project, and enforcing it away would delete the finding. Revisit only as an
  explicit, preregistered ablation.
- A 4-exemplar run uses the **first 4** of the frozen 8, so it is a strict subset of
  the 8-exemplar run and the two remain comparable.

## 3h. Whole-file verification checklist

Run before signing, and again before the final report. Every row is checkable by
reading the file; **`match` is filled in by a human, not asserted here.**

| # | section | expected | present? | match? |
|---|---------|----------|----------|--------|
| 1 | §1 Primary | macro-F1 named primary; accuracy, format-failure, truncation, escalation named secondary | ✅ | |
| 2 | §1 Cost | USD/1,000 clauses; all rates from `configs/costs.yaml` (hard rule 5) | ✅ | |
| 3 | §1 Uncertainty | ≥3 seeds mean±std (hard rule 2); paired bootstrap for same-row comparisons | ✅ | |
| 4 | §1 Splits | four disjoint roles; enforced by `src/train/splits.py`, tested in `tests/test_splits.py` | ✅ | |
| 5 | §1a | noise floors measured BEFORE any accept rule was written | ✅ ±0.0145 / ±0.0154 | |
| 6 | §1b | cost-axis blockers listed with the specific null fields | ✅ | |
| 7 | §2 | H1, H2, H3 each with a rationale and a way to be wrong | ✅ | |
| 8 | E1 | anchored to a cited source (LexGLUE Table 3); INT8 headline | ✅ 0.80 | |
| 9 | E2 | margin uses the paired/seed component, not the unpaired floor | ✅ [C3-gated] | |
| 10 | E3 | INT8 vs FP32 paired; FP32-as-headline explicitly prohibited | ✅ ≤0.01 | |
| 11 | E4 | Tier 1 model recorded before the run; Llama-vs-Qwen substitution rule stated | ✅ | |
| 12 | E4b | two-tier fallback registered BEFORE E4 runs | ✅ | |
| 13 | E5 | dev-only (hard rule 1); all three signals from the same logits | ✅ | |
| 14 | E6 | curve is the deliverable; V\* reported not tested; power mandatory | ✅ | |
| 15 | E7 | per-class deltas required, not just the aggregate | ✅ | |
| 16 | C1 | tolerance stated in BOTH threshold and escalation-rate units | ✅ | |
| 17 | C2 | scoped to Sonnet 5; null on compressed models ≠ evidence against anchoring | ✅ | |
| 18 | C3 | gates every rule with a margin under 0.02; no accept rule of its own | ✅ | |
| 19 | C4 | direction predicted in advance; <1% rate fallback registered in advance | ✅ | |
| 20 | §3e | five instances; class stated as "failures that present as normal operation" | ✅ | |
| 21 | §3f | Stage 1 output budgets; Sonnet residual risk stated as unmeasured | ✅ | |
| 22 | §3g | `test_stratified_764` retired for the correct reason, rebuild trigger stated | ✅ | |
| 23 | all | every accept rule has a falsification condition | ✅ | |
| 24 | all | every number is anchored, or explicitly flagged as picked | ✅ | |
| 25 | hard rule 7 | negative/rejected results stay in the report | ✅ E1, E2, E4→E4b, E6 | |

**Known-unsettable at signing time, by design:**

| rule | why it cannot be filled yet | unblocked by |
|------|----------------------------|--------------|
| E2 margin | `seed_sd` unmeasured | C3 |
| E3 / E7 tolerances | provisional under 0.02 | C3 |
| E6 conditions 2 and 3 | `device_cost_usd`, throughput, power all null | §1b + the Mac Mini harness |

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
