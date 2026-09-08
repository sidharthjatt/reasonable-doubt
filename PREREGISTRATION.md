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

### 1b. The cost axis — capital and volume, not energy *(rewritten 2026-09-08)*

An earlier version of this section, and E6 with it, treated **energy as the asymptote
that decides whether local inference ever wins**. **Measurement falsified that.**

Measured on the Mac Mini, ONNX-INT8, DeBERTa-v3-base architecture:

| quantity | value |
|----------|-------|
| throughput | **28.97 ± 0.24 req/s** (4 runs, bs=1, max_length 512) |
| SoC power under load | 16.0 W *(package only — see caveat)* |
| energy | 0.526 J / request |

```
1000 clauses / 28.97 req/s        = 34.5 s
16.0 W x 34.5 s                   = 552 J = 0.000153 kWh
at $0.085/kWh                     = $0.0000130 per 1000 clauses
Sonnet 5 (batch, cached)          = $0.4483   per 1000 clauses
                          ratio   ~ 1 : 34,000
```

**Energy is ~1/34,000 of the API line.** The local curve's asymptote is effectively
zero, and **capital dominates entirely**. Even a 10x correction from SoC to wall power
leaves it ~1/3,400 — the conclusion is insensitive to the measurement caveat below.

**E6 therefore turns on capital treatment and volume.** Both cases are reported; neither
is privileged, which removes the researcher degree of freedom that picking an
amortisation window would have introduced:

| case | assumption | local cost/1k | result |
|------|-----------|---------------|--------|
| **Greenfield** | hardware bought for this workload; capital attributable | $633.86 / V per 1k | **V\* ≈ 1.41M clauses** for Tier 0 alone, higher for the cascade by 1/(1 − escalation_rate) |
| **Sunk capital** | the Mac Mini already exists (it does) | ~$0.000013 per 1k | **local wins from the first clause** |

**Both are correct under their own assumption, and the honest report gives both.** The
greenfield number carries a hard consequence worth stating plainly: **LEDGAR is ~80,000
clauses in total, so at realistic contract-review volume greenfield local hardware never
breaks even** — V\* is ~18x the entire corpus. A reader deploying on hardware they
already own reaches the opposite conclusion. The volume at which those two answers
swap is the finding.

`V_max` (what the device can process in its life) is **685,230,000** clauses at 3 years
and 25% duty — far above V\*, so under the greenfield case the device does not die
before break-even; it simply needs ~18 corpora of work to get there.

**Power measurement caveat, stated precisely.** `powermetrics --samplers cpu_power`
reports "Combined Power (CPU + GPU + ANE)" — **SoC package power only**. It excludes
RAM, SSD, PSU losses, networking and fans, so it is **not wall power**, and a machine
reporting 0.2 W idle here draws several watts at the wall. Wall power cannot be
obtained from `powermetrics` at all; it needs an external meter. Fields are named
`*_soc_watts` so no report can claim more than was measured. Given the 34,000x margin
this does not change any conclusion, but the number must be what it says it is.

**Still outstanding for E6:**

| input | status |
|-------|--------|
| Tier 0 throughput | **measured** 28.97 ± 0.24 req/s |
| `device_cost_usd` | **derived** $633.86 (59,900 INR ÷ 94.50, FX dated 2026-09-07) |
| Tier 0 SoC power | first reading taken with a defective harness; **re-measure** |
| Tier 0 wall power | not obtainable without a meter; **optional given the margin** |
| Tier 1 throughput | **not measured** — needs the trained adapter under MLX |
| escalation rate | from the router sweep, simulated offline at zero cost |

## 2. Hypotheses## 2. Hypotheses

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
- **Training configuration:** registered in §3t (effective batch 16, 1 visible GPU,
  3 epochs, lr 2e-5, max_length 512, seeds 1/2/3).
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
- **Training configuration:** registered in §3t (effective batch 16 as BS 4 x GA 4,
  1 visible GPU, 1 epoch, lr 2e-4, max_length 2560, LoRA r=16, seeds 1/2/3).
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
  2. **Crossover, reported under BOTH capital treatments** (see §1b): greenfield
     (capital attributable, V\* finite and ≤ V_max) and sunk-capital (marginal cost
     only). Neither is privileged; both curves appear in the report; **and**
  3. **Asymptote:** the cascade's high-volume cost/1k is **< 50%** of
     Sonnet-5-alone's. Measurement has made this nearly automatic — energy is ~1/34,000
     of the API line — so condition 3 is now a **sanity check, not a discriminator**,
     and E6 rests on conditions 1 and 2.
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
**and, where a claim spans models or endpoints, one reading per model** — and show the
delta** — and include a control that is known to behave the opposite way
(the `definitely_not_a_real_param` → HTTP 400 probe is what made the temperature result
conclusive rather than merely suggestive). One sample describes a moment, not a rule.

**Third retraction, a second axis (2026-09-07).** `temperature` was validated on
Haiku 4.5 — range-checked, type-checked, with an unknown-field control — and the
conclusion generalised to Sonnet 5. It does not hold: **Sonnet 5 returns
`` `temperature` is deprecated for this model``**. Rung 1 passed only because Haiku
ran first. The note above said "two readings **over time**"; this failure was one
reading **across models**. Both axes now apply:

| axis | failure | rule |
|------|---------|------|
| time | one counter reading read as a quota model | two readings, show the delta |
| **model / endpoint** | **a parameter validated on one model assumed for another** | **one reading per model, and per endpoint — sync and batch are different paths** |

A parameter validated on one model is **not** validated on another. The standing mitigation is that any library default touching a
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

### 3j. FINDING — the cheaper model cannot cache, inverting its cost advantage

Measured at Rung 2, then confirmed against the published per-model minimums
(`platform.claude.com/docs/en/build-with-claude/prompt-caching`, retrieved 2026-09-07):

| model | documented minimum cacheable prefix | our prefix | result |
|-------|-----------------------------------|-----------|--------|
| Claude Sonnet 5 | **1,024** | 1,084 | cacheable, clears by **60 tokens** |
| Claude Haiku 4.5 | **4,096** | 793 | **cannot cache** — short by 3,303 |

Rung 2 observation, same batch, same 1h TTL, 10 requests each:

```
Haiku : input 9,431 | cache_write     0 | cache_read     0   hit rate  0.0%
Sonnet: input 2,307 | cache_write 1,080 | cache_read 9,720   hit rate 74.2%
```

Haiku 4.5's floor is **four times** our prefix, so no prompt of this shape can ever
cache on it. Projected over Stage 1's 3,000 rows at the measured behaviour:

| | cost | per 1,000 clauses |
|---|------|-------------------|
| Haiku 4.5, caching impossible | $1.5879 | **$0.5293** |
| Sonnet 5, caching active | $1.3449 | **$0.4483** |

**The cheaper model costs 1.18x MORE per clause than the dearer one — and is also less
accurate** (Rung 1: 0.70 vs 0.75 on 20 rows). Headline rates of $1/$5 versus $2/$10
invert once the caching floor is applied. For a cost-focused thesis this is a result,
not an aside: **published per-token prices do not determine relative cost when a
cacheable prefix sits between two models' minimums.**

Two further notes:

1. **Silent by design.** The documentation states that shorter prompts "cannot be
   cached, and no error is returned — the request is simply processed without caching."
   That is §3e's failure class *in the API itself*: the cheaper path degrades silently
   and the only way to detect it is to read `cache_creation_input_tokens` and
   `cache_read_input_tokens` off every response — which hard rule 10 already requires.
2. **Sonnet's margin is 60 tokens.** Any edit to the label list or instructions that
   shortens the prefix below 1,024 would silently disable caching and raise Sonnet's
   cost ~2.5x with no error. The prompt sha256 is pinned; **treat the prefix length as
   a pinned quantity too, and assert it before any run.**

**CONSIDERED AND REJECTED: padding Haiku's prefix to 4,096 tokens.** It would make
Haiku cacheable and cut its cost substantially. It is rejected because the prompt is
**preregistered and hashed** (`zeroshot` sha256 `29a4c26e4330…`): padding would change
the prompt, change Haiku's predictions, and invalidate comparison with every result
already produced under the current prompt — including Rung 1. Optimising a
preregistered artefact after seeing its cost is exactly what hard rule 6 forbids.
Recorded here so the option is on the record as rejected, not overlooked.

### 3m. Pre-run notebook audit — seven defects, all silent (2026-09-08)

Both Kaggle notebooks were reviewed line by line before any GPU time was spent. Seven
defects found, **none of which would have raised an error**; five would have produced
plausible-looking but wrong results after 9–15 hours of GPU time. Same class as §3e:
failures that present as normal operation.

| # | notebook | defect | how it would have surfaced | verified |
|---|----------|--------|---------------------------|----------|
| 1 | tier1 | eval on `range(3000)` — positional — instead of the `test_3000` manifest | `test_3000` spans indices 1..9992; only **881 of 3000** rows overlap. The join to Claude's Stage 1 results would silently match 881 rows. **70.6% of the manifest missed** | measured |
| 2 | tier1 | `padding_side` left at Qwen's default `right` for batched generation | every sequence but the longest in each batch emits garbage; reads as "the model is bad" | **reproduced** (below) |
| 3 | tier1 | `max_seq_length=1024` | **0.7% of train examples exceed it and the LABEL is at the end** — those rows train on a prompt with no answer. 0.0% at 1536 | measured, 300 rows |
| 4 | tier1 | `max_seq_length` passed to `TrainingArguments` | `TypeError` at startup (loud, but wastes a session slot) — it is an `SFTConfig` field | confirmed: not a `TrainingArguments` field |
| 5 | tier1 | `gradient_checkpointing` sets `use_cache=False`, never restored before `.generate()` | generation very slow or failing | confirmed: Qwen default `use_cache=True` |
| 6 | both | checkpoints deleted only after the completion marker was written | a session dying during eval or export retrains the whole seed | logic review |
| 7 | tier0 | INT8 quantised with `avx512_vnni` | targets the **x86 Kaggle host**, not the Apple Silicon Mac Mini that deploys it | logic review |

**Defect 2 reproduced locally** on Qwen2.5-0.5B-Instruct, same family and chat template,
four prompts of differing lengths in one batch:

```
padding_side='right'                    padding_side='left'
  France -> "What would you like to      France -> "Paris"
             know about France"          Japan  -> "Tokyo"
  Japan  -> "What would you like to..."  Italy  -> "Rome"
  Italy  -> "What would you like to..."
  USA(longest) -> "Washington DC"        USA    -> "Washington DC"
```

**Only the longest sequence in the batch is correct under right padding** — 3 of 4 wrong,
and the wrong answers are fluent text, not errors. At 3,000 eval rows in batches of 16
this would have produced a format-failure rate near 90% and been read as Tier 1 failing
E4, triggering E4b on a harness bug rather than a finding.

**Also fixed, not in the original list:** `max_new_tokens` raised 12 → 16 (longest label
is 5 tokens + eos; 12 was sufficient but thin), generation truncation is now counted and
reported, per-row `row_indices` are saved alongside predictions so the offline join
cannot be silently misaligned, and tier0 now evaluates the **INT8** artefact on
`test_3000` and reports the E3 delta directly — without which E1's accept rule, which
attaches to INT8, could not be computed from the notebook's output at all.

### 3v. The resume rests on an unchecked filesystem assumption (2026-09-08)

Lever 1 (§3u) spreads seeds across quota windows and depends entirely on a resume. The
resume depends on checkpoints in `/kaggle/working` still existing next session.
**Nothing has ever checked that they do**, and the belief differs by run mode:

* **interactive**, Persistence = "Files only": believed to persist;
* **commit** (Save & Run All): believed **not** to. Each version runs in a fresh
  container, and the previous run's `/kaggle/working` becomes **that version's output**
  rather than the next run's working directory.

If the second is right, every commit starts seed 1 at `(fresh)` and lever 1 does not
work on the commit path at all — discovered nine hours in. **The plan's load-bearing
assumption was a filesystem behaviour nobody had tested**, which is §3e's shape exactly:
not a wrong number, an unexamined default that would have presented as normal operation
until it silently repeated 13.4 h of work.

Kaggle's documentation cannot settle it — its pages render client-side and return no
text to a fetch, the same wall met on quota accounting and the session cap. **So it is
settled by measurement**, not argument: `notebooks/kaggle_probe_persist.py`, three
commits of ~1 minute. It runs with **Accelerator = None**, so answering a filesystem
question costs **zero GPU quota**. Run 1 writes a marker plus a checkpoint-shaped
directory; run 2 reports `MARKER FOUND` or `NO MARKER`; run 3, with the notebook's own
output attached as an input, prints the exact `/kaggle/input/<slug>/` restore path.
Verified locally against all three branches with the Kaggle paths redirected.

**A second finding, which is worse, and which the persistence question was hiding.**
Two open Kaggle threads report that output from a **timed-out** commit cannot be
retrieved: "Unable to download the output of 'timeout exceeded' notebook" and
"[Bug] Can't access output files after 12-hour timeout". The logs show the files were
written; users cannot get them.

The plan as stated **required** commits to hit the cap: a seed needs ~13.4 h of training
plus eval, so a single commit can only ever end in timeout — the one path with a known
retrieval bug. **Consequence, and it holds whichever way the persistence probe
resolves:** each commit must train a step budget it can finish and exit cleanly,
producing a real version output. At 0.07 it/s, ~8 h is ~2,000 steps, so a seed is two
bounded commits plus eval.

**This is a change to how the run is executed, not to what is being measured.**
Splitting one 13.4 h training run into two bounded ones alters neither the data order,
the batch composition, the optimizer trajectory nor the number of steps — the checkpoint
carries RNG state, optimizer state and scheduler position, and `ignore_data_skip=False`
replays the dataloader to position (§3u). So it is **not** a preregistration amendment,
and §3t's registered configuration is unchanged. It is recorded here because the
distinction matters: a bounded commit is the same experiment, whereas
`group_by_length` — refused in §3u — would not be.

**Not yet done, and blocking.** `kaggle_tier1.py` still assumes `/kaggle/working`
persists and has no step budget. The restore path and the bound cannot be written until
the probe reports, because the restore's source path is one of the probe's outputs.
**Seed 1 must not start for real until then.**

### 3u. Measured Tier 1 throughput; probe projection wrong by 13x (2026-09-08)

**Measured on Kaggle, seed 1, one T4:** `15/3563 [02:56 < 13:21:05, 0.07 it/s]`, i.e.
**~13.4 h per seed**, ~40 h for three against a 30 h weekly quota. Also confirmed on the
real GPU: `trainable param dtypes after re-cast: {'torch.float32'} (392 tensors changed)`
— TRL's bf16 cast (§3r) did occur and the fp32 re-cast reached all 392 tensors — and
`train token length: mean 607 p99 1061 MAX 2378 | over MAX_LEN: 0`, confirming §3r's
length distribution and that MAX_LEN 2560 truncates nothing.

**Two estimates, one right and one badly wrong.**

| estimate | predicted | measured | error |
|----------|-----------|----------|-------|
| §3r FLOPs arithmetic (6ND + 33% recompute, 8–13 TFLOPS sustained) | 13–21 h/seed | **13.4 h** | correct, at the fast end |
| probe's measured-step extrapolation | ~1.0 h/seed | **13.4 h** | **13x low** |

**The measurement was worse than the arithmetic, and the reason matters.** §3r named the
probe's biases and still under-weighted them: one warm-up-inclusive step, on 10 rows of
~40 tokens against a fit-set mean of 607, at `max_steps=1` so no steady state was ever
reached. Roughly 15x on sequence length alone, which is most of the 13x. The lesson is
not "prefer arithmetic to measurement" — it is that **a measurement taken outside the
regime it is extrapolated to is not a measurement of that regime**, and labelling it an
estimate does not repair that. §3e's rule about one observation applies to a
badly-scoped measurement exactly as it does to one reading of a counter.

The probe's projection should therefore either be removed or run on realistic-length
rows for enough steps to reach steady state. It has served its purpose — the number is
now known from the real run — and is not worth the GPU minutes to fix.

**Quota decision: lever 1 — spread the seeds across quota windows.** Recorded with the
reason, since it constrains what may be traded later:

* **Lever 2 (DDP across both T4s) is not taken.** Its premise — that quota is charged in
  wall-clock session hours rather than per-GPU hours — could not be established, and the
  six unverified interactions are enumerated in the session record.
* **Lever 3 (shrink the fit set) is rejected on experimental, not engineering, grounds.**
  E4's accept rule is macro-F1 ≥ E1 + 0.04. Cutting training data to buy quota makes the
  central hypothesis *harder to pass* by weakening the arm under test, so a negative E4
  would no longer distinguish "the middle tier is not justified" from "we under-trained
  it to save GPU hours." That confound is not worth the hours.
* **Lever 1 costs calendar time and nothing else.** Correctness is unaffected, seeds stay
  comparable, and the resume machinery already exists.

**Consequence: the resume path is now load-bearing and has never been exercised.** At
13.4 h no seed completes in one session under either candidate cap, so *every* seed
spans at least two. Operational procedure — cap, checkpoint cadence, the commit path and
the exact resume confirmation — is in `notebooks/RUNNING.md`.

**`group_by_length` and similar: rejected, and it is a preregistration question.**
Padding waste at BS 4 is **27%** (34.6M real tokens against 44.0M padded, simulated from
the measured mean/p99/max), so length-grouped batching could recover at most ~21% — it
is not a lever that changes the quota arithmetic. It is refused on comparability
grounds regardless: **it changes batch composition, therefore the sequence of gradient
updates, therefore what "seed 1" means.** §3t registers the effective batch and the
seeds; batching policy belongs in that registration. Since seed 1 is already running
under random batching, adopting it now would make seed 1 incomparable to seeds 2 and 3,
which is a straightforward violation of hard rule 2's mean ± std over three seeds. **Off
the table for this run entirely.** Any future adoption is an amendment recorded before
*any* seed of that run executes.

### 3t. Training configurations, registered — they never were (2026-09-08)

Found while adding an effective-batch assertion to Tier 0. The instruction was to state
Tier 0's **registered** effective batch in the assertion message rather than infer it
from the file's constants. **There is no registered batch size** — not for E1, not for
E2, not for E4. E1's entry registers hypothesis, metric, split, accept rule, provenance
and falsification; E2's and E4's do the same. **No experiment registers a single
training hyperparameter.**

**This makes an existing assertion an overclaim.** `kaggle_tier1.py` line 161 reads:

```python
assert BS * GA == 16, f"effective batch must stay 16 (registered for E4), got {BS*GA}"
```

The assertion is right and the parenthetical is false: 16 is the value in the file, and
attributing it to a registration that does not exist dresses a constant up as a
commitment. Written by me in `d84c53b`, caught here only because someone asked the
assertion to cite its source. **An assertion that names a wrong authority is worse than
one that names none** — it defeats exactly the check a reader would otherwise make.

**Registered now, before any training run**, so hard rule 6 is satisfied and so both
notebooks' assertions have something real to cite. These are the values the notebooks
already contain; nothing is being changed, it is being *recorded*.

**Tier 0 — E1, E2, E3** (`kaggle_tier0.py`)

| parameter | registered value |
|-----------|------------------|
| model | `microsoft/deberta-v3-base` |
| max sequence length | 512 |
| epochs | 3, best epoch selected on `train_holdout_3000` |
| learning rate | 2e-5 |
| `per_device_train_batch_size` | 16 |
| `gradient_accumulation_steps` | 1 (library default, not set in the file) |
| **effective train batch** | **16** |
| visible GPUs | **1** |
| precision | fp16 AMP |
| seeds | 1, 2, 3 |

**Tier 1 — E4** (`kaggle_tier1.py`)

| parameter | registered value |
|-----------|------------------|
| model | `Qwen/Qwen2.5-1.5B-Instruct` |
| max sequence length | 2560 |
| epochs | 1 |
| learning rate | 2e-4 |
| `per_device_train_batch_size` | 4 |
| `gradient_accumulation_steps` | 4 |
| **effective train batch** | **16** |
| visible GPUs | **1** |
| precision | fp16 AMP, 4-bit NF4 base, fp32 trainable params (§3r) |
| LoRA | r=16, alpha=32, dropout=0.05, 7 target modules |
| seeds | 1, 2, 3 |

**"Visible GPUs: 1" is a registered experimental parameter, not an implementation
detail.** §3r established that a second visible device silently doubles the effective
batch through `train_batch_size = per_device_train_batch_size * max(1, n_gpu)`. That
makes device count part of the training configuration, so it is registered alongside the
batch size rather than left to the session's accelerator setting. The **documented
fallback** for Tier 1 (BS 2 / GA 8 under memory pressure, §3r) preserves the registered
effective batch of 16 and is therefore *within* this registration; any change to the
effective batch itself is an amendment.

**Tier 0's exposure, now fixed.** Tier 0 set no `CUDA_VISIBLE_DEVICES`, and DeBERTa is
neither 4-bit nor 8-bit, so `Trainer._wrap_model`'s `not is_loaded_in_8bit` guard passes
and `nn.DataParallel` wraps it on "GPU T4 x2" — training E1/E2 at an effective batch of
**32** against the 16 registered above, silently, because DataParallel on a standard
fp16 model simply works. `kaggle_probe_tier0.py` had the same omission, which is its own
defect: **a probe running on a different device configuration than the notebook it
certifies is not certifying that notebook.** Both now pin one device before torch is
imported and assert it.

**Practice note.** Two of this session's defects were assertions that were individually
correct while pointing at the wrong thing — `ast.parse` checking parseability instead of
compilability (§3s), and this one citing a registration that did not exist. The
mitigation is the same in both cases and is not "write more assertions": it is that **an
assertion must be traceable to the thing it claims authority from**, and that claim must
be checkable. Where the authority is a preregistered value, the file says so and the
value exists here.

### 3s. The checks never ran — the file did not compile (2026-09-08)

`kaggle_tier1.py` shipped from the §3r commit with `fp16` passed **twice** to the same
`dict()` call: the pre-existing `fp16=True` was left in place when the explicit
`fp16=USE_FP16, bf16=USE_BF16` pair was added beside it.

```
cfg_kw = dict(..., fp16=True, logging_steps=100,
              ..., gradient_checkpointing=True,
              fp16=USE_FP16, bf16=USE_BF16)
SyntaxError: keyword argument repeated: fp16
```

**This is a compile-time error, so nothing in CELL 2 executes.** Not the version gate
(§3p), not PREFLIGHT's signature checks (§3o), not the mixed-install crash-site check,
not the precision/device gate written the same day to catch §3r's own failure class.
Four sessions of accumulated guards, every one of them downstream of a file that never
parsed. **A check is worth exactly as much as the file that reaches it.**

**The compounding defect: the verification used the wrong instrument.** A CELL 2 parse
check WAS run in that session and reported `tier1 CELL 2 parses`. It used `ast.parse`.
Duplicate keyword arguments are rejected during symbol-table construction, not during
parsing:

```
ast.parse("f = dict(a=1, a=2)")             -> succeeds
compile("f = dict(a=1, a=2)", "<t>", "exec") -> SyntaxError: keyword argument repeated: a
```

So the check ran, passed, and was reported as passing, on a file that could not run.
**This is §3o's class for the fourth time** — alongside import-presence checked for
signature compatibility, one model generalised to all models, and one reading
generalised to a rule. The shape is always *verifying a proxy for the property you care
about*: here, "is this parseable" standing in for "will this execute".

Instance 4 of §3e also applies: an operation with a success path and no failure path.
`ast.parse` cannot report this defect, so its success carried no information about it.

**Structural fix — `tests/test_notebooks_compile.py`, in the main suite (now 366 tests),
not a one-off.** It compiles the `CELL 2 of 2` region of `kaggle_tier0.py` and
`kaggle_tier1.py` and both probe files whole, with `compile(..., "exec")`. Four
properties are pinned so the test cannot decay into the weaker check it replaces:

* `test_ast_parse_would_not_have_caught_it` asserts the *divergence itself* — that
  `ast.parse` accepts the exact defect that shipped and `compile` rejects it. Anyone
  "simplifying" this back to `ast.parse` fails that test.
* the `CELL 2 of 2` banner must exist exactly once per two-cell notebook, because the
  banner is also what `notebooks/RUNNING.md`'s run protocol depends on — if it moves,
  the test is compiling the wrong region *and* the run instructions are wrong.
* CELL 1's exclusion is justified by an assertion that it still contains `!pip` and
  still fails to compile, so the carve-out cannot outlive its reason.
* verified by reintroducing the real defect: the suite fails with
  `keyword argument repeated: fp16 (line 392 of the cell)`, and passes once reverted.

**`kaggle_tier0.py` does NOT have this defect, and the premise that it might was
wrong.** The §3r commit (`d84c53b`) touched `PREREGISTRATION.md`,
`kaggle_probe_qlora.py` and `kaggle_tier1.py` only — **Tier 0 received no precision
edits at all**, so there was no second `fp16` to duplicate. Its CELL 2 compiles clean.

**But Tier 0 not receiving those edits is itself a finding, and two of §3r's defects
apply to it unfixed:**

| §3r defect | applies to Tier 0? | why |
|------------|--------------------|-----|
| `dtype="auto"` reading a bf16 config | **no** | Tier 0 pins `transformers==4.57.6`, which never defaults `dtype` to `"auto"` — verified two ways in a venv built from Tier 0's own pins: zero occurrences of `dtype = "auto"` in `from_pretrained`, against an explicit `dtype = "auto"` line in 5.0.0. The pin taken for `optimum-onnx` incidentally closes this. |
| TRL's unconditional bf16 cast | **no** | Tier 0 uses `Trainer`, not `SFTTrainer`, and is not 4-bit |
| `bf16` defaulting to `None` | **no** | `TrainingArguments.bf16` defaults to `False` in 4.57.6, not `None` as in 5.0.0 |
| **DataParallel on two visible devices** | **YES** | Tier 0 sets no `CUDA_VISIBLE_DEVICES`. DeBERTa is neither 4-bit nor 8-bit, so `Trainer._wrap_model`'s `not is_loaded_in_8bit` guard passes and `nn.DataParallel` wraps it on "GPU T4 x2" |
| **effective batch silently doubling** | **YES** | `train_batch_size = per_device_train_batch_size * max(1, n_gpu)`. Tier 0 runs `BS = 16`, so on two devices the effective batch becomes **32**. Unlike Tier 1's case this does not crash — DataParallel on a standard fp16 model works — it simply trains E1/E2 at a batch size no output reports. |

The second pair is the more dangerous presentation of the two: Tier 1's version of this
would have been loud, Tier 0's is silent. **Not fixed in this commit** — recorded here,
and to be fixed before Tier 0 runs.

### 3r. T4 audit — every implicit precision, device and memory assumption (2026-09-08)

Three probe runs on Kaggle each found a **different** T4-specific defect, and each was
visible only after the previous one was fixed. That is the signature of debugging one
traceback at a time. This entry enumerates every precision, device and memory value the
two notebooks left to a library default, states what T4 + transformers 5.0.0 +
peft 0.20.0 + trl 1.12.0 actually does with it, and fixes them in one commit.

**The reported failure.** `NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'`, in `accelerate` `unscale_gradients` → `scaler.unscale_`.

**The diagnosis on hand was that `dtype="auto"` reads Qwen2.5's config `bfloat16`. That
is true, and it is not the cause.** Traced through the actual path (CPU, 4-bit flag
forced so peft's and TRL's kbit branches both execute):

| stage | dtype of trainable params |
|-------|----------------------------|
| `from_pretrained(dtype=bfloat16)` | bf16 |
| `prepare_model_for_kbit_training` | **fp32** — the `from_pretrained` dtype is already overwritten here |
| `SFTTrainer.__init__` | **bf16** |
| same run, but `from_pretrained(dtype=float16)` | **still bf16** |

The bf16 that reaches the scaler is written by **trl 1.12.0 `SFTTrainer.__init__`,
~line 1154**:

```python
if _is_quantized_model:
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.bfloat16)
```

Unconditional whenever the model is 4-bit — **no GPU capability check, no reference to
`args.fp16`**. It follows the QLoRA paper, which is right on Ampere+ and fatal on
sm_75. **So passing `dtype=torch.float16` does not fix this failure**; it is still
correct, and it closes a different exposure, but the fix is to re-cast trainable
parameters to **fp32** after `SFTTrainer` is constructed. fp32, not fp16: AMP keeps
master weights in fp32 and autocasts the matmuls.

**Full audit. "Implicit" = the value was never written in our code.**

| # | assumption | left implicit? | what actually happens on T4 + this stack | fix |
|---|------------|----------------|-------------------------------------------|-----|
| 1 | `from_pretrained` dtype | yes | transformers 5.0.0 defaults `dtype="auto"` (`modeling_utils` line 262–263), reads Qwen2.5's config `bfloat16`. In 4.x the default was fp32 — **a v5 behaviour change inherited by silence** | pass `dtype=MODEL_DTYPE` (fp16) at both load sites |
| 2 | `bnb_4bit_compute_dtype` | no, already fp16 | correct, but unlinked to the model dtype | bind both to named constants and assert equal |
| 3 | **trainable-param dtype after `SFTTrainer`** | yes | **TRL casts to bf16 unconditionally — this is the reported crash** | re-cast to fp32 after construction, assert, and warn if TRL stops doing it |
| 4 | `bf16` | yes | `SFTConfig.bf16` defaults to **`None`, not `False`**. Falsy today, but never asserted | set `bf16=False` explicitly, assert `not USE_BF16` and `USE_FP16 != USE_BF16` |
| 5 | **device count** | yes | **`Trainer._wrap_model` line ~1665: `if n_gpu > 1 and not getattr(model, "is_loaded_in_8bit", False): model = nn.DataParallel(model)`. The guard tests 8-bit ONLY — never `is_loaded_in_4bit`, never `hf_device_map`. On "GPU T4 x2" our 4-bit model gets DataParallel-wrapped** | `CUDA_VISIBLE_DEVICES=0` before `import torch`, assert `device_count() == 1` |
| 6 | **effective batch** | yes | **`TrainingArguments`: `train_batch_size = per_device_train_batch_size * max(1, n_gpu)`. With two devices the effective batch silently becomes 32, not the registered 16** — a change to the experiment reported by nothing | same pin; plus `assert BS * GA == 16` |
| 7 | `device_map={"": 0}` | no | places weights on GPU 0 but **does not set `n_gpu`**, so it does not prevent 5 or 6 | keep, but it is not the guard |
| 8 | memory at BS 4 × 2560 | yes | see below | documented fallback |
| 9 | per-seed runtime | yes | the 3–5 h figure assumed two devices | measured by the probe now |

**Memory, one 16GB T4, gradient checkpointing on.** 4-bit base ~1.1 GB; LoRA r=16 over 7
modules = 18.5M params, fp32 weights + grads + Adam states ~0.3 GB; checkpointed layer
boundaries at BS 4 × 2560 ~0.9 GB. The deciding term is the **lm_head logits**, BS × seq
× 151,936 vocab: ~0.7 GB at the 607-token mean, but ~2.9 GB in fp16 plus its fp32
cross-entropy upcast — **~8.7 GB** — for a batch padded to 2378, the longest fit row.
Dynamic padding means this only bites when a long row lands in a batch, and only **17 of
57,000 rows exceed 1536 tokens, 2 exceed 2048**. So the common case fits with room and
the tail case is the risk. **Fallback: BS 2 / GA 8**, written into the code comment —
not BS 2 alone, because the effective batch is BS × GA and must stay **16**, which is
what E4 is registered at.

**Runtime, re-estimated for ONE T4 — and it does not fit the quota.** 57,000 rows × 1
epoch, ~607-token mean, ≈45–55M padded tokens. Forward+backward ≈ 6ND ≈ 4.6e17 FLOPs,
plus ~33% for checkpointing recompute ≈ **6.2e17 FLOPs**. A T4 peaks at 65 TFLOPS fp16;
QLoRA with 4-bit dequant per matmul, small batches and checkpointing realistically
sustains 8–13 TFLOPS. That is **13–21 h per seed**, so **3 seeds ≈ 40–60 h against a 30 h
weekly quota** — Tier 1 alone does not fit, before Tier 0's 3–5 h.

**This estimate has roughly ±2× error bars and is not a basis for a decision on its
own** (§3e). Two consequences, both taken:

1. `kaggle_probe_qlora.py` now **measures** step time and prints a projected per-seed and
   3-seed figure, labelled an estimate, with its biases named — so the quota question is
   answered from the real GPU before three seeds are committed rather than after one.
2. The resume design already tolerates this: seeds are skipped by completion marker and
   interrupted seeds resume from checkpoint, so **3 seeds spanning two or three weekly
   quota windows is a supported path**, not a failure. If the measured projection
   confirms 13–21 h/seed, that is the plan, and RUNNING.md's "9–15h" figure for Tier 1
   is wrong and must be corrected against the measurement.

**Early gate, which is the structural fix.** All three T4 failures were disagreements
between values knowable in the first ten seconds — model dtype, compute dtype, AMP mode,
device count, GPU capability. Both files now resolve and cross-check all five **before
any weights load**, printing each and raising on any disagreement. That converts a
step-5 crash after a backward pass into a step-2 message. Same lesson as §3o and §3p:
the check must sit at the layer where the values are, and must compare them against each
other rather than confirm each one alone.

**What is NOT verified.** None of this ran on a T4. The dtype trace, the re-cast block
and the parse checks were executed on CPU (and the trace with a stubbed `bitsandbytes`);
the DataParallel guard, the `n_gpu` batch multiplication and the transformers dtype
default were read from installed source, not observed on two devices. The memory
arithmetic is arithmetic — no allocation was measured. The runtime figure is an estimate,
explicitly. Every one of these is now instrumented in the probe, which is where they get
measured.

### 3q. `use_reentrant` — a question settled by measurement, and the answer is "no"

Decided **before** the Kaggle session rather than mid-session, because the failure it
guards against costs GPU hours to discover.

`gradient_checkpointing=True` together with `prepare_model_for_kbit_training` is a
known sticking point: with **reentrant** checkpointing, if nothing upstream of the LoRA
layers requires grad, the backward pass raises *"element 0 of tensors does not require
grad and does not have a grad_fn"*. The widely-repeated fix is to pass
`gradient_checkpointing_kwargs={"use_reentrant": False}`.

**For peft 0.20.0 + trl 1.12.0 + transformers 5.0.0 the kwarg is not needed, and is not
added.** Two independent mechanisms make it unnecessary, and both were measured, not
assumed.

**How this was determined.** Source reading first, then an executed test with a control
— per §3e, introspecting cleanly is not evidence that a path works.

*Read:*

* `transformers 5.0.0` `PreTrainedModel.gradient_checkpointing_enable`: when
  `gradient_checkpointing_kwargs is None` it substitutes `{"use_reentrant": False}`.
  **Non-reentrant is already the default.** The same method then calls
  `self.enable_input_require_grads()` whenever `main_input_name == "input_ids"`, which
  is the guard against the no-grad error *independently of reentrancy*.
* `trl 1.12.0` `SFTTrainer.__init__` line 1351: it applies
  `setdefault("use_reentrant", False)` **only when `transformers.__version__ < 5.0.0`**,
  with a comment saying it expects 5.0.0 to make non-reentrant the default. TRL has
  already reasoned about this exact version boundary, in both directions.
* `trl 1.12.0` line 1147: for a PEFT model with gradient checkpointing it calls
  `model.enable_input_require_grads()` itself, citing transformers issue #42489.
* `peft 0.20.0` `prepare_model_for_kbit_training`: on the kbit branch it also calls
  `enable_input_require_grads()`, then `gradient_checkpointing_enable({})`.

*Executed*, on CPU with a tiny causal LM through the real `SFTTrainer` path, forcing
peft's kbit branch. 4-bit itself is CUDA-only, but the no-grad failure is **not**
4-bit-specific — it is "frozen base + checkpointing, nothing upstream requiring grad",
which reproduces here. The value actually bound into each module's checkpoint
`functools.partial` was read after `train()`, not inferred:

| arm | `gradient_checkpointing_kwargs` passed | bound after `train()` | result |
|-----|----------------------------------------|------------------------|--------|
| A — as `kaggle_tier1.py` is written | *(none)* | `{'use_reentrant': False}` | trains, loss 6.4949 |
| B — the folklore fix | `{'use_reentrant': False}` | `{'use_reentrant': False}` | trains, loss 6.4954 |
| C — **control**, forced reentrant | `{'use_reentrant': True}` | `{'use_reentrant': True}` | **trains**, loss 6.4954 |

A and B bind an identical value: **the kwarg is inert here**, and adding it would look
like configuring something while changing nothing.

**The control is the informative arm, and it did not fail.** C genuinely selects
reentrant checkpointing and still trains, which locates the protection in
`enable_input_require_grads()` rather than in the reentrancy setting. Had C failed, the
kwarg would have been load-bearing and would have been added to both files. This is the
§3e discipline applied deliberately: a control known to behave the opposite way, to make
the result conclusive rather than merely suggestive.

**One measurement error worth recording, because it nearly produced the wrong answer.**
The first run read the bound value after `SFTTrainer.__init__` and saw `{}` in all three
arms — the kwarg apparently ignored. `Trainer` activates gradient checkpointing inside
`_inner_training_loop`, **not** in `__init__`, so the reading was taken before the call
under test had happened. Reading after `train()` separates the arms cleanly. Same shape
as §3o: measuring a proxy, at the wrong moment, and getting a plausible number.

**What is NOT verified.** Nothing here ran on CUDA or on real bitsandbytes 4-bit
weights, and the model was a tiny GPT-2, not Qwen2.5-1.5B or Llama-3.2-3B. The
reentrancy binding and `enable_input_require_grads` are device- and
architecture-independent code paths, which is why the CPU result is informative — but
that is an argument, not a measurement, and §3e forbids promoting it to a structural
claim. Kaggle also supplies its own `transformers`, and if it is **below 5.0.0** the
mechanism changes hands: TRL's `setdefault` at line 1351 then applies it instead. Both
branches are covered, by different code, which is why the check is behavioural.

**Consequence, in `kaggle_probe_qlora.py` step 5:** rather than pass an inert kwarg, the
probe now *measures* both facts on the real GPU stack — it prints the checkpoint kwargs
actually bound after `train()`, and asserts a positive finite `grad_norm`, because a step
that runs while moving nothing is a failure presenting as success (§3e). The assertion
message names the fix and instructs that it be applied to **both** files together. So
the decision is recorded here, the assumption is instrumented there, and if CUDA
disagrees the probe says so in two minutes instead of after a seed.

### 3p. Notebook defect 10 — a check that validated the wrong layer, again

`PREFLIGHT` printed **PASSED** while running `transformers 5.0.0` and `peft 0.19.1` —
neither of which was the version introspected. It validated **signatures**, never
**versions**, so it certified an environment it had never seen.

The crash was inside peft, not our code:
`AttributeError: 'LoraConfig' object has no attribute 'velora_config'` in
`peft/tuners/lora/bnb.py`. Diagnosed locally: in peft **0.20.0** that field exists on
`LoraConfig` *and* is referenced by `bnb.py` — the two agree. The Kaggle environment
was executing 0.20.0's `bnb.py` against 0.19.1's `LoraConfig`: **a mixed install**,
because pip cannot replace a package the kernel has already imported.

Three compounding defects, all mine:

1. **`!pip -q install ... | tail -3`** discarded the resolver output that would have
   shown the pins failing. Same silent-failure class as `chars/4`.
2. **No version assertion.** PREFLIGHT now compares running versions against the pins
   and raises, naming both.
3. **No kernel restart.** Every notebook is now explicitly two cells with a mandatory
   restart between them.

**Version decision, with the reason.** Kaggle ships `transformers 5.0.0`. Rather than
downgrade it for Tier 1 — which means uninstalling a preloaded package, the very thing
that caused the mixed install — only `peft` is forced, because `trl 1.12.0` declares
`transformers>=4.56.2` with **no upper bound** and `peft 0.20.0` declares none either.
Fewer packages replaced, fewer chances of a partial install. Tier 1's PREFLIGHT was
executed locally under transformers **5.0.0** and passes.

**Tier 0 must differ:** `optimum-onnx` declares `transformers<4.58.0,>=4.36`, and Tier 0
needs optimum for the ONNX export, so it pins `4.57.6`. The two notebooks run in
separate sessions, so the conflict is never resolved in one environment. This
constraint was found only by attempting the install locally.

**What still cannot be verified off-CUDA, and the response.** `bitsandbytes` 4-bit
loading and the peft LoRA-on-4-bit path cannot be exercised on macOS at all. Rather
than assert they work, `notebooks/kaggle_probe_qlora.py` is a ~2-minute cell that runs
4-bit load → LoRA wrap → one real training step → generation, and checks peft's
internal consistency at the exact crash site. **It is run before Tier 1, not instead of
knowing.**

### 3o. Notebook defect 9 — verifying the wrong property, twice

`SFTConfig` was guarded with `try: from trl import SFTConfig / except ImportError`.
The import succeeds; **the signature is what changed.** TRL renamed `max_seq_length`
to `max_length`, so the run died with `TypeError` at seed 1.

**This is the third instance of the same reasoning error**, alongside the two already
in §3e: the temperature parameter validated on Haiku and assumed for Sonnet, and one
rate-limit reading generalised to a quota model. The shape is *checking a proxy for the
property you care about*: import presence for signature compatibility, one model for
all models, one reading for a rule.

**Structural fix — a PREFLIGHT block at the top of both notebooks**, before the dataset
downloads and before any weights load, so a signature error surfaces in ~10 seconds:

* the sequence-length kwarg is **resolved by introspection** (`max_length` vs
  `max_seq_length`), never assumed, and the chosen name is printed;
* every kwarg passed to `SFTConfig`, `SFTTrainer`, `LoraConfig`, `TrainingArguments`,
  `Trainer`, `DataCollatorWithPadding`, `AutoQuantizationConfig.arm64` and
  `ORTQuantizer.quantize` is checked against the **installed** signature;
* semantic defaults are asserted, not trusted: `packing is False` (packing would
  concatenate examples so the LABEL no longer ends its own prompt) and
  `dataset_text_field == "text"`. Note `max_length` defaults to **1024** — a silently
  dropped kwarg would have truncated exactly as §3m defect 3 did;
* callables taking `**kwargs` are reported as **NOT VERIFIABLE**, not as passing.
  Claiming introspection had verified `from_pretrained` would repeat the original error.

Versions are pinned to what was introspected: **transformers 4.57.6, trl 1.12.0,
peft 0.20.0**. torch is deliberately unpinned — Kaggle's build is CUDA-matched.

**Correction to §3m defect 2, found here:** TRL's collator pads via its own
`trl.trainer.utils.pad()`, whose `padding_side` defaults to `"right"`, and `SFTTrainer`
never reads `tok.padding_side`. So the training-side padding fix is **belt-and-braces,
not the operative mechanism**; it remains only because a future TRL version could start
honouring it. The measured 1.34 loss drift is real, and the fix is genuinely load-bearing
for the **eval** loop, which calls `tok()` directly.

### 3n. Notebook defect 8 — an invariant established once, mutated per iteration

Found in review after §3m's fixes, before any GPU time.

`tok.padding_side = "right"` was set **once at module level**, outside the seed loop.
Seed 1's eval switched it to `"left"` for generation and nothing restored it, so seed 2
would fail its training assert **after seed 1's 5–6 hours**. Re-running would not help:
the module-level line runs again, seed 1 is skipped as complete, seed 2 trains and
leaves `"left"`, and seed 3 fails identically. **One session per seed**, defeating the
resume design entirely.

Simulated across both paths (fresh, and post-crash with seed 1's adapter present):

```
OLD:  seed1 train ok -> AssertionError: seed2 TRAIN ASSERT FAILED
NEW:  seed1 train/eval ok, seed2 train/eval ok, seed3 train/eval ok   (both paths)
```

**The assert was not the defect — it was the only reason this surfaced at all.** Without
it, seeds 2 and 3 would have trained under left padding and silently produced a
corrupted loss (§3m defect 2 measured that drift at 1.34), and the two runs would have
been incomparable to seed 1 in a way no output would have shown.

**General lesson, and the reason this is recorded separately:** an invariant required by
a loop body must be **established inside the loop**, not inherited from before it. A
static sweep for attribute mutations on pre-loop objects found `tok.padding_side` to be
the only such case; `eval_bs`, `preds`, `raws`, `trunc`, `model` and `base` are all
rebound per iteration and are safe.

### 3k. PREDICTION, recorded BEFORE Stage 1's results land (2026-09-07)

**Cache economics measured at n=10 do not extrapolate to batch concurrency.**

Cache entries become available to other requests only **after the first response
begins**. A batch processes requests concurrently, so many start before the first
cache write has landed and each of those pays a write instead of a read. Rung 2 had 10
requests per leg; Stage 1 has 3,000. The effect scales with how many requests are
in flight before the first write completes.

**Predicted, before observing:**

| quantity | Rung 2 (n=10) | Stage 1 prediction (n=3000) |
|----------|---------------|------------------------------|
| Sonnet cache writes | 1 | **> 1, plausibly many** |
| Sonnet hit rate (read / all input) | 74.2% | **materially below 74.2%** |
| Haiku cache writes / reads | 0 / 0 | **0 / 0** (prefix 793 < the 4,096 minimum, §3j) |
| Sonnet projected cost | — | **> $1.3449** — that projection assumed one write |

**How this is scored:** report observed `cache_creation_input_tokens` and
`cache_read_input_tokens` totals per leg against the above. A Sonnet hit rate at or
above 74.2% falsifies the prediction; a materially lower rate with more than one write
confirms it. Either way the number is reported, and the n=10 → n=3000 extrapolation is
recorded as unsound or sound accordingly.

**Budget is unaffected** — gating assumed zero cache hits throughout (hard rule 8), so
this changes what we report, not what we may spend.

**Caveat on the documented minimum.** §3j cites Sonnet 5's published minimum as 1,024.
That figure has been reported not to hold in practice (anthropic-sdk-python issue
#1194: caching does not fire at 1,024 for Sonnet-tier models and appears near 2,048;
at least one third-party reference lists Sonnet 4.6 at 2,048 against the docs' 1,024).
**Our Sonnet 5 did cache at a 1,084-token prefix**, which bounds its true threshold at
or below 1,084 empirically — so §3j's finding stands on measurement, not on the doc.
But the "60-token margin" in §3j is **not** a safety margin: it is measured against a
number already shown to be unreliable. The guard is therefore behavioural —
`assert_caching_engaged` (src/api/usage.py) fails a run whose cache fields are all
zero — not a comparison against any published minimum.

### 3k-RESULT. Stage 1 scored against §3k — the prediction is FALSIFIED (2026-09-08)

Batch `msgbatch_014FYs1h3L4cVNYDke9hnXvn` (6,000 requests) and the few-shot batch
`msgbatch_01QbUfqihLGmwG3wkYJVZuMU` (1,000) both ended; all 7,000 succeeded, 0 failed.
Recorded once each in `results/spend_ledger.jsonl`. Cumulative spend **$3.5815** of the
$15.00 hard stop.

| leg | n | input (uncached) | cache_write | cache_read | output | est → actual |
|-----|---|------------------|-------------|------------|--------|--------------|
| stage1 Haiku 4.5 | 3000 | 2,797,272 | 0 | 0 | 72,246 | $6.9373 → **$1.5793** (−77.2%) |
| stage1 Sonnet 5 | 3000 | 641,463 | **0** | 3,240,000 | 68,570 | $4.5543 → **$1.3083** (−71.3%) |
| fewshot Sonnet 5 | 1000 | 216,946 | **2,702** | 2,699,298 | 22,789 | $3.1401 → **$0.6062** (−80.7%) |

**Scored against §3k's own accept rule, which was written before submission:**

| §3k predicted | observed | verdict |
|---------------|----------|---------|
| Sonnet cache writes **> 1, plausibly many** | **0** | **falsified** — fewer than Rung 2's one, not more |
| Sonnet hit rate **materially below 74.2%** | **83.5%** | **falsified** — §3k: "a hit rate at or above 74.2% falsifies the prediction" |
| Sonnet cost **> $1.3449** | **$1.3083** | **falsified** |
| Haiku **0 writes / 0 reads** | 0 / 0 | **confirmed** (793-token prefix, below the minimum, §3j) |

**Why Stage 1's Sonnet leg wrote nothing.** 3,240,000 / 3,000 = exactly **1,080 tokens
read per request** — every one of the 3,000 was a read. Rung 2 wrote that same prefix
(identical `prompt_sha256` `29a4c26e…`) at 17:55 UTC under a 1h TTL; Stage 1 was
submitted ~18:00, and each read refreshes the TTL. **Stage 1 never met a cold prefix, so
it never tested the concurrency mechanism at all.** Its hit rate beats Rung 2's precisely
because Rung 2 already paid the one write that Stage 1 then rode for free. Recording the
prediction as falsified on this leg alone would be scoring it on a run that could not
have confirmed it either.

**The few-shot leg does test the mechanism, and also refutes it.** Different prompt
(`prompt_sha256` `cb9498ce…`), therefore a genuinely **cold** prefix, with 1,000 requests
submitted at once:

    cache_creation_input_tokens  2,702       = one prefix, written once
    cache_read_input_tokens  2,699,298       = 2,699,298 / 2,702 = 999.0 exactly

**One cold prefix, one thousand concurrent requests, exactly one write and 999 reads.**
§3k's premise — that requests starting before the first write lands each pay their own
write — did not hold at n=1000. The mechanism as described is wrong, not merely
mis-sized.

**What this does and does not license.** It falsifies §3k as written; that is settled.
It is **not** a basis for asserting "batch caching always writes once." Per §3e this is
**one cold-prefix batch, on one model, at one n**: untested at n=3000, untested on Haiku,
untested for a prefix near the caching minimum, and untested across the
submission-timing variation that §3k's mechanism was actually about. The honest summary
is that the n=10 → n=3000 extrapolation §3k doubted turned out **sound** on this
evidence, and the reason it doubted it turned out **not to be a real effect at n=1000**.

**Retained consequences, which stand regardless.** §3k's budget note holds: gating
assumed zero cache hits throughout (hard rule 8), so every leg came in far under its
estimate and nothing was over-spent. And the behavioural guard `assert_caching_engaged`
remains the right instrument — the two Sonnet legs would have been indistinguishable
from "caching silently stopped working" on any check that compared token counts against
a published minimum.

**Kept under hard rule 7.** This is a preregistered prediction that was wrong, scored by
the rule it shipped with, and it stays in the report.

### 3l. E2 arm selection, recorded BEFORE the run (2026-09-07)

**Decision: one arm, `sqrt_inv_freq`, at 3 seeds. Not three arms at fewer seeds.**

GPU budget: Tier 1 (9–15h) + Tier 0 CE (3–5h) = 12–20h of Kaggle's 30h weekly quota.
Three arms × 3 seeds is a further 9–15h; worst case 35h, over quota.

**Why not cut seeds instead.** E2's margin *is* `√2 × 1.96 × seed_sd`, and `seed_sd`
comes from C3, which comes from the CE baseline's seeds. Reducing seeds on the arms
would both widen the required margin and degrade the estimate that sets it — a double
penalty on the comparison that matters — and would breach hard rule 2 on precisely the
runs being compared.

**Why `sqrt_inv_freq`.** `inv_freq` at 137.7× is the arm most likely to destabilise
training (already noted in E2), and `effective_number` carries a free parameter β that
we would be choosing without evidence. `sqrt_inv_freq` is the moderate option and the
one most likely to reveal a real effect if H3 holds.

**Reporting rule.** If `sqrt_inv_freq` clears the C3-derived margin, H3 is supported and
the remaining arms are a refinement. If it does not, **E2 is reported as tested on one
arm with `inv_freq` and `effective_number` UNRUN**, stated plainly — not as "class
weighting does not help". Sequencing: **CE 3 seeds → C3 → `sqrt_inv_freq` 3 seeds.**

### 3i. Amendment — Sonnet output budget 48 → 40 (2026-09-07)

**Trigger:** a pre-submission upper bound breached `stage_1.max_usd`.

Input-token counts were first estimated from a **40-row** sample. A **600-row** sample
(free `count_tokens`, paced under the 100 rpm org limit) showed the 40-row means were
**low**, and gave sd 137.0 (Haiku) / 204.2 (Sonnet) — a long tail, as
`results/data_report.md` predicted (p99 ≈ 550 clause tokens, 1.2–1.6% over 512).

Gating is computed from a **one-sided upper bound, mean + 2·SE** on the total, with a
finite-population correction, not from the point estimate:

| Sonnet output budget | Batch 1 upper bound | vs `max_usd` $7.00 |
|----------------------|---------------------|--------------------|
| 48 (previous) | $7.0571 | **BREACHES by $0.0571** |
| **40 (adopted)** | **$6.9371** | clears by $0.0629 |

Note the point estimate at 48 was **$7.0013** — over the line on its own, so no amount
of sampling precision would have rescued it. The trim was necessary, not marginal.

**Why trim rather than raise `max_usd`:** loosening a budget guard to fit a change is
the wrong direction. `max_usd` is unchanged at $7.00.

**Residual risk, restated.** §3f already recorded that Sonnet's 48-token budget was
~1.9× its observed maximum of 25 (n=20, zero truncations) and that 0/20 is compatible
with a true overrun rate near 15%. **At 40 the headroom drops to ~1.6× that observed
maximum, so the risk recorded in §3f increases.** It is still not measured. Stage 1
must report Sonnet's truncation count at 40; if non-zero, both §3f and this amendment
were wrong, and both stay in the record (hard rule 7).

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
- **Sampling temperature: PER-MODEL, and now DROPPED for both.** *(Second correction,
  2026-09-07.)* The entry below is **true for Haiku 4.5 only**. **Sonnet 5 rejects
  `temperature` outright** — `` `temperature` is deprecated for this model`` — verified
  on both the sync and batch paths.

  Setting it on Haiku while omitting it on Sonnet would make the two Stage 1 legs differ
  by an uncontrolled parameter, and comparing those legs is the point of running both.
  **`temperature` is therefore omitted for BOTH models.**

  **Consequence: the seeds-based variance plan is load-bearing again.** With no
  temperature control on either leg, run-to-run variability is neither pinned nor
  bounded, and hard rule 2's mean ± std over ≥3 seeds is the *only* mechanism
  characterising it. C3's seed-variance measurement is now doubly gating: it sets E2's
  margin **and** it is the sole evidence about determinism.

  The verified-for-Haiku detail is retained below for the record:
- **(Haiku 4.5 only) temperature is pinnable via `extra_body`.** *(Corrected 2026-09-07; an
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
