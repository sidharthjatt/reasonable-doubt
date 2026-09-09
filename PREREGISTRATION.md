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
| **Greenfield** | hardware bought for this workload; capital attributable | $633,862.43 / V + $1.475e-5 per 1k | **V\* = 1,413,971 clauses** for Tier 0 alone (energy-inclusive, §3aa, trained artefacts), higher for the cascade by 1/(1 − escalation_rate) |
| **Sunk capital** | the Mac Mini already exists (it does) | ~$0.000013 per 1k | **local wins from the first clause** |

**Both are correct under their own assumption, and the honest report gives both.** The
greenfield number carries a hard consequence worth stating plainly: **LEDGAR is ~80,000
clauses in total, so at realistic contract-review volume greenfield local hardware never
breaks even** — V\* is ~18x the entire corpus. A reader deploying on hardware they
already own reaches the opposite conclusion. The volume at which those two answers
swap is the finding.

`V_max` (what the device can process in its life) is **714,999,960** clauses at
30.23 rps, 3 years and 25% duty — **506x V\***, so under the greenfield case the device
does not die before break-even; it simply needs ~17.7 corpora of work to get there.

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
| E1b | Tier 0 retrained at **10 epochs**, all else identical to E1 | **macro-F1 ≥ 0.80 on `test_3000`, INT8, mean over 3 seeds** — the SAME bar as E1. Seed 1 first; seeds 2–3 gated on it | **registered — see E1b below** |
| E2 | Tier 0 loss arms vs E1 (sqrt-inv-freq, effective-number, inv-freq) | best arm beats E1 by **≥ √2·1.96·seed_sd** (paired, same rows) — **NOT YET SETTABLE** | **[C3-gated]** |
| E3 | INT8 vs FP32 at the deployed precision | **\|INT8 − FP32\| ≤ 0.01** macro-F1, paired. FP32-as-headline prohibited | planned **[C3-gated]** |
| E4 | Tier 1: Qwen2.5-1.5B-Instruct LoRA, 3 seeds | macro-F1 **≥ E1 + 0.04**; else Tier 1 not justified → E4b | planned |
| E4b | Two-tier `Tier 0 → Claude` fallback | E6's rule with Tier 1 removed. **Registered before E4 runs** | registered |
| E5 | Routing signal: margin vs max-softmax vs entropy | best **AUROC ≥ 0.75** AND **≥ 0.05** above worst, on `dev_2000` only | planned |
| E6 | **Cost-vs-volume break-even curve** — the headline | (1) within **0.04** macro-F1 of Sonnet-5-alone; (2) finite crossover **V\* ≤ V_max**; (3) asymptote **< 50%** of Sonnet-5. **V\* is reported, not tested** | **BLOCKED by §1b** |
| E7 | `max_length` 256 vs 512 ablation | loses **< 0.02** macro-F1 paired AND **≥ 1.5×** faster; per-class deltas required | planned **[C3-gated]** |
| E8 | Few-shot vs zero-shot Sonnet-5, paired on 1,000 rows | **`fraction_closed` ≥ 0.50 AND paired-bootstrap CI excludes 0** ⇒ premise live, defer restatement; **≤ 0.20** ⇒ capability not prompting. Band in full below | **registered — see E8 below** |
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
- **STATED LIMITATION — E1's training environment is unrecorded, permanently.** E1's
  per-seed JSONs (`results/tier0_ce_seed*.json`) carry the manifests, hyperparameters and
  metrics, and **nothing identifying the machine that produced the weights**: no GPU
  model, no CUDA or cuDNN version, no driver, no torch version, no TF32 state. The only
  environment fact ever emitted was `transformers 4.57.6`, and it was *printed* rather
  than persisted. **This cannot be filled retroactively — the runs are finished and the
  session is closed.**

  Two consequences a reader of E1 needs, independent of anything E1b does:

  1. **E1 is not reproducible to the precision this project claims elsewhere.** Its
     numbers are 0.7636 / 0.7597 / 0.7478 on `test_3000` FP32, and a reproduction attempt
     that lands outside that spread cannot be attributed between "different training
     setup" and "different hardware", because one of the two was never written down.
     `seed_sd = 0.0032` is a **within-host** spread and does not bound the cross-host
     term.
  2. **The GPU model is a reconstruction, not a record.** It is inferred from the
     notebook's `GPU T4 x2` comments and a terminal scrollback. Anywhere E1's hardware is
     described, it must be marked as inferred.

  What *is* solid: the manifests are sha-verified, the effective batch and single-GPU pin
  were asserted in-run rather than assumed, and E1's INT8 headline is re-scored on the Mac
  (`scripts/score_int8_local.py`), so the **evaluation** is fully specified even though
  the **training** is not. `train_env()` records all of the above from 2026-09-09 onward;
  see §3ak for how a host change is handled going forward.
- **Falsification:** mean macro-F1 **< 0.80** across 3 seeds ⇒ Tier 0 as specified does
  not reproduce the published DeBERTa baseline on LEDGAR; report it and investigate the
  training setup before changing the encoder or dropping the tier. **Negative result
  stays in the report (hard rule 7).**

### E1b — Tier 0 at 10 epochs *(registered 2026-09-09, BEFORE the run)*

- **Hypothesis:** E1's shortfall is caused by **undertraining**, not by the encoder or
  the task. E1 ran 3 epochs / 10,686 optimizer steps; LexGLUE's published DeBERTa result
  came from up to 20 epochs at batch 8 — **2.8×–14× more steps**, at lr 3e-5 rather than
  2e-5.
- **Change from E1: EXACTLY ONE VARIABLE.** `EPOCHS` 3 → 10. Batch 16, lr 2e-5,
  max_length 512, fp16, seeds, splits, manifests, selection metric and checkpoint
  selection are all unchanged and byte-identical. **This is deliberate.** A
  LexGLUE-faithful configuration would move epochs, batch size and learning rate
  together; if it cleared 0.80 we would learn that the bundle works and not which part
  mattered, and if it did not we would have spent up to 18 GPU-hours and still hold
  three hypotheses. One variable buys an attributable result.
- **Metric / split:** macro-F1 on `test_3000`, **INT8 headline, FP32 reported alongside**
  — identical to E1, including E1's reason for attaching to INT8.
- **Accept rule:** **macro-F1 ≥ 0.80 on `test_3000`, measured on the ONNX-INT8 artefact,
  mean over seeds 1/2/3.** This is **the same bar as E1, deliberately not a bar chosen to
  match what 10 epochs is expected to reach.** If undertraining is the cause, 0.80
  remains the right target and E1b clears it; if E1b lands at, say, 0.79, that is a
  near-miss to be reported as a near-miss, not a rule to be relaxed afterwards.
- **E1b DOES NOT REPLACE E1.** E1 stays in the report **as falsified** (hard rule 7). Its
  0.80 was anchored to LexGLUE Table 3 before any run and the rule did its job; a rule
  that fails is evidence, not an error. E1b answers a **different question** — *does this
  encoder reach the published baseline when trained closer to the way that baseline was
  trained?* — and both results appear, with this distinction stated wherever either is
  quoted.
- **Staged execution, and the stopping rule is registered now.** **Seed 1 runs alone
  first** (~3.3–7.3 h). Seeds 2 and 3 are scheduled **only if seed 1 moves macro-F1
  materially toward 0.80**. If it does not, **undertraining is not the cause**, E1b is
  reported as not-accepted on one seed with that reasoning, and no further quota is spent
  on this hypothesis. Reporting a 1-seed E1b as a *result* would violate hard rule 2 —
  it is a **gate**, and is labelled as one.
- **`EarlyStoppingCallback` is deliberately NOT added.** `load_best_model_at_end=True`
  with per-epoch eval already gives best-checkpoint selection, so early stopping would
  only save wall clock while adding a registered config change — and E1b's whole design
  is one variable. Declining it is recorded here so the omission is known to be a
  decision rather than an oversight.
- **Falsification:** mean macro-F1 < 0.80 across 3 seeds (or a seed-1 gate that does not
  move the number) ⇒ **undertraining does not account for E1's shortfall**, and the
  investigation moves to the encoder, the fine-tuning recipe, or the task setup. Negative
  result stays in the report (hard rule 7).
- **Artefact collision — the trap this run would otherwise walk into.** E1's outputs are
  named `tier0_ce_seed*.json`, `logits_ce_seed*.npz`, `fp32_ce_*`. Re-running with
  `LOSS_ARM = "ce"` would find E1's completion markers and **skip every seed**, or worse,
  overwrite E1's artefacts with E1b's. E1b therefore runs under a distinct arm tag
  (`ce10ep`), so the two experiments cannot collide in `/kaggle/working` and E1's
  artefacts remain exactly as measured.

### E2 — Tier 0 class-imbalance arms

- **Hypothesis (H3):** Class weighting raises macro-F1 over unweighted CE.
- **Metric / split:** macro-F1 on `test_3000`, 3 seeds per arm. Arm *selection* happens
  on `train_holdout_3000`; only the selected arm is reported on test.
- **Accept rule (SUBSTITUTED from C3, 3 seeds, 2026-09-09):** the arm beats E1 by **≥ 0.0089 macro-F1** on `test_3000`, paired on identical rows.
  - Derived mechanically by `scripts/c3_substitute.py` from the preregistered formula `√2 × 1.96 × seed_sd`, with the C3-measured `seed_sd = 0.003195` — the across-seed macro-F1 std on `train_holdout_3000`, which is C3's registered metric. **No E1, E2 or E3 value was read to produce this number**, and none appears in the script's output.
  - **Uncertainty the rule inherits:** at 3 seeds the true sd lies in roughly [0.0017, 0.0201] (95%, chi-square), so the margin implied by that interval spans [0.0046, 0.0557] — a 12× band. The point estimate is what the formula registers and is what is used; the band is stated because the conclusion is not robust to it.
  - **Regime, per §3ae, written before this number existed:**
    - REGIME A — E2 is comfortably testable. The margin is at or below the 0.03 gain E2 itself calls plausible at 137.7x imbalance, so a real class-weighting effect of that size would be detected.
    - UNCERTAINTY ON THE MARGIN ITSELF: with 3 seeds the true sd lies in roughly [0.0017, 0.0201] at 95%, so the margin implied by that interval spans [0.0046, 0.0557]. A 12x band. The point estimate is what the formula registers, and it is used, but the regime above is not robust to this interval.
  - *Formerly: "NOT YET SETTABLE. [C3-gated — blocking]", registered as the formula above.*
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
     Sonnet-5-alone's. Measurement has made this nearly automatic — the asymptote is
     **0.0031%** of the API line, energy being **1/32,240** of it (§3aa) — so condition 3
     is a **sanity check, not a discriminator**, and E6 rests on conditions 1 and 2.
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

- **⚠ PARTIALLY UNBLOCKED (2026-09-08).** `measured_throughput_rps` (30.23),
  `power_draw_soc_watts` (18.20), `energy_joules_per_request` (0.591) and
  `electricity_cost_usd_per_kwh` (0.0847) are now measured and in `configs/costs.yaml`;
  the energy term is no longer null and the curve in §3aa is computed with it.
  `device_cost_usd` remains **derived, not stored** (59,900 INR ÷ 94.50, per hard rule
  5 and the FX-auditability note). Tier 1 and Tier 2 `per_tier_throughput` entries are
  still null, so the **cascade** curve is not yet computable — only Tier 0's.

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

### E8 — Does few-shot prompting close the Sonnet↔Tier 0 gap? *(registered 2026-09-09, BEFORE the numbers were opened)*

- **Why this exists.** E6's frontier put Tier 0 alone **above** Sonnet-5 alone on
  macro-F1, which is the reverse of what H1, E4, E4b and E6's conditions all presuppose.
  Before any of those is restated, the one alternative explanation that costs nothing has
  to be ruled out: **Stage 1 was zero-shot.** If the gap is a prompting artefact rather
  than a capability difference, the premise is not falsified and nothing should be
  rewritten. `results/fewshot_results.json` — 1,000 Sonnet-5 rows, batch
  `msgbatch_01QbUfqihLGmwG3wkYJVZuMU`, already paid for at $0.6062 — answers it offline.
- **Arms, paired.** Sonnet-5 **few-shot** (8 exemplars, `exemplars_8`) vs Sonnet-5
  **zero-shot** (`stage1_results.json`), **restricted to the same 1,000 rows**. The
  few-shot ids are a verified strict subset of `stage1`'s 3,000 and of `test_3000`'s
  manifest indices, so the comparison is row-for-row, not 1000-vs-3000. Both arms are
  scored over the **identical label-averaging set** — the classes present in the gold of
  those 1,000 rows — because macro-F1 is defined relative to that set and a 1,000-row
  subset does not cover all 100 classes. `classes_averaged` is reported with every
  figure.
- **Metric:** macro-F1 primary, accuracy alongside. Parsing via
  `src.data.schema.parse_response`; label matching EXACT per §3ag; unmatched counted
  WRONG per `src.eval.metrics`.
- **Quantities, defined before they are computed.**
  - `delta_fs` = macro-F1(few-shot) − macro-F1(zero-shot), same 1,000 rows.
  - `gap` = macro-F1(Tier 0 INT8, mean over seeds 1–3, same 1,000 rows) −
    macro-F1(zero-shot, same 1,000 rows). This is what few-shot would have to close.
  - `fraction_closed` = `delta_fs` / `gap`.
  - A **paired bootstrap over rows** (10,000 resamples, seed 20260909, percentile 95%
    CI) on `delta_fs`. Paired because it is the same rows under two prompts; an unpaired
    or across-seed sd is the wrong yardstick and must not be quoted here.
- **Accept rule — the band, mechanical, registered before the numbers.**

  | `fraction_closed` | CI on `delta_fs` | disposition |
  |---|---|---|
  | **≥ 0.50** | excludes 0 | **PREMISE LIVE.** The gap is substantially a prompting artefact. Restatement of H1/E4/E4b/E6 is **deferred**; the correct next step is few-shot at full `test_3000` scale, not a rewrite. |
  | **≤ 0.20** | either | **CAPABILITY, NOT PROMPTING.** Eight exemplars move Sonnet a small amount; the deficit is in the task, not the prompt. Restatement proceeds. |
  | **0.20 < f < 0.50** | any | **INDETERMINATE.** Neither disposition is licensed; more few-shot rows, or a class-covering prompt, before anything is restated. |
  | any | includes 0 | few-shot has **no measurable effect** at n=1,000; treat as ≤ 0.20 and say the CI, not the point estimate. |

- **Stated in advance, and it BOUNDS what E8 can conclude.** `exemplars_8` covers
  **6 distinct classes of 100** (`classes_represented: 6`, `max_class_count: 3`,
  `min_class_count: 0`). Eight examples cannot teach a 100-class taxonomy — 94 classes
  are never shown. So:
  - A **small** `delta_fs` is the *expected* outcome under the capability hypothesis and
    is legitimate evidence for it, in the specific form: *the deficit is not fixed by
    in-context examples at this budget.*
  - A small `delta_fs` **does NOT** establish that no prompt closes the gap. E8 falsifies
    "prompting closes it" **only for the 8-exemplar prompt actually run.** A
    class-covering prompt (≥100 exemplars) is untested and unbudgeted. Any write-up
    sentence that generalises past the 8-exemplar prompt is out of E8's evidence.
  - Under §3d/C4 the 8 exemplars are also expected to *bias* prediction toward their 6
    classes, which on a 100-class macro-F1 can push `delta_fs` **negative**. A negative
    `delta_fs` is a real result, is reported, and is not re-run away (hard rule 7).
- **Falsification:** `fraction_closed ≥ 0.50` with a CI excluding 0 falsifies "the
  cascade premise is dead" and stops the restatement.
- **Cost:** $0.00. Responses are on disk; no API call, no GPU. Nothing is appended to
  the spend ledger.

### E7 — `max_length` latency ablation

- **Hypothesis:** 256 tokens is enough for Tier 0, at materially lower latency.
- **Metric / split:** macro-F1 on `test_3000` (paired, same rows) and measured p50/p95
  latency locally.
- **Accept rule:** 256 loses **< 0.02 macro-F1 paired** AND is **≥ 1.5× faster** at p50.
- **Reasoning for the numbers — AMENDED 2026-09-09, number unchanged.** This paragraph
  previously justified 0.02 with *"only **1.2–1.6%** of clauses exceed **512** tokens
  (measured, `results/data_report.md`), and the p50 clause is ~100 tokens, so most rows
  are unaffected either way"*. **That is the wrong threshold's number.** E7 is about
  **256**, and `data_report.md` has only an `over 512` column — no over-256 figure was
  ever measured when this rule was written. Measured now, on the full splits:

  | threshold | train rows truncated | test rows truncated |
  |---|---|---|
  | 512 | 1.487% | 1.420% |
  | **256** | **12.075%** | **11.140%** |

  (The over-512 figures cross-validate `data_report.md`'s 2,000-row sample, 1.2% / 1.6%.)

  So **"most rows are unaffected either way" is true at 512 and false at 256** — roughly
  **8× more rows** are truncated than the cited figure implied. The p50 clause at ~100
  tokens is still unaffected, but the claim that the tail is negligible does not hold.

  **0.02 is retained**, because it was never derived from that percentage: it is the
  point at which the deployed system's accuracy would need reporting as a materially
  different number, the same standard E3's 0.01 is set by. What changes is that 0.02 is
  now understood as a **real risk of being exceeded** rather than a formality — the
  argument for it is a decision threshold, not "the tail is small". 1.5× is the smallest
  speedup that would change a deployment decision, unchanged.
- **The per-class note is now load-bearing, not a caveat.** At 512, `Capitalization`
  (n=434) already truncates at **29.49%** — the worst of any class, and 20× the corpus
  rate. At 256 the concentration is worse. A uniform-looking aggregate delta under 0.02
  can therefore hide one class collapsing, which is exactly what macro-F1 is supposed to
  catch. **Report per-class deltas; an aggregate-only E7 result is not a result.**
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

- **Accept rule (REGISTERED 2026-09-08):** the resulting escalation rates differ by
  **≤ 3 percentage points** — **this is the binding clause** — AND the two thresholds
  agree within **1 decile of the routing signal's distribution** on the full 10k
  validation split, as a **rank-invariance sanity check**.
- **Reasoning, anchored not picked.** A threshold is only meaningful through the
  escalation rate it produces, so the rule is stated in both units. `dev_2000` is drawn
  from `validation` (its manifest records `split: validation`), so the two sets are
  **nested**, and the sampling SD of the escalation-rate difference is
  `sqrt((1 - n/N)^2 * p(1-p) * (1/n + 1/(N-n)))` with n=2,000, N=10,000:

  | escalation rate p | SD of the difference | 3pp expressed in SD |
  |---|---|---|
  | 0.2 | 0.80pp | 3.8 SD |
  | 0.3 | 0.92pp | 3.3 SD |
  | 0.5 | 1.00pp | 3.0 SD |

  **3pp is ~3 SD of that difference under the null that both sets calibrate to the same
  threshold** — computed, not chosen for looking reasonable. Derived independently twice
  before registration.
- **The decile clause is a sanity check, not a second test, and is labelled so.** The
  empirical quantile rank at n=2,000 has SD ~1pp, so one decile is **~9-11 SD** and can
  never bind before the 3pp clause does. It is kept because expressing the tolerance in
  rank terms is what makes it survive any monotone rescaling of the signal (temperature
  included) — but reporting the rule as two independent conditions would overstate what
  it tests.
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
- **Accept rule (REGISTERED 2026-09-07, adopted VERBATIM 2026-09-08):** few-shot
  verbalized-confidence **sd ≤ 0.5× zero-shot sd**
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
- **RECORDED LIMITATION — the threshold is conservative, and that is a Type II risk.**
  At n~1,000 few-shot rows the sampling spread of an sd *ratio* is about +/-4.4%
  (chi-square, 95%), so 0.5 sits ~11 SD from unity: even a ratio of 0.9 would be
  detectable. A tighter threshold would have been defensible **had it been set before the
  run**. It was not; the Stage 1 and few-shot results already exist on disk, and
  tightening it now would be fitting the rule to the answer (see 3ad). **The rule stands
  at 0.5, and this note is the accounting**: a null result here rules out a *large*
  anchoring effect, not a small one.
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
- **Accept rule (REGISTERED 2026-09-07, adopted VERBATIM 2026-09-08):** the class
  appearing 3× in `exemplars_8` is predicted in few-shot at **≥ 2× its zero-shot
  prediction rate** on the same rows, AND its shift is
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
  - **Procedure, since both runs are already on disk:** read the **zero-shot** prediction
    rate of the 3x class from `results/stage1_results.json` and record which form of the
    rule applies, in this file, **before opening any few-shot prediction.** The zero-shot
    rate is the trigger's only input, so reading it does not contaminate the test; the
    few-shot number is what must stay unseen until the form is fixed.
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

| 6 | `AutoQuantizationConfig.arm64(...)` chosen "because the artefact is deployed on the Apple Silicon Mac Mini" | naming a target ISA when quantising is ordinary practice, and the factory is literally named for the ISA | **the rationale was never true.** `.arm64`, `.avx512` and `.avx512_vnni` are identical in every parameter (QInt8 weights, QUInt8 activations, `reduce_range=False`, `per_channel=True`); the factory emits the same bytes. A choice that changed nothing was recorded, in a code comment and in RUNNING.md, as a deliberate deployment-driven decision — so the real question (which *kernel* executes the graph) was never asked | Block 5, diagnosing the INT8 collapse |

| 7 | scaling a wall-clock estimate by **optimizer steps** when the batch size also changes | steps are the natural unit for "how much training happened", and for a fixed batch size the two are proportional | **training time tracks SAMPLES (rows x epochs), not steps.** At batch 8 you get 2x the steps in the same wall clock. Scaling by steps double-counted the batch-size change and inflated a LexGLUE-faithful config from ~6.7x to 14x, producing a confident **"3 seeds — impossible"** verdict on an option that is merely expensive. Nothing failed; a plausible table with a wrong column drove an options recommendation | Block 5, costing the E1b options |

| 8 | shipping a fix into the repo while a run using the OLD code was still in flight | a fix is landed as soon as it is written and tested; the runs are separate | `results/` ended up holding **six bench files from two code versions**, distinguishable only by correlating file mtimes against a terminal scrollback. Nothing in any file recorded which code wrote it. One artefact's first-batch measurement (seed 1, 29.63 rps) was **overwritten twice** and no longer exists, after that number had already been used in a summary | Block 5, the throughput replication |

| 9 | an experiment's environment **printed** to the console instead of persisted to its artefact | printing versions at the top of a run is standard practice and looks like diligence; the operator sees it and moves on | E1's per-seed JSONs record hyperparameters and metrics but **no GPU, CUDA, cuDNN, driver, torch version or TF32 state**. `transformers 4.57.6` was printed and never written. The runs are complete and the session closed, so **the loss is permanent** — E1 can never be attributed between a training-setup difference and a hardware difference | Block 5, planning E1b's host |

**Instance 9 is instance 8's shape with the repair removed.** Instance 8's bench files
could not say which code wrote them; that was fixable by re-running ten minutes of
benchmarks. Here the runs cost GPU-hours, are finished, and the terminal that held the
only copy is closed — **so the mitigation is prospective only and the gap is a permanent
property of E1.** Recorded in E1's own entry as a limitation, not merely as a constraint
on E1b, because a reader of E1 must learn it from E1.

**The deceptive part is that printing looks like recording.** A run that prints its
versions appears instrumented; the operator watching the log sees exactly the information
they would want. The defect is invisible until someone asks the question *after* the
session ends — which is precisely when provenance is needed and precisely when it is
gone. §3e's other instances degrade a value; this one degrades a value's *lifetime*.

**Mitigation, now enforced:** `train_env()` writes the training host into every seed's
JSON, and unavailable fields are `null` rather than guessed. Standing rule: **if a fact
would be needed to interpret a result later, printing it is not recording it.**

**Instance 8's defect is not the overwrite — the overwrite is instance 5's shape again.
It is that the artefacts could not say what produced them.** The collision was fixable
and was fixed twice; what made this one expensive is that *after* it happened, the only
way to reconstruct which file came from which code version was a human's terminal
scrollback. An artefact whose provenance lives outside the artefact is not evidence once
the session ends, and every rule in this document about re-derivable numbers depends on
artefacts being evidence.

**Aggravating factor, recorded because it is the reusable part:** the fix that was
shipped mid-flight was itself a fix for the *previous* round of the same collision. Two
renames did not end it; the third change added a REFUSAL to overwrite, which does. A
naming scheme only defends against the collisions its author anticipated.

**Mitigation, now enforced rather than intended.** `src/serve/bench_local.py` records
`code_commit` and `code_dirty` in every result, and `scripts/bench_aggregate.py`
**refuses** to aggregate runs spanning more than one code version, or any run carrying no
version at all — a mean over two code versions describes neither. `code_dirty` is
recorded alongside the sha because a dirty tree means the commit identifies an ancestor,
not the exact code, which is the state the six mixed files were produced in.

**Instance 7 is the same class in an ANALYSIS rather than in code**, which is why it is
recorded here rather than dismissed as arithmetic. It threw no error, produced a
well-formed table, and its output was *directionally* right — more epochs cost more —
which is what made it survive a first reading. What it did was **remove an option from
consideration**: "impossible" is a verdict, and a verdict reached from a miscomputed
number is indistinguishable, at the point of decision, from one reached correctly.

It was caught only because the same table was rebuilt from a different starting point a
turn later and the two disagreed. That is not a repeatable mitigation. The transferable
one is narrower: **a derived quantity must be scaled by the unit it is actually
proportional to, and the proportionality must be stated when the scaling is written** —
"time is proportional to samples" would not have survived being written next to a column
scaled by steps. This is instance 3's shape (`zero_division=0` deflating in proportion to
absent classes) moved from a library default into our own arithmetic: **a quantity
silently scaled by the wrong denominator.**

Recorded at Sid's instruction, about my own analysis, on the same standing as the C4/C2
entry that records his.

**Instance 6 is a new sub-class: a false rationale attached to a harmless choice.**
Instances 1–5 are all *actions* that silently did the wrong thing. Instance 6 is an
action that did nothing at all, wearing an explanation that made it look load-bearing.
That is worse than an unexplained choice, because a documented rationale is not
re-examined: the comment answered "why arm64?" convincingly enough that nobody asked
"does this flag do anything?" — and the answer, across all **15** dataclass fields,
was no. Verified here rather than asserted: `arm64`, `avx512` and `avx512_vnni` compare
equal on 15/15 fields; `avx2` differs on exactly one, `weights_dtype` (QUInt8 vs QInt8).

**It also cost a diagnosis.** Believing the export was arm-specific made "we quantised
for the wrong ISA" the natural first hypothesis for the INT8 collapse, and it is false.
The graph is not arm-specific; what differs between Kaggle and the Mac Mini is the
*runtime kernel*, and the environment that would have shown this — `onnxruntime`
version, CPU model, `avx512_vnni` — was recorded nowhere. The false rationale did not
merely fail to help; it pointed at the wrong layer, which is the same failure as
instance 3 in §3p ("a check that validated the wrong layer").

**Mitigation, and it is not "comment more carefully".** A rationale asserting that two
options differ must be accompanied by the comparison that shows they differ — the same
standard §3ab imposed after a confident comparison between quantities never put side by
side, and the same one that retracted the `Indemnity`/`Indemnifications` collision claim
in §3ag. Three retractions now share this shape: **a difference asserted without the two
things ever being placed next to each other.** Where the comparison is cheap — and
printing two dataclasses is cheap — it is now required before the rationale is written.

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

**A SECOND NEW CLASS — revising a rule whose data already exists (2026-09-08).**
Recorded at Sid's instruction, about his own request.

The ask was to write concrete accept rules for C1, C2 and C4 before Tier 0's results
landed, on the grounds that they were unwritten and hard rule 6 was about to be breached.
**They were not unwritten.** All three carried numbers and reasoning, committed at
11:36-11:43 UTC on 2026-09-07 — **six hours before** the runs they score were submitted
(few-shot 17:42 UTC, Stage 1 17:43 UTC). Hard rule 6 was already satisfied. The
timestamps read `17:06 +0530`, which is what made them look later than the runs.

**Acting on the request as stated would have converted a satisfied hard rule 6 into a
breach.** C2 and C4 score the Stage 1 and few-shot results, which have been complete and
polled on disk since 2026-09-07. Re-anchoring their thresholds now — however much better
the new anchor — is fitting the rule to data that already exists. The revision would have
*looked* like an improvement in rigour while being the exact thing preregistration
prohibits.

**The generalisable part, and it is the useful bit:**

* **"The rule is marked DRAFT" is not evidence that it is unwritten.** A label describing
  the author's confidence is not a fact about the registration. The commit history is the
  fact.
* **The check before revising ANY accept rule is whether the data it scores exists yet —
  not how well-reasoned the revision is.** Quality of reasoning is irrelevant to this
  question; only chronology matters. A better-anchored threshold chosen after the data
  exists is worse than a cruder one chosen before.
* Where the two split apart, chronology wins: C1 was legitimately revisable because Tier 1
  has produced nothing, so its 3pp got a real sampling-theory anchor. C2 and C4 were
  adopted **verbatim**, with C2's now-known conservatism recorded as a Type II limitation
  rather than repaired.

**Relation to the other entries.** Every §3e instance before the false positive is a real
defect presenting as normal operation; the false positive is the inverse. This one is a
third shape: **a correct-looking process improvement that would have destroyed the
property it was meant to protect.** All three share a root — acting on a plausible reading
of the situation without checking the fact it depends on — but this is the only one where
the damage would have been to the method rather than to a measurement.

**A NEW CLASS — the false positive (2026-09-08).** Every instance above is a **real
defect that presented as normal operation**. This one is the inverse: **normal operation
flagged as a defect.**

Reviewing the measured E6 energy inputs, I divided **load** SoC power by throughput —
18.20 W ÷ 30.23 req/s = 0.6021 J/req — compared it against the reported **0.591**, found
a 1.87% gap, and recorded an "internal inconsistency" in `configs/costs.yaml` and in
§3aa, instructing that it be resolved on the next measurement.

**There was no inconsistency.** The harness reports **marginal** power, which is the
correct quantity:

    (18.20 − 0.32) / 30.23  =  17.88 / 30.23  =  0.59147 J/req      vs 0.591 reported

Exact agreement. I compared two quantities that were never meant to be equal, and 0.6021
is a number nobody had claimed. The measurement was sound; **the divisor was wrong.**

**This is not harmless, which is why it gets its own class rather than a footnote.** A
spurious flag on a measured input to E6 discredits a sound measurement: the next reader
finds a "known inconsistency" annotation on the energy term and reasonably discounts the
whole energy analysis, or spends a re-measurement resolving a discrepancy that does not
exist. False positives spend the same trust and attention that real findings need, and
this project's entire method rests on flagged findings being worth reading. A checker
that cries wolf degrades every other check in the file.

**The mechanism, and the guard.** A derived quantity is only checkable against the
**baseline it is defined against**, and that baseline was never written down — the field
was a bare `power_draw_soc_watts: 18.20`, which does not say whether it is load, idle or
marginal. The name admitted both readings and I took the wrong one.

Mitigation, now in place:

* the field is **split into three** — `power_draw_soc_watts_idle`, `_load`, `_marginal` —
  so no single name can be read ambiguously;
* `energy_joules_per_request` states **in the file** that it is derived from the marginal
  figure, and why marginal is correct for E6 (idle is already carried by the capital
  term, so charging it per-clause double-counts);
* `tests/test_local_hardware.py` asserts the identity
  `|marginal/rps − energy_j_per_req| < 0.005`, so the comment cannot drift from the
  numbers — **and separately asserts that `load/rps` does NOT equal it**, pinning the
  exact distinction that was missed. Verified both directions: perturbing the energy
  figure to 0.650 fails the identity test with the full arithmetic in the message.

**Standing rule added: state which baseline a derived quantity is defined against before
checking it.** If the definition is not recorded, the check is not possible — and
performing it anyway produces a confident comparison between unrelated numbers. The
prior §3e instances all say *verify the property you care about, not a proxy*; this one
adds that **you must first know what the property is defined against.**

**Fourth retraction — two hypotheses weighted at once, one of which forbids the other
(2026-09-08).** Diagnosing Tier 1's silent commit (§3z), I argued the leading explanation
was that `ProgressCallback` writes through `tqdm` to **stderr** and the Batch log viewer
was not surfacing it — and then, in the same analysis, published a table row reading
*"implied by no step-100 log at 8,520 s → **< 0.0122 it/s**"* and concluded the run
"cannot finish under any cap".

**Those two claims are incompatible.** The throughput bound is derived *entirely* from
the absence of a log line. If the log path is broken — the leading hypothesis — then the
absence of a log line carries **no information whatsoever about throughput**, and the run
may be perfectly healthy at the measured 0.07 it/s. The inference is only valid under the
hypothesis it was offered as an alternative to. I ranked the hypotheses correctly and
then reasoned as though the lowest-ranked one were established.

This is §3e's original shape — *a structural fact inferred from a single observation* —
with an extra turn: **the single observation was an absence, and an absence is evidence
only if the channel that would have carried the signal is known to work.** Ruling out
`dataloader_num_workers`, `dist.barrier()` and the rest from source was sound; converting
silence into a number was not.

**Restated on its actual grounds.** The decision to stop the run was right and the
justification was wrong. It is **not** "the run is provably too slow" — that is
unknowable from here. It is: **3 seeds × 13.4 h against a 30 h weekly quota cannot be run
blind.** Every subsequent commit depends on reading throughput and step progress from the
log. Spending 2.4 h once to buy legibility for all of them is cheap; discovering at the
cap that the instrumentation never worked is not. The heartbeat, the flushed log and
`logging_steps=20` are the deliverable, and the stopped run is what paid for them.

**Standing rule added:** an inference from *absence of output* must state, and check, the
assumption that the output channel works. Where it cannot be checked, the absence is not
evidence and must not be converted into a bound.

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

### 3al. I quoted the wrong yardstick, and the conservative direction is not a defence (2026-09-09)

Reporting E6's frontier I said the best Pareto point's **+0.0063 macro-F1 over Tier 0
alone** is *"inside one seed sd (0.0034-0.0056)"*. **Two things are wrong with that
sentence, and they are different in kind.**

**The arithmetic.** 0.0063 is not inside 0.0034-0.0056. It is **1.1x to 1.9x** it. The
claim was stated as though the reader could check it, and it does not check.

**The yardstick, which matters more.** Even with the arithmetic fixed, the across-seed sd
is **the wrong quantity for this comparison.** Tier 0 alone and Tier 0-plus-escalation are
scored on **the same 3,000 rows with the same 3 seeds**, differing only in whether a row
was escalated. An across-seed sd measures how much **retraining** moves the score; the
question being asked is how much **escalating these rows** moves it. Those are different
sources of variance, and the shared row-draw noise that dominates both arms **cancels in
the difference** — which a paired test exploits and an unpaired sd throws away.

**Being conservative did not make it safe.** The wrong test happened to point at the
cautious conclusion, so the error produced no visibly bad claim, which is exactly why it
would have survived. A number that is right by luck cannot be relied on the next time the
luck runs the other way — and here it very nearly did.

**The paired test, run** (`scripts/cascade_paired_bootstrap.py`, 10,000 resamples over
rows, seed 20260909, dev-calibrated threshold **held fixed** while test rows are
resampled so hard rule 1's dev/test separation cannot leak through the resampler):

| quantity | value |
|---|---|
| cascade − Tier 0, paired | **+0.0063**, 95% CI **[−0.0004, +0.0103]**, p = 0.0756 |
| across-seed sd of Tier 0 (the wrong yardstick) | 0.0056 |

**The conclusion survives, and the reason for it changes completely.** The interval
includes zero, so the cascade's gain is not distinguishable from zero — but only just, at
p = 0.076, with the lower bound at **−0.0004**. The paired interval is far **tighter**
than the unpaired spread implied: the correct test is more powerful, and it puts this
result at *borderline*, not at *comfortably null*. Reporting it as "well inside the noise"
would have been a second wrong claim in the opposite direction.

**Standing rule.** When two arms are evaluated on the same rows, the reported interval is
a **paired** one. An across-seed sd may be quoted only for a question about seeds. The
error-in-the-safe-direction is recorded here rather than quietly fixed, because hard rule
7 covers my own mistaken statements and not only failed experiments.

### 3ak. Running E1b on a different host — what it costs (2026-09-09)

E1b may have to run on different hardware or a different account. §3ah exists because one
host's INT8 *kernel* differed from another's; the same exposure applies to **training**
and was never instrumented.

**THE BASELINE'S ENVIRONMENT WAS NEVER RECORDED. This is the binding constraint.**
E1's per-seed JSONs carry `epochs`, `lr`, `max_length`, manifest shas and metrics — and
nothing about the GPU, CUDA, cuDNN, driver or torch version that produced the weights.
Only `transformers 4.57.6` was ever *printed*, to a terminal. So the problem is not
symmetric: instrumenting the new host does not make the comparison sound, because **there
is nothing on the E1 side to compare the record against.** §3e instance 8's lesson
(provenance must live in the artefact) in the one place this project had not looked.

`train_env()` now records the training host in every seed's JSON. It cannot retroactively
record E1's.

**What must be recorded — and the one most likely to bite.**

| field | why it can change macro-F1 |
|---|---|
| `gpu_name`, `gpu_capability` | different SM arch ⇒ different kernels; T4 has tensor cores, P100 does not |
| **`tf32_matmul`, `tf32_cudnn`** | **on Ampere+ TF32 silently reduces matmul precision by default; T4 has no TF32 at all.** Two hosts can run identical code at different arithmetic precision with nothing in the output saying so |
| `torch`, `cuda_runtime`, `cudnn`, `driver` | kernel and algorithm selection, reduction order |
| `cudnn_benchmark`, `cudnn_deterministic` | benchmark picks algorithms by *timing*, so even one host can vary under different load |
| `bf16_supported` | availability differs by arch; `fp16=True` is set, so the fp16 path itself differs across arch |
| `transformers`, `tokenizers`, `datasets`, `numpy` | tokenisation and data-order effects |

**WHAT CANNOT BE MADE COMPARABLE, stated plainly.** fp16 training numerics and cuDNN
algorithm selection are hardware-dependent and **cannot be equalised by recording them**.
Metadata tells you the runs differed; it does not tell you *by how much*. There is no set
of fields that makes a cross-host training comparison sound, and `seed_sd = 0.0032` does
not help — it is a **within-host** training-seed spread, and cross-host variance is
unmeasured with no reason to assume it is smaller.

**What IS host-independent, and it is more than expected.** E1b's accept rule is
**absolute** — macro-F1 ≥ 0.80 on `test_3000`, INT8, mean over 3 seeds — not a comparison
to E1. And the INT8 headline is re-scored on the Mac (`scripts/score_int8_local.py`), so
the *evaluation* is common to both experiments even when the *training* is not. A host
change therefore does not threaten E1b's accept rule. It threatens E1b's **diagnostic
purpose** — "does undertraining explain E1's shortfall?" — and the **seed-1 gate**, both
of which are comparisons to E1.

**REGISTERED PROTOCOL for a host change.** Do not attempt to correct for a host delta.
**Move the comparison onto the new host:**

1. Run **E1-baseline seed 1** on the new host: `EPOCHS = 3`, everything else identical,
   `RUN_TAG = "ce_hostB"`. Cost **1.0–2.2 h**.
2. Run **E1b seed 1** on the same host. Cost **3.3–7.3 h**.
3. The gate becomes **within-host**: does 10 epochs beat 3 epochs *on this host*? That is
   the one-variable question E1b was designed to ask, and it is answered without any
   cross-host term.
4. Report the new host's E1-baseline against the original 0.7636 as a **measured host
   effect at n=1**, explicitly labelled as having no variance estimate.

**The cost of the host change is therefore +1.0–2.2 h, roughly +30–45% on E1b's seed-1
gate.** That is the price of keeping the comparison one-variable, and it is cheaper than
any analysis that tries to reason across hosts after the fact.

**Rejected:** running E1b alone on a new host and comparing to E1's 0.7570. That is a
two-variable comparison (epochs AND host) reported as one, and no recorded metadata
converts it back into one variable.

### 3aj. Tier 0 throughput replication — the rule, registered BEFORE the runs (2026-09-09)

Three trained INT8 artefacts benchmarked once each gave 29.63 / **27.19** / 30.24 rps,
with seed 2 also showing p95 118.7 ms against 96.2 / 94.8. The graphs are
architecturally identical, so this cannot be a property of the model.

> **CORRECTION (2026-09-09), and the reason this rule is unchanged by it.** The seed-1
> figure quoted above, 29.63 rps, **no longer has an artefact on disk**: its file was
> overwritten twice by a mid-flight code change (§3e instance 8), and it survives only in
> a terminal scrollback. It is retained in this paragraph as the *motivating observation*
> that prompted the rule, explicitly labelled unreproducible, and **no number in this
> paragraph may be carried into `costs.yaml`, §3aa or any reported figure.** The rule
> below was registered on the *structure* of the problem — n=1 per artefact cannot
> separate a slow graph from a slow run — and that reasoning does not depend on the
> particular values, so the rule stands as registered. The 9 fresh runs supply every
> number that gets used.

**Three estimators were offered and all three are rejected, for one reason.** At n=1 per
artefact, "this graph is slower" and "this *run* was slower" are not separable — and only
the second is possible. Choosing between mean, median, and re-running the low sample is
picking an estimator to solve a measurement problem.

- **Mean of 3** carries a contaminated sample into the headline and reports an sd
  inflated by contamination rather than by the device.
- **Median of 3** discards information, has no estimable uncertainty at n=3, and drops
  the outlier without ever stating that it did.
- **Re-running seed 2 alone is outlier-hunting**, even though the outlier is probably
  spurious. Re-running only the sample one dislikes and keeping the preferred result
  cannot distinguish "contention fixed" from "drew again from a wide distribution".
  *(Offered by Sid and withdrawn by him on this reasoning; recorded because the withdrawn
  option is the instructive one.)*

  **DEMONSTRATED, not argued (2026-09-09).** Under replication artefact 2's mean is 27.60
  against 28.50 / 28.98, and `F(artefact) = 1.04` on df=(2,4) — the 27.19 that started
  this was a run, not a graph. Had seed 2 been re-run alone it would almost certainly
  have returned ~27.6-28.5, and **that number would have been uninterpretable**: it could
  not distinguish "the contention is gone" from "the second draw landed higher", because
  a single artefact measured twice still has no estimate of run-to-run variance to
  compare against. The replication supplies exactly that estimate, which is the whole
  difference between the two designs.

**REGISTERED RULE — replication, not selection.** Each of the three artefacts is
benchmarked **3 times (9 runs)**, and:

1. **E6 uses the mean over ALL 9 runs, with the sd over all 9.** No run is excluded for
   being an outlier.
2. **Exclusion requires an identifiable, stated cause** (a logged thermal throttle, a
   known contending process), is recorded with that reason, and **applies symmetrically
   to fast and slow runs**. `scripts/bench_aggregate.py` enforces this: `--exclude`
   without `--exclude-reason` is refused.
3. **The between-artefact / within-artefact variance split is reported every time.** It
   is a TEST, not a summary: §3aa's weight-independence claim predicts the between term
   is indistinguishable from run-to-run noise. A ratio > 2 contradicts a claim E6's
   provenance now rests on, and `costs.yaml` is not updated until it is explained.
4. `costs.yaml` is updated in **one edit** after the 9 runs, including the recomputed
   `assumed_lifetime_requests`, so no derived field lags its input.

**Scale check, stated in advance so the rule is not mistaken for a claim that this
matters.** Across every candidate estimator `assumed_lifetime_requests` moves 1–4%, and
V* is capital-dominated (§3aa: a 5× throughput loss moves it 0.012%). **No E6 conclusion
turns on this choice.** The replication is run because the provenance should be clean,
not because the frontier is at risk.

### 3ai. E5 RESULT — rank instability at BOTH precisions is the finding (2026-09-09)

**Lead with this, because the tempting summary is the wrong one.** The four signals are
not merely close; **their ordering is not reproducible across seeds**, and the single
piece of evidence that could have argued for abandoning the registered signal existed
only at the precision that is not deployed.

| signal | INT8 (deployed) | ranks | FP32 | ranks |
|---|---|---|---|---|
| max_softmax | 0.8636 ± 0.0048 | **[1, 2, 1]** | 0.8646 ± 0.0033 | **[1, 1, 1]** |
| neg_entropy | 0.8621 ± 0.0068 | [3, 1, 2] | 0.8624 ± 0.0041 | [3, 3, 2] |
| margin | 0.8602 ± 0.0042 | [2, 3, 3] | 0.8622 ± 0.0035 | [2, 2, 4] |
| trained_difficulty | 0.8564 ± 0.0027 | [4, 4, 4] | 0.8597 ± 0.0023 | [4, 4, 3] |

**On FP32, `max_softmax` held rank 1 on all three seeds — the only basis on which a
switch away from `margin` could have been argued. On the deployed INT8 artefact that
disappears: `[1, 2, 1]`, and NO signal holds rank 1 on every seed.** Had E5 been reported
on FP32 alone, a post-hoc case for `max_softmax` would have looked considerably stronger
than the evidence supports.

**The registered clause fires identically on both precisions:**

> Spread < 0.05 ⇒ signal choice does not matter; report that and use margin for its
> temperature-invariance property alone.

Spread of means: **0.0072 (INT8)**, 0.0050 (FP32) — both far below 0.05. Best AUROC
≥ 0.75 is met at both. So the clause's *premise* is confirmed twice, and its instruction
is unchanged.

**Decision-irrelevance, measured on the decision variable rather than on AUROC.** The
largest difference in retained Tier 0 accuracy between any two signals, at any escalation
rate from 5% to 50%, is **0.0035** — inside seed noise and invisible on the frontier.

**Reported, and labelled as such:** `max_softmax` is *reliably* higher than `margin` on
FP32 (paired +0.00227 / +0.00211 / +0.00295, sign-consistent 3/3) and *negligibly* higher
— 4.9% of the spread the rule requires before calling it a finding. Both are true. It is
recorded as a post-hoc descriptive observation, **not** a selection.

**`trained_difficulty` is stably last ([4,4,4] on INT8), and the cause is identified.**
Cross-fitting is correct — `StandardScaler` is fit inside each fold, so there is no leak.
The diagnostic is that its **in-sample** AUROC is *below* `max_softmax` on 2 of 3 seeds,
even though `max_softmax` is one of its own input features and coefficients (1,0,0,0,0,0)
would reproduce it exactly. That is not overfitting: `LogisticRegression` maximises
log-likelihood while E5 scores **AUROC**, so nothing forces a likelihood-optimal
combination to out-rank its best input. The supportable claim is therefore narrower than
"a learned combination does not help": **a log-loss-fitted linear combination of these six
features does not out-rank max-softmax.** A rank-based objective is a new experiment, not
a fix.

### 3ah. E3 discriminator — registered BEFORE the arms are scored (2026-09-08)

INT8 measured at chance on Kaggle (macro-F1 0.0037 / 0.0030 vs FP32 0.7636), while the
identical export reproduced locally on arm64 agreed with FP32 at r=+0.9982, 8/8 argmax.
Three candidates survived and none could be separated from the saved artefacts, because
the FP32 ONNX graph was deleted immediately after quantising — leaving only torch-FP32
and INT8, which differ in **two** steps at once (export *and* quantisation).

**The discriminator is diagnostic. It does not rescue E3.** E3's falsification clause
has already fired at a delta of −0.7599 against a 0.01 tolerance; the INT8 number is the
headline and reporting 0.7636 as the system's accuracy stays prohibited. What the
discriminator decides is *what re-verification means*, not whether E3 passed.

Three arms, identical rows (`test_3000`), one code path (`onnx_predict`):

| arms compared | isolates | outcome registered in advance |
|---|---|---|
| torch-FP32 vs ONNX-FP32 | the **export** | disagree ⇒ the export is the locus; the quantiser is exonerated |
| ONNX-FP32 vs ONNX-INT8 | the **quantised kernel** | disagree ⇒ the kernel is the locus |
| both agree | neither | the collapse is in the harness or the scoring join, not the artefact |

Recorded with them: `onnxruntime` version, `get_available_providers()`, CPU model, and
`avx512_vnni` / `avx512f` / `avx2` from `/proc/cpuinfo`. **The environment was recorded
nowhere before now**, which is why the first failure was un-diagnosable from its own
output. Unmeasurable fields are written as `null`, never `False` (hard rule 11).

**Deployment is arm64, and that is the point.** The local arm64 reproduction of this
exact export preserved the function. So if the locus is the quantised kernel on a
non-VNNI x86 host, **the broken thing is the Kaggle measurement environment, not the
serving path** — Tier 0 may deploy INT8 correctly on the Mac Mini while every INT8
number this project has recorded is garbage. That is not a lesser finding: it means
E1's INT8 headline, E3's delta and E5's INT8 dev logits must all be re-scored on arm64
via `scripts/score_int8_local.py` before any of them is believed **in either direction**.
A confirmatory arm64 re-score is required even if the discriminator points at x86.

**Registered as the reaction, before the arms are run:** if ONNX-FP32 matches torch-FP32
and only INT8 diverges, the finding is recorded as an *environment* defect and E3 is
re-run on arm64. If ONNX-FP32 *also* diverges, the export is implicated and the arm64
local result becomes the anomaly to explain, not the reference. Neither outcome permits
E3 to be scored as met.

### 3af. The offline scorer is authoritative; the notebook's macro-F1 is a convenience number (2026-09-08)

Both Kaggle notebooks compute macro-F1 inline with a bare
`f1_score(..., zero_division=0)` call. That is the library default §3e instance 3 records
as wrong for this project, it reports no `classes_averaged`, and it bypasses
`src.eval.metrics.score` and its `allow_absent_classes` guard entirely.

**Neither running notebook is being changed.** Instead the convention is settled by
making the offline path authoritative:

* **E1, E2 and E3 are scored by `src/eval/score_tier0.py`**, from the `.npz` logits,
  through `src.eval.metrics.score`, with `allow_absent_classes` passed **explicitly** and
  `classes_averaged` recorded on every result. Each result carries
  `"scored_by": "src.eval.metrics.score (AUTHORITATIVE)"`.
* **The notebook's in-run number is a CONVENIENCE figure** — it exists so a running seed
  prints something legible — and is labelled as such wherever it appears.

**The two conventions are identical on `test_3000`, so E1's and E4's headline numbers are
unaffected.** `test_3000` and `train_holdout_3000` both record `classes_represented: 100`
with `min_class_count: 1` in their committed manifests, so no class is averaged in at 0.0
by either route. **The divergence is confined to `dev_2000`** (`classes_represented: 99`,
`min_class_count: 0`) and therefore to **router threshold calibration**, never to a
reported accuracy.

**Demonstrated, not asserted** — `tests/test_score_tier0.py`:

| test | shows |
|---|---|
| manifest coverage | reads 100/100/99 from the committed manifests, so a manifest change breaks the claim |
| all classes present | the two numbers agree on `test_3000` |
| a class absent, default label space | they still agree at 99/100 — **absence alone does not cause divergence**; what changes is `classes_averaged`, which the notebook never reports |
| full label space | averaging dev over all 100 deflates macro-F1 by **exactly 99/100** — the §3e instance 3 failure, reproduced |
| the flag never defaults | `score` raises without `allow_absent_classes` |
| E1 headline | unchanged under the switch |

**One nuance the tests surfaced, worth stating precisely:** the two numbers are equal to
floating-point precision but **not bit-identical** — the routes average the same per-class
F1 values in a different *order* (`sorted(set(gold))` versus the label list), so the sums
differ in the last ulp (~1e-16). "Arithmetically identical" is the correct claim;
"byte-identical" would have been false. The tests assert agreement to 1e-12, four orders
tighter than anything that could affect a macro-F1 reported to four decimals.

### 3ag. Pluralisation: exact matching is KEPT, and one justification for it was wrong (2026-09-08)

Tier 1 emits a label **string**, matched **exactly** — the normaliser folds case,
whitespace and edge punctuation and does nothing else ("no fuzzy rescue"). LEDGAR's label
names are mostly plural (**73 of 100** end in `s`), so the natural model output
`Governing Law` fails against the label `Governing Laws`, and every such near-miss is
**both a format failure and a wrong answer**.

**DECISION: exact matching is kept.**

**The reason that holds.** `format_failure_rate` is a **measured quantity of this
project**, not an obstacle to a better accuracy number. Folding singular to plural would
repair the metric by deleting the finding — the same reasoning that left the SDK's JSON
schema enforcement unused. A cascade whose middle tier cannot reliably emit a label in
the required surface form is a cascade with a real defect, and E4 is supposed to be able
to see it. The number is the point.

**The reason that does NOT hold, retracted here.** It was argued — by me, and the
argument was then relied on — that folding would be dangerous because *the label space
contains both `Indemnity` and `Indemnifications`, so naive folding would create real
collisions.* **That is false, and I did not check it before asserting it.**
`collision_audit` over all 100 names finds **zero** pairs differing only by a trailing
`s`, and zero under a broader `s` / `es` / `y↔ies` fold as well. `Indemnity` and
`Indemnifications` are different stems and never collide. **A naive plural fold would
create no collisions in this label space.** The decision therefore rests on the
format-failure-rate argument **alone**, which is enough — but the collision claim is
struck, and `tests/test_format_failures.py` pins the zero-collision fact so the discarded
justification cannot creep back. This is §3ab's class again: a confident comparison
between quantities that had not actually been put side by side.

**The cost of the choice is reported, not assumed away.** `src/eval/format_failures.py`
categorises every format failure as `pluralisation` (one trailing `s` from exactly one
real label — what a fold *would* have repaired), `ambiguous` (a near-miss for more than
one label, which no fold repairs without choosing arbitrarily), `case_or_edge` (must be
0; non-zero means the normaliser regressed) or `other` (hallucinated labels, prose,
truncation). **`recoverable_by_plural_fold` is reported with every Tier 1 result**, so the
price of keeping exact matching is a number in the report.

**Measured so far — 8 rows, UNTRAINED adapter (§3-smoke, not a Tier 1 result):**
3 failures of 8, of which **pluralisation 0, ambiguous 0, other 3**
(`Indemnities`, `Information Disclosure`, `Restrictions On Transfer`). On this tiny
untrained sample a fold would have repaired **nothing**. That is 8 rows from a model that
has not been trained; it bounds nothing and is recorded only so the instrument is known
to work before the real predictions land.

### 3ae. C3 substitution — the reaction, registered before the number (2026-09-08)

Tier 0's three CE seeds land in a few hours and will fix `seed_sd`, which is the only
free parameter in E2's registered accept rule. **What each possible value would mean is
recorded here first**, so the reaction is preregistered along with the rule.

`scripts/c3_substitute.py` performs the substitution mechanically:
`margin = √2 × 1.96 × seed_sd`, with `seed_sd` the across-seed macro-F1 std on
`train_holdout_3000` — **C3's registered metric, which is the selection split**. Because
C3's metric is the selection split, the script has no reason to open a test field and
does not. **The sealing is structural, not disciplinary.** A standard deviation carries no
information about the level it was computed around, so printing it reveals nothing about
whether E1 passed; a test asserts that two seed sets with the same spread and very
different means produce byte-identical output.

**Regimes, fixed in advance:**

| margin | seed_sd | regime |
|---|---|---|
| ≤ 0.030 | ≤ 0.0108 | **A — comfortably testable.** At or below the 0.03 gain E2 itself calls plausible at 137.7× imbalance, so a real class-weighting effect of that size is detectable. |
| 0.030 – 0.050 | 0.0108 – 0.0180 | **B — testable only for a LARGE effect.** The margin now exceeds E2's own plausible effect size, so the expected result would be declared not-accepted. E2 is still run and reported, with the margin stated and the fact that a plausible-sized effect sat outside detection stated with it. |
| > 0.050 | > 0.0180 | **C — effectively untestable at 3 seeds.** The margin exceeds realistic headroom above E1 (LexGLUE DeBERTa macro-F1 is 0.831), so essentially no attainable arm could clear it. |
| > 0.083 | > 0.030 | **C3's own falsification line**, registered in C3 before any run: several rules are unfalsifiable as written and must be loosened or moved to paired comparisons. |

**Yes, regime C is possible, and it is named now rather than after the fact.** If it
occurs, running `sqrt_inv_freq` for 3–5 GPU-hours would produce a foregone conclusion.
The registered response is to **report E2 as not testable at this seed count with this
margin**, state the measured `seed_sd`, and treat the GPU hours as available for
something that can still discriminate — *not* to quietly widen the seed count until the
margin shrinks, which would be choosing a threshold after seeing that the first one
failed.

**The margin inherits a 12× uncertainty band, and this is the part most likely to be
forgotten.** The sample sd of **three** numbers is a very noisy estimate: with ν=2,
`(n−1)s²/σ²` is χ²₂, so at 95% the true σ lies in `[0.52·s, 6.28·s]`. Whatever margin
comes out, the interval consistent with the data spans a factor of twelve. **A regime-A
result is therefore not proof that E2 is comfortably testable** — it is a point estimate
that happens to fall in regime A, and the upper end of its own interval may be regime C.
The script prints the band beside the margin and writes it into the substituted rule.

**DECIDED (2026-09-08): `seed_sd` is the POINT ESTIMATE.** The formula names `seed_sd`
without saying whether that is the point estimate or an upper confidence bound, and the
ambiguity is real. It is resolved toward the point estimate, and the reasoning is the
part worth keeping:

**Resolving an ambiguity toward the reading that makes E2 fail — while knowing that is
what it does — is exactly as post-hoc as resolving it the other way.** A one-sided 95%
upper bound on σ is ~1.92× larger and would push most plausible outcomes into regime B
or C, i.e. would make E2 unacceptable-by-construction. Choosing it now, with that
consequence known, would be choosing an outcome rather than a method. The point estimate
is the plain reading of "the across-seed macro-F1 std measured by C3", it is what a
reader of the registered formula would have computed, and it is what the script does.
The upper bound is recorded here as the alternative that was considered and declined,
with its effect stated, so the choice is auditable.

**MANDATORY REPORTING CONDITION.** Because the point estimate was chosen over the
conservative reading, **the χ² band must be reported alongside E2's result every single
time it appears** — in the report, in any summary, and in any statement of whether E2
passed. The required form states both:

> E2 margin = *M* (from seed_sd = *s*, 3 seeds). At 3 seeds the 95% interval on the true
> sd is [0.52·*s*, 6.28·*s*], so the margin consistent with the data spans
> [*M*/1.92, *M*×6.28] — **a regime-A point estimate may have an interval reaching
> regime C.**

A margin quoted without that band overstates its precision by up to an order of
magnitude. `c3_substitute.py` prints the band next to the margin and writes it into the
substituted rule, so the two cannot become separated by accident — but the obligation is
on the *report*, not only on the script.

**Order of operations, which is the whole point:** run the script, read *only* `seed_sd`
and the margin, commit the substituted rule, and **only then** open any E1/E2/E3 number.
The same discipline kept with `stage1_results.json` for C2 and C4 (§3ad).

### 3ad. C1/C2/C4 registered; two of them adopted verbatim (2026-09-08)

Full description in §3e under "A SECOND NEW CLASS". Short form: a request to re-anchor
C1, C2 and C4 rested on the premise that they were unwritten. They were committed
2026-09-07 at 11:36-11:43 UTC, six hours before the runs they score (17:42/17:43 UTC), so
hard rule 6 was already satisfied and revising C2 or C4 would have breached it.

**Outcome, signed off before any edit:**

| check | disposition | why |
|---|---|---|
| **C1** | **reworded and anchored** | scores Tier 1 routing thresholds; Tier 1 has produced nothing, so revision is legitimate. 3pp is now derived as ~3 SD of the nested-sample escalation-rate difference, and the decile clause is relabelled a rank-invariance sanity check because at ~9-11 SD it can never bind |
| **C2** | **adopted verbatim** | scores few-shot vs Stage 1, both on disk. Conservatism of the 0.5x threshold recorded as a Type II limitation instead of being repaired |
| **C4** | **adopted verbatim** | same. The <1% fallback trigger is read from the ZERO-SHOT rate only, recorded before any few-shot number is opened |

No confidence value or predicted label from `stage1_results.json` or
`fewshot_results.json` was read while these rules were being finalised.

**What actually lands with Tier 0 is C3**, which the checks table already records as
descriptive with no accept rule of its own. It gates E2, E3 and E7. E2's rule is
registered as a **formula** — `>= sqrt(2) * 1.96 * seed_sd` — with one parameter C3
measures; substituting a measured value into a preregistered formula is not writing a new
rule, so hard rule 6 holds through that step.

### 3ac. The electricity tariff is ASSUMED, and E6 does not depend on it (2026-09-08)

`electricity_cost_usd_per_kwh: 0.0847` was sitting in `configs/costs.yaml` beside
measured quantities with nothing marking it as unmeasured. **It is a guess** — roughly
Rs 8/kWh at the recorded FX, an estimate of a Rajasthan domestic slab, supplied in
conversation and **never read off a bill.**

This is §3ab's class again — *a value whose basis is not stated* — with the basis being
worse than ambiguous. The power field was ambiguous between load and marginal; this one
is not a measurement at all, and sat in a file whose other entries are.

**Field split, and a refusal rather than a fallback.**

```yaml
electricity_cost_usd_per_kwh_measured: null
electricity_cost_usd_per_kwh_assumed: 0.0847
```

`src/eval/breakeven.py` reads `_measured` first and **raises `TariffUnavailableError`**
rather than reaching for `_assumed`. The assumption is available only to a caller that
passes `allow_assumed_tariff=True` — asked for by name — and every `BreakevenResult`
returned then carries `tariff_is_assumed=True` and a `tariff_basis` string that begins
"ASSUMED". This is hard rule 11 applied exactly: an approximation may be used when it is
legitimately needed, but it must be requested explicitly, marked in the output, and never
substituted silently. The assumed tariff cannot reach `src/api/cost.py`, which is the
billing path, because nothing in that path reads `local_hardware`.

**Sensitivity, in the same form as §3aa's wall-power table.** Reference is V\* with the
energy term excluded entirely (1,413,925):

> **SUPERSEDED 2026-09-09, conclusion unchanged.** The table below was computed with the
> UNTRAINED probe's energy figure (0.591 J/req). The trained artefacts measure
> **0.6269 ± 0.0311 J/req** (§3aa discharge), so the energy term is ~6% larger and every
> V\* shifts by 1–27 clauses on a base of ~1.41M: x0.5 → 1,413,948, **x1 → 1,413,971**,
> x2 → 1,414,018, x10 → **1,414,390 (+0.0329%)**. The table is left as the historical
> record rather than rewritten; the live figures are in `configs/costs.yaml` and are
> pinned by `tests/test_local_hardware.py`. **§3ac's conclusion — that the assumed tariff
> is not load-bearing — is unaffected**, and is if anything better supported: a 10x
> tariff still moves V\* by 0.033%, far inside the 0.05% threshold §3ac set.

| tariff | USD/kWh | energy $/1k | V\* | vs no-energy |
|---|---|---|---|---|
| 0.5x | 0.0423 | $6.952e-6 | 1,413,947 | +0.0016% |
| **1.0x (assumed)** | **0.0847** | **$1.390e-5** | **1,413,969** | **+0.0031%** |
| 2.0x | 0.1694 | $2.781e-5 | 1,414,012 | +0.0062% |
| **10.0x** | 0.8470 | $1.390e-4 | **1,414,363** | **+0.0310%** |

**A tenfold error in the tariff moves the crossover by 439 clauses out of 1.41 million —
0.0310%, comfortably under 0.05%.** Combining the two unmeasured quantities at their
worst — a 10x tariff error *and* a 3x SoC-to-wall correction together — gives
V\* = 1,415,242, **+0.0931%**. Still under a tenth of one percent.

**So the assumption is not load-bearing — but that is a finding, not a premise.** It
would have been easy, and wrong by this project's standards, to write "energy is tiny so
the tariff cannot matter" and move on. The claim is only worth anything because the table
above was computed: §3ab was caused by exactly that kind of confident reasoning about
numbers that had not been put side by side. E6's headline rests on **capital and
volume**; the tariff, the SoC-only power figure, and both together are shown to be
immaterial rather than assumed to be.

**What would change this.** If a measured tariff arrives it goes in `_measured`, the flag
clears automatically (tested), and V\* is recomputed — a difference of at most a few
hundred clauses. If E6's API baseline ever drops by three orders of magnitude, energy
stops being negligible and this table must be recomputed; the guard is that
`breakeven()` **raises** rather than returning a negative or infinite V\* when energy
meets or exceeds the API line, because that case is an E6 result to report, not an error
to swallow.

**Tests (`tests/test_local_hardware.py`, 10 total).** That `resolve_tariff` refuses
without `allow_assumed_tariff`; that **every** result built on the assumed tariff carries
`tariff_is_assumed=True`, across all four multipliers; that supplying a measured tariff
**clears** the flag — without which the flag could be a hardcoded `True` and the previous
test would still pass, verifying a constant instead of a property (§3o); that the four
published V\* values match the code, so this table cannot drift from
`src/eval/breakeven.py`; and that no `device_cost_usd` is ever stored, since hard rule 5
requires it derived from INR and the FX rate.

### 3ab. A false positive on a sound measurement (2026-09-08)

Recorded as its own entry because it is a **new failure class for this project**, and the
full description is in §3e under "A NEW CLASS — the false positive".

**Short form.** I flagged a 1.87% "internal inconsistency" between the measured
18.20 W, 30.23 req/s and 0.591 J/request, and wrote it into `configs/costs.yaml` and
§3aa. It was wrong. The harness reports **marginal** power, and
`(18.20 − 0.32) / 30.23 = 0.59147` matches 0.591 exactly. I divided by **load** power —
a quantity nobody had claimed — and reported the mismatch as a defect.

**Every other §3e instance is a real defect presenting as normal operation. This is the
inverse: normal operation presenting as a defect.** It is not harmless — a spurious flag
on a measured E6 input discredits a sound measurement and spends the attention that real
findings need.

**Root cause:** the field was a bare `power_draw_soc_watts`, which does not say whether
it is load, idle or marginal. A derived quantity cannot be checked without knowing the
baseline it is defined against, and that baseline was never written down.

**Fixed:** field split into `_idle` / `_load` / `_marginal`; the marginal basis and its
justification stated in `configs/costs.yaml`; and `tests/test_local_hardware.py` asserts
`|marginal/rps − energy_j_per_req| < 0.005` **and** that `load/rps` does not equal it,
pinning the distinction. The flag is removed from both files. **E6's numbers are
unchanged** — 0.591 J/req, $1.390e-5 per 1k, V\* = 1,413,969 — because the measurement
was always right.

### 3aa. E6 energy term MEASURED; the crossover barely moves (2026-09-08)

`power_draw_watts` and `electricity_cost_usd_per_kwh` were the last two nulls blocking
E6's energy term. Both are now measured, so **E6's local curve is computed with a
measured per-clause floor rather than without one.**

| quantity | value | source |
|---|---|---|
| throughput | **30.23 req/s** | measured, ONNX INT8, bs 1, max_length 512 |
| SoC power, idle | **0.32 W** | measured, `powermetrics` Combined Power |
| SoC power, load | **18.20 W** | measured |
| **SoC power, marginal** | **17.88 W** | load − idle; **the figure E6 uses** |
| energy | **0.591 J/request** | = 17.88 / 30.23 = 0.59147 |
| electricity | **$0.0847/kWh** | **ASSUMED, not measured — see §3ac** |
| **energy cost** | **$1.390e-5 per 1,000 clauses** | derived |
| Sonnet 5 (batch, cached) | $0.4483 per 1,000 | §1 |
| **ratio** | **1 / 32,240** | derived |

**The crossover, recomputed.** `cost_local(V) = 633,862.43/V + 1.390e-5` per 1,000
clauses, against Sonnet 5's flat $0.4483:

| | V\* |
|---|---|
| energy term excluded (previous figure) | 1,413,925 |
| **energy term included** | **1,413,969** |
| shift | **+44 clauses (+0.0031%)** |

**The shift is UP, not down, and the direction is the whole point.** Adding a positive
per-clause cost to the local curve makes local break even **later**:
`V* = 633,862.43 / (0.4483 − e)`, and `e > 0` shrinks the denominator. So the previous
figure was **an optimistic lower bound** — it understated local cost by omitting a real
term — and the corrected figure sits above it. The conclusion is unchanged (~1.41M
clauses, **17.7x the entire 80,000-clause LEDGAR corpus**, and 1/506 of `V_max`), but it
is now an *inclusive* number rather than a bound. That is the only thing the measurement
bought, and it is worth having: E6's whole design (§1b) is that the energy term is what
stops the local curve tending to zero, and asserting it was negligible without measuring
it would have been the §3e error in its usual costume.

**The asymptote condition is now measured too.** E6's accept condition 3 requires the
high-volume cost to be < 50% of Sonnet-5-alone. The measured asymptote is **0.0031%**.
Condition 3 is a sanity check, not a discriminator; E6 rests on conditions 1 and 2.

**SoC-only limitation, stated separately because it is a real limit on the measurement
and not a caveat on the conclusion.** `powermetrics --samplers cpu_power` reports
"Combined Power (CPU + GPU + ANE)" — **SoC package power only.** It excludes RAM, SSD,
PSU losses, networking and fans. **It is not wall power, and wall power cannot be
obtained from `powermetrics` at all**; it requires an external meter, which we do not
have. So 18.20 W is a floor on the machine's true draw, and 0.591 J/request is a floor on
true per-request energy. The field is named `power_draw_soc_watts` for that reason, with
`power_draw_wall_watts` left null rather than filled with the SoC figure.

**The limitation cannot change E6's answer, and that is demonstrable rather than
asserted:**

| wall correction | energy per 1k | V\* | vs no-energy |
|---|---|---|---|
| ×1 (SoC, measured) | $1.390e-5 | 1,413,969 | +0.0031% |
| **×3** | $4.171e-5 | **1,414,056** | **+0.0093%** |
| ×5 | $6.952e-5 | 1,414,144 | +0.0155% |
| ×10 | $1.390e-4 | 1,414,363 | +0.0310% |

A 3x wall-power correction — generous, since SoC-to-wall on a Mac Mini is typically well
under that — moves V\* by **87 clauses out of 1.41 million**. Even 10x moves it by 438.
**E6 turns on capital and volume; energy is not a lever at any plausible correction**,
and that is now shown rather than assumed.

**Why MARGINAL power is the right baseline, and not merely the one that was measured.**
Idle draw (0.32 W) is incurred whether or not Tier 0 serves a single clause. Charging it
to per-clause cost would double-count baseline machine cost that the **capital term
already carries** — the $633,862.43/V numerator exists precisely because the machine
exists. The incremental term must therefore be the incremental power. The harness made
this choice correctly.

**One input here is not a measurement.** The electricity tariff is an **assumption**,
not a reading from a bill. §3ac states its basis, shows E6's sensitivity to it, and
records the code and tests that stop it reaching the report unlabelled.

**One data-quality flag.** `measured_throughput_rps_sd` is now **null, not 0.24**. 0.24
was the spread of the *28.97* measurement and does not describe *30.23*. Pairing an old
spread with a new mean would misreport the precision of a number E6 depends on. The new
figure arrived without a spread and the field says so.

*(An earlier version of this entry carried a second flag, asserting a 1.87% internal
inconsistency between the power, throughput and energy figures. **That flag was wrong and
has been removed** — it compared LOAD power against a MARGINAL-power derivation. The
three numbers agree exactly. See §3ab.)*

**DISCHARGED 2026-09-09 — weight-independence is now a MEASUREMENT, not an argument.**
This paragraph previously read: *"every one of these numbers was measured on the UNTRAINED
architecture probe... Throughput is weight-independent — identical ops on identical shapes
— so it transfers... This must still be re-verified on the real INT8 artefact when it
exists."* That was an argument from architecture. It has now been tested.

Nine runs — three trained INT8 artefacts x 3 runs, commit `8c5a5ed4`, clean tree, under
§3aj's registered rule (mean over all runs, no outlier excluded):

| field | mean | sd | between/within |
|---|---|---|---|
| throughput_rps | **28.3615** | 1.1839 | **0.74** |
| p50_latency_ms | 25.3969 | 0.8824 | 0.87 |
| p95_latency_ms | 99.7950 | 3.5058 | 0.72 |
| marginal_soc_watts | 17.7546 | 0.5449 | 0.51 |
| joules_per_request | 0.6269 | 0.0311 | 0.37 |

**It passes the strong way: within-artefact variance exceeds between-artefact variance on
every field.** The formal test is `F(artefact) = 1.04` on df=(2,4) against a 95% critical
value of **6.94** — the artefact explains nothing beyond run-to-run noise. Per-artefact
means 28.50 / 27.60 / 28.98.

**The probe is retrospectively validated, with a correction to how comfortably.** It
reported 30.23 rps against the trained mean of 28.3615 — **1.58 sd, over-estimating by
6.6%**. Energy: 0.591 vs 0.6269 ± 0.0311 J/req, 1.2 sd. So every probe-derived E6 figure
was optimistic by ~6.6% on throughput, and `assumed_lifetime_requests` falls 6.18% from
714,999,960 to 670,805,531. **The conclusions are unchanged** — V* is capital-dominated —
but "validated" here means "within noise", not "correct".

> **A retracted figure.** An earlier comparison put the probe at **0.75 sd** from the
> trained mean. That was computed on the contaminated n=3 set of §3e instance 8, one of
> whose values (29.63 rps) has no artefact on disk. **0.75 sd is withdrawn and must not
> be carried anywhere; 1.58 sd is the figure.** The direction of the error matters: the
> retracted number made the probe look more accurate than it is.

**The seed-2 "outlier" was a run, not a graph — and this is now demonstrated.** The
single-measurement set showed 27.19 rps for artefact 2 against 29.63 / 30.24. Under
replication its mean is 27.60 against 28.50 / 28.98, and `F(artefact) = 1.04` says that
spread is noise. See §3aj: re-running seed 2 alone would have produced a number that
could not be interpreted, because at n=1 per artefact "slow graph" and "slow run" are not
separable.

**Run-order drift is present and is carried as a stated limitation.** Block means declined
29.06 → 28.23 → 27.79 over ~10 minutes (4.4%), 22.3% of total variance. It is not a clean
thermal story: **only 2 of 3 artefacts are monotonic in run order**, and `F(run) = 0.87`
does not reach significance. Because the loop interleaved artefacts, drift lands inside
the *within* term, which would flatter the weight-independence ratio — the concern is
real, and modelling the block as a covariate moves the ratio to **0.59**, i.e. *more*
favourable, because the degrees-of-freedom correction dominates. `F(artefact)` is reported
because it does not depend on which of those effects wins.

**What is still not known:** whether 28.3615 describes sustained serving. The measured
window is 10 minutes and had not demonstrably plateaued. At `duty_cycle` 0.25 the device
has idle time to cool, so a duty-cycled box plausibly runs at or above this rate while
sustained serving plausibly runs below it. The mean over all 9 runs is used because the
true operating regime is unmeasured; selecting run 1 or run 3 would be choosing a regime
we have not measured. **A 30+ minute plateau run is what would settle it**, and until then
`assumed_lifetime_requests` inherits this uncertainty.

E1's accept rule attaches to the trained artefact, not to the probe — unchanged.

**Still blocking the full E6 curve:** Tier 1 and Tier 2 `per_tier_throughput` entries
remain null, so only **Tier 0's** curve is computable. The *cascade* curve — which is
what E6's headline claim is about — needs Tier 1 serving numbers that do not exist until
Tier 1 trains.

### 3z. A run with no observable state (2026-09-08)

Tier 1 commit 1 reached 2h22m with **no output at all** after the re-cast line at 328 s.
The first `logging_steps=100` line was due at ~1758 s. The session still showed
"Running". **There was no way to distinguish a hang from a silent log**, which is the
same class as §3e instance 4 — a wait with a success condition and no failure condition —
except here there was no condition at all, only absence.

**Why the log was invisible, from source.** `disable_tqdm` defaults to `False`, so
`ProgressCallback` is active, and its `on_log` writes through **`tqdm.write()` — to
stderr** — with the bar itself `\r`-based. Our own prints use `flush=True` on stdout and
**did** appear. A Batch log viewer that surfaces stdout promptly and stderr late, or not
at all, produces exactly what was seen. `PrinterCallback`, the `disable_tqdm=True`
alternative, is no better: a bare `print(logs)` with no flush into a block-buffered
non-TTY stdout. Compounding it, `logging_steps=100` emits nothing for the first 100
optimizer steps — ~24 min at the measured rate, and unbounded if the rate is wrong.

**What can be ruled out from source, for Tier 1:**

| candidate | ruled out because |
|---|---|
| DataLoader worker deadlock | `dataloader_num_workers` is never passed (0 occurrences; the `_SFTCONFIG_KWARGS` allowlist is cross-checked complete), so it is the default **0** — no worker processes exist |
| distributed barrier | single process, `CUDA_VISIBLE_DEVICES=0`, PREFLIGHT asserts `device_count()==1`, no `WORLD_SIZE`/torchrun in a commit, so accelerate stays single-process and no `dist.barrier()` runs |
| checkpoint load / data skip | `resume=False` on commit 1; RESTORE printed "starting fresh" |
| network after the HF rate-limit warning | `report_to=[]`, and model, tokenizer and dataset all load **before** the re-cast print |
| `torch.compile` | not used anywhere |
| eval during training | `eval_strategy` never set → `"no"` |

**What cannot be ruled out without CUDA:** actual throughput in a Batch container versus
the interactive session where 0.07 it/s was measured; bitsandbytes 4-bit kernel behaviour
on that specific container; whether the viewer surfaces stderr at all; a genuine
CUDA-level stall.

**Ranked, and the ranking is decision-relevant because the hypotheses predict different
outcomes:** (1) log-path invisibility — consistent with every observation, and predicts
the run is near step 573 and will stop cleanly at the 2,000-step budget around 8 h with
retrievable output; (2) genuinely slower; (3) a real hang. **(2) and (3) both predict a
cap timeout and total loss of the commit's output** (§3v).

**The arithmetic that decides it.** The budget is 2,000 steps and setup cost 328 s:

| | required rate |
|---|---|
| finish 2,000 steps inside a 9 h cap | ≥ **0.0624 it/s** |
| inside a 12 h cap | ≥ **0.0467 it/s** |
| measured earlier, interactive | 0.0700 it/s → 8.0 h ✓ |
| ~~implied by no step-100 log at 8,520 s~~ | ~~< 0.0122 it/s → 46 h~~ **RETRACTED** |

**That last row is retracted — see the fourth retraction in §3e.** It derives a
throughput bound entirely from the absence of a log line, while this same entry argues
the leading explanation is that the log channel does not work. Under that hypothesis the
absence carries **no information about throughput at all** and the run may be healthy at
0.07 it/s. The two cannot both be weighted, and the bound was the weaker of the two.

**What is actually known:** the required rates above are real, the measured 0.07 it/s is
real, and **nothing observed distinguishes a healthy run from a stalled one.** That is
the finding.

**Stopping it manually forfeits exactly what a cap timeout would** — neither produces
retrievable output — so "stop to save the output" is not a reason either way.

**The real ground for stopping: 3 seeds × 13.4 h against a 30 h weekly quota cannot be
run blind.** Every subsequent commit depends on reading throughput and step progress out
of the log, and none of them can be steered without it. Spending 2.4 h once to buy
legibility for all of them is cheap. Discovering at the cap that the instrumentation
never worked is not.

**Fix — three parts, and the thread is the one that matters.**

1. **A daemon-thread heartbeat every 2 minutes**, not a Trainer callback. This is the
   whole point: a callback heartbeat only fires when the loop reaches `on_step_end`, so
   it is silent both when training is hung *and* when it is merely quiet — it cannot tell
   them apart, which is the exact failure being fixed. A separate thread keeps printing
   while the main thread is blocked. Silence from it now means the process is dead or the
   log pipe is broken; output from it while `step` does not advance means the loop is
   stuck. Different diagnoses, different fixes. Verified: it prints throughout a
   deliberately blocked main thread, and reports step, it/s and projected hours once
   training begins.
2. **`FlushingLog`**, re-emitting Trainer's log dict on stdout with `flush=True`, and
   announcing train-begin and each checkpoint save. It runs *alongside* `ProgressCallback`
   rather than replacing it, so nothing is lost if the stderr path does work.
3. **`logging_steps` 100 → 20** and **`disable_tqdm=True`** (added to the allowlist, per
   §3-allowlist discipline). First evidence the loop is turning arrives in ~5 min instead
   of ~24. Neither changes any training mathematics.

**A live risk in Tier 0, which is running in parallel as this is written.** Tier 0 passes
**`dataloader_num_workers=2`** — the one candidate ruled out for Tier 1 is *not* ruled out
there, because worker processes do exist. Tier 0 also has no heartbeat, so if it goes
quiet it is as unobservable as Tier 1 was. Recorded rather than changed: its commit is
already running and an edit cannot reach it.

### 3y. Tier 0 sizing — restore yes, step budget no (2026-09-08)

Tier 1 seed 1 commit 1 is running on Kaggle: PREFLIGHT passed, RESTORE OK, the fp32
re-cast fired on 392 tensors, training under way. **The commit path works** — CELL 1
uninstalled peft 0.19.1 and installed 0.20.0 cleanly in a Batch kernel and the
mixed-install checks (§3p, ported in `8cc3221`) confirmed it. **No kernel restart is
needed inside a commit**, because a commit starts from a fresh container. The
restart remains mandatory for interactive runs.

**Tier 0 runtime, computed from measured token lengths rather than the earlier guess.**
Tokenised all 60,000 LEDGAR train rows with `microsoft/deberta-v3-base`:

| | tokens |
|---|---|
| mean / median | **136 / 101** |
| p90 / p99 / max | 278 / 561 / 1706 |
| over `MAX_LENGTH=512` | 886 rows (**1.48%**, truncated) |
| padded tokens per step, BS 16, dynamic padding | **6,141** (vs 2,139 real — **187% padding waste**) |

3 epochs × 3,562 steps = **10,686 steps**, 66M padded tokens, ≈ **7.24e16 FLOPs**
(no gradient checkpointing in Tier 0, so no recompute penalty):

| sustained | h/seed (train) |
|---|---|
| 10 TFLOPS | 2.01 |
| 15 TFLOPS | 1.34 |
| 25 TFLOPS | 0.80 |

Plus per seed ≈ 2 min eval (3×3,000 selection + 3,000 fp32 test on GPU), 3–6 min INT8 on
x86 CPU at batch 1, and 12 s for the ONNX export and arm64 quantise (measured, §3
probe). **1.0–2.2 h per seed, 3.0–6.6 h for all three**, against a 9 h design cap.

**Decision: NO step budget for Tier 0.** It fits in one commit at both ends of the
estimate, so a budget would be machinery that never fires — and untested machinery on a
path that does not need it is a liability, not insurance. **The check on that decision is
a printed per-seed wall clock**, with the threshold stated in the output: a seed over
~2.5 h means the pessimistic end is real, and the remaining seeds should be split into
separate commits by editing `SEEDS` — no code change, since a commit needs a real edit
anyway (§3w).

**Decision: restore block YES**, identical in mechanism to Tier 1's. Two reasons, and
the second is not a contingency:

1. a commit that dies or overruns otherwise leaves nothing retrievable (§3v);
2. **E2's second arm is a separate commit by design.** §3l sequences it CE 3 seeds → C3
   → `sqrt_inv_freq` 3 seeds, so the CE results must be carried into the later commit for
   one final output to hold every arm. Without restore, the arms end up in two outputs
   that must be merged by hand — an invitation to compare arms that were never in the
   same place.

Verified across four branches with redirected paths: first commit with nothing attached
(proceeds), CE results carried into the E2-arm commit (restores), attached-but-empty
(raises), two attached inputs (raises as ambiguous).

**`group_by_length` is NOT adopted for Tier 0 either, and the temptation here is real.**
187% padding waste means length-grouped batching could cut Tier 0's training time by
roughly two thirds — far more than Tier 1's 27%/21%. Unlike Tier 1 it is not too late:
Tier 0 has not started, so it *could* be registered before any seed runs. It is still
refused, because **Tier 0 fits in one commit without it.** Buying time that is not needed,
by changing batch composition on the tier whose accept rule is a reproduction of a
published baseline (E1, macro-F1 ≥ 0.80 against LexGLUE's DeBERTa 0.831), is a bad trade:
it would make any shortfall ambiguous between our training setup and our batching policy.
Recorded so the option and the reason for declining it are both on the record.

**Tier 0 and Tier 1 must never share a Kaggle notebook, and RUNNING.md now says so
structurally rather than in passing.** `4.57.6 < 4.58` and `5.0.0 ≥ 4.58`: **no
transformers version satisfies both**, so in one notebook whichever CELL 1 ran last wins
and the other tier runs against a version it was never validated on. The doc now
instructs creating two differently-named notebooks, forbids pasting either tier's cells
into the other, states that each tier's probe belongs only in its own notebook, and notes
that attaching the wrong tier's output raises "contains no Tier n artefacts" — which is
the symptom of exactly this mistake. Both notebooks' CELL 2 already assert their own
version range and raise within ~10 s; that is named in the doc as **the backstop, not the
plan**.

**Not verified:** the Tier 0 runtime is arithmetic over measured token lengths, not a
measured run — the same method that was right for Tier 1 (§3u) where the probe's
extrapolation was 13x wrong, but arithmetic nonetheless. The restore block has not run on
Kaggle. The per-seed wall clock print is what converts the estimate into a measurement on
the first commit.

### 3x. Attached inputs track latest; the stall guard is kept anyway (2026-09-08)

**§3w finding (2) resolved on Kaggle: an attached notebook input tracks the LATEST
version, not one pinned at attach time.** Run 5 read `run_index 2`, written by run 4.
The chain works and no manual re-attach is needed between commits.

**The guard is kept regardless, at the explicit instruction of the person running it,
and the reason is the right one:** the pinning risk turned out not to exist, but *a
stalled chain looks exactly like a normal resume*, and that is the failure class this
project keeps paying for (§3e). The check costs one integer comparison. Being wrong
about it costs 13.4 h per occurrence, repeatedly, with no output ever indicating a
problem.

**Where "strictly greater" belongs, which is not where it first appears to.**
`RESUME_TOTAL_STEPS_AT_LEAST` is the total the *previous commit ended at*, so the
restored total being **equal** to it is the healthy case. A strict comparison at restore
time would fail every healthy resume. Strictness belongs on **progress made during the
commit**: each seed must end strictly above both its own start and the previous commit's
recorded step for that seed. Both checks now exist, at the layers where each is true.

**Implementing that surfaced two defects in the code committed an hour earlier
(`35060ff`), both of which would have presented as normal operation.**

1. **The guard read checkpoint directories, which are deleted when a seed completes.**
   `shutil.rmtree(ck)` runs once the adapter is saved, so a checkpoint-derived maximum
   **drops to zero** the moment seed 1 finishes. The next commit would have raised
   "the attached input is STALE" on a perfectly healthy chain — and the message would
   have sent the operator hunting a nonexistent attachment problem. The guard now reads
   the **per-seed progress files**, which persist, and compares a **total across seeds**,
   which is monotonic where per-seed counts restart at 0.
2. **The step budget was per seed, not per commit.** `StopAfterNSteps(RUN_STEP_BUDGET)`
   was constructed inside the seed loop, so a commit that finished seed 1 and started
   seed 2 handed seed 2 a fresh full budget. One commit could therefore train
   2 × `RUN_STEP_BUDGET` — ~16 h at the measured rate — and hit the very time cap that
   bounded commits exist to avoid (§3v), **while each seed's own budget was correctly
   respected**. The counter is now class-level and shared across every seed the commit
   touches, and the loop breaks before starting a seed it cannot fund.

Defect 2 is the more instructive: the budget was not ignored, it was *applied at the
wrong scope*. Every individual check passed. This is the same shape as §3o and §3t —
a correct check pointed at the wrong object — and it is the fourth time in this project
that scope, not logic, was the error.

**Verified.** Restore across nine branches with the Kaggle paths redirected, including
both regressions above: a completed seed with its checkpoints deleted (**passes**, where
the previous implementation would have raised), and seed 1 complete plus seed 2 partway
with totals summed (**passes**). Stalled chain, stale total, nothing attached on a
non-first commit, an input holding nothing of ours, and two ambiguous inputs all
**raise**. A differing notebook slug is still found, confirming the glob. Commit budget
verified on the CPU harness with `RUN_STEP_BUDGET = 7` over three 10-step "seeds": seed 1
ran to 7, seed 2 was not started, total 7 — where the per-seed bug would have permitted
21.

**Not verified:** none of this has run on Kaggle. The restore, both guards and the shared
budget were exercised locally against redirected paths and a CPU harness with a tiny
model. The first real commit is the test.

### 3w. Persistence measured; Tier 1 restructured for bounded commits (2026-09-08)

**§3v's question, answered on Kaggle. Three Batch commits, zero GPU quota.**

**`/kaggle/working` does NOT carry over between commits.** A fresh commit sees only
`__notebook__.ipynb` — no marker, no checkpoint-shaped directory. The previous run's
working directory becomes **that version's output**. The suspicion was right, and the
resume design as written would have restarted seed 1 at `(fresh)` on every commit,
**discovered 13.4 h in**.

**Restore path**, with the notebook's own output attached via *Add input → Your Work →
Notebook*: `/kaggle/input/notebooks/<username>/<notebook-slug>/`.

**Three further findings from the same three commits, none of which was the question
being asked:**

1. **A commit with a `+0 -0` diff is skipped**, reporting "Ran in 0 seconds" without
   running. `kaggle_probe_persist.py`'s own instructions said "commit the SAME notebook
   again, changing nothing" — **the probe's procedure did not work as written**, and it
   was caught only by someone running it. Fixed there and documented in RUNNING.md.
2. **A notebook input may be pinned to the version current when it was attached.**
   Unresolved at time of writing; a fourth commit is testing it. If pinned, every commit
   restores the same checkpoint and training never advances — *while every run looks
   like a normal resume*. This is §3e's shape in its purest form so far: not a wrong
   value, a correct-looking loop that never terminates.
3. Therefore the restore cannot rely on the environment being honest about freshness.

**The guard, which works under both branches of (2).** Two constants are set per commit:
`RUN_STEP_BUDGET` (steps to train this commit) and `RESUME_FROM_STEP_AT_LEAST` (the
`global_step` the previous commit reported). After restoring, the highest checkpoint
step must be **≥ `RESUME_FROM_STEP_AT_LEAST`**, or the run raises naming a stale input.
A pinned input fails on the second chained commit, in seconds, instead of looping. The
notebook prints the value to use next, so the loop is closed by the notebook rather than
by memory.

**This composes with finding (1) rather than merely coexisting with it:** since every
commit *must* carry a real edit to run at all, and `RESUME_FROM_STEP_AT_LEAST` must
change every commit, **the edit that makes the commit execute is the same act as
recording progress.** A forgotten update is not a silent stale resume; it is a skipped
commit or a raise.

**Step budget: a callback, NOT `max_steps`.** `max_steps` becomes `num_training_steps`,
which is what `Trainer.create_scheduler` builds the LR schedule from — so
`max_steps=2000` would decay the learning rate to zero over 2,000 steps instead of the
true 3,563. **A different LR trajectory is a different experiment, and nothing would
have reported it.** Measured on the CPU harness, learning rate at each of 10 steps:

| run | LR trajectory |
|-----|---------------|
| uninterrupted | `9e-4, 8e-4, 7e-4, 6e-4, 5e-4, 4e-4, 3e-4, 2e-4, 1e-4, 0` |
| **budget callback**, 4 steps × 3 commits | **identical at every step** |
| `max_steps=4` | `7.5e-4, 5e-4, 2.5e-4, 0` — decayed to zero in 4 steps |

The callback sets `should_training_stop` **and** `should_save`, so the stop point is
always checkpointed rather than losing back to the last `save_steps` boundary.

**Bounded commits remain a change to execution, not to measurement** (§3v): same data
order, batch composition, optimizer trajectory, step count and — now verified — the same
learning rate at every step. **Not an amendment**; §3t stands unchanged.

**Restore verified across seven branches** with the Kaggle paths redirected: first
commit with nothing attached (proceeds); not-first commit with nothing attached
(raises); input attached holding nothing of ours (raises); normal resume (proceeds);
**pinned input** (raises); two attached inputs (raises as ambiguous); and a slug
differing from the probe's (found, confirming the glob). A run that restores a
checkpoint and then advances zero steps also raises, so the pinned case is caught after
the fact as well as before.

**Not verified:** none of the restructured code has run on Kaggle. The restore, the
budget callback and the guards were exercised locally against redirected paths and a
CPU harness. The pinned-vs-latest behaviour of an attached notebook input is still
unknown; the guard is written to be correct either way, which is why it did not wait for
that answer.

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

### E8 — few-shot vs zero-shot Sonnet-5, paired *(2026-09-09)*

- **Experiment id:** E8 (registered `b26d686`, before the numbers were opened)
- **Date:** 2026-09-09
- **Config / command:** `python scripts/score_fewshot.py` — `results/fewshot_results.json`
  (batch `msgbatch_01QbUfqihLGmwG3wkYJVZuMU`) vs `results/stage1_results.json`, both
  `claude-sonnet-5`, restricted to the same **1,000 rows**; 97 of 100 classes present, and
  **both arms averaged over exactly those 97**. Parsed with
  `src.data.schema.parse_response`; **0 format failures in either arm.**
- **Seeds:** n/a for the API arms (one batch each). Tier 0 comparator is mean over seeds 1-3.
- **Result:**

  | arm | macro-F1 | accuracy | classes predicted |
  |---|---|---|---|
  | Sonnet-5 **zero-shot** | 0.6258 | 0.7490 | 91 / 97 |
  | Sonnet-5 **few-shot (8)** | **0.6276** | 0.7570 | 89 / 97 |
  | Tier 0 INT8, same rows | **0.7782** ± 0.0053 | — | — |

  - `delta_fs` = **+0.0018** macro-F1 (accuracy +0.0080)
  - paired bootstrap, 10,000 resamples, seed 20260909: **95% CI [-0.0180, +0.0213]**,
    p = 0.869 — **includes 0**
  - `gap` (Tier 0 − zero-shot, same rows) = **+0.1524**, 95% CI [+0.1076, +0.1757],
    p = 0.0000 — excludes 0
  - `fraction_closed` = **+0.0119** (1.2% of the gap)
- **Cost:** **$0.00.** Responses were already on disk; no API call, nothing appended to
  `results/spend_ledger.jsonl`.
- **Manifest:** `test_3000` (`text_sha256 e719c110…`), restricted to the 1,000 few-shot
  ids, verified a strict subset of both `stage1`'s 3,000 and the manifest's indices.
  Exemplars: `exemplars_8` (`text_sha256 ac7e7be8…`).
- **Accept rule met?** The registered band resolves to **CAPABILITY, NOT PROMPTING** — the
  CI includes 0, which the band routes to the ≤ 0.20 disposition, and the point estimate
  closes 1.2% of the gap. **Restatement of H1/E4/E4b/E6 proceeds.**
- **Notes.** Few-shot moved Sonnet-5 by an amount indistinguishable from zero, while Tier 0
  beats it by 0.1524 on the same rows with an interval nowhere near zero. **The deficit is
  in the task, not the prompt.**
  **The bound registered in advance holds and must be carried into the write-up:**
  `exemplars_8` covers **6 distinct classes of 100** — 94 classes are never shown, so eight
  examples cannot teach the taxonomy. E8 therefore falsifies "prompting closes the gap"
  **only for the 8-exemplar prompt actually run**; a class-covering prompt (≥100 exemplars)
  is untested and unbudgeted, and no sentence may generalise past that.
  Few-shot predicted **fewer** distinct classes than zero-shot (89 vs 91), consistent with
  the §3d/C4 expectation that exemplars bias prediction toward their own classes.

### E6 best-Pareto-point — paired test *(2026-09-09)*

- **Experiment id:** E6 (supporting test; supersedes a wrong yardstick — see §3al)
- **Date:** 2026-09-09
- **Config / command:** `python scripts/cascade_paired_bootstrap.py` — Tier 0 alone vs
  Tier 0 + Sonnet-5 escalation at the best Pareto point, **same 3,000 rows, same 3 seeds**,
  margin signal, threshold calibrated on `dev_2000` and **held fixed while test rows are
  resampled**. Averaged over all 100 classes (`test_3000` covers 100).
- **Seeds:** 1, 2, 3
- **Result:** Tier 0 **0.7523 ± 0.0056**; cascade **0.7585 ± 0.0034**; paired delta
  **+0.0063**, 95% CI **[-0.0004, +0.0103]**, p = **0.0756** — includes 0. The dev-quantile
  threshold at target 0.0406 applies as 0.0400 / 0.0323 / 0.0493 per seed.
- **Cost:** $0.00.
- **Accept rule met?** n/a — instrumentation under C3's remit (paired-bootstrap floors),
  not an arm with its own rule.
- **Notes.** The conclusion ("would not claim the cascade beats Tier 0 alone") stands, but
  **not for the reason first given.** See §3al: the across-seed sd was the wrong yardstick,
  and 0.0063 was 1.1-1.9× it rather than inside it. The correct paired interval is
  **tighter**, and puts the result at **borderline** (p = 0.076, lower bound -0.0004),
  not comfortably null.

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
