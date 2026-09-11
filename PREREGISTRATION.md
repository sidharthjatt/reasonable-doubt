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

> **⚠ SUPERSEDED 2026-09-09 — the figures below were replaced and this block is the
> historical record.** This table read **28.97 ± 0.24 req/s (4 runs)**, measured on an
> **UNTRAINED architecture probe** and on a contaminated n=4 set (**§3e instance 8**). The
> live figures are **9 runs over the three TRAINED INT8 artefacts** (§3aj's registered rule:
> mean over all runs, no outlier excluded). The stale number stood here as "measured" while
> §3aj already carried the replacement — corrected under §3aw, and found only by the §3ax
> sweep, not by any check.

| quantity | value (LIVE, 9 runs, 3 trained artefacts) | superseded (n=4, untrained probe) |
|----------|-------|---|
| throughput | **28.3615 ± 1.1839 req/s** (bs=1, max_length 512) | ~~28.97 ± 0.24~~ |
| p50 latency | **25.3969 ± 0.8824 ms** | — |
| marginal SoC power | **17.7546 ± 0.5449 W** (load − idle) | ~~16.0 W package~~ |
| energy | **0.6269 ± 0.0311 J / request** | ~~0.526 J~~ |

**Per-run values, required by §3aw and carried here rather than only in the artefact:**

| artefact | run | rps | p50 ms | p95 ms | marginal W | J/req |
|---|---|---|---|---|---|---|
| int8_ce_1 | 1 | 28.6402 | 25.0328 | 101.6378 | 17.3140 | 0.6045 |
| int8_ce_1 | 2 | 27.3710 | 26.4489 | 106.0808 | 17.7396 | 0.6481 |
| int8_ce_1 | 3 | 29.4894 | 24.7160 | 97.2942 | 18.6136 | 0.6312 |
| int8_ce_2 | 1 | 28.7610 | 24.9569 | 95.9016 | 16.8503 | 0.5859 |
| int8_ce_2 | 2 | 27.8682 | 25.9455 | 101.5982 | 17.4198 | 0.6251 |
| int8_ce_2 | 3 | 26.1640 | 27.0067 | 102.5825 | 18.1729 | 0.6946 |
| int8_ce_3 | 1 | 29.7825 | 24.5410 | 95.7084 | 18.0896 | 0.6074 |
| int8_ce_3 | 2 | 29.4628 | 24.5560 | 97.3014 | 18.1293 | 0.6153 |
| int8_ce_3 | 3 | 27.7141 | 25.3685 | 100.0497 | 17.4625 | 0.6301 |

**All nine are the same sign and the spread is run-to-run, not artefact-to-artefact:**
`F(artefact) = 1.04` on df=(2,4) against a 95% critical value of 6.94 — the artefact
explains nothing beyond noise (§3aj). Per-artefact means 28.50 / 27.60 / 28.98.

The arithmetic below is **left at the superseded figures as the historical record**; the
live cost line is computed from `configs/costs.yaml`, which carries the 9-run values.

```
[SUPERSEDED ARITHMETIC — kept per hard rule 7, do not cite]
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
| Tier 0 throughput | **measured 28.3615 ± 1.1839 req/s** (9 runs, 3 trained INT8 artefacts, §3aj). ~~28.97 ± 0.24 (n=4, untrained probe)~~ **superseded — §3e instance 8**; per-run table in §1 |
| `device_cost_usd` | **derived** $633.86 (59,900 INR ÷ 94.50, FX dated 2026-09-07) |
| Tier 0 SoC power | first reading taken with a defective harness; **re-measure** |
| Tier 0 wall power | not obtainable without a meter; **optional given the margin** |
| Tier 1 throughput | **not measured** — needs the trained adapter under MLX |
| escalation rate | from the router sweep, simulated offline at zero cost |

## 2. Hypotheses## 2. Hypotheses

> **DRAFT.**

| id | hypothesis | rationale | how it could be wrong |
|----|-----------|-----------|-----------------------|
| H1 | **⚠ RESTATED → H1-A (§3av); original preserved, and it is SATISFIED TRIVIALLY in the wrong direction.** A 3-tier cascade reaches macro-F1 within 0.04 of Sonnet-5-alone on `test_3000` at **under half** the USD/1k-clause cost. | Tier 0 answers head-class clauses in milliseconds at near-zero marginal cost; the top 10 classes are 31.5% of the corpus. | Escalation is driven by rare classes, which are also where Tier 0 is weakest, so the router escalates most of the tail and saves little. Or Tier 0's errors are confident, so the router does not catch them. |
| H2 | Tier 0's softmax **margin** is a better routing signal than LLM **verbalized confidence**, measured by AUROC for predicting own-correctness on `dev_2000`. | Verbalized confidence is severely compressed in every model measured: sd 0.031 (Gemini, non-reasoning), 0.083 (gpt-oss), 0.174 (Haiku), 0.184 (Sonnet), all with means 0.86–0.97 against accuracies 0.65–0.84. Haiku emitted **5 distinct values in 20 answers**. A continuous margin has no such plateaus. | The encoder is also badly calibrated, or its margin is compressed in a different way; or verbalized confidence, despite discretization, still ranks correctness well enough that AUROC is comparable. |
| H3 | Class weighting raises macro-F1 over unweighted CE at 137.7x train imbalance. | Unweighted CE optimises accuracy, which the head dominates; macro-F1 weights all 100 classes equally. | Weighting destabilises training or trades so much head accuracy that macro-F1 does not move beyond the noise floor. At `inv_freq` the weight ratio is 137x and may simply not converge. |

## 3. Planned experiments

> **DRAFT — proposed accept rules with reasoning. Edit and own before running.**
> An experiment with a blank accept rule must not be executed (hard rule 6).

| id | description | accept rule | status |
|----|-------------|-------------|--------|
| C3 | Measure seed-variance and paired-bootstrap floors | descriptive — no accept rule; **gates E2, E3, E7** | planned |
| E1 | Tier 0: DeBERTa-v3-base, CE baseline, 3 seeds | **macro-F1 ≥ 0.80 on `test_3000`, measured on the ONNX-INT8 artefact**, FP32 reported alongside. Anchored to LexGLUE Table 3 (DeBERTa m-F1 83.1) | planned |
| E1b | Tier 0 retrained at **10 epochs**, all else identical to E1 | **macro-F1 ≥ 0.80 on `test_3000`, INT8, mean over 3 seeds** — the SAME bar as E1. Seed 1 first; seeds 2–3 gated on it | **NOT ACCEPTED (§3bi): 0.798744 ± 0.007018, short by 0.0013 = 0.18σ. Undertraining supported by the ONE-VARIABLE within-host comparison — +0.0385 vs the 3-epoch host baseline, n = 1 (§3ak/§3at) — without reaching the bar. The cross-host +0.0465 / 8.31σ is DESCRIPTIVE ONLY. Seed 1 breaches E3's 0.01 quantisation tolerance at −0.0137** |
| E2 | Tier 0 loss arms vs E1 (sqrt-inv-freq, effective-number, inv-freq) | best arm beats E1 by **≥ √2·1.96·seed_sd** (paired, same rows) — **NOT YET SETTABLE** | **[C3-gated]** |
| E3 | INT8 vs FP32 at the deployed precision | **\|INT8 − FP32\| ≤ 0.01** macro-F1, paired — **the text names NO seed aggregation (§3bj.1)** | **MEASURED on both Tier 0 families; VERDICT WITHHELD pending the statistic (§3bj, §4).** Scope: **artefact-general across Tier 0, not E1-scoped** (§3bi). Tolerance stands at 0.01 (C3 `seed_sd = 0.003195`). **E1** −0.005378 / −0.008317 / −0.000693, mean −0.004796 — **passes under both readings**. **E1b** −0.013668 / −0.001751 / −0.007598, mean −0.007672 — **FAILS per-seed, PASSES on the mean** |
| E4 | Tier 1: Qwen2.5-1.5B-Instruct LoRA, 3 seeds | macro-F1 **≥ E1 + 0.04** = **0.7923** (E1 INT8 0.7523, §3ar) | **seed 1: 0.7254 — misses by 0.0669 = 12× test_3000 σ** |
| E4b | Two-tier `Tier 0 → Claude` fallback | E6's rule with Tier 1 removed. **Registered before E4 runs** | **SUPERSEDED BY MEASUREMENT (E4b-A, §3av)** — the E6 frontier IS this configuration; no Tier 1 experiment remains |
| E5 | Routing signal: margin vs max-softmax vs entropy | best **AUROC ≥ 0.75** AND **≥ 0.05** above worst, on `dev_2000` only | planned |
| E6 | **Cost-vs-volume break-even curve** — the headline | (1) within **0.04** macro-F1 of Sonnet-5-alone — **MET TRIVIALLY**; (2) finite crossover **V\* ≤ V_max** and (3) asymptote **< 50%** of Sonnet-5 — **PREMISE-FALSIFIED**. **V\* is reported, not tested** | **RESTATED → E6-A (§3av); verdict UNCHANGED** |
| E7 | `max_length` 256 vs 512 ablation | loses **< 0.02** macro-F1 paired AND **≥ 1.5×** faster; per-class deltas required | planned **[C3-gated]** |
| E8 | Few-shot vs zero-shot Sonnet-5, paired on 1,000 rows | **`fraction_closed` ≥ 0.50 AND paired-bootstrap CI excludes 0** ⇒ premise live, defer restatement; **≤ 0.20** ⇒ capability not prompting. Band in full below | **registered — see E8 below** |
| C1 | Calibration-set robustness (§3a) | thresholds agree within **1 decile** AND escalation within **3pp** | planned |
| C2 | Few-shot confidence anchoring (§3b) | **Sonnet 5 ONLY**: few-shot sd **≤ 0.5×** zero-shot sd AND mode within 0.02 of 0.9 | **RUN — FAILS both clauses; no anchoring** |
| C4 | Exemplar class over-prediction (§3d) | 3× class predicted at **≥ 2×** its zero-shot rate AND above every 1× class; **+2pp fallback if zero-shot rate < 1%** | **RUN — ratio form (§3am); FAILS both clauses** |

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
  first** (~~3.3–7.3 h~~ → **~8.1–8.6 h**, PER SEED; §3ba). Seeds 2 and 3 are scheduled **only if seed 1 clears the
  gate registered in §3ar** — "materially" is no longer undefined; §3ar partitions the
  outcome space with no unassigned region. If it does not, **undertraining is not the cause**, E1b is
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
- **SCOPE — settled 2026-09-11, §3bi, because E1b forced the question.** This rule names
  **no run**. Its hypothesis is about **Tier 0**, and "same model, same rows" pairs each
  artefact against **itself** rather than restricting which artefact is eligible.
  **E3 is artefact-general across every Tier 0 artefact, not scoped to E1.** Consequences,
  recorded either way as §3bi required:
  - **E1b's artefacts are in scope.** Seeds: −0.013668 / −0.001751 / −0.007598, mean
    **−0.007672**; E1's largest was **|0.0083|** (E1 per seed: −0.005378 / −0.008317 /
    −0.000693, mean −0.004796). **⚠ AMENDED — see §3bj.1.** "Seed 1 BREACHES" and "the
    first measured breach" **assume a per-seed reading that E3's text does not state**.
    Under the 3-seed-mean reading E1b **passes** and there has been **no breach**. Both
    readings are recorded; **neither is adopted**, because choosing now means choosing
    after seeing which one fails.
  - **The tolerance is NOT loosened.** 0.01 was registered **[C3-gated]** against the risk
    that seed sd ≈ 0.03 would void it; C3 measured **0.003195**, so the gate lifts in
    favour of 0.01 standing. Substituting now — after a number came near it — is exactly
    what the gate's ordering forbids.
- **THE STATISTIC IS UNSPECIFIED, and this rule's own text is the evidence (§3bj.1).**
  The accept rule reads `|INT8 − FP32| ≤ 0.01 … paired` and **never names a seed
  aggregation** — not per-seed, not mean, not a seed count. Per seed, E1b seed 1 **fails**
  at 0.013668; on the 3-seed mean, E1b **passes** at 0.007672. E1 passes either way, which
  is why the divergence surfaces only now. **NEITHER READING IS ADOPTED**: choosing after
  seeing which one fails is the ordering hard rule 6 exists to prevent. **Resolution is a
  registration act owed BEFORE the next INT8-vs-FP32 number is opened**, and it belongs in
  this rule's text, not in a result entry.
  - **The falsification clause does not bind the running service today:** no E1b artefact
    is deployed (§3bg serves `fp32_ce_1`, an E1 artefact). It binds on any decision to
    serve an E1b artefact at INT8, on top of §3bc's recalibration gate.
  - ~~**E3 itself remains UNSCORED.**~~ **⚠ WITHDRAWN — see §3bj.2.** E3 has been
    **measured on both Tier 0 families** and its numbers are already load-bearing (§3bc's
    canary floor). Corrected status: **MEASURED, VERDICT WITHHELD pending the statistic.**
    Result entry filed in §4.

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
| Sonnet 5, caching active | $1.3449 | **$0.4483** *(PROJECTED — Stage 1 later measured **$0.43610**; §3ax item 6)* |

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

### 3ap. Four corrections to §3an/§3ao, and the finding is the opposite shape (2026-09-09)

**A. The C2 direction claim was backwards.** I wrote that the few-shot mode "moved *away*
from the anchored 0.9". |0.98 − 0.9| = **0.08**; |0.85 − 0.9| = **0.05**. It moved
**toward** it. The verdict is unchanged — 0.05 still misses the registered ±0.02 window
and the sd clause fails at 0.8837 vs ≤ 0.5 — but *"the fixed confidence did not anchor
Sonnet" overstates what was measured.* Location moved; spread did not:

| statistic | zero-shot | few-shot | toward 0.9? |
|---|---|---|---|
| mode | 0.98 | 0.85 | **yes** (0.080 → 0.050) |
| median | 0.920 | **0.900** | **yes** (0.020 → 0.000) |
| mean \|c−0.9\| | 0.1196 | 0.1083 | **yes** |
| mass within ±0.05 of 0.9 | 202 | **307** | **yes**, +52% |
| mean | 0.8514 | 0.8443 | no (0.0486 → 0.0557) |
| mass **exactly** at 0.90 | 52 | 50 | **no — flat** |
| sd | 0.1598 | 0.1412 | **no compression** (ratio 0.88) |

**Supported statement:** the anchor produced a **measurable pull on location** — the
median lands exactly on 0.9 and the mass within ±0.05 rises by half — that **fell well
short of the registered window, with no compression at all.** Pile-up on the *exact*
demonstrated value did not happen (52 → 50 rows).

**The §3b falsification clause does not cleanly fire, and that is a defect in the rule.**
It reads: *"sd ratio > 0.5 **and** mode away from 0.9 ⇒ the fixed exemplar confidence did
not anchor, and verbalized confidence is compressed for reasons intrinsic to the models
rather than to our prompt."* The first conjunct holds; the second does **not** — the mode
moved toward 0.9, merely not far enough. **So C2 neither passes its accept rule nor
satisfies its falsification clause.** The rule is silent on "moved toward but missed",
which is exactly what happened. My earlier claim that compression is *"intrinsic to the
model, not induced by our prompt"* invoked a clause that never fired and is **withdrawn**.
What survives: **compression is untouched by the anchor** (sd ratio 0.88) — that much is
measured, and it is enough for H2's purposes. Whether *location* is prompt-induced is
answered "partly, weakly", which the rule was not written to grade.

**B. McNemar tests ACCURACY; the registered statistic is MACRO-F1.** `delta = +0.0063` is
macro-F1; per-row correct/incorrect is accuracy. Net +12 of 365 **cannot determine a
macro-F1 sign**, because macro-F1 upweights rare classes and *which* classes those rows
belong to decides it. **This is §3e instance 2's shape recurring in the adjudicating test
itself** — the same defect that left `macro_f1` NaN in `build_frontier.py` while accuracy
was reported. **§3e instance 10.** That it recurred in the test brought in to *settle* the
question is the aggravating part: the check inherited the flaw it was auditing.

**The correct test, paired bootstrap of macro-F1 restricted to escalated rows:**

| seed | n | classes | Tier 0 | API | delta | 95% CI |
|---|---|---|---|---|---|---|
| 1 | 120 | 45 | 0.2646 | 0.3191 | **+0.0545** | [−0.0527, +0.1025] |
| 2 | 97 | 44 | 0.3094 | 0.3589 | **+0.0495** | [−0.0578, +0.1114] |
| 3 | 148 | 59 | 0.2668 | 0.3475 | **+0.0807** | [−0.0382, +0.1098] |

**Sign-consistent 3/3 positive**, every interval including 0. **This reverses my
conclusion.** I wrote *"on the rows the cascade acts, the API is not reliably better than
the encoder"* — that rested on the wrong statistic. On the **registered** statistic the
API is better on escalated rows on **all three seeds**, by +0.05 to +0.08, though no
single interval separates from zero at n ≈ 100–150 over 44–59 classes. The
macro-F1/accuracy divergence is itself informative: the API's gains on escalated rows are
concentrated in **rare** classes, which is what macro-F1 is built to see and accuracy is
built to miss. **Withdrawn.** McNemar is retained **as accuracy evidence, labelled as
such**: pooled 79/91, net +12 of 365, p = 0.399.

*Caveat that stops this from being over-read:* escalated-row macro-F1 is computed over the
escalated rows' **own** label space (44–59 classes), a different quantity from the
contribution to global 100-class macro-F1. **It is not a decomposition of the +0.0063**
and does not explain why the global delta fails sign-consistency.

**C. The pooled 365 are not independent, and the distinct-row count is a finding.** All
three seeds run on the same 3,000 rows, so a row escalated by more than one seed is
counted more than once; the 170 discordant pairs are **not** 170 independent observations.
**The direction is conservative** — dependence inflates apparent precision, and the test
was already null at p = 0.399, so the null conclusion is safe. Stated rather than left
implicit.

**The overlap refutes my own mechanism claim:**

> 365 escalations cover **250 distinct rows**. Escalated by 1 seed: **160**; by 2: 65;
> by 3: **25 (10.0%)**. Pairwise Jaccard **0.213–0.258**.

I claimed the variance reduction was *"by construction"* — escalated rows getting a
seed-invariant API prediction. **That mechanism requires the router to select the same
rows each seed, and it does not: ~75% of escalated rows are non-overlapping between any
two seeds, and only 10% form an all-three core.** The claim is **withdrawn**. What
replaces it is the more serious finding the overlap actually shows: **which rows get
escalated is itself highly seed-unstable.** And on variance, the honest position is that
**nothing is established at n = 3** — an sd ratio (0.611) and a correlation (−0.965) on
three points are not testable quantities. I substituted one story for another when the
supported answer was "not determinable at this seed count".

**D. An artefact number disagreed with the prose, because its label space was unrecorded.**
`e6_seed_structure.json` gave first1000 − rest2000 as +0.0096 / +0.0204 / +0.0171; the
prose said +0.0075 / +0.0204 / +0.0151. **Both are correct and they answer different
questions**, which nothing in the file said:

| basis | classes | per seed | mean |
|---|---|---|---|
| each half over its **own** present classes | 97 vs 98 | +0.0096 / +0.0204 / +0.0171 | +0.0157 |
| **common label space** (held constant) | 95 | +0.0075 / +0.0204 / +0.0151 | **+0.0143** |

**The common-95 figure is the one reported**, because holding the label space constant is
the entire point of separating row difficulty from label-space effects; the own-class-set
version is confounded by construction. This is **§3e instance 9's shape with a
disagreement attached** — a number whose denominator was not recorded. The script now
writes **both**, each with its `classes_averaged`, and marks which is `reported`.

### 3ar. E1b's seed-1 GATE — registered, with every boundary derived (2026-09-10)

E1b's **accept rule** has been registered since 2026-09-09 (macro-F1 ≥ 0.80 on
`test_3000`, INT8, mean over seeds 1/2/3). Its **seed-1 gate** was not: the entry says
seeds 2–3 run only if seed 1 *"moves macro-F1 materially toward 0.80"*, and **"materially"
was never defined.** That is C2's defect in advance — a rule with an unassigned region that
interpretation would fill once the number arrived (§3ap A). Registered now, before any E1b
run.

**Two corrections that move every boundary, made first.**

1. **E1's headline is 0.7523, not 0.7570.** `0.7570` is E1's **FP32** 3-seed mean
   (0.7636 / 0.7597 / 0.7478). Every rule here attaches to the **deployed INT8 artefact**,
   whose arm64 3-seed mean is **0.7523** (0.7582 / 0.7514 / 0.7471). §3ak's "compare to
   E1's 0.7570" quoted an FP32 figure against an INT8 rule, and **E4's miss inherited it**:
   E4's bar is `E1 + 0.04` = **0.7923**, and Tier 1's 0.7254 misses by **0.0669**, not
   0.0716 — still 12× the `test_3000` σ, so E4's verdict is unchanged.
2. **σ = 0.0056, not 0.0032.** `0.0032` is the **selection-split** seed sd. This rule is
   stated on `test_3000` INT8, whose measured across-seed sd is **0.0056**. Using a
   selection-split sd for a test-split rule is §3ap(B)'s class of error — the wrong
   statistic for the stated question.

**ONE construction, and every boundary derives from it.** Two anchors that mean something —
the accept target **0.80** and the baseline **E1 = 0.7523** — with the band between them
split at its midpoint. **Nothing is stacked.** An earlier draft mixed a stacked-2σ ladder
with an `E1 + 2σ` anchor and opened a **0.7635–0.7681 dead zone** in which a result **more
than 2σ above E1** would still have stopped the run.

| symbol | derivation | arithmetic | value |
|---|---|---|---|
| σ | measured across-seed sd, `test_3000` INT8 arm64, seeds 1–3 | sd(0.7582, 0.7514, 0.7471) | **0.0056** |
| **T** | 0.80 − 1.645σ | 0.80 − 1.645 × 0.0056 = 0.80 − 0.009212 = 0.790788 | **0.7908** |
| **B** | E1 + 2σ | 0.7523 + 2 × 0.0056 = 0.7523 + 0.0112 | **0.7635** |
| **M** | (B + T) / 2 | (0.7635 + 0.7908) / 2 = 0.777144 | **0.7771** |

**The partition — total, with no unassigned region:**

| seed-1 INT8 macro-F1 | region | action |
|---|---|---|
| **≥ 0.7908** (≥ T) | ON TRACK | **run seeds 2–3** |
| **0.7771 – 0.7907** (M ≤ x < T) | MOVED, UPPER | **run seeds 2–3** |
| **0.7635 – 0.7770** (B ≤ x < M) | MOVED, LOWER | **STOP.** Partial movement: undertraining contributes but does not account for E1's shortfall |
| **0.7523 – 0.7634** (E1 ≤ x < B) | WITHIN NOISE OF E1 | **STOP.** Undertraining is not the cause |
| **< 0.7523** | WORSE THAN E1 | **STOP.** Falsified |

**WHAT T BOUNDS, stated rather than left to inference.** T is a **seed-level** bound. It
asks: *given this one observation, is a true mean of ≥ 0.80 still plausible at one-sided
95%?* It does **not** ask whether the 3-seed mean will reach 0.80 — that question uses the
standard error of the mean, σ/√3, giving 0.80 − 1.645 × 0.0056/√3 = **0.7947**, a
**tighter** bar.

Seed-level is correct **because we have one seed, not three.** Grading a single observation
against the standard error of a three-seed mean would understate that observation's
uncertainty — borrowing a mean's precision to judge a single draw. The seed-level bound is
also deliberately the **more permissive** of the two (0.7908 vs 0.7947), and the asymmetry
is intended: a false STOP abandons a possibly-true hypothesis permanently, while a false
CONTINUE costs two seeds of quota. **The gate is a spending decision under one observation,
not a test of the accept rule** — the accept rule is still decided on the 3-seed mean,
unchanged.

**The gate is evaluated on INT8, which the training host cannot compute — see §3as.**

### 3az. HOST BASELINE RESULT — §3at applied; this is NOT the gate (2026-09-10)

**Stated first, because conflating them would be the §3av error:** this is the **host
baseline** (§3ak step 1, `ce_hostB`, `EPOCHS = 3`, seed 1). **The §3ar gate has NOT been
reached** — it applies to **E1b seed 1**, which has not run. Nothing below is a gate outcome.

**Steps 4–6 executed.** `train_env()` **was persisted** into both the seed JSON and the npz —
§3e instance 9's mitigation working on its first real run: Tesla T4, capability 7.5, CUDA
12.8, cuDNN 91002, driver 580.159.04, torch 2.10.0+cu128, `tf32_matmul: false`,
`tf32_cudnn: true`, transformers 4.57.6.

**A CORRECTION TO §3as STEP 4, MADE BEFORE RUNNING IT.** §3as step 4 said *"local arm64
export + quantise"*. **Following that literally would have introduced a second variable into
`d`.** E1's anchor was produced by scoring **Kaggle-quantised** INT8 bytes on arm64 —
verified: `models/int8_ce_1/model_quantized.onnx` is byte-identical (sha256
`a964075d…`) to the Kaggle download, and `scripts/score_int8_local.py` only *scores*, it
never exports or quantises. So the host baseline was scored the **same way**: Kaggle's
`int8_ce_hostB_1`, scored on arm64. **`d` is then a one-variable comparison (host), which is
what §3at requires.** A local re-quantise would have made it host + quantisation-run.

**RESULT.**

| quantity | value |
|---|---|
| hostB seed 1, `test_3000` **INT8 arm64** | **0.7588558562** |
| E1 seed 1, `test_3000` **INT8 arm64** | **0.7582129450** |
| **`d = hostB − E1`** | **+0.0006429112** |
| \|d\| vs the 2σ threshold | **0.000643 < 0.0112** |
| in σ units | **0.11 σ** |

**§3at BRANCH 1 FIRES: \|d\| < 0.0112 → UNINTERPRETABLE AT n = 1.** No host effect is
detected *at the only scale available*. **This does NOT establish that the hosts are
equivalent** — n = 1, no variance estimate, and 2σ is a within-host across-**seed** spread
borrowed for a cross-host question whose variance is unmeasured. **Both §3at branches
conclude "not attributable", and that line holds here:** E1's training environment was never
recorded, so even a large `d` could not have been decomposed. What this result licenses is
"no host effect was detected", nothing stronger.

**The FP32 figure is NOT the registered quantity.** Kaggle reported `test_3000_fp32`
0.7638615 against E1 seed 1's 0.7636. **`d` is not computed from it and no FP32 comparison
is made** — §3at is stated on INT8 arm64, which is the deployed precision (E3's falsification
clause prohibits reporting FP32 as the system's accuracy). The number is recorded here only
so its omission is visible as a decision.

**Kaggle's INT8 collapse reproduced exactly as expected: 0.000777** against arm64's 0.7589,
a gap of **+0.7581**. That is E3's discriminator on a non-VNNI Xeon (§3as), an input to
nothing.

**THE GUARD FIRED FOR REAL, AND ON THIS RUN RATHER THAN E1b.**
`tier0_ce_hostB_seed1.json` was produced by the **pre-port** notebook, so it carried a
`macro_f1` with no `classes_averaged` and
`test_committed_artefacts_record_classes_averaged` **failed**. It was **not** grandfathered:
the registered scorer was re-run over the logits already on disk, and every Kaggle value
reproduced **within 4 ULP** (`test_3000_fp32` 3 ULP, `test_3000_int8` 0 ULP,
`test_3000_onnx_fp32` 4 ULP, `selection` exact) — the §3ay summation-order signature.
Kaggle's own values are **untouched**; `classes_averaged: 100` and a
`registered_scorer_backfill` block were added beside them. **The list did not grow.**

### 3bm. ESCALATION TURNED OFF — Tier 0 only, by measurement (2026-09-12)

**A DEPLOYMENT DECISION, registered under §3bc. Not an experiment, no accept rule.**
The operator's decision, taken on the served-config check filed in §4.

#### What it rests on

| | macro-F1 |
|---|---|
| Tier 0 alone | **0.809052** |
| cascade | 0.809874 |
| delta | **+0.000822**, 95% CI **[−0.0060, +0.0072]**, p = **0.857** |

McNemar on the 129 escalated rows: **28 / 21, net +7, exact p = 0.392**.

> **The escalation target does not pay for itself in accuracy at the served operating
> point.** Escalation is therefore **off**: `tier2.enabled: false`.

**THE ROUTER STAYS ON, AND THAT IS THE POINT.** Tier 0 scores **0.3488** on the rows the
margin signal selects against **0.8733** overall — the signal identifies genuinely hard
clauses. What it cannot do is hand them to something that answers them better. Those rows
are now **flagged**, not escalated.

#### The flag is NOT a skipped escalation, and the response says so

`escalation_skipped` means *"the router picked this row and we could not escalate it"* — a
degraded state. With Tier 2 off by config nothing is picked, so reporting `skipped: true`
would make a deliberate architecture read as a failure to anything counting that field.

| field | low-margin row, escalation OFF |
|---|---|
| `low_confidence` / `needs_review` | **true** |
| `escalation_enabled` | false |
| `escalation_selected` | **false** — nothing was selected |
| `escalation_skipped` | **false** |
| `escalation_skipped_reason` | null |
| `tiers_invoked` | `["tier0"]` |

Verified live with no API key on `test_3000`'s **lowest-margin row** (margin 0.001359
against threshold 0.528864): `label Definitions`, flagged, not escalated, Tier 0 billed
alone.

#### Reversible by config alone, and held to it

Tier 2's code, tests, spend cap, ledger path and threshold are **unchanged**. The five
Tier 2 behaviour tests now run against a fixture that forces `tier2_enabled=True`, so the
path stays exercised — **deleting or skipping them would have left "reversible by config"
as a claim with nothing holding it up.** `test_escalation_is_reenableable_by_config_alone`
asserts the round trip; `test_served_config_has_escalation_off` pins the served default so
a flip back is a deliberate act that breaks a test rather than silent drift.

#### What this does NOT claim

- **Not** that escalation is worthless in general — it is one operating point, **n = 1
  seed**, and **in-sample for the 4.056% rate** (§3bc).
- **Not** that the low-confidence rows are handled. They are **flagged and returned with a
  Tier 0 answer**; the review path they imply does not exist. The flag is honest about the
  uncertainty, not a fix for it.
- **Not** a change to any published result. E6, H1 and E8 are untouched.

### 3bl. FP32 ENERGY MEASURED; the throughput gap is SESSION, not artefact (2026-09-12)

**Not an experiment.** Fills §3bh group A's hard block and corrects the live pricing path.

#### 1. The energy bench — 3 runs, served artefact, NO run excluded

| run | rps | idle W | load W | marginal W | J/req |
|---|---|---|---|---|---|
| 1 | 31.2755 | 0.11465 | 15.85605 | 15.74140 | 0.503315 |
| 2 | 31.3247 | 0.07695 | 16.22230 | 16.14535 | 0.515419 |
| **3** | 30.1408 | **1.36195 ⚠** | 16.38020 | 15.01825 | 0.498270 |
| **mean ± sd** | **30.9137 ± 0.6698** | 0.5179 ± 0.7313 | 16.1529 ± 0.2689 | **15.6350 ± 0.5710** | **0.505668 ± 0.008813** |

**RUN 3'S IDLE BASELINE IS CONTAMINATED AND THE RUN IS STILL REPORTED.** Idle reads
**1.36195 W** against 0.11465 and 0.07695 — background applications were open and the
machine could not be brought fully idle. Because `marginal = load − idle`, **a high idle
biases marginal DOWN**, so run 3 pulls the reported figure **down**, not up. It is kept
because §3aj's protocol excludes no run.

> **SENSITIVITY, so the no-exclusion rule is not asked for on trust.** Runs 1–2 only:
> marginal **15.9434 W** (+1.97%), J/req **0.509367** (+0.73%), sunk **$1.198427e-05**/1k
> (+0.73%), local **$0.00086820**/1k (−1.21%). **No conclusion moves under either choice**,
> which is the only reason the contaminated run is harmless rather than merely disclosed.

#### 2. THE THROUGHPUT CONTRADICTION — resolved, and it is not the weights

Two FP32 throughput figures disagreed by 26% (24.5969 ± 2.2374 in the morning session vs
30.9137 above), and FP32 appeared to overtake INT8's 28.3615. **Interleaved control**, no
sudo, `--skip-power`, both artefacts alternated, 3 runs each, one session:

| artefact | runs | mean ± sd |
|---|---|---|
| `onnx_ce_1_fp32` (E1) | 30.3178 / 26.9431 / 30.1701 | 29.1437 ± 1.9072 |
| `onnx_ce10ep_1_fp32` (E1b, served) | 29.9017 / 29.5456 / 30.1395 | 29.8622 ± 0.2989 |

| effect | size |
|---|---|
| **artefact** (E1b − E1, one session) | **+0.7186 rps** — inside E1's own within-session sd |
| **same artefact, across sessions** (E1 today − E1 this morning) | **+4.8471 rps = +19.95%** |

> ### THE GAP IS MEASUREMENT CONDITIONS, NOT ARTEFACT.
>
> And the consequence is larger than the question asked: **no cross-session throughput or
> energy comparison in this project is safe**, including **FP32 vs INT8**. Arithmetically
> the file now says FP32 is faster *and* 8.4% cheaper than INT8, **reversing §3bh's
> projected +15% to +20%** — but the INT8 row is a third session, and the drift measured
> here is bigger than the gap. **The precision penalty on this host is UNMEASURED**: not
> 1.15×, not zero, not negative. Settling it needs an interleaved INT8-vs-FP32 run, which
> has **not** been done and is not chased here.
>
> §3bg's deployment decision is unaffected — it rests on FP32 being the only precision
> qualified on a Linux target (§3bf/§3bg), never on speed or cost.

The morning row is **retained, not deleted** (§3ax's class) under
`superseded_2026_09_11:` in `costs.yaml`, with the control's numbers beside it.

#### 3. Derived, and what stays on E1

`tier0_usd_per_1k_sunk` **1.189724e-05** (was 1.4750e-05); Sonnet-5-to-Tier-0 ratio
**36,656×** (was 29,567×); local **$0.00087881**/1k (was $0.00095968). Tariff is **still
assumed** (§3ac) and every figure here inherits that.

> **E6, THE FRONTIER AND V\* ARE NOT REBUILT and stay on E1 as registered.** No Pareto
> point, break-even volume or frontier figure has been recomputed. §3bh group C already
> records that V\* is precision-independent to ~0.01%, so nothing is pending.

#### 4. The live pricing bug §3bh flagged is FIXED

`local_usd_per_request()` is **precision-keyed**: it reads the served precision's own
`per_tier_throughput` row and **raises `MeasurementUnavailable`** for a precision whose
energy is unmeasured rather than borrowing the other row's (hard rule 11). The INT8 path
reproduces **$0.00095968** exactly, which is the control that says the refactor moved
nothing.

> **§3bh's own estimate of the bug was wrong in sign.** It said `/classify` "understates by
> ~15%". The INT8 figure it was quoting **overstated** by 9.2%. Corrected rather than
> quietly dropped — and per §2 above, that 9.2% is itself cross-session and not a precision
> effect.

#### 5. ORT telemetry — answered, not fixed

`bench_local` **does** route through `src.ort_runtime.import_onnxruntime()`, so §a7dcf59's
disable **is applied** on this path. The `telemetry.cc … Failed to persist telemetry device
ID` line is emitted by ORT's **native library load**, which happens at `import onnxruntime`
— *before* `disable_telemetry_events()` can be called — and it appears only under `sudo`,
where the device-ID path is not writable as root. It is **not** the exit-time
`recursive_mutex` crash the disable exists to prevent, and that crash did not occur: all
runs exited 0. **Not fixed:** suppressing a message emitted during dylib load is not a
one-line change, and the disable is doing its actual job.

### 3bk. SERVED MODEL SWAPPED to E1b seed 1 FP32 — registered under §3bc (2026-09-11)

**NOT AN EXPERIMENT, NO ACCEPT RULE.** A deployment change, registered because §3bc exists
so that every chosen serving value carries its basis. **Nothing here restates E6, H1 or
E8 — those stay on E1 and were not rebuilt.**

#### The choice of seed, made on SELECTION and not on test

| seed | **selection** (`train_holdout_3000`) | test_3000 FP32 |
|---|---|---|
| **1** | **0.819297 ← chosen** | 0.811111 |
| 2 | 0.815113 | 0.794219 |
| 3 | 0.815256 | **0.813919 ← test-best** |

> **THE SELECTION RULE COST 0.0028 OF TEST MACRO-F1 AND WAS FOLLOWED ANYWAY.** Seed 1 is
> **not** the test-best; seed 3 is. Picking seed 3 would have turned `test_3000` into a
> selection split (hard rule 1). That the two disagree is the only reason this is worth
> recording: a selection rule that never costs anything has never been tested.

**E1b DID NOT MEET ITS ACCEPT RULE (§3bi) AND THAT DOES NOT DECIDE THIS.** §3bi governs
whether the *experiment* is accepted, on a 3-seed INT8 mean. This is a *deployment* choice
between two artefacts, measured in the configuration that actually serves.

#### What the served configuration measures — both artefacts, arm64 ONNX FP32, same scorer, same rows

| | E1 seed 1 (served until now) | **E1b seed 1 (served now)** |
|---|---|---|
| `test_3000` macro-F1 | 0.763053 | **0.809052** |
| canary, macOS-arm64 | 183/200 = 0.9150 | **191/200 = 0.9550** |
| canary, Linux-aarch64 | 183/200 = 0.9150 | **191/200 = 0.9550** |
| router threshold @ 4.056% | 0.134668 | **0.528864** |
| retained Tier 0 acc on dev | 0.8775 | **0.9004** |

**+0.045998 on the served configuration.** This is a **cross-host training** comparison and
therefore **descriptive** under §3ak — but the question a deployment answers is *"what does
the artefact score when served?"*, and both sides of this table were measured identically
on this host, through the registered scorer, on the same rows.

#### THE EXPORT PATH, CONTROLLED BEFORE IT WAS TRUSTED

Kaggle **never ships** the FP32 ONNX (`kaggle_tier0.py` deletes it after the E3
discriminator), so it must be exported locally. **The repo venv has drifted off the pin —
`transformers 5.0.0` against the pinned `4.57.6` — and optimum's exporter does not import
against it at all.** The pinned toolchain (`transformers 4.57.6`, `huggingface-hub 0.36.2`,
`optimum 2.1.0`, `optimum-onnx 0.1.0`, `onnx 1.22.0`, `onnxruntime 1.29.0` — the exact
line `kaggle_tier0.py` installs) was reconstructed to do the export.

> **POSITIVE CONTROL, run before exporting anything new.** E1's `fp32_ce_1` was re-exported
> through that reconstructed toolchain and compared against `models/onnx_ce_1_fp32` — the
> bytes §3bg has been **serving**:
>
> `sha256 4363c460…f301` **both**. **BYTE-IDENTICAL.**
>
> So the export path used here reproduces the deployed artefact exactly, and the drift in
> the repo venv did not touch the result. Without this control the new export would rest on
> an environment nobody had validated.

`scripts/export_fp32_onnx.py` is new and delegates to §3be's already-validated
`export_fp32_onnx`, so the served export path and the validated export path cannot drift.
It **refuses** to overwrite an existing export.

#### A FINDING THAT GOES AGAINST THE SWAP, recorded in full

§3bg established FP32 as platform-stable on **0/200** prediction disagreements for E1.
**E1b seed 1 is measurably less stable:**

| | E1 FP32 (§3bg) | **E1b FP32** |
|---|---|---|
| prediction disagreements, host vs container | **0/200** | **1/200** |
| logit correlation | 0.99969 | **0.999144** |
| median \|delta logit\| | 0.000002 | 0.00000286 |
| max \|delta logit\| | 1.08 | **2.785** |
| rows with max\|delta\| > 2.0 | 0/200 | **1/200** |

**Deterministic, not flaky:** two container runs are **bit-identical** (max \|delta\|
0.0000000000, 0/200), and the host-vs-container disagreement reproduces at exactly 1/200.

**The divergent row, opened rather than summarised** — canary row 112, dataset index 30917,
gold `Defined Terms`:

| | prediction | margin |
|---|---|---|
| macOS-arm64 | `Representations` | **0.100370** |
| Linux-aarch64 | `Definitions` | **0.572361** |

- **Both platforms are WRONG on it**, which is why accuracy is identical at 191/200 on
  each. Accuracy parity here is **not** evidence of prediction parity, and would have
  hidden this entirely.
- **IT STRADDLES THE ESCALATION THRESHOLD (0.528864), so the platform difference changes a
  ROUTING decision, not just a label.** On the 200 canary rows the host escalates **3/200**
  and the container **2/200** — that row is the whole difference.

> **WHAT THIS DOES AND DOES NOT LICENSE.** It does not trip the canary floor, does not move
> measured accuracy on either platform, and does not block the swap. It **does** mean
> §3bg's *"FP32 shows 0/200 disagreements"* is a statement about **E1's artefact**, not a
> general property of FP32, and it must not be carried forward as one. Whether a 1/200
> cross-platform routing difference matters at the deployed escalation rate is **not
> measured here** and is not claimed either way.

#### Carried forward unchanged, and still wrong

- **THE CANARY FLOOR STAYS AT 0.80, WHICH IS NOW SLACKER THAN §3bc'S CONSTRUCTION.** That
  construction puts the floor ~0.10–0.115 below measured; at 0.9550 it would give
  ~0.84–0.855. 0.80 sits **0.155** below. Held so the swap changes exactly one thing and
  the refusal boundary stays the one §3bf/§3bg were measured against — **a choice, not a
  derivation**, and it makes the canary less sensitive than the construction would.
- **`estimated_cost_usd` IS STILL AN INT8 FIGURE** on every `/classify` response
  (9.5968e-07 observed post-swap). §3bh's block is unchanged: correcting it requires an
  unmeasured FP32 energy term in a billing path, which hard rule 11 forbids.
- **E1's threshold is superseded, not archived** — `configs/router_threshold_fp32.json` was
  overwritten and E1's calibration lives in git history, which is where a superseded
  version-controlled config belongs. Unlike §3bg's precision rename there is no
  unmarked-default risk: the file records its `artefact`, and `ServiceConfig` refuses at
  startup when it does not match the served weights.

#### Verified after the swap

`/health` reports `tier0_fp32 → claude-sonnet-5`, `model_dir …/onnx_ce10ep_1_fp32`,
canary `ran: true, accuracy 0.955, 191/200, floor 0.8, passed`. `/classify` with **no API
key** answers `Governing Laws` at `tier_used=tier0`. Container built for `linux/arm64` and
the startup canary fires inside it. **553 passed.** **No live Tier 2 call was made and
nothing was appended to `results/spend_ledger.jsonl`.**

#### Explicitly NOT done

**E6, H1 and E8 were not rebuilt and still describe E1.** No frontier, break-even, gap or
cost figure in this repository has been restated against E1b, and none may be read as
having been.

### 3bj. E3's STATISTIC IS AMBIGUOUS, and E3 is NOT unscored — two corrections to §3bi (2026-09-11)

**§3bi claimed "the first measured breach" and "E3 itself remains UNSCORED". Both are
wrong as stated.** Corrected here rather than by editing §3bi, per the append-only rule.

#### 1. THE RULE, QUOTED VERBATIM — and it does not say which statistic

> **Accept rule:** **|INT8 − FP32| ≤ 0.01 macro-F1**, paired. **[C3-gated]**
>
> **Metric / split:** macro-F1 of both precisions on `test_3000`, **paired on identical
> rows** — same model, same rows, so the row draw cancels entirely.

**There is no seed language anywhere in E3.** Not "per seed", not "mean over seeds", not a
seed count. §3bi silently read it **per seed**, which is what produced the breach claim.
**That reading was picked, not derived**, and this project has a registered name for
picking a number and presenting it as anchored (§3e).

**BOTH READINGS ARE DEFENSIBLE. Neither is adopted here.**

| | **reading A — PER SEED** | **reading B — 3-SEED MEAN** |
|---|---|---|
| textual basis | *"same model"* is **singular** — it identifies **one artefact** paired against itself. The stated variance — *"the only variance is quantisation-induced prediction flips"* — is a **within-artefact** quantity, and averaging three *different* quantisations reintroduces the across-seed variance the pairing exists to remove | **hard rule 2**: *"Every model result reports mean ± std over ≥ 3 seeds. Never a single best run."* Every other Tier 0 rule (E1, E1b) binds on the 3-seed mean; reading E3 per seed makes it the **only** Tier 0 rule with a different aggregation, and §2 lists it beside them |
| consequence basis | E3's falsification is *"the **deployed** system is not the measured system"*, and **what deploys is always one artefact, never a mean** | E3's metric line names `test_3000`, the same split the 3-seed rules use, and gives no other unit |
| **E1 (arm64)** | max \|−0.008317\| — **PASSES** | −0.004796 — **PASSES** |
| **E1b** | **\|−0.013668\| — FAILS** | **−0.007672 — PASSES** |

**E1, per seed, arm64, computed for this entry:** −0.005378 / −0.008317 / −0.000693, mean
**−0.004796**, max **\|0.008317\|** — which reproduces the **0.0083** figure §3bc and §3bi
both cite, so the two experiments are on a common basis.

> ### THE BREACH EXISTS ONLY UNDER READING A.
>
> Under **A**, E1b seed 1 is the **first and only** measured breach and §3bi's claim stands.
> Under **B**, **no Tier 0 artefact has ever breached E3**, E1b included, and §3bi's claim
> is simply false. **E1 passes under both**, so the readings have never diverged before and
> nothing earlier in the project decides between them.

**A NOTE THAT CUTS AGAINST READING A, recorded because it is against the reading §3bi
used.** A per-seed breach rule fires on the **single worst** of three runs. Hard rule 2
forbids reporting the single **best** run; it does not literally address the worst, but a
max-over-seeds rule is that prohibition's mirror image and inherits its problem —
**3 draws from a distribution have a worse maximum than 1 draw**, so reading A's effective
tolerance tightens as seed count rises, with no compensation registered anywhere.

**NOT RESOLVED HERE, AND THE REASON MATTERS.** Choosing now means choosing **after** seeing
which reading breaches, on the one artefact family where they disagree. That is the
ordering hard rule 6 exists to prevent, and it is the same failure §3bi's own
[C3-gated]-tolerance paragraph refuses three lines further down. **The ambiguity is the
finding.** Resolving it is a registration act that must happen **before** the next
INT8-vs-FP32 number is opened, and it must state its statistic in E3's own text.

**WHAT IS TRUE UNDER BOTH READINGS, so the deployment decision does not wait on this:**
E1b seed 1's INT8 artefact carries a **measured \|delta\| of 0.013668**, the largest this
project has recorded. §3bk serves that seed at **FP32**, so no INT8 delta sits in the
served path at all; E3's falsification consequence is **not engaged by the swap** under
either reading.

#### 2. E3 IS MEASURED, NOT PLANNED — §2's status was wrong

§3bi wrote *"E3 itself remains UNSCORED"* and left §2's status at `planned`. **E3 has been
measured twice**: on E1 (arm64, all 3 seeds, max \|0.0083\|) and now on E1b (all 3 seeds).
Its numbers have been **in use** — §3bc's canary-floor reasoning is built on E3's largest
delta. Carrying `planned` beside numbers the project already depends on is §3ax's class
exactly: **a status that no longer describes the thing it labels.**

**Corrected status: MEASURED on both Tier 0 families; VERDICT WITHHELD pending the
statistic (§3bj.1).** The measurement is complete; what is missing is which function of it
the rule reads. A results-log entry is filed with both readings and no verdict, and §2 now
says so.

**Why not simply score it under reading B and be done:** B is the reading under which E3
passes. Adopting the passing reading at the moment the other one fails is the choice this
entry exists to refuse.

### 3bi. E1b VERDICT — NOT ACCEPTED, by 0.0013. The near-miss is reported as a near-miss (2026-09-11)

**THE REGISTERED RULE, quoted before the number:** *"macro-F1 ≥ 0.80 on `test_3000`,
measured on the ONNX-INT8 artefact, mean over seeds 1/2/3 … the same bar as E1,
deliberately not a bar chosen to match what 10 epochs is expected to reach. If E1b lands
at, say, 0.79, that is a **near-miss to be reported as a near-miss, not a rule to be
relaxed afterwards**."*

#### The result

| seed | E1b INT8 `test_3000` (arm64) | E1 INT8 (arm64) |
|---|---|---|
| 1 | 0.797443 | 0.758213 |
| 2 | 0.792468 | 0.751423 |
| 3 | **0.806321** | 0.747119 |
| **mean ± sd** | **0.798744 ± 0.007018** | 0.752252 ± 0.005593 |

> ## VERDICT: **NOT ACCEPTED.** 0.798744 < 0.80.
>
> Shortfall **0.001256** — **0.18×** E1b's own seed sd. **1 of 3** seeds clears the bar
> individually. The rule is not relaxed, not re-anchored, and not restated to fit.

**AND THE HYPOTHESIS IS SUPPORTED — on the one-variable comparison, which is not the one
this entry first used.**

> **CORRECTION, applied before this entry was final (§3ak).** The first draft rested the
> undertraining claim on **+0.046492 = 8.31× E1's seed sd**, E1b's 3-seed mean against
> E1's. That is a **cross-host, two-variable** comparison — epochs **and** host — reported
> as one, and §3ak registers exactly that construction as **rejected**: E1's training
> environment was never recorded (§3e instance 9), so a difference across it is
> **detectable but not attributable**, and no metadata converts it back into one variable.
> The figure is retained below as **descriptive only** and carries no causal weight.

The comparison §3ak step 1 was spent to make, and the one the claim now rests on — **one
variable, both arms on this host, both training environments persisted by `train_env()`,
both scored INT8 on arm64**:

| arm | epochs | seeds | INT8 `test_3000` |
|---|---|---|---|
| host baseline (`ce_hostB`) | 3 | **n = 1** | 0.7589 |
| **E1b seed 1** | **10** | **n = 1** | **0.7974** |
| **within-host epochs effect** | | **n = 1** | **+0.0385** |

**n = 1 ON BOTH SIDES, AND THAT BOUNDS THE CLAIM.** The host baseline was never run at
more than one seed, so this comparison has **no variance estimate of its own**. Against
§3at's borrowed 2σ yardstick of 0.0112, +0.0385 is **3.4×** the noticing threshold, so an
epochs effect is **detected** — §3at licenses that reading and licenses no stronger one. It
is not a 3-seed result and hard rule 2 forbids presenting it as one.

> Stated with the scope it actually has: *on this host, at n = 1, going from 3 to 10
> epochs moves INT8 `test_3000` by **+0.0385**, and 10 epochs still does not reach 0.80.*
> Undertraining is a **real and substantial** cause of E1's shortfall and is **not the
> whole** of it. A report that said only "E1b failed" would be as wrong as one that said
> "E1b confirmed undertraining".

**Descriptive only, retained and labelled (§3ak step 4, §3at):** E1b's 3-seed mean against
E1's 3-seed mean is **+0.046492**; E1b seed 1 against E1 seed 1 is **+0.0392**. Both cross
the unrecorded-host boundary. They agree in sign and rough magnitude with the within-host
+0.0385, which is **corroboration, not evidence**.

**THE EPOCH BUDGET IS STILL BINDING, which bears directly on the near-miss.** Selection
best was **epoch 10 — the last one — for seeds 2 and 3** (0.8151 / 0.8153), and epoch 9
for seed 1. For two of three seeds the run was **still improving when it hit the cap**, so
10 epochs is plausibly still undertrained. **This is an observation, not a result, and it
does NOT rescue the verdict**: no rule permits inferring where the curve would have gone,
and a 20-epoch run is a new experiment requiring its own registration and its own quota.

**FP32, REPORTED ALONGSIDE AND EXPLICITLY NOT THE RULE.**

| | E1b FP32 | E1 FP32 |
|---|---|---|
| mean ± sd | **0.806416 ± 0.010656** | 0.757048 ± 0.008227 |
| per seed | 0.811111 / 0.794219 / 0.813919 | — |

> **FP32 CLEARS 0.80 AND THAT IS NOT THE RULE.** The accept rule names the ONNX-INT8
> artefact, and §3ar correction 1 exists *precisely because* an FP32 figure was once
> quoted against an INT8 rule. Reporting 0.8064 as "E1b met its bar" would repeat that
> error with the direction that flatters the project. It did not.
>
> All three FP32 values were recomputed locally from `logits_ce10ep_seed*.npz` through the
> registered scorer with the `test_3000_indices` guard, and reproduce Kaggle's recorded
> figures to **0 delta** at n=3000.

**THE PAIRED QUANTISATION DELTA, WHICH E3 OWNS — and it is larger than the miss.**

| seed | INT8 | FP32 | **INT8 − FP32** |
|---|---|---|---|
| 1 | 0.797443 | 0.811111 | **−0.013668** |
| 2 | 0.792468 | 0.794219 | −0.001751 |
| 3 | 0.806321 | 0.813919 | −0.007598 |
| **mean** | 0.798744 | 0.806416 | **−0.007672** |

> **Seed 1's |−0.013668| exceeds 0.01; the 3-seed mean of |−0.007672| does not.** E1's
> largest was |0.0083| and its mean −0.004796, so E1 sits under 0.01 **both ways**.
>
> **⚠ AMENDED — see §3bj.1.** This entry originally called seed 1 "the first measured
> breach". **That assumes a per-seed reading which E3's registered text does not state.**
> Under the 3-seed-mean reading **E1b passes and no Tier 0 artefact has ever breached E3.**
> Both readings are recorded in §3bj and in §4's E3 entry; **neither is adopted**, because
> adopting one now means adopting it after seeing which one fails.

**IS E3 SCOPED TO E1, OR TO EVERY TIER 0 ARTEFACT? CHECKED RATHER THAN ASSUMED — and the
answer is the one that costs us something.** E3's registration **names no run**: its
hypothesis is *"INT8 dynamic quantisation does not materially degrade **Tier 0**"*; its
metric is *"macro-F1 of both precisions on `test_3000`, paired on identical rows — same
model, same rows"*, where "same model" pairs **each artefact against itself** rather than
restricting which artefact; and §2's E3 row reads *"INT8 vs FP32 **at the deployed
precision**"*. **E3 is artefact-general across Tier 0, not E1-scoped.** E1b's artefacts are
therefore in scope. **Whether seed 1 fires E3's falsification clause depends on a
statistic the rule never specifies — see §3bj.1.** It fires under a per-seed reading and
does not fire on the 3-seed mean.

**What that does and does not mean, against E3's own falsification text** — *"delta > 0.01
⇒ the deployed system is not the measured system. Report the INT8 number as the headline
regardless, with the delta stated. Reporting the FP32 figure as the system's accuracy is
prohibited."*

- It **does** mean the INT8 figure stays the headline and the delta is stated. Both done
  above, and the FP32 figure is not reported as the system's accuracy anywhere in this
  entry.
- It does **not** bind the running system today: **no E1b artefact is deployed.** §3bg
  serves `fp32_ce_1`, an **E1** artefact, at FP32. The clause binds on any future decision
  to serve an E1b artefact at INT8 — a swap §3bc already gates behind recalibration, and
  which this breach makes materially harder rather than merely procedural.
- **E3's tolerance is unchanged at 0.01.** It was registered **[C3-gated]** against the
  risk that a seed sd around 0.03 would render 0.01 meaningless; C3 measured
  **`seed_sd = 0.003195`**, so the gate lifts **in favour of 0.01 standing**. No
  substitution was ever due under the gate, and none is made now that a number has
  breached it — that ordering is the whole point of the gate.
- ~~**E3 REMAINS UNSCORED AS AN EXPERIMENT.**~~ **⚠ WITHDRAWN — see §3bj.2.** E3 has been
  **measured on both Tier 0 families** (E1 arm64, all 3 seeds; E1b, all 3 seeds) and its
  numbers are already load-bearing — §3bc's canary floor is built on E3's largest delta.
  Calling it `planned` beside figures the project depends on is §3ax's class. **Corrected
  status: MEASURED, VERDICT WITHHELD pending the statistic.** Result entry filed in §4
  with both readings and no verdict.

**THE MISS IS SMALLER THAN THE QUANTISATION COST — stated, and explicitly not a rescue.**
FP32 clears 0.80 at **0.806416**. The shortfall is **0.001256**. The mean INT8−FP32 penalty
is **0.007672**, which is **6.1× the miss**; seed 1's penalty alone is **10.9×** it.

> **THIS DOES NOT CONVERT THE VERDICT INTO A PASS.** The rule names INT8, INT8 is what was
> measured, and 0.798744 < 0.80. What the comparison legitimately says is *where the
> missing 0.0013 most plausibly sits* — inside a quantisation penalty this project already
> registered a tolerance for and has now exceeded on one seed. That is a pointer to the
> next experiment, not an argument about this one.

**PROVENANCE, CHECKED RATHER THAN ASSUMED.** Seeds 2 and 3 were quantised under the
**pinned** toolchain — `onnxruntime 1.29.0`, `optimum 2.1.0`, `optimum-onnx 0.1.0`,
`onnx 1.22.0`, `transformers 4.57.6`, with `quantiser_versions_complete: true`, the first
runs to record optimum at all (§3bd's capture fix, working). **Seed 1 predates the pin**
and was quantised under **ORT 1.30.0** with optimum/onnx unrecorded. **The 3-seed mean
therefore mixes two toolchains.** That is licensed by §3be, which measured 1.29.0 against
1.30.0 on this exact model as **byte-identical across all 376 initializers** — but it is
stated here rather than left for a reader to discover.

All three scored on arm64 under ORT 1.29.0 through `scripts/score_int8_local.py`, n=3000,
0 unmatched, `classes_averaged=100`.

**DISCLOSURE, VOLUNTEERED, AND IT CHANGES NOTHING.** E1b's accept rule names **INT8
because INT8 was the deployed precision when the rule was written**. **§3bg (2026-09-11)
moved serving to FP32**, so the rule's premise no longer describes the running system — and
FP32 is the arm that clears 0.80. **The verdict is unchanged and the rule is not re-read.**
E1b was registered against INT8, measured on INT8, and is scored on INT8 at **0.798744**. A
rule may not be re-anchored to whichever precision turns out to pass once the number is
known, and the fact that such a re-anchoring is now *available* is precisely why it is
refused **in writing** rather than left unmentioned for a reader to notice later.

**What does NOT follow from this verdict:** that Tier 0 is unfit to deploy. §3bg's serving
decision rests on FP32 platform-stability and the canary, not on E1b's accept rule, and
the deployed artefact is still E1 seed 1. Swapping to an E1b artefact remains a separate
decision requiring recalibration (§3bc).

### 3bh. FP32 Tier 0 BENCHED; which published numbers need restating (2026-09-11)

**Not an experiment.** §3bg moved serving to FP32. Every local cost and latency figure
published so far was measured on **INT8**, so this records the FP32 measurement and lists
exactly what it does and does not license.

**MEASURED — same protocol and replicate count as the INT8 row** (3 artefacts × 3 runs = 9,
batch 1, `max_length` 512, mean over all runs, no outlier excluded, §3aj). The three FP32
ONNX exports come from `fp32_ce_{1,2,3}` through the path §3be validated byte-for-byte.

| | INT8 (unchanged) | **FP32** |
|---|---|---|
| throughput req/s | 28.3615 ± 1.1839 | **24.5969 ± 2.2374** |
| p50 latency ms | 25.3969 | **29.3805 ± 2.7296** |
| p95 latency ms | 99.7950 | **110.9449 ± 10.4128** |
| n runs | 9 | **9** |
| energy J/req | 0.6269 | **UNMEASURED** |

**FP32 is only 1.15× slower** — far less than the 715 MB vs 244 MB artefact gap suggests.
Per-artefact means 24.2966 / 25.1584 / 24.3356, a 3.5% spread against a 9.1% within-run
sd, consistent with the INT8 row's weight-independence though not formally re-tested.

**ENERGY IS UNMEASURED AND THAT IS A HARD BLOCK, not a gap to be filled with the INT8
number.** `powermetrics` needs sudo and none is available here, so all 9 runs are marked
`complete_for_e6=false` and `scripts/bench_aggregate.py` **refuses** them. That refusal is
correct: E6's asymptote *is* the power term. The INT8 row is untouched — it remains the
measured basis of every E6 number published, and overwriting it would restate results
rather than add to them.

---

#### What needs restating, in three groups

**A.** ~~**BLOCKED ENTIRELY on the FP32 energy measurement.**~~ **✅ UNBLOCKED 2026-09-11
— the FP32 energy is MEASURED (§3bl).** `sudo powermetrics` was obtained and the bench ran
on the **served** artefact, 3 runs, no run excluded.

| figure | ~~INT8 value~~ superseded for the served row | **FP32, measured** |
|---|---|---|
| `energy_joules_per_request` | ~~0.6269~~ | **0.505668 ± 0.008813** |
| `power_draw_soc_watts_marginal` | ~~17.7546~~ | **15.6350 ± 0.5710** |
| `tier0_usd_per_1k_sunk` | ~~1.4750e-05~~ | **1.189724e-05** |
| Sonnet-5-to-Tier-0 cost ratio | ~~**29,567×**~~ | **36,656×** |
| local $/1k (capital + energy) | ~~0.00095968~~ | **0.00087881** |
| E6's **sunk** curve and its Pareto points | **NOT REBUILT — E6 stays on E1 as registered** | — |
| `costs.yaml` energy_note's "1/30,395 of $0.4483" | superseded twice over | rewritten against $0.43610 |

> **THE INT8 ROW ITSELF IS UNTOUCHED.** It remains exactly as measured and remains the
> basis of every published E6 number. "Superseded" above means *superseded as the basis
> for the **served** row*, not withdrawn.
>
> **AND THE DIRECTION IS THE OPPOSITE OF WHAT GROUP B PREDICTED — do not read it as a
> finding.** These numbers say FP32 is both faster and ~8.4% CHEAPER than INT8, against
> group B's projected +15% to +20%. **That reversal is not supported**: the two rows come
> from different sessions, and §3bl's interleaved control measured the *same artefact*
> moving **19.95%** between sessions — larger than the gap. The precision penalty on this
> host is **UNMEASURED**, and settling it needs an interleaved INT8-vs-FP32 run that has
> not been done.

**B. COMPUTABLE from today's throughput; only the last ~1.5% waits on energy.** The capital
term dominates: energy is **1.54%** of the INT8 local per-request cost.

| figure | INT8 | FP32 |
|---|---|---|
| `assumed_lifetime_requests` | 670,805,531 | **581,765,879** (−13.3%) |
| capital per request | 9.4493e-07 | **1.0895e-06** |
| local $/1k | $0.00095968 | **$0.0011043 – $0.0011486** (+15% to +20%), the range spanning FP32 energy from 1× to 4× INT8 |
| `tier0_usd_per_1k_greenfield` at V=1e6 | 0.63388 | restate with the above |

> ~~**LIVE AND CURRENTLY WRONG-PRECISION:**~~ **✅ FIXED 2026-09-11 (§3bl).**
> `local_usd_per_request()` is now **precision-keyed**: it reads the served precision's own
> `per_tier_throughput` row and **raises `MeasurementUnavailable`** for a precision whose
> energy is unmeasured, rather than borrowing the other one's. `/classify` now reports
> **$8.788115e-07 per request ($0.00087881 per 1k)** with an FP32 basis string.
> The old prediction that it "understates by ~15%" was **wrong in sign**: the INT8 figure
> it was quoting **over**stated by 9.2%. See the caveat above — that gap is cross-session,
> not a precision effect. A regression test pins the served precision into the response.

**C. ESSENTIALLY UNCHANGED — and this is the one that matters most.**

> **V\* ≈ 1,453,465 clauses does NOT move with precision.** Its denominator is
> `api_usd_per_1k − marginal_local_per_1k` = 0.43610 − 1.475e-05, and the local term is
> **0.003%** of it. Even at 4× the INT8 energy, V\* shifts by ~0.01%. The break-even volume
> is set by the device price against the API price; throughput changes the *amortisation*,
> not the crossover. **E6's headline conclusion is precision-independent.**

**D. SEPARATE AXIS.** E7's latency inputs move and are measured: p50 25.40 → **29.38 ms**,
p95 99.80 → **110.94 ms**. E6 does not consume latency (§3aa), so no cost conclusion
depends on this.

### 3bg. DEPLOYMENT DECISION — serve FP32 ONNX, not INT8 (2026-09-11)

**Not an experiment. A deployment decision, with the verification that licensed it.**
§3bf established that INT8 is qualified on no Linux target. FP32 ONNX was then checked on
the same two platforms, with the same metrics, before anything was built on it.

#### 1. FP32 across macOS-arm64 and Linux-aarch64 — the §3bf comparison, repeated

Exported from `fp32_ce_1` through the export path the §3be control validated byte-for-byte
(`transformers 4.57.6`, `optimum 2.1.0`, `optimum-onnx 0.1.0`, `onnx 1.22.0`), then scored
on the 200 frozen canary rows on both platforms under `onnxruntime 1.29.0`.

| metric | INT8 (§3bf) | **FP32** |
|---|---|---|
| host accuracy (macOS arm64) | 0.9000 | **0.9150** |
| container accuracy (Linux aarch64) | **0.6400** | **0.9150** |
| prediction disagreements | 60/200 | **0/200** |
| logit correlation | 0.8660 | **0.99969** |
| median \|Δlogit\| | 0.5171 | **0.000002** |
| rows with max \|Δlogit\| > 2.0 | 118/200 | **0/200** |
| rows with max \|Δlogit\| < 0.5 | 3/200 | **194/200** |

**FP32 is platform-stable where INT8 is not, and it is also more accurate** — 0.9150
against 0.9000 on the host. The FP32 residual is not zero (max \|Δ\| 1.08 on one row), so
this is "no prediction moved", not "bit-identical".

#### 2. The only x86 evidence, from E3's discriminator arm

Two runs recorded the ONNX-FP32 arm; E1's three seeds predate it. Both on Kaggle's
Intel Xeon — `avx512f` and `avx2` present, **`avx512_vnni` absent**:

| run | torch FP32 | ONNX FP32 | ONNX INT8 | argmax agree (torch↔onnx FP32) | argmax agree (onnx FP32↔INT8) |
|---|---|---|---|---|---|
| host baseline (3 ep) | 0.763862 | 0.764108 | **0.000777** | **99.93%** | **1.2%** |
| E1b seed 1 (10 ep) | 0.811111 | 0.811466 | **0.000166** | **99.97%** | **0.57%** |

**On the one x86 host this project has measured, ONNX-FP32 tracks torch FP32 to within
+0.0004 macro-F1 at ~99.95% argmax agreement, while INT8 collapses to chance on the same
machine, in the same process, with no error raised.** The export step is not the problem;
quantisation is.

> **Both conditions held, so the decision proceeds.** Stated precisely: FP32 ONNX is
> verified on **two** platforms directly and on a **third** (x86) only against torch FP32
> from the same run. HF Spaces x86 is still unqualified as a *serving* host — the canary
> is what would qualify it, and it now runs on FP32 numbers.

#### 3. What is now keyed by precision

`configs/serve.yaml` carries `tier0.precision` (`fp32` | `int8`), and **every
precision-dependent artefact is keyed by it**: the model directory, the router threshold
file, and the canary's measured accuracy and floor. A bare unkeyed value is **refused** —
sharing one would silently serve one precision's calibration to the other.

**The threshold is calibrated on dev logits of the SERVED precision**, which is not a
formality: at the same registered 4.056% escalation rate the thresholds are

| precision | threshold | dev logits |
|---|---|---|
| fp32 | **0.134668** | `results/dev_logits_fp32_local_ce_seed1.npz` (fp32 / arm64_local) |
| int8 | 0.115365 | `results/dev_logits_int8_local_ce_seed1.npz` (int8 / arm64_local) |

`ServiceConfig` refuses to start when the threshold's recorded precision differs from the
served one. **The INT8 floor stays at 0.80** (§3bf) and the INT8 arm remains fully
configured — it is not deleted, it is not selected.

#### 4. Container measurements

| | |
|---|---|
| image size | **1.01 GB** — model NOT included |
| model, mounted | **715 MB** FP32 (against 244 MB INT8) |
| startup | canary 183/200 = **0.9150** in-container, matching the host reference exactly |
| latency, n=60, end-to-end HTTP | mean **167 ms**, p50 **136 ms**, p95 **459 ms**, min 38 / max 874 |

Median clause 612 characters. **The FP32 cost is size and latency, not accuracy**: p50
136 ms against the INT8 encoder's 25.4 ms p50 measured bare-metal on the host
(`configs/costs.yaml`), though those two numbers are not comparable — one is end-to-end
HTTP inside a VM, the other is in-process inference on the host. A like-for-like FP32
latency measurement has **not** been taken.

**Fixed in passing:** `/health` reported `architecture: "tier0_int8 -> claude"` as a
hardcoded string and kept saying so while serving FP32 — a health endpoint describing a
different system than the one answering. It now reads the served precision.

### 3bf. THE CANARY FIRED IN A CONTAINER — INT8 diverges macOS-arm64 vs Linux-aarch64 (2026-09-11)

**Not an experiment. A deployment measurement, and the §3bc canary doing exactly what it
was registered to do.** The service image was built and run locally with
`models/int8_ce_1` mounted. It **refused to start**:

> `TIER 0 CANARY FAILED: accuracy 0.6400 on 200 frozen TRAIN rows is below the floor
> 0.8000.` `machine: aarch64, system: Linux, onnxruntime 1.29.0`

The same artefact scores **0.9000** on the host. Nothing was misconfigured — the container
mounted the same weights, loaded the same `configs/`, and ran the same code.

**RULED OUT, in order, rather than assumed.**

| candidate | result |
|---|---|
| tokenizer version | container `transformers` **4.57.6** (the version that TRAINED the model), host 5.0.0 |
| tokenization itself | **byte-identical** `input_ids` on the same text, all 25 tokens |
| onnxruntime version | **1.29.0 in both** |
| numpy version | **2.4.6 in both** |

**IT IS THE INT8 KERNEL, and the divergence is broad rather than a few bad rows.** Over the
200 canary rows:

| | |
|---|---|
| host accuracy (macOS arm64) | **0.9000** (180/200) |
| container accuracy (Linux aarch64) | **0.6400** (128/200) |
| prediction disagreements | **60/200** |
| logit correlation | **0.8660** |
| rows with max \|Δlogit\| > 2.0 | **118/200** |
| rows with max \|Δlogit\| < 0.5 | **3/200** |

A few broken rows would leave correlation near 1.0 with a handful of outliers. 0.866 with
118/200 rows moving more than 2 logits is **pervasive numeric divergence in the quantised
kernel**, between two builds of the same onnxruntime version on the same physical CPU.

**MECHANISM — HYPOTHESIS, NOT ESTABLISHED.** onnxruntime prints
`cpuid_info warning: Unknown CPU vendor. cpuinfo_vendor value: 0` inside the container.
If ORT cannot identify the CPU it cannot detect the ARM dot-product extensions
(`dotprod`/`i8mm`) its INT8 kernels use, and would fall back to a generic accumulation
path. That fits the evidence but has **not** been confirmed — confirming it means reading
ORT's dispatch, not reading its warning.

**CONSEQUENCES.**

1. **The canary is vindicated as a refusal rather than a warning.** Without it this image
   would have served 0.64-accuracy labels with plausible margins, and the margin signal —
   which the router thresholds on — would have been computed from the same divergent
   logits. Nothing else in the stack would have raised.
2. **§3ah's finding is broader than "x86 without VNNI".** It was recorded as an x86
   non-VNNI problem; it reproduces between two ARM platforms. Any statement of the form
   "INT8 is fine on arm64" is now too coarse: it is fine on *this* host, measured.
3. **Every INT8 number in this project is host-qualified.** The gate figure
   **0.7974** (§3be) was measured on macOS-arm64 and is not transportable to a Linux
   serving host without re-measurement. The gate verdict is unaffected — it compares E1b
   against the host baseline *on the same machine* — but a deployed accuracy claim is not
   established by it.
4. **HF Spaces is x86 Linux**, i.e. a third platform, neither of the two measured here.
   **No deployment target is currently qualified.** Qualifying one means running the canary
   on it, which is exactly what the container refusing to start already demonstrates.

**The floor was NOT lowered, and lowering it would be the error the floor exists to
prevent.** 0.80 was derived in §3bc from a measured 0.9000 with ~12× the largest INT8
delta this project had seen as headroom. A floor moved to accommodate a failing platform
stops being a measurement of health and becomes a record of what that platform happens to
produce.

### 3be. E1b SEED-1 GATE READ — ON TRACK; the ORT confound is CLOSED (2026-09-11)

**THE CONFOUND REGISTERED OPEN IN §3bd IS NOW CLOSED BY MEASUREMENT, NOT BY ARGUMENT.**
`scripts/verify_quantiser_repro.py`, both phases, on the downloaded artefacts:

| phase | onnxruntime | result |
|---|---|---|
| **control** | 1.30.0 (the reference's own) | **376 / 376 initializers byte-identical** |
| **candidate** | 1.29.0 (host baseline's) | **376 / 376 initializers byte-identical** |

The control passing rules out arm64-vs-x86, the local export, optimum and onnx in one
step; the candidate then shows **onnxruntime 1.29.0 and 1.30.0 produce the same quantised
tensors** for this model. **There is no quantiser confound, the gate reads the downloaded
`int8_ce10ep_1` bytes unmodified, and §3bd's open item is discharged.**

> Scope, stated because it is narrower than "the versions are equivalent": this is
> byte-identity of the QUANTISED WEIGHTS for one model under one config. It says nothing
> about inference-kernel differences between ORT versions at scoring time. Scoring was
> done under 1.29.0, the same version the host baseline was scored under.

**A CORRECTION TO THE RECORD, made because the claim was mine and it was wrong.** §3bd
said the FP32 ONNX was "retained deliberately as E3's middle arm". **It is not retained.**
`kaggle_tier0.py` deletes `onnx_<tag>_<seed>/` after the discriminator uses it, and the
E1b commit-2 output contains no `onnx_ce10ep_1`. The comment above the export — "previously
it was deleted immediately after quantising" — governs **ORDER**, not survival, and was
misread as a retention guarantee. Consequences: `onnx_*` in `_RESTORE_GLOBS` can never
match, and no check may assume the FP32 ONNX is downloadable. It is **reproducible**
instead — the control phase exported it locally from `fp32_ce10ep_1` and the result
quantised to Kaggle's exact bytes, so the local export is itself validated (under torch
2.14.0 locally vs 2.10.0 on Kaggle, which therefore did not matter here).

---

#### THE GATE (§3ar), READ

> **E1b seed 1, INT8, `test_3000`, arm64: macro-F1 = 0.7974** (accuracy 0.8720, n = 3000,
> 98/100 classes predicted, 0 unmatched). Artefact `int8_ce10ep_1`, scored by
> `scripts/score_int8_local.py` through the registered scorer.

| boundary | value | |
|---|---|---|
| **T** (0.80 − 1.645σ) | 0.7908 | **0.7974 ≥ T** |
| M | 0.7771 | |
| B | 0.7635 | |

> ### VERDICT: **ON TRACK → RUN SEEDS 2–3.**

**What it is NOT.** E1b's accept rule is macro-F1 **≥ 0.80 on `test_3000`, INT8, MEAN OVER
3 SEEDS**. 0.7974 is **one seed** and sits **below 0.80**. It clears the gate that
authorises spending quota on seeds 2–3; it does not satisfy the accept rule, and hard rule
2 forbids reporting a 1-seed figure as a result.

**The comparison E1b was designed to make (§3ak step 1) — one variable, both arms on this
host, both scored on arm64:**

| arm | epochs | INT8 `test_3000` macro-F1 |
|---|---|---|
| host baseline | 3 | 0.7589 |
| **E1b seed 1** | **10** | **0.7974** |
| **within-host epochs effect** | | **+0.0385** |

For context only, and **not** the within-host comparison: E1's 3-seed INT8 mean is 0.7523
(+0.0451) and E1 seed 1 alone is 0.7582 (+0.0392).

**FP32, REPORTED ALONGSIDE AND EXPLICITLY NOT THE GATE.** Every boundary in §3ar is
defined on INT8, the deployed precision; §3ar correction 1 exists because an FP32 figure
was once quoted against an INT8 rule.

| arm | FP32 `test_3000` macro-F1 |
|---|---|
| host baseline (3 ep) | 0.7639 |
| E1b seed 1 (10 ep) | **0.8111** |

Recomputed from `logits_ce10ep_seed1.npz` through the registered scorer and matching
Kaggle's recorded `test_3000_fp32` to **0 ULP**. Selection macro-F1 rose monotonically
across epochs 6–10 (0.7995 / 0.8083 / 0.8083 / 0.8193 / 0.8183) and `load_best_model_at_end`
selected **epoch 9** (0.8193), confirming §3bb's checkpoint-survival analysis end to end.
Wall clock 12,927 s.

**KAGGLE'S OWN INT8 FIGURE IS 0.00017 AND IS DIAGNOSTIC ONLY.** That is the third time this
non-VNNI x86 host has scored a healthy INT8 artefact at chance with no error raised, against
**0.7974** for the same bytes on arm64. It is exactly the failure the deployment canary
(§3bc) exists to refuse to start on, and the gap between the two numbers is now measured
rather than argued.

### 3bd. E1b RUN RECORD — what commit 2 actually ran, and the ORT drift (2026-09-11)

**Recorded because the file no longer says.** §3bb registered the plan; this records the
execution, which differed from the committed file in two ways that nothing on disk
captured.

| | commit 1 | commit 2 |
|---|---|---|
| ran | 2026-09-10 | 2026-09-11 |
| `RUN_STEP_BUDGET` | 17815 | **None** |
| `RESUME_FROM_STEP_AT_LEAST` | 0 | **17815** |
| `EPOCHS` | 10 | 10 |
| trains | steps 1 → 17,815 (epochs 1–5) | 17,815 → 35,630 (epochs 6–10) |
| onnxruntime | **1.29.0** | **1.30.0** |
| code | pre-pin | **pre-pin** |

**CORROBORATION — WHAT IS OBSERVED AND WHAT IS ONLY DECLARED.** Commit 2's arm was set on
the Kaggle copy, so none of it is attested by this repository. The run log attests part of
it:

| value | status | evidence |
|---|---|---|
| `RESUME_FROM_STEP_AT_LEAST = 17815` | **corroborated** | `restored 1: ['ck_ce10ep_1']`, `=== seed 1 (RESUMING) ===`, `[train] begin at step 17815 of 35630` |
| resume actually advanced | **corroborated** | same three lines — the chain did not restart from zero |
| `RUN_STEP_BUDGET = None` | **corroborated** | the run ended `ALL SEEDS DONE` with **no** `COMMIT STEP BUDGET EXHAUSTED` line — the budget never fired |

All three are now observed. The budget line mattered: a budget that silently fired would
have stopped commit 2 early and left a partially trained seed reporting as complete.

**COMMIT 2'S ARM NEVER REACHED THE REPOSITORY.** It was set on the Kaggle copy, so the
committed file continued to hold commit 1's values. The two disagreed with nothing saying
so, and a blind re-upload would have retrained **epochs 1–5 again** — ~4 h of quota
producing a checkpoint indistinguishable from progress. Fixed structurally: `RUN_ARM` is
now a required declaration that defaults to `None`, PREFLIGHT refuses an undeclared arm
before a GPU slot is spent, and `RUN_STEP_BUDGET` / `RESUME_FROM_STEP_AT_LEAST` are
**derived** from it rather than hand-set, so they cannot drift apart again. The file is
currently **UNARMED**; seeds 2–3 are armed only after the seed-1 gate (§3ar) is read,
because arming them earlier pre-commits the spend the gate exists to decide.

**THE QUANTISER MOVED MID-EXPERIMENT, AND IT IS THE ONLY THING THAT DID.** The install
cell pinned only `transformers`, so onnxruntime drifted **1.29.0 → 1.30.0 between two
halves of one seed.** Training is torch-only, so the drift touches only commit 2's
**ONNX-export + INT8-quantise tail** — but that tail is precisely what the gate reads.
§3ar's gate compares **E1b INT8 against the host baseline INT8**, so a quantiser version
change is **a second variable inside a difference registered to isolate epochs**.

**Scope of the change, from the Kaggle pip logs — NOT from `train_env`.** This distinction
is load-bearing: `train_env` recorded `optimum: null` in every run (the reason is below),
so it could not have established this and is not the evidence here.

| component | host baseline (`rd-tier0-hostb`) | E1b commit 2 | |
|---|---|---|---|
| `onnxruntime` | **1.29.0** | **1.30.0** | **MOVED** |
| `optimum` | 2.1.0 | 2.1.0 | same |
| `optimum-onnx` | 0.1.0 | 0.1.0 | same |
| `onnx` | 1.22.0 *(already satisfied)* | 1.22.0 *(already satisfied)* | same |
| `transformers` | 4.57.6 | 4.57.6 | same |
| `huggingface-hub` | 0.36.2 | 0.36.2 | same |

> **`onnxruntime` is the only pip-visible change between the two runs.** The host
> baseline's `ONNX ENV: ort 1.29.0` line, printed at quantise time, confirms the installed
> version is also the one that ran the quantiser rather than merely the one resolved at
> install.

"Pip-visible" is the honest bound on that claim. It covers what the install cell reports;
it does not cover a base-image change beneath those packages, and no run recorded enough
to rule that out. The whole toolchain is now pinned (`onnxruntime==1.29.0`,
`optimum==2.1.0`, `optimum-onnx==0.1.0`, `onnx==1.22.0`, `transformers==4.57.6`) and
asserted in PREFLIGHT from `importlib.metadata`, so the next run's parity is a check
rather than a reconstruction.

**THE CONFOUND STAYED OPEN UNTIL MEASURED, AND IS NOW CLOSED — see §3be.** Same versions
everywhere else narrowed it to onnxruntime but did not establish that 1.29.0 and 1.30.0
produce the same tensors. Both phases of the check have now run and both are byte-identical
across all 376 initializers, so the gate reads the downloaded bytes unmodified.

**Status: CLOSED by §3be. The protocol below is retained as registered, because it is what
the outcome is evidence from.** `scripts/verify_quantiser_repro.py`
re-quantises the FP32 ONNX and compares initializer tensors against the downloaded
artefact. **(Correction: the FP32 ONNX is NOT retained — `kaggle_tier0.py` deletes it after
the discriminator uses it. The script exports it locally instead; see §3be.)** It runs in two phases and
**the first is a positive control**: re-quantise with onnxruntime held at the reference's
own 1.30.0 and require byte-identity, which rules out arm64-vs-x86, optimum and onnx at
once. Only behind a passing control does the 1.29.0 comparison mean anything, and the
script **refuses** to run the second phase otherwise — a NOT-IDENTICAL from a broken
instrument is indistinguishable from a real quantiser difference. Outcomes:

- **control fails** → the confound **could not be tested**; the gate carries it as a
  stated limitation.
- **control passes, candidate identical** → no confound; the gate proceeds unchanged.
- **control passes, candidate differs** → real ORT difference; the gate is scored against
  the locally re-quantised **1.29.0** bytes, matching the host baseline, and the
  difference is disclosed.

**A RELATED FIELD WAS NULL IN EVERY RUN TO DATE.** The quantiser is `ORTQuantizer` —
optimum's front end over `onnxruntime.quantization` — and `train_env()` read
`optimum.__version__`, which does not exist (it lives at `optimum.version.__version__`).
The bare `except` wrote `null` for optimum every time while the field looked populated, so
the quantiser's front end is **unversioned in every artefact on disk**. Version capture now
goes through `importlib.metadata`, adds `onnx`, and records
`quantiser_versions_complete`. onnxruntime is pinned to **1.29.0** and asserted in
PREFLIGHT. **optimum is now pinned at 2.1.0** with `optimum-onnx` at 0.1.0 and `onnx` at 1.22.0,
recovered from the `rd-tier0-hostb` pip log — the versions were always in the log, just
never in an artefact. `quantiser_versions_complete` now covers all four.

### 3bc. DEPLOYMENT CONFIG — canary floor and router operating point (2026-09-11)

**THIS IS NOT AN EXPERIMENT AND HAS NO ACCEPT RULE.** Nothing here is tested, nothing
here can be confirmed or falsified, and no number below may be cited as a result. It is
registered because both values are thresholds chosen from measurements, and hard rule 6
exists so that a chosen threshold is written down with its basis rather than appearing
in a config file with no provenance. The deployed service is `src/serve/`
(**E4b-A**: Tier 0 INT8 → Claude Sonnet 5, no Tier 1).

---

> **⚠ ESCALATION IS NOW OFF — see §3bm (2026-09-12).** `tier2.enabled: false`. The router
> still runs and flags low-margin rows (`low_confidence` / `needs_review`); nothing is
> escalated, and a flagged row is **not** an `escalation_skipped` row. The threshold,
> spend cap and Tier 2 code are unchanged and re-enableable by config alone.

> **⚠ THE SERVED ARTEFACT AND ITS CANARY NUMBER HAVE CHANGED — see §3bk (2026-09-11).**
> Tier 0 now serves **`models/onnx_ce10ep_1_fp32`** (E1b seed 1), canary **191/200 =
> 0.9550** on macOS-arm64 *and* Linux-aarch64, router threshold **0.528864** at the same
> registered 4.056%. **The floor below stays 0.80** and is now 0.155 under measured rather
> than the ~0.10–0.115 this section constructs — held deliberately, recorded in §3bk. The
> INT8 numbers in this section are unchanged and still describe `models/int8_ce_1`.

#### 1. Tier 0 startup canary — floor **0.80**, from a measured **0.9000**

| | |
|---|---|
| measured | **180/200 = 0.9000** accuracy, arm64 (Apple Silicon), `models/int8_ce_1`, 2026-09-10 |
| row set | `configs/canary_200.json` — 200 TRAIN rows, seed 20260910 |
| floor | **0.80** |
| behaviour | `create_app()` raises `CanaryFailure`; the process does not start |

**Why a refusal rather than a warning.** Kaggle's non-VNNI x86 scored these same weights
at INT8 macro-F1 **0.0037** against FP32's 0.7636 — chance, **with no error raised
anywhere**. INT8 kernels are ISA-specific and degrade silently without AVX-512 VNNI. The
deployment target (HF Spaces) is x86, i.e. the same hardware class. A service that
started anyway would serve chance-level labels with plausible confidences on every
request, which is exactly the silent-degradation path hard rule 11 forbids.

**Why 0.80 and not something tighter.** The gap to the measured value is 0.10 absolute,
≈12× the largest FP32/INT8 delta this project had measured at the time (E3, max
|delta| 0.0083) — **SUPERSEDED 2026-09-11 (§3ax's class): E1b seed 1 measures |0.013668|,
so the correct multiple is ≈7.3×, not ≈12×. The floor is UNCHANGED and the reasoning still
holds** — 7.3× benign jitter is still far above it, and §3bi records the breach — so
benign cross-ISA numeric jitter cannot trip it; and it sits far above any degenerate
outcome, so the failure it exists to catch cannot pass it. The floor is deliberately
loose: it is a smoke test for a catastrophic mode, not a precision instrument.

> **THE 0.9000 IS NOT AN ACCURACY RESULT AND IS NOT COMPARABLE TO ANYTHING REPORTED.**
> These are TRAIN rows the model was fitted on. `test_3000` INT8 accuracy for the same
> artefact is **0.8563** (macro-F1 0.7582). The canary number is higher *because* it is
> contaminated by construction; that is acceptable for a fixed reference point and
> disqualifying for anything else.

The rows are drawn from TRAIN **excluding** `train_holdout_3000` and `exemplars_8`, so no
evaluation role is consumed: `test_3000` reports, `dev_2000` is thresholds only (hard
rule 1), `train_holdout_3000` selects checkpoints.

---

#### 2. Router operating point — percentile calibration at **4.056%**

| | |
|---|---|
| mode | **percentile**, not absolute (§3aq) |
| target escalation rate | **4.056%** |
| derived threshold | margin **0.115365** on `dev_2000`, seed 1, `models/int8_ce_1` |
| achieved on dev | 4.10% |
| source | `results/dev_logits_int8_local_ce_seed1.npz` (int8 / arm64_local) |

**Percentile, per §3aq.** Absolute dev-calibrated margin thresholds span **1.49×** across
E6's three seeds (0.1154 / 0.1177 / 0.1718), and that spread *causes* the 1.53× spread in
realized escalation rate. Margin is not on a comparable scale across retrainings, so an
absolute value does not transfer. **§3au narrows what this buys:** percentile selection
stabilises the **RATE ONLY** (set Jaccard 0.2370 → 0.2419, f = +0.0065). *Which* clauses
escalate is not stabilised, so no SLA, audit or reproducibility claim may rest on it.

**THE RATE FIXES THE BILL. IT IS NOT AN ACCURACY CLAIM.** §3av's registered verdict
(**E6-A**) stands unchanged: cascading adds **no measurable accuracy** — `test_3000`
macro-F1 delta **+0.0063**, 95% CI **[−0.0004, +0.0103]**, sign-consistency **2/3**;
under percentile selection **+0.0065**, 2/3. Routing works (margin drops accuracy ~0.85 →
~0.37 on its own selection, AUROC 0.86), but the escalation target has no marginal value
on precisely those rows. The operating point therefore buys a bounded, predictable API
bill and a defined worst-case path, and nothing else.

> **⚠ THE OPERATING POINT WAS CHOSEN AFTER SEEING TEST, AND THIS IS THE DISCLOSURE.**
> 4.056% is the **mean applied escalation rate across E6's three seeds on `test_3000`**
> ((120 + 97 + 148) / 3 ÷ 3000), and §3au's rank-based top-4.056% selection is built from
> the same figure. It was **not** derived on dev in advance. The *threshold value* is
> computed on `dev_2000` only, so hard rule 1 is intact for the threshold — but **the
> RATE those thresholds target came from test.**
>
> **Consequence, stated plainly: `test_3000` numbers at this operating point are NOT a
> held-out estimate for the deployed configuration.** Any cascade macro-F1, escalation
> rate or USD/1k figure read off `test_3000` at 4.056% is an in-sample figure for the
> rate that selected it, and must be reported as such. A held-out estimate for the
> deployed config would need a rate fixed without reference to test, or a fresh
> reporting split. Neither exists, and the number is not repaired by restating it.
>
> This is recorded rather than corrected because the alternative — silently picking a
> different rate now — would replace a disclosed dependence with an undisclosed one.

**A recalibration is REQUIRED whenever the weights change**, E1b's 10-epoch artefact
included: the percentile is transferable, the derived absolute value is not.
`configs/router_threshold.json` records the artefact it was calibrated against and
`ServiceConfig` refuses to start when the served weights differ.

---

#### 3. Live-serving spend

Escalation appends to `results/spend_ledger.jsonl` (hard rule 12) under run id
`serve_tier0_sonnet5`, separating serving spend from the offline runs. A configurable
cap (`tier2.spend_cap_usd`, shipped at **$1.00**, under the $15 hard stop) is checked
against cumulative recorded spend before every call; at the cap the service **refuses to
escalate** and returns the Tier 0 answer with `escalation_skipped=true`. A cache hit
spends nothing, so it is served without consulting the cap and writes no ledger entry.
**As of registration no live escalation has been made and the ledger contains no serving
entry.**

### 3bb. E1b runs as TWO BOUNDED COMMITS — and §3y's step-budget decision is reversed (2026-09-10)

**Does a mid-run split need registering? The split itself, NO. The mechanism, YES — and it
is a bigger change than the split.**

- **The split changes no registered parameter.** Same seed, same `EPOCHS`, same batch, same
  lr, same `max_length`, same manifests, same selection metric. Splitting one training run
  into two bounded ones alters neither the data order, the batch composition, the optimizer
  trajectory nor the number of steps — the checkpoint carries all of it. §3s established
  exactly this for Tier 1. **No new accept rule; §3ar's gate is untouched.**
- **But `kaggle_tier0.py` had NO mechanism to do it**, and §3y decided that deliberately:
  *"Tier 0 is 3.0-6.6h for all three seeds against a 9h design cap, so a budget would be
  machinery that never fires."* **That premise is now false.** §3ba measures E1b seed 1
  ALONE at ~8.1–8.6 h against a 9 h hard cap. §3y's decision is **reversed for E1b**, and
  the reversal is recorded here rather than made silently — a change to the training path is
  exactly what hard rule 6 exists to surface.

**THE TRAP THE OBVIOUS SPLIT WALKS INTO, and it is why `EPOCHS` must not move.** The
intuitive way to run "epochs 1–5 then 6–10" is to set `EPOCHS = 5` for commit 1.
**That would silently change the experiment.** `num_train_epochs` is what
`Trainer.create_scheduler` builds `num_training_steps` from, so `EPOCHS = 5` decays the
learning rate to zero over 17,815 steps instead of the true 35,630 — **a different LR
trajectory, reported by nothing.** The same argument forbids `max_steps`. §3s recorded this
for Tier 1 and verified on a CPU harness that the callback reproduces the uninterrupted
schedule exactly while `max_steps` does not.

> **`EPOCHS` stays 10 in EVERY commit of the chain.** Only `CommitStepBudget` stops early.

**EPOCHS = 10 CONFIRMED, and it is the hypothesis — not a cost knob.** 10 epochs = **35,630
steps**, which clears LexGLUE's ~30,000-step minimum; **6 epochs would be 21,378 and would
not**. E1b exists to test whether E1's shortfall is undertraining (§E1b: *one variable*), so
trading epochs for wall clock would answer a different question and is refused.

**`CommitStepBudget` ported from `kaggle_tier1.py`** with its reasoning intact, because the
reasoning is what makes it correct: class-level counter (a per-seed budget hands each new
seed a fresh full budget), and `should_save` set alongside `should_training_stop` so the stop
point is checkpointed rather than losing back to the last **epoch** boundary — which here is
3,563 steps, ~47 min at the measured rate. `RUN_STEP_BUDGET = None` is the default and a
**no-op**, so E1 and the 3-epoch arms keep exactly the behaviour §3y chose for them.

**A resume that does not advance is asserted, not trusted.** `RESUME_FROM_STEP_AT_LEAST`
raises if the chain resumes at or below the step the previous commit reported. A stalled
chain looks identical to a healthy resume in the logs, and that is the failure class this
project keeps paying for (§3s).

**THE PLAN.**

| | commit 1 | commit 2 |
|---|---|---|
| `E1B` | `True` | `True` |
| `HOST_BASELINE` | `False` | `False` |
| `EPOCHS` (derived) | **10** | **10** |
| `RUN_STEP_BUDGET` | **17815** | **None** |
| `RESUME_FROM_STEP_AT_LEAST` | **0** | **the step commit 1 prints** |
| notebook input attached | **none** | **commit 1's output** |
| trains | steps 1 → 17,815 (epochs 1–5) | 17,815 → 35,630 (epochs 6–10) |
| estimated | **~4.0–4.3 h** | **~4.0–4.6 h** (carries the eval/export/quantise tail) |

Both are **less than half the 9 h cap**, so the cap risk goes to zero at no extra GPU cost —
the same 35,630 steps are trained either way. The attach path is confirmed working: the host
baseline logged *"no notebook input attached → first commit for this arm, starting fresh."*

**TWO PRECONDITIONS VERIFIED AGAINST THE PINNED SOURCE BEFORE ARMING — the split rests on
both, and neither follows from the ordering already checked.**

**(1) The checkpoint at 17,815 IS written.** Read from `transformers==4.57.6`'s own
`trainer.py` (downloaded and inspected — the local venv is 5.0.0 and would have answered a
different question):

| line | statement |
|---|---|
| 2755 | `self.control = self.callback_handler.on_step_end(...)` ← our callback sets `should_save` **and** `should_training_stop` |
| 2756 | `self._maybe_log_save_evaluate(...)` |
| 3227 | `if self.control.should_save: self._save_checkpoint(model, trial)` |
| 2772 | `if ... should_training_stop: break` |

**`_maybe_log_save_evaluate` runs AFTER `on_step_end` and BEFORE the break**, and the save is
gated on the flag our callback just set. So the stop point is checkpointed in the same
iteration. **No off-by-one workaround is needed**, and `save_total_limit=2` cannot evict it
because it is the newest. Had the order been the other way, commit 2 would have resumed from
epoch 4 and silently trained 4+5 epochs = 9, not 10.

**(2) Commit 1 exits CLEANLY, with no exception.** The first implementation used
`raise SystemExit(0)`. **That was replaced**: Kaggle executes this file as a Script on some
paths and through IPython as a Notebook on others; `SystemExit` is a clean exit code 0 in the
first and can surface as an error in the second. Since the entire point of a bounded commit
is that **its output stays attachable**, an ambiguous exit is not acceptable. The callback now
sets a `_BUDGET_STOPPED` flag and `break`s, the script reaches its natural end, and
**AST-verified: no `SystemExit` is raised anywhere in the file.** The closing banner is
conditional — a bounded stop prints `COMMIT COMPLETE (BOUNDED)` with the resume values, and
explicitly states that **no fp32 model was saved and no completion marker written**, so the
next commit resumes rather than skipping.



**Quota:** 27.6 h remaining, E1b seed 1 ≈ 8.1–8.6 h across the two commits, leaving ~19 h —
enough for seeds 2–3 **only if §3ar's gate opens**, which is what the gate is for.

### 3ba. E1b RE-COST from measured rate — the registered estimate was too optimistic (2026-09-10)

**The wall clock was 2.38 h against a registered 1.0–2.2 h**, with throughput degrading
**1.575 → 1.255 it/s** across the run. **Effective rate = 10,689 steps / 2.38 h = 1.248
it/s.**

**The wall clock was NOT persisted by the notebook** — `tier0_ce_hostB_seed1.json` has no
timing field, which is §3e instance 9 recurring in the run whose whole purpose was to record
its environment. It is recorded here from the operator's observation, and **the notebook must
persist `seed_wall_clock_s` before E1b runs** (`kaggle_tier0_dev.py` already does; the main
notebook does not).

**E1b is 10 epochs = 35,630 steps.** Re-costed:

| basis | h / seed | 3 seeds |
|---|---|---|
| start-of-run 1.575 it/s | 6.28 | 18.85 |
| **effective measured 1.248 it/s** | **7.93** | **23.80** |
| end-of-run 1.255 it/s | 7.89 | 23.66 |

**The registered range was 3.3–7.3 h/seed. The measured basis gives 7.93 — OUTSIDE it.**
§3ak's estimate is superseded (§3ax's class: do not leave 3.3–7.3 standing as current).

**Against the constraints:** Kaggle's session cap is **9 h**; quota remaining **27.6 h**.

- **Seed 1 fits, but the margin is ~1 h and that is before overhead.** The 7.93 h is
  training steps only. E1b runs `eval_strategy="epoch"` — **10 evaluations of `sel_ds`
  instead of 3** — plus the constant tail (full-10k predict, ONNX export, INT8 quantise,
  three ONNX inference passes over ~16k rows). Realistic seed 1: **~8.1–8.6 h against a 9 h
  cap.**
- **3 seeds at 23.8 h against 27.6 h remaining leaves 3.8 h** — and would consume the quota
  E7 would need, which §3as already declined to schedule.

**IS A TIMEOUT RECOVERABLE? YES, with one real caveat.** `tr.train(resume_from_checkpoint=
resume)` exists, `_RESTORE_GLOBS` includes `ck_*`, and `save_strategy="epoch"` with
`save_total_limit=2` keeps the last two checkpoints. So a second commit resumes.
**The caveats, both load-bearing:**

1. **Checkpoints are per EPOCH, not per step** — at 3,563 steps/epoch (~47 min) a timeout
   loses back to the last epoch boundary, up to ~47 min.
2. **Resume depends on `/kaggle/working` surviving into the next session** (§3s already
   records this). If it does not, seed 1 restarts from step 0 and the 8+ h is spent again.

**RECOMMENDATION, given §3ar's gate is a spending decision:** run E1b seed 1 alone and let
the gate decide, which is what §3ar was designed for. At ~8.1–8.6 h it is the single largest
run in the project and it consumes ~30% of the remaining quota — but the gate's whole
function is to stop the other two seeds if seed 1 does not move. **Do not schedule 3 seeds up
front.**

### 3ay. THIRD SWEEP — the registered scorer is bypassed everywhere the numbers were actually produced (2026-09-10)

The 1-ULP discrepancy in `tier1_seed1.json` was **not a precision artefact — it was a
provenance signal**, and following it found that the bypass is **systemic, not isolated**.

**Every macro-F1 in the record that was produced by a training run came from
`sklearn.f1_score` directly, never from `src.eval.metrics.score`:**

| site | what it produced |
|---|---|
| `notebooks/kaggle_tier0.py:307,310` | E1's `tier0_ce_seed*.json` — selection, `test_3000_fp32`, `test_3000_int8`, `test_full10k` |
| `notebooks/kaggle_tier0_dev.py:123` | `tier0_dev_ce_seed*.json` |
| `notebooks/kaggle_tier1.py:816` | `tier1_seed1.json` — **where the 1 ULP surfaced** |
| `src/train/tier0_encoder.py:306,338` | the local training path's metrics |

The registered scorer is imported only by the **analysis** scripts written afterwards
(`score_fewshot`, `score_int8_local`, `build_frontier`, `cascade_paired_bootstrap`,
`score_cached`, `rung0_freetier`). **The measurement path and the reporting path use
different scorers**, and nothing compared them until a 1-ULP mismatch forced it.

**MEASURED DIVERGENCE, on every committed number recomputable from logits on disk:**

| number | committed (sklearn) | registered scorer | divergence |
|---|---|---|---|
| E1 seed 1 `test_3000` FP32 | 0.7635909027071933 | 0.7635909027071929 | **4 ULP** |
| E1 seed 2 `test_3000` FP32 | 0.7597399679922866 | 0.7597399679922866 | **exact** |
| E1 seed 3 `test_3000` FP32 | 0.7478116906220197 | 0.7478116906220192 | **4 ULP** |
| dev_2000 INT8 seed 1 / 2 / 3 | — | — | **1 / 1 / 3 ULP** |
| Tier 1 seed 1 `test_3000` | 0.7254043543132834 | 0.7254043543132833 | **1 ULP** |

**Cause identified, not assumed:** `sum(per_class)/len(per_class)` versus numpy's pairwise
`mean()`. Summation order, nothing else.

**THE CONSEQUENCE SO FAR IS NIL; THE UNGUARDED RISK IS NOT.** `classes_averaged` matched in
every case (100 on `test_3000`, 99 on `dev_2000`), so the two calls happened to average over
the **same label set** — which is why the divergence stayed at ULP scale. **That was luck,
not design.** `src.eval.metrics.score` exists precisely to refuse the case that would have
diverged materially: it **raises** unless `allow_absent_classes=True` when the label set
contains classes with no gold examples, because sklearn scores each such class 0.0 and
deflates macro-F1 in proportion to how many there are. **That guard was never in the path
that produced any committed number.** A future arm evaluated on a subset covering fewer
classes — exactly E8's 97-of-100 situation — would have diverged by far more than 4 ULP, and
**nothing would have announced it.**

**`tier1_seed1.json` is left with its sklearn-derived `macro_f1` UNCHANGED**, and a
`macro_f1_provenance` field recording that it is sklearn-derived and differs from the
registered scorer by exactly 1 ULP with the cause named. **Loosening the bit-exact guard to
absorb it would have destroyed the only evidence of the bypass** — the guard earned its
keep by refusing to write. `per_class_f1` was added *from the registered scorer* (a new
field, moving nothing committed): 100 classes averaged, **4 at F1 = 0.0** (`Assigns`,
`Books`, `Powers`, `Qualifications`), worst non-zero `Applicable Laws` 0.111.

**This is a THIRD defect class, distinct from §3aw and §3ax:** *the measurement path and the
reporting path use different implementations of the same registered metric, and nothing
compares them.* §3aw is a missing table, §3ax is a stale value; this is **two live
implementations of one definition**, where the guard the project wrote to protect the metric
sits in the path that never produces the numbers.

**VERIFIED 2026-09-10 — the gate's path is CLEAN, and this was checked rather than assumed.**
`kaggle_tier0.py:307,310` is a bypass site and it is the script running E1b, so the question
is whether the gate reads a bypassed number. It does not:

| quantity | produced by | scorer |
|---|---|---|
| **E1 = 0.7523**, the anchor every §3ar boundary derives from | `scripts/score_int8_local.py` → `results/int8_local_seed*.json` | **`src.eval.metrics.score`** (line 27/75) |
| **E1b's gate input** (§3as step 5) | `scripts/score_int8_local.py`, same script | **`src.eval.metrics.score`** |
| E8's numbers | `scripts/score_fewshot.py` (line 35/137/146) | **`src.eval.metrics.score`** |

**The anchor and the gate input come from the SAME implementation**, and both record
`classes_averaged`. Kaggle's own macro-F1 is never read by the gate. E8 likewise used the
registered scorer — verified in the code and in the artefact (`classes_averaged: 97`,
`absent_classes: []` on both arms), not inferred from "written afterwards".

**THE DIVERGENCE IS NOW BOUNDED, NOT HYPOTHESISED.** On E8's own 97-of-100 rows:

| scoring | macro-F1 |
|---|---|
| registered scorer, 97 gold classes | **0.6276** |
| sklearn default (`average="macro"`, no `labels=`) | **0.6088** |
| **deflation** | **−0.0188 (−3.00%)** |

The mechanism is exact: sklearn averages over **gold ∪ predicted**, so the moment a model
predicts a class with no gold examples, that class scores 0.0 and drags the mean down. On
`test_3000` this is impossible (gold covers all 100), which is the only reason the committed
numbers agreed to 4 ULP. **On any subset arm it is a 3%-scale error, and nothing would have
announced it.**

**MITIGATION IMPLEMENTED — `tests/test_scorer_provenance.py`, 5 tests.** The first proposal
("assert sklearn and the registered scorer agree on the committed artefacts") was
**rejected**: it is green today *only because* `classes_averaged` happened to match
everywhere, and it would stay green until a subset arm appeared — which is exactly when it
would be needed. **A test that cannot fail on the defect it guards is the proxy-check shape
again (§3e).** What is committed instead:

1. `test_subset_case_must_diverge` — constructs the case where the two implementations
   **must** disagree and asserts the divergence is **large (> 0.05), not ULP scale**.
   Mutation-checked: if `score()` were reimplemented on sklearn's default the difference
   becomes 0.0000 and **the test goes red** (measured: registered 0.9167 vs sklearn 0.7333,
   diff +0.1833).
2. `test_scorer_refuses_absent_classes` — the guard **raises**, and averages absent classes
   in only under an explicit `allow_absent_classes=True`.
3. `test_e8_subset_divergence_is_real_and_recorded` — pins E8's 97-class label set.
4. `test_committed_artefacts_record_classes_averaged` — **every** artefact carrying a
   `macro_f1` must record the label set it averaged over, with a named `GRANDFATHERED` list
   (the 6 Kaggle seed JSONs, 2 rung-0 files, and the frontier) so the debt is **visible and
   may only shrink**.
5. `test_grandfathered_list_only_shrinks` — a stale entry would hide a real gap.

**BYPASS CLOSED 2026-09-10, before E1b was armed — and it had to be, because the rule and
the run collided.** E1b runs the same unmodified `kaggle_tier0.py`, so it would have emitted
new seed JSONs carrying a `macro_f1` with no `classes_averaged`. Then **either**
`test_committed_artefacts_record_classes_averaged` fails, **or** the files get added to
`GRANDFATHERED` — which `test_grandfathered_list_only_shrinks` forbids. **One of the two
breaks on E1b's first write.** A guard that forces a choice between failing and being
weakened is not yet a guard.

**The notebooks are STANDALONE** — they run on Kaggle where the repo is absent, so
`from src.eval.metrics import score` is not available. The fix is therefore a **port**, and a
port that is never compared to its original is §3ay's defect again in a new place. So:

- `notebooks/kaggle_tier0.py` and `kaggle_tier0_dev.py` now define `_macro_report`, a port of
  `src.eval.metrics.score` between `# --- BEGIN PORT` / `# --- END PORT` sentinels. It
  averages over an explicit label set, **raises** on absent classes, and returns
  `classes_averaged`, `classes_in_gold`, `classes_predicted`, `absent_classes` and
  `per_class_f1`.
- Every reported metric block is now the **full report**. `macro_f1` and `accuracy` keep
  their names, so `scripts/score_int8_local.py` and the E3 discriminator read unchanged.
- `dev_2000` gains a full `dev_2000_fp32` report — **it is the split where the deflation is
  actually reachable**, covering 99 of 100 classes.
- **`test_notebook_port_matches_registered_scorer`** extracts the real shipped block from
  each notebook by its sentinels, `exec`s it, and asserts agreement with
  `src.eval.metrics.score` on four cases **including the divergence case**, that it inherits
  the raising guard, and that it does **not** reproduce sklearn's deflation.
- **`test_notebooks_no_longer_call_sklearn_directly_for_reported_metrics`** fails if
  `def macro(...): return f1_score(` reappears.

**`e6_frontier.json` came OFF the grandfathered list the same day it went on** — regenerated
under the exact-reproduction guard, which passed bit-exactly, now recording
`classes_averaged: 100` and `scorer: src.eval.metrics.score`. The regeneration that added
per-seed arrays was the free moment to add this and it was missed; adding it separately cost
one more run. **The list shrank, which is the only direction its test permits.**

Suite: **481 passed, 1 skipped.** Remaining grandfathered: the 6 Kaggle seed JSONs already on
disk and 2 rung-0 files — **historical artefacts that cannot be reissued**, not future ones.
`notebooks/kaggle_tier1.py` still calls sklearn directly and is **not** closed; it is
recorded here rather than fixed, because no Tier 1 run remains (E4-A/E4b-A) and editing a
notebook nothing will execute would be change without verification.

### 3ax. SECOND DEFECT CLASS — "superseded figure still presented as current" (2026-09-10)

§3ax's sweep found the 28.97 throughput defect **incidentally**, while looking for something
else. That is the tell: **no targeted check existed**, so a separate sweep was run. It found
**five more**. This is a different failure from §3aw — there the number was correct and the
structure was missing; here **the number itself has been replaced and the replacement did not
propagate.**

**THE CLASS.** *A figure that a §3e instance or a later measurement replaced, still stated
elsewhere in the record without a supersession marker — so the report asserts two
contradictory values and a reader cannot tell which is live.*

Why it evades everything already registered: hard rule 7 says negative and rejected results
**stay** in the report, which is right — but "stays" was silently read as "stays *unmarked at
every site*". Hard rule 5 routes costs through `configs/costs.yaml`, and **`costs.yaml` was
correct in every case below** — the stale values live only in the prose. The artefacts and
the narrative drifted apart with nothing comparing them.

| # | figure stated as current | live value | where | impact |
|---|---|---|---|---|
| **1** | **28.97 ± 0.24 req/s** (n=4, untrained probe) | **28.3615 ± 1.1839** (n=9, trained) | §1 L75, §1b L123 | **FIXED 2026-09-10** — both sites now carry the live figure, the strike-through and the per-run table |
| **2** | **30.23 req/s / 18.20 W / 17.88 W marginal / 0.591 J/req**, headed "measured" | **28.3615 / 18.5108 / 17.7546 / 0.6269** | §3aa's own table (~L2440) | **the §3ac marker at L2360 covers §3ac's sensitivity table, NOT this one.** Four figures, all superseded, all unmarked |
| **3** | *"E6's numbers are unchanged — 0.591 J/req, $1.390e-5 per 1k, V\* = 1,413,969"* | energy 0.6269; V\* 1,413,971 per §3ac | §3ab close (~L2431) | asserted as live in a sentence whose point is that nothing changed |
| **4** | `measured_throughput_rps` **(30.23)**, `power_draw_soc_watts` **(18.20)**, `energy_joules_per_request` **(0.591)** described as *"now measured and in `configs/costs.yaml`"* | costs.yaml holds 28.3615 / 17.7546 / 0.6269 | E6 entry L479–482 | **the cited file disagrees with the citation.** `power_draw_soc_watts` is not even a field any more — §3ab split it into `_idle`/`_load`/`_marginal` |
| **5** | **`V_max` = 714,999,960 clauses at 30.23 rps** | **670,806,198** at 28.3615 rps (**−6.2%**) | §1b L139 | conclusion unchanged (still 506× → 474× V\*), number stale |
| **6** | **Sonnet 5 $0.4483 per 1,000**, used live in `V* = 633,862.43 / (0.4483 − e)` | **$0.43610** — `e6_frontier.json`'s `api_usd_per_1k`, from actual Stage 1 usage | §1 L114, §3aa, §3ac | **FIXED 2026-09-10.** 0.4483 was a PROJECTION made before Stage 1 ran — a stale **input to a live formula**, not merely a stale value. **V\* moves 1,413,925 → 1,453,465 (+2.80%)**; no conclusion changes (still ~18× the corpus) |

**Item 6 is the one that matters most, and it is the subtlest.** Every other entry is a
measurement superseded by a better measurement of the same thing. Item 6 is a **projection
that was never replaced by the measurement it was projecting** — Stage 1 ran, the actual
per-1k cost is on disk in `e6_frontier.json`, and the V\* arithmetic still divides by the
forecast. **No conclusion changes** (V\* remains ~1.45M, still ~18× the entire 80,000-clause
LEDGAR corpus), but the headline break-even number is 2.8% stale and its input is labelled
"measured" when it is projected.

**None of these change a conclusion.** All six are reporting defects. That is precisely why
they survived: **nothing that depended on them broke.**

**MITIGATION, registered:** when a figure is superseded, the supersession is recorded **at
every site that states it**, not only at the site that discovered it — and the live value is
carried with the strike-through so the two are legible together. A `grep` for the old value
is the check, and it is cheap. **Items 1 and 6 are FIXED; items 2–5 are REPORTED and not yet fixed.** Item 6 was
promoted out of the report-only set because it is a stale *input to a live formula* rather
than a stale value — the formula was still dividing by a forecast whose outcome had already
been measured.

### 3aw. DEFECT CLASS — "aggregate hides structure", with a registered mitigation (2026-09-10)

Three occurrences, and **every one was caught by review, not by the code.** That is what
makes it a class rather than three mistakes: the failure has a stable shape, it survives
scrutiny of the numbers themselves (each aggregate was *arithmetically correct*), and
nothing in the pipeline objects.

| # | where | the aggregate | the structure it hid |
|---|---|---|---|
| 1 | §3an | E6 delta **+0.0063**, one interval | per-seed **−0.0015 / +0.0049 / +0.0153** — opposite signs, sd 0.0085 **exceeding the mean** |
| 2 | §3ap(D) | first-1000 vs rest **+0.0157** | scored over **different label spaces** (97 vs 98) — a different estimand, not a different number |
| 3 | §3au | Jaccard moved **+0.005** | per-pair **+0.058 / −0.050 / +0.007** — opposite signs, **10× the aggregate's size** |

**THE CLASS.** *A mean, pooled statistic or single interval reported without the per-unit
structure underneath it, where that structure carries opposite signs, unequal magnitudes,
or differing estimands.*

Why it evades the existing rules: hard rule 2 requires **mean ± std over ≥ 3 seeds**, and
**all three defects satisfied it.** A standard deviation is itself an aggregate — it reports
*spread* but not *sign*, so `+0.0063 ± 0.0085` is fully rule-2-compliant and still conceals
that one seed went the other way. Hard rule 11 forbids silent degradation of a *measurement*;
this degrades a *report* of a correct measurement. The gap was real.

**THE MITIGATION, REGISTERED:**

> **Any reported aggregate over seeds, pairs, splits or arms must be accompanied by the
> per-unit table it summarises — in the same artefact AND the same prose.** If the per-unit
> values differ in sign, the aggregate **may not be stated without that fact adjacent to
> it.**

**Scope, so it is enforceable rather than aspirational.**

- **"Same artefact"** means the JSON/npz carries the per-unit values, not only the summary.
  A per-unit table that exists solely in terminal scrollback is §3e instance 9 (printing is
  not recording); one that exists solely in the artefact is §3au's chance-floor error
  (computing is not reporting). **Both places, or it does not count.**
- **"Sign" includes the null.** A per-unit set spanning zero must say so even when every
  value is nominally positive — §3ap(B)'s escalated-row macro-F1 is 3/3 positive with all
  three CIs including zero, and is reported that way.
- **Bootstrap intervals are aggregates too.** A single CI over pooled units hides per-unit
  structure exactly as a mean does; §3an's row-level p = 0.0756 carried no seed-level
  information and was reported alongside the 2/3 sign count, not instead of it.
- **Where n = 1 there is no per-unit table**, and the mitigation is satisfied by saying
  n = 1 explicitly (E4-A, §3at's host baseline).

**This rule is retroactive as a REPORTING obligation, not as a re-measurement obligation.**
No number below changes; what changes is whether it may be stated alone. The sweep against
this rule is §3ax.

### 3av. RESTATEMENTS adopted for H1, E4, E4b and E6 (2026-09-10)

Adopted after E8 (few-shot does not close the gap), §3an/§3ap (E6's per-seed structure) and
§3au (percentile stabilises rate, not set). **E1 is deliberately NOT restated** — the host
baseline and E1b are in flight and E1's wording depends on their outcome.

**Hard rule 7 governs all four: every falsified version is preserved verbatim alongside its
replacement. Nothing below deletes a rule.**

---

#### H1 → **H1-A**

**FIRST, THE DISCLOSURE, BECAUSE THE ORDER MATTERS.** H1 as registered reads: *"A 3-tier
cascade reaches macro-F1 within 0.04 of Sonnet-5-alone on `test_3000` at under half the
USD/1k-clause cost."* **H1 is SATISFIED — trivially, and in the wrong direction.** Tier 0
alone is **0.137 ABOVE** Sonnet-5-alone, so "within 0.04" is met by a cascade that never
escalates at all, at ~1/1,700th the cost. H1 presupposed Sonnet as the accuracy **ceiling**;
it is the **floor**. A rule that is satisfied by the opposite of the mechanism it was written
to test has not been confirmed — it has been bypassed. **H1-A below is written after seeing
that, and is therefore post-hoc; this paragraph is the mitigation, and it appears first for
that reason.**

> **H1-A.** *A fine-tuned Tier 0 encoder matches or exceeds zero-shot Sonnet-5 on
> `test_3000` macro-F1 at under 1% of its cost; cascading to a frontier model adds no
> measurable accuracy.*

**Argument.** Preserves both of H1's clauses — a joint accuracy-and-cost claim, with the
0.04 tolerance still meaningful — and flips only the direction the evidence flipped. It is
testable and already tested (E6, E8, §3ap).

**Survives untouched from the original registration:** the *form* of the hypothesis; the
0.04 tolerance; the entire cost side; and H1's "how it could be wrong" column, which
anticipated *escalation driven by rare classes where Tier 0 is weakest* — **the wrong
failure mode.** H1 failed by a route it did not consider, which is worth more than a
correct guess.

---

#### E4 → **E4-A** (no restatement)

**E4's rule is not restated, because it fired correctly.** On the corrected bar of
`E1 + 0.04 = 0.7923` (E1 INT8 0.7523, §3ar), Tier 1 reached **0.7254** — a **0.0669** miss,
**12× the `test_3000` σ** of 0.0056.

> **E4-A.** E4 is reported as **NOT ACCEPTED at n = 1**, with the hard-rule-2 shortfall
> stated as an explicit limitation: *n = 1, no seed variance measured. Tier 1's own seed sd
> would have to be ~11× Tier 0's for two further seeds to bridge 0.0669. That is an
> argument, not a measurement.*

**Argument.** Seeds 2–3 cannot change a verdict at 12σ; they would satisfy a reporting
requirement, not a scientific one, at ~28 h of a 30 h quota that E1b needs.

**Survives untouched:** all of it. **Explicitly rejected:** retro-fitting E4's bar to what
Tier 1 reached — that is fitting a rule to its answer, which hard rule 6 exists to prevent.

---

#### E4b → **E4b-A** (superseded by measurement)

> **E4b-A.** E4b's configuration **has already been measured.** The E6 frontier **is**
> Tier 0 → Sonnet-5 with no middle tier. E4b's result is E6's result; **no separate run is
> required.**

**Argument.** E4b was registered assuming the open question was *removing Tier 1*. §3aq
shows the open question is whether **Tier 2 adds anything at all** — and E6 answers that for
exactly this two-tier configuration. Re-running it would spend budget to re-derive a number
on disk.

**THE CONSEQUENCE, STATED PLAINLY BECAUSE IT IS EASY TO MISS:** with E4b superseded by E6's
existing measurement and E4 not accepted at n = 1, **no Tier 1 experiment remains in the
plan. Tier 1 stands as a pure negative result at n = 1** — trained, evaluated, below bar,
with no successor experiment and no further quota allocated. That is the honest end state,
and it is reported as such rather than left looking like unfinished work.

**Survives untouched:** E4b's original text, and the fact that it was **registered in
advance of E4's outcome** — preregistration working as intended, even though the experiment
it authorised is now answered by other means.

**Considered and rejected — E4b-B, re-scoping to Haiku-4.5** (0.7210) as the escalation
target: Haiku is *worse* than Sonnet, which is already worse than Tier 0. §3aq predicts no
headroom, so this would spend budget to confirm a stronger form of a settled result.

---

#### E6 → **E6-A**

> **E6-A.** **Routing works; the escalation target has no marginal value on the rows routing
> correctly identifies as hard.** The margin signal drops accuracy from ~0.85 to ~0.37 on
> its own selection (AUROC 0.86, E5), and Sonnet-5 does not beat Tier 0 on those rows by any
> amount the evidence supports. **A frontier model adds nothing on exactly the examples a
> fine-tuned encoder finds hard.**

**THE REGISTERED VERDICT IS UNCHANGED.**

> `test_3000` macro-F1 delta **+0.0063**, 95% CI **[−0.0004, +0.0103]**, sign-consistency
> **2/3**. Under percentile selection: **+0.0065**, 2/3 (§3au). **The cascade is not
> established as better than Tier 0 alone.**
>
> **§3ap(B)'s escalated-row result does NOT revive it.** That macro-F1 is 3/3 positive
> (+0.0545 / +0.0495 / +0.0807) with **all three CIs including zero**, computed over the
> escalated rows' *own* 44–59-class label space. It is **not a decomposition of the
> +0.0063** and does not transfer to it. It says where value *would* sit if any existed; it
> does not establish that any does.

**Survives untouched:** **condition 1** (within 0.04 macro-F1 of Sonnet-5-alone) is **MET** —
by Tier 0 with no cascade at all, and is marked *met trivially*. **Conditions 2 and 3**
(finite crossover V\* ≤ V_max; asymptote < 50% of Sonnet-5) are marked
**premise-falsified**, not wrong: their arithmetic was never tested, because both assume
escalation buys accuracy and that assumption is what failed. V\* is still *reported, not
tested*, as registered.

**Also preserved:** §3an, §3ap and §3au in full, **including the corrections that reversed
my own claims** — the accuracy-for-macro-F1 substitution (§3e instance 10), the withdrawn
"by construction" variance mechanism, and the chance-floor overreach. Those are part of the
result, not scaffolding to be removed once the conclusion is tidy.

### 3au. Does PERCENTILE calibration stabilise the escalation SET, or only the RATE? — registered before the run (2026-09-10)

§3aq recommended calibrating the router at a **percentile** of the signal distribution
rather than at an absolute threshold value, because the dev-calibrated thresholds are
0.1154 / 0.1177 / 0.1718 (1.49×) and margin is therefore not on a comparable scale across
retrainings. **That recommendation is currently untested**, and §3an's overlap measurement
cannot test it: it was taken under **absolute** thresholds, so its Jaccard of 0.213–0.258
**confounds two things** —

| cause of non-overlap | does percentile calibration fix it? |
|---|---|
| the thresholds sit at different **scales** | **yes, by construction** |
| the margin **ranking** of rows differs between seeds | **no** |

**The test.** Select the top **4.056%** by margin **per seed** (the same target rate, but
rank-based, so scale is removed by construction), then measure pairwise Jaccard, all-three
overlap, distinct rows covered, and the per-seed macro-F1 delta under that selection.

**Reference points, both computed before the run:**

| quantity | value |
|---|---|
| J under **absolute** thresholds (§3an), mean of the three pairs | **0.2370** |
| J under **chance** — two 122-row sets drawn at random from 3,000 | **0.0208** |

**The chance floor is REPORTED, not merely computed.** It is printed by
`scripts/percentile_vs_absolute.py` and appears in the result table below, because a floor
that exists only inside an artefact cannot discipline the prose that cites it.

So the absolute-threshold sets already overlap ~11× chance; the question is how much of the
remaining gap to 1.0 is scale rather than ranking.

**Accept band, mechanical, using this project's registered `fraction_closed` construction
(E8, §3ar's midpoint logic):** `f = (J_pct − 0.2370) / (1 − 0.2370)`.

| J_pct | f | disposition |
|---|---|---|
| **≥ 0.6185** | ≥ 0.50 | **STRONG SUPPORT.** Percentile calibration substantially stabilises the escalation **SET**. §3aq's recommendation stands as worded — it stabilises rate *and* set |
| **0.3896 – 0.6184** | 0.20–0.50 | **PARTIAL.** Set stability improves but is not achieved. The recommendation must be reworded to claim rate stabilisation, with set stability stated as improved-not-solved, and quantified |
| **< 0.3896** | < 0.20 | **WEAK / NULL.** Non-overlap is dominated by **ranking disagreement**, which percentile calibration does not touch. The recommendation must be reworded to say **only** that it stabilises the escalation **RATE** — and that is a weaker, different claim than the one §3aq currently makes |

**Registered in advance, because it is the outcome I expect to have to write.** If the
result lands in the WEAK band, §3aq's deployment recommendation is **narrowed, not
withdrawn**: holding the escalation rate fixed across retrainings is still worth having —
it is what makes the API bill predictable, which was the stated motivation — but the claim
that the *same clauses* get escalated would be unsupported, and any SLA or audit argument
resting on set stability would have to go.

**A high J_pct would also be a finding about the SIGNAL, not just the calibration:** it
would mean margin's *ranking* is reproducible across retrainings even though its *scale* is
not, which is the property a percentile threshold needs and the property §3an's absolute
measurement could not see.

**Cost: $0.00, no GPU.** All inputs are on disk.

**RESULT (2026-09-10) — WEAK / NULL, and decisively so.**

| seed | k | rate | tier0 | cascade | delta |
|---|---|---|---|---|---|
| 1 | 122 | 4.067% | 0.7582 | 0.7568 | −0.0014 |
| 2 | 122 | 4.067% | 0.7514 | 0.7568 | +0.0054 |
| 3 | 122 | 4.067% | 0.7471 | 0.7628 | +0.0157 |

Mean delta **+0.0065**, sd 0.0086, **sign-consistency 2/3** — materially identical to the
absolute-threshold result (+0.0063, 2/3), so **nothing about E6's verdict changes.**

| pair | absolute | **percentile** | move |
|---|---|---|---|
| 1 & 2 | 0.240 | **0.298** | **+0.058** |
| 1 & 3 | 0.258 | **0.208** | **−0.050** |
| 2 & 3 | 0.213 | **0.220** | +0.007 |
| **mean** | 0.2370 | **0.2419** | +0.0049 |
| **chance floor** | 0.0208 | 0.0208 | — |
| distinct rows | 250 | **249** (157 + 67 + 25) | |
| in all 3 seeds | 25 (10.0%) | **25 (10.0%)** | |

**THE MEAN IS STABLE; THE PAIRS ARE NOT.** "Percentile selection moved Jaccard by 0.005" is
true of **no individual pair**: 1&2 rose 0.058, 1&3 *fell* 0.050, 2&3 barely moved. The
aggregate concealed **opposite-sign movement ten times its own size.** This is the third
appearance of the same shape in this project — §3an (E6's aggregate delta hiding per-seed
−0.0015/+0.0049/+0.0153) and §3ap(D) (a mean concealing which label space it used) are the
prior two. **The conclusion is unchanged and now rests on the per-pair table rather than the
mean: no pair approaches high overlap** — the maximum is 0.298, and the worst got worse.

**`f = (0.2419 − 0.2370) / (1 − 0.2370) = +0.0065`** — 0.65% of the achievable improvement.
The registered band puts this in **WEAK / NULL** with room to spare; it is not a near-miss.

**What this means, in the wording the band requires.** The 1.49× spread in threshold
*scale* contributed **almost nothing** to escalation-set instability. Removing it by
construction moved the mean Jaccard by 0.005 and left the all-three core at exactly 25 rows.
**The non-overlap is ranking disagreement, essentially all of it.**

**§3aq's deployment recommendation is NARROWED, not withdrawn** — as registered in advance:

> Calibrate at a percentile, **because it holds the escalation RATE — and therefore the API
> bill — fixed across retrainings.** That is its whole benefit. It does **NOT** stabilise
> *which clauses* are escalated: under percentile selection the escalation sets still
> overlap only ~24% pairwise, with 10% common to all three seeds. **Any SLA, audit or
> reproducibility argument that depends on the same clauses being escalated is
> unsupported.**

**And the finding about the SIGNAL is the inverse of the one §3au said a high J would give.**
Margin's *scale* is not reproducible across retrainings (§3an) **and neither is its
ranking** — at least not in the tail that matters. Only 10% of escalated rows are common to
all three seeds under either calibration.

**DIFFICULTY IS PARTLY INTRINSIC AND PARTLY MODEL-SPECIFIC — both, and the chance floor is
what settles it.** An earlier draft of this entry said difficulty is *"substantially
model-specific rather than intrinsic to the clause"*, which **contradicts the floor computed
three lines above it**: the chance Jaccard for two 122-row sets drawn at random from 3,000 is
**0.0208**, and the observed mean is **0.2419 — 11.6× chance**. Three independently
retrained routers agreeing on which clauses are hard at 11.6× the random rate is a **large**
effect, not a residue. Computing a floor and then arguing past it is §3e instance 9's shape
in the analysis rather than the artefact — so the floor is now **printed by the script and
carried in this table**, not left in the JSON.

**The supported statement is both at once:**

> Difficulty is **partly intrinsic to the clause** — independently retrained routers select
> overlapping rows at **11.6× chance** — and **partly model-specific** — that overlap is
> **0.24, far from 1.0**, with only 10% of escalated rows common to all three seeds.

**This does NOT contradict §3aq's "routing works".** Each seed's router correctly identifies
rows that are hard *for that model* (0.85 → ~0.37 on its own selection, AUROC 0.86); the
clauses those sets share are hard for the task, and the ones they do not share are hard for
that particular retraining. **The claim that the non-overlap is essentially all ranking
disagreement stands unchanged** — that is about the *cause* of the gap between 0.24 and 1.0,
and is unaffected by where 0.24 sits relative to chance.


### 3at. HOST BASELINE — interpretation rule, registered BEFORE the run is spent (2026-09-10)

The host-baseline run (§3ak step 1: E1's config, `EPOCHS = 3`, `RUN_TAG = "ce_hostB"`,
seed 1, new host) had **no registered interpretation rule** — the same hard-rule-6 gap that
was blocking E1b, in the run scheduled to go first. Registered now.

**WHICH COMPARISON IS THE SCIENTIFIC ONE — and it is not the one the name suggests.**

| comparison | status | why |
|---|---|---|
| **E1b (10 ep) vs host baseline (3 ep), both on THIS host** | **THE SCIENTIFIC COMPARISON** | one variable (epochs), **both environments persisted** via `train_env()`. This is what step 1 exists to make possible |
| host baseline vs **E1 seed 1** (0.7582) | **descriptive only — detectable, not attributable** | E1's training environment is **permanently unrecorded** (§3e instance 9). A difference can be *measured* but cannot be decomposed into hardware / driver / TF32 / library / setup |
| seed-1 **gate** vs E1 = 0.7523 (§3ar) | **unchanged, and NOT scientific** | the gate is a **spending decision** about whether to buy two more seeds. It is not an epochs finding and must never be reported as one |

**Those three must not be conflated later.** The gate keeps its registered anchor
(E1 = 0.7523, §3ar) even though the scientific epochs comparison uses the host baseline —
because the gate answers *"is 0.80 still reachable, so is more quota worth spending?"*,
which is a question about the accept target, not about epochs.

**Interpretation rule for `d = hostB_seed1 − E1_seed1`**, both `test_3000` INT8 arm64,
same seed, same config, different host. E1 seed 1 = **0.7582**.

| outcome | reading |
|---|---|
| **\|d\| < 0.0112** (2σ) | **UNINTERPRETABLE AT n = 1.** No host effect detected *at the only scale available*. This does **NOT** establish the hosts are equivalent |
| **\|d\| ≥ 0.0112** | **A host difference is DETECTED and CANNOT BE ATTRIBUTED.** Report the magnitude and sign; refuse the cause. E1's side has no environment record to compare against |

**Why the scale is a proxy, stated so it is not mistaken for a test.** σ = 0.0056 is the
**within-host across-SEED** sd of `test_3000` INT8. The question here is **cross-host at
fixed seed**, whose variance is **unmeasured**, with no reason to assume it is smaller
(§3ak says exactly this). 2σ is therefore a **borrowed yardstick**, used because it is the
only measured scale on this quantity — the same borrowing §3an refused for C1's 3pp, and it
is permitted here **only because the conclusion in every branch is "not attributable"**, so
the yardstick cannot change a causal claim. It bounds *noticing*, not *concluding*.

**In BOTH branches, E1b proceeds unaffected.** The epochs comparison lives entirely on the
recorded host, so nothing about `d` gates it. A large `d` makes E1's original numbers less
transferable; it does not make E1b's question harder.

**The epochs comparison itself is n=1 vs n=1 at first, and is NOT the answer.** A delta
between host baseline (1 seed) and E1b seed 1 (1 seed) below 2σ is uninterpretable for the
same reason `d` is. It becomes interpretable only when E1b has its 3 seeds — and E1b's
**accept rule is absolute** (≥ 0.80 on the 3-seed mean) and never became a comparison.

**Cost of being wrong about this rule: ~~1.0–2.2 h~~ → 2.38 h MEASURED (§3ba).** Registered before spending it.

### 3as. E1b's execution loop: Kaggle trains FP32, arm64 scores INT8 (2026-09-10)

**The gate is stated on INT8 and Kaggle cannot produce an INT8 number.** Measured, not
assumed: E1 seed 1's `test_3000_int8` on Kaggle's Xeon read macro-F1 **0.0037** (accuracy
0.0173) against FP32 0.7636 — chance, on a non-VNNI CPU. Any INT8 figure written in the
notebook is invalid and must not be recorded.

**Six steps. Step 3 is manual and blocking.**

| # | where | action |
|---|---|---|
| 1 | Kaggle | Train FP32, `EPOCHS = 10`, arm tag **`ce10ep`** (E1b's collision guard). `train_env()` **PERSISTED** into the seed JSON *and* the npz — §3e instance 9: printing is not recording |
| 2 | Kaggle | The INT8 figure computed there is the **E3 discriminator's diagnostic ONLY** and may never be the gate's input — see the correction below. Save `fp32_ce10ep_1/` as notebook output |
| 3 | **manual** | **DOWNLOAD** `fp32_ce10ep_1/` → `~/Downloads/tier0_e1b/`. Verify `model.safetensors` ≈ 738 MB, plus `config.json`, `spm.model`, `tokenizer.json`, `training_args.bin`. **Nothing downstream runs until this completes** |
| 4 | local arm64 | Export ONNX + dynamic quantise → `models/int8_ce10ep_1/` |
| 5 | local arm64 | Score `test_3000` via `scripts/score_int8_local.py` |
| 6 | local | Evaluate §3ar's gate **on step 5's number only** |

**CORRECTION to step 2, made before the run rather than after.** This entry first read
*"Do NOT compute or write any INT8 number"* on Kaggle. That would have **deleted working
instrumentation**: `kaggle_tier0.py` computes INT8 there deliberately, as **E3's
three-arm discriminator** (torch-FP32 → ONNX-FP32 isolates the export; ONNX-FP32 → INT8
isolates the quantised kernel), which is how the Xeon INT8 collapse was localised to a
non-VNNI kernel rather than to the serving path. The code already labels it diagnostic and
already says the number must be re-scored on arm64 before being believed either way. **The
rule is therefore about which number the GATE reads, not about what may be computed:**
Kaggle's INT8 figure is a diagnostic, the gate reads step 5's arm64 figure, and no INT8
number produced on a non-VNNI host may be reported as an accuracy.

The §3ak host-baseline run (`EPOCHS = 3`, tag `ce_hostB`) follows the identical six steps.

**Option A is the registered plan** — host baseline seed 1 + E1b seed 1, gate, then seeds
2–3: gate cost **4.3–9.5 h**, full path **10.9–24.1 h** against the 30 h quota. §3ak's
~~3.3–7.3 h~~ is **per seed**, not a 3-seed total — and is **superseded by §3ba's measured ~8.1–8.6 h/seed**.

**E7 is NOT scheduled.** It is a latency ablation, and E1b tests the number the whole report
rests on. E7 holds a **conditional slot** if quota survives Option A. It is a **3-seed
training job**: E7 says "paired, same rows", deliberately omitting E3's "same model", and
its `train rows truncated` column and `[C3-gated]` tag cohere only on that reading — so it
is not cheap and does not fit alongside a full Option A.

### 3aq. THE ACTUAL FINDING: routing works; the escalation target has no marginal value (2026-09-09)

Everything above was framed as "the cascade premise is falsified", and **that framing is
wrong** — it buries the result and makes E5 and E6 look like they conflict. They do not.

**The router is working, and working well.** Base accuracy on `test_3000` is ~0.85. On the
rows the router selects:

| seed | Tier 0 accuracy on escalated rows | API accuracy on the same rows |
|---|---|---|
| 1 | 0.392 | 0.400 |
| 2 | 0.340 | 0.433 |
| 3 | 0.338 | 0.351 |

**0.85 → ~0.37.** The margin signal identifies genuinely hard rows, exactly as E5's AUROC
of 0.86 said it would. **E5 and E6 are one coherent finding, not two conflicting ones.**

**The negative result is therefore not "cascading doesn't work". It is:**

> **Routing works. The escalation target has no marginal value on precisely the rows
> routing correctly identifies as hard.**

That is the stronger claim and the more interesting one: **a frontier model adds nothing
on exactly the examples a fine-tuned encoder finds hard.** The cascade is not badly
engineered and the router is not miscalibrated — there is simply **no accuracy headroom at
the top of the cascade to route into.** The write-up is built around this, and does not
apologise for it.

**Supporting it — and this is a stronger statement than the escalation-rate spread.** The
per-seed dev-calibrated **thresholds** are **0.1154 / 0.1177 / 0.1718**, a **1.49× spread**
— and *that* is the cause of the 1.53× spread in realized escalation rate, not a separate
fact about it. **The margin signal is not on a comparable scale across retrainings.** An
**absolute** dev-calibrated threshold therefore does not transfer between seeds of the
same model on the same data.

> **Deployment recommendation, which follows directly: calibrate the router at a
> PERCENTILE of the signal distribution, not at an absolute threshold value.** A
> percentile is invariant to the monotone rescaling that retraining applies to margin, and
> would hold the escalation rate — and therefore the API bill — fixed across retrainings.
> This is the one actionable engineering result the E6 work produces.

**⚠ NARROWED BY §3au (2026-09-10), which TESTED this recommendation.** Percentile selection
was measured, and it stabilises the **RATE ONLY**. Mean escalation-set Jaccard moves from
0.2370 (absolute) to **0.2419** (percentile) — `f = +0.0065`, 0.65% of the achievable
improvement — with the all-three-seed core unchanged at 25 rows (10.0%). **The recommendation
stands for cost predictability and for nothing else:** *which clauses* get escalated is
**not** stabilised, so any SLA, audit or reproducibility claim resting on set stability is
unsupported. The paragraph above is retained as written, with this correction attached
(hard rule 7).

### 3an. E6's aggregate delta hid its per-seed structure — and the structure changes the claim (2026-09-09)

§3al replaced a wrong yardstick with a paired row bootstrap. That was necessary and not
sufficient: **the row bootstrap resamples ROWS, not SEEDS**, so its p = 0.0756 is a
row-level p and carries **none** of the seed-level uncertainty. Reporting it alone
repeated the original error's shape at one remove — a single interval standing in for a
structure it cannot see.

| seed | tier0 | cascade | delta | applied esc. | esc. n | T0 correct | API correct |
|---|---|---|---|---|---|---|---|
| 1 | 0.7582 | 0.7567 | **−0.0015** | 4.00% | 120 | 47 | 48 |
| 2 | 0.7514 | 0.7564 | +0.0049 | 3.23% | 97 | 33 | 42 |
| 3 | 0.7471 | 0.7625 | +0.0153 | 4.93% | 148 | 50 | 52 |

**1. SIGN CONSISTENCY — FAILS, at the bar this project already uses.** E5 required
**sign-consistent 3/3** before calling `max_softmax` *reliably* better than `margin`
(§3ai). The cascade is positive on **2 of 3** seeds. The per-seed delta sd is **0.0085**,
which **exceeds the mean delta of +0.0063**. Applying our own bar: **the cascade is not
reliably better than Tier 0 alone.** Both tests are reported, neither replaces the other:

| test | result |
|---|---|
| paired row bootstrap (3,000 rows) | +0.0063, 95% CI [−0.0004, +0.0103], p = 0.0756 |
| seed sign-consistency (3 seeds) | **2/3 — FAILS** |

**2. ⚠ SUPERSEDED BY §3ap(B) AND §3ap(C) — READ THEM WITH THIS.** The McNemar test below is ACCURACY, not the registered macro-F1; on macro-F1 the API is better on escalated rows on 3/3 seeds. The "by construction" mechanism is WITHDRAWN: the escalation sets are only ~24% overlapping across seeds. Retained as written so the correction is legible.

**2. IT IS VARIANCE REDUCTION, NOT A LEVEL GAIN — and the direct test settles it.**
corr(Tier 0 quality, delta) = **−0.965**: the worst seed gains most and the best seed
*loses*. Across-seed sd falls from **0.0056 (tier0) to 0.0034 (cascade)**, ratio 0.61.
That is a variance signature, and the decisive measurement is on the escalated rows
themselves — **McNemar, pooled over seeds, on exactly the rows the router chose**:

> Tier 0 right & API wrong: **79**. Tier 0 wrong & API right: **91**. Net **+12 of 365**.
> Exact binomial **p = 0.399**.

**On the very rows the cascade acts, the API is not reliably better than the encoder it
is replacing.** So "the cascade adds accuracy" is **NOT supported**. What is supported is
narrower and partly mechanical: escalated rows get a **seed-invariant** API prediction,
and the router selects precisely the low-margin rows where Tier 0 is *most* seed-unstable,
so replacing ~4% of rows removes a disproportionate share of across-seed variance **by
construction**. At n = 3 seeds an sd ratio is not testable anyway. **The claim we may
make is "the cascade reduces across-seed variance, by a mechanism that would produce that
result even with no accuracy benefit" — not "the cascade is more accurate."**

**3. THRESHOLD TRANSFER — a single dev-calibrated threshold does not hold its escalation
rate.** One `dev_2000` margin quantile at target 4.06% realizes **4.00% / 3.23% / 4.93%**
on test: spread **1.70pp**, ratio **1.53×**.

- This is **C1-shaped territory arriving unregistered.** It is **not C1**: C1 compares
  `dev_2000` against the nested full 10k validation split for one model, and its 3pp bar
  is ~3 SD of *that* nested-sampling null (§3a). Across **retraining seeds** the null is
  different and 3pp has no derivation here. **Quoting C1's 3pp against this number would
  be borrowing a bar from a different question** — it is recorded as a measurement, not
  scored against a rule it was not written for.
- **Deployment consequence, stated plainly:** a threshold chosen once and shipped has a
  realized escalation rate that moves by **~50% relative** across retrainings of the same
  model on the same data. Since Tier 2 dominates marginal cost, **the API bill moves with
  it** — the cost axis of the E6 frontier is drawn at a rate the deployed system would
  not reliably hold. Any SLA or budget stated as "escalates ~4%" is unsupported; the
  honest form is a range.

### 3ao. E8's 1,000 rows were taken POSITIONALLY, and that is the take-first-N defect again (2026-09-09)

`scripts/submit_batch.py` selects `rows = man.indices[:n]`. `test_3000`'s manifest records
*"Indices are ascending, preserving the split's chronological row order."* So the 1,000
few-shot rows are **not a sample** — they are the **chronologically earliest third** of a
**deliberately chronological** test split (dataset rows 1–3424 of 9,992). §3d already
records the take-first-8 problem for exemplars; this is the same defect on the evaluation
set, and it went unnoticed because a subset of a stratified manifest *looks* like a sample.

**It is measurable, and it is not negligible.** Tier 0 on the first 1,000 vs the remaining
2,000, scored over the **95 classes both cover** so the label space is held constant:

| seed | first-1000 | rest-2000 | difference |
|---|---|---|---|
| 1 | 0.8000 | 0.7924 | +0.0075 |
| 2 | 0.8007 | 0.7803 | +0.0204 |
| 3 | 0.7895 | 0.7744 | +0.0151 |

**Sign-consistent 3/3, mean +0.0143.** The earliest rows are genuinely easier. **E8's
pairing is unaffected** — both arms ran on the same rows, so the *prompt* comparison is
valid and `delta_fs` stands. What is affected is **generalisation**: E8's numbers describe
the earliest third of the 2019 test split, not `test_3000`, and must be labelled that way.

**Number hygiene — two figures that must never be set side by side unqualified.**

| figure | value | what it is |
|---|---|---|
| E8 `gap` | **+0.1524** | Tier 0 − Sonnet zero-shot, **first-1,000 rows, 97 classes** |
| frontier | **+0.1374** | Tier 0 − Sonnet zero-shot, **test_3000, 100 classes** |

**These are not the same quantity and their difference is not a finding.** Both the row set
and the label-averaging space differ, and both differences push the same way: the earlier
rows are easier (+0.0143 measured above), and the 3 classes absent from the subset are rare
ones Tier 0 handles poorly, so excluding them from the average inflates the subset score.
Tier 0 reads **0.7782** on the subset against **0.7523** on the full set — a **+0.0260**
artefact of scope, not of model quality. **Wherever either number appears next to the
other, this note travels with it.**

### 3am. C4's rate-fallback switch, recorded BEFORE any few-shot prediction is scored (2026-09-09)

C4's registered procedure requires the *form* of its rule to be fixed from the
**zero-shot** rate alone, before the few-shot number is opened. Discharging that now.

- **The 3× class in `exemplars_8` is `Governing Laws`** (counts: `Governing Laws` 3;
  `Compliance With Laws`, `Organizations`, `Remedies`, `Solvency`, `Waivers` 1 each —
  six distinct classes over eight exemplars, as §3d records).
- **Zero-shot prediction rate of `Governing Laws`, on the 1,000 rows both runs share:
  58/1000 = 5.80%.**
- **5.80% is above the 1% trigger, so the RATIO form applies.** The bar is
  **≥ 2× = ≥ 11.60%** few-shot prediction rate, AND the shift must exceed that of every
  1× class. **The +2pp absolute fallback does NOT apply and is not used.**
- Single-exemplar zero-shot rates, for the second clause: `Compliance With Laws` 2.30%,
  `Solvency` 0.70%, `Remedies` 0.50%, `Organizations` 0.30%, `Waivers` 0.30%.

**Disclosure, because the procedure's whole point is that the order is auditable.** E8 was
scored first, so few-shot *aggregate* numbers (macro-F1, accuracy, count of distinct
classes predicted) were already open when this switch was recorded. Those aggregates do
not contain `Governing Laws`' few-shot rate, which is the only quantity the rule tests, so
the switch is uncontaminated **in substance** — but the ideal order would have fixed the
form before E8 ran, and it did not. Recorded rather than glossed.

**"89 vs 91 distinct classes predicted" is NOT C4** and must not be reported as if it
were. That was a descriptive aggregate noted alongside E8. C4 is the specific test above,
and it is what decides the question.

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
   `RUN_TAG = "ce_hostB"`. ~~Cost **1.0–2.2 h**~~ — **MEASURED 2.38 h (§3ba)**; the
   estimate was low and E1b's is re-costed from the measured rate. **ITS PURPOSE, WHICH IS EASY TO LOSE:** it
   supplies a **3-epoch number on a RECORDED host**, so that E1b (10 ep) vs host baseline
   (3 ep) is a **clean one-variable epochs comparison with both environments persisted**.
   That — not the comparison back to E1 — is the scientific reason to spend the run (~~1.0–2.2 h~~ → **2.38 h measured**).
   Its interpretation rule is registered in **§3at**.
2. Run **E1b seed 1** on the same host. ~~Cost **3.3–7.3 h**~~ → **~8.1–8.6 h (§3ba, measured basis)**, which no longer fits one commit; see §3bb.
3. The gate becomes **within-host**: does 10 epochs beat 3 epochs *on this host*? That is
   the one-variable question E1b was designed to ask, and it is answered without any
   cross-host term.
4. Report the new host's E1-baseline against E1 seed 1's **INT8** figure of **0.7582** as
   a **measured host effect at n=1**, explicitly labelled as having no variance estimate.
   (This step previously named **0.7636**, which is E1 seed 1's **FP32** number quoted
   against an INT8 rule — the same substitution §3ar corrected for 0.7570. The comparison
   is INT8-to-INT8 or it is not a comparison.) **This reading is DESCRIPTIVE ONLY: a
   difference here is detectable but NOT attributable**, because E1's environment was never
   recorded — see §3at.

**The cost of the host change is therefore +2.38 h MEASURED (~~+1.0–2.2 h~~), roughly +28% on E1b's seed-1
gate.** That is the price of keeping the comparison one-variable, and it is cheaper than
any analysis that tries to reason across hosts after the fact.

**Rejected:** running E1b alone on a new host and comparing to E1's INT8 headline of **0.7523** (this line previously read *"E1's 0.7570"* — E1's **FP32** mean quoted against an INT8 rule; corrected in §3ar). That is a
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
| Sonnet 5 (batch, cached) | ~~$0.4483~~ → **$0.43610 measured** | §1; **projection superseded by Stage 1's own usage — §3ax item 6** |
| **ratio** | **1 / 32,240** | derived |

**The crossover, recomputed.** `cost_local(V) = 633,862.43/V + 1.390e-5` per 1,000
clauses, against Sonnet 5's flat $0.4483:

> **⚠ ITEM 6 FIXED 2026-09-10 (§3ax). The $0.4483 above was a PROJECTION, and it is a stale
> INPUT TO A LIVE FORMULA — not merely a stale value.** It was forecast from Rung-2 caching
> behaviour *before Stage 1 ran*. Stage 1 has since completed, and the **measured** Sonnet-5
> batch+cached cost is **$0.43610 per 1,000** (`e6_frontier.json` `api_usd_per_1k`, derived
> from that run's own `usage` blocks per hard rule 10). Recomputed on the measured input:
>
> | | V\* (energy excluded) | V\* (energy included) |
> |---|---|---|
> | projected $0.4483 | 1,413,925 | 1,413,969 |
> | **measured $0.43610** | **1,453,465** | **1,453,511** |
> | shift | **+39,540 (+2.80%)** | **+39,542 (+2.80%)** |
>
> **No conclusion changes.** V\* remains ~1.45M clauses — **~18× the entire 80,000-clause
> LEDGAR corpus** — so the local device still does not break even on any realistic volume,
> which is E6's finding either way. What changes is that the break-even number is now
> computed from a **measurement of the run that happened** rather than from a forecast of it,
> and the input is no longer labelled "measured" when it was projected.


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
probe). ~~**1.0–2.2 h per seed, 3.0–6.6 h for all three**~~, against a 9 h design cap.

> **⚠ SUPERSEDED 2026-09-10 (§3ba/§3bb).** Measured: **2.38 h** for a 3-epoch seed, and
> **~8.1–8.6 h** for E1b's 10-epoch seed. **This entry's step-budget decision rested on
> these numbers and is reversed for E1b in §3bb** — the budget it called "machinery that
> never fires" now fires.

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

  **Per-seed, required by §3aw** (the mean above summarises these; all three are on
  the same side of both Sonnet arms, so the sign is not in question):

  | seed | Tier 0 macro-F1 | `gap` vs zero-shot 0.6258 |
  |---|---|---|
  | 1 | 0.7813 | +0.1555 |
  | 2 | 0.7813 | +0.1555 |
  | 3 | 0.7720 | +0.1462 |
  | **mean** | **0.7782 ± 0.0053** | **+0.1524** |

  - `delta_fs` = **+0.0018** macro-F1 (accuracy +0.0080)
  - paired bootstrap, 10,000 resamples, seed 20260909: **95% CI [-0.0180, +0.0213]**,
    p = 0.869 — **includes 0**
  - `gap` (Tier 0 − zero-shot, same rows) = **+0.1524** (per-seed +0.1555 / +0.1555 /
    +0.1462 — **3/3 same sign**), 95% CI [+0.1076, +0.1757], p = 0.0000 — excludes 0
  - `fraction_closed` = **+0.0119** (1.2% of the gap)
- **Cost:** **$0.00.** Responses were already on disk; no API call, nothing appended to
  `results/spend_ledger.jsonl`.
- **Manifest:** `test_3000` (`text_sha256 e719c110…`), restricted to the 1,000 few-shot
  ids, verified a strict subset of both `stage1`'s 3,000 and the manifest's indices.
  Exemplars: `exemplars_8` (`text_sha256 ac7e7be8…`).
- **Accept rule met?** The registered band resolves to **CAPABILITY, NOT PROMPTING** — the
  CI includes 0, which the band routes to the ≤ 0.20 disposition, and the point estimate
  closes 1.2% of the gap. **Restatement of H1/E4/E4b/E6 proceeds.**
- **Row provenance — POSITIONAL, not sampled (§3ao).** The 1,000 rows are
  `manifest.indices[:1000]`, i.e. the **chronologically earliest third** of a deliberately
  chronological test split, not a seeded draw. The pairing is unaffected (both arms ran on
  the same rows), so `delta_fs` stands; **generalisation is affected** — these numbers
  describe the earliest third of the 2019 test split, not `test_3000`.
- **Number hygiene (§3ao).** E8's `gap` of **+0.1524** (first-1,000 rows, 97 classes) and
  the frontier's **+0.1374** (`test_3000`, 100 classes) are **not the same quantity**;
  Tier 0 alone reads 0.7782 vs 0.7523 across the two scopes, a +0.0260 artefact of row set
  and label space. This note travels with either figure whenever they appear together.
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
- **Per-seed structure, which the aggregate hides (§3an):** deltas −0.0015 / +0.0049 /
  +0.0153 — **sign-consistent 2/3, FAILING** the 3/3 bar E5 used for "reliably better";
  per-seed delta sd **0.0085 exceeds the mean delta**. The row bootstrap's p is a
  **row-level** p and does not carry seed-level uncertainty; both tests are reported.
  McNemar on the escalated rows, pooled: **79 / 91, net +12 of 365, p = 0.399** — on the
  rows the cascade acts, the API is **not** reliably better than the encoder. The
  supported claim is **variance reduction by construction**, not a level gain.
- **Notes.** The conclusion ("would not claim the cascade beats Tier 0 alone") stands, but
  **not for the reason first given.** See §3al: the across-seed sd was the wrong yardstick,
  and 0.0063 was 1.1-1.9× it rather than inside it. The correct paired interval is
  **tighter**, and puts the result at **borderline** (p = 0.076, lower bound -0.0004),
  not comfortably null.

### C2 — few-shot confidence anchoring *(2026-09-09)*

- **Experiment id:** C2 (registered 2026-09-07, adopted verbatim 2026-09-08, §3ad/§3b)
- **Config / command:** `python scripts/score_c2_c4.py`; `claude-sonnet-5`, same 1,000
  rows, zero-shot vs few-shot. **Scope: Sonnet 5 only**, per §3b.
- **Result:**

  | arm | mean | sd | mode | distinct values |
  |---|---|---|---|---|
  | zero-shot | 0.8514 | **0.1598** | 0.98 | 31 |
  | few-shot | 0.8443 | **0.1412** | **0.85** | 26 |

  - sd ratio **0.8837** vs bar **≤ 0.5** — **FAIL**
  - few-shot mode **0.85** vs bar **within 0.02 of 0.90** — **FAIL**
- **Cost:** $0.00.
- **Accept rule met?** **NO — C2 fails both clauses.**
- **Notes.** The predicted direction was **wrong**. A fixed `"confidence": 0.9` in all
  eight exemplars did **not** anchor Sonnet 5's verbalized confidence: the spread narrowed
  by ~12%, not the ~50% required, and the mode moved to 0.85 — *away* from the anchored
  value, not toward it. Per §3b's falsification clause this means the compression of
  verbalized confidence is **intrinsic to the model, not induced by our prompt**, which
  strengthens rather than weakens H2's case for a continuous margin over verbalized
  confidence. **⚠ AMENDED — see §3ap(A).** The mode moved **toward** 0.9 (0.080 → 0.050
  from the anchor), not away; the median lands exactly on 0.9 and the mass within ±0.05
  rises 202 → 307. The supported statement is a **measurable pull on location that fell
  well short of the registered window, with no compression** — not "did not anchor". The
  §3b falsification clause requires "mode away from 0.9" and therefore **does not fire**;
  the inference that compression is "intrinsic to the model, not induced by our prompt" is
  **withdrawn** as resting on a clause that never triggered. What stands is that
  **compression is untouched by the anchor** (sd ratio 0.88), which is what H2 needs. §3b's recorded Type II limitation still applies in the other direction: at
  0.5 the bar rules out a *large* anchoring effect, and a ratio of 0.88 is not evidence of
  *no* effect, only of no large one.

### C4 — exemplar class over-prediction *(2026-09-09)*

- **Experiment id:** C4 (registered 2026-09-07, adopted verbatim 2026-09-08; rate-fallback
  form fixed in **§3am before any few-shot prediction was read**)
- **Config / command:** `python scripts/score_c2_c4.py`; same 1,000 rows. Form: **RATIO**
  (`Governing Laws` zero-shot 5.80% > 1% trigger), bar **≥ 2×**. The +2pp fallback does
  not apply.
- **Result:**

  | class | exemplars | zero-shot | few-shot | ratio | shift |
  |---|---|---|---|---|---|
  | **Governing Laws** | **3×** | 5.80% | 5.90% | **1.02×** | +0.10pp |
  | Compliance With Laws | 1× | 2.30% | 2.20% | 0.96× | −0.10pp |
  | Organizations | 1× | 0.30% | 0.80% | 2.67× | +0.50pp |
  | Remedies | 1× | 0.50% | 0.90% | 1.80× | +0.40pp |
  | Solvency | 1× | 0.70% | 0.70% | 1.00× | +0.00pp |
  | Waivers | 1× | 0.30% | 1.10% | 3.67× | +0.80pp |

  - ratio **1.02×** vs bar **≥ 2×** — **FAIL**
  - 3× shift **+0.10pp** vs largest 1× shift **+0.80pp** — **FAIL** (second clause)
- **Cost:** $0.00.
- **Accept rule met?** **NO — C4 fails both clauses.**
- **Notes.** The predicted direction was **wrong**. Repeating a class three times in the
  exemplar set did **not** bias predictions toward it: `Governing Laws` moved by 1 row in
  1,000. Per §3d's falsification clause, **exemplar repetition did not bias predictions,
  and the uniform-draw exemplar policy carries less risk than §3c records.**
  **Caveat on the second clause, which cuts against reading it as a finding:** the 1×
  classes that "moved more" sit on single-digit counts (`Waivers` 3→11 rows, `Organizations`
  3→8), where a ratio is unstable — precisely the instability C4's own 1% rate-fallback
  exists to guard against, applied to the comparison classes rather than to the 3× class.
  The safe statement is that **no class shifted materially**, not that the 1× classes
  shifted more.
  **This supersedes the "89 vs 91 distinct classes predicted" remark in the E8 entry**,
  which was a descriptive aggregate and **not** C4 (§3am).

### E6 per-seed structure — instrumentation *(2026-09-09)*

- **Experiment id:** E6 supporting; see §3an
- **Config / command:** `python scripts/e6_seed_structure.py`
- **Result:** sign-consistency **2/3 (fails)**; corr(tier0 quality, delta) **−0.965**;
  across-seed sd **0.0056 → 0.0034**; McNemar on escalated rows **79/91, p = 0.399**;
  realized escalation **4.00 / 3.23 / 4.93%** from one dev threshold (spread 1.70pp,
  1.53×); Tier 0 first-1000 vs rest-2000 over 95 common classes **+0.0143, 3/3 sign-consistent**.
- **Cost:** $0.00.
- **Accept rule met?** n/a — instrumentation under C3's remit.

### E3 — INT8 vs FP32, both Tier 0 families *(2026-09-11)*

- **Experiment id:** E3 (registered pre-signing; scope settled §3bi, statistic ambiguity
  §3bj.1)
- **Date:** 2026-09-11
- **Commit:** `e6edcb6` + this entry
- **Config / command:** every INT8 figure from `scripts/score_int8_local.py` on **arm64**
  under ORT 1.29.0 — the ISA that deploys, per §3ah. FP32 figures are the Kaggle run JSONs'
  `test_3000_fp32`, which §3bi verified reproduce locally through the registered scorer
  with the `test_3000_indices` guard to **0 delta** at n=3000 for the E1b family. Paired:
  same artefact, same 3,000 rows, `manifest_sha256 e719c110…`.
- **Seeds:** 1 / 2 / 3 for **both** families.
- **Result — `INT8 − FP32` macro-F1 on `test_3000`:**

  | seed | **E1** INT8 | E1 FP32 | **delta** | **E1b** INT8 | E1b FP32 | **delta** |
  |---|---|---|---|---|---|---|
  | 1 | 0.758213 | 0.763591 | −0.005378 | 0.797443 | 0.811111 | **−0.013668** |
  | 2 | 0.751423 | 0.759740 | **−0.008317** | 0.792468 | 0.794219 | −0.001751 |
  | 3 | 0.747119 | 0.747812 | −0.000693 | 0.806321 | 0.813919 | −0.007598 |
  | **mean** | 0.752252 | 0.757048 | **−0.004796** | 0.798744 | 0.806416 | **−0.007672** |

- **Cost:** **$0.00.** No API call; nothing appended to `results/spend_ledger.jsonl`.
- **Manifest:** `test_3000`, `text_sha256 e719c110…`; n=3000, 0 unmatched,
  `classes_averaged=100` on every run.
- **Accept rule met? — NO VERDICT. WITHHELD, and that is this entry's substance.**

  | reading | E1 | E1b | rule met? |
  |---|---|---|---|
  | **per seed** (max \|delta\|) | 0.008317 ≤ 0.01 | **0.013668 > 0.01** | E1 yes, **E1b NO** |
  | **3-seed mean** | 0.004796 ≤ 0.01 | 0.007672 ≤ 0.01 | **both yes** |

  **E3's registered text names no seed aggregation (§3bj.1).** The readings agree on E1 and
  disagree on E1b, so the project never had to choose — and cannot choose now without
  choosing *after* seeing which one fails. **Both recorded; neither adopted.** A verdict is
  owed once the statistic is written into E3's own text, which must happen **before** the
  next INT8-vs-FP32 number is opened.
- **Notes.**
  - **Kaggle's own `e3_int8_minus_fp32_macro_f1` is unusable and is NOT the source above.**
    E1's run JSONs record **−0.7599 / −0.7567 / −0.7455** — near-total collapse — because
    Kaggle evaluates INT8 on non-VNNI x86 where the kernels degrade silently (§3ah, §3as).
    Every delta here is arm64.
  - **Supersedes and rescues nothing.** E1b's verdict is unchanged (§3bi: **NOT ACCEPTED**
    at 0.798744), and E3's FP32-as-headline prohibition is untouched.
  - §3bc's canary-floor margin cites E3's largest delta; that figure moved 0.0083 →
    0.013668 and was corrected in place at `e6edcb6` (≈12× → ≈7.3×, **floor unchanged**).
  - **Not engaged by the §3bk swap:** the served path is **FP32**, so no INT8 delta sits in
    it under either reading.

### Served-config escalation check — instrumentation *(2026-09-11)*

- **Experiment id:** none. **DESCRIPTIVE INSTRUMENTATION under §3bk**, reported because
  the served artefact changed. **NOT an E6 rebuild** — E6, H1 and E8 remain on E1 and no
  frontier, break-even, gap or USD figure was recomputed or touched.
- **Date:** 2026-09-11
- **Config / command:** `python scripts/served_config_escalation.py`. Tier 0 =
  `models/onnx_ce10ep_1_fp32` (E1b seed 1) FP32 logits on **arm64**; threshold **0.528864**
  **read from the served `configs/router_threshold_fp32.json`, not recomputed**; served
  rule `margin < threshold`; Tier 2 arm = Sonnet-5 **zero-shot** predictions already on
  disk in `results/stage1_results.json`. The script **refuses** to run if the logits'
  precision, ISA or artefact disagree with the threshold file's.
- **Seeds:** **n = 1** — the served artefact. Hard rule 2 forbids reading this as a model
  result; it is a property of one deployed configuration.
- **Result — `test_3000`, 3,000 rows:**

  | | macro-F1 |
  |---|---|
  | Tier 0 alone | **0.809052** |
  | cascade (Tier 0 → Sonnet 5) | 0.809874 |
  | **delta** | **+0.000822** |

  - escalated **129 / 3000 = 4.30%** (target rate 4.056%)
  - paired bootstrap, 10,000 resamples, seed 20260911: **95% CI [−0.0060, +0.0072]**,
    **p = 0.857 — INCLUDES 0**
  - **McNemar on the 129 escalated rows** (accuracy, not macro-F1): API right / Tier 0
    wrong **28**, Tier 0 right / API wrong **21**, net **+7**, exact two-sided
    **p = 0.392**
  - accuracy **on those rows**: Tier 0 **0.3488**, Sonnet 5 **0.4031**
- **Cost:** **$0.00.** No API call; nothing appended to `results/spend_ledger.jsonl`.
- **Accept rule met?** n/a — no accept rule. **Stated plainly: the cascade does NOT beat
  Tier 0 alone at the served operating point.** The interval includes 0 and the point
  estimate is +0.0008, roughly a tenth of E1's +0.0063.
- **IN-SAMPLE, per §3bc.** The threshold *value* is `dev_2000`-calibrated (hard rule 1
  intact), but the **4.056% RATE was selected on test**, so this is an in-sample figure for
  the rate that chose it. It is not a held-out estimate of the deployed configuration.
- **Notes.**
  - **THE ROUTER IS WORKING; THE ESCALATION TARGET IS NOT PAYING.** Tier 0 scores **0.3488**
    on the rows it routes away against **0.8733** overall — the margin signal is finding
    genuinely hard rows. Sonnet 5 manages **0.4031** on those same rows. This **replicates
    §3aq/§3av's E6-A verdict on the new served artefact**, and slightly more starkly: E1's
    delta was +0.0063 with a lower bound of −0.0004 (borderline); this one is +0.0008 with
    an interval comfortably spanning 0.
  - **What this does NOT say.** It does not say escalation is harmless to remove — the
    escalation path is also the defined handling route for low-confidence rows, and this
    measures accuracy only, not that. Whether escalation stays enabled is a deployment
    decision and is **not** made here.
  - Fixed while producing this: `scripts/score_int8_local.py` stamped `precision: "int8"`
    into every saved logits npz regardless of the artefact — the same defect corrected in
    `scripts/dev_logits_int8_local.py` at §3bk. It had already mislabelled this run's FP32
    logits once; `--precision` is now required with no default, and the file was
    regenerated.

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
