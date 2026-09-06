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
