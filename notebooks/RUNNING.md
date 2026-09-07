# Running the Kaggle notebooks

Two notebooks, run in this order. **Start Tier 1 first** — it is the long pole
(9–15 GPU-hours vs 3–5), and Tier 0 can follow in the same or a later session.

---

## Before you start

1. Go to <https://www.kaggle.com/code> → **New Notebook**.
2. Right-hand panel → **Session options**:
   - **Accelerator**: `GPU T4 x2` (or `GPU P100` — either works; T4 x2 is usually
     easier to get).
   - **Internet**: **ON**. Both notebooks `pip install` and download the dataset.
     Without this they fail immediately.
   - **Persistence**: `Files only` if offered. Not required — the notebooks re-derive
     everything — but it speeds up a resume.
3. **Check your quota before starting**: the GPU hours remaining figure is shown in the
   session options panel. Tier 1 needs 9–15h, Tier 0 needs 3–5h, of a 30h weekly quota.
4. Paste the whole `.py` file into **one cell**. Do not split it.

---

## Tier 1 — `kaggle_tier1.py` (start this first)

Paste, then **Run All**. Expect 3–5 hours per seed, 3 seeds.

**A seed is finished when you see, in order:**

```
=== seed 1 (fresh) ===
   ... training progress ...
   eval 0/3000
   eval 480/3000
   ...
{
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "seed": 1,
  "macro_f1": 0.xxxx,
  ...
}
```

and a file `tier1_seed1.json` appears in `/kaggle/working`. **That file is the
completion marker** — its presence is what makes a re-run skip the seed.

**Final line when all three are done:**
`DONE — download /kaggle/working/tier1_*.json and t1_adapter_* dirs`

## Tier 0 — `kaggle_tier0.py`

Same procedure. Expect ~1–1.5h per seed.

Early on you must see:
```
train_holdout_3000 verified 97eebc04d0c7...
```
**If that line does not appear, stop.** It means the frozen selection set does not
match the committed manifest, and the run would not be comparable to anything else.

A seed is finished when `tier0_ce_seed1.json` appears in `/kaggle/working` and you see
the selection metrics printed. Final line:
`ALL SEEDS DONE — download /kaggle/working/*.json, *.npz and int8_* dirs`

**To run E2's arm afterwards:** change one line near the top,
`LOSS_ARM = "ce"` → `LOSS_ARM = "sqrt_inv_freq"`, and Run All again. Results land under
different filenames, so nothing is overwritten. **Do this only after C3 reports**
(see PREREGISTRATION §3l — the sequencing is CE 3 seeds → C3 → sqrt_inv_freq 3 seeds).

---

## If a session dies

This is expected — Kaggle caps session length, and both notebooks are built for it.

**Do exactly this: open the notebook again and Run All. Nothing else.**

- Seeds whose `*_seed*.json` exists are **skipped** — you will see
  `seed 1: already complete, skipping`.
- A seed that was interrupted **resumes from its last checkpoint** — you will see
  `=== seed 2 (RESUMING) ===`.
- Nothing is lost and nothing is double-counted.

**Do not** delete `/kaggle/working` between runs — that is where the completion markers
and checkpoints live. Deleting it forces a full restart.

If you see `=== seed 1 (fresh) ===` when you expected `RESUMING`, the working directory
was cleared. That costs time but is not incorrect; the run is still valid.

---

## What to download when finished

From `/kaggle/working`:

| file | why |
|------|-----|
| `tier0_ce_seed*.json` | E1 selection + FP32 test metrics |
| `logits_ce_seed*.npz` | **needed for E5** (routing signals) and C3 |
| `int8_ce_*/` | **the deployed artefact** — E1's accept rule attaches to it |
| `tier1_seed*.json` | E4 metrics |
| `tier1_preds_seed*.json` | per-row predictions, for the offline cascade simulation |
| `t1_adapter_*/` | LoRA adapters, for local MLX serving later |

The `.npz` logits matter most — every routing threshold in E5 and E6 is simulated from
them offline at zero cost, so losing them means re-running the GPU work.

---

## Common failures

| symptom | cause | fix |
|---------|-------|-----|
| `pip` errors, no internet | Internet toggle off | turn it on, restart session |
| `CUDA out of memory` (Tier 1) | batch too large for the assigned GPU | lower `BS` from 4 to 2; `GA` compensates |
| `assert ... holdout sha` | dataset or sampling changed | **stop and report** — do not work around it |
| session ends silently mid-training | Kaggle time cap | re-run the cell; it resumes |
| `bitsandbytes` import error | CPU-only session | accelerator is not set to GPU |
