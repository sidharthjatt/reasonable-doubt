# Running the Kaggle notebooks

Two notebooks, run in this order. **Start Tier 1 first** — it is the long pole
(9–15 GPU-hours vs 3–5), and Tier 0 can follow in the same or a later session.

---

## THE INSTALL PROTOCOL — read this first

Two runs have now been lost to dependency problems, both from the same cause: **pip
cannot replace a package the running kernel has already imported.** The result is a
*mixed* install — `peft` reported 0.19.1 while executing 0.20.0's `bnb.py`, which
crashed with `LoraConfig object has no attribute 'velora_config'`.

**Every notebook is now TWO cells.** For each one:

1. Paste **CELL 1** (the `!pip install` block) and run it. **Read the output** — it is
   no longer suppressed, and resolver errors appear there.
2. **RESTART THE KERNEL.** Kaggle: `Run -> Restart & clear cell outputs`, or the
   Restart button in the session panel. This step is not optional.
3. Paste **CELL 2** (everything after the `CELL 2 of 2` banner) and run it.

**Never use Run All, on any notebook, on any run — first or resumed.** Run All executes
both cells in one kernel with no restart in between, which is the failure this protocol
exists to prevent. The three steps above are the only supported way to start a cell.

CELL 2 begins by asserting the running versions match the pins and **raises** if they
do not. Previously PREFLIGHT passed while running versions that had never been
introspected; that is what this check prevents.

### Run the probe before the notebook — both notebooks have one

Each notebook has a matching probe. Run CELL 1, restart the kernel, then run the probe
in its own cell — in the same kernel CELL 2 will use. **If the probe fails, do not start
that notebook.**

| probe | for | ~time | what it exercises |
|-------|-----|-------|-------------------|
| `kaggle_probe_qlora.py` | Tier 1 | ~2 min | the path that cannot be tested off-CUDA: 4-bit load → LoRA wrap → one real training step → generation |
| `kaggle_probe_tier0.py` | Tier 0 | ~90 s | the path that otherwise first runs 3–5 h in: untrained DeBERTa-v3-base → ONNX export → ORTQuantizer arm64 INT8 → one onnxruntime forward pass on 4 rows |

Tier 0's probe matters at least as much as Tier 1's. Tier 0 is the notebook that
**downgrades a preloaded package** (transformers 5.0.0 → 4.57.6), which is the exact
operation that caused the mixed peft install — Tier 1 sidesteps it by forcing only
`peft`, Tier 0 cannot. And an export failure discovered after training costs the whole
session, checkpoint included.

Both probes end in one line: `PASSED …` or `FAILED — DO NOT START TIER n`.

The 90 s figure assumes the DeBERTa weights are already in the HF cache; a cold ~370 MB
download is on top of that.

### Why the two notebooks pin different versions

| notebook | transformers | why |
|----------|--------------|-----|
| Tier 1 | **Kaggle's own (5.0.0) — not changed** | `trl 1.12.0` needs `>=4.56.2` with no upper bound; `peft 0.20.0` has none. Only `peft` is forced, so only one package is replaced. |
| Tier 0 | **pinned `==4.57.6`** | `optimum-onnx` declares `transformers<4.58.0,>=4.36`, and Tier 0 needs optimum for the ONNX export. This IS a downgrade from Kaggle's 5.0.0. |

They run in separate sessions, so the conflict never has to be resolved in one
environment.

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
4. **Split the `.py` file into two cells at the `CELL 2 of 2` banner.** Everything
   above the banner is CELL 1; everything below it is CELL 2. Do **not** paste the file
   into one cell, and do **not** use **Run All** — either one runs the install and the
   training code in the same kernel, with no restart between them, which is exactly how
   the last two sessions were lost.

---

## Tier 1 — `kaggle_tier1.py` (start this first)

CELL 1 → **restart the kernel** → run `kaggle_probe_qlora.py` (~2 min) → CELL 2.
Expect 3–5 hours per seed, 3 seeds.

The probe runs in its own cell *after* the restart and *before* CELL 2, in the same
kernel CELL 2 will use. **If the probe fails, do not start Tier 1.**

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

Same protocol: CELL 1 → **restart the kernel** → run `kaggle_probe_tier0.py` (~90 s) →
CELL 2. Expect ~1–1.5h per seed.

Tier 0 is the notebook that **downgrades a preloaded package** (`transformers`
5.0.0 → 4.57.6), so it carries *more* mixed-install risk than Tier 1, not less, and its
ONNX export does not run until 3–5 hours in. **If the probe fails, do not start
Tier 0** — a failure there is a failure you would otherwise have met after a full
training run.

Early on you must see:
```
train_holdout_3000 verified 97eebc04d0c7...
```
**If that line does not appear, stop.** It means the frozen selection set does not
match the committed manifest, and the run would not be comparable to anything else.

A seed is finished when `tier0_ce_seed1.json` appears in `/kaggle/working` and you see
the selection metrics printed. Final line:
`ALL SEEDS DONE — download /kaggle/working/*.json, *.npz and int8_* dirs`

**To run E2's arm afterwards:** change one line near the top of CELL 2,
`LOSS_ARM = "ce"` → `LOSS_ARM = "sqrt_inv_freq"`, then re-run **CELL 2 only** — the
environment is already installed and the kernel already restarted, so CELL 1 must not
be re-run in that kernel. Results land under different filenames, so nothing is
overwritten. **Do this only after C3 reports**
(see PREREGISTRATION §3l — the sequencing is CE 3 seeds → C3 → sqrt_inv_freq 3 seeds).

---

## If a session dies

This is expected — Kaggle caps session length, and both notebooks are built for it.
The *resume* is checkpoint-driven and needs nothing from you; the *environment* is what
you have to get right, and **Run All is now the wrong answer for a resume too.**

Two invariants govern every run, first or resumed:

> **CELL 1 may only run in a kernel that has not yet imported the pinned packages.**
> **CELL 2 may only run in a kernel that was restarted after CELL 1 last ran.**

Run All violates the second on a first run, and can violate the first on a resume.

**Which case are you in?** Restart the kernel, then run this in a scratch cell — it
imports nothing that CELL 1 replaces, so it is safe in either case:

```python
import importlib.metadata as md
for pkg in ("transformers", "peft", "trl"):   # Tier 0: transformers, optimum, onnxruntime
    try: print(pkg, md.version(pkg))
    except md.PackageNotFoundError: print(pkg, "NOT INSTALLED")
```

| what it prints | you are in | do this |
|----------------|-----------|---------|
| the pinned versions | **same session**, packages survived — kernel restart or interrupt only | run **CELL 2 only**. Do not re-run CELL 1. |
| Kaggle's stock versions, or `NOT INSTALLED` | **new session**, fresh container | CELL 1 → **restart** → probe → CELL 2, the full protocol |

If you cannot tell, take the second row: CELL 1 → restart → probe → CELL 2 is always
correct, and costs about three minutes.

Once CELL 2 is running, the resume itself is automatic:

- Seeds whose `*_seed*.json` exists are **skipped**. The two notebooks print
  *different* strings, so match the one for the notebook you are running:
  Tier 1 prints `seed 1: complete, skipping`; Tier 0 prints
  `seed 1: already complete, skipping`.
- A seed that was interrupted **resumes from its last checkpoint** — you will see
  `=== seed 2 (RESUMING) ===`.
- Nothing is lost and nothing is double-counted.

**Do not** delete `/kaggle/working` between runs — that is where the completion markers
and checkpoints live. Deleting it forces a full restart.

If you see `=== seed 1 (fresh) ===` when you expected `RESUMING`, the working directory
was cleared. That costs time but is not incorrect; the run is still valid.

---

## Long runs: the session cap, and running as a commit

**Tier 1 is measured at ~13.4 h per seed on one T4** (0.07 it/s over 3,563 steps,
seed 1, 2026-09-08). **No seed fits in one session under any cap**, so the resume path
below is load-bearing rather than a contingency.

**Session cap.** Two independent secondary readings agree that Kaggle raised notebook
runtime from 9 h to **12 h for CPU and GPU** (9 h remains for TPU), which explains the
9-vs-12 disagreement between sources: anything written before the increase says 9.
**Neither reading is primary** — Kaggle's own pages render client-side and return no
quota text to a fetch — so **plan for 9 h.** It is safe under both, and the difference
is real: at `save_steps=500` a 9 h cut loses ~1.1 h of work, a 12 h cut ~0.1 h.

**Recommended: lower `save_steps` from 500 to 200.** At 0.07 it/s a 500-step cadence is
one checkpoint every **2.0 h**, so a kill lands on average 1 h — worst case 2 h — after
the last one. 200 steps is ~48 min, bounding the loss at ~0.8 h. A checkpoint here is
the LoRA adapter plus Adam state, roughly 220 MB, and writing it takes seconds.
**This changes no training mathematics** — checkpoint cadence does not affect the model
— so it is not a preregistration amendment.

### Interactive sessions will not survive this. Use Save & Run All.

An interactive session shows an **"Are you still there?"** prompt and terminates on
inactivity. A 13 h run left in a browser tab will be killed. **Save Version → Save & Run
All (Commit)** runs a copy of the notebook **in the background on a separate machine**,
and survives closing the browser and the interactive session timing out.

**But a commit runs every cell in ONE kernel with no restart between CELL 1 and CELL 2 —
the exact condition the two-cell protocol exists to prevent.** Whether that is safe turns
on a question this repo has not answered: a commit starts from a *fresh container*, so
the mixed install only recurs if Kaggle's kernel has already imported `peft` before
CELL 1 runs. That is plausible-sounding either way, and this project has now lost three
sessions to plausible-sounding.

**Do not assume it. Measure it, in this order:**

1. **Commit the PROBE notebook first** (`kaggle_probe_qlora.py` as its own notebook:
   CELL 1, then the probe). ~5 minutes of quota. It exercises CELL 1 → import in a
   single kernel with no restart — structurally identical to what a Tier 1 commit does,
   at 1/150th the cost.
2. **If the probe commit passes**, the commit path is safe: commit Tier 1.
3. **If it fails at the version gate or the `velora_config` crash-site check**, the
   commit path genuinely needs restructuring. The fix is not to re-order cells — no cell
   order gives you a restart inside a commit — it is to **run CELL 2 in a subprocess**,
   which gets a fresh interpreter and therefore a guaranteed-clean import state:
   write CELL 2 to a file, then `!python /kaggle/working/tier1_cell2.py`.

A commit run also makes the probe's exit behaviour matter: `kaggle_probe_qlora.py`
currently **prints** `PROBE FAILED` without raising, so in a commit the run would carry
straight on into CELL 2 and spend 13 h anyway. Give it a hard `raise` before committing
anything that depends on it gating.

### Exactly what to do when a seed is killed mid-training

Seed 1 dying at, say, step 2,100 of 3,563 is the expected case, not a failure.

**Step 1 — new session or same one?** Open the notebook, restart the kernel, and run the
version snippet from **If a session dies** above. A commit that hit the cap always means
a new container, so after a commit: assume new.

**Step 2 — run the cells.**

| situation | what to run |
|-----------|-------------|
| new container (the usual case) | CELL 1 → **restart** → probe → CELL 2 |
| same session, packages survived | **CELL 2 only** |

**Step 3 — confirm it RESUMED and did not restart.** Four lines, in order:

```
=== seed 1 (RESUMING) ===          <- the word RESUMING. "(fresh)" means it restarted.
  trainable param dtypes after re-cast: {'torch.float32'} (392 tensors changed)
```
then a progress bar that **starts near the last checkpoint, not at 0/3563**.

- `=== seed 1 (RESUMING) ===` is the marker. It appears only when
  `t1_ck_1/checkpoint-*` exists.
- **`=== seed 1 (fresh) ===` means the checkpoint directory is gone** and you are paying
  the full 13.4 h again. Stop and check `/kaggle/working` before letting it run.
- **Expect a delay of several minutes before the progress bar moves.** `ignore_data_skip`
  defaults to `False`, so Trainer replays the dataloader to the resume position — for
  ~2,000 steps that is 32,000 examples of collation. **This is the skip, not a hang**,
  and it is what preserves the data order for the seed.

**Why a resume stays comparable to an uninterrupted run:** the checkpoint carries RNG
state, optimizer state and scheduler position, and the data skip restores the batch
sequence. A resumed seed is the same experiment as an unresumed one. That is the reason
`ignore_data_skip` must stay `False` even though it costs minutes.

**Never delete `/kaggle/working` between runs** — the completion markers
(`tier1_seed*.json`), the adapters (`t1_adapter_*`) and the checkpoints (`t1_ck_*`) all
live there.

## After Tier 0: re-score INT8 on the Mac

Kaggle quantises for **arm64** but evaluates on its own **x86** CPU. INT8 kernels are
ISA-specific and numerics can differ, so Kaggle's INT8 accuracy is measured on hardware
that never serves the model. **E1's accept rule attaches to the number measured here.**

Download `int8_ce_<seed>/` and `tier0_ce_seed<seed>.json`, then:

```bash
.venv/bin/python scripts/score_int8_local.py \
    --int8-dir <path to downloaded int8_ce_1> \
    --seed 1 \
    --kaggle-result <path to downloaded tier0_ce_seed1.json>
```

It prints both figures and their difference. If they differ, the **local** number is
authoritative and the gap is reported as a property of the deployed system.

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
| session ends silently mid-training | Kaggle time cap | see **If a session dies** — check the versions first, then CELL 2 or the full protocol. Not Run All |
| `bitsandbytes` import error | CPU-only session | accelerator is not set to GPU |
