# =============================================================================
# PERSISTENCE PROBE — does /kaggle/working survive from one COMMIT to the next?
#
# The entire multi-session plan (PREREGISTRATION 3u, lever 1) rests on a resume, and
# the resume rests on checkpoints in /kaggle/working still being there next time. That
# has never been checked. Under an interactive session with Persistence = "Files only"
# it is believed to hold; under Save & Run All (Commit) each version is thought to start
# from a fresh container, with the previous run's /kaggle/working becoming that
# VERSION'S OUTPUT rather than the next run's working directory.
#
# If that is right, every commit restarts seed 1 at "(fresh)" and the plan fails --
# discovered nine hours in. Kaggle's documentation could not settle it (its pages render
# client-side and return no text to a fetch), so it is settled by measurement.
#
# ---------------------------------------------------------------------------
# SET ACCELERATOR TO **NONE**. Persistence is not GPU-specific, so this costs ZERO
# GPU quota. Do not burn T4 hours answering a filesystem question.
# Internet is not needed either.
# ---------------------------------------------------------------------------
#
# HOW TO RUN — three commits, about a minute each:
#
#   RUN 1  Save Version -> Save & Run All (Commit). Read the output.
#          Expect: "no marker" and run_index 1.
#   RUN 2  Commit again. **YOU MUST MAKE A REAL EDIT FIRST** — a commit whose diff is
#          +0 -0 is SKIPPED by Kaggle and reports "Ran in 0 seconds" without running.
#          Changing a comment is enough. Read the output.
#          -> marker FOUND  => /kaggle/working carries over between commits.
#          -> marker ABSENT => it does not. This is the expected result.
#   RUN 3  Only if RUN 2 says ABSENT. In the editor: Add Input -> Notebook Output ->
#          this notebook -> its RUN 2 version. Commit again.
#          This does not re-ask the question; it VALIDATES THE FIX and prints the exact
#          /kaggle/input/<slug>/ mount path that kaggle_tier1.py will need to restore
#          checkpoints from.
#
# Each run appends to the marker, so RUN 3's marker shows every prior run that survived.
# =============================================================================
import json, os, socket, sys, time
from pathlib import Path

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
MARKER = WORK / "persistence_marker.json"

print("=" * 72)
print("PERSISTENCE PROBE")
print("=" * 72)
print(f"hostname        : {socket.gethostname()}   <- differs per container")
print(f"boot/run time   : {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
print(f"KAGGLE_KERNEL_RUN_TYPE = {os.environ.get('KAGGLE_KERNEL_RUN_TYPE')!r}"
      "   <- 'Batch' for a commit, 'Interactive' in the editor")

# ---- 1. THE QUESTION: is the previous run's marker still in /kaggle/working? ----
print("\n[1] /kaggle/working")
existing = sorted(p.name for p in WORK.iterdir()) if WORK.exists() else []
print(f"    contents ({len(existing)}): {existing[:25]}{' …' if len(existing) > 25 else ''}")

prior = None
if MARKER.exists():
    try:
        prior = json.loads(MARKER.read_text())
    except Exception as e:
        print(f"    marker present but unreadable: {e!r}")
        prior = {"runs": []}

if prior is not None:
    print(f"\n    >>> MARKER FOUND. /kaggle/working DOES carry over between runs.")
    print(f"    >>> previous runs recorded: {len(prior.get('runs', []))}")
    for r in prior.get("runs", []):
        print(f"          run {r['run_index']}  {r['time']}  {r['run_type']}  {r['host']}")
else:
    print(f"\n    >>> NO MARKER. /kaggle/working did NOT carry over from a previous run.")
    print(f"    >>> If this is RUN 2 or later, the resume design cannot rely on it.")

# ---- 2. THE FIX: is a previous version's output mounted as an input? ----------
print("\n[2] /kaggle/input  (this is where a previous version's OUTPUT appears, if")
print("    you added it via Add Input -> Notebook Output)")
if not INPUT.exists():
    print("    /kaggle/input does not exist — no data sources attached")
else:
    srcs = sorted(p for p in INPUT.iterdir())
    if not srcs:
        print("    (empty — no data sources attached)")
    for src in srcs:
        inner = sorted(q.name for q in src.iterdir()) if src.is_dir() else []
        print(f"    {src}/   -> {inner[:15]}{' …' if len(inner) > 15 else ''}")
        found = list(src.rglob("persistence_marker.json"))
        for f in found:
            print(f"       >>> MARKER FOUND VIA INPUT: {f}")
            print(f"       >>> THIS IS THE RESTORE PATH kaggle_tier1.py must copy from.")
            try:
                d = json.loads(f.read_text())
                print(f"       >>> it records {len(d.get('runs', []))} prior run(s)")
            except Exception as e:
                print(f"       >>> unreadable: {e!r}")

# ---- 3. Write/extend the marker so the NEXT run can see it -------------------
runs = (prior or {}).get("runs", [])
# Carry forward anything visible through /kaggle/input too, so RUN 3 shows the chain.
if not runs and INPUT.exists():
    for f in INPUT.rglob("persistence_marker.json"):
        try:
            runs = json.loads(f.read_text()).get("runs", [])
            print(f"\n    (seeding marker history from the mounted input {f})")
            break
        except Exception:
            pass

runs.append({
    "run_index": len(runs) + 1,
    "time": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    "run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "unknown"),
    "host": socket.gethostname(),
})
WORK.mkdir(parents=True, exist_ok=True)
MARKER.write_text(json.dumps({"runs": runs}, indent=2))

# A second, larger file: checkpoints are ~220MB, and a marker that survives is not
# proof that a real checkpoint directory does. This is small enough to be free and
# structured like what actually has to survive.
ckdir = WORK / "fake_checkpoint-200"
ckdir.mkdir(exist_ok=True)
(ckdir / "adapter_model.safetensors").write_bytes(b"\0" * (1 << 20))   # 1 MB
(ckdir / "trainer_state.json").write_text(json.dumps({"global_step": 200}))

print("\n" + "=" * 72)
print(f"WROTE marker as run_index {runs[-1]['run_index']}  ->  {MARKER}")
print(f"WROTE {ckdir}/ (a checkpoint-shaped directory, 1MB)")
print("\nNow EDIT something (a comment is enough — a +0 -0 diff is skipped with")
print("'Ran in 0 seconds'), commit again, and read section [1].")
print("  MARKER FOUND  -> /kaggle/working carries over; the resume design works as is.")
print("  NO MARKER     -> it does not; attach this notebook's output as an input and")
print("                   commit once more to confirm the restore path in section [2].")
print("=" * 72)
