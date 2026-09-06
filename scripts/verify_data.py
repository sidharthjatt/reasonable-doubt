"""Checkpoint 1: data-layer verification and EDA.

    python scripts/verify_data.py

Prints a report and writes it to ``results/data_report.md``. Exits non-zero if any
integrity check fails. No API calls; the only network is the dataset download.

The leakage check is the one that silently invalidates a project, so it runs first
and its failure is unmissable.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loading import (  # noqa: E402
    EXPECTED_SPLIT_SIZES,
    get_split,
    label_names,
    load_ledgar,
    text_sha256,
)
from src.data.manifest import (  # noqa: E402
    DEFAULT_MANIFEST_DIR,
    MANIFEST_SPECS,
    ManifestVerificationError,
    load_manifest,
    verify_manifest,
)

SPLIT_ORDER = ["train", "validation", "test"]
SPLIT_ERAS = {"train": "2016-2017", "validation": "2018 (dev)", "test": "2019"}


class Report:
    """Accumulates markdown lines, echoes them, and tracks failures."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.failures: list[str] = []

    def __call__(self, line: str = "") -> None:
        self.lines.append(line)
        print(line)

    def check(self, ok: bool, passed: str, failed: str) -> bool:
        if ok:
            self(f"- PASS — {passed}")
        else:
            self(f"- **FAIL — {failed}**")
            self.failures.append(failed)
        return ok

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def section(rep: Report, title: str) -> None:
    rep()
    rep(f"## {title}")
    rep()


def check_leakage(rep: Report, ds, hashes: dict[str, list[str]]) -> None:
    section(rep, "1. Leakage check")
    rep(
        "Every example text is sha256-hashed. Cross-split overlap means the "
        "chronological split has been compromised and no held-out number is "
        "trustworthy. Within-split duplicates are reported but are a property of the "
        "corpus, not an error."
    )
    rep()

    sets = {name: set(h) for name, h in hashes.items()}
    rep("| pair | overlapping examples |")
    rep("|------|----------------------|")
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    overlaps = {}
    for a, b in pairs:
        n = len(sets[a] & sets[b])
        overlaps[(a, b)] = n
        rep(f"| {a} ∩ {b} | {n} |")
    rep()
    rep.check(
        all(v == 0 for v in overlaps.values()),
        "zero cross-split overlap — train, dev and test are disjoint.",
        "CROSS-SPLIT LEAKAGE DETECTED: "
        + ", ".join(f"{a}∩{b}={n}" for (a, b), n in overlaps.items() if n)
        + ". Every held-out result is invalid until this is resolved.",
    )
    rep()

    rep("| split | rows | distinct texts | duplicate rows | most-repeated text |")
    rep("|-------|------|----------------|----------------|--------------------|")
    for name in SPLIT_ORDER:
        counts = Counter(hashes[name])
        n_rows = len(hashes[name])
        dupes = n_rows - len(counts)
        top = counts.most_common(1)[0][1]
        rep(f"| {name} | {n_rows:,} | {len(counts):,} | {dupes:,} | {top}× |")


def check_chronology(rep: Report, ds) -> None:
    section(rep, "2. Chronological integrity")
    rep(
        "LEDGAR's split is chronological, not random. Sizes are asserted at load "
        "time; row order within each split is the dataset's own order and is never "
        "reshuffled, concatenated or re-split. The manifests address rows by index "
        "into that order, so any reordering is caught by the sha256 in section 6."
    )
    rep()
    rep("| split | era | rows | expected | match |")
    rep("|-------|-----|------|----------|-------|")
    ok = True
    for name in SPLIT_ORDER:
        got, exp = len(ds[name]), EXPECTED_SPLIT_SIZES[name]
        ok &= got == exp
        rep(f"| {name} | {SPLIT_ERAS[name]} | {got:,} | {exp:,} | {'yes' if got == exp else 'NO'} |")
    rep()
    rep.check(ok, "split sizes are 60k/10k/10k as expected.", "split sizes have drifted.")


def check_distribution(rep: Report, ds, names: list[str]) -> None:
    section(rep, "3. Class distribution")
    counts = {n: Counter(int(x) for x in ds[n]["label"]) for n in SPLIT_ORDER}

    rep("| split | classes present | min | median | max | imbalance (max/min) |")
    rep("|-------|-----------------|-----|--------|-----|---------------------|")
    for name in SPLIT_ORDER:
        vals = [counts[name].get(i, 0) for i in range(len(names))]
        present = [v for v in vals if v > 0]
        lo, hi = min(vals), max(vals)
        ratio = f"{hi / lo:.1f}×" if lo else f"inf (min=0, max={hi})"
        rep(
            f"| {name} | {len(present)}/{len(names)} | {lo} | "
            f"{statistics.median(vals):.0f} | {hi} | {ratio} |"
        )
    rep()

    train = counts["train"]
    rep("**Train split — 10 most frequent classes**")
    rep()
    rep("| class | count | share |")
    rep("|-------|-------|-------|")
    total = sum(train.values())
    for cls, n in train.most_common(10):
        rep(f"| {names[cls]} | {n:,} | {n / total:.2%} |")
    rep()

    rep("**Train split — 10 least frequent classes**")
    rep()
    rep("| class | count | share |")
    rep("|-------|-------|-------|")
    tail = sorted(range(len(names)), key=lambda i: (train.get(i, 0), i))[:10]
    for cls in tail:
        n = train.get(cls, 0)
        rep(f"| {names[cls]} | {n:,} | {n / total:.2%} |")
    rep()

    head10 = sum(n for _, n in train.most_common(10))
    tail50 = sum(sorted(train.get(i, 0) for i in range(len(names)))[:50])
    rep(
        f"The long tail is severe: the 10 largest classes hold {head10 / total:.1%} of "
        f"train, while the 50 smallest hold {tail50 / total:.1%} combined. This is why "
        "**macro-F1 is the primary metric** — accuracy is dominated by the head and "
        "would hide total failure on the tail."
    )


def check_lengths(rep: Report, ds) -> None:
    section(rep, "4. Clause length (characters)")
    rep("| split | p50 | p90 | p99 | max | mean |")
    rep("|-------|-----|-----|-----|-----|------|")
    for name in SPLIT_ORDER:
        lens = sorted(len(t) for t in ds[name]["text"])

        def pct(p: float) -> int:
            return lens[min(len(lens) - 1, int(p * len(lens)))]

        rep(
            f"| {name} | {pct(0.50):,} | {pct(0.90):,} | {pct(0.99):,} | "
            f"{lens[-1]:,} | {statistics.mean(lens):,.0f} |"
        )


def check_token_estimate(rep: Report, ds) -> None:
    section(rep, "5. Token length — ESTIMATE ONLY")
    rep(
        "> **ESTIMATE. NOT FOR BILLING.** These counts come from a local tokenizer and "
        "are for capacity planning only (encoder `max_length`, batch sizing). Under "
        "hard rule 9, token counts are per-model and must come from that model's own "
        "`count_tokens` call or `usage` block. Claude 4.7+ uses a newer tokenizer that "
        "produces roughly 30% more tokens for the same text, so these numbers will "
        "understate Claude token counts substantially. Never budget from this table."
    )
    rep()

    # Hard rule 11: no silent degradation. If the tokenizer cannot be loaded we RAISE.
    # An earlier version fell back to a chars/4 heuristic here; it produced a plausible
    # table that was ~30% wrong — the same magnitude as the tokenizer effect this
    # project exists to measure.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    def encode(text: str) -> int:
        return len(tok(text, add_special_tokens=True)["input_ids"])

    rep("Tokenizer: microsoft/deberta-v3-base (the Tier 0 encoder's own tokenizer)")
    rep()
    rep("| split | sample | p50 | p90 | p99 | max | over 512 |")
    rep("|-------|--------|-----|-----|-----|-----|----------|")
    for name in SPLIT_ORDER:
        split = ds[name]
        step = max(1, len(split) // 2000)
        sample = list(split["text"])[::step]
        lens = sorted(encode(t) for t in sample)

        def pct(p: float) -> int:
            return lens[min(len(lens) - 1, int(p * len(lens)))]

        over = sum(1 for x in lens if x > 512) / len(lens)
        rep(
            f"| {name} | {len(lens):,} | {pct(0.50)} | {pct(0.90)} | {pct(0.99)} | "
            f"{lens[-1]} | {over:.1%} |"
        )
    rep()
    rep(
        "The `over 512` column is the share of clauses that would be truncated at "
        "DeBERTa's usual 512-token window — a Tier 0 design input."
    )


def check_manifests(rep: Report, ds, manifest_dir: Path) -> None:
    section(rep, "6. Manifest verification")
    rep(
        "Each manifest is re-read from disk, its indices re-applied to the dataset, "
        "and its text and label-space sha256 recomputed. A mismatch means the rows we "
        "would evaluate are not the rows the manifest was frozen on."
    )
    rep()
    for name in MANIFEST_SPECS:
        try:
            m = load_manifest(name, manifest_dir)
            verify_manifest(m, ds, manifest_dir)
        except (FileNotFoundError, ManifestVerificationError) as exc:
            rep.check(False, "", f"manifest {name}: {exc}")
            continue

        absent = m.num_labels - m.classes_represented
        rep.check(
            True,
            f"`{name}` verified — sha256 `{m.text_sha256[:16]}…`",
            "",
        )
        rep(f"  - split `{m.split}`, n={m.n}, seed={m.seed}, sampling `{m.sampling}`")
        rep(f"  - purpose: {m.purpose}")
        rep(
            f"  - classes represented **{m.classes_represented}/{m.num_labels}** "
            f"({absent} with zero examples), per-class min **{m.min_class_count}**, "
            f"max {m.max_class_count}"
        )
        if absent:
            missing = [c for c, n in m.class_counts.items() if n == 0]
            rep(
                f"  - classes with zero examples: {', '.join(missing)}. These are NOT "
                "dropped from the label space and are NOT oversampled; macro-F1 over "
                f"this manifest is therefore an average over {m.classes_represented} "
                "classes, and that must be stated wherever the number is reported."
            )
        rep()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    ap.add_argument("--out", type=Path, default=Path("results/data_report.md"))
    args = ap.parse_args()

    rep = Report()
    rep("# Data report — Checkpoint 1")
    rep()
    rep("Dataset: `coastalcph/lex_glue`, config `ledgar`.")
    rep(
        "Generated by `scripts/verify_data.py`. Every downstream script must call "
        "`verify_manifest()` before running."
    )

    ds = load_ledgar()
    names = label_names(ds)
    hashes = {n: [text_sha256(t) for t in get_split(ds, n)["text"]] for n in SPLIT_ORDER}

    check_leakage(rep, ds, hashes)
    check_chronology(rep, ds)
    check_distribution(rep, ds, names)
    check_lengths(rep, ds)
    check_token_estimate(rep, ds)
    check_manifests(rep, ds, args.manifest_dir)

    section(rep, "Summary")
    if rep.failures:
        rep(f"**{len(rep.failures)} CHECK(S) FAILED.** Do not proceed:")
        rep()
        for f in rep.failures:
            rep(f"- {f}")
    else:
        rep("All integrity checks passed.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(rep.text(), encoding="utf-8")
    print(f"\nWrote {args.out}")
    return 1 if rep.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
