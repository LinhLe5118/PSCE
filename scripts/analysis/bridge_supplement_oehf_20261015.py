"""BRIDGE follow-up scores for the supplement, written with the final naming (decision of 15/10/2026, option a).

The BRIDGE manuscript chain of 9/10/2026 (``finalize_verified_manuscripts.py``, run 15/10/2026 18:23) generated its
fragments before the final method existed and calls the 108-field head "Full OEH". Under the final naming that head
is **OEH-F**; the shift-gated OEH was never run on BRIDGE. This script regenerates the supplement section from the
same hash-checked audit (``artifacts/week8/difficulty_common_metric_final_2026-10-09``) with the chain's own table
code, renames the methods and checks that no bare "OEH" is left. It does not touch the chain's outputs.

Usage: ``python -m scripts.analysis.bridge_supplement_oehf_20261015``.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from scripts.analysis import finalize_verified_manuscripts as Fv

ROOT = Fv.REPO
OUTPUT = ROOT / "artifacts/week8/bridge_supplement_oehf_2026-10-15"
TEX = ROOT / "paper/sections/supplement_bridge_oehf.tex"
RENAMES = (  # order matters: longest first
    ("Full OEH (exploratory)", "OEH-F (exploratory)"),
    ("Reduced OEH", "OEH-F without direct OEH-B"),
    ("Full OEH", "OEH-F"),
    ("full OEH", "OEH-F"),
    (r"\OEH{}'s", "OEH-F's"),
    (r"\OEH{}", "OEH-F"),
    ("OEH's", "OEH-F's"),
    ("& CF/18 & OEH & CF/18 & OEH\\\\", "& CF/18 & OEH-F & CF/18 & OEH-F\\\\"),
    ("MiniGrid-14 was opened before the full run", "MiniGrid-14 was opened before OEH-F was run on it"),
    (r"\section{Completed BRIDGE follow-up scores}", r"\subsection{All scores}"),
)
INTRO = (r"\section{BRIDGE follow-up for OEH-F (descriptive)}\label{sec:bridge-oehf}" "\n"
         r"These BRIDGE panels predate the final method. The forecaster scored here is \textbf{OEH-F}, the full 108-field "
         r"head without the shift gate; the final, shift-gated OEH was not run on BRIDGE. The prospective five-task panel was "
         r"sealed before its labels. The 14 MiniGrid tasks had been opened before OEH-F was run on them, so that panel is "
         r"post-outcome exploratory. All intervals are descriptive." "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rename(text: str) -> str:
    for old, new in RENAMES:
        text = text.replace(old, new)
    bare = re.findall(r"OEH(?!-F|-B)", text.replace(r"\OEH", ""))
    if bare or r"\OEH" in text:
        raise RuntimeError(f"bare OEH left after renaming: {len(bare)}")
    return text


def main() -> None:
    folder, manifest, summary, paired = Fv.load_verified(False)  # status and score hashes are checked here
    coverage = Fv.load_posterior_coverage(False)
    results_tex, supplement_tex = Fv.make_latex(summary, paired, coverage)
    body = rename(results_tex.rstrip("\n") + "\n" + supplement_tex)
    TEX.write_bytes((INTRO + body).encode("utf-8"))  # LF on every OS: paper/ is normalized to LF by git
    OUTPUT.mkdir(parents=True, exist_ok=True)
    doc = {"status": "descriptive", "decision": "Linh, 15/10/2026: option (a), OEH-F naming, supplement only",
           "created_utc": datetime.now(timezone.utc).isoformat(), "renames": [list(r) for r in RENAMES],
           "inputs_sha256": {Path(p).relative_to(ROOT).as_posix(): sha(Path(p)) for p in (
               folder / "manifest.json", folder / "summary.csv", folder / "paired.csv",
               ROOT / "artifacts/week8/bridge_expanded_full_oehb_preoutcome_2026-10-08/manifest.json")},
           "audit_status": manifest["status"],
           "outputs_sha256": {TEX.relative_to(ROOT).as_posix(): sha(TEX)},
           "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in (
               Path(__file__), ROOT / "scripts/analysis/finalize_verified_manuscripts.py",
               ROOT / "scripts/analysis/verify_common_difficulty_metrics.py")}}
    (OUTPUT / "manifest.json").write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"wrote {TEX.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
