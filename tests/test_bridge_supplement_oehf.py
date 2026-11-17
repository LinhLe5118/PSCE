"""BRIDGE follow-up supplement with the final naming (OEH-F): reproducible from the hash-checked audit."""
from __future__ import annotations

import hashlib
import json
import re

from scripts.analysis import bridge_supplement_oehf_20261015 as B
from scripts.analysis import finalize_verified_manuscripts as Fv


def test_section_is_regenerated_exactly_from_the_audit():
    folder, _, summary, paired = Fv.load_verified(False)
    results_tex, supplement_tex = Fv.make_latex(summary, paired, Fv.load_posterior_coverage(False))
    expected = B.INTRO + B.rename(results_tex.rstrip("\n") + "\n" + supplement_tex)
    assert B.TEX.read_text(encoding="utf-8") == expected
    manifest = json.loads((B.OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    for rel, digest in {**manifest["inputs_sha256"], **manifest["outputs_sha256"]}.items():
        assert hashlib.sha256((B.ROOT / rel).read_bytes()).hexdigest() == digest


def test_full_head_is_called_oeh_f_everywhere_in_the_bridge_sections():
    for name in ("supplement_bridge_oehf.tex", "supplement_reduced_bridge.tex"):
        text = (B.ROOT / "paper/sections" / name).read_text(encoding="utf-8")
        assert "Full OEH" not in text and "full OEH" not in text and "full-\\OEH" not in text
        assert not re.search(r"Reduced OEH(?!-F)", text)
    body = B.TEX.read_text(encoding="utf-8")[len(B.INTRO):]
    assert not re.findall(r"OEH(?!-F|-B)", body)


def test_supplement_inputs_the_tracked_section_only():
    text = (B.ROOT / "paper/supplement.tex").read_text(encoding="utf-8")
    assert "\\IfFileExists" not in text and "\\input{sections/supplement_bridge_oehf}" in text
