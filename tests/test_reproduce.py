"""The reproduction command (``python -m scripts.reproduce``), without the two-minute refit.

The full command refits the heads, regenerates the forecasts and reruns the registered analyses; these tests check
the parts that need no refit: the seals of the final tests, the printed tables it compares with, and that its
scoring code gives the printed Tables 11-15, 17 and 18 from the committed scored files.
"""
from __future__ import annotations

import shutil
import subprocess
import sys

from scripts.reproduce import common as C
from scripts.reproduce import seals
from scripts.reproduce import tables as T


def _documented(line: str, doc: dict) -> bool:
    return any(line.startswith(f"Table {k['table']}, {k['row']} / {k['column']}: printed {k['printed']}")
               for k in doc["known_differences"])


def test_command_line_starts_in_a_fresh_process():
    out = subprocess.run([sys.executable, "-m", "scripts.reproduce", "--help"], capture_output=True, text=True, cwd=C.ROOT,
                         check=True).stdout
    assert "--out" in out and "--steps" in out and "--workers" in out


def test_output_folder_inside_artifacts_is_refused():
    run = subprocess.run([sys.executable, "-m", "scripts.reproduce", "--out", str(C.ART / "reproduce_out"), "--steps", "seals"],
                         capture_output=True, text=True, cwd=C.ROOT)
    assert run.returncode != 0 and "must not be inside artifacts" in run.stderr
    assert not (C.ART / "reproduce_out").exists()


def test_final_tests_document_matches_the_artifacts():
    from scripts.reproduce import final_tests_doc as D

    assert D.TARGET.read_text(encoding="utf-8") == D.build(), "rerun python -m scripts.reproduce.final_tests_doc"


def test_seals_of_the_three_final_tests_hold():
    report = seals.check()
    assert report["all_ok"], report
    assert set(report["tests"]) == {"Test 1", "Test 2", "Test 3"}


def test_printed_numbers_parse_and_format_back_unchanged():
    doc = C.load_json(T.EXPECTED)
    tokens = [tok for key in doc["tables"] for _, _, text in T.numeric_cells(doc["tables"], key) for tok in T._tokens(text)]
    assert len(tokens) > 600
    for tok in tokens:
        assert T._format(T._value(tok)[0], tok) == tok, tok


def test_scoring_code_gives_the_printed_tables_from_committed_scores(tmp_path):
    for t in C.TESTS.values():
        target = tmp_path / "3_registered" / t["folder"]
        target.mkdir(parents=True)
        shutil.copy(C.ART / t["folder"] / "scored.csv.gz", target / "scored.csv.gz")
    d = T.load_cases(tmp_path)
    t11, t12, t13 = T.table_11_13(d)
    computed = {"11": t11, "12": t12, "13": t13, "14": T.table_14(d), "15": T.table_15(d), "17": T.table_17(d),
                "18": T.table_18(d)}
    doc = C.load_json(T.EXPECTED)
    differences, rendered = T.compare(computed, doc["tables"])
    assert all(_documented(line, doc) for line in differences), differences
    assert len(differences) == len(doc["known_differences"])
    assert rendered["11"][1] == ["OEH", ".0476 / .562", ".1080 / .833", ".0189 / .358", ".1097 / .790", ".0662 / .599"]


def test_differences_are_flagged():
    doc = C.load_json(T.EXPECTED)
    computed = {"14": [["OEH", [13.1], [21.1], [8.0], [20.0], [8.6], [21.8]], ["Robust head", [15.1], [22.9], [7.0], [28.1], [9.3], [24.1]],
                       ["B2", [16.5], [24.9], [9.1], [35.8], [10.3], [29.6]]]}
    differences, _ = T.compare(computed, doc["tables"])
    assert differences == ["Table 14, OEH / Test 3: printed 21.7, computed 21.800000"]
