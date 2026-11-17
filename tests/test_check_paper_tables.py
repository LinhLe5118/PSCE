"""The hand-typed result tables of the paper (Tests 1-3) match the committed artifacts."""
from scripts.analysis import check_paper_tables

TEX = check_paper_tables.ROOT / "paper/sections/06_results.tex"


def test_paper_result_tables_match_artifacts():
    assert check_paper_tables.main(verbose=False)


def test_checker_detects_a_changed_number():
    text = TEX.read_text(encoding="utf-8")
    assert r"\textbf{.1080}" in text
    assert not check_paper_tables.main(verbose=False, tex=text.replace(r"\textbf{.1080}", r"\textbf{.1090}", 1))
