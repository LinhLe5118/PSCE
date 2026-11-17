"""Build the Chapter 3 draft DOCX with the thesis renderer and the school template.

Imports ``docs/thesis/v2_2026-10-13/build_thesis_v2.py`` (read-only) so headings, tables, equations, captions and references look exactly
like the thesis build. Tables marked ``@@TABLE:name@@`` and numbers marked
``@@NUM:key@@`` in ``03_dataset.md`` are filled from ``tables/*.csv`` and
``numbers.json`` written by ``build_ch3_assets.py``; no number is typed by hand.

Usage (repository root, Windows with Word's MML2OMML.XSL):
    python docs/thesis/chapter3_dataset_2026-10-16/scripts/build_ch3_docx.py
The DOCX is a build product and is not committed.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import docx
import pandas as pd
from docx.oxml.ns import qn

REPO = Path(__file__).resolve().parents[4]  # repository root
THESIS = REPO / "docs/thesis/v2_2026-10-13"
HERE = Path(__file__).resolve().parents[1]
TAB = HERE / "tables"

spec = importlib.util.spec_from_file_location("build_thesis_v2", THESIS / "build_thesis_v2.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
B.HERE = HERE  # figure paths in the chapter are relative to this folder

NUMBERS = json.loads((HERE / "numbers.json").read_text())


def fmt_int(x) -> str:
    return f"{int(x):,}"


def md_table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def table_registry() -> str:
    r = pd.read_csv(TAB / "registry.csv")
    rows = []
    for _, x in r.iterrows():
        sa = "—" if pd.isna(x.sa_min) else f"{int(x.sa_min)}–{int(x.sa_max)}"
        rows.append([x.panel, x.role, x.source, x.mdps, x.behaviors, fmt_int(x.rows), x.budgets, fmt_int(x.jobs), sa,
                     f"{x.censored_pct:.1f}"])
    rows.append(["**Total**", "", "", f"**{fmt_int(r.mdps.sum())}**", "", f"**{fmt_int(r.rows.sum())}**", "",
                 f"**{fmt_int(r.jobs.sum())}**", "", ""])
    return md_table(["Panel", "Role", "Source", "MDPs", "Beh.", "Rows", "Budgets", "Learner jobs", "S×A", ">x50 (%)"], rows)


def table_profiles() -> str:
    p = pd.read_csv(TAB / "pilot_profiles.csv")
    cols = ["observed_sa_fraction", "normalized_visitation_entropy", "empirical_stochastic_cell_fraction",
            "low_count_cell_fraction", "weighted_transition_entropy"]
    rows = [[x.panel, fmt_int(x.groups), f"{x.sa_median:.0f}", *[x[c] for c in cols]] for _, x in p.iterrows()]
    cap = ("Table: Table 3.10. Pilot statistics by panel: median [interquartile range] over MDP–behavior groups "
           "(stochastic groups only; pre-outcome).")
    cap = cap.replace("(stochastic groups only; pre-outcome).",
                      "(stochastic groups only; pre-outcome). N = number of groups; S×A = median number of state–action pairs.")
    return cap + "\n" + md_table(["Panel", "N", "S×A", "Observed S–A fraction", "Visitation entropy",
                                  "Stochastic-cell fraction", "Low-count fraction", "Transition entropy"], rows)


def table_labels() -> str:
    d = pd.read_csv(TAB / "label_distribution.csv")
    keep = ["Final internal", "Colosseum 2029", "SPIBB-2", "Forest", "FrozenLake", "Gambler", "MDP Playground", "labmaze"]
    d = d[d.panel.isin(keep)].copy()
    d["order"] = d.panel.map({k: i for i, k in enumerate(keep)})
    rows = [[x.panel, x.learner, x.n, x.x1, x.x2, x.x5, x.x10, x.x20, x.x50, f"{x['>x50']} ({x.censored_pct:.1f}%)"]
            for _, x in d.sort_values(["order", "learner"]).iterrows()]
    cap = ("Table: Table 3.11. Label counts of the final test panels, by learner "
           "(number of rows in each requirement class).")
    return cap + "\n" + md_table(["Panel", "Learner", "Rows", "x1", "x2", "x5", "x10", "x20", "x50", ">x50"], rows)


def table_noise() -> str:
    d = pd.read_csv(TAB / "label_noise.csv")
    d = d[d.censored.notna()]
    rows = [[x.panel, fmt_int(x.rows), f"{x.nonmonotone_pct:.1f}", f"{x.fallback_pct:.1f}", fmt_int(x.censored),
             int(x.resolved_x100), int(x.resolved_x200), f"{100 * x.still_censored / x.censored:.1f}"]
            for _, x in d.iterrows()]
    tot = d[d.panel.isin(["Final internal", "Colosseum 2029", "SPIBB-2", "Forest"])]
    rows.append(["**Four final panels**", fmt_int(tot.rows.sum()), "", "", fmt_int(tot.censored.sum()),
                 int(tot.resolved_x100.sum()), int(tot.resolved_x200.sum()),
                 f"{100 * tot.still_censored.sum() / tot.censored.sum():.1f}"])
    cap = ("Table: Table 3.12. Seed variability and stability of censoring. Non-monotone: the success count "
           "decreases between two consecutive primary budgets. Fall-back: a row reaches 8/10 and later drops below "
           "it. The last three columns follow the rows labeled >x50 to the secondary budgets.")
    return cap + "\n" + md_table(["Panel", "Rows", "Non-monotone (%)", "Fall-back (%)", "Censored rows",
                                  "First success at x100", "First success at x200", "Still censored (%)"], rows)


TABLES = {"registry": table_registry, "profiles": table_profiles, "labels_final": table_labels, "noise": table_noise}

ADDITIONAL_REFERENCES = """
@gulcehre2020: C. Gulcehre *et al.*, "RL Unplugged: A suite of benchmarks for offline reinforcement learning," in *Advances in Neural Information Processing Systems*, vol. 33, 2020.
@conserva2022: M. Conserva and P. Rauber, "Hardness in Markov decision processes: Theory and practice," in *Advances in Neural Information Processing Systems*, vol. 35, 2022.
@chades2014: I. Chadès, G. Chapron, M.-J. Cros, F. Garcia, and R. Sabbadin, "MDPtoolbox: A multi-platform toolbox to solve stochastic dynamic programming problems," *Ecography*, vol. 37, no. 9, pp. 916–920, 2014.
@towers2024: M. Towers *et al.*, "Gymnasium: A standard interface for reinforcement learning environments," arXiv:2407.17032, 2024.
@rajan2023: R. Rajan, J. L. Borja Diaz, S. Guttikonda, F. Ferreira, A. Biedenkapp, J. O. von Hartz, and F. Hutter, "MDP Playground: An analysis and debug testbed for reinforcement learning," *Journal of Artificial Intelligence Research*, vol. 77, pp. 821–890, 2023.
@beattie2016: C. Beattie *et al.*, "DeepMind Lab," arXiv:1612.03801, 2016.
@misra2020: D. Misra, M. Henaff, A. Krishnamurthy, and J. Langford, "Kinematic state abstraction and provably efficient rich-observation reinforcement learning," in *Proc. 37th Int. Conf. Machine Learning (ICML)*, PMLR 119, 2020, pp. 6961–6971.
"""


def fill(text: str) -> str:
    text = re.sub(r"@@TABLE:(\w+)@@", lambda m: TABLES[m.group(1)](), text)
    text = re.sub(r"@@NUM:(\w+)@@", lambda m: "`" + str(NUMBERS[m.group(1)]).replace(" / ", "` / `") + "`", text)
    if "@@" in text.replace("@@REFERENCES@@", ""):
        raise ValueError("unfilled placeholder")
    return text


def main() -> None:
    chapter = fill((HERE / "03_dataset.md").read_text(encoding="utf-8"))
    (HERE / "03_dataset_filled.md").write_text(chapter, encoding="utf-8")
    refs = (THESIS / "07_references.md").read_text(encoding="utf-8").replace("# References", "# References of Chapter 3")
    refs = refs + "\n" + ADDITIONAL_REFERENCES
    (HERE / "references_additions.md").write_text(ADDITIONAL_REFERENCES.strip() + "\n", encoding="utf-8")

    doc = docx.Document(str(B.TEMPLATE))
    body = doc.element.body
    for el in list(body.iterchildren()):
        if el.tag != qn("w:sectPr"):
            body.remove(el)
    for name, text in B.resolve_citations([("03_dataset.md", chapter), ("refs", refs)]):
        B.render(doc, text)
    # the first heading needs no page break in a chapter-only draft
    first = next(p for p in doc.paragraphs if p.style.name.startswith("Heading"))
    first.paragraph_format.page_break_before = False
    doc.core_properties.title = "Difficulty-Aware Offline Reinforcement Learning — Chapter 3 draft"
    out = HERE / "DAORL_Chapter3_Dataset_draft.docx"
    doc.save(str(out))
    print("Wrote", out)


if __name__ == "__main__":
    main()
