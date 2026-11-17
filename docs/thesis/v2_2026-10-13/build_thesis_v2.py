"""Build the 2026-10-13 thesis DOCX from Markdown chapters on the school template.

Keeps the cover page and styles of the root ``DAORL_thesis.docx`` (never
modified), replaces everything after the cover with the maintained chapters,
renders LaTeX math as native Word equations (OMML) and inserts table/figure
captions that feed the generated lists. Run ``update_fields.ps1`` afterwards so
Word fills the table of contents and exports the review PDF.

Markdown subset: ``#``/``##``/``###`` headings, paragraphs, ``- `` bullets,
``**bold**``, ``*italic*``, `` `code` ``, ``$inline$`` and ``$$display$$ (n)``
math, ``Table: caption`` followed by a pipe table, ``![Figure x. caption](path)``,
``<!-- pagebreak -->`` and ``::: refs`` blocks for the bibliography.
"""
from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor, Twips
from lxml import etree

HERE = Path(__file__).resolve().parent
# School template: the copy committed next to this script, else the original
# root DAORL_thesis.docx of the working folder (never modified).
TEMPLATE = next((path for path in (HERE / "template" / "DAORL_thesis_template.docx",
                                   HERE.parents[2] / "DAORL_thesis.docx") if path.exists()),
                HERE / "template" / "DAORL_thesis_template.docx")
XSL = Path(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL")
SOURCES = ["00_front.md", "01_introduction.md", "02_background.md", "03_dataset.md", "04_method.md",
           "05_results.md", "06_conclusion.md", "07_references.md", "08_appendix.md"]
HEADER_FILL, BORDER = "0b75af", "8aa6b8"
_OMML = None


def omml(latex: str, display: bool):
    global _OMML
    from latex2mathml.converter import convert

    if _OMML is None:
        _OMML = etree.XSLT(etree.parse(str(XSL)))
    mathml = convert(latex, display="block" if display else "inline")
    tree = _OMML(etree.fromstring(mathml.encode()))
    node = tree.getroot()
    return copy.deepcopy(node)


def run(paragraph, text, bold=False, italic=False, code=False, size=None, color=None):
    r = paragraph.add_run(text)
    r.bold, r.italic = bold or None, italic or None
    if code:
        r.font.name = "Consolas"
        r.font.size = Pt(10.5)
    if size:
        r.font.size = Pt(size)
    if color:
        r.font.color.rgb = RGBColor.from_string(color)
    return r


INLINE = re.compile(r"(\$[^$]+\$|\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`)")


def inline(paragraph, text, size=None, bold=False, color=None):
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("$") and part.endswith("$") and len(part) > 2:
            paragraph._p.append(omml(part[1:-1], False))
        elif part.startswith("**"):
            run(paragraph, part[2:-2], bold=True, size=size, color=color)
        elif part.startswith("`"):
            run(paragraph, part[1:-1], code=True)
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            run(paragraph, part[1:-1], italic=True, bold=bold, size=size, color=color)
        else:
            run(paragraph, part, bold=bold, size=size, color=color)


def body_paragraph(doc, text, indent=True):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(1.0) if indent else Cm(0)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    inline(p, text)
    return p


def heading(doc, text, level):
    p = doc.add_paragraph(style=f"Heading {level}")
    run(p, text, bold=level >= 2)
    if level == 1:
        p.paragraph_format.page_break_before = True
    return p


def ensure_style(doc, name):
    if name not in [s.name for s in doc.styles]:
        style = doc.styles.add_style(name, 1)
        style.base_style = doc.styles["Caption"]
        style.font.bold = False
        style.font.size = Pt(11)
        style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        style.paragraph_format.space_after = Pt(10)
        style.paragraph_format.keep_with_next = name == "Table Caption"
    return name


def caption(doc, text, kind):
    p = doc.add_paragraph(style=ensure_style(doc, f"{kind} Caption"))
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    label, _, rest = text.partition(". ")
    run(p, label + ". ", bold=True, size=11)
    inline(p, rest, size=11)
    return p


def display_math(doc, latex, number):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0)
    stops = p.paragraph_format.tab_stops
    stops.add_tab_stop(Cm(8.5), 1)   # center
    stops.add_tab_stop(Cm(17.0), 2)  # right
    run(p, "\t")
    math = omml(latex, True)
    # oMathPara forces its own line; place the bare oMath inline between tabs.
    inner = math.find(qn("m:oMath")) if math.tag == qn("m:oMathPara") else math
    p._p.append(inner)
    if number:
        run(p, f"\t({number})")
    return p


def set_cell(cell, text, header):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), HEADER_FILL if header else "ffffff")
    tcPr.append(shd)
    margins = OxmlElement("w:tcMar")
    for side, width in (("top", 60), ("left", 90), ("bottom", 60), ("right", 90)):
        m = OxmlElement(f"w:{side}")
        m.set(qn("w:w"), str(width))
        m.set(qn("w:type"), "dxa")
        margins.append(m)
    tcPr.append(margins)
    p = cell.paragraphs[0]
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    numeric = re.fullmatch(r"[\s\-−+.,0-9%\[\];/()×x≥≤<>]+", text.replace("**", "")) is not None
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if header or numeric else WD_ALIGN_PARAGRAPH.LEFT
    inline(p, text, size=10, bold=header, color="FFFFFF" if header else None)


def table(doc, rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [r for r in cells if not all(re.fullmatch(r":?-{3,}:?", c) for c in r)]
    t = doc.add_table(rows=len(cells), cols=len(cells[0]))
    t.alignment = 1
    tblPr = t._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        b = OxmlElement(f"w:{side}")
        b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), "6")
        b.set(qn("w:color"), BORDER)
        borders.append(b)
    tblPr.append(borders)
    width = OxmlElement("w:tblW")
    width.set(qn("w:w"), "9600")
    width.set(qn("w:type"), "dxa")
    tblPr.append(width)
    # Column widths proportional to typical cell length (fixed layout, 17 cm text width).
    total = 9600
    lengths = []
    for j in range(len(cells[0])):
        texts = [re.sub(r"\$[^$]*\$", "xxxx", r[j]) for r in cells if j < len(r)]
        longest_word = max(len(w) for t in texts for w in (t.split() or [""]))
        mean = sum(len(t) for t in texts) / len(texts)
        lengths.append(max(4.0, min(45.0, max(mean, longest_word * 0.9))))
    floor = 700  # twips; keeps short labels such as "G1" or "108" on one line
    widths = [int(total * w / sum(lengths)) for w in lengths]
    short = [j for j, w in enumerate(widths) if w < floor]
    if short:
        spare = sum(floor - widths[j] for j in short)
        wide = [j for j in range(len(widths)) if j not in short]
        wide_total = sum(widths[j] for j in wide) or 1
        for j in wide:
            widths[j] -= int(spare * widths[j] / wide_total)
        for j in short:
            widths[j] = floor
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tblPr.append(layout)
    grid = t._tbl.tblGrid
    for col, w in zip(grid.findall(qn("w:gridCol")), widths):
        col.set(qn("w:w"), str(w))
    for row in t.rows:
        for cell, w in zip(row.cells, widths):
            cell.width = Twips(w)
    for i, row in enumerate(cells):
        if i == 0:
            trPr = t.rows[0]._tr.get_or_add_trPr()
            h = OxmlElement("w:tblHeader")
            h.set(qn("w:val"), "1")
            trPr.append(h)
        for j, text in enumerate(row):
            set_cell(t.cell(i, j), text, i == 0)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(4)


def figure(doc, path, text, width_cm):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str((HERE / path).resolve()), width=Cm(width_cm))
    caption(doc, text, "Figure")


def field(doc, instruction, placeholder):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0)
    for kind, text in (("begin", None), ("instr", instruction), ("separate", None), ("text", placeholder),
                       ("end", None)):
        r = p.add_run()
        if kind == "instr":
            node = OxmlElement("w:instrText")
            node.set(qn("xml:space"), "preserve")
            node.text = text
            r._r.append(node)
        elif kind == "text":
            r.text = text
        else:
            node = OxmlElement("w:fldChar")
            node.set(qn("w:fldCharType"), kind)
            r._r.append(node)


def front_lists(doc):
    p = doc.add_paragraph(style="TOC Heading")
    run(p, "Table of Contents", bold=True, size=16)
    p.paragraph_format.page_break_before = True
    field(doc, 'TOC \\o "1-3" \\h \\z \\u', "Right-click and update field to build the table of contents.")
    p = doc.add_paragraph(style="TOC Heading")
    run(p, "List of Figures", bold=True, size=16)
    p.paragraph_format.page_break_before = True
    field(doc, 'TOC \\h \\z \\t "Figure Caption,1"', "Update field to list figures.")
    p = doc.add_paragraph(style="TOC Heading")
    run(p, "List of Tables", bold=True, size=16)
    field(doc, 'TOC \\h \\z \\t "Table Caption,1"', "Update field to list tables.")


def render(doc, text):
    lines = text.splitlines()
    i, para = 0, []

    def flush():
        if para:
            body_paragraph(doc, " ".join(s.strip() for s in para))
            para.clear()

    pending_caption = None
    in_refs = False
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s == "::: refs":
            flush()
            in_refs = True
        elif s == ":::":
            flush()
            in_refs = False
        elif in_refs and s:
            p = doc.add_paragraph(style="Thesis Reference")
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            inline(p, s)
        elif s == "<!-- pagebreak -->":
            flush()
            doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
        elif s.startswith("<!--"):
            flush()
            while "-->" not in lines[i]:
                i += 1
        elif s.startswith("#"):
            flush()
            level = len(s) - len(s.lstrip("#"))
            title = s[level:].strip()
            if title == "Abstract":
                p = doc.add_paragraph(style="Heading 1")
                run(p, title)
            elif title.startswith("Table of Contents"):
                front_lists(doc)
            else:
                heading(doc, title, level)
        elif s.startswith("$$"):
            flush()
            latex = s[2:]
            while not latex.rstrip().endswith("$$") and "$$" not in latex:
                i += 1
                latex += " " + lines[i].strip()
            body, _, tail = latex.partition("$$")
            number = re.search(r"\(([\d.]+)\)", tail)
            display_math(doc, body.strip(), number.group(1) if number else None)
        elif s.startswith("Table:"):
            flush()
            pending_caption = s[len("Table:"):].strip()
        elif s.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            if pending_caption:
                caption(doc, pending_caption, "Table")
                pending_caption = None
            table(doc, rows)
            continue
        elif s.startswith("!["):
            flush()
            m = re.match(r"!\[(.+)\]\(([^)]+)\)(\{width=([\d.]+)\})?", s)
            figure(doc, m.group(2), m.group(1), float(m.group(4) or 15))
        elif s.startswith("- "):
            flush()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.left_indent = Cm(1.0)
            p.paragraph_format.first_line_indent = Cm(-0.5)
            p.paragraph_format.tab_stops.add_tab_stop(Cm(1.0))
            run(p, "•	")
            inline(p, s[2:])
        elif re.match(r"^\d+\. ", s):
            flush()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.left_indent = Cm(1.0)
            p.paragraph_format.first_line_indent = Cm(-0.6)
            number, _, rest = s.partition(" ")
            run(p, number + " ")
            inline(p, rest)
        elif not s:
            flush()
        else:
            para.append(s)
        i += 1
    flush()


CITE = re.compile(r"\[(@[\w-]+(?:\s*;\s*@[\w-]+)*)\]")
REF_LINE = re.compile(r"^@([\w-]+):\s*(.+)$")


def resolve_citations(texts):
    """Number ``[@key]`` citations by first appearance; emit only cited references."""
    entries, order = {}, []
    for _, text in texts:
        for line in text.splitlines():
            m = REF_LINE.match(line.strip())
            if m:
                entries[m.group(1)] = m.group(2)
    for name, text in texts:
        for group in CITE.findall(text):
            for key in re.findall(r"@([\w-]+)", group):
                if key not in entries:
                    raise KeyError(f"{name}: citation without reference entry: {key}")
                if key not in order:
                    order.append(key)
    number = {key: i + 1 for i, key in enumerate(order)}
    unused = sorted(set(entries) - set(order))
    if unused:
        print("Unused reference entries:", ", ".join(unused))

    def replace(match):
        keys = re.findall(r"@([\w-]+)", match.group(1))
        return "[" + ", ".join(str(number[k]) for k in keys) + "]"

    result = []
    for name, text in texts:
        lines = []
        for line in text.splitlines():
            if REF_LINE.match(line.strip()):
                continue
            if line.strip() == "@@REFERENCES@@":
                lines.extend(f"[{number[k]}] {entries[k]}" for k in order)
                continue
            lines.append(CITE.sub(replace, line))
        result.append((name, "\n".join(lines)))
    return result


def main(output: Path) -> None:
    doc = docx.Document(str(TEMPLATE))
    body = doc.element.body
    children = list(body.iterchildren())
    cover_end = next(i for i, el in enumerate(children) if el.find(".//" + qn("w:sectPr")) is not None)
    for el in children[cover_end + 1:]:
        if el.tag != qn("w:sectPr"):
            body.remove(el)
    texts = [(name, (HERE / name).read_text(encoding="utf-8")) for name in SOURCES]
    for name, text in resolve_citations(texts):
        render(doc, text)
    doc.core_properties.title = "Difficulty-Aware Offline Reinforcement Learning"
    doc.core_properties.subject = "Pilot-only forecasting of learner-specific data requirements"
    doc.save(str(output))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "DAORL_Thesis_v2_2026-10-13.docx")
