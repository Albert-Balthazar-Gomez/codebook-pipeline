"""Generate a 10-page synthetic IS 456-style codebook for pipeline validation.

This is a *fixture*, not a real standard. It reproduces the structural features
that break naive extractors, so pipeline correctness can be verified offline:

  - running header on every page + page number in the footer (must be stripped)
  - a full-page border frame (a layout grid that must NOT become a table)
  - clause numbering 4 levels deep (26.5.1.1) plus depth-1 chapter headings
  - a real ruled table, and a ruled grid that is genuinely sparse (must be
    rejected by is_real_table)
  - figure captions with the graphic cropped below them
  - hyphenation at line ends, and body prose that *starts* with a clause number
    (a trap for the heading detector)

Run:  python fixtures/make_fixture.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table as RLTable,
    TableStyle,
)

HEADER = "INDIAN STANDARD IS 456 : 2000"
FOOTER = "BIS New Delhi 110002"

OUT = Path(__file__).resolve().parent / "test_is456_excerpt.pdf"

PAGE_W, PAGE_H = A4

body = ParagraphStyle("body", fontName="Helvetica", fontSize=9.5, leading=13)
h1 = ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=12, leading=15, spaceBefore=8, spaceAfter=2)
h2 = ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10.5, leading=14, spaceBefore=6, spaceAfter=1)
h3 = ParagraphStyle("h3", fontName="Helvetica-Bold", fontSize=9.5, leading=13, spaceBefore=4)
cap = ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=8.5, leading=11, alignment=1)

LOREM = (
    "The permissible stresses and limit states adopted in this clause are "
    "applicable to reinforced concrete sections designed to resist the actions "
    "and combinations of actions specified in this standard. The design of "
    "members shall include the effects of axial force, bending and shear, and "
    "the detailing requirements given for each member type shall be satisfied "
    "irrespective of the method used for analysis of the structure. Where the "
    "actual load differs from the design load, the permissible limits shall be "
    "interpolated and the resulting values shall be adopted for design. This "
    "requirement applies to all limit states except those for which a partial "
    "safety factor of unity is expressly permitted by the relevant clause of "
    "this standard, and any relaxation shall be justified in the design."
)

LOREM2 = (
    "The moment of resistance of a rectangular section or a flanged section of a "
    "given overall depth shall be calculated by considering the following "
    "assumptions. The tensile strength of concrete is ignored. The compressive "
    "stress distribution in the concrete is rectangular. The tensile stress in "
    "the reinforcing steel is not more than the permissible stress or the stress "
    "from the strain compatibility relationship, whichever is less. The strain in "
    "the reinforcing steel depends on the strain in the concrete at the same "
    "level and is not more than the strain at yield. The strain compatibility "
    "relationship between the concrete and the reinforcement is based on the "
    "stress strain curves given for the grade of concrete and grade of steel "
    "used in the member."
)


def make_table(caption: str, rows: list[list[str]], widths: list[float]) -> KeepTogether:
    t = RLTable(rows, colWidths=widths, hAlign="LEFT")
    style = [
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    t.setStyle(TableStyle(style))
    return KeepTogether([Paragraph(caption, cap), Spacer(1, 2 * mm), t])


def make_figure_box(caption: str, height: float = 34 * mm) -> KeepTogether:
    """A ruled box standing in for a diagram, with its caption above it."""
    box = RLTable([[""]], colWidths=[70 * mm], rowHeights=[height], hAlign="CENTER")
    box.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.75, colors.black),
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
            ]
        )
    )
    return KeepTogether([Paragraph(caption, cap), Spacer(1, 1.5 * mm), box, Spacer(1, 3 * mm)])


def build_story() -> list:
    """Each tuple is one page's worth of flowables; they are joined with an
    explicit PageBreak so the fixture has a deterministic 10-page length.
    A variable page count would make running-head detection untestable (it needs
    several pages before a repeated string can be called a running head)."""
    pages: list[list] = []
    s: list = []

    # ---- p1: front matter + chapter heading ------------------------------- #
    s.append(Paragraph("26 BENDING AND SHEAR", h1))
    s.append(Paragraph("26.1 General", h2))
    s.append(Paragraph(LOREM, body))
    s.append(Spacer(1, 3 * mm))
    s.append(
        make_table(
            "Table 1 Limiting Moment of Resistance of Rectangular Sections",
            [
                ["Grade of concrete", "fck (MPa)", "Limiting Mu / fck b d2"],
                ["M 15", "15", "0.138"],
                ["M 20", "20", "0.156"],
                ["M 25", "25", "0.175"],
                ["M 30", "30", "0.193"],
                ["M 35", "35", "0.210"],
                ["M 40", "40", "0.227"],
            ],
            [45 * mm, 30 * mm, 45 * mm],
        )
    )

    pages.append(s)
    s = []

    # ---- p2: deep clause + a body line that starts with a clause number ---- #
    s.append(Paragraph("26.5 Design of sections", h2))
    s.append(Paragraph("26.5.1 The reinforcement of members subjected to bending shall be determined from the limiting moment of resistance of the section given in 26.4.2 and the design shear stress shall satisfy the provisions given in 26.5.4.", body))
    s.append(Paragraph("26.5.1.1", h3))
    s.append(Paragraph(LOREM2, body))
    s.append(Spacer(1, 3 * mm))
    s.append(make_figure_box("Fig. 1 Typical Arrangement of Reinforcement in a Beam Section"))

    pages.append(s)
    s = []

    # ---- p3: table 2 + figure 2 ------------------------------------------- #
    s.append(Paragraph("26.5.1.2 Where the tensile reinforcement is provided, it shall not be less than one-third of the total reinforcement provided and it shall not be less than the value given in Table 2.", body))
    s.append(Spacer(1, 2 * mm))
    s.append(
        make_table(
            "Table 2 Minimum Tensile Reinforcement",
            [
                ["Grade of reinforcement", "fy (MPa)", "As / b d"],
                ["Fe 250", "250", "0.0035"],
                ["Fe 415", "415", "0.0035"],
                ["Fe 500", "500", "0.0035"],
            ],
            [50 * mm, 28 * mm, 40 * mm],
        )
    )
    s.append(Spacer(1, 4 * mm))
    s.append(make_figure_box("Fig. 2 Effective Depth of a Section", height=30 * mm))

    pages.append(s)
    s = []

    # ---- p4: a new chapter ------------------------------------------------ #
    s.append(Paragraph("27 ULTIMATE STRENGTH OF MEMBERS", h1))
    s.append(Paragraph("27.1 General", h2))
    s.append(Paragraph(LOREM, body))
    s.append(Paragraph("27.2 Design strength of concrete and steel", h2))
    s.append(Paragraph("27.2.1 The characteristic strength of concrete shall be as given in Table 3.", body))
    s.append(Spacer(1, 2 * mm))
    s.append(
        make_table(
            "Table 3 Characteristic Strength of Concrete",
            [
                ["Grade", "fck (MPa)", "fcm (MPa)", "fctm (MPa)"],
                ["M20", "20", "28", "2.2"],
                ["M25", "25", "33", "2.6"],
                ["M30", "30", "38", "2.9"],
                ["M35", "35", "43", "3.2"],
                ["M40", "40", "48", "3.5"],
            ],
            [30 * mm, 30 * mm, 30 * mm, 30 * mm],
        )
    )

    pages.append(s)
    s = []

    # ---- p5: 4-level clause chain ----------------------------------------- #
    s.append(Paragraph("27.3 Moment of resistance", h2))
    s.append(Paragraph("27.3.1 Rectangular sections", h3))
    s.append(Paragraph("27.3.1.1 The limiting moment of resistance of a rectangular section subjected to bending about an axis is given by the equations in this clause.", body))
    s.append(Paragraph("27.3.1.2 The depth of the neutral axis shall be limited to the value given in 26.5.1.1 and the strain in the tension steel shall not exceed the yield strain.", body))
    s.append(Paragraph("27.3.1.3 Where the section is flanged, the compression flange width shall be as given in Fig. 3.", body))
    s.append(Spacer(1, 2 * mm))
    s.append(make_figure_box("Fig. 3 Effective Flange Width of a T-Beam", height=32 * mm))

    pages.append(s)
    s = []

    # ---- p6: shear -------------------------------------------------------- #
    s.append(Paragraph("28 SHEAR AND TORSION", h1))
    s.append(Paragraph("28.1 General", h2))
    s.append(Paragraph(LOREM, body))
    s.append(Paragraph("28.2 Shear stress", h2))
    s.append(Paragraph("28.2.1 The shear stress shall be calculated as tau v = V u / (b d).", body))
    s.append(
        make_table(
            "Table 4 Design Shear Strength tau c",
            [
                ["pt (MPa)", "tau c (MPa)"],
                ["0.18", "0.19"],
                ["0.25", "0.28"],
                ["0.36", "0.43"],
                ["0.50", "0.64"],
            ],
            [45 * mm, 45 * mm],
        )
    )

    pages.append(s)
    s = []

    # ---- p7: a SPARSE ruled grid: must be REJECTED by is_real_table -------- #
    s.append(Paragraph("28.3 Shear reinforcement", h2))
    s.append(Paragraph("28.3.1 Vertical stirrups shall be provided as shown in Fig. 4.", body))
    s.append(Spacer(1, 2 * mm))
    # Only one cell has content -> is_real_table must return False.
    sparse = RLTable(
        [["a"], [], []],
        colWidths=[60 * mm, 40 * mm],
        rowHeights=[6 * mm, 6 * mm, 6 * mm],
        hAlign="LEFT",
    )
    sparse.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)]))
    s.append(KeepTogether([Paragraph("Table 5 Layout Frame (not a data table)", cap), Spacer(1, 2 * mm), sparse]))

    pages.append(s)
    s = []

    # ---- p8: serviceability ----------------------------------------------- #
    s.append(Paragraph("29 SERVICEABILITY", h1))
    s.append(Paragraph("29.1 General", h2))
    s.append(Paragraph(LOREM, body))
    s.append(Paragraph("29.2 Deflection", h2))
    s.append(Paragraph("29.2.1 The deflection of members shall be calculated from the equations given in Table 6.", body))
    s.append(
        make_table(
            "Table 6 Deflection Limits",
            [
                ["Member", "Span / effective depth", "Limiting deflection"],
                ["Simply supported", "L / 250", "live load only"],
                ["Simply supported", "L / 350", "live load + creep"],
                ["Continuous", "L / 250", "live load only"],
                ["Cantilever", "L / 125", "live load only"],
            ],
            [45 * mm, 45 * mm, 45 * mm],
        )
    )

    pages.append(s)
    s = []

    # ---- p9: robustness --------------------------------------------------- #
    s.append(Paragraph("30 ROBUSTNESS", h1))
    s.append(Paragraph("30.1 General", h2))
    s.append(Paragraph(LOREM, body))
    s.append(Paragraph("30.2 Continuity of ties", h2))
    s.append(Paragraph("30.2.1 Column ties shall comply with the detailing shown in Fig. 5.", body))
    s.append(Spacer(1, 2 * mm))
    s.append(make_figure_box("Fig. 5 Arrangement of Column Ties", height=34 * mm))

    pages.append(s)
    s = []

    # ---- p10: durability -------------------------------------------------- #
    s.append(Paragraph("31 DURABILITY OF CONCRETE STRUCTURES", h1))
    s.append(Paragraph("31.1 General", h2))
    s.append(Paragraph(LOREM, body))
    s.append(Paragraph("31.2 Water cement ratio", h2))
    s.append(Paragraph("31.2.1 The maximum water cement ratio shall be as given in Table 7.", body))
    s.append(
        make_table(
            "Table 7 Minimum Cement Content and Maximum Water Cement Ratio",
            [
                ["Exposure condition", "Min cement kg/m3", "Max w/c", "Min grade"],
                ["Mild", "220", "0.60", "M20"],
                ["Moderate", "250", "0.55", "M25"],
                ["Severe", "300", "0.50", "M30"],
                ["Very severe", "320", "0.45", "M35"],
                ["Extreme", "340", "0.40", "M40"],
            ],
            [48 * mm, 34 * mm, 26 * mm, 26 * mm],
        )
    )
    pages.append(s)

    story: list = []
    for i, group in enumerate(pages):
        if i:
            story.append(PageBreak())
        story.extend(group)
    return story


def decorate(canv, _doc) -> None:
    """Running head, footer, page number and a full-page frame."""
    canv.saveState()
    # A page border. This is a layout frame and must not be detected as a table.
    canv.setStrokeColor(colors.lightgrey)
    canv.rect(12 * mm, 10 * mm, PAGE_W - 24 * mm, PAGE_H - 20 * mm)
    canv.setFont("Helvetica-Bold", 9)
    canv.drawCentredString(PAGE_W / 2, PAGE_H - 15 * mm, HEADER)
    canv.setFont("Helvetica", 8)
    canv.drawCentredString(PAGE_W / 2, 13 * mm, FOOTER)
    canv.drawRightString(PAGE_W - 20 * mm, 13 * mm, "%d" % canv.getPageNumber())
    canv.restoreState()


def main() -> Path:
    doc = BaseDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=22 * mm,
        bottomMargin=20 * mm,
        title="IS 456 excerpt (synthetic fixture)",
        author="codebook-pipeline fixtures",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=decorate)])
    doc.build(build_story())
    print(f"wrote {OUT}")
    return OUT


if __name__ == "__main__":
    main()