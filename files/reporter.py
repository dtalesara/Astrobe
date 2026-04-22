"""
Astrobe — Terminal Reporter
============================
Prints a formatted AstrobeReport to stdout using ANSI colours.
"""

from __future__ import annotations
from astrobe.pipeline import AstrobeReport

# ── ANSI ──────────────────────────────────────────────────────────────────────
R      = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BLUE   = "\033[94m"
MAGENTA= "\033[95m"
GREY   = "\033[90m"
WHITE  = "\033[97m"

STATUS_COLOUR = {"viable": GREEN, "marginal": YELLOW, "unlikely": RED}
STATUS_ICON   = {"viable": "✔", "marginal": "~", "unlikely": "✘"}

VERDICT_COLOUR = {
    "STRONG CANDIDATE": GREEN,
    "PROMISING":        GREEN,
    "MARGINAL":         YELLOW,
    "UNLIKELY":         YELLOW,
    "INHOSPITABLE":     RED,
}


def _bar(score: float, width: int = 24) -> str:
    filled = round(score * width)
    empty  = width - filled
    colour = GREEN if score >= 0.70 else (YELLOW if score >= 0.40 else RED)
    return f"{colour}{'█' * filled}{GREY}{'░' * empty}{R}"


def _div(char="─", width=68):
    return GREY + char * width + R


def _row(label, value, colour=WHITE):
    return f"  {GREY}{label:<22}{R} {colour}{value}{R}"


def print_report(report: AstrobeReport, show_all_organisms: bool = False):
    p  = report.planet
    a  = report.astrobe
    b  = report.biosearch
    d  = report.digital_dna
    vc = VERDICT_COLOUR.get(report.verdict, WHITE)

    print()
    print(_div("═"))
    print(f"  {BOLD}{WHITE}{p.name}{R}  {GREY}·  {p.location}{R}")
    if p.description:
        print(f"  {DIM}{p.description}{R}")
    print(_div())
    print()

    # ── Composite verdict ──────────────────────────────────────────────────────
    print(f"  {BOLD}COMPOSITE SCORE{R}  {_bar(report.composite_score)}  "
          f"{BOLD}{report.composite_score:.3f}{R}  "
          f"{vc}{BOLD}[ {report.verdict} ]{R}")
    print()

    # ── MODULE 1: Astrobe Core ─────────────────────────────────────────────────
    print(f"  {CYAN}{BOLD}── MODULE 1 · ASTROBE CORE{R}")
    print(_row("Habitability score",   f"{a.habitability_score:.3f}  {_bar(a.habitability_score, 16)}"))
    print(_row("RF life probability",  f"{a.rf_life_probability:.1%}",
               GREEN if a.rf_life_probability >= 0.5 else RED))
    if a.habitable_zone:
        hz_str = f"{a.habitable_zone.inner_au:.3f}–{a.habitable_zone.outer_au:.3f} AU"
        in_hz  = f"  {'✔ IN ZONE' if a.in_habitable_zone else '✘ OUTSIDE'}"
        print(_row("Habitable zone", hz_str + in_hz,
                   GREEN if a.in_habitable_zone else YELLOW))
    print(_row("AGN proximity",
               "⚠ DETECTED — radiation risk" if a.near_agn else "Clear",
               RED if a.near_agn else GREEN))
    print(_row("Surface temperature", f"{p.temperature_c:+.1f} °C"))
    print(_row("Projected +10 yr",    f"{a.future_temp_10yr:+.1f} °C"))
    print(_row("Projected +50 yr",    f"{a.future_temp_50yr:+.1f} °C"))
    if a.composition_analysis:
        parts = ", ".join(
            f"{el} {v['proportion']:.1f}%" for el, v in a.composition_analysis.items() if v["present"]
        )
        print(_row("Key atm. gases", parts or "none detected"))
    print()

    # ── MODULE 2: BioSearch ───────────────────────────────────────────────────
    print(f"  {GREEN}{BOLD}── MODULE 2 · BIOSEARCH{R}")
    print(_row("BioSearch score",  f"{b.biosearch_score:.3f}  {_bar(b.biosearch_score, 16)}"))
    print(_row("Life phase",       f"{b.phase}  —  {b.phase_label}"))
    print(_row("Emergence score",  f"{b.emergence:.2f} / 10 (raw)"))
    print(_row("Preservation",     f"{b.preservation:.1%}"))
    print(_row("Detectability",    f"{b.detectability:.1f} / 10"))
    atm_col = {"oxidizing": GREEN, "neutral": YELLOW, "reducing": YELLOW}
    print(_row("Atmosphere state", b.atmosphere_state.upper(),
               atm_col.get(b.atmosphere_state, WHITE)))
    print()

    # ── MODULE 3: HOPE ────────────────────────────────────────────────────────
    viable   = report.viable_organisms
    marginal = report.marginal_organisms
    unlikely = [r for r in report.hope if r.status == "unlikely"]

    print(f"  {MAGENTA}{BOLD}── MODULE 3 · HOPE — EXTREMOPHILE COMPATIBILITY{R}")
    print(f"  {GREEN}{len(viable)} viable{R}  ·  "
          f"{YELLOW}{len(marginal)} marginal{R}  ·  "
          f"{RED}{len(unlikely)} unlikely{R}")
    print()

    display = (viable + marginal) if not show_all_organisms else report.hope
    if not display:
        display = report.hope[:3]

    for res in display:
        sc = STATUS_COLOUR[res.status]
        ic = STATUS_ICON[res.status]
        print(f"  {sc}{ic}{R} {BOLD}{res.organism.name}{R}  "
              f"{DIM}{res.organism.classification}  ·  {res.organism.domain}{R}")
        print(f"    Compatibility  {_bar(res.score, 16)}  {sc}{res.score:.1%}{R}")
        if res.matched_conditions:
            for m in res.matched_conditions[:3]:
                print(f"    {GREEN}✔{R} {DIM}{m}{R}")
        if res.limiting_factors:
            for l in res.limiting_factors[:3]:
                print(f"    {RED}!{R} {DIM}{l}{R}")
        print()

    if not show_all_organisms and unlikely:
        print(GREY + f"  {len(unlikely)} 'unlikely' organism(s) hidden. "
              "Run with --show-all-organisms to display them." + R)
        print()

    # ── MODULE 4: DigitalDNA ──────────────────────────────────────────────────
    BASE_COL = {"A": RED, "C": BLUE, "T": YELLOW, "G": GREEN}

    print(f"  {BLUE}{BOLD}── MODULE 4 · DIGITALDNA{R}")
    seq_coloured = "".join(f"{BASE_COL.get(b, WHITE)}{b}{R}" for b in d.sequence)
    print(f"  {GREY}Encoded sequence:{R}  {seq_coloured}")
    print()
    print(f"  {GREY}Habitability class distribution:{R}")
    for cls, prob in d.class_probabilities.items():
        bar_col = GREEN if prob >= 0.5 else (YELLOW if prob >= 0.2 else GREY)
        marker = f"  ◀ {BOLD}TOP CLASS{R}" if cls == d.top_class else ""
        print(f"    {cls:<26} {_bar(prob, 16)}  {bar_col}{prob:.1%}{R}{marker}")
    print()
    if d.evolution_best_fitness is not None:
        best_seq = "".join(f"{BASE_COL.get(b, WHITE)}{b}{R}"
                           for b in (d.evolution_best_sequence or ""))
        print(_row("Evolution best fitness", f"{d.evolution_best_fitness:.6f}"))
        print(f"  {GREY}{'Optimal sequence':<22}{R}  {best_seq}")
    print()

    # ── Footer ─────────────────────────────────────────────────────────────────
    print(_div("═"))
    print(GREY + "  Astrobe v1.0  ·  github.com/your-username/astrobe" + R)
    print()
