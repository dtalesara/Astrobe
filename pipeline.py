"""
Astrobe — Unified Pipeline
===========================
Runs a CandidatePlanet through all four analysis modules in sequence
and returns a single AstrobeReport containing every result.

Pipeline order:
  1. Astrobe Core    — astrophysical habitability, RF prediction
  2. BioSearch       — planetary emergence, phase, detectability
  3. HOPE            — extremophile organism compatibility
  4. DigitalDNA      — DNA-encoded habitability + evolutionary simulation
"""

from __future__ import annotations
from dataclasses import dataclass, field

from astrobe.models import CandidatePlanet
from astrobe.core import AstrobeResult, AstrobePredictor, run_astrobe
from astrobe.biosearch import BioSearchResult, run_biosearch
from astrobe.hope import HOPEResult, run_hope
from astrobe.digital_dna import DigitalDNAResult, run_digital_dna


@dataclass
class AstrobeReport:
    planet: CandidatePlanet
    astrobe: AstrobeResult
    biosearch: BioSearchResult
    hope: list[HOPEResult]
    digital_dna: DigitalDNAResult

    # ── Derived summary properties ─────────────────────────────────────────────

    @property
    def viable_organisms(self) -> list[HOPEResult]:
        return [r for r in self.hope if r.status == "viable"]

    @property
    def marginal_organisms(self) -> list[HOPEResult]:
        return [r for r in self.hope if r.status == "marginal"]

    @property
    def composite_score(self) -> float:
        """
        Weighted composite of all module scores → single 0–1 life likelihood.
          Astrobe habitability   30%
          BioSearch score        25%
          DigitalDNA top prob    25%
          HOPE viable ratio      20%
        """
        hope_ratio = len(self.viable_organisms) / max(len(self.hope), 1)
        ddna_score = self.digital_dna.class_probabilities.get("Habitable", 0.0) \
                   + 0.5 * self.digital_dna.class_probabilities.get("Potentially Habitable", 0.0)

        return round(
            0.30 * self.astrobe.habitability_score
            + 0.25 * self.biosearch.biosearch_score
            + 0.25 * min(ddna_score, 1.0)
            + 0.20 * hope_ratio,
            4,
        )

    @property
    def verdict(self) -> str:
        s = self.composite_score
        if s >= 0.70:
            return "STRONG CANDIDATE"
        elif s >= 0.50:
            return "PROMISING"
        elif s >= 0.30:
            return "MARGINAL"
        elif s >= 0.15:
            return "UNLIKELY"
        return "INHOSPITABLE"


def analyse_planet(planet: CandidatePlanet,
                   predictor: AstrobePredictor = None,
                   run_evolution: bool = True,
                   evo_generations: int = 20,
                   verbose: bool = False) -> AstrobeReport:
    """
    Run the complete Astrobe pipeline on a CandidatePlanet.

    Parameters
    ----------
    planet          : The planet to analyse.
    predictor       : Pre-trained AstrobePredictor (created if None).
    run_evolution   : Whether to run DigitalDNA evolutionary simulation.
    evo_generations : Number of evolutionary generations.
    verbose         : Print progress to stdout.

    Returns
    -------
    AstrobeReport with all module results.
    """
    if verbose:
        print(f"\n  ▶ Running Astrobe pipeline for: {planet.name}")

    if verbose: print("    [1/4] Astrobe core ...")
    astrobe_result = run_astrobe(planet, predictor)

    if verbose: print("    [2/4] BioSearch ...")
    biosearch_result = run_biosearch(planet)

    if verbose: print("    [3/4] HOPE extremophile analysis ...")
    hope_results = run_hope(planet)

    if verbose: print("    [4/4] DigitalDNA ...")
    ddna_result = run_digital_dna(planet, run_evo=run_evolution,
                                   evo_generations=evo_generations)

    report = AstrobeReport(
        planet=planet,
        astrobe=astrobe_result,
        biosearch=biosearch_result,
        hope=hope_results,
        digital_dna=ddna_result,
    )

    if verbose:
        print(f"    ✔ Complete. Composite score: {report.composite_score:.3f}  [{report.verdict}]")

    return report
