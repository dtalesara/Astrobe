"""
Astrobe — BioSearch Module
===========================
Planetary biosignature pipeline.

Given a CandidatePlanet, computes:
  - emergence_score      : how likely life could arise
  - phase                : 0 prebiotic → 3 advanced biosphere
  - preservation         : probability biosignatures survived
  - detectability        : signal-to-noise for remote sensing
  - biosearch_score      : master weighted composite (0–1)

The planet is also evolved forward in time so age-dependent
atmosphere transitions are reflected in the result.
"""

from __future__ import annotations
import random
from dataclasses import dataclass

from astrobe.models import CandidatePlanet


PHASE_LABELS = {
    0: "Prebiotic",
    1: "Hydrothermal Biosphere",
    2: "Marine Biosphere",
    3: "Advanced / Oxygen-influenced Biosphere",
}


# ─── Atmosphere Evolution ──────────────────────────────────────────────────────

def _evolved_atmosphere(planet: CandidatePlanet) -> str:
    if planet.age_ga < 2.5:
        return "reducing"
    elif planet.age_ga < 3.0:
        return "neutral"
    return "oxidizing"


# ─── Sub-scores ────────────────────────────────────────────────────────────────

def emergence_score(planet: CandidatePlanet) -> float:
    score = 0.0
    if planet.has_liquid_water:
        score += 3.0
    score += 0.6 * planet.geothermal
    score += 0.5 * planet.silica
    score += 1.0 - abs(planet.tectonics - 5) / 5.0   # optimal mid-range
    return max(score, 0.0)


def classify_phase(emergence: float) -> int:
    if emergence < 3:
        return 0
    elif emergence < 6:
        return 1
    elif emergence < 9:
        return 2
    return 3


def preservation_probability(planet: CandidatePlanet) -> float:
    silica    = planet.silica / 10.0
    stability = 1.0 - abs(planet.tectonics - 5) / 5.0
    thermal   = 1.0 - (planet.geothermal / 10.0)
    return max(0.0, min(1.0,
        0.5 * silica + 0.3 * stability + 0.2 * thermal
    ))


def detectability_score(planet: CandidatePlanet, phase: int) -> float:
    signal = phase * 2.5
    if planet.has_liquid_water:
        signal += 2.0
    atm = _evolved_atmosphere(planet)
    if atm == "oxidizing":
        signal += 2.0
    noise = (planet.star.radiation if planet.star else 4.0) * 0.3
    return min(max(signal - noise, 0.0), 10.0)


# ─── Master Score ──────────────────────────────────────────────────────────────

@dataclass
class BioSearchResult:
    emergence: float
    phase: int
    phase_label: str
    preservation: float
    detectability: float
    biosearch_score: float
    atmosphere_state: str


def run_biosearch(planet: CandidatePlanet) -> BioSearchResult:
    """Run the full BioSearch pipeline on a CandidatePlanet."""
    em    = emergence_score(planet)
    phase = classify_phase(em)
    pres  = preservation_probability(planet)
    det   = detectability_score(planet, phase)
    atm   = _evolved_atmosphere(planet)

    em_n  = min(em / 10.0, 1.0)
    total = (
        0.4 * em_n
        + 0.3 * pres
        + 0.3 * (det / 10.0)
    )

    return BioSearchResult(
        emergence       = round(em, 4),
        phase           = phase,
        phase_label     = PHASE_LABELS[phase],
        preservation    = round(pres, 4),
        detectability   = round(det, 4),
        biosearch_score = round(total, 4),
        atmosphere_state = atm,
    )
