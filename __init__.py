"""
Astrobe
========
Unified astrobiology analysis pipeline.

Quick start:
    from astrobe import CandidatePlanet, Star, analyse_planet

    earth = CandidatePlanet(
        name="Earth",
        star=Star("G", luminosity=1.0, radiation=3.0),
        orbital_distance_au=1.0,
        temperature_c=15,
        has_atmosphere=True,
        has_liquid_water=True,
        age_ga=4.5,
        geothermal=5.0,
        tectonics=6.0,
        silica=5.5,
        pressure_atm=1.0,
        ph=7.0,
        salinity_ppt=35.0,
        radiation_gy=0.003,
        water_activity=0.99,
    )

    report = analyse_planet(earth, verbose=True)
    print(f"Composite score: {report.composite_score}")
    print(f"Verdict: {report.verdict}")
"""

from astrobe.models import CandidatePlanet, Star, ChemicalComposition
from astrobe.pipeline import analyse_planet, AstrobeReport
from astrobe.core import AstrobePredictor, run_astrobe
from astrobe.biosearch import run_biosearch
from astrobe.hope import run_hope, EXTREMOPHILE_DB
from astrobe.digital_dna import run_digital_dna, train as train_digital_dna

__all__ = [
    "CandidatePlanet",
    "Star",
    "ChemicalComposition",
    "analyse_planet",
    "AstrobeReport",
    "AstrobePredictor",
    "run_astrobe",
    "run_biosearch",
    "run_hope",
    "run_digital_dna",
    "train_digital_dna",
    "EXTREMOPHILE_DB",
]

__version__ = "1.0.0"
from astrobe.reporter import print_report
