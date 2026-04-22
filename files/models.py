"""
Astrobe — Unified Data Models
==============================
CandidatePlanet is the single shared schema that flows through every
analysis module: Astrobe core → BioSearch → HOPE → DigitalDNA.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Star:
    star_type: str          # "M", "G", "K", "F"
    luminosity: float       # solar luminosity units
    radiation: float        # 0–10 arbitrary index


@dataclass
class ChemicalComposition:
    """Atmospheric composition in percentage by volume."""
    components: dict[str, float]   # e.g. {"N2": 78, "O2": 21, ...}
    KEY_ELEMENTS = ["O2", "CO2", "H2O"]

    def validate(self):
        if any(v < 0 for v in self.components.values()):
            raise ValueError("All proportions must be non-negative.")
        total = sum(self.components.values())
        if not (99.0 <= total <= 101.0):
            raise ValueError(f"Proportions must sum to ~100%, got {total:.2f}%.")

    def analyze(self) -> dict:
        return {
            el: {
                "present": self.components.get(el, 0) > 0,
                "proportion": self.components.get(el, 0),
            }
            for el in self.KEY_ELEMENTS
        }


@dataclass
class CandidatePlanet:
    """
    The universal input object for the Astrobe pipeline.

    Every module reads from this dataclass.  Fields cover:
      - astrophysical context  (star, orbital distance, AGN proximity)
      - surface/climate        (temperature, atmosphere, liquid water)
      - geological             (geothermal, tectonics, silica, age)
      - chemical               (atmospheric composition)
      - HOPE environment       (pressure, pH, salinity, radiation dose, water activity)
    """

    # Identity
    name: str
    location: str = "Unknown"
    description: str = ""

    # ── Astrophysical ──────────────────────────────────────────────
    star: Optional[Star] = None
    orbital_distance_au: Optional[float] = None   # AU from host star
    near_agn: Optional[bool] = None               # populated by AGNProximity if run

    # ── Climate / surface ─────────────────────────────────────────
    temperature_c: float = 15.0
    has_atmosphere: bool = True
    has_liquid_water: bool = True
    atmosphere_type: str = "neutral"   # "reducing" | "neutral" | "oxidizing"
    composition: Optional[ChemicalComposition] = None

    # ── Geological ────────────────────────────────────────────────
    age_ga: float = 4.5               # billions of years
    geothermal: float = 5.0           # 0–10
    tectonics: float = 5.0            # 0–10
    silica: float = 5.0               # 0–10

    # ── HOPE environment parameters ───────────────────────────────
    pressure_atm: float = 1.0
    ph: float = 7.0
    salinity_ppt: float = 35.0
    radiation_gy: float = 0.003       # annual absorbed dose
    water_activity: float = 0.99

    # ── DigitalDNA encoding ───────────────────────────────────────
    dna_sequence: Optional[str] = None   # set by encode_to_dna()

    def to_epigenetic_factors(self) -> dict[str, float]:
        """Return environmental parameters as DigitalDNA epigenetic factors."""
        return {
            "temperature":    self.temperature_c / 100.0,
            "atmosphere":     1.0 if self.has_atmosphere else 0.0,
            "liquid_water":   1.0 if self.has_liquid_water else 0.0,
            "geothermal":     self.geothermal / 10.0,
            "tectonics":      self.tectonics / 10.0,
            "silica":         self.silica / 10.0,
            "radiation":      min(self.radiation_gy / 1000.0, 1.0),
            "water_activity": self.water_activity,
            "ph_neutral":     1.0 - abs(self.ph - 7.0) / 7.0,
        }
