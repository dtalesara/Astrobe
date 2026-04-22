"""
Astrobe — Core Astrophysical Engine
=====================================
Handles:
  - Random Forest life prediction
  - Habitability scoring (0–1)
  - Stellar habitable zone calculation
  - Future temperature projection
  - AGN proximity flag (offline heuristic; astroquery version optional)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional

from astrobe.models import CandidatePlanet, Star


# ─── Stellar Habitable Zone ────────────────────────────────────────────────────

@dataclass
class HabitableZone:
    inner_au: float
    outer_au: float

    def contains(self, distance_au: float) -> bool:
        return self.inner_au <= distance_au <= self.outer_au


def stellar_habitable_zone(star: Star) -> HabitableZone:
    """Calculate the classical habitable zone bounds (AU) from stellar luminosity."""
    inner = np.sqrt(star.luminosity / 1.1)
    outer = np.sqrt(star.luminosity / 0.53)
    return HabitableZone(inner_au=round(inner, 4), outer_au=round(outer, 4))


# ─── Habitability Score ────────────────────────────────────────────────────────

def habitability_score(planet: CandidatePlanet) -> float:
    """
    Weighted 0–1 score from core physical parameters.
    atmosphere (0.35) + temperature (0.35) + liquid_water (0.20) + orbital (0.10)
    """
    atm_score  = 1.0 if planet.has_atmosphere else 0.0
    temp_score = max(0.0, min(1.0, (planet.temperature_c + 20) / 70.0))
    water_score = 1.0 if planet.has_liquid_water else 0.0

    orbital_score = 0.0
    if planet.star and planet.orbital_distance_au is not None:
        hz = stellar_habitable_zone(planet.star)
        if hz.contains(planet.orbital_distance_au):
            orbital_score = 1.0
        else:
            dist = min(
                abs(planet.orbital_distance_au - hz.inner_au),
                abs(planet.orbital_distance_au - hz.outer_au),
            )
            span = hz.outer_au - hz.inner_au
            orbital_score = max(0.0, 1.0 - dist / (span + 1e-9))

    score = (
        0.35 * atm_score
        + 0.35 * temp_score
        + 0.20 * water_score
        + 0.10 * orbital_score
    )
    return round(score, 4)


# ─── AGN Proximity (heuristic offline) ────────────────────────────────────────

def agn_proximity_heuristic(planet: CandidatePlanet) -> bool:
    """
    Offline heuristic: flag high radiation as a proxy for AGN influence.
    For a real implementation swap in the astroquery/Vizier version.
    """
    return planet.radiation_gy > 100.0


# ─── Future Temperature Projection ────────────────────────────────────────────

def project_temperature(planet: CandidatePlanet, years: float,
                         rate_per_year: float = 0.02) -> float:
    """Simple linear warming projection (°C per year)."""
    return round(planet.temperature_c + rate_per_year * years, 3)


# ─── RF Life Predictor ────────────────────────────────────────────────────────

class AstrobePredictor:
    """
    Random Forest life predictor.  Trains on a list of labelled telescope
    observations and predicts life likelihood for new planets.
    """

    DEFAULT_TRAINING_DATA = [
        {"atmosphere": True,  "temperature": 15,  "liquid_water": True,  "has_life": True},
        {"atmosphere": True,  "temperature": 22,  "liquid_water": True,  "has_life": True},
        {"atmosphere": True,  "temperature": 10,  "liquid_water": True,  "has_life": True},
        {"atmosphere": True,  "temperature": -5,  "liquid_water": True,  "has_life": True},
        {"atmosphere": True,  "temperature": 40,  "liquid_water": True,  "has_life": True},
        {"atmosphere": False, "temperature": -67, "liquid_water": False, "has_life": False},
        {"atmosphere": False, "temperature": -40, "liquid_water": False, "has_life": False},
        {"atmosphere": True,  "temperature": 90,  "liquid_water": False, "has_life": False},
        {"atmosphere": False, "temperature": 200, "liquid_water": False, "has_life": False},
        {"atmosphere": True,  "temperature": -30, "liquid_water": False, "has_life": False},
    ]

    def __init__(self, training_data: Optional[list[dict]] = None):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        data = training_data or self.DEFAULT_TRAINING_DATA
        X = np.array([[int(d["atmosphere"]), d["temperature"], int(d["liquid_water"])]
                      for d in data])
        y = np.array([int(d["has_life"]) for d in data])

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        self._model = RandomForestClassifier(n_estimators=100, random_state=42)
        self._model.fit(X_tr, y_tr)
        self.accuracy = accuracy_score(y_te, self._model.predict(X_te))

    def predict(self, planet: CandidatePlanet) -> bool:
        X = np.array([[int(planet.has_atmosphere), planet.temperature_c,
                       int(planet.has_liquid_water)]])
        return bool(self._model.predict(X)[0])

    def predict_proba(self, planet: CandidatePlanet) -> float:
        X = np.array([[int(planet.has_atmosphere), planet.temperature_c,
                       int(planet.has_liquid_water)]])
        return float(self._model.predict_proba(X)[0][1])


# ─── Convenience wrapper ───────────────────────────────────────────────────────

@dataclass
class AstrobeResult:
    habitability_score: float
    rf_life_prediction: bool
    rf_life_probability: float
    in_habitable_zone: Optional[bool]
    habitable_zone: Optional[HabitableZone]
    near_agn: bool
    future_temp_10yr: float
    future_temp_50yr: float
    composition_analysis: Optional[dict]


def run_astrobe(planet: CandidatePlanet,
                predictor: Optional[AstrobePredictor] = None) -> AstrobeResult:
    """Run the full Astrobe astrophysical analysis on a CandidatePlanet."""
    if predictor is None:
        predictor = AstrobePredictor()

    hz = None
    in_hz = None
    if planet.star and planet.orbital_distance_au is not None:
        hz = stellar_habitable_zone(planet.star)
        in_hz = hz.contains(planet.orbital_distance_au)

    near_agn = planet.near_agn if planet.near_agn is not None \
               else agn_proximity_heuristic(planet)

    return AstrobeResult(
        habitability_score      = habitability_score(planet),
        rf_life_prediction      = predictor.predict(planet),
        rf_life_probability     = predictor.predict_proba(planet),
        in_habitable_zone       = in_hz,
        habitable_zone          = hz,
        near_agn                = near_agn,
        future_temp_10yr        = project_temperature(planet, 10),
        future_temp_50yr        = project_temperature(planet, 50),
        composition_analysis    = planet.composition.analyze()
                                  if planet.composition else None,
    )
