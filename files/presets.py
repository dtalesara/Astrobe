"""
Astrobe — Planet Presets
=========================
Ready-to-use CandidatePlanet objects for known and hypothetical worlds.
"""

from astrobe.models import CandidatePlanet, Star, ChemicalComposition

SUN   = Star(star_type="G", luminosity=1.0,   radiation=3.0)
M_RED = Star(star_type="M", luminosity=0.04,  radiation=7.0)

PRESETS: dict[str, CandidatePlanet] = {

    "earth": CandidatePlanet(
        name="Earth",
        location="Solar System",
        description="Our home — the reference standard for habitable worlds.",
        star=SUN, orbital_distance_au=1.0,
        temperature_c=15, has_atmosphere=True, has_liquid_water=True,
        atmosphere_type="oxidizing",
        composition=ChemicalComposition({"N2": 78, "O2": 21, "CO2": 0.04, "H2O": 0.4, "Ar": 0.56}),
        age_ga=4.5, geothermal=5.0, tectonics=6.0, silica=5.5,
        pressure_atm=1.0, ph=7.0, salinity_ppt=35.0,
        radiation_gy=0.003, water_activity=0.99,
    ),

    "mars": CandidatePlanet(
        name="Mars",
        location="Solar System",
        description="Cold, thin-atmosphered desert world. Once had liquid water.",
        star=SUN, orbital_distance_au=1.52,
        temperature_c=-40, has_atmosphere=True, has_liquid_water=False,
        atmosphere_type="neutral",
        composition=ChemicalComposition({"CO2": 95, "N2": 2.6, "Ar": 1.6, "O2": 0.15, "H2O": 0.03, "other": 0.62}),
        age_ga=4.5, geothermal=2.0, tectonics=1.5, silica=6.0,
        pressure_atm=0.006, ph=8.0, salinity_ppt=0.0,
        radiation_gy=0.3, water_activity=0.10,
    ),

    "europa_ocean": CandidatePlanet(
        name="Europa Subsurface Ocean",
        location="Jupiter's moon Europa",
        description="Liquid saltwater ocean under ~20 km of ice. Tidal heating may drive hydrothermal vents.",
        star=SUN, orbital_distance_au=5.2,
        temperature_c=2, has_atmosphere=True, has_liquid_water=True,
        atmosphere_type="reducing",
        age_ga=4.5, geothermal=6.0, tectonics=3.0, silica=4.0,
        pressure_atm=1300.0, ph=7.5, salinity_ppt=50.0,
        radiation_gy=0.001, water_activity=0.97,
    ),

    "enceladus": CandidatePlanet(
        name="Enceladus Subsurface Ocean",
        location="Saturn's moon Enceladus",
        description="Active geysers confirmed to contain water, salts, silica, and organics.",
        star=SUN, orbital_distance_au=9.5,
        temperature_c=90, has_atmosphere=True, has_liquid_water=True,
        atmosphere_type="reducing",
        age_ga=4.5, geothermal=8.0, tectonics=4.0, silica=7.0,
        pressure_atm=50.0, ph=9.0, salinity_ppt=10.0,
        radiation_gy=0.01, water_activity=0.99,
    ),

    "kepler_442b": CandidatePlanet(
        name="Kepler-442b (analog)",
        location="Lyra constellation, ~1,200 ly",
        description="Super-Earth in the habitable zone of a K-type star. High habitability estimates.",
        star=Star("K", luminosity=0.31, radiation=4.0),
        orbital_distance_au=0.41,
        temperature_c=10, has_atmosphere=True, has_liquid_water=True,
        atmosphere_type="neutral",
        age_ga=2.9, geothermal=6.5, tectonics=5.5, silica=6.0,
        pressure_atm=1.2, ph=7.2, salinity_ppt=30.0,
        radiation_gy=0.005, water_activity=0.98,
    ),

    "proxima_b": CandidatePlanet(
        name="Proxima Centauri b (analog)",
        location="Proxima Centauri, 4.2 ly",
        description="Nearest known exoplanet candidate. Tidally locked to a flare star.",
        star=M_RED, orbital_distance_au=0.05,
        temperature_c=-10, has_atmosphere=True, has_liquid_water=False,
        atmosphere_type="reducing",
        age_ga=4.8, geothermal=4.0, tectonics=3.5, silica=5.0,
        pressure_atm=0.8, ph=7.0, salinity_ppt=10.0,
        radiation_gy=2.5, water_activity=0.70,
    ),
}
